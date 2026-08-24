"""No scored field may carry a default.

A field the model can omit, that then arrives at scoring carrying a default, is
a prediction the Engine never made being graded as though it had. Default
`category_id="other"` / `severity="medium"` would make silence a scoreable
answer, and the benchmark would be measuring the default's luck against the
ground-truth distribution instead of the model's judgement.

These tests assert on the JSON schema actually generated for the structured
output — the artefact the provider enforces — so a default reintroduced later
fails CI rather than quietly biasing a benchmark run.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from engine.analysis import _as_extractions, analyze_trace, consolidate
from engine.consolidate import assemble_board, fallback_plan
from engine.models import (
    Cluster,
    ConsolidationPlan,
    FindingExtraction,
    FindingExtractionList,
    RawFinding,
)
from tests.fakes import FakeChatModel

SCORED_FIELDS = ("category_id", "severity")


def object_schema(model, name: str) -> dict:
    """The generated schema for `name`, whether inlined or under $defs."""
    schema = model.model_json_schema()
    if schema.get("title") == name:
        return schema
    return schema["$defs"][name]


# -- the generated schema marks the scored fields required -----------------


@pytest.mark.parametrize(
    "model,name",
    [(FindingExtractionList, "FindingExtraction"), (ConsolidationPlan, "Cluster")],
)
@pytest.mark.parametrize("field", SCORED_FIELDS)
def test_scored_fields_are_required_in_the_generated_schema(model, name, field):
    schema = object_schema(model, name)
    assert field in schema["required"], (
        f"{name}.{field} is optional in the schema sent to the model, so the "
        f"model may omit it and a default will be scored as its prediction"
    )
    assert "default" not in schema["properties"][field]


@pytest.mark.parametrize(
    "model,name",
    [(FindingExtractionList, "FindingExtraction"), (ConsolidationPlan, "Cluster")],
)
def test_title_and_description_are_required_too(model, name):
    """They feed the description-deviation scorer and the issue pairing."""
    required = object_schema(model, name)["required"]
    assert {"title", "description"} <= set(required)


def test_severity_is_constrained_to_the_three_levels():
    """Required is not enough: the enum is what stops "unknown" arriving as a
    severity the scorer cannot place."""
    schema = object_schema(FindingExtractionList, "FindingExtraction")
    severity = schema["properties"]["severity"]
    enum = severity.get("enum") or schema["$defs"][severity["$ref"].split("/")[-1]]["enum"]
    assert sorted(enum) == ["high", "low", "medium"]


def test_the_localization_hints_stay_optional_deliberately():
    """Unscored: they help a human audit an occurrence, and forcing them would
    only push the model to invent a span id it could not find."""
    schema = object_schema(FindingExtractionList, "FindingExtraction")
    required = set(schema["required"])
    assert not ({"evidence", "span_id", "turn_index"} & required)


# -- trace_id is the orchestrator's, not the model's -----------------------


def test_trace_id_is_absent_from_the_extraction_schema():
    """The orchestrator knows which trace it asked about. Asking the model
    invites a mis-attributed finding, which corrupts the {trace_id, error_id}
    matrix scoring consumes."""
    schema = object_schema(FindingExtractionList, "FindingExtraction")
    assert "trace_id" not in schema["properties"]
    # No object anywhere in the schema offers it either. Checked on properties
    # rather than the raw JSON, because the model docstring — which explains
    # precisely why trace_id is absent — is itself sent as a description.
    full = FindingExtractionList.model_json_schema()
    for definition in [full, *full.get("$defs", {}).values()]:
        assert "trace_id" not in (definition.get("properties") or {})


def test_raw_finding_requires_the_stamped_trace_id():
    """No default here either: a skipped stamping step must raise, not file a
    finding against ""."""
    with pytest.raises(ValidationError):
        RawFinding(
            title="t", description="d", category_id="tool_misuse", severity="high"
        )


def test_every_finding_downstream_carries_the_stamped_trace_id(index, categories):
    extraction = FindingExtraction(
        title="t", description="d", category_id="tool_misuse", severity="high"
    )
    fake = FakeChatModel(
        responses=[AIMessage(content="ok")],
        structured=[FindingExtractionList(findings=[extraction, extraction])],
    )
    findings = analyze_trace(fake, index, "trace-planted-ticket", [], categories)
    assert [f.trace_id for f in findings] == ["trace-planted-ticket"] * 2


# -- an omitted scored field fails, and is counted -------------------------


@pytest.mark.parametrize("missing", SCORED_FIELDS)
def test_a_response_omitting_a_scored_field_fails_validation(missing):
    payload = {"title": "t", "description": "d", "category_id": "tool_misuse",
               "severity": "high"}
    payload.pop(missing)
    with pytest.raises(ValidationError):
        FindingExtractionList.model_validate({"findings": [payload]})


@pytest.mark.parametrize("missing", SCORED_FIELDS)
def test_the_analysis_pass_raises_rather_than_defaulting(index, categories, missing):
    """Straight into the per-trace failure accounting — never a defaulted finding."""
    payload = {"title": "t", "description": "d", "category_id": "tool_misuse",
               "severity": "high"}
    payload.pop(missing)
    fake = FakeChatModel(
        responses=[AIMessage(content="ok")], structured=[{"findings": [payload]}]
    )
    with pytest.raises(ValidationError):
        analyze_trace(fake, index, "trace-planted-ticket", [], categories)


def test_that_failure_lands_in_the_counted_per_trace_path(
    traces_file, monkeypatch, categories, capsys
):
    """End to end: the graph counts it as a failed trace rather than reporting
    the trace as clean."""
    from engine import graph as graph_module

    payload = {"title": "t", "description": "d", "category_id": "tool_misuse"}

    def emit_missing_severity(*args, **kwargs):
        return FindingExtractionList.model_validate({"findings": [payload]})

    monkeypatch.setattr(graph_module, "build_model", lambda name: FakeChatModel())
    monkeypatch.setattr(graph_module, "analyze_trace", emit_missing_severity)
    with pytest.raises(Exception, match="analysis failed on 6 of 6 traces"):
        graph_module.graph.invoke(
            {
                "trace_file": str(traces_file),
                "seed_issueboard": {},
                "categories": [c.model_dump() for c in categories],
            },
            config={"configurable": {"analysis_concurrency": 1}},
        )
    assert "ValidationError" in capsys.readouterr().err


@pytest.mark.parametrize("missing", SCORED_FIELDS)
def test_consolidation_retries_then_falls_back_rather_than_defaulting(
    categories, capsys, missing
):
    """A plan missing a scored field takes the existing retry->fallback path."""
    cluster = {"title": "t", "description": "d", "category_id": "tool_misuse",
               "severity": "high", "finding_indices": [0]}
    cluster.pop(missing)

    class Broken:
        def invoke(self, messages, **kwargs):
            return ConsolidationPlan.model_validate({"clusters": [cluster]})

    class BrokenModel(FakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            return Broken()

    findings = [
        RawFinding(trace_id="t1", title="Mode", description="d",
                   category_id="tool_misuse", severity="high")
    ]
    board = consolidate(BrokenModel(responses=[], structured=[]), findings, None, categories)
    assert "falling back to deterministic clustering" in capsys.readouterr().err
    # The fallback derives category and severity from findings that DID validate.
    assert board.issues[0].category_id == "tool_misuse"
    assert board.issues[0].severity == "high"


def test_as_extractions_rejects_a_non_plan_result():
    with pytest.raises(ValueError, match="no usable structured output"):
        _as_extractions("not a finding list")


# -- fallback_plan still works with guaranteed-present severities ---------


def test_fallback_plan_reads_severities_that_are_now_always_present():
    findings = [
        RawFinding(trace_id="t1", title="Mode", description="d",
                   category_id="formatting", severity="low"),
        RawFinding(trace_id="t2", title="Mode", description="d",
                   category_id="formatting", severity="high"),
    ]
    plan = fallback_plan(findings)
    assert len(plan.clusters) == 1
    assert plan.clusters[0].severity == "high"
    assert plan.clusters[0].category_id == "formatting"


def test_a_cluster_severity_reaches_the_issue_unchanged(categories):
    """No clamping, no "medium" floor — what the model predicted is what is scored."""
    for severity in ("low", "medium", "high"):
        plan = ConsolidationPlan(
            clusters=[Cluster(title="t", description="d", category_id="formatting",
                              severity=severity, finding_indices=[0])]
        )
        findings = [RawFinding(trace_id="t1", title="t", description="d",
                               category_id="formatting", severity="low")]
        board = assemble_board(plan, findings, None, categories)
        assert board.issues[0].severity == severity


# -- out-of-vocabulary coercion survives, and is counted ------------------


def test_an_invented_category_is_still_coerced_to_other(categories, capsys):
    """A category the model invented is a real prediction we cannot map — kept
    and coerced, unlike an absent one, which is now impossible."""
    plan = ConsolidationPlan(
        clusters=[Cluster(title="Novel mode", description="d",
                          category_id="sycophancy", severity="high",
                          finding_indices=[0])]
    )
    findings = [RawFinding(trace_id="t1", title="t", description="d",
                           category_id="sycophancy", severity="high")]
    board = assemble_board(plan, findings, None, categories)
    assert board.issues[0].category_id == "other"

    err = capsys.readouterr().err
    assert "coerced 1 out-of-vocabulary" in err
    assert "sycophancy" in err


def test_the_coercion_count_reflects_how_many_clusters_were_coerced(categories, capsys):
    plan = ConsolidationPlan(
        clusters=[
            Cluster(title="A", description="d", category_id="sycophancy",
                    severity="high", finding_indices=[0]),
            Cluster(title="B", description="d", category_id="verbosity",
                    severity="low", finding_indices=[1]),
            Cluster(title="C", description="d", category_id="formatting",
                    severity="low", finding_indices=[2]),
        ]
    )
    findings = [
        RawFinding(trace_id=f"t{i}", title=f"t{i}", description="d",
                   category_id="formatting", severity="low")
        for i in range(3)
    ]
    assemble_board(plan, findings, None, categories)
    assert "coerced 2 out-of-vocabulary" in capsys.readouterr().err


def test_no_coercion_line_when_every_category_is_in_vocabulary(categories, capsys):
    plan = ConsolidationPlan(
        clusters=[Cluster(title="A", description="d", category_id="formatting",
                          severity="low", finding_indices=[0])]
    )
    findings = [RawFinding(trace_id="t1", title="t", description="d",
                           category_id="formatting", severity="low")]
    assemble_board(plan, findings, None, categories)
    assert "coerced" not in capsys.readouterr().err
