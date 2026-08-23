"""Gate: all schemas round-trip serialize/deserialize with validation."""

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from benchmark.schemas import (
    AblationAction,
    AblationRecord,
    AblationSpec,
    AblationSplit,
    BenchmarkReport,
    CategoryScore,
    Dimension,
    EngineConfig,
    ErrorCategory,
    ErrorMatch,
    FaultConfig,
    FilterStep,
    GenerationConfig,
    InputDataset,
    InputSpec,
    Issue,
    Issueboard,
    IssueOccurrence,
    OccurrenceMatch,
    Persona,
    Span,
    Trace,
    TraceDataset,
    TraceFilter,
    Turn,
)

NOW = datetime(2026, 8, 23, 12, 0, 0, tzinfo=UTC)


def make_trace(trace_id: str = "t1", input_id: str = "i1") -> Trace:
    span = Span(
        span_id="s1",
        name="retriever.search",
        span_type="retrieval",
        start_time=NOW,
        end_time=NOW,
        inputs={"query": "refund policy"},
        outputs={"docs": ["doc-a"]},
        attributes={"k": 3},
    )
    turn = Turn(turn_index=0, user_message="hi", final_response="hello", spans=[span])
    return Trace(trace_id=trace_id, input_id=input_id, mode="single_turn", turns=[turn])


def make_input_dataset() -> InputDataset:
    dim = Dimension(dim_id="d1", name="query_topic", kind="safe", variations=["refunds"])
    persona = Persona(persona_id="p1", name="angry", kind="adversarial", description="…")
    spec = InputSpec(
        input_id="i1", mode="single_turn", dim_id="d1", variation="refunds", prompt="hi"
    )
    cfg = GenerationConfig(safe_dims=[dim], adversarial_personas=[persona], seed=7)
    return InputDataset(created_at=NOW, generation_config=cfg, inputs=[spec])


def make_issueboard(source: str = "ground_truth") -> Issueboard:
    issue = Issue(
        error_id="e1",
        title="hallucinated refund policy",
        description="fabricates a 90-day window",
        category_id="hallucination",
        severity="high",
        injection_mode="replay_edit" if source == "ground_truth" else None,
    )
    occ = IssueOccurrence(error_id="e1", trace_id="t1", turn_index=0)
    return Issueboard(source=source, issues=[issue], occurrences=[occ])


def make_report() -> BenchmarkReport:
    return BenchmarkReport(
        engine_config=EngineConfig(model="gpt-5.1-mini"),
        base_rates={"control_fraction": 0.3},
        matcher_fallback_rate=0.05,
        occurrence_matches=[
            OccurrenceMatch(trace_id="t1", predicted_error_id="p1", resolved_error_id="e1")
        ],
        matches=[ErrorMatch(predicted_error_id="p1", matched_error_id="e1", overlap=3)],
        category_scores=[
            CategoryScore(
                category_id="hallucination",
                precision=1.0,
                recall=0.5,
                f1=0.667,
                cohens_kappa=0.6,
                support=4,
            )
        ],
        severity_loss=0.25,
        description_scores={"p1": 0.8},
        headline={"macro_f1": 0.7},
    )


ROUNDTRIP_CASES = [
    make_trace(),
    TraceDataset(traces=[make_trace()]),
    make_input_dataset(),
    make_issueboard(),
    make_issueboard("engine_predicted"),
    ErrorCategory(category_id="hallucination", name="hallucination", description="…"),
    AblationSpec(
        error_id="e1",
        mode="replay_edit",
        filter=TraceFilter(steps=[FilterStep(field="turns[0].spans", op="exists")]),
        ablation_actions=[
            AblationAction(target="turns[0].final_response", transform="llm_rewrite")
        ],
    ),
    AblationSpec(
        error_id="e2",
        mode="dependency_fault",
        fault_config=FaultConfig(shim="retriever", target="docs", behavior="irrelevant_docs"),
    ),
    AblationRecord(
        ablation_id="a1", error_id="e1", trace_id="t1", before_after=[("old", "new")]
    ),
    AblationSplit(seed=7, control_fraction=0.3, control_input_ids=["i1"], ablate_input_ids=["i2"]),
    make_report(),
]


@pytest.mark.parametrize("obj", ROUNDTRIP_CASES, ids=lambda o: type(o).__name__)
def test_roundtrip(obj: BaseModel):
    restored = type(obj).model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_validation_rejects_bad_enum():
    with pytest.raises(ValidationError):
        Issue(
            error_id="e1",
            title="x",
            description="y",
            category_id="c",
            severity="catastrophic",  # not in {low, medium, high}
        )


def test_validation_rejects_bad_span_type():
    with pytest.raises(ValidationError):
        Span(
            span_id="s1",
            name="x",
            span_type="database",
            start_time=NOW,
            end_time=NOW,
        )
