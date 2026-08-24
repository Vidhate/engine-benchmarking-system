"""App-side helper: emit a board from the Engine's real consolidation code.

Deliberately a separate process from `gate1_contract.py`. The gate script plays
the benchmark: it validates JSON against `benchmark.schemas` and must not import
`engine`. This file plays the app: it imports `engine` and must not import
`benchmark`. Keeping them in one file would collapse the boundary the gate is
supposed to be checking.

    python scripts/_produce_board.py <seed_board.json> <categories.json>  -> board JSON on stdout
"""

from __future__ import annotations

import json
import sys

from engine.consolidate import assemble_board
from engine.models import Category, Cluster, ConsolidationPlan, RawFinding, SeedIssueboard

FINDINGS = [
    RawFinding(
        trace_id="trace-planted-ticket",
        title="Tool error reported to the user as success",
        description="create_ticket returned an error; the answer claims a ticket was created.",
        category_id="tool_misuse",
        severity="high",
        evidence="TicketServiceError: upstream ticketing API returned 503",
        span_id="s-t-2",
        turn_index=0,
    ),
    RawFinding(
        trace_id="trace-planted-truncated",
        title="Answer stops mid-sentence",
        description="The final response is cut off before the instructions finish.",
        category_id="formatting",
        severity="medium",
        evidence="Once enabled, Nimbus caches the",
        span_id="s-o-2",
        turn_index=0,
    ),
]

PLAN = ConsolidationPlan(
    clusters=[
        Cluster(
            title="Tool failure hidden behind a success message",
            description="A failed tool call is reported to the user as having succeeded.",
            category_id="tool_misuse",
            severity="high",
            finding_indices=[0],
            matches_seed_error_id="seed-tool-failure-hidden",
        ),
        Cluster(
            title="Truncated response",
            description="The assistant's answer is cut off mid-sentence.",
            category_id="formatting",
            severity="medium",
            finding_indices=[1],
        ),
    ]
)


def main() -> int:
    seed = SeedIssueboard.model_validate(json.loads(sys.argv[1]))
    categories = [Category(**item) for item in json.loads(sys.argv[2])]
    print(assemble_board(PLAN, FINDINGS, seed, categories).model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
