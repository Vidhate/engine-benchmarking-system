"""Gate 3 — checkpoint time-travel (the Mode A replay surface).

Runs two turns, forks the thread at the checkpoint that produced turn 1's
answer, rewrites that assistant message, and resumes. The continuation must
build on the rewritten content, which is exactly what the ablation engine's
`replay_edit` mode needs from this app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _contract import banner, load_contract  # noqa: E402
from langgraph_sdk import get_sync_client  # noqa: E402

TURN_1 = "What is your refund policy for an annual plan?"
TURN_2 = "Thanks. Anything different for a monthly plan?"
# A recall question: the point is to see the model reuse what it "said".
TURN_2_REPLAYED = (
    "Sorry, I got distracted and did not read that. Without looking anything up again, "
    "just repeat the refund window in days that you gave me a moment ago."
)
CORRUPTED_ANSWER = (
    "Annual plans on Nimbus Notes are refundable in full within 365 days of the charge "
    "date, and refunds are paid out in Nimbus Credit rather than to your card."
)
CORRUPTION_MARKERS = ("365", "nimbus credit")


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def last_ai_message(messages: list[dict]) -> dict:
    return next(m for m in reversed(messages) if m.get("type") == "ai" and not m.get("tool_calls"))


def main() -> int:
    contract = load_contract()
    banner("GATE 3 — time travel: fork a checkpoint, edit state, resume")
    client = get_sync_client(url=contract["base_url"])
    assistant = contract["assistant_id"]

    thread_id = client.threads.create()["thread_id"]
    print(f"thread_id: {thread_id}")

    def run(message: str, checkpoint: dict | None = None) -> dict:
        return client.runs.wait(
            thread_id,
            assistant,
            input={"messages": [{"role": "user", "content": message}]},
            checkpoint=checkpoint,
        )

    first = run(TURN_1)
    original_answer = text_of(last_ai_message(first["messages"]))
    print(f"\nturn 1 answer (original):\n  {original_answer[:300]}")

    run(TURN_2)

    # --- state persists across separate runs on the same thread ---------------
    state = client.threads.get_state(thread_id)
    print(f"\nmessages on the thread after two runs: {len(state['values']['messages'])}")
    assert len(state["values"]["messages"]) > 4, "thread did not keep state across runs"

    # --- find the checkpoint that ended turn 1 -------------------------------
    history = list(client.threads.get_history(thread_id, limit=100))
    print(f"checkpoints in history: {len(history)}")
    target = None
    for snapshot in reversed(history):  # oldest -> newest
        messages = snapshot["values"].get("messages") or []
        if not messages:
            continue
        last = messages[-1]
        if last.get("type") == "ai" and text_of(last) == original_answer:
            target = snapshot
            break
    assert target is not None, "could not locate the turn-1 checkpoint"
    checkpoint_id = target["checkpoint"]["checkpoint_id"]
    edit_target = last_ai_message(target["values"]["messages"])
    print(f"forking at checkpoint: {checkpoint_id}")
    print(f"editing message id   : {edit_target['id']}")

    # --- fork with edited state ---------------------------------------------
    forked = client.threads.update_state(
        thread_id,
        values={
            "messages": [
                {"role": "ai", "id": edit_target["id"], "content": CORRUPTED_ANSWER}
            ]
        },
        checkpoint={"checkpoint_id": checkpoint_id},
    )
    fork_checkpoint = forked["checkpoint"]
    print(f"fork checkpoint      : {json.dumps(fork_checkpoint)}")

    forked_state = client.threads.get_state(thread_id, subgraphs=False)
    forked_messages = forked_state["values"]["messages"]
    assert text_of(last_ai_message(forked_messages)) == CORRUPTED_ANSWER
    print(f"messages on the fork : {len(forked_messages)} (turn 2 is gone, as expected)")

    # --- resume from the fork ------------------------------------------------
    resumed = run(TURN_2_REPLAYED, checkpoint=fork_checkpoint)
    continuation = text_of(last_ai_message(resumed["messages"]))
    print(f"\ncontinuation after the edit:\n  {continuation[:400]}")

    lowered = continuation.lower()
    hits = [marker for marker in CORRUPTION_MARKERS if marker in lowered]
    print(f"\noriginal answer mentioned '30 days': {'30 days' in original_answer.lower()}")
    print(f"continuation echoes the edited content: {hits}")
    assert hits, "continuation did not build on the edited checkpoint content"
    print("\nOK — forked from an earlier checkpoint, edited state, resumed coherently")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
