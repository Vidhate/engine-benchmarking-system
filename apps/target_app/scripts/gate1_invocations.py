"""Gate 1 — successful invocations through `langgraph_sdk` only.

Drives the served app exactly the way the Phase-4 harness will: config file in,
SDK calls out, zero knowledge of the app's internals. Prints the session ids so
Gate 2 can find the traces in LangSmith.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _contract import banner, load_contract  # noqa: E402
from langgraph_sdk import get_sync_client  # noqa: E402

PROMPTS = [
    "What is your refund policy for an annual plan?",
    "My notebook has been stuck syncing with a spinner for an hour. What do I do?",
    "I was charged twice in June. Please open a ticket with billing for me.",
]

SESSIONS_FILE = Path(__file__).resolve().parent / ".last_sessions.json"


def final_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return " ".join(block.get("text", "") for block in content or [] if isinstance(block, dict))


def main() -> int:
    contract = load_contract()
    banner("GATE 1 — invocations via langgraph_sdk (config-only knowledge)")
    print(json.dumps(contract, indent=2))

    client = get_sync_client(url=contract["base_url"])
    sessions = []

    for index, prompt in enumerate(PROMPTS, start=1):
        session_id = f"gate1-{uuid.uuid4().hex[:12]}"
        thread = client.threads.create()
        result = client.runs.wait(
            thread["thread_id"],
            contract["assistant_id"],
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {}},
            metadata={"session_id": session_id},
        )
        messages = result["messages"]
        tool_calls = [
            call["name"]
            for message in messages
            for call in (message.get("tool_calls") or [])
        ]
        answer = final_text(messages[-1])
        sessions.append(
            {
                "session_id": session_id,
                "thread_id": thread["thread_id"],
                "prompt": prompt,
                "tool_calls": tool_calls,
            }
        )
        print(f"\n--- invocation {index} -------------------------------------------")
        print(f"prompt      : {prompt}")
        print(f"thread_id   : {thread['thread_id']}")
        print(f"session_id  : {session_id}")
        print(f"messages    : {len(messages)}")
        print(f"tool calls  : {tool_calls}")
        print(f"answer      : {answer[:400]}")
        assert answer.strip(), "empty final response"

    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))
    print(f"\nOK — {len(sessions)} successful invocations; session ids -> {SESSIONS_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
