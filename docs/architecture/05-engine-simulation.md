# Stage IV — Engine Under Test (Simulated)

Real Engine isn't available to run, so a **coding agent with custom instructions** simulates
it: a prompt-based deep-agent with tool-call access and instructions on how to report errors.

```
[N, M, T*]  →  [N, E_P]
ablated traces   predicted issueboard
```

The simulated Engine is a *benchmark consumer*, not part of the benchmark itself — anything
that reads the trace JSON and emits an issueboard in the schema can be scored. This is what
makes the model comparison (Sol vs 5.1-mini) a drop-in swap.

## High-level flow

```mermaid
flowchart LR
    T["AblatedTraceDataset [N,M,T*]<br/>(ablation markers stripped)"] --> W["Per-trace analysis agent"]
    SEED["Seed Issueboard<br/>(possibly empty)"] --> W
    W --> RL["Running error list<br/>(raw findings per trace)"]
    RL --> META["Meta deep-agent<br/>consolidation pass"]
    META --> OUT["Predicted Issueboard [N,E_P]<br/>(updated over seed)"]
```

```python
class EngineConfig(BaseModel):
    model: str                       # "sol" | "5.1-mini" | ...
    system_prompt: str               # the 'custom instructions' simulating Engine
    tools: list[str]                 # trace-inspection tool names exposed to the agent
    max_tool_calls_per_trace: int
    seed: int

def run_engine(traces: TraceDataset, seed_board: Issueboard,
               cfg: EngineConfig) -> Issueboard:
    """[N,M,T*] → [N,E_P]. Sequential per-trace analysis + meta consolidation."""
```

## Atomic view: per-trace analysis pass

Runs sequentially on every trace, maintaining a running list of raw findings.

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant A as Analysis Agent (LLM)
    participant TL as Trace Tools
    loop each trace in [N,M,T*]
        O->>A: analyze(trace_id) + running context<br/>(known issue titles so far)
        A->>TL: get_trace / list_spans / read_span / search_text
        TL-->>A: trace fragments
        A->>A: identify failures in this trace
        A-->>O: raw findings [{title, description,<br/>category, severity, evidence}]
        O->>O: append to running error list
    end
```

```python
def analyze_trace(trace: Trace, running_titles: list[str],
                  cfg: EngineConfig) -> list[RawFinding]:
    """One trace → zero or more raw findings (unconsolidated)."""

class RawFinding(BaseModel):
    trace_id: str
    title: str
    description: str
    category_id: str
    severity: Literal["low", "medium", "high"]
    evidence: str                    # cited span/snippet
```

## Atomic view: meta consolidation pass

A second deep-agent pass clusters raw findings into issues — mirroring real Engine's
"identify clusters of issues" behavior — and merges with the seed board.

```mermaid
flowchart TB
    RL["Running error list<br/>(per-trace raw findings)"] --> CL["Cluster findings<br/>same failure mode → one Issue"]
    SEED["Seed Issueboard"] --> MERGE
    CL --> DEDUP["Write canonical Issue per cluster<br/>(title, description, category, severity)"]
    DEDUP --> MERGE["Merge with seed board:<br/>attach occurrences to existing issues<br/>or add new issues"]
    MERGE --> OUT["Issueboard(source=engine_predicted)<br/>issues + occurrences {trace_id, error_id}"]
```

```python
def consolidate(findings: list[RawFinding], seed_board: Issueboard,
                cfg: EngineConfig) -> Issueboard:
    """Cluster raw findings into canonical Issues, merge over the seed board,
    emit occurrences — the {trace_id, error_id} matrix scoring consumes."""
```

## Model comparison harness (Sol vs 5.1-mini)

The assignment's final question is answered by running the identical benchmark twice:

```mermaid
flowchart LR
    T["[N,M,T*] + seed board<br/>(identical inputs)"] --> E1["run_engine(cfg model=sol)"]
    T --> E2["run_engine(cfg model=5.1-mini)"]
    E1 --> P1["[N,E_P] (sol)"]
    E2 --> P2["[N,E_P] (5.1-mini)"]
    GT["[N,E_K]"] --> S1["score()"] 
    GT --> S2["score()"]
    P1 --> S1
    P2 --> S2
    S1 --> CMP["Side-by-side<br/>BenchmarkReports"]
    S2 --> CMP
```

Everything except `EngineConfig.model` is held constant (same prompt, tools, seed, traces),
so report deltas are attributable to the model.

## Anti-leak guarantees (what the simulated Engine must NOT see)

- `ablation_ids`, `AblationRecord`s, ground-truth issueboard — stripped before Stage IV.
- The `C_E` taxonomy given to Engine's prompt is the **public category vocabulary only**
  (category names/descriptions), never the concrete injected error definitions.
- No access to the original (pre-ablation) traces — diffing them would trivially reveal
  every injection.
