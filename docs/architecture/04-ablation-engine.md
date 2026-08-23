# Stage III — The Ablation Engine

Injects known errors `E` into traces and records the injections as ground truth.

```
[N, M, T]  →  [N, M, T*], [N, E_K]
traces        ablated traces  ground-truth issueboard
```

**Goal**: manufacture ground truth by construction — every injected error is fully known
(what, where, how), so scoring against it is exact.

**Nuance**: input traces may already contain unknown hidden errors `E_h`. Ablations must be
compatible with those (see [bias analysis](#hidden-error-bias-analysis)).

## Injection modes (locked design decision)

Naive post-hoc editing of a single span breaks **causal consistency**: downstream spans and
the final response remain consistent with the *original* content, so ablated traces carry
artificial inconsistencies organic failures never produce. The system therefore supports two
injection modes, chosen per error definition:

| | **Mode A — `replay_edit` (default)** | **Mode C — `dependency_fault` (optional)** |
|---|---|---|
| What is injected | The **error manifestation itself** — corrupted content at a turn boundary | The **error mechanism** — a fault in an external dependency |
| How consistency is kept | Turns after the injection point are **regenerated organically** through the app | The whole trace is **regenerated organically** with the fault active |
| Error classes covered | Content errors: hallucination, tone, formatting, instruction violation, wrong/incomplete answers | Mechanism errors: retrieval garbage, tool failure/timeout, empty results, flaky APIs, degraded model |
| Ground truth is exact for | The corrupted content (we wrote it) | The activated mechanism fault (visible in the trace by construction) |
| Target-app requirement | Endpoint accepts **client-supplied conversation history** (stateless invoke) | **Config-level control of external dependencies** (LLM `base_url` proxy, retriever endpoint, tool shims) — no source access needed |

### Mode A — `replay_edit`

Corrupt the content of one turn, then replay the remainder of the conversation through the
real app so everything downstream coherently builds on the corruption.

```mermaid
sequenceDiagram
    participant AE as Ablation Engine
    participant TT as Turn k (target)
    participant A as Target App (via harness client)
    AE->>TT: apply ablation_actions to turn-k response
    AE->>TT: rewrite turn-k internal spans for consistency<br/>(last LLM span output must equal corrupted response)
    loop turns k+1 … M
        AE->>A: invoke(history[0..k] with corrupted turn k, next user msg)
        A-->>AE: fresh response + spans (organic)
    end
    AE->>AE: assemble T* = turns[0..k-1] + corrupted k + regenerated k+1..M
```

- **Single-turn degenerate case** (`M = 1`, the bulk of the corpus): there is no downstream
  to replay — Mode A reduces to a consistency-managed post-hoc edit (corrupt the response,
  rewrite the one producing LLM span to match). Acceptable: the consistency surface is a
  single span.
- **Intra-turn limitation** (why Mode C exists): the endpoint cannot be forced to make a bad
  tool call or garbage retrieval — editing those spans post-hoc would evaporate on replay.
  Mechanism errors are out of Mode A's reach by construction.
- **Target-app contract**: replay requires a stateless endpoint that accepts client-supplied
  message history. A server-side-stateful app cannot be made to "believe" the corrupted
  turn; such apps are Mode-A-eligible only at `k = 0` (corrupt the first exchange).

```python
def apply_replay_edit(trace: Trace, spec: AblationSpec,
                      target: TargetAppClient) -> tuple[Trace, AblationRecord]:
    """Corrupt turn k per spec.ablation_actions, rewrite turn-k spans for
    consistency, organically regenerate turns k+1..M via the harness client."""
```

### Mode C — `dependency_fault`

Activate a fault in an external dependency and re-run the input through the app; the
resulting trace is organically, causally consistent end-to-end — including intra-turn spans.

```mermaid
flowchart LR
    subgraph SHIMS["Fault shims (config-level)"]
        P1["LLM proxy<br/>(base_url swap: truncate, degrade model)"]
        P2["Retriever shim<br/>(irrelevant/empty/stale docs)"]
        P3["Tool wrapper<br/>(error, timeout, corrupted result)"]
    end
    IN["InputSpec (re-run)"] --> APP["Target App"]
    APP <--> SHIMS
    APP --> TR["Regenerated trace<br/>fault visibly activated in spans"]
```

**Ground truth is defined at the mechanism level, not the outcome level.** The known error
is e.g. *"retrieval returned irrelevant documents"* — guaranteed true and trace-observable
by construction — not *"final answer is wrong"*, which the app may dodge by self-correcting
from parametric memory.

**Activation is sufficient; there is no manifestation verifier.** If the app recovers and
the final answer happens to be accurate, the trace still contains a real, flaggable issue
(e.g. answer sourced from training memory instead of the degraded store — an
ungroundedness/retrieval-bypass problem: unauditable, stale-data-prone). Catching activated
mechanism faults regardless of outcome accuracy falls squarely within Engine's
responsibilities, so these traces keep their `E_K` entries. Validation checks **activation
only** (the fault is visible in the trace spans), never outcome.

```python
def apply_dependency_fault(input_spec: InputSpec, spec: AblationSpec,
                           target: TargetAppClient,
                           shims: ShimRegistry) -> tuple[Trace, AblationRecord]:
    """Arm the fault shim per spec.fault_config, re-run the input through the
    app, verify activation is visible in the regenerated trace's spans."""
```

> **Note on trace identity**: both modes produce (partially or fully) *regenerated* traces.
> The ablated dataset replaces the original trace for that input; `E_K` occurrences
> reference the new `trace_id`, and `parent_dataset_id` + `AblationRecord`s preserve lineage.

### Locked decisions (from design review)

1. **Mode A is the default**; Mode C is enabled per target app when config-level dependency
   control exists. The benchmark degrades gracefully to A-only (content errors) without it.
2. **No manifestation verifier** — mechanism-level ground truth + activation check only.
3. `Issue.injection_mode ∈ {replay_edit, dependency_fault}` exists **only on `E_K`
   entries**, used solely for post-hoc analysis of which ablation kind yields fairer
   benchmarks. It is stripped from everything Engine sees, and scorers do not consume it.
4. The submission includes test errors of **both kinds** plus a short commentary on which
   class (content vs mechanism ablation) is more informative for benchmarking Engine.

## Prevalence control — the input-level control/ablate split (locked)

If every trace gets an injection, "flag everything" scores perfect recall and false-positive
measurement collapses. Before any ablation runs, the corpus is split once, up front:

```mermaid
flowchart LR
    IN["All inputs [N]"] --> SPL["seeded, stratified split<br/>(by mode, safe/adversarial, dimension)"]
    SPL --> CTRL["CONTROL set<br/>never ablated, never re-run with shims"]
    SPL --> ABL["ABLATE set<br/>sole population filters run against"]
    ABL --> S4x["step 4: filter → sub-sample → inject"]
    CTRL --> OUTx["ship unmodified in [N,M,T*]"]
```

- **Split at the input level, not the trace level.** `dependency_fault` regenerates traces
  by re-running inputs — a trace-level split would let a "control" trace be silently
  replaced by a shimmed re-run. Each `input_id` is assigned once; control inputs are never
  ablated and never re-run.
- **Seeded and stratified** by input provenance (single/multi-turn, safe/adversarial,
  dimension) so control and ablate sets have matched distributions — otherwise Engine could
  learn distributional tells ("adversarial traces are the injected ones").
- **Filters and the `min_eligible ≥ 5` gate run within the ablate set only** (steps 3–4).
- **Reported, not hidden**: control fraction and per-error injection counts are recorded in
  the benchmark report so precision/recall are interpretable against known base rates.
- **Honesty note**: control traces measure Engine's FP rate on *non-injected* traces, not
  on error-free traces — they still carry `E_h`. Consistent with precision-as-lower-bound.
- The split assignment lives on the ground-truth side (like `injection_mode`) and is
  stripped from everything Engine sees.

## The four-step loop (base case III.A: traces assumed golden)

```mermaid
flowchart TB
    T["TraceDataset [N,M,T]"] --> S1
    CE["Error categories C_E<br/>(taxonomy)"] --> S1
    S1["STEP 1 · Propose errors<br/>agent + SDK tools explore traces<br/>out: [E, C_E]"] --> S2
    S2["STEP 2 · Plan ablations<br/>per error: mode + TraceFilter +<br/>actions / fault config<br/>out: [AblationSpec]"] --> S3
    S3{"STEP 3 · Validate<br/>filter ≥ 5 eligible traces?<br/>mode-specific dry-run clean?"}
    S3 -- "failures surfaced" --> S2
    S3 -- pass --> S4["STEP 4 · Apply<br/>filter → sub-sample → inject (per mode) → record"]
    S4 --> OUT1["AblatedTraceDataset [N,M,T*]"]
    S4 --> OUT2["GT Issueboard [N,E_K]<br/>(trace_id, error_id) + AblationRecords"]
```

### Step 1 — Propose errors from taxonomy + traces

An agent with SDK tools (trace search, span inspection, sampling) explores the trace corpus
and drafts concrete, app-specific errors under each high-level category.

```python
def propose_errors(traces: TraceDataset,
                   categories: list[ErrorCategory],
                   n_per_category: int) -> list[Issue]:
    """in: [N,M,T], C_E → out: [E, C_E]
    Each Issue: {error_id, title, description, category_id, severity∈{low,med,high},
    injection_mode}. Content-shaped errors → replay_edit; mechanism-shaped errors →
    dependency_fault (proposed only if the app's shim registry supports them)."""
```

Grounding proposals in *real traces of this app* (not a generic error list) is what makes
injected errors plausible — they must look like failures this app could actually produce.

### Step 2 — Plan filter + ablation strategy per error

```python
def plan_ablation(traces: TraceDataset, issue: Issue) -> AblationSpec:
    """in: [N,M,T], [E,C_E] → out: AblationSpec
    - filter: predicate steps over trace properties selecting traces where this
      error CAN plausibly exist (e.g. 'has a tool span', 'retrieval returned docs')
    - replay_edit:      ablation_actions — str-in → str-out mutations on the target turn
    - dependency_fault: fault_config — which shim, which fault behavior, which inputs"""
```

### Step 3 — Validate every spec (the quality gate)

```mermaid
flowchart LR
    SPEC["AblationSpec"] --> F{"filter matches<br/>≥ 5 traces?"}
    F -- no --> FAIL["reject: error not<br/>expressible in this corpus"]
    F -- yes --> M{"mode?"}
    M -- replay_edit --> A{"actions run clean on a sample?<br/>turn-k spans consistent?<br/>downstream replay succeeds?"}
    M -- dependency_fault --> C{"shim armable?<br/>fault ACTIVATES —<br/>visible in regenerated spans?"}
    A -- yes --> V{"result schema-valid<br/>& coherent?"}
    C -- yes --> V
    A -- no --> BACK
    C -- no --> BACK
    V -- yes --> PASS["validated spec"]
    V -- no --> BACK["surface all errors<br/>→ back to STEP 2"]
```

```python
def validate_specs(traces: TraceDataset,
                   specs: list[AblationSpec],
                   target: TargetAppClient, shims: ShimRegistry,
                   min_eligible: int = 5) -> tuple[list[AblationSpec], list[ValidationError]]:
    """Dry-run every spec (mode-aware). Valid specs pass through; failures return
    to planning. dependency_fault validation asserts activation, never outcome."""
```

### Step 4 — Apply and record ground truth

```python
def apply_ablations(traces: TraceDataset,
                    specs: list[AblationSpec],
                    target: TargetAppClient, shims: ShimRegistry,
                    seed: int) -> tuple[TraceDataset, Issueboard, list[AblationRecord]]:
    """in: [N,M,T], E → out: [N,M,T*], [N,E_K]
    For each validated spec:
      1. apply filter               → eligible traces (ABLATE SET ONLY)
      2. sub-sample if too large    → target_count traces (seeded RNG)
      3. inject per mode            → replay_edit: corrupt + replay downstream
                                      dependency_fault: arm shim + regenerate trace
      4. store {trace_id, error_id} → IssueOccurrences + AblationRecords (before/after)
    Compound errors: a trace may be selected by >1 spec → carries multiple E_K entries,
    BUT never two errors of the same category (disjointness invariant below)."""
```

Design invariants:

- **Originals are immutable** — ablation writes a new dataset with `parent_dataset_id` set.
- **Full audit trail** — every injection stores its before/after (replay_edit) or fault
  config + activation evidence (dependency_fault).
- **Leak-proofing** — the copy shipped to Engine strips `ablation_ids`, `AblationRecord`s,
  `injection_mode`, and any formatting artifacts that would fingerprint ablated traces.
- **Controlled prevalence** — the control set is untouchable by construction; within the
  ablate set, sub-sampling sets the injection rate per error.
- **Same-category disjointness (the exact-match key)** — two errors of the **same category
  are never injected into the same trace**; enforced during sub-sampling. Compound errors
  on one trace must come from different categories. Consequence: `(trace_id, category_id)`
  uniquely identifies the injected known error, which is what lets scoring match
  predictions to ground truth exactly instead of by text similarity
  (see [06-scoring.md](06-scoring.md#error-matcher--mapping-e_p--e_k)).

## Hidden-error bias analysis (III.B)

Traces already carry unlabeled errors `E_h`. After ablation, the true error set is
`E_h + E_K`, but scoring only knows `E_K`. Three regimes, by how much ablation overwrites
the hidden errors:

```mermaid
flowchart TB
    subgraph R1["① Ablation fully overwrites E_h"]
        A1["E_h ⊂ E_K ⇒ E_K is complete ground truth<br/>TP = E_K∩E_P · FP = E_P−E_K · FN = E_K−E_P<br/>all metrics exact"]
    end
    subgraph R2["② Ablation doesn't touch E_h"]
        A2["measured TP = E_K∩E_P — under-estimated<br/>(actual TP = E_K∩E_P + E_h∩E_P)<br/>measured FP = E_P−E_K — over-estimated<br/>(actual FP = E_P−E_K−E_h)<br/>measured FN = E_K−E_P — under-estimated<br/>(actual FN = (E_K+E_h)−E_P)"]
    end
    subgraph R3["③ Partial overwrite"]
        A3["same direction of bias as ②,<br/>smaller deviation from true values"]
    end
```

| Measured quantity | vs. truth | Why |
|---|---|---|
| True positives `E_K ∩ E_P` | under-estimate | Engine's hits on `E_h` aren't credited |
| False positives `E_P − E_K` | over-estimate | `E_h` hits are miscounted as FP |
| False negatives `E_K − E_P` | under-estimate | misses on `E_h` are invisible |

Note the modes interact with `E_h` differently: `replay_edit` *overwrites* whatever hid in
the corrupted turn and its regenerated tail (pushing toward regime ①/③), while
`dependency_fault` regenerates the whole trace — old `E_h` instances are discarded with the
old trace, but the app may organically produce fresh ones. This is exactly the analysis
`injection_mode` on `E_K` exists to support.

**Conclusions** (from the notes, kept as system principles):

1. Ablation-based ground truth is a **biased estimator** of Engine's performance — expected
   and acceptable, because the bias direction is known (measured precision is a *lower
   bound* on true precision w.r.t. all real issues).
2. The only foolproof way to find `E_h` is **expert annotation**; ablations improve
   continually as annotations reveal previously hidden errors (fold them into `C_E`).
3. Bias magnitude is a direct function of how many `E_h` ablation misses — reducible by
   spending more test-time compute and widening ablation hyperparameters (more categories,
   more proposals per category, overwrite-style ablations that replace regions where `E_h`
   could live).
