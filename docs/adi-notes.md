# Things to think about — assignment questions

Short answers with provenance. Each section links the architecture doc that argues the design and the code that ships it.

## 1. What is the right data structure for a trace?

Organized data structrue in 3 levels : `Trace → Turn → Span`
- **span** is the atomic unit of app behavior (one LLM call, tool call, or retrieval, with inputs/outputs/attributes)
- **turn** groups the spans behind one user↔assistant exchange
- **trace** carries relationships between spans (`input_id`, `parent_dataset_id`) to create order.  

I heavily relied on terminologies on how the field treats traceing with an additional constraint that traces should be ablate-able cleanly.
Clean ablation implies : ability to post-hoc edit a trace *with consistency guarantees* (for ex - execution orders should not change or tool calls should not be left hanging without tool results)
For this version I started dev with post-processing on Langsmith tracing to ETL into my trace dataobject. This was due to a combination of late decisions that would take prohibitively long to backtrack and also speed to submit the assignment.

- Design: [`docs/architecture/01-data-schemas.md`](architecture/01-data-schemas.md)
- Code: [`benchmark/schemas/traces.py:20-50`](../benchmark/schemas/traces.py#L20-L50) (`Span`, `Turn`, `Trace`, `TraceDataset`); boundary in [`benchmark/tracing/store.py`](../benchmark/tracing/store.py)

## 2. What is the right data structure for an issueboard?

IssueBoard is a collection of "Issues" and "Occurrences".  
I modeled an Issue against the info I grasped from our first interview - An **`Issue`** names a *failure mode* (title, category from a shared taxonomy, severity, description);  
An **`IssueOccurrence`** names one *sighting* of it in actual traces on which these issues are being searched in (`error_id`, `trace_id`, optional turn/span + evidence quote).   

The split makes scoring well-behaved later for engine benchmarking: detection ("did you find this failure mode?") and localization ("on which traces?") score independently, and the same board shape serves ground truth and prediction (separated by the value of the `source` field). Category and severity are required predictions with no defaults, so a model can never score points by omission.  
Beyond the Issue itself it's worth to know I also define ErrorCategory - a super high level bucket into which target-app-specific errors can be bucketed (in line again with our interview discussions). ErrorCategory can, for ex, be - "hallucination", "formatting issues" etc.

- Design: [`docs/architecture/01-data-schemas.md`](architecture/01-data-schemas.md)
- Code: [`benchmark/schemas/issues.py:18-47`](../benchmark/schemas/issues.py#L18-L47) (`ErrorCategory`, `Issue`, `IssueOccurrence`, `Issueboard`)

## 3. What is the right evaluation function?

I define four scorers at different hierarchies:  
1. per-error-category detection (were the right categories predicted by Engine for the right traces in which they were ablated?) -> P/R/F1 + Cohen's κ category-level
2. per-error localization (was a specific Issue predicted by Engine on the right IssueOccurrences in which they were ablated?) -> P/R/F1 + Cohen's κ error-level
3. severity calibration under an asymmetric loss (uner-predicting severity is penalized quadratically vs over-predicting is penalized linearly) -> asymmetric loss scalar (can be made higher granular i.e. category-level or error-level)
4. description fidelity (How much does Engine's predicted descriptions of specific Issue deviate from ground truth) -> Tf-iDF vector based cosine similarity (can be made granular and also improved with better models like LLM-based embedding models or a hybrid scorer).

Because the corpus may hold genuine app defects we never injected (E_h), unmatched predictions count against precision *and* surface as an auditable candidate list: **precision is a lower bound, recall over injected errors is unbiased**. (Can explain math behind this more on a call).

- Design: [`docs/architecture/06-scoring.md`](architecture/06-scoring.md) (esp. "Error matcher" and "Interpreting numbers under hidden-error bias")
- Code: matcher [`benchmark/scoring/matcher.py:29`](../benchmark/scoring/matcher.py#L29), scorers [`benchmark/scoring/`](../benchmark/scoring/), disjointness invariant [`benchmark/ablation/apply.py:10-18`](../benchmark/ablation/apply.py#L10-L18)

## 4. How do you create realistic traces?

I define a very detailed Synthetic Input Candidate Generator that can create seed for single-turn or multi-turn convos. An Input Harness drives these candidates through the target app's endpoint. Only a config about the target app needs to be known beforehand to do this (abstractions like LangGraph help make this config == langgraph.json). Realism is controlled at the *input* level — a dimension grid (topic, length, ambiguity, language, complexity, goal) crossed with personas, including adversarial ones (prompt injection, jailbreak, scope creep), LLM-expanded into concrete requests with every cell populated.
I have detailed math on what the tradeoffs here are and why this can be trusted to be realistic given context about any target-app.

Every trace, hence, is a **real execution of a real LangGraph target app** (a two-tool RAG + ticketing support agent) collected through a black-box harness.  Injected ablations are of 2 types : Mode A - what I proposed in my interview - (`replay_edit`) corruptions are span-consistency-managed and leak-scrubbed (timestamp normalization, token scans) so the Engine cannot detect the edit from format artifacts, and Mode C - what Johannes proposed at the end of the interview - (`dependency_fault`) re-runs the app with a faulty dependency so the error is the model's *own organic reaction*

There's a cool nuance in error activation vs manifestation that would be really interestin to discuss in person.

- Design: [`docs/architecture/02-input-generation.md`](architecture/02-input-generation.md), [`docs/architecture/03-trace-harness.md`](architecture/03-trace-harness.md), [`docs/architecture/04-ablation-engine.md`](architecture/04-ablation-engine.md)
- Code: grid expansion [`benchmark/generation/generators.py:210`](../benchmark/generation/generators.py#L210), harness [`benchmark/harness/runner.py:132`](../benchmark/harness/runner.py#L132), leak-stripped export [`benchmark/ablation/export.py`](../benchmark/ablation/export.py)

## 5. How does the methodology scale to many tasks of similar or larger size?

The Benchmarking system I propose has the following qualities which makes it super extensible. 

- It is **Target-App-agnostic**: the benchmark touches the target app only through a YAML config and standard LangGraph server surfaces — swapping in a new app is a new config file, zero benchmark-code changes. See the black-box contract: [`docs/execution-plan.md#L51`](execution-plan.md) and [`configs/target_app.yaml`](../configs/target_app.yaml).
- **Scale of benchmark study is controlled via configs**: synthetic input corpus size, synthetic input variations, number and fraction of ablations / injection counts, and concurrency are all knobs ([`configs/pipeline/submission.yaml`](../configs/pipeline/submission.yaml)); generation is cached and pure-function keyed, the harness and Engine both run batched-parallel, and cost grows roughly linearly in trace count.
- **Long runs are survivable**: every stage checkpoints its artifacts and `--resume` skips completed stages with lineage-hash verification ([`benchmark/pipeline/resume.py`](../benchmark/pipeline/resume.py)) — the 5.4 h submission run survived a mid-harness crash without re-computing from scratch any completed work.
- **New error categories are config updates** - any new error categories can be very easily added to the benchmark system - even opening doors to RSI style self-adding new categories in future. 
- **Target-app specific errors are generated** - specific target app errors are not presumed or known before jumping into the benchmarking run - meaning the injected errors are truly novel, discovered and predicted - imagine them as P(error | category, target-app-context) - and not known before running the benchmark pipeline.

The taxonomy is shared and extensible (`other` as escape hatch + E_h audit trail feeds new categories), and the tracing backend is replaceable behind `TraceStore`, so neither the vocabulary nor LangSmith is a scaling bottleneck.

## 6. Does Sol perform better than 5.1-mini on this task?

Pending. I ran the benchmark against 5.1-mini and am submitting the reports based on the results back from it. I will kick off the engine with Sol today and do a comparison and update the repo once ready.

**Pending — the comparison is designed and one arm is complete.** Arm 1 (`gpt-5-mini`) ran on the full 398-trace corpus: category F1 0.094, matched-error F1 0.315, perfect localization precision on detected errors, systematic misses on failures-of-omission (full numbers: [`data/pipeline/submission/report.md`](../data/pipeline/submission/report.md), analysis: [`docs/paper/main.pdf`](paper/main.pdf)). The comparison harness holds *everything* except the model id constant — same prompts, tools, seeds, traces, and ground truth. Arm 2 (larger model) reuses the frozen traces + ground truth via the copy-then-resume recipe and costs only the Engine pass (~2–3 h); deltas will be reported per-scorer with bootstrap CIs before any comparative claim.

- Design: [`docs/architecture/05-engine-simulation.md`](architecture/05-engine-simulation.md) ("Model comparison harness")
- Code: model as the single comparison axis [`apps/engine/engine/prompts.py:1-7`](../apps/engine/engine/prompts.py#L1-L7), readback verification [`benchmark/pipeline/runner.py:591`](../benchmark/pipeline/runner.py#L591)
