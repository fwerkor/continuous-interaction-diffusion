# Evaluation contract

CID evaluation separates model/output quality from interaction behavior. Runtime traces are the
source of truth for interaction timing; task datasets define expected external observations and,
when a tokenizer-specific harness is available, expected display tokens.

## Per-runtime interaction metrics

`cid.metrics.summarize_runtime()` reports:

- **latent-to-executable steps**: for each `need_id`, the first step where the need is executable
  minus the first step where the latent need is visible to the runtime;
- **binding-to-observation steps**: the first observation availability step (including cache hits)
  minus the first active-binding step;
- **observation-to-projection steps**: the first cognitive projection after an observation becomes
  available minus that observation step;
- external refresh count, cognitive projection count, cache hits, and deduplicated jobs;
- model steps that overlap external I/O and the wall-clock duration of that overlap;
- cognitive reclamation and compaction counts.

The three latency metrics above use CID model/runtime steps, not seconds. A zero value can therefore
mean same-step progress. Wall-clock overlap remains a separate metric because source and model
latencies vary across hardware.

## Task-level freshness and display metrics

`cid.evaluation.evaluate_runtime_result()` combines a `RuntimeResult` with its
`TrajectoryExample`:

- **converged**: whether the runtime reached an accepted terminal state before its budget;
- **exact display**: exact token-ID equality when the benchmark harness supplies expected display
  IDs; the core evaluator deliberately does not own a tokenizer;
- **observation coverage**: observed expected work items divided by unique expected work items;
- **fresh observations**: observed work items matching the latest dataset event for the same
  `(source, canonical arguments)` key;
- **stale observations**: observed work items whose value or explicit version differs from that
  latest expected event;
- **missing observations**: expected work items for which no observation was available;
- **stale observation rate**: stale observations divided by observed work items.

If a trajectory contains multiple versions for the same source/argument work key, only the event
with the greatest `arrival_step` is treated as the final freshness target. This makes dynamic and
streaming tasks measure whether the model/runtime ends on the latest state rather than whether it
ever observed an earlier valid value.

`RuntimeEvaluationSummary` aggregates task results into convergence rate, exact-display accuracy,
global observation coverage/staleness, and mean interaction delays. Exact-display accuracy is
computed only over tasks for which expected display token IDs were supplied.

## Benchmark harness boundary

The core evaluation code is intentionally tokenizer- and benchmark-independent. A benchmark runner
may provide model-specific tokenization, source replayers, wall-clock budgets, and dataset splits,
then feed each `RuntimeResult` into this contract. This keeps RQ metrics stable when the model,
backbone, or task suite changes.
