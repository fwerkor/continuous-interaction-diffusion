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

## Deterministic dataset replay

`ScheduledReplaySource` and `run_replay_case()` provide a step-exact replay layer for
`TrajectoryExample.events`. Replay uses a logical runtime clock that is distinct from the count of
model forward passes, so an event with `arrival_step=4` becomes visible on logical runtime step 4
regardless of CPU/GPU speed. During quiescence the replay clock may advance to the next scheduled
event without consuming a model step. Dynamic sources return the newest not-yet-observed version
available at that logical step; later refreshes block until a new dataset version becomes visible.
No wall-clock sleep multiplier is used to approximate dataset time.

The replay runner constructs source descriptors and protected facts from the trajectory, executes
the supplied `CIDPolicy`, then immediately evaluates the resulting runtime trace/final state with
the metric contract above. Model-specific initialization still belongs outside the core: a neural
benchmark harness chooses tokenizer-dependent display IDs, TCT width/capacity, checkpoints, and
dataset splits before calling `run_replay_case()`.
