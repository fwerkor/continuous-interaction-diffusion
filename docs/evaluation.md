# Evaluation contract

CID evaluation separates model/output quality from interaction behavior. Runtime traces are the
source of truth for interaction timing; task datasets define expected external observations and,
when a tokenizer-specific harness is available, expected display tokens.

## Per-runtime interaction metrics

`cid.metrics.summarize_runtime()` reports both step-level interaction behavior and wall-clock runtime
telemetry. The runtime records explicit tool start, ready, bind, model-step, and trajectory boundary
events so tool service time is separated from scheduler/drain delay. Reported fields include:

- **latent-to-executable steps**: for each `need_id`, the first executable step minus the first step
  where the latent need is visible to the runtime;
- **binding-to-observation steps** and **observation-to-projection steps**;
- external refresh, cognitive projection, cache-hit, and deduplicated-job counts;
- **runtime wall time** and cumulative model-compute time;
- **tool latency** for completed external reads, including the full per-call sample vector plus mean,
  P50, P95, and maximum;
- **tool wait time**: the union of intervals in which at least one external read is pending, and
  **tool wait ratio**: that duration divided by runtime wall time;
- **model-tool overlap** and **latency hidden ratio**: overlap duration divided by tool wait time;
- time-weighted mean and peak concurrent tool calls;
- **ready-to-bind delay**: wall-clock time from an external read becoming ready to its first binding
  observation update, reported as individual samples plus mean and P95;
- cognitive reclamation and compaction counts.

`tool_wait_overlap_s` is retained as a compatibility alias for `model_tool_overlap_s`. All wall-clock
ratios are computed from interval unions, so concurrent calls do not double-count elapsed time.
Open calls that are still pending when a trajectory ends contribute to tool wait and concurrency but
not to the completed-call latency distribution.

The three step latency metrics use CID model/runtime steps, not seconds. A zero value can therefore
mean same-step progress. Wall-clock metrics use `time.monotonic()` and are intended for comparing
actual execution behavior on a fixed evaluation setup.

### Raw trace persistence

Each neural benchmark JSONL row includes `trace_events`. Timestamps are seconds relative to the
trajectory's first trace event rather than process-global monotonic timestamps. This preserves the
full event stream needed to recompute alternate latency, concurrency, cache, deduplication,
lifecycle, or interaction statistics after a benchmark has completed.

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
global observation coverage/staleness, interaction delays, tool-call latency distributions,
wall-clock wait/overlap ratios, ready-to-bind delay, and concurrency. Ratio aggregation is weighted
by elapsed time rather than averaging per-task percentages. Exact-display accuracy is computed only
over tasks for which expected display token IDs were supplied.

## Deterministic dataset replay

`ScheduledReplaySource` and `run_replay_case()` provide a step-exact replay layer for
`TrajectoryExample.events`. Replay uses a logical runtime clock that is distinct from the count of
model forward passes, so an event with `arrival_step=4` becomes visible on logical runtime step 4
regardless of CPU/GPU speed. During quiescence the replay clock may advance to the next scheduled
event without consuming a model step. Dynamic sources return the newest not-yet-observed version
available at that logical step; later refreshes block until a new dataset version becomes visible.
No wall-clock sleep multiplier is used to approximate dataset time. Consequently, wall-clock tool
latencies from `ScheduledReplaySource` describe replay/runtime behavior and must not be presented as
network or live-tool latency. For live-tool efficiency measurements, the same trace and metric
contract should be used with actual source implementations.

The replay runner constructs source descriptors and protected facts from the trajectory, executes
the supplied `CIDPolicy`, then immediately evaluates the resulting runtime trace/final state with
the metric contract above. Model-specific initialization still belongs outside the core: a neural
benchmark harness chooses tokenizer-dependent display IDs, TCT width/capacity, checkpoints, and
dataset splits before calling `run_replay_case()`.
