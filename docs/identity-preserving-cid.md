# Identity-preserving CID (experimental)

The optional identity-preserving architecture keeps the pretrained language backbone's masked-token distribution unchanged when the cognitive controller elects not to reason and no external information is available. It preserves CID Thought/Display iterative updates and typed Source/Fact/Percept ABI while computing the pretrained backbone once per model step.

## Internal thinking requires no tool

Each step has two logical phases that share one Backbone pass:

1. Thought queries attend to the Prompt, **current Display**, existing Thought, and independently routed Sources/Facts/Percepts. Original CID heads predict provisional Thought, local uncertainty, lifecycle, need/action logits, and **new internal Thought-cell allocations**.
2. Display cross-attention reads the newly updated Thought representation, including cells selected for allocation **in the same step**. No Source or Tool is required. Next step uses the persistent TCT state and revised Display as inputs, permitting further recurrent internal calculation.

The selection is the same prefix allocation rule as CIDMaterializer, including FP32 sigmoid threshold and budget. Straight-through gradient allows Display supervision to help teach when to allocate Thought. Defaults: allocation threshold 0.8, max 32 new cells per step. When overriding Runtime materializer values, update the identity allocation model configuration to match.

New Thought allocations also suppress convergence for the current step so the runtime will execute at least one subsequent cognitive iteration. This is a trainable recurrent computational path; it does not establish that an untrained controller can reason well. Non-tool multi-step reasoning supervision is essential.

## Conditional native-language preservation

- No active or newly allocated Thought and no external evidence: Display logits exactly equal the pretrained native Backbone's logits (or its native-language Stage C adapter, when enabled).
- Any active or newly allocated Thought: bounded learned residual updates Display. Tools are not required.
- Source or Fact: conditional Display revision is permitted.
- Percept: routing masks allow revision only at the addressed Display positions if Thought is inactive.
- Diffusion noise conditions the bounded residual query, never the native token embeddings.
- Existing CID control heads, TCT cell lifetime, semantic updates, materializer, diffusion scheduler and tensor contract are preserved.

## Entry points and compatibility

Stage A: cid train ... --interaction-architecture identity-preserving

Stage B: cid train-full ... --interaction-architecture identity-preserving

Default remains joint for legacy checkpoint compatibility. The flag is saved in adapter_config, so unified CID checkpoints restore the same path. New Stage A/B training is required; old joint weights cannot be treated as an already-trained control head for this architecture.

## Verification and remaining limitations

Branch tests cover exact native language identity, immediate non-tool Thought allocation, recurrent Thought/Display exchange, targeted Percept revisions, Source/Fact handling, gradients, Stage A/B optimizer inclusion, Stage C isolation, checkpoint roundtrip, and legacy ABI.

Stage A/B trainers completed 4 and 2 Tiny LFM actual optimizer updates. Real 350M end-to-end Runtime smoke produced some coherent English/code beginnings, but termination/repetition remained poor with the untrained new controller. These tests prove code wiring, **not** fully trained CID reasoning.

A release gate requires real multi-step non-tool reasoning trajectories, tool benchmarks, multiple random seeds, Chinese and code execution, EOS/loop checking and runtime latency. Do not merge to main based solely on structural identity.

## cid-engine

The new network keeps CIDTensorBatch/CIDTensorOutput and all preexisting Thought diffusion, allocation, Display revision and materialization operator signatures. A kernel or ABI change is not required on current evidence. The independent engine branch is reserved for compatibility regression; CANN requires testing on actual NPU hardware.

## Pretraining gate: preallocated MASK and revisable EOS

New identity-preserving adapters default to the preallocated Display scheme.
The existing joint path remains legacy. A new run begins with an editable
MASK canvas and a movable fallback EOS at its final position:

- Legacy: [MASK, EOS, MASK, ...]
- Preallocated: [MASK, MASK, ..., MASK, EOS]

An earlier EOS predicted by the model is accepted only when every position
preceding it has already been revealed. Otherwise the earlier EOS is deferred
and the previous EOS is retained, preventing incomplete high-confidence
prefixes from prematurely closing the answer. The safeguard applies to both
native cid-engine and PyTorch refinement. Structural insertion/deletion and
revision remain supported. Existing clients providing the legacy initial
canvas are upgraded at the first step for preallocated checkpoints only.

Stage A/B bootstrap states use the exact same preallocated initial canvas.
Supervised partially revealed states reveal each correct answer token with
probability 1-t. When the true EOS remains concealed, only the last canvas
position carries a provisional EOS. Teacher noise is quantized to the same
discrete Runtime schedule k/D with k=1..D, normally D=8. The entire Display
canvas receives the same noise level including latent tail. Unknown Display
states are never supervised as completed empty answers. Closed-loop training
still consumes real prior Display tokens and Runtime noise. Stage C also uses
scheduled partial preallocated masking. Legacy training remains unchanged.

An isolated Tiny LFM smoke test performed six Stage A optimizer steps and
two Stage B closed-loop optimizer steps, with nonzero actual parameter changes
in control attention, Display attention, and Need prediction heads on a
source-backed tool trajectory. This confirms gradient and tool training
connections, not real-world task correctness.

A fixed StageC500 0.4B backbone and newly initialized identity controller
were tested on six paired questions with fixed seeds. Initial EOS1, EOS24,
and EOS-last preallocation converged 2/6, 0/6, and 0/6 respectively. Several
preallocated answers improved their opening phrases, but long output,
termination and Chinese quality remained poor. No Thought cells were
allocated in the 18 runs. This establishes that proper Stage A/B training
is still necessary, not that this untrained checkpoint already solves
language generation. Full results are on ise2 at
/data/cid4b/vnext-experiment-20261101/paired_runtime.json.

### Optional bounded MASK window

A separate compatibility experiment can be enabled with
--display-initialization window --identity-initial-mask-tokens 8.
It uses an eight-token initial MASK window followed by a revisable EOS,
retaining the earlier bounded expansion mechanism. The vNext recommended
training configuration remains the preallocated full-canvas mode.
This selector is stored in adapter_config; the legacy joint default is unchanged.
