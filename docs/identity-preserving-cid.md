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
