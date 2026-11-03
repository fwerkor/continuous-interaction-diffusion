# LFM2.5 Encoder 350M Diffusion: native CID compatibility

For the original LiquidAI/LFM2.5-Encoder-350M-Diffusion only, choose
backbone_protocol="lfm-native" and interaction_architecture="identity-preserving".
Load the original model through LFMCIDAdapter.from_pretrained with
freeze_backbone=True and pass its tokenizer to ILLaDAContextTensorizer.

The original tokenizer's chat template is applied without added vocabulary.
CID reserves a physical logical EOS sentinel at the end of the Display, and
excludes this sentinel from the native Backbone forward. For N native MASK
tokens, allocate N+1 physical CID Display cells. The original native forward
logits and position IDs for the N generated cells then match in neutral state.

The specialized decoder keeps the original one-token, highest-confidence
reveal for MASK cells, while the CID revision controller can reopen visible
tokens after a persistent confidence gain. External evidence invalidates old
confidence anchors and can reopen a completed answer; a shifted suffix can
trigger a bounded insertion/deletion without discarding the stable text.
Inference and detached training rollout both apply the same revision gates.
It suppresses the same special tokens as the original reference. The original
lexical answer terminator is recognized internally and mapped to logical CID
EOS, without changing the required on-wire CID output protocol.

Limitations:
- Exact native language parity applies only in a neutral Thought/tool state.
  TCT, tools, memory and control residuals still need separate Stage A/B
  training and rollout validation.
- The lexical terminator must be filled and all earlier MASK cells resolved
  before the bridge finalizes. Intermediate partial markup may be visible.
- Ordinary native decoding reveals one new token per CID update, excluding
  explicit CID revisions; callers must budget enough steps for complete
  responses and recovery after late tool evidence. The normal eight-step CID
  budget is too small for a long native response.
- If native EOS is predicted while earlier MASKs are unresolved, the decoder
  chooses an alternative non-EOS token; this rare case can differ from native.
- Compare quality against the same original native model, not against a
  larger language model. Keep prior KL-trained checkpoints as experimental
  baselines; do not replace the original pretrained Backbone weights.
