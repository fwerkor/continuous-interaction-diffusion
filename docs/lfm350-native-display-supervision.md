# LFM-native complete Display supervision (experimental)

This is a new supervision contract for the LFM2.5-Encoder-350M-Diffusion CID adapter. It is not a validated trained model.

## Why the previous objective fails

A position-invariant minimum cross-entropy over a short answer span provides gradients only to the selected span. A one-token label in a 64-token canvas leaves 63 positions without language-target gradients. Falling Display loss is therefore compatible with nonsense or repeated output elsewhere.

## New target contract

A trajectory must contain a **factually checked complete response** in `metadata.native_full_display`. A complete intermediate draft differing from the final answer requires its own `metadata.native_full_display_by_step[step]` entry. Unknown drafts remain unsupervised; their Thought / Need supervision remains.

Example (add these fields to an ordinary existing trajectory):

```json
{
  "target_display": "131 ms",
  "metadata": {
    "native_full_display": "The documentation reports a latency of 131 ms.",
    "native_full_display_verified": true,
    "native_full_display_reviewer": "verified-grounded-teacher"
  }
}
```

The full reply is tokenized with `\n[/Answer]` appended; the lexical marker is learned while the physical CID EOS remains hidden from the backbone. An actual LFM tokenizer check is essential: concatenating `.[/Answer]` directly changes the BPE tokens and the runtime may not recognize the stop sequence.

No fixed **answer token** position is imposed; its position is determined by the verified full response. Full-reply denoising is the primary language objective. The old location-free answer loss contributes at 0.1 weight only if the answer token sequence is actually found in the full reply tokenization. Incompatible tokenization or verified semantic paraphrases must not produce contradictory token targets.

The independently reviewed metadata field `native_full_display_verified: true` permits semantic paraphrases without literal answer text. This flag must be set only by an external trusted reviewer / verifier; the loader cannot verify paraphrases independently.

An independently verified response may be a semantic paraphrase if the native_full_display_semantic_paraphrase flag is true.

The full reply must be sourced from genuinely correct grounded teacher answers (including the observed tool value). Do **not** generate it from ungrounded base-model guesses. Do **not** silently use `target_display` as a complete natural reply for the v1 dataset. An additional diverse language-replay mixture is recommended and must be evaluated for semantic and linguistic quality.

## Preflight

```bash
python scripts/validate_native_display_data.py <training.jsonl>
python scripts/validate_native_display_data.py <validation.jsonl>
```

To merge verified teacher reply JSONL, provide records containing
example_id, full_reply, verified=true, reviewer, and optional by_step.
Run python scripts/merge_native_full_replies.py --trajectories original.jsonl --verified-replies reviewed.jsonl --output annotated.jsonl.
The merger uses atomic output writing and rejects missing or unverified rows.

The validator returns nonzero if any complete answer lacks a checked full reply. It can also be used with `--max-examples 1000` for quick audits. The actual neural tensorizer enforces this contract again.

## Gating

Start full training only after independent full-reply data auditing, no-answer-before-observation causal checks, unit/gradient tests, a native-model control, and **held-out end-to-end** comparisons with both tools and ordinary chat. Compare with the original native LFM and last released CID model. Reject configurations that regress language quality, grounding, or tool routing even if training / teacher-forced loss falls.

The historical Stage A / Stage B checkpoints trained under the prior fragment loss are not repaired by this code change and require fresh training.
