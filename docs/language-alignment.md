# Language Alignment (LA)

LA precedes CID Stage A/B without renaming or replacing the existing Stage 0.
It trains only the original Backbone using the same Prompt + preallocated Display
masked-denoising path that CID will subsequently use. No Thought or tool
controller is included; the result is a normal Hugging Face Backbone directory.

## Input and objective

* Student: native BOS plus the raw user prompt, followed by a fixed-size
  preallocated Display canvas (all MASK, last token an EOS sentinel).
* Sample a mask probability uniformly between min_mask_probability and 1.
  Reveal some gold answer tokens, train cross-entropy on concealed answer tokens,
  and optionally mask/supervise the gold EOS boundary.
* Unused Display canvas positions remain latent MASK, not padding. The fixed
  sentinel EOS is never treated as a supervised answer token.
* Teacher mode: load an unchanged copy of the instruction-tuned 350M Backbone,
  but apply its own tokenizer chat_template.jinja only to the teacher prompt.
  The teacher and student see identical corrupted Display tokens. The
  temperature-scaled KL term applies only at masked answer-content positions.
  The CID EOS is supervised separately by the student answer labels; the
  template-specific text [/Answer] is not used as CID EOS.
* Native noncausal LFM2/iLLaDA backbones run as Masked LM. Supported causal
  Llama and Qwen3 Base checkpoints undergo the existing CID bidirectional
  conversion and CID mask-token initialization before LA training.
  Ordinary autoregressive SFT would not match this architecture.

## Data and artifacts

Use cid prepare-stage-c-data to export public chat/instruction data to
prompt/answer JSONL; review dataset provenance, licenses, length/language
balance and deduplicate by prompt before training.
The same reader also accepts messages records. Do not include hidden reasoning.

Example, smoke test on the original diffusion instruction model:

    cid train-language-alignment --model LiquidAI/LFM2.5-Encoder-350M-Diffusion \
      --data data/la-train.jsonl --validation-data data/la-valid.jsonl \
      --output-dir outputs/la-350m-sft --max-steps 500 --grad-accum 4 \
      --canvas-tokens 64 --max-answer-tokens 64

    cid train-language-alignment --model LiquidAI/LFM2.5-Encoder-350M-Diffusion \
      --native-teacher LiquidAI/LFM2.5-Encoder-350M-Diffusion \
      --distillation-weight 0.2 --temperature 2 \
      --data data/la-train.jsonl --validation-data data/la-valid.jsonl \
      --output-dir outputs/la-350m-distilled --max-steps 500 --grad-accum 4 \
      --canvas-tokens 64 --max-answer-tokens 64

Each la-step-NNNNNN/ directory contains a full save_pretrained Backbone,
tokenizer and la_manifest.json. Pass this directory as --model to the
existing Stage A/B launcher; initialize new CID modules as usual. Do not
resume a previous CID checkpoint with a different LA Backbone.

## Validation requirements

Evaluate matching raw prompt vs native-template teacher on held-out user
questions and ground-truth answers, plus full native decoding for short/long,
Chinese/English, code, EOS/repetition and tool grounded cases. Compare pure
SFT and teacher KL with identical data, seeds, steps and evaluation harness.

LA reduces initial distribution mismatch but does not guarantee language
retention during Stage B. Test Stage B language replay separately before
calling the full pipeline ready.

For a reproducible pilot sample from the public 350M model's original
instruction-data source (the default exporter replaces the inaccessible
Nemotron v4 dataset):

    cid prepare-language-alignment-data \
      --model LiquidAI/LFM2.5-Encoder-350M-Diffusion \
      --dataset mlabonne/open-perfectblend \
      --max-examples 2500 --max-scan 60000 \
      --output-dir data/la-pilot

This writes deduplicated train/validation JSONL and a dataset manifest.
Use the filenames la-train.jsonl and la-validation.jsonl from that directory.
Review upstream source licenses before distributing any derivative dataset.
