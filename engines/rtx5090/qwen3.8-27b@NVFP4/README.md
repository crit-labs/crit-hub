# Qwen3.8-27B @ NVFP4 · RTX 5090

Engines for Qwen3.8-27B, a dense model of 48 gated DeltaNet and 16 gated full-attention layers with
a bundled MTP layer, in NVFP4: the NVFP4 MLPs in layers 0–55 and FP8 per-channel weights elsewhere.

Checkpoint: [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). Each
engine's `model.json` pins the revision and the SHA-256 of every file it was built on.

Decode and prefill are crit Eval scores over six speed cases; each engine README has the workload,
checks and limits behind them.
