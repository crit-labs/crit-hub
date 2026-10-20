# Qwen3.8-27B @ NVFP4 · RTX 5090

Engines for Qwen3.8-27B, a dense model of 48 gated DeltaNet and 16 gated full-attention layers with
a bundled MTP layer, in NVFP4.

Checkpoints: [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (NVFP4
MLPs in layers 0–55, FP8 per-channel weights elsewhere) and
[`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
(QAT, every Text linear NVFP4). Each engine's `model.json` pins its own checkpoint revision and the
SHA-256 of every file it was built on.

Each engine folder's README has its design, measurements, checks and limits.
