# Gemma-4-31B @ NVFP4 · RTX 5090

Engines for Gemma 4 31B, a dense model of 50 sliding-window and 10 full-attention layers (the full
layers' keys equal their values), in NVFP4: NVFP4 MLPs with BF16 attention projections and a tied
BF16 output head. Text only; the vision tower is not used.

Checkpoint: [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4), the
instruction-tuned model. Each engine's `model.json` pins the revision and the SHA-256 of every file it
was built on.

Each engine folder's README has its design, measurements, checks and limits. The drafter is
downloaded separately.
