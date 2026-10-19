# Qwen3.8-Flash-Next @ FP4-Local · RTX 5090

Engines for Qwen3.8-Flash-Next, a mixture-of-experts model (`qwen4_exp`: 48 layers of gated delta-net
and QSA attention, 512 routed experts top-10 plus a shared expert, an MTP block) in NVIDIA's FP4-Local
encoding: NVFP4 experts with CSF-encoded block scales, MXFP8 attention and shared-expert projections.
The checkpoint is about 102 GB, so on a 32 GB GPU the routed experts live in host RAM.

Checkpoint: [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local).
Each engine's `model.json` pins the revision and the SHA-256 of every file it was built on.

Decode and prefill are crit Eval scores over six speed cases; the engine needs about 82 GiB of host
RAM. Its README has the workload, checks, known issues and limits behind them.
