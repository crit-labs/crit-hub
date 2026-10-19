# Nemotron-3.5-Super-120B-A12B @ NVFP4 · RTX 5090

Engines for Nemotron 3.5 Super, a 120B-total, 12B-active hybrid (`nemotron_h`: 40 Mamba-2 layers,
40 latent-MoE layers with 512 experts top-22 plus a shared expert, 8 attention layers and an MTP
block), in NVFP4. The checkpoint is about 75 GB, 63 GB of it routed experts, so on a 32 GB GPU the
experts live in host RAM.

Checkpoint: [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4);
the engines here run its text path. Each engine's `model.json` pins the revision and the SHA-256 of
every file it was built on.

Decode and prefill are crit Eval scores over six speed cases (ranges are two Eval runs); decode
depends on the CPU and needs about 82 GiB of host RAM. The engine README has the workload, checks and
limits behind them.
