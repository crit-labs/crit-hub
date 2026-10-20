# Qwen3.8-Flash-Next @ NVFP4 · RTX 5090

Engines for Qwen3.8-Flash-Next, a mixture-of-experts model (`qwen4_exp`: 48 layers of gated delta-net
and QSA attention, 512 routed experts top-10 plus a shared expert, an MTP block) in NVIDIA's NVFP4
encoding: routed experts keep their stored NVFP4 codes and scales, the n-gram table is one row-scaled
FP8 matrix, the attention, gated delta-net, hyper-connection and PLE projections are Q8, and the
token embedding is Q8 with a Q6 output head. The checkpoint is about 180B parameters and the
converted artifact about 119 GB, so on a 32 GB GPU the routed experts live in host RAM.

Checkpoint: [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
Each engine's `model.json` pins the revision and the SHA-256 of every file it was built on.

Each engine folder's README has its design, measurements, checks and limits. The engine needs about
65 GiB of host RAM and about 127 GB of disk for the artifact.
