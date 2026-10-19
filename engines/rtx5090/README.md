# NVIDIA RTX 5090

Engines built for one GeForce RTX 5090: Blackwell (GB202), 32 GB of GDDR7, compute capability 12.0,
compiled for `sm_120a` (`CMAKE_CUDA_ARCHITECTURES=120a`) to use its FP4 and FP8 tensor cores. Every
engine here was built and measured on Windows 11 with WSL2 Ubuntu 24.04 and the CUDA 13.1 toolkit;
the two mixture-of-experts engines also need about 82 GiB of host RAM visible to WSL, because their
experts do not fit in VRAM.

The engines use the whole 32 GB card at their full context, so close other GPU programs before
loading one. Engines for other GPUs go in their own folder next to this one, see
[engines/](../README.md).
