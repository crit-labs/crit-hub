// Prompt GEMMs on FP16 tensor cores: C = A W^T with FP16 activations A [M][K] and weights W [N][K] as stored
// (BF16, FP8 E4M3 with a per-tensor scale, NVFP4 with E4M3 group-16 scales), converted to FP16 tiles in shared
// memory (exact for E4M3 and scaled E2M1 values). FP32 accumulation. Grouped form for routed experts.
#pragma once
#include "common.cuh"

namespace nmh {

enum class WFmt : int { bf16 = 0, e4m3 = 1, fp4 = 2 };
enum class Epi : int {
    f32 = 0,      // C[r][c] = alpha * acc
    f32_add = 1,  // C[r][c] += alpha * acc
    f16 = 2,      // Cb[r][c] = f16(alpha * acc)
    relu2 = 3,    // Cb[r][c] = f16(relu2(alpha * acc) * out_scale)
};

struct GemmArgs {
    const __half* A = nullptr;
    int lda = 0;
    const int* a_rows = nullptr;     // optional: A row of logical row r is a_rows[r]
    int M = 0, N = 0, K = 0;
    const void* W = nullptr;         // dense weights
    const uint8_t* Ws = nullptr;     // fp4 scales [N][K/16]
    float alpha = 1.f;
    float out_scale = 1.f;
    float* C = nullptr;
    __half* Cb = nullptr;
    int ldc = 0;
    // Grouped (routed experts): tiles[i] = {group, row_begin, row_end}, ntiles on the device; weights of group g at
    // group_base[g] + w_off, scales at group_base[g] + s_off, alpha group_alpha[g * 2 + alpha_idx].
    const int4* tiles = nullptr;
    const int* ntiles = nullptr;
    const uint8_t* const* group_base = nullptr;
    std::size_t w_off = 0, s_off = 0;
    const float* group_alpha = nullptr;
    int alpha_idx = 0;
};

// Dense: grid (ceil(N/128), ceil(M/128)). Grouped: grid (ceil(N/128), max_tiles).
void gemm(WFmt fmt, Epi epi, const GemmArgs& args, int grid_tiles, cudaStream_t stream);

}  // namespace nmh
