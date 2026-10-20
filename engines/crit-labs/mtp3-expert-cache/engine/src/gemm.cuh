// Prompt GEMMs on FP16 tensor cores: C = A W^T with FP16 activations A [M][K] and weights W [N][K] as stored
// (BF16, MXFP8 or NVFP4), converted to FP16 tiles in shared memory (exact for MXFP8/NVFP4 values in the FP16 range).
// FP32 accumulation.
#pragma once
#include "common.cuh"

namespace fnl {

enum class WFmt : int { bf16 = 0, mx8 = 1, fp4 = 2, f16 = 3 };
enum class Epi : int {
    f32 = 0,          // C[r][c] = alpha * acc
    bf16_siluq = 1,   // Cb[r][c] = f16(silu(acc / 4))
    f32_sigmoid = 2,  // C[r][c] = sigmoid(acc)
    act = 3,          // B rows interleave gate/up in groups of 8; Cb[r][f] = f16(silu(ag acc_g) * (au acc_u))
    bf16 = 4,         // Cb[r][c] = f16(alpha * acc)
    idx = 5,          // C[r / 4][c] = sum over the 4 rows of a query of relu(acc) / sqrt(128)
};

struct GemmArgs {
    const __half* A = nullptr;
    int lda = 0;
    const int* a_rows = nullptr;     // optional: A row of logical row r is a_rows[r]
    int M = 0, N = 0, K = 0;
    // Dense weights: W/Ws (and W2/Ws2 for the up half of an act GEMM).
    const void* W = nullptr;
    const uint8_t* Ws = nullptr;
    const void* W2 = nullptr;
    const uint8_t* Ws2 = nullptr;
    float alpha = 1.f, alpha2 = 1.f;
    float* C = nullptr;
    __half* Cb = nullptr;
    int ldc = 0;
    // Grouped (routed experts): tiles[i] = {group, row_begin, row_end}, ntiles on the device.
    const int4* tiles = nullptr;
    const int* ntiles = nullptr;
    uint8_t* const* group_base = nullptr;   // slot base per group (expert)
    std::size_t w_off = 0, w2_off = 0, s_off = 0, s2_off = 0;  // byte offsets of the matrices in a slot
    const float* group_alpha = nullptr;     // per group: alpha at [group * 3 + alpha_idx]
    int alpha_idx = 0, alpha2_idx = 0;
};

// Dense: grid (ceil(N/128), ceil(M/128)). Grouped: grid (ceil(N/128), max_tiles).
void gemm(WFmt fmt, Epi epi, const GemmArgs& args, int grid_tiles, cudaStream_t stream);

}  // namespace fnl
