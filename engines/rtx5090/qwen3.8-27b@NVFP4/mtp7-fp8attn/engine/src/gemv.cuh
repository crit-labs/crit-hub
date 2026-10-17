// Decode / verify projections for 1..8 columns: FP8 or NVFP4 weights converted exactly to FP16 in registers,
// FP16 activations, FP32 tensor-core accumulation (mma.m16n8k16). A fixed K split per call and a fixed-order
// reduction make every column's result independent of how many columns run together.
#pragma once
#include "common.h"

namespace kestrel {

enum class GemvEpi : int {
    store = 0,     // y[n * ldy + row] = v
    residual = 1,  // y[n * ldy + row] += v
    swiglu = 2,    // rows interleaved in 8-row blocks (gate block, up block): h[n * ldh + i] = silu(g) * u (FP16)
    residual_norm = 3,  // y[n * ldy + row] += v, then h[n * ldh + :] = RMSNorm(y row) * (1 + norm_w) (needs splits > 1)
};

struct GemvOut {
    GemvEpi epi = GemvEpi::store;
    float* y = nullptr;
    int ldy = 0;
    __half* h = nullptr;
    int ldh = 0;
    const float* norm_w = nullptr;
    float* ssq = nullptr;  // residual_norm scratch: N * (rows / 16) floats
};

// x: FP16 [N][ldx] (N <= 8). splits > 1: eight warps of a block split K over the same 16 rows and reduce in a
// fixed order (column invariant); `partial` is unused. swiglu needs splits == 1.
void gemv_fp8(const Fp8Mat& m, const __half* x, int ldx, int N, const GemvOut& out, int splits, float* partial,
              cudaStream_t stream);
void gemv_fp4(const Fp4Mat& m, const __half* x, int ldx, int N, const GemvOut& out, int splits, float* partial,
              cudaStream_t stream);

}  // namespace kestrel
