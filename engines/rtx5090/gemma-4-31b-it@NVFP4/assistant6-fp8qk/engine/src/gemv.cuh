// Decode / verify projections for 1..8 columns: FP8 or NVFP4 weights converted exactly to BF16 in registers,
// BF16 activations (Gemma's hidden states exceed the FP16 range), FP32 tensor-core accumulation (mma.m16n8k16). A fixed K split per call and a fixed-order
// reduction make every column's result independent of how many columns run together.
#pragma once
#include "common.h"

namespace g4 {

enum class GemvEpi : int {
    store = 0,     // y[n * ldy + row] = v
    residual = 1,  // y[n * ldy + row] += v
    geglu = 2,     // rows interleaved in 8-row blocks (gate block, up block): h[n * ldh + i] = gelu_tanh(g) * u (BF16)
    softcap = 3,   // y[n * ldy + row] = 30 * tanh(v / 30) (Gemma's final logit soft-capping)
};

struct GemvOut {
    GemvEpi epi = GemvEpi::store;
    float* y = nullptr;
    int ldy = 0;
    __nv_bfloat16* h = nullptr;
    int ldh = 0;
};

// x: BF16 [N][ldx] (N <= 8). splits > 1: eight warps of a block split K over the same 16 rows and reduce in a
// fixed order (column invariant); `partial` is unused. geglu works with either.
void gemv_fp8(const Fp8Mat& m, const __nv_bfloat16* x, int ldx, int N, const GemvOut& out, int splits, float* partial,
              cudaStream_t stream);
void gemv_fp4(const Fp4Mat& m, const __nv_bfloat16* x, int ldx, int N, const GemvOut& out, int splits, float* partial,
              cudaStream_t stream);
void gemv_set_variant(int v);

}  // namespace g4
