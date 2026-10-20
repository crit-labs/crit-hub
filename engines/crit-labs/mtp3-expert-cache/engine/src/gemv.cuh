// Decode GEMVs: FP32 activations of 1..4 tokens against weights as stored (MXFP8, BF16); each weight byte is
// read once per call whatever the token count.
#pragma once
#include "common.cuh"

namespace fnl {

enum class Post : int { none = 0, silu_quarter = 1, sigmoid = 2, inject = 3 };

// One output segment of a multi-matrix GEMV sharing the input vectors.
struct Mx8Seg {
    const uint8_t* w;   // rows x cols E4M3
    const uint8_t* s;   // rows x cols/32 UE8M0
    float* y;           // token t, row r at y[t * ldy + r]
    int rows;
    int ldy = 0;        // 0: rows
};
struct Mx8Batch {
    Mx8Seg seg[4];
    int nseg;
    int cols;
    const float* x;      // token t at x[t * ldx]
    int tokens = 1;
    int ldx = 0;         // 0: cols
};

void mx8_gemv(const Mx8Batch& b, cudaStream_t stream);

struct Bf16Gemv {
    const __nv_bfloat16* w;  // rows x cols
    const float* x;          // token t at x[t * ldx]
    float* y;                // token t, row r at y[t * ldy + r]
    int rows, cols;
    Post post;
    int tokens = 1;
    int ldx = 0, ldy = 0;    // 0: cols / rows
};
void bf16_gemv(const Bf16Gemv& g, cudaStream_t stream);

}  // namespace fnl
