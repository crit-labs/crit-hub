// Decode GEMVs: FP32 activations of 1..4 tokens against weights as stored (FP8 E4M3 with a per-tensor scale,
// BF16, NVFP4 with E4M3 group-16 scales and an FP32 global scale). Every weight byte is read once per call
// whatever the token count, and each token's sum uses the same per-lane order for any token count.
#pragma once
#include "common.cuh"

namespace nmh {

enum class WFormat : int { fp8 = 0, bf16 = 1, fp4 = 2 };
enum class Out : int { store = 0, add = 1, relu2 = 2 };

struct GemvSeg {
    const void* w = nullptr;      // rows x cols (fp8 bytes, bf16, or packed fp4 bytes)
    const uint8_t* s = nullptr;   // fp4: rows x cols/16 E4M3 scales
    float alpha = 1.f;            // per-tensor scale (fp8 weight_scale, fp4 weight_scale_2)
    float* y = nullptr;           // token t, row r at y[t * ldy + r]
    int rows = 0;
    int ldy = 0;                  // 0: rows
};
struct Gemv {
    WFormat fmt = WFormat::fp8;
    Out out = Out::store;
    GemvSeg seg[3];
    int nseg = 1;
    int cols = 0;
    const float* x = nullptr;     // token t at x[t * ldx]
    int ldx = 0;                  // 0: cols
    int tokens = 1;
};
void gemv(const Gemv& g, cudaStream_t stream);

}  // namespace nmh
