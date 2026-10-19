// Prompt-path NVFP4 W4A4 routed experts: activation quantization into the Sm1xx block-scaled layout and the
// per-chunk copy of resident experts' weight scales into that layout (fp4_grouped.h runs the GEMMs).
//
// Activations are quantized as the checkpoint's ModelOpt calibration intends: one static global scale per MoE layer
// and projection (input_scale, shared by all experts), gs = 1 / input_scale; per 16 values a UE4M3 block scale
// e4m3(amax / 6 * gs) and E2M1 codes of x * gs / scale (round to nearest even, saturating). Each row is quantized on
// its own, so a token's result does not depend on the other tokens of the chunk.
#pragma once
#include "common.cuh"

namespace nmh {

constexpr int kUpK16 = kLatent / 16;    // 64 scale columns of the up rows
constexpr int kDownK16 = kInter / 16;   // 168 scale columns of the down rows
constexpr std::size_t kSfUpBytes = std::size_t(kInter) * kUpK16;      // 172,032
constexpr std::size_t kSfDownBytes = std::size_t(kLatent) * kDownK16; // 172,032
constexpr std::size_t kSfExpertBytes = kSfUpBytes + kSfDownBytes;

// wsf[e]: expert e's up then down weight scales in the Sm1xx layout, for every e with group_base[e] != nullptr.
void swizzle_expert_scales(const uint8_t* const* group_base, uint8_t* wsf, cudaStream_t stream);

// Gathered rows r of the routed tiles (tile = {expert, r0, r1}): source row x[route_token[r]] (FP16, kLatent wide)
// quantized to xq[r][kLatent / 2] with scales at xsf + sf_offset(sfrow[e] + r - offsets[e], b, kUpK16).
void quant_gather_fp4(const __half* x, const int* route_token, const int4* tiles, int ntiles, const int* offsets,
                      const int* sfrow, float gs, uint8_t* xq, uint8_t* xsf, cudaStream_t stream);

// relu2 of the up outputs (BF16 [rows][kInter], gathered order) quantized to aq[r][kInter / 2] and asf.
void relu2_quant_fp4(const __nv_bfloat16* up, const int4* tiles, int ntiles, const int* offsets, const int* sfrow,
                     float gs, uint8_t* aq, uint8_t* asf, cudaStream_t stream);

// Dense rows: x[T][K] (FP16) or relu2(in[T][K]) (BF16) quantized to q[T][K / 2] with scales sf in the Sm1xx layout
// (rows padded to 128). K multiple of 64.
void quant_rows_fp4(const __half* x, int T, int K, float gs, uint8_t* q, uint8_t* sf, cudaStream_t stream);
void relu2_quant_rows_fp4(const __nv_bfloat16* in, int T, int K, float gs, uint8_t* q, uint8_t* sf, cudaStream_t stream);
// Linear [rows][k16] E4M3 weight scales to the Sm1xx layout.
void swizzle_rows(const uint8_t* src, int rows, int k16, uint8_t* dst, cudaStream_t stream);
// h[i] += d[i]
void add_f32(float* h, const float* d, std::size_t n, cudaStream_t stream);

}  // namespace nmh
