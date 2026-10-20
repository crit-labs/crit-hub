// Prompt-path kernels: activation quantization for the cuBLASLt GEMMs, epilogues and the chunked gated-delta scan.
#pragma once
#include "common.h"
#include "ops.cuh"

namespace kestrel {


// RMSNorm (1 + w) then per-row FP8 quantization: codes[r][k], scale[r] = amax / 448; optionally also the BF16 rows.
void rmsnorm_quant_fp8(const float* x, const float* w, int T, int K, std::uint8_t* codes, float* scale,
                       __nv_bfloat16* bf_out, cudaStream_t s);
// RMSNorm (1 + w) to BF16 with the tensor's absolute maximum (atomic, *amax must start at 0).
void rmsnorm_bf16_amax(const float* x, const float* w, int T, int K, __nv_bfloat16* out, unsigned* amax, cudaStream_t s);
// Per-row FP8 quantization of a BF16 or FP16 matrix.
void quant_fp8_rows(const __nv_bfloat16* in, int T, int K, std::uint8_t* codes, float* scale, cudaStream_t s);
void quant_fp8_rows(const __half* in, int T, int K, std::uint8_t* codes, float* scale, cudaStream_t s);
// NVFP4 quantization with a dynamic global scale gs = 448 * 6 / amax: packed [T_pad][K/2], swizzled block scales
// (T outer), alpha_out = w_alpha / gs for the GEMM. Rows T..T_pad-1 are zero.
void quant_fp4(const __nv_bfloat16* in, int T, int T_pad, int K, const unsigned* amax, float w_alpha, std::uint8_t* packed,
               std::uint8_t* sf, float* alpha_out, cudaStream_t s);
// hidden[r] += proj(r, :), then RMSNorm (1 + w) of the new row quantized to FP8 per row (codes/scale, optional BF16
// rows) or, when codes is null, written as BF16 with the tensor's absolute maximum (atomic, *amax starts at 0).
void residual_norm(Proj p, int T, float* hidden, const float* w, std::uint8_t* codes, float* scale, __nv_bfloat16* bf_out,
                   unsigned* amax, cudaStream_t s);
// Static-scale NVFP4 variants (gs = checkpoint input_global_scale): residual (optional, p.f32/bf null = none) +
// RMSNorm + NVFP4 quantization of the hidden rows, and SwiGLU + NVFP4 quantization of the interleaved gate/up output.
// Rows T..T_pad-1 are written as zeros.
void residual_norm_fp4(Proj p, int T, int T_pad, float* hidden, const float* w, float gs, std::uint8_t* packed,
                       std::uint8_t* sf, cudaStream_t s);
void swiglu_fp4(Proj p, int T, int T_pad, float gs, std::uint8_t* packed, std::uint8_t* sf, cudaStream_t s);
// hidden[r][c] += proj(r, c) for c < 5120.
void residual_add(Proj p, int T, float* hidden, cudaStream_t s);
// SwiGLU over interleaved 8-row gate/up blocks: out[r][i] (BF16, [T][17408]) and its absolute maximum.
void swiglu_bf16(Proj p, int T, __nv_bfloat16* out, unsigned* amax, cudaStream_t s);
// attention output * sigmoid(gate) (gate from the q projection), quantized per row to FP8.
void attn_gate_quant(const float* attn, Proj qkv, int T, std::uint8_t* codes, float* scale, cudaStream_t s);

// Gated DeltaNet prompt path for one chunk of T tokens continuing from the committed state (pending must be 0).
struct GdnPrefillScratch {
    float* qk;   // [T][16][2][128]
    float* v;    // [T][48][128]
    float* gb;   // [T][48][2]
    float* o;    // [T][48][128]
};
// qkvz projection (q|k|v|z, 16384 columns); xn: the normed hidden as BF16 [T][5120] for the a/b gates.
// ab: FP32 [T][96] gate projections (a for value heads 0..47, then b).
struct WyScratch;
// wy: chunked WY scan scratch, or null for the sequential scan.
void gdn_prefill(const GdnLayer& L, GdnState st, GdnPrefillScratch sc, Proj qkvz, const float* ab, int T,
                 std::uint8_t* codes, float* scale, const WyScratch* wy, cudaStream_t s);

void set_scan3(bool on);  // prompt scan: 3 value-column slices per head (default) or 2

}  // namespace kestrel
