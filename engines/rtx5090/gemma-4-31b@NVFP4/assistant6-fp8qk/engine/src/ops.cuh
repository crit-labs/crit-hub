// Element-wise, normalization, attention-preparation and head kernels for decode and prompt passes.
#pragma once
#include "attn.cuh"
#include "common.h"

namespace g4 {

// out[n][0..dim) = table[tok[n]] * scale (BF16 table, FP32 math); table may be mapped host memory.
void embed(const int* tok, int N, const __nv_bfloat16* table, int dim, float scale, float* out, int ldo, cudaStream_t s);
// Gemma RMSNorm: out = x * rsqrt(mean(x^2) + eps) * w, FP32 math, BF16 output.
void rmsnorm_bf16(const float* x, int ldx, const float* w, int N, int dim, __nv_bfloat16* out, int ldo, cudaStream_t s);

// Decode sandwich around a projection output v (FP32 [N][ldv]): y = (y + RMSNorm(v) * w_post) * scalar, then
// h = RMSNorm(y) * w_next as BF16. scalar: device pointer or null (1).
void sandwich(const float* v, int ldv, float* y, int ldy, const float* w_post, const float* scalar, const float* w_next,
              __nv_bfloat16* h, int ldh, int N, int dim, cudaStream_t s);

// Attention preparation for N rows at positions pos + n (pos.dev or pos.host), FP32 or raw BF16 projections.
// Sliding: qkv columns q (8192) | k (4096) | v (4096): queries q_norm + RoPE -> BF16 q_out [N][8192]; keys k_norm +
// RoPE and values v_norm quantized into the ring. Global: q (16384) | k (2048): queries q_norm + proportional RoPE ->
// BF16 q_out [N][16384]; u = v_norm(k) quantized into the cache. which: 0 all, 1 queries only, 2 cache only.
// q_out may be null with which = 2.
void slide_prep(Proj qkv, int N, Pos pos, const float* q_norm, const float* k_norm, const float* inv, __nv_bfloat16* q_out,
                int ldq, const SlideCache& c, int which, cudaStream_t s);
void global_prep(Proj qkv, int N, Pos pos, const float* q_norm, const float* inv, __nv_bfloat16* q_out, int ldq,
                 const GlobalCache& c, int which, cudaStream_t s);

// out[i] = BF16(in[i]) for i < n (n % 4 == 0).
void to_bf16(const float* in, int n, __nv_bfloat16* out, cudaStream_t s);

// Greedy token per column: argmax over `vocab` logits (lowest id on ties).
void argmax(const float* logits, int ld, int vocab, int N, int* out, cudaStream_t s);
// The same in two stages over kArgParts slices per column (scratch: N * kArgParts (value, index) pairs); same result.
constexpr int kArgParts = 64;
void argmax_wide(const float* logits, int ld, int vocab, int N, int* out, float2* scratch, cudaStream_t s);
// Log-softmax gather for scoring: for each row r, logprob of targets[r] and of queries[r][0..nq), and the argmax.
void logprob_rows(const float* logits, int ld, int rows, const int* targets, const int* queries, int nq,
                  float* target_lp, float* query_lp, int* argmax_out, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ prompt path
// RMSNorm(x) * w then per-row FP8 quantization (codes [r][5376], scale[r] = amax / 448).
void rmsnorm_quant_fp8(const float* x, const float* w, int T, std::uint8_t* codes, float* scale, cudaStream_t s);
// Prompt sandwich: hidden[r] = (hidden[r] + RMSNorm(v[r]) * w_post) * scalar, then RMSNorm(hidden[r]) * w_next into
// one of: FP8 rows (codes, scale), static NVFP4 (packed, sf with global scale gs; rows T..T_pad-1 zero) or BF16 rows.
struct RowOut {
    std::uint8_t* codes = nullptr;
    float* scale = nullptr;
    std::uint8_t* packed = nullptr;
    std::uint8_t* sf = nullptr;
    float gs = 0.f;
    float* rscale = nullptr;  // per-row NVFP4: gs_r = 448 * 6 / row amax, writes 1 / gs_r per row (static gs unused)
    int T_pad = 0;
    __nv_bfloat16* bf = nullptr;
};
void sandwich_rows(Proj v, int T, float* hidden, const float* w_post, float scalar, const float* w_next, const RowOut& out,
                   cudaStream_t s);
// GeGLU over interleaved 8-row gate/up blocks of the gate/up GEMM output, quantized to NVFP4 [T_pad][21504]: with
// rscale, per row (global scale 448 * 6 / row amax, 1 / scale written per row), else with the static scale gs.
void geglu_fp4(Proj gu, int T, int T_pad, float gs, float* rscale, std::uint8_t* packed, std::uint8_t* sf, cudaStream_t s);
// Per-row FP8 quantization of an FP32 matrix [T][K].
void quant_fp8_rows(const float* in, int T, int K, std::uint8_t* codes, float* scale, cudaStream_t s);

}  // namespace g4
