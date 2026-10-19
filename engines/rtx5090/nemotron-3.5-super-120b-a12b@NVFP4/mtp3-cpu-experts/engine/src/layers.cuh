// Small per-layer kernels shared by the decode and prompt paths.
#pragma once
#include "common.cuh"

namespace nmh {

// Per-step facts of one decode token, copied to the device by the pass graph so kernels never bake a position in.
struct Step {
    int pos;     // position of the input token
    int token;   // input token id
};

// out[t][i] = table[ids[t]][i] (bf16 rows of a device-mapped host table).
void embed_rows(const __nv_bfloat16* table, const int* ids, float* out, int tokens, cudaStream_t stream);
// Same with the ids in Step records.
void embed_steps(const __nv_bfloat16* table, const Step* steps, float* out, int tokens, cudaStream_t stream);
// y[t] = x[t] * rsqrt(mean(x[t]^2) + eps) * w over kHidden (FP32 out, or FP16 / BF16 rows).
void rmsnorm(const float* x, const float* w, float* y, int tokens, cudaStream_t stream, int ldy = kHidden);
void rmsnorm_h(const float* x, const float* w, __half* y, int tokens, cudaStream_t stream, int ldy = kHidden);
void rmsnorm_bf(const float* x, const float* w, __nv_bfloat16* y, int tokens, cudaStream_t stream);
void to_bf16(const float* x, __nv_bfloat16* y, std::size_t n, cudaStream_t stream);
void to_f16(const float* x, __half* y, std::size_t n, cudaStream_t stream);
// y[i] += x[i]
void add_inplace(float* y, const float* x, std::size_t n, cudaStream_t stream);
// Keys/values of T decode tokens (FP32 rows of the q|k|v projection, row t at proj + t * ld) into the FP8 cache at
// steps[t].pos; queries converted to BF16 q16[t][kQDim].
struct KvCache {
    uint8_t* k;      // [capacity][kKvHeads][kHeadDim] E4M3
    uint8_t* v;
    float* ks;       // [capacity][kKvHeads]
    float* vs;
};
void kv_write_steps(const KvCache& c, const float* proj, int ld, const Step* steps, __nv_bfloat16* q16, int tokens,
                    cudaStream_t stream);
// E4M3 codes of T FP32 rows (row t at x + t * ldx, K values) with one scale per row: scale = max |x| / 448
// (1 for a zero row), q[t][k] = e4m3(x / scale) saturating.
void quant_rows_fp8(const float* x, int ldx, int K, int T, uint8_t* q, float* rscale, cudaStream_t stream);
// out[t][n] = d[t][n] * rscale[t] * alpha (add: out += ...), rows of N values (d and out with their own strides).
void scale_rows(const float* d, int ldd, const float* rscale, float alpha, float* out, int ldo, int N, int T, bool add,
                cudaStream_t stream);
// For each of `rows` logits rows (stride ld): tlp[r] = log p(targets[r]), qlp[r][j] = log p(queries[r][j]),
// amax[r] = argmax (lowest id on ties); any output pointer may be null.
void row_scores(const float* logits, int ld, int valid, int rows, const int* targets, const int* queries, int per,
                float* tlp, float* qlp, int* amax, cudaStream_t stream);

}  // namespace nmh
