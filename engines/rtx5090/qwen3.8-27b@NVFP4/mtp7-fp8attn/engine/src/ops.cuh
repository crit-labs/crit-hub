// Element-wise, normalization, attention-preparation, gated-delta and head kernels.
#pragma once
#include "common.h"
#include "fp8_kv_attention.h"

namespace kestrel {

// Positions: kernels read the first position from *pos_dev when it is set, else use pos0.
struct Pos {
    const int* dev = nullptr;
    int host = 0;
};

// out[n][0..5120) = table[tok[n]] (BF16 -> FP32); tok on the device, table may be mapped host memory.
void embed(const int* tok, int N, const __nv_bfloat16* table, float* out, cudaStream_t s);
// Zero-centred RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + w), FP32 math, FP16 output.
void rmsnorm_f16(const float* x, int ldx, const float* w, int N, int dim, __half* out, int ldo, cudaStream_t s);
// The same for an FP16 input (MTP embedding / hidden inputs).
void rmsnorm_f16_from_f16(const __half* x, int ldx, const float* w, int N, int dim, __half* out, int ldo, cudaStream_t s);

// Attention preparation for N rows at positions pos..pos+N-1. qkv: FP32 [N][ld] with q|gate (12288), k (1024),
// v (1024). Writes normalized, rotated queries (BF16 [N][6144]) and quantizes normalized, rotated keys and values
// into the FP8 cache. rope_inv: 32 FP32 inverse frequencies.
void attn_prep(Proj qkv, int N, Pos pos, const float* q_norm, const float* k_norm, const float* rope_inv,
               __nv_bfloat16* q_out, const crit::fp8_kv::Cache& cache, cudaStream_t s, bool kv_only = false,
               std::uint8_t* q8 = nullptr, float* q8_scale = nullptr);
// out[n][h*256+d] = attn[n][h*256+d] * sigmoid(gate) with gate = qkv[n][h*512+256+d]; FP16 output.
void attn_gate(const float* attn, Proj qkv, int N, __half* out, cudaStream_t s);

// Gated DeltaNet decode / verify step, deferred commit.
struct GdnLayer {
    const float* conv_w;    // [10240][4]
    const float* a_w;       // unused (BF16 weights below)
    const __nv_bfloat16* in_a;  // [48][5120]
    const __nv_bfloat16* in_b;  // [48][5120]
    const float* neg_a;     // [48] = -exp(A_log)
    const float* dt_bias;   // [48]
    const float* norm_w;    // [128]
};
struct GdnState {
    float* conv;        // [2][3][10240] committed window, oldest first; the prompt path uses buffer 0
    float* conv_stash;  // [2][8][10240] pre-conv inputs of the last two passes
    float* S;           // [48][512][32] recurrent state (thread-owned layout)
    float* stash_k;     // [8][48][128] normalized keys
    float* stash_decay; // [8][48]
    float* stash_delta; // [8][48][128]
};
struct GdnScratch {
    float* qk;    // [8][16][2][128] normalized q (scaled) and k
    float* v;     // [8][48][128]
    float* gb;    // [8][48][2] decay, beta
};
// qkvz: FP32 [N][16384] projections (qkv | z); xn: FP16 normed hidden [N][5120]. pending: device count of the previous
// round's columns to commit before this step (nullptr = 0). Output FP16 [N][6144] for out_proj.
void gdn_step(const GdnLayer& L, GdnState st, GdnScratch sc, const float* qkvz, const __half* xn, int N,
              const int* pending, __half* out, cudaStream_t s);
// One-kernel step: pass = device pass counter (parity selects the window / stash buffers).
void gdn_fused(const GdnLayer& L, GdnState st, const float* qkvz, const __half* xn, int N, const int* pending,
               const int* pass, __half* out, cudaStream_t s);

// Greedy token per column: argmax over the first kVocabValid logits (lowest id on ties).
void argmax(const float* logits, int ld, int N, int* out, cudaStream_t s);

// Log-softmax gather for scoring: for each row r, logprob of targets[r] and of queries[r][0..nq), and the argmax.
void logprob_rows(const float* logits, int ld, int rows, const int* targets, const int* queries, int nq,
                  float* target_lp, float* query_lp, int* argmax_out, cudaStream_t s);

}  // namespace kestrel
