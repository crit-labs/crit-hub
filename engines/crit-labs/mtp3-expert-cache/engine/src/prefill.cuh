// Kernels used only by the multi-token (prompt) path.
#pragma once
#include "common.cuh"
#include "experts.cuh"
#include "layers.cuh"

namespace fnl {

void to_bf16(const float* x, __nv_bfloat16* y, std::size_t n, cudaStream_t stream);
void to_f16(const float* x, __half* y, std::size_t n, cudaStream_t stream);
// inj[t][s] = 2 sigmoid((w_s . xn[t]) / 4)
void hc_inject(const float* xn, const __nv_bfloat16* w, float* inj, int tokens, cudaStream_t stream);
// sg[t] = sigmoid(w . x[t])
void shared_gate(const float* x, const __nv_bfloat16* w, float* sg, int tokens, cudaStream_t stream);
// Per token softmax / top-10 / renormalize of router logits [T][512].
void route_batch(const float* logits, Route* routes, int tokens, cudaStream_t stream);
// counts[s][e]: routes of tokens in sub-chunk s (sub tokens each) to expert e. counts zeroed by the caller.
void count_routes(const Route* routes, int tokens, int sub, int* counts, cudaStream_t stream);
// For tokens [t0, t1): route_token[off[e] + i] = t, pos_of[t][k] = that position. cursor zeroed by the caller.
// cpu_row[t][k] >= 0: the route is computed on the CPU (pos_of = -1 - row of the CPU results).
void scatter_routes(const Route* routes, int t0, int t1, const int* offsets, int* cursor, int* route_token,
                    int* pos_of, cudaStream_t stream, const int* cpu_row = nullptr);
// out[t] = sum_k w_k route_out[pos_of[t][k]] + sg[t] shared[t], t in [t0, t1); route rows relative to the sub-chunk.
void moe_combine(const Route* routes, const int* pos_of, const float* route_out, const float* shared, const float* sg,
                 float* out, int t0, int t1, cudaStream_t stream, const float* cpu_rows = nullptr);
// Decode CSF scales of n experts (expert id, destination slot) straight from mapped host records.
struct ScaleJob { int expert; uint8_t* slot; };
void decode_scales(const ExpertTables& t, int layer, const ScaleJob* jobs, int n, cudaStream_t stream);

// Attention preparation for T tokens at positions pos0..pos0+T-1 (see AttnPrep for one token).
struct AttnPrepBatch {
    const float* proj;          // [T][kProjAttn]
    const float* qn1p, *kn1p, *iqn1p, *ikn1p, *inv_freq;
    __half* q;                  // [T][kQHeads][256] f16
    float* gate;                // [T][kQHeads][256]
    float* iq;                  // [T][kIdxHeads][128]
    __half* iq3;                // [T][kIdxHeads][384] = [hi, hi, lo] for the index GEMM
    float* raw;                 // [T][128] raw index keys of the chunk
    uint8_t* kcache, *vcache;   // E4M3
    float* kscale, *vscale;     // per key / value vector
    float* rawkeys;             // [kRawRing][128] raw keys at position % kRawRing (the open block's earlier positions)
    float* blockkeys;
    int capacity, pos0, tokens;
};
constexpr int kProjAttn = kQProj + 2 * kKvHeads * kHeadDim + kIdxProj;  // 13952
void attn_prep_batch(const AttnPrepBatch& a, cudaStream_t stream);

// Pools the index keys of blocks completed in the chunk (fp32 and bf16 copies) and keeps the open block's keys.
void pool_blocks(const AttnPrepBatch& a, cudaStream_t stream);
// keys3[b] = [hi, lo, hi] f16 split of the FP32 pooled key of block b < n: with queries [hi, hi, lo] one
// f16 GEMM over K = 384 gives the index score to about FP32 precision.
void split_keys(const float* bk, int n, __half* keys3, cudaStream_t stream);
// Top kTopBlocks per query row (only rows with more complete blocks than kTopBlocks); sel row stride kTopBlocks.
void topk_batch(const float* scores, int ld, int pos0, int q0, int G, int* sel, cudaStream_t stream);
// Gathered attention on tensor cores for queries q0..q0+G-1; sel rows (kTopBlocks) for sparse queries.
struct AttnPrefill {
    const __half* q;            // [T][kQHeads][256]
    const float* gate;          // [T][kQHeads][256]
    const uint8_t* kcache, *vcache;   // E4M3
    const float* kscale, *vscale;
    const int* sel;             // [G][kTopBlocks] relative to q0
    int capacity, pos0, q0, G;
    __half* out;                // [T][kAttnOut] gated
};
void attn_prefill(const AttnPrefill& a, cudaStream_t stream);

}  // namespace fnl
