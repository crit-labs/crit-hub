// Per-layer kernels for one token (decode) and their multi-token forms where they are elementwise.
#pragma once
#include "common.cuh"

namespace fnl {

// Per-step facts of one decode position, written by the host into pinned memory and copied to the device by the
// step graph, so captured kernels never bake a position in.
struct Step {
    int pos;          // position of the input token
    int token;        // input token id
    int nb;           // complete index blocks visible to the token ((pos + 1) / 4)
    int sparse;       // nb > kTopBlocks: attend the selected blocks
    int nsel;         // blocks attended (kTopBlocks when sparse, else nb)
    int tail_start;   // first token of the open block (nb * 4)
    int nkeys;        // nsel * 4 + (pos + 1 - tail_start)
    int pad;
};
constexpr int kMaxSplits = (kTopBlocks * kBlockTokens + kBlockTokens + 63) / 64;   // 33
constexpr int kMaxTokens = 4;      // tokens of one decode pass (verify of 3 drafts)
constexpr int kRawRing = 8;        // raw index keys kept per layer, at position % 8 (two blocks)
// E4M3 of x / scale with scale = amax / 448 over a 256-dim vector (1 when the vector is zero).
__device__ __forceinline__ uint8_t kv_e4m3(float x, float scale) {
    const __nv_fp8_e4m3 v(x / scale);
    return v.__x;
}
constexpr std::size_t kPartialStride = std::size_t(kKvHeads) * kMaxSplits * kGroup * (2 + kHeadDim);

// ---- hyper-connections ----
// xn[t][s][i] = H[t][s][i] * rsqrt(mean_i H[t][s]^2 + eps) * w1p[s*kHidden+i]  (w1p = 1 + weight)
void hc_norm(const float* H, const float* w1p, float* xn, int tokens, cudaStream_t stream);
// x[t][i] = mean_s m[t][s][i] * xn[t][s][i]
void hc_mix(const float* xn, const float* m, float* x, int tokens, cudaStream_t stream);
// H[t][s][i] += out[t][i] * inj[t][s]
void hc_combine(float* H, const float* out, const float* inj, int tokens, cudaStream_t stream);
// H[t][s][i] = emb[t][i] (bf16 rows gathered from a mapped host table)
void embed_rows(const __nv_bfloat16* table, const int* ids, float* H, int tokens, cudaStream_t stream);

// ---- gated delta net ----
struct GdnStep {
    const float* qkv;      // [T][kQkvDim] projection output
    const float* ab;       // [T][2*kVHeads] (a then b)
    const float* conv_w;   // [kQkvDim][4]
    float* conv_state;     // [kQkvDim][3] (oldest first)
    const float* neg_a;    // [kVHeads] -exp(A_log)
    const float* dt_bias;  // [kVHeads]
    float* q;              // [T][kKHeads][kKDim] normalized and scaled
    float* k;              // [T][kKHeads][kKDim] normalized
    float* v;              // [T][kVHeads][kVDim]
    float* g;              // [T][kVHeads]
    float* beta;           // [T][kVHeads]
    int tokens;
    bool update_state = true;   // false: the state is committed later (commit_conv_states)
};
void gdn_conv(const GdnStep& s, cudaStream_t stream);
constexpr std::size_t kGdnStateFloats = std::size_t(kVHeads) * kKDim * kVDim;
constexpr int kGdnRing = 4;
// Sequential gated delta rule over T tokens. The layer's state lives in a ring of kGdnRing states and the
// current one is ring[*cur]. With snapshots, the state after token t goes to ring[(*cur + 1 + t) % kGdnRing]
// (the caller advances *cur by the tokens it keeps); without, the final state replaces ring[*cur].
void gdn_recur(const float* q, const float* k, const float* v, const float* g, const float* beta, float* ring,
               const int* cur, bool snapshots, float* out, int tokens, cudaStream_t stream);
// After a pass whose first *n tokens are kept: every GDN layer's conv state <- the last 3 of (state, inputs),
// the PLE conv state <- the last kPleState of (state, gvn rows), and *cur advances by *n.
struct Commit {
    float* conv_states;         // [layers][kQkvDim][3]
    const float* inputs;        // [layers][kMaxTokens][kQkvDim] conv inputs of the pass
    int layers;
    float* ple_state;           // [kHyper][kPleState]
    const float* ple_gvn;       // [kMaxTokens][kHyper]
    int* cur;
    const int* n;
};
void commit_states(const Commit& c, cudaStream_t stream);
// out[t][h][d] = norm(o[t][h]) * w[d] * sigmoid(z[t][h][d])
void gdn_gate_norm(const float* o, const float* z, const float* w, float* out, int tokens, cudaStream_t stream);

// ---- attention (QSA) ----
struct AttnPrep {
    const float* proj;          // token t at proj + t * ldp: q|k|v|idx projection
    const float* qn1p, *kn1p;   // [256] 1 + q_norm / k_norm
    const float* iqn1p, *ikn1p; // [128] 1 + indexer norms
    const float* inv_freq;      // [kRotary/2]
    float* q;                   // [T][kQHeads][256] normed + roped
    float* gate;                // [T][kQHeads][256]
    float* iq;                  // [T][kIdxHeads][128]
    uint8_t* kcache;            // layer [kKvHeads][capacity][256] E4M3
    uint8_t* vcache;
    float* kscale;              // layer [kKvHeads][capacity] scale of each key vector
    float* vscale;
    float* rawkeys;             // layer [kRawRing][128] raw index keys at position % kRawRing
    float* blockkeys;           // layer [capacity/4][128]
    int capacity;
    const Step* step;           // device step facts, one per token
    int tokens;
    int ldp;
};
// Normalizes, rotates and caches the T tokens' projections, then pools the index blocks they complete.
void attn_prep(const AttnPrep& a, cudaStream_t stream);
// Token t: scores[t * ld + b] = sum_h relu(iq_t,h . bk_b) / sqrt(128), b < step[t].nb (grid covers max_blocks);
// nothing for a token that is not sparse.
void idx_scores(const float* iq, const float* blockkeys, const Step* step, int max_blocks, float* scores, int ld,
                int tokens, cudaStream_t stream);
// Token t: the top kTopBlocks of scores[t * ld ..] into sel[t * kTopBlocks ..] (ascending); only when sparse.
void topk_blocks(const float* scores, int ld, const Step* step, int* sel, int tokens, cudaStream_t stream);
struct AttnDecode {
    const float* q;             // [T][kQHeads][256]
    const uint8_t* kcache, *vcache;  // layer [kKvHeads][capacity][256] E4M3
    const float* kscale, *vscale;    // layer [kKvHeads][capacity]
    const int* sel;             // [T][kTopBlocks] selected blocks of sparse tokens
    const Step* step;           // [T]
    int capacity;
    float* partial;             // [T][kPartialStride]
    const float* gate;          // [T][kQHeads][256]
    float* out;                 // [T][kAttnOut]
    int tokens;
};
void attn_decode(const AttnDecode& a, cudaStream_t stream);

// ---- MoE helpers ----
// act[i] = silu(g[i]) * u[i]
void silu_mul(const float* g, const float* u, float* act, int n, cudaStream_t stream);

// ---- PLE ----
void ple_dequant(const uint8_t* rows, float s2, float* e, int tokens, cudaStream_t stream);
struct PleMix {
    float* H;                   // [T][kHyper]
    const float* key;           // [T][kHyper]
    const float* value;         // [T][kHidden]
    const float* nk1p, *nq1p, *nc1p;  // [kHyper]
    const float* conv_w;        // [kHyper][4]
    float* conv_state;          // [kHyper][kPleState] previous normalized gated values, oldest first
    int tokens;
};
// gv, gvn: [T][kHyper] workspace. Without update_state the conv state is committed later (commit_states).
void ple_mix(const PleMix& p, float* gv, float* gvn, cudaStream_t stream, bool update_state = true);

// ---- head ----
// argmax over the first n logits of each row (ties to the lower id): out[r] for rows r < rows (ld apart).
// scratch: [rows][256] values and indices.
void argmax(const float* logits, int n, int* out, float* scratch, int* scratch_idx, cudaStream_t stream,
            int rows = 1, std::size_t ld = 0);

// ---- MTP block helpers ----
// y[r] = x[rows ? rows[r] : r] * rsqrt(mean(x^2) + eps) * w1p over width values (one RMS per row).
void rms_rows(const float* x, int width, const int* rows, const float* w1p, float* y, int count, cudaStream_t stream);
// out[r][i] = bf16 table row ids[r] (one stream).
void embed_single(const __nv_bfloat16* table, const int* ids, float* out, int count, cudaStream_t stream);
// H[r][s][i] = fe[r][i] + fh[r][s][i]
void mtp_fuse(const float* fe, const float* fh, float* H, int count, cudaStream_t stream);

}  // namespace fnl
