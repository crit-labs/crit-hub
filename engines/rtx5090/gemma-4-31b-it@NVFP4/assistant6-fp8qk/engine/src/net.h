// g4's model: weights on the device, the target pass for 1..8 columns, the chunked prompt pass, the Gemma 4
// assistant drafter and a device speculative round. All work runs on the backend's compute stream.
#pragma once
#include <crit/cuda.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "attn.cuh"
#include "common.h"
#include "lt.h"
#include "ops.cuh"

namespace g4 {

struct Config {
    int context = 262144;      // KV capacity in tokens
    int prefill_chunk = 2048;  // prompt tokens per chunk (<= ring - window - kMaxCols)
    int ring = 3200;           // sliding-window cache slots per layer
    int draft = 6;             // drafter tokens per round (0 = plain decoding)
    int slide_blocks = 8;      // decode attention blocks per sliding KV head
    int global_blocks = 40;    // decode attention blocks per full-attention KV head
    int splits = 8;            // K split of the 5376-row decode projections
    bool row_fp4 = true;
    bool head_fp4 = false;
    bool prefill_q8 = true;    // full-attention prompt attention over stored key codes (rotated-key cache)
    bool prefill_qb = false;   // ... with BF16 queries against the codes (else FP8 queries, FP8 Q K^T)     // tied output head re-encoded to NVFP4 (else FP8 per row)       // prompt MLP activations: NVFP4 with a per-row global scale (else the checkpoint's static scale)
    std::string gemm_table;    // recorded cuBLASLt algorithm indices (file)
};

struct LayerW {
    bool global = false;
    const float* in_norm = nullptr;
    const float* post_attn = nullptr;
    const float* pre_ff = nullptr;
    const float* post_ff = nullptr;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    const float* scalar = nullptr;  // layer_scalar (device)
    float scalar_host = 1.f;
    Fp8Mat qkv, o;  // FP8 per-row re-encodings of the BF16 projections: q|k|v (sliding) or q|k (global); o_proj
    Fp4Mat gu, down;  // checkpoint NVFP4 MLP (gate/up interleaved in 8-row blocks)
};

struct DraftLayerW {
    bool global = false;
    const float* in_norm = nullptr;
    const float* post_attn = nullptr;
    const float* pre_ff = nullptr;
    const float* post_ff = nullptr;
    const float* q_norm = nullptr;
    const float* scalar = nullptr;
    Fp4Mat q, o, gu, down;
};

struct DraftW {
    Fp4Mat pre, post, head;  // pre_projection [1024][10752], post_projection [5376][1024], tied head [262144][1024]
    DraftLayerW layers[kDLayers];
    const float* norm = nullptr;
};

// Device control block of one request (int32).
namespace ctl {
constexpr int pos = 0;        // position of the first column of the next target pass
constexpr int last = 2;       // last emitted token (input of the next pass)
constexpr int emitted = 3;    // tokens emitted by the last round
constexpr int budget = 4;     // output allowance of the round
constexpr int hcol = 5;       // column of the last pass whose final hidden feeds the drafter, plus one
constexpr int forced_at = 6;  // replay: index of the next forced token
constexpr int stopped = 7;    // a stop token was emitted
constexpr int fields = 16;
}  // namespace ctl

class Net {
public:
    Net(std::shared_ptr<crit::core::CudaBackend> backend, const std::string& model_dir, const std::string& draft_dir,
        const Config& config);
    ~Net();
    const Config& config() const { return cfg_; }
    cudaStream_t stream() const { return stream_; }

    void reset();
    // Prompt tokens [p0, p0 + T) (host ids). When `last` is set, the final position's token is computed.
    void prefill(const int* tokens, int T, int p0, bool last);
    int first_token();            // synchronizes
    void force_first(int token);  // replay: the continuation decides the first token

    // One speculative round (or one plain step when draft is 0), captured as a CUDA graph. Afterwards the pinned
    // result holds {emitted, tokens[draft + 1]}.
    void launch_round(int budget, bool replay, int draft = -1);
    const int* round_result() const { return result_host_; }
    crit::core::CudaEvent& round_event() { return round_done_; }
    void set_forced(const int* tokens, int n);
    void truncate_round(int accepted);
    void set_stops(const std::vector<int>& stops);
    void set_sampling(float temperature, float top_p, int top_k, std::uint64_t seed);

    // Scoring: log-probabilities of ids[i], i >= first, through `path` (0 prefill, 1 decode, 2 verify).
    void score(const int* ids, int n, int first, int path, const int* queries, int nq, float* target_lp, float* query_lp,
               int* argmax_out);

    std::size_t bytes_per_pass() const { return bytes_per_pass_; }
    std::size_t draft_bytes_per_step() const { return draft_bytes_; }
    std::size_t device_bytes() const { return device_bytes_; }
    std::string describe() const;

private:
    void load(const std::string& dir, const std::string& draft_dir);
    void alloc_buffers();
    void* dalloc(std::size_t bytes);
    // Target pass over N columns: tokens d_tok_ at positions ctl[pos]..; logits -> d_logits_ (soft-capped),
    // greedy -> d_pred_, final normed hidden -> d_final_.
    void target_pass(int N, cudaStream_t s);
    // One drafter step: token *tok (device), hidden d_din_ + kHidden (BF16), at position ctl[pos]. Draft -> *out;
    // next hidden -> d_din_ + kHidden.
    void draft_step(const int* tok, int* out, cudaStream_t s);
    cudaGraphExec_t build_round_graph(int draft, bool replay, bool sampling);
    void drop_graphs();
    void head_rows(const __nv_bfloat16* xn, int rows, float* logits, cudaStream_t s);
    void prompt_chunk(int n, int pos0);
    SlideCache slide_cache(int layer) const;
    GlobalCache global_cache(int layer) const;

    std::shared_ptr<crit::core::CudaBackend> backend_;
    Config cfg_;
    cudaStream_t stream_ = nullptr;
    std::vector<crit::core::CudaBuffer> owned_;
    std::unique_ptr<Lt> lt_;

    // Weights.
    std::vector<LayerW> layers_;
    DraftW dw_{};
    Fp8Mat head_{};
    Fp4Mat head4_{};
    void head_gemv(const __nv_bfloat16* x, int N, float* logits, cudaStream_t s);
    const float* final_norm_ = nullptr;
    const __nv_bfloat16* embed_ = nullptr;  // mapped pinned host memory
    float embed_scale_ = 1.f;
    float* inv_s_ = nullptr;  // 128 sliding inverse frequencies
    float* inv_g_ = nullptr;  // 64 full-attention inverse frequencies
    float2* rope_ = nullptr;  // [context + 64][64] full-attention (cos, sin) table
    std::size_t bytes_per_pass_ = 0, draft_bytes_ = 0, device_bytes_ = 0;

    // State.
    std::vector<SlideCache> slide_;   // per layer (null for full-attention layers)
    std::vector<GlobalCache> glob_;   // per layer (null for sliding layers)
    int* ctl_ = nullptr;

    // Decode buffers (8 columns).
    float* d_hidden_ = nullptr;
    __nv_bfloat16* d_xn_ = nullptr;
    float* d_proj_ = nullptr;
    __nv_bfloat16* d_q_ = nullptr;
    float* d_attn_ = nullptr;
    __nv_bfloat16* d_attn_h_ = nullptr;
    __nv_bfloat16* d_mlp_ = nullptr;
    float* d_v_ = nullptr;
    float* d_partial_ = nullptr;
    float* d_logits_ = nullptr;
    __nv_bfloat16* d_final_ = nullptr;  // target final normed hidden [8][5376]
    int* d_tok_ = nullptr;
    float2* d_amx_ = nullptr;  // argmax slice results
    int* d_pred_ = nullptr;
    int* d_draft_ = nullptr;
    int* d_result_ = nullptr;
    int* d_stops_ = nullptr;
    int* d_forced_ = nullptr;
    int forced_cap_ = 0;
    float* d_sampling_ = nullptr;
    int* d_cand_ids_ = nullptr;
    float* d_cand_p_ = nullptr;
    int* d_cand_n_ = nullptr;
    // Drafter buffers (one column).
    float* d_demb_ = nullptr;           // FP32 scaled embedding [5376]
    __nv_bfloat16* d_din_ = nullptr;    // [embedding | hidden] BF16 [10752]
    float* d_dx_ = nullptr;             // drafter residual [1024]
    __nv_bfloat16* d_dxn_ = nullptr;    // [1024]
    float* d_dh_ = nullptr;             // post_projection output FP32 [5376]
    int* result_host_ = nullptr;
    crit::core::CudaBuffer result_pinned_;
    crit::core::CudaEvent round_done_;

    // Prompt buffers.
    int chunk_pad_ = 0;
    int* p_tok_ = nullptr;
    float* p_hidden_ = nullptr;
    std::uint8_t* p_xq_ = nullptr;
    float* p_xs_ = nullptr;
    __nv_bfloat16* p_raw_ = nullptr;
    __nv_bfloat16* p_raw2_ = nullptr;
    __nv_bfloat16* p_q_ = nullptr;
    float* p_attn_ = nullptr;
    std::uint8_t* p_aq_ = nullptr;
    float* p_as_ = nullptr;
    std::uint8_t* p_x4_ = nullptr;
    std::uint8_t* p_x4sf_ = nullptr;
    std::uint8_t* p_hq_ = nullptr;
    float* p_rs1_ = nullptr;  // per-row 1 / global scale of the NVFP4 MLP input and GeGLU output
    float* p_rs2_ = nullptr;
    std::uint8_t* p_h4sf_ = nullptr;
    __nv_bfloat16* p_final_ = nullptr;  // final normed hidden rows [chunk][5376]

    std::map<int, cudaGraphExec_t> graphs_;
    bool sampling_ = false;
};

}  // namespace g4
