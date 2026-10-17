// kestrel's model: weights on the device, the target pass for 1..8 columns, the chunked prompt pass, the MTP drafter
// and a device speculative round. All work runs on the backend's compute stream.
#pragma once
#include <crit/cuda.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "fp8_kv_attention.h"
#include "lt.h"
#include "ops.cuh"
#include "gdn_wy.cuh"
#include "prefill_ops.cuh"

namespace kestrel {

struct Config {
    int context = 262144;      // KV capacity in tokens
    int prefill_chunk = 4096;  // prompt tokens per chunk
    int draft = 4;             // MTP draft tokens per round (0 = plain decoding)
    int attn_blocks = 40;      // decode attention blocks per KV head
    int splits = 8;            // K split of the 5120-row decode projections
    bool static_fp4 = true;
    bool attn_q8 = true;       // prompt attention with FP8 Q K^T
    bool attn_pv8 = true;      // ... and FP8 P V (E4M3 probabilities scaled by the value scales), 64-key tiles
    bool decode_q8 = false;    // decode / verify attention with FP8 Q K^T (queries quantized in attention prep); measured slower in rounds
    bool wy_scan = false;      // chunked WY gated-delta scan for prompts (FP32; slower than the staged scan here)
    std::string gemm_table;    // recorded cuBLASLt algorithm indices (file)    // NVFP4 prompt activations with the checkpoint's static input global scales
};

struct LayerW {
    bool attn = false, fp4 = false;
    const float* in_norm = nullptr;
    const float* post_norm = nullptr;
    Fp8Mat qkv, o;  // attention: q|gate, k, v (14336 rows); o_proj
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    Fp8Mat qkvz, out;  // DeltaNet: qkv|z (16384 rows); out_proj
    GdnLayer gdn{};
    Fp4Mat gu4, down4;  // NVFP4 MLP (gate/up interleaved in 8-row blocks)
    Fp8Mat gu8, down8;  // FP8 MLP
};

struct MtpW {
    Fp8Mat fc8, kv8;                    // prompt path: fc and the key/value rows of qkv (FP8 per row)
    Fp4Mat fc, qkv, o, gu, down, head;  // drafting: NVFP4 re-encodings; head = NVFP4 copy of lm_head
    const float* in_norm = nullptr;
    const float* post_norm = nullptr;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    const float* norm = nullptr;
    const float* pre_emb = nullptr;
    const float* pre_hidden = nullptr;
};

// Device control block of one request (int32).
namespace ctl {
constexpr int pos = 0;        // position of the first column of the next target pass
constexpr int pending = 1;    // committed columns of the last pass still to fold into recurrent state
constexpr int last = 2;       // last emitted token (input of the next pass)
constexpr int emitted = 3;    // tokens emitted by the last round
constexpr int budget = 4;     // output allowance of the round
constexpr int mtp_cols = 5;   // MTP catch-up columns of the next round
constexpr int forced_at = 6;  // replay: index of the next forced token
constexpr int stopped = 7;    // a stop token was emitted
constexpr int pass = 8;       // target passes since the prompt (parity selects DeltaNet buffers)
constexpr int fields = 16;
}  // namespace ctl

class Net {
public:
    Net(std::shared_ptr<crit::core::CudaBackend> backend, const std::string& model_dir, const Config& config);
    ~Net();
    const Config& config() const { return cfg_; }
    cudaStream_t stream() const { return stream_; }

    // New request: zero recurrent state, pending and positions.
    void reset();
    // Prompt tokens [p0, p0 + T) (host ids). When `last` is set, the final position's greedy token is computed and
    // MTP catch-up for the last position is staged. Returns nothing; the first token is read with first_token().
    void prefill(const int* tokens, int T, int p0, bool last);
    int first_token();  // synchronizes
    void force_first(int token);  // replay: the continuation decides the first token

    // One speculative round (or one plain step when draft is 0), captured as a CUDA graph. Afterwards the pinned
    // result holds {emitted, tokens[draft + 1]}. budget = output allowance; forced = replay continuation or null.
    void launch_round(int budget, bool replay, int draft = -1);
    const int* round_result() const { return result_host_; }  // valid after round_event() completes
    crit::core::CudaEvent& round_event() { return round_done_; }
    // Replay continuation (device copy) for replay rounds.
    void set_forced(const int* tokens, int n);
    // Truncate the last round to `accepted` tokens (Core trimmed at a stop the device did not know).
    void truncate_round(int accepted);
    // Stop tokens checked on the device.
    void set_stops(const std::vector<int>& stops);
    // Sampled decoding (chat): temperature / top-p / top-k / seed; temperature 0 = greedy.
    void set_sampling(float temperature, float top_p, int top_k, std::uint64_t seed);

    // Scoring: log-probabilities of ids[i], i >= first, through `path` (0 prefill, 1 decode, 2 verify).
    void score(const int* ids, int n, int first, int path, const int* queries, int nq, float* target_lp, float* query_lp,
               int* argmax_out);

    std::size_t bytes_per_pass() const { return bytes_per_pass_; }
    std::size_t draft_bytes_per_step() const { return draft_bytes_; }
    std::string describe() const;

private:
    void load(const std::string& dir);
    void alloc_buffers();
    void* dalloc(std::size_t bytes);
    // Target pass over N columns: tokens d_tok_ at positions ctl[pos]..; logits -> d_logits_, greedy -> d_pred_,
    // final normed hidden -> d_final_.
    void target_pass(int N, cudaStream_t s);
    void mtp_pass(int N, const int* tok_dev, const __half* hidden_dev, int hidden_ld, Pos pos, cudaStream_t s);
    void mtp_head(int col, int* out_tok, cudaStream_t s);
    cudaGraphExec_t build_round_graph(int draft, bool replay, bool sampling);
    void drop_graphs();
    void head_rows(const __half* xn, int rows, float* logits, cudaStream_t s);
    void prompt_chunk(int n, int pos0, bool has_next);

    std::shared_ptr<crit::core::CudaBackend> backend_;
    Config cfg_;
    cudaStream_t stream_ = nullptr;
    std::vector<crit::core::CudaBuffer> owned_;
    std::unique_ptr<Lt> lt_;

    // Weights.
    std::vector<LayerW> layers_;
    MtpW mtp_{};
    Fp8Mat head_{};
    const float* final_norm_ = nullptr;
    const __nv_bfloat16* embed_ = nullptr;  // mapped pinned host memory
    float* rope_inv_ = nullptr;
    std::size_t bytes_per_pass_ = 0, draft_bytes_ = 0;

    // State.
    std::vector<crit::fp8_kv::Cache> kv_;  // per attention layer
    crit::fp8_kv::Cache mtp_kv_{};
    std::vector<GdnState> gdn_;             // per DeltaNet layer
    int* ctl_ = nullptr;                    // device control block

    // Decode buffers (8 columns).
    float* d_hidden_ = nullptr;
    __half* d_xn_ = nullptr;
    float* d_proj_ = nullptr;
    __nv_bfloat16* d_q_ = nullptr;
    float* d_attn_ = nullptr;
    __half* d_gated_ = nullptr;
    __half* d_mlp_ = nullptr;
    float* d_partial_ = nullptr;
    float* d_ssq_ = nullptr;
    std::uint8_t* d_q8_ = nullptr;   // FP8 queries for the FP8 decode attention [8][24][256]
    float* d_q8s_ = nullptr;
    float* d_attn_partial_ = nullptr;
    float* d_logits_ = nullptr;
    __half* d_final_ = nullptr;      // target final normed hidden [8][5120]
    __half* d_mtp_in_ = nullptr;     // [8][10240]
    __half* d_mtp_h_ = nullptr;      // MTP output hidden (after mtp.norm) [8][5120]
    int* d_tok_ = nullptr;           // target pass input tokens [8]
    int* d_pred_ = nullptr;          // target greedy predictions [8]
    int* d_draft_ = nullptr;         // drafts [8]
    int* d_mtp_tok_ = nullptr;       // MTP input tokens [8]
    int* d_result_ = nullptr;        // {emitted, tokens[8]}
    int* d_stops_ = nullptr;         // [16] + count
    int* d_forced_ = nullptr;        // replay continuation
    int forced_cap_ = 0;
    float* d_sampling_ = nullptr;    // temperature, top_p, top_k, seed lo/hi
    int* d_cand_ids_ = nullptr;      // sampling candidates per verify column [8][64]
    float* d_cand_p_ = nullptr;
    int* d_cand_n_ = nullptr;
    GdnScratch gsc_{};
    int* result_host_ = nullptr;
    crit::core::CudaBuffer result_pinned_;
    crit::core::CudaEvent round_done_;

    // Prompt buffers.
    int chunk_pad_ = 0;
    int* p_tok_ = nullptr;
    float* p_hidden_ = nullptr;
    std::uint8_t* p_xq_ = nullptr;
    float* p_xs_ = nullptr;
    __nv_bfloat16* p_xbf_ = nullptr;
    __nv_bfloat16* p_raw_ = nullptr;
    __nv_bfloat16* p_raw2_ = nullptr;
    __nv_bfloat16* p_h_ = nullptr;
    std::uint8_t* p_hq_ = nullptr;   // FP8 codes [chunk][17408] or NVFP4 packed
    float* p_hs_ = nullptr;
    float* p_ab_ = nullptr;          // prompt a/b gate projections [chunk][96]
    std::uint8_t* p_x4_ = nullptr;
    std::uint8_t* p_x4sf_ = nullptr;
    std::uint8_t* p_h4sf_ = nullptr;
    unsigned* p_amax_ = nullptr;     // [4]
    float* p_alpha_ = nullptr;       // [4]
    __nv_bfloat16* p_q_ = nullptr;
    float* p_attn_ = nullptr;
    std::uint8_t* p_aq_ = nullptr;
    float* p_as_ = nullptr;
    GdnPrefillScratch psc_{};
    WyScratch wy_{};
    __half* p_final_ = nullptr;      // final normed hidden rows for MTP prefill [chunk][5120]

    std::map<int, cudaGraphExec_t> graphs_;
    bool sampling_ = false;
};

}  // namespace kestrel
