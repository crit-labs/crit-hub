// fnl network: weights, state and the forward passes (no Core types here).
#pragma once
#include "common.cuh"
#include "experts.cuh"
#include "gemv.cuh"
#include "layers.cuh"
#include "prefill.cuh"
#include "spec.cuh"
#include "copy_pool.h"
#include "cpu_experts.h"
#include <crit/cuda.h>
#include <cstdio>
#include <algorithm>
#include <crit/safetensors.h>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fnl {

struct Options {
    std::filesystem::path model;
    int capacity = 262144;           // KV positions (prompt + output)
    std::size_t expert_slots = 0;    // 0: as many as fit
    std::size_t vram_margin = std::size_t(1) << 30;   // device memory left free at load
    int prefill_chunk = 16384;
    int load_threads = 12;
    bool mtp = true;                 // load the MTP block for speculative decoding
    int cpu_threads = 6;             // CPU workers for missed experts (0: the device reads every miss)
    float cpu_share = 0.f;           // share of a pass's misses computed by the CPU (0: off; slower on this PC)
};

// Checkpoint tensors through Core's safetensors reader (shards listed by model.safetensors.index.json).
class Tensors {
public:
    struct Entry {
        std::span<const std::byte> bytes;   // mapped view
        std::string dtype;
        std::vector<std::uint64_t> shape;
        std::uint32_t file = 0;
        std::uint64_t offset = 0;            // absolute file offset of the data
    };
    // One destination of a sequential gather.
    struct Copy {
        const Entry* src;
        void* dst;
    };
    explicit Tensors(const std::filesystem::path& model);
    const Entry& at(const std::string& name) const;
    bool has(const std::string& name) const { return map_.count(name) != 0; }
    // Reads every source with direct I/O in file order (files in parallel) and copies it to its destination.
    void gather(std::vector<Copy> copies, int threads) const;
private:
    std::vector<crit::core::SafetensorsFile> files_;
    std::vector<std::shared_ptr<const void>> owners_;
    std::map<std::string, Entry> map_;
};

struct Mx8 { const uint8_t* w = nullptr; const uint8_t* s = nullptr; int rows = 0, cols = 0; };
struct Hc { const float* norm1p = nullptr; const __nv_bfloat16* down = nullptr; const __nv_bfloat16* up = nullptr;
            const __nv_bfloat16* inject = nullptr; };
struct LayerWeights {
    bool attention = false;
    Hc ahc, mhc;
    // gated delta net
    Mx8 qkv, z, a, b, out;
    const float* conv_w = nullptr; const float* neg_a = nullptr; const float* dt_bias = nullptr;
    const float* gnorm = nullptr;
    // attention
    Mx8 q, k, v, o, idx;
    const float* qn1p = nullptr; const float* kn1p = nullptr; const float* iqn1p = nullptr; const float* ikn1p = nullptr;
    // moe
    const __nv_bfloat16* router = nullptr; const __nv_bfloat16* shared_gate = nullptr;
    Mx8 sh_gate, sh_up, sh_down;
};
// The MTP block (one attention layer with routed experts, BF16 dense weights, experts resident on the device).
struct MtpWeights {
    const float* emb1p = nullptr;    // 1 + pre_fc_norm_embedding [kHidden]
    const float* hid1p = nullptr;    // 1 + pre_fc_norm_hidden [kHyper]
    const __nv_bfloat16* fc_emb = nullptr; const __nv_bfloat16* fc_hid = nullptr;
    Hc ahc, mhc, mixer;
    const __nv_bfloat16 *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *idx = nullptr;
    const float* qn1p = nullptr; const float* kn1p = nullptr; const float* iqn1p = nullptr; const float* ikn1p = nullptr;
    const __nv_bfloat16* router = nullptr; const __nv_bfloat16* shared_gate = nullptr;
    const __nv_bfloat16 *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr;
    Mx8 head;                        // MXFP8 re-encoding of lm_head for drafting
};
struct PleWeights {
    const __nv_bfloat16* key = nullptr; const __nv_bfloat16* value = nullptr;
    const float* nk1p = nullptr; const float* nq1p = nullptr; const float* nc1p = nullptr; const float* conv_w = nullptr;
    float s2 = 0;
    std::vector<const uint8_t*> shard_w, shard_s;   // mapped host rows of the n-gram table
    std::vector<int64_t> sizes, offsets, mult;      // per head vocab sizes / row offsets, 3 multipliers
    int rows_per_shard = 0;
};

class Net {
public:
    Net(const Options& options, crit::core::CudaBackend& backend, crit::core::Resources& resources);
    ~Net();
    const Options& options() const { return options_; }
    int capacity() const { return options_.capacity; }
    std::size_t expert_slots() const { return tables_.nslots; }
    int own_slots() const { return own_slots_; }
    // Before a request touches KV: positions it may use (prompt + output). Positions beyond lend their memory
    // to the expert cache.
    void begin_request(int positions);

    // Clears all request state (KV positions are overwritten as they are used) and the pass timing.
    void reset();
    // Device milliseconds since reset() in target passes (head through acceptance and state commit) and in the
    // drafter's steps of speculative rounds; waits for the last pass.
    std::pair<double, double> pass_timing();
    // One token at position pos (tokens[0..pos] is the request's history, tokens[pos] the input).
    // Enqueues work; next token id lands in *device_token and logits in logits().
    void decode(const std::vector<int>& history, int pos);
    const float* logits() const { return logits_; }
    const int* next_token() const { return next_; }
    cudaStream_t stream() const;

    // Prompt tokens history[pos0 .. pos0 + count) in one pass (count <= prefill_chunk). The last position's
    // logits()/next_token() are always produced; with keep_all, head_rows() gives the logits of any position.
    // With drafter, the MTP block's cache also covers the chunk's positions whose next token is known.
    void prefill(const std::vector<int>& history, int pos0, int count, bool keep_all = false, bool drafter = false);

    // Speculative decoding (greedy). spec_begin() after the prompt's last chunk (position pos = last prompt
    // position): the first output token and the first drafts. spec_round(): one verified pass of the input token
    // history[pos] and drafts at positions pos..pos+3 (history holds the request's tokens up to pos), then the
    // drafter. Both end with the round's result copied to spec_result() (pinned; ready after spec_done()).
    bool has_mtp() const { return mtp_loaded_; }
    void spec_begin(int pos);
    // After spec_begin: gathers the first verified pass's n-gram rows while the drafter runs (history holds the
    // prompt; pos its last position).
    void spec_begin_prefetch(const std::vector<int>& history, int pos);
    void spec_round(const std::vector<int>& history, int pos, const int* drafts);
    // Scoring through verify passes: tokens history[pos .. pos + T) in one pass (T <= 4), all kept; logits() holds
    // T rows (kVocab apart).
    void verify_pass(const std::vector<int>& history, int pos, int T);
    const Accept& spec_result() const { return *spec_host_; }
    cudaEvent_t spec_done() const { return step_done_; }
    // Logits [n][kVocab] of chunk positions r0 .. r0 + n of the last prefill(keep_all = true) (prompt-path head).
    void head_rows(int r0, int n, float* out);
    int prefill_chunk() const { return options_.prefill_chunk; }

    // Debug: hidden streams after each layer (device, kHyper floats per token) when set.
    std::function<void(int layer, const float* H, int tokens)> on_layer;

    unsigned long long expert_lookups() const;
    unsigned long long expert_misses() const;
    unsigned long long prefill_fetched() const { return prefill_fetched_; }

private:
    void load_dense(const Tensors& t);
    void load_experts(const Tensors& t);
    void load_mtp(const Tensors& t);
    void allocate_state();
    void allocate_prefill();
    void allocate_slots();
    void hc_pre(const Hc& w, float* H, int tokens, float* inj);
    void decode_layer(int l, int T);
    void decode_head(int T);
    void decode_tail(int T);
    void set_steps(const std::vector<int>& history, int pos, int T, const int* tokens);
    void launch_pass(const std::vector<int>& history, int pos, int T, bool spec, bool have_rows = false);
    void spec_accept();
    void spec_drafts();
    void collect_pass_timing();
    // While the drafter runs, gathers the n-gram rows of the next round's pass as its tokens become known.
    void prefetch_next(const int* window, int base, int pos_last, const int* inputs, int ninputs);
    // Drafter: extend pass over R rows (ids, hidden rows of `hidden`), then its token into acc_->drafts[k].
    void mtp_pass(int R, const int* ids, const float* hidden, const int* rows, const Step* steps);
    void mtp_draft(int R, int k);
    void mtp_batch(const std::vector<int>& history, int pos0, int T);
    // n-gram rows of positions first..first+count; with parallel, copied by the gather pool (one caller at a time).
    void ple_rows(const int* tokens, int base, int first, int count, uint8_t* host_rows, bool parallel = false) const;
    // prompt path
    void hc_pre_batch(const Hc& w, int T, bool inject);
    void gdn_batch(const LayerWeights& L, int gi, int T);
    void attn_batch(const LayerWeights& L, int ai, int pos0, int T);
    void moe_batch(const LayerWeights& L, int layer, int T, bool prefetch_next);
    void ple_batch(const std::vector<int>& history, int pos0, int T);
    void directory_download();
    void directory_upload();
    void evict_slot(int s, bool keep_valid);
    int host_victim(int protect_epoch);
    void host_plan_layer(int layer, const std::vector<int>& want, std::vector<int>& fetch);

    Options options_;
    std::unique_ptr<Tensors> tensors_;   // keeps the mapped n-gram table alive
    std::unique_ptr<CopyPool> gather_pool_;
    crit::core::CudaBackend& backend_;
    crit::core::Resources& resources_;
    std::vector<crit::core::CudaBuffer> buffers_;
    std::vector<LayerWeights> layers_;
    PleWeights ple_;
    Hc mixer_;
    const __nv_bfloat16* lm_head_ = nullptr;
    const __nv_bfloat16* embed_ = nullptr;   // mapped pinned host
    const float* inv_freq_ = nullptr;
    ExpertTables tables_;
    std::vector<uint8_t*> host_layers_;
    // host views of the expert tables for the CPU
    std::vector<uint32_t> exc_off_h_, exc_cnt_h_;
    std::vector<float> s2_h_;
    const uint32_t* exc_pool_h_ = nullptr;
    CpuReply* cpu_reply_ = nullptr;
    static constexpr int kInsertJobs = 64;
    cudaStream_t insert_stream_ = nullptr;
    cudaEvent_t insert_done_ = nullptr;
    ScaleJob* insert_jobs_ = nullptr;   // pinned ring of scale jobs for background insertions
    int insert_next_ = 0;
    std::unique_ptr<CpuMissServer> cpu_server_;
    void sync_insertions();
public:
    unsigned long long cpu_jobs() const { return cpu_server_ ? cpu_server_->jobs() : 0; }
    unsigned long long cpu_experts() const { return cpu_server_ ? cpu_server_->experts() : 0; }
    void cpu_report() const {
        if(!cpu_server_) return;
        const double j = double(cpu_server_->jobs());
        std::fprintf(stderr, "cpu: %.0f jobs, %.2f experts/job, %.1f us/job compute, %.1f us/job insert issue\n", j,
                     cpu_server_->experts() / std::max(j, 1.0), cpu_server_->busy_ns() / 1e3 / std::max(j, 1.0),
                     cpu_server_->insert_ns() / 1e3 / std::max(j, 1.0));
    }
private:

    // state
    float* gdn_state_ = nullptr;     // [36][kGdnRing][48][128][128], current state ring[*gdn_cur_]
    int* gdn_cur_ = nullptr;
    float* conv_state_ = nullptr;    // [36][kQkvDim][3]
    float* conv_in_ = nullptr;       // [36][kMaxTokens][kQkvDim] conv inputs of the last decode pass
    uint8_t* kcache_ = nullptr;         // [12][2][cap][256] E4M3
    uint8_t* vcache_ = nullptr;
    float* kscale_ = nullptr;           // [12][2][cap] one scale per key / value vector
    float* vscale_ = nullptr;
    float* rawkeys_ = nullptr;       // [12][kRawRing][128]
    float* blockkeys_ = nullptr;     // [12][cap/4][128]
    float* ple_state_ = nullptr;     // [kHyper][9]

    // decode workspace (up to kMaxTokens tokens)
    float *H_, *xn_, *lora_, *mixw_, *inj_, *x_, *proj_, *z_, *ab_, *q_, *k_, *v_, *g_, *beta_, *o_, *gn_, *out_;
    float *aq_, *agate_, *aiq_, *scores_, *partial_, *rlogits_, *act_, *shg_, *shu_, *sha_, *sho_, *sg_;
    float *logits_, *amax_v_, *ple_e_, *ple_k_, *ple_v_, *ple_gv_, *ple_gvn_;
    int *sel_, *next_, *amax_i_, *ids_, *one_, *kept_;   // kept_[t] = t
    uint8_t* ple_rows_dev_ = nullptr;
    uint8_t* ple_rows_host_ = nullptr;  // pinned [kMaxTokens][16][90]
    int* ids_host_ = nullptr;           // pinned [kMaxTokens]
    Route* routes_ = nullptr;           // [kMaxTokens]
    float* down_part_ = nullptr;        // [kMaxUnion][kMaxPassTokens][kHidden] per-entry down projections
    ExpertPlan* plan_ = nullptr;
    unsigned long long* stats_ = nullptr;
    Step* step_ = nullptr;              // device step facts [kMaxTokens]
    Step* step_host_ = nullptr;         // pinned [kMaxTokens]
    int graph_blocks_ = 0;              // index-score grid of the pass being enqueued (0: dense)
    struct StepGraphs { cudaGraphExec_t head = nullptr, tail = nullptr, draft = nullptr; };
    std::map<std::pair<int, int>, StepGraphs> graphs_;  // by (pass kind, index-score bucket)
    cudaEvent_t step_done_ = nullptr;   // end of the last enqueued pass
    cudaEvent_t pass_marks_[3] = {};    // start of the last pass, end of its target part, end of its drafts
    bool pass_pending_ = false, pass_drafted_ = false;
    double target_ms_ = 0, draft_ms_ = 0;

    // MTP block
    bool mtp_loaded_ = false;
    MtpWeights mtp_;
    uint8_t* mtp_experts_ = nullptr;    // [kExperts][kSlotBytes]
    uint8_t *mkcache_ = nullptr, *mvcache_ = nullptr;  // [2][cap][256] E4M3
    float *mkscale_ = nullptr, *mvscale_ = nullptr;    // [2][cap]
    float* mrawkeys_ = nullptr;         // [kRawRing][128]
    float* mblockkeys_ = nullptr;       // [cap/4][128]
    float *me_ = nullptr, *men_ = nullptr, *fe_ = nullptr, *hn_ = nullptr, *fh_ = nullptr, *mH_ = nullptr;
    float* dlogits_ = nullptr;          // [kVocab]
    Accept* acc_ = nullptr;
    Accept* spec_host_ = nullptr;       // pinned
    DraftInputs dinp_{};
    SpecProgress* progress_ = nullptr; // pinned, written by the device during a round
    uint8_t* next_rows_ = nullptr;     // pinned [kMaxTokens][16][90] rows gathered ahead
    struct RowsAhead { bool valid = false; int pos = 0; int tokens[kMaxTokens] = {0}; } ahead_;
    int* rows_last_ = nullptr;          // {3, 0}: last row of a 4-row pass, row of a 1-row pass

    // prompt workspace (chunk T, MoE sub-chunk S)
    static constexpr int kSub = 2048;
    static constexpr int kQueryGroup = 128;
public:
    static constexpr int kScoreRows = 128;
    float* score_logits() const { return pscore_logits_; }
private:
    __half* pkeys3_ = nullptr;                // [cap/4][384] split keys of the current layer (index GEMM)
    float *pH_ = nullptr, *px_ = nullptr, *pinj_ = nullptr, *pout_ = nullptr;
    __half* px_bf_ = nullptr;
    Route* proutes_ = nullptr;
    int* ppos_of_ = nullptr;
    uint8_t* pscratch_ = nullptr;             // phase scratch
    std::size_t pscratch_bytes_ = 0;
    __half* pact_ = nullptr;                  // [kSub * kTopK][kInter]
    float* proute_out_ = nullptr;             // [kSub * kTopK][kHidden]
    int* proute_token_ = nullptr;             // [kSub * kTopK]
    int* pcounts_ = nullptr;                  // [nsub][512]
    int* poffsets_ = nullptr;                 // [512]
    int* pcursor_ = nullptr;                  // [512]
    int4* ptiles_ = nullptr;                  // [kSub*kTopK/128 + 512]
    int* pntiles_ = nullptr;
    uint8_t** pgroup_base_ = nullptr;         // [512]
    ScaleJob* pjobs_ = nullptr;               // [1024]
    float* pscores_ = nullptr;                // [kQueryGroup][cap/4]
    float* pscore_logits_ = nullptr;          // [kScoreRows][kVocab] scoring head rows
    int* psel_ = nullptr;                     // [kQueryGroup][512]
    uint8_t* prows_dev_ = nullptr;            // [T][16][90]
    // pinned host staging
    uint8_t* hstage_ = nullptr;
    int* hcounts_ = nullptr;
    int* hoffsets_ = nullptr;
    int4* htiles_ = nullptr;
    int* hntiles_ = nullptr;
    uint8_t** hgroup_base_ = nullptr;
    ScaleJob* hjobs_ = nullptr;
    uint8_t* hrows_ = nullptr;
    // host mirror of the expert directory during a prompt pass
    std::vector<int> mslot_of_, mowner_, mstamp_;
    std::vector<uint8_t> mref_, mvalid_;
    std::vector<uint8_t*> mslot_addr_;
    static constexpr int kOwnSlot = 0, kKvSlot = 1, kWorkspaceSlot = 2;
    std::vector<int> slot_kind_, slot_first_pos_;
    int own_slots_ = 0;
    std::vector<std::pair<uint8_t*, std::size_t>> workspace_lend_;
    int mhand_ = 0, mepoch_ = 0;
    std::vector<int> prefetched_;             // experts of the next layer already being copied
    crit::core::CudaEvent copies_done_;
    unsigned long long prefill_fetched_ = 0;
    // Prompt-only experts of short chunks: copied into the workspace tail, not cached.
    static constexpr int kTransientChunk = 512;
    static constexpr int kTransientTokens = 4;
public:
    unsigned long long prompt_transient_ = 0;
private:
    // CPU share of short prompt chunks' missing experts
    static constexpr int kCpuPromptTokens = 511;
    static constexpr int kCpuPromptRows = kCpuPromptTokens * kTopK;
    std::unique_ptr<CpuExperts> prompt_cpu_;
    float prompt_cpu_share_ = 0.43f;
    float* hx_ = nullptr;            // pinned [kCpuPromptTokens][kHidden] MoE inputs of the chunk
    Route* hroutes_ = nullptr;       // pinned [kCpuPromptTokens]
    float* hcpu_rows_ = nullptr;     // pinned [kCpuPromptRows][kHidden] CPU results
    int* hcpu_row_ = nullptr;        // pinned [kCpuPromptTokens][kTopK]
    int* pcpu_row_ = nullptr;        // device copy
public:
    unsigned long long prompt_cpu_routes_ = 0;
private:
};

}  // namespace fnl
