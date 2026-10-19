// nmh network: weights, state and the forward passes (no Core session types here).
#pragma once
#include "common.cuh"
#include "experts.cuh"
#include "fp8_kv_attention.h"
#include "gemm.cuh"
#include "gemv.cuh"
#include "layers.cuh"
#include "mamba.cuh"
#include "prefill.cuh"
#include "spec.cuh"
#include "cpu_experts.h"
#include "lt.h"
#include "fp4_grouped.h"
#include "fp4_prompt.cuh"
#include <crit/cuda.h>
#include <crit/safetensors.h>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nmh {

constexpr int kMaxTokens = 4;            // tokens of one decode pass
constexpr float kActScale = 1.f / 64.f;  // FP16 prompt-path relu2 activations are stored times this
constexpr int kMaxForced = 65536;        // longest replayed continuation

struct Options {
    std::filesystem::path model;
    int capacity = 1048576;          // KV positions (prompt + output)
    int prefill_chunk = 32768;
    int moe_sub = 2048;              // prompt MoE sub-chunk (routes grouped by expert per sub-chunk)
    std::size_t vram_margin = std::size_t(3) << 29;   // device memory left free when the expert cache is sized
    std::size_t expert_slots = 0;    // 0: as many as fit
    bool lend_kv = true;             // KV positions a request can't reach hold expert slots
    bool lend_workspace = true;      // the prompt workspace holds expert slots between prompts
    int load_threads = 12;
    int attn_blocks = 64;            // decode attention blocks per KV head
    bool mtp = true;                 // load the MTP block for speculative decoding
    float cpu_share = 0.f;           // share of a decode pass's misses computed by the CPU (0: none)
    int cpu_min = 2;                 // misses a pass needs before the CPU takes part
    int cpu_threads = 6;
    int drafts = 3;                  // drafts per speculative round (1 .. kDrafts)
    bool prefill_fp8 = false;        // prompt FP8-weight GEMMs on FP8 tensor cores (activations quantized per row)
    int cpu_prompt_max = 384;        // prompt chunks up to this many tokens give a share of their experts to the CPU
    float cpu_prompt_share = 0.35f;  // share of the experts (fixed by expert id, not by cache state)
    bool prompt_attn_fp8 = true;     // prompt attention on E4M3 tensor cores (crit fp8_kv::prefill_fp8)
    bool prompt_fp4 = true;          // prompt routed experts as NVFP4 W4A4 grouped GEMMs (checkpoint input scales)
    float dt_min = 0.f;              // Mamba dt lower limit (config.json time_step_limit; transformers clamps at time_step_min)
    // Keep the pinned expert records and embeddings in the process for the next engine opened on the same model
    // (Eval opens engines one after another; docs/eval/ADAPTER.md). Off for serving, where candidate switches
    // load other libraries that could not reuse them.
    bool keep_host_weights = false;
};

// Checkpoint tensors through Core's safetensors reader (shards listed by model.safetensors.index.json).
class Tensors {
public:
    struct Entry {
        std::span<const std::byte> bytes;
        std::string dtype;
        std::vector<std::uint64_t> shape;
        std::uint32_t file = 0;
        std::uint64_t offset = 0;
    };
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

struct HostBlock;   // pinned host memory filled before pinning (net.cu)

struct Fp8W { const uint8_t* w = nullptr; float scale = 0.f; };
struct Fp4W {
    const uint8_t* w = nullptr;
    const uint8_t* s = nullptr;
    float scale = 0.f;            // weight_scale_2
    float in_scale = 0.f;         // input_scale (static activation global scale)
    const uint8_t* sf = nullptr;  // prompt_fp4: scales in the Sm1xx layout
    const float* alpha = nullptr; // prompt_fp4: device in_scale * scale
};

struct MoeWeights {
    int group = 0;                           // expert-table group
    const __nv_bfloat16* router = nullptr;   // [512][4096]
    const float* bias = nullptr;             // [512] e_score_correction_bias
    const __nv_bfloat16* fc1 = nullptr;      // [1024][4096]
    const __nv_bfloat16* fc2 = nullptr;      // [4096][1024]
    Fp4W sh_up, sh_down;
};

struct LayerWeights {
    Kind kind = Kind::mamba;
    int index = 0;                    // index among layers of its kind
    const float* norm = nullptr;      // [kHidden]
    // mamba
    Fp8W in_proj, out_proj;
    MambaWeights m;
    // attention
    Fp8W q, k, v, o;
    // moe
    MoeWeights moe;
};

// The MTP block: fusion of the next token's embedding with a hidden state, an attention layer (BF16) and a MoE
// layer (experts in expert-table group kMtpGroup), then the shared head.
struct MtpWeights {
    const float *enorm = nullptr, *hnorm = nullptr, *norm0 = nullptr, *norm1 = nullptr, *final_norm = nullptr;
    const __nv_bfloat16* eh_proj = nullptr;   // [4096][8192]: rows over (enorm(embedding), hnorm(hidden))
    const __nv_bfloat16 *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    MoeWeights moe;
};

class Net {
public:
    Net(const Options& options, crit::core::CudaBackend& backend);
    ~Net();
    const Options& options() const { return options_; }
    int capacity() const { return options_.capacity; }
    int prefill_chunk() const { return options_.prefill_chunk; }
    cudaStream_t stream() const;
    std::size_t expert_slots() const { return std::size_t(tables_.nslots); }
    int own_slots() const { return own_slots_; }
    bool has_mtp() const { return mtp_loaded_; }
    int drafts() const { return options_.drafts; }

    // Clears request state (KV positions are overwritten as they are used).
    void reset();
    // Before a request touches KV: positions it may use. Lent slots in positions below it are evicted.
    void begin_request(int positions);
    // Prompt tokens history[pos0 .. pos0 + count) (count <= prefill_chunk). The last position's logits() and
    // next_token() are produced; with keep_all, head_rows() gives the logits of any chunk position. With drafter,
    // the MTP block's cache also covers the chunk's positions whose next token is known (all but the prompt's last).
    void prefill(const std::vector<int>& history, int pos0, int count, bool keep_all = false, bool drafter = false);
    // Logits [n][kVocab] of chunk rows r0 .. r0 + n of the last prefill(keep_all = true) into score_logits().
    void head_rows(int r0, int n);
    static constexpr int kScoreRows = 128;
    float* score_logits() const { return pscore_; }
    // One decode pass of T tokens (history[pos .. pos + T) at positions pos ..): logits rows and argmax per token.
    void decode(const std::vector<int>& history, int pos, int T = 1);
    const float* logits() const { return logits_; }
    const int* next_token() const { return next_; }

    // Speculative decoding (greedy). set_forced(): the replayed continuation of this request (empty: none).
    // spec_begin(pos): after the prompt's last chunk (pos = last prompt position): the first output token and the
    // first drafts. spec_round(pos, input, drafts): one verified pass of the input token at pos and the drafts at
    // pos + 1 .., acceptance, state commit and the next drafts. Both end with spec_result() (pinned) ready at
    // spec_done().
    void set_forced(const std::vector<int>& forced);
    void spec_begin(int pos);
    void spec_round(int pos, int input, const int* drafts);
    const Accept& spec_result() const { return *spec_host_; }
    cudaEvent_t spec_done() const { return spec_done_; }
    // Device milliseconds of verify passes and drafter steps since reset().
    std::pair<double, double> pass_timing();

    unsigned long long expert_lookups() const;
    unsigned long long expert_misses() const;
    unsigned long long expert_cpu() const;
    unsigned long long prefill_fetched() const { return prefill_fetched_; }
    std::size_t dense_bytes_per_pass() const { return dense_bytes_; }
    std::size_t draft_bytes_per_step() const { return draft_bytes_; }

private:
    void load_dense(const Tensors& t);
    void load_mtp(const Tensors& t);
    void load_experts(const Tensors& t);
    void allocate_state();
    void allocate_prefill();
    void allocate_slots();
    void decode_layers(int T, bool spec);
    void moe_decode(const MoeWeights& w, const float* x, float* h, int T);
    void moe_prefill(const MoeWeights& w, const __half* x, float* h, int T);
    void mtp_prefill(const std::vector<int>& history, int pos0, int rows);
    // Drafter rows: fused inputs, q / k / v and KV writes for R rows (steps: position, token after it) from hidden
    // rows (normed target or drafter outputs, FP32 [R][kHidden]).
    void mtp_rows(int R, const Step* steps, const float* hidden);
    // The drafter's output for row *sel of the last mtp_rows: attention, MoE, final norm (into mtp_out_) and the
    // drafting head's argmax into draft_tok_.
    void mtp_finish(const int* sel, const Step* step);
    void draft_chain(int first_k);
    void directory_download();
    void directory_upload();
    int host_victim(int protect_epoch);
    void lend_workspace(bool lend);

    Options options_;
    crit::core::CudaBackend& backend_;
    std::vector<crit::core::CudaBuffer> buffers_;
    std::shared_ptr<HostBlock> experts_block_, embed_block_;   // pinned host records
    std::unique_ptr<CpuMissServer> cpu_server_;
    crit::core::CudaBuffer mailbox_;
    std::vector<LayerWeights> layers_;
    const float* norm_f_ = nullptr;
    Fp8W lm_head_;
    const __nv_bfloat16* embed_ = nullptr;   // mapped pinned host
    std::size_t dense_bytes_ = 0, draft_bytes_ = 0;
    bool mtp_loaded_ = false;
    MtpWeights mtp_;

    // experts
    ExpertTables tables_;
    uint8_t* host_experts_ = nullptr;
    std::vector<float> s2_host_;          // [kGroups * kExperts * 2]
    float* palpha_ = nullptr;             // device [kGroups * kExperts * 2]: up s2, down s2 / kActScale
    float* palpha4_ = nullptr;            // device [kGroups * kExperts * 2]: W4A4 alphas, input_scale * s2 (up, down)
    std::vector<float> in_scale_;         // [kGroups * 2] routed experts' input_scale (up, down), one per layer
    int own_slots_ = 0;
    std::vector<int> slot_first_pos_;     // KV-lent slots: first position they occupy; -1 own memory; -2 workspace
    std::vector<std::pair<uint8_t*, std::size_t>> workspace_lend_;
    bool workspace_lent_ = false;

    // state
    float* conv_state_ = nullptr;         // [kMambaLayers][kConvDim][3]
    float* conv_rec_ = nullptr;           // [kMambaLayers][kMaxTokensRec][kConvDim] conv inputs of the last pass
    float* ssm_ = nullptr;                // [kMambaLayers][ring][kSsmFloats]
    int ring_ = 1;                        // SSM states per layer (kStateRing with speculation)
    int* ssm_cur_ = nullptr;              // [1]
    std::vector<KvCache> kv_;             // per attention layer, then the drafter's
    std::vector<crit::fp8_kv::Cache> kvc_;

    // decode workspace (up to kMaxTokens tokens)
    Step* step_ = nullptr;                // device [kMaxTokens]
    Step* step_host_ = nullptr;           // pinned [kMaxTokens]
    float *H_ = nullptr, *X_ = nullptr, *P_ = nullptr, *XC_ = nullptr, *Y_ = nullptr, *G_ = nullptr;
    float *AO_ = nullptr, *RL_ = nullptr, *LAT_ = nullptr, *SH_ = nullptr, *ACT_ = nullptr, *PART_ = nullptr;
    float *LO_ = nullptr, *logits_ = nullptr, *attn_partial_ = nullptr;
    __nv_bfloat16* Q16_ = nullptr;
    int* next_ = nullptr;
    Route* routes_ = nullptr;
    ExpertPlan* plan_ = nullptr;
    unsigned long long* stats_ = nullptr;
    std::map<int, cudaGraphExec_t> graphs_;   // plain decode pass per token count
    // speculative rounds
    Accept* acc_ = nullptr;               // device
    Accept* spec_host_ = nullptr;         // pinned
    DraftCtl* dctl_ = nullptr;            // device
    int* forced_ = nullptr;               // device [kMaxForced]
    int forced_len_ = 0;
    float *mE_ = nullptr, *mF_ = nullptr, *mX_ = nullptr, *mP_ = nullptr, *mR_ = nullptr, *mO_ = nullptr;
    float* mtp_out_ = nullptr;            // [kHidden] drafter output of the last step (normed)
    float* mlogits_ = nullptr;            // [kVocab]
    __nv_bfloat16 *mQ16_ = nullptr, *mQsel_ = nullptr;
    int* draft_tok_ = nullptr;            // [1]
    cudaGraphExec_t round_graph_ = nullptr, begin_graph_ = nullptr;
    cudaEvent_t spec_done_ = nullptr;
    cudaEvent_t marks_[3] = {};           // round start, end of the verify part, end of the drafts
    bool marks_pending_ = false;
    double verify_ms_ = 0, draft_ms_ = 0;

    // prompt workspace (chunk C, MoE sub-chunk S)
    int* pids_ = nullptr;                 // device [C]
    int* pids_host_ = nullptr;            // pinned [C]
    float* pH_ = nullptr;                 // [C][kHidden]
    __half* pX_ = nullptr;                // [C][kMInner] f16 (normed inputs, gated-norm outputs)
    float* pP_ = nullptr;                 // [C][kInProj]
    float* pXC_ = nullptr;                // [C][kConvDim]
    float* pY_ = nullptr;                 // [C][kMInner]  (also attention output)
    __nv_bfloat16* pQKV_ = nullptr;       // [C][kQkv]
    __half* pLAT_ = nullptr;              // [C][kLatent]
    __half* pSH_ = nullptr;               // [C][kShared]
    float* pRL_ = nullptr;                // [C][kExperts]
    Route* proutes_ = nullptr;            // [C]
    __half* pACT_ = nullptr;              // [S*kTopK][kInter]
    float* pROUT_ = nullptr;              // [S*kTopK][kLatent]
    __half* pLO_ = nullptr;               // [S][kLatent]
    int* ptoken_ = nullptr;               // [S*kTopK]
    int* ppos_ = nullptr;                 // [S*kTopK]
    int* pcounts_ = nullptr;              // [nsub][kExperts]
    int* poffsets_ = nullptr;             // [nsub][kExperts]
    int* pcursor_ = nullptr;              // [kExperts]
    int4* ptiles_ = nullptr;              // [nsub][max tiles]
    int* pntiles_ = nullptr;              // [nsub]
    const uint8_t** pgroup_ = nullptr;    // [kExperts] slot base per expert of the layer
    float* pscore_ = nullptr;             // [kScoreRows][kVocab]
    uint8_t* pq_ = nullptr;               // [C][kMInner] E4M3 activation codes (prefill_fp8)
    float* prs_ = nullptr;                // [C] their row scales
    std::unique_ptr<Lt> lt_;
    // NVFP4 W4A4 routed experts (prompt_fp4): gathered quantized rows of a sub-chunk and their scale rows (each
    // expert's rows padded to 128), the layer's resident weight scales in the Sm1xx layout, per-expert scale rows.
    uint8_t *pXq_ = nullptr, *pXsf_ = nullptr, *pAq_ = nullptr, *pAsf_ = nullptr, *pWsf_ = nullptr;
    uint8_t *pShq_ = nullptr, *pShsf_ = nullptr, *pShaq_ = nullptr, *pShasf_ = nullptr;   // shared expert rows
    std::size_t rout_bytes_ = 0;
    // CPU share of short prompts: routes and latent rows on the host, fixed rows of the CPU experts' routes
    Route* hroutes_ = nullptr;            // pinned [cpu_prompt_max]
    uint16_t* hlat_ = nullptr;            // pinned [cpu_prompt_max][kLatent] FP16 bits
    int* hfixed_ = nullptr;               // pinned [cpu_prompt_max][kTopK]
    int* pfixed_ = nullptr;               // device copy
    float* hcpu_out_ = nullptr;           // pinned [cpu_prompt_max * kTopK][kLatent]
    std::vector<int> cpu_row_token_;
    std::vector<CpuPromptExpert> cpu_list_;
    void shared_fp4(const MoeWeights& w, const __half* x, float* h, int T, cudaStream_t s);
    int* psfrow_ = nullptr;               // device [nsub][kExperts]
    int* hsfrow_ = nullptr;               // pinned [nsub][kExperts]
    std::unique_ptr<Fp4Grouped> fp4g_;
    std::vector<Fp4Group> fp4_groups_;
    void moe_experts_fp4(int group, int k, int t0, cudaStream_t s);
    // out (=/+=) x . W^T for a prompt chunk: FP8 tensor cores from FP32 rows x32 (prefill_fp8), else FP16 tensor
    // cores from the FP16 rows x16.
    void dense_fp8(const Fp8W& w, int N, int K, const float* x32, int ldx, int T, float* out, int ldo, bool add,
                   float* tmp);
    // pinned staging
    int* hcounts_ = nullptr;
    int* hoffsets_ = nullptr;
    int4* htiles_ = nullptr;
    int* hntiles_ = nullptr;
    const uint8_t** hgroup_ = nullptr;
    int max_tiles_ = 0;
    int last_count_ = 0;
    // host mirror of the expert directory during a prompt pass
    std::vector<int> mslot_of_, mowner_, mstamp_;
    std::vector<uint8_t> mref_, mvalid_;
    std::vector<uint8_t*> mslot_addr_;
    int mhand_ = 0, mepoch_ = 0;
    // Prompt chunks of at least kPrefetchTokens route to nearly every expert of a layer: the next MoE layer's missing
    // experts are copied on the transfer stream while the layers before it compute.
    static constexpr int kPrefetchTokens = 256;
    void prefetch_group(int group);
    int prefetch_group_ = -1, prefetch_epoch_ = 0;
    cudaEvent_t prefetch_ready_ = nullptr, moe_done_ = nullptr;
    unsigned long long prefill_fetched_ = 0;
};

}  // namespace nmh
