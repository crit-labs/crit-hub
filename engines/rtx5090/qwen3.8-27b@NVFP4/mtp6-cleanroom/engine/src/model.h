#pragma once
// Qwen3.8-27B NVFP4 model state on crit Core's CUDA backend: resident weights in
// their stored encodings, private request state and the device speculative round.
#include "kernels.cuh"
#include "safetensors.h"
#include <crit/cuda.h>
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>

namespace crit::techniques::device_round { class DeviceSpeculativeRound; }

namespace q38 {
struct Config {
    std::string model_dir;
    std::string main_file, mtp_file; // default: model_dir/model.safetensors, model_dir/model_mtp.safetensors
    int context = 4096;
    int prefill_chunk = 512;
    int draft_tokens = 3;        // K; 0 = ordinary decoding through the same round
    int draft_rows = 0;          // proposal head rows (0 = full vocabulary)
    bool graphs = true;
    int attn_split = 256;        // keys per attention partition in decode/verify
    bool mtp_fp8 = false;        // draft with a derived per-row FP8 copy of the BF16 MTP matrices
    static Config parse(const std::string& json_text, const std::string& model_dir);
};

constexpr int kVocab = 248320, kValidVocab = 248077, kHidden = 5120, kInter = 17408, kLayers = 64;
constexpr int kKmax = 7;                 // maximum draft width
constexpr int kTv = kKmax + 1;           // maximum verify columns
constexpr int kResSplit = 4;             // split-K partials for residual-producing decode projections

struct Upload { std::string name; std::size_t file_offset, bytes; void* device; const SafetensorsFile* file; };

struct LayerW {
    bool full = false;
    QW qkv, z, ab, out;          // GDN: qkv/z FP8, ab BF16 [a;b], out FP8. Attention: qkv FP8 [q|k|v], out = o_proj
    QW gate, up, down;
    const bf16 *in_norm = nullptr, *post_norm = nullptr;
    const bf16 *q_norm = nullptr, *k_norm = nullptr;
    GdnLayer gdn{};
    bf16 *kc = nullptr, *vc = nullptr;
};
struct MtpW {
    QW fc, qkv, o, gate, up, down;
    const bf16 *in_norm, *post_norm, *q_norm, *k_norm, *norm, *pre_e, *pre_h;
    bf16 *kc = nullptr, *vc = nullptr;
};

class Model {
public:
    Model(std::shared_ptr<crit::core::CudaBackend>, const Config&);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Config& config() const { return cfg_; }
    cudaStream_t stream() const { return stream_; }
    // Request lifetime
    void reset_request();                                   // zero private state
    void prefill(const int* host_tokens, int n, int pos0);  // materialize n tokens at pos0 (n <= prefill_chunk)
    void finish_prefill(int prompt_len, int last_token);     // seed the first round
    crit::techniques::device_round::DeviceSpeculativeRound& round() { return *round_; }
    // Log probabilities of tokens[first..n) (first >= 1).
    std::vector<float> score(const std::vector<int>& tokens, int first);
    std::vector<float> score_decode(const std::vector<int>& tokens, int first);
    // Weight audit: device read-back compared with the original file words.
    std::size_t audit(std::string& report);
    std::size_t weight_bytes() const { return weight_bytes_; }
    std::size_t state_bytes() const { return state_bytes_; }
    double prefill_ms_total = 0;
    // Eager per-operation timing of one target forward (and optional MTP step) at T columns.
    std::string profile(int T, int reps);
    std::string micro(int T);
private:
    std::shared_ptr<crit::core::CudaBackend> backend_;
    Config cfg_;
    cudaStream_t stream_;
    cublasHandle_t cublas_ = nullptr;
    std::unique_ptr<SafetensorsFile> main_, mtpfile_;
    std::vector<crit::core::CudaBuffer> buffers_;
    std::vector<Upload> uploads_;
    std::size_t weight_bytes_ = 0, state_bytes_ = 0;
    std::vector<LayerW> layers_;
    MtpW mtp_{}, mtp_draft_{};   // draft_ = mtp_ or its derived FP8 copy
    const bf16* embed_ = nullptr;
    const bf16* final_norm_ = nullptr;
    QW lm_head_{};
    // Activations
    float *partial_res_ = nullptr;
    float *resid_, *big_, *conv_out_, *core_, *qbuf_, *abuf_, *partial_, *logits_, *tmp_, *score_out_;
    bf16 *xn_, *act_, *xfinal_, *cat_, *dq_, *vhid_, *mtp_hin_, *mtp_h_, *pend_h_;
    int *tok_, *pos_, *mtok_, *mpos_, *pending_, *prev_n_, *prev_tok_, *targets_;
    int* host_ints_ = nullptr;
    int max_splits_ = 16, prefill_splits_ = 4;
    bool have_pending_ = false;
    std::unique_ptr<crit::techniques::device_round::DeviceSpeculativeRound> round_;

    void* alloc(std::size_t bytes, bool state = false);
    void upload(const SafetensorsFile& f, const std::string& name, void* dst, std::size_t offset = 0);
    QW load_fp8(const std::string& prefix);
    QW load_mlp(const std::string& prefix);
    QW load_bf16(const SafetensorsFile& f, const std::string& name);
    const bf16* load_vec(const SafetensorsFile& f, const std::string& name);
    void proj(const QW& w, const bf16* x, int ldx, int T, float* y, int ldy, bool accumulate);
    void mlp_up(const QW& gate, const QW& up, int T);
    void proj2(const QW& a, float* ya, const QW& b, float* yb, int ldy, const bf16* x, int ldx, int T);
    void target_forward(int T, const int* tokens, const int* base, int mode, int splits, int split);
    void mtp_forward(const MtpW& m, int T, const int* tokens, const bf16* hidden, int ldh, const int* base, int splits, int split, bf16* out);
    void draft_step(const int* control, int* drafts, int K);
    void verify_step(const int* control, const int* drafts, int* predictions, int K);
    void commit_step(const int* control, const int* result, int K);
    void set_device_int(int* dst, int value);
    bool profiling_ = false;
    std::vector<std::pair<const char*, cudaEvent_t>> marks_;
    void mark(const char* name);
};
}
