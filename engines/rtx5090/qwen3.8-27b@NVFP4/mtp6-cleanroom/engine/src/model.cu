// Model loading, forward passes and device speculative-round steps.
#include "model.h"
#include "forge/device_round.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace q38 {
namespace dr = crit::techniques::device_round;
namespace {
void check(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void checkb(cublasStatus_t e, const char* what) {
    if(e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(e)));
}
constexpr float kEps = 1e-6f;
constexpr int kGdnLd = 10240 + 6144;

__global__ void mtp_positions_kernel(const int* control, const int* prev_n, int* mpos, int K) {
    const int j = threadIdx.x;
    if(j > K) return;
    const int frontier = control[dr::control::frontier];
    mpos[j] = j == 0 ? frontier - *prev_n : frontier + j - 1;
}
__global__ void gather_row_kernel(const bf16* src, const int* prev_n, bf16* dst) {
    int r = *prev_n - 1;
    if(r < 0) r = 0;
    for(int d = threadIdx.x; d < kHidden; d += blockDim.x) dst[d] = src[std::size_t(r) * kHidden + d];
}
__global__ void commit_kernel(const int* control, const int* result, int K, int* pending, int* prev_n, int* prev_tok) {
    const int i = threadIdx.x;
    const int n = control[dr::control::n_emitted];
    if(i == 0) { *pending = n; *prev_n = n; }
    if(i <= K) prev_tok[i] = result[1 + i];
}
__global__ void seed_kernel(int* pending, int* prev_n, int* prev_tok, int n, int last) {
    *pending = 0; *prev_n = n; prev_tok[0] = last;
}
}

Config Config::parse(const std::string& text, const std::string& dir) {
    Config c;
    c.model_dir = dir;
    c.main_file = dir + "/model.safetensors";
    c.mtp_file = dir + "/model_mtp.safetensors";
    if(text.empty()) return c;
    const auto j = Json::parse(text);
    if(j.has("context")) c.context = int(j["context"].integer());
    if(j.has("prefill_chunk")) c.prefill_chunk = int(j["prefill_chunk"].integer());
    if(j.has("draft_tokens")) c.draft_tokens = int(j["draft_tokens"].integer());
    if(j.has("draft_rows")) c.draft_rows = int(j["draft_rows"].integer());
    if(j.has("graphs")) c.graphs = j["graphs"].b;
    if(j.has("attn_split")) c.attn_split = int(j["attn_split"].integer());
    if(j.has("mtp_fp8")) c.mtp_fp8 = j["mtp_fp8"].b;
    if(c.draft_tokens < 0 || c.draft_tokens > kKmax) throw std::invalid_argument("draft_tokens outside 0..7");
    if(c.draft_rows % 16 || c.draft_rows < 0 || c.draft_rows > kVocab) throw std::invalid_argument("draft_rows must be a multiple of 16");
    if(c.prefill_chunk < kTv + 1 || c.prefill_chunk > 1024) throw std::invalid_argument("prefill_chunk outside 9..1024");
    return c;
}

void* Model::alloc(std::size_t bytes, bool state) {
    bytes = (bytes + 255) & ~std::size_t(255);
    buffers_.push_back(backend_->allocate(crit::core::Memory::device, bytes));
    (state ? state_bytes_ : weight_bytes_) += bytes;
    return buffers_.back().data();
}

// Copy exact file words to device memory in bounded synchronous slices.
void Model::upload(const SafetensorsFile& f, const std::string& name, void* dst, std::size_t offset) {
    const auto src = f.bytes(name);
    constexpr std::size_t slice = std::size_t(64) << 20;
    for(std::size_t done = 0; done < src.size(); done += slice) {
        const std::size_t n = std::min(slice, src.size() - done);
        check(cudaMemcpyAsync(static_cast<std::byte*>(dst) + offset + done, src.data() + done, n, cudaMemcpyHostToDevice, stream_), "weight upload");
        check(cudaStreamSynchronize(stream_), "weight upload sync");
    }
    uploads_.push_back({name, f.info(name).begin, src.size(), static_cast<std::byte*>(dst) + offset, &f});
}

QW Model::load_fp8(const std::string& p) {
    const auto& wi = main_->info(p + ".weight");
    const auto& si = main_->info(p + ".weight_scale");
    if(wi.dtype != "F8_E4M3" || si.dtype != "BF16") throw std::runtime_error("Unexpected FP8 encoding for " + p);
    QW w;
    w.fmt = W_FP8; w.N = int(wi.shape[0]); w.K = int(wi.shape[1]);
    auto* wd = alloc(wi.bytes());
    auto* sd = alloc(si.bytes());
    upload(*main_, p + ".weight", wd);
    upload(*main_, p + ".weight_scale", sd);
    w.w = static_cast<const std::uint8_t*>(wd);
    w.rs = static_cast<const bf16*>(sd);
    return w;
}
QW Model::load_mlp(const std::string& p) {
    if(!main_->has(p + ".weight_packed")) return load_fp8(p);
    const auto& wi = main_->info(p + ".weight_packed");
    const auto& si = main_->info(p + ".weight_scale");
    const auto& gi = main_->info(p + ".weight_global_scale");
    if(wi.dtype != "U8" || si.dtype != "F8_E4M3" || gi.dtype != "F32" || gi.bytes() != 4) throw std::runtime_error("Unexpected NVFP4 encoding for " + p);
    QW w;
    w.fmt = W_NVFP4; w.N = int(wi.shape[0]); w.K = int(wi.shape[1]) * 2;
    if(si.shape[1] * 16 != w.K) throw std::runtime_error("NVFP4 scale geometry mismatch for " + p);
    auto* wd = alloc(wi.bytes());
    auto* sd = alloc(si.bytes());
    upload(*main_, p + ".weight_packed", wd);
    upload(*main_, p + ".weight_scale", sd);
    std::memcpy(&w.gs, main_->bytes(p + ".weight_global_scale").data(), 4);
    w.w = static_cast<const std::uint8_t*>(wd);
    w.bs = static_cast<const std::uint8_t*>(sd);
    return w;
}
QW Model::load_bf16(const SafetensorsFile& f, const std::string& name) {
    const auto& wi = f.info(name);
    if(wi.dtype != "BF16" || wi.shape.size() != 2) throw std::runtime_error("Unexpected BF16 matrix " + name);
    QW w;
    w.fmt = W_BF16; w.N = int(wi.shape[0]); w.K = int(wi.shape[1]);
    auto* d = alloc(wi.bytes());
    upload(f, name, d);
    w.w = static_cast<const std::uint8_t*>(d);
    return w;
}
const bf16* Model::load_vec(const SafetensorsFile& f, const std::string& name) {
    const auto& wi = f.info(name);
    if(wi.dtype != "BF16") throw std::runtime_error("Unexpected BF16 tensor " + name);
    auto* d = alloc(wi.bytes());
    upload(f, name, d);
    return static_cast<const bf16*>(d);
}

Model::Model(std::shared_ptr<crit::core::CudaBackend> backend, const Config& cfg) : backend_(std::move(backend)), cfg_(cfg) {
    stream_ = backend_->stream(crit::core::Queue::compute);
    checkb(cublasCreate(&cublas_), "cublasCreate");
    checkb(cublasSetStream(cublas_, stream_), "cublasSetStream");
    main_ = std::make_unique<SafetensorsFile>(cfg_.main_file);
    mtpfile_ = std::make_unique<SafetensorsFile>(cfg_.mtp_file);
    const std::string P = "model.language_model.";
    const int ctx = cfg_.context;
    embed_ = load_vec(*main_, P + "embed_tokens.weight");
    layers_.resize(kLayers);
    for(int i = 0; i < kLayers; ++i) {
        auto& L = layers_[i];
        const std::string p = P + "layers." + std::to_string(i) + ".";
        L.full = (i % 4) == 3;
        L.in_norm = load_vec(*main_, p + "input_layernorm.weight");
        L.post_norm = load_vec(*main_, p + "post_attention_layernorm.weight");
        if(L.full) {
            // Whole-row concatenation q | k | v of the stored FP8 rows and their row scales.
            const std::string a = p + "self_attn.";
            const auto& qi = main_->info(a + "q_proj.weight");
            const auto& ki = main_->info(a + "k_proj.weight");
            const auto& vi = main_->info(a + "v_proj.weight");
            QW w; w.fmt = W_FP8; w.K = int(qi.shape[1]);
            w.N = int(qi.shape[0] + ki.shape[0] + vi.shape[0]);
            auto* wd = static_cast<std::uint8_t*>(alloc(std::size_t(w.N) * w.K));
            auto* sd = static_cast<std::uint8_t*>(alloc(std::size_t(w.N) * 2));
            upload(*main_, a + "q_proj.weight", wd, 0);
            upload(*main_, a + "k_proj.weight", wd, qi.bytes());
            upload(*main_, a + "v_proj.weight", wd, qi.bytes() + ki.bytes());
            upload(*main_, a + "q_proj.weight_scale", sd, 0);
            upload(*main_, a + "k_proj.weight_scale", sd, std::size_t(qi.shape[0]) * 2);
            upload(*main_, a + "v_proj.weight_scale", sd, std::size_t(qi.shape[0] + ki.shape[0]) * 2);
            w.w = wd; w.rs = reinterpret_cast<const bf16*>(sd);
            L.qkv = w;
            L.out = load_fp8(a + "o_proj");
            L.q_norm = load_vec(*main_, a + "q_norm.weight");
            L.k_norm = load_vec(*main_, a + "k_norm.weight");
            L.kc = static_cast<bf16*>(alloc(std::size_t(ctx) * 4 * 256 * 2, true));
            L.vc = static_cast<bf16*>(alloc(std::size_t(ctx) * 4 * 256 * 2, true));
        } else {
            const std::string a = p + "linear_attn.";
            L.qkv = load_fp8(a + "in_proj_qkv");
            L.z = load_fp8(a + "in_proj_z");
            L.out = load_fp8(a + "out_proj");
            const auto& ai = main_->info(a + "in_proj_a.weight");
            QW ab; ab.fmt = W_BF16; ab.K = int(ai.shape[1]); ab.N = int(ai.shape[0]) * 2;
            auto* abd = alloc(std::size_t(ab.N) * ab.K * 2);
            upload(*main_, a + "in_proj_a.weight", abd, 0);
            upload(*main_, a + "in_proj_b.weight", abd, ai.bytes());
            ab.w = static_cast<const std::uint8_t*>(abd);
            L.ab = ab;
            auto& g = L.gdn;
            g.conv_w = load_vec(*main_, a + "conv1d.weight");
            g.A_log = load_vec(*main_, a + "A_log");
            g.dt_bias = load_vec(*main_, a + "dt_bias");
            g.norm_w = load_vec(*main_, a + "norm.weight");
            g.conv_hist = static_cast<float*>(alloc(3 * 10240 * 4, true));
            g.conv_stash = static_cast<float*>(alloc(std::size_t(kTv) * 10240 * 4, true));
            g.state = static_cast<float*>(alloc(std::size_t(48) * 128 * 128 * 4, true));
            g.rec_stash = static_cast<float*>(alloc(std::size_t(48 * 4) * kTv * kGdnStash * 4, true));
        }
        L.gate = load_mlp(p + "mlp.gate_proj");
        L.up = load_mlp(p + "mlp.up_proj");
        L.down = load_mlp(p + "mlp.down_proj");
    }
    final_norm_ = load_vec(*main_, P + "norm.weight");
    lm_head_ = load_fp8("lm_head");
    // Multi-token prediction module (BF16).
    auto& m = mtp_;
    m.fc = load_bf16(*mtpfile_, "mtp.fc.weight");
    {
        const auto& qi = mtpfile_->info("mtp.layers.0.self_attn.q_proj.weight");
        const auto& ki = mtpfile_->info("mtp.layers.0.self_attn.k_proj.weight");
        QW w; w.fmt = W_BF16; w.K = int(qi.shape[1]); w.N = int(qi.shape[0] + 2 * ki.shape[0]);
        auto* wd = alloc(std::size_t(w.N) * w.K * 2);
        upload(*mtpfile_, "mtp.layers.0.self_attn.q_proj.weight", wd, 0);
        upload(*mtpfile_, "mtp.layers.0.self_attn.k_proj.weight", wd, qi.bytes());
        upload(*mtpfile_, "mtp.layers.0.self_attn.v_proj.weight", wd, qi.bytes() + ki.bytes());
        w.w = static_cast<const std::uint8_t*>(wd);
        m.qkv = w;
    }
    m.o = load_bf16(*mtpfile_, "mtp.layers.0.self_attn.o_proj.weight");
    m.gate = load_bf16(*mtpfile_, "mtp.layers.0.mlp.gate_proj.weight");
    m.up = load_bf16(*mtpfile_, "mtp.layers.0.mlp.up_proj.weight");
    m.down = load_bf16(*mtpfile_, "mtp.layers.0.mlp.down_proj.weight");
    m.in_norm = load_vec(*mtpfile_, "mtp.layers.0.input_layernorm.weight");
    m.post_norm = load_vec(*mtpfile_, "mtp.layers.0.post_attention_layernorm.weight");
    m.q_norm = load_vec(*mtpfile_, "mtp.layers.0.self_attn.q_norm.weight");
    m.k_norm = load_vec(*mtpfile_, "mtp.layers.0.self_attn.k_norm.weight");
    m.norm = load_vec(*mtpfile_, "mtp.norm.weight");
    m.pre_e = load_vec(*mtpfile_, "mtp.pre_fc_norm_embedding.weight");
    m.pre_h = load_vec(*mtpfile_, "mtp.pre_fc_norm_hidden.weight");
    m.kc = static_cast<bf16*>(alloc(std::size_t(ctx) * 4 * 256 * 2, true));
    m.vc = static_cast<bf16*>(alloc(std::size_t(ctx) * 4 * 256 * 2, true));
    mtp_draft_ = m;
    if(cfg_.mtp_fp8) {
        // Draft-only FP8 copies derived on the device; the BF16 originals stay resident and unchanged.
        auto derive = [&](const QW& src) {
            QW q; q.fmt = W_FP8; q.N = src.N; q.K = src.K;
            auto* qd = static_cast<std::uint8_t*>(alloc(std::size_t(src.N) * src.K, true));
            auto* sd = static_cast<bf16*>(alloc(std::size_t(src.N) * 2, true));
            quantize_fp8_rows(src, qd, sd, stream_);
            q.w = qd; q.rs = sd;
            return q;
        };
        auto& d = mtp_draft_;
        d.fc = derive(m.fc); d.qkv = derive(m.qkv); d.o = derive(m.o);
        d.gate = derive(m.gate); d.up = derive(m.up); d.down = derive(m.down);
    }

    // Activations and scratch.
    const std::size_t TP = std::size_t(cfg_.prefill_chunk) + 1;
    max_splits_ = (ctx + cfg_.attn_split - 1) / cfg_.attn_split;
    prefill_splits_ = (ctx + 1023) / 1024;
    auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4, true)); };
    auto b16 = [&](std::size_t n) { return static_cast<bf16*>(alloc(n * 2, true)); };
    auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4, true)); };
    resid_ = f32(TP * kHidden);
    partial_res_ = f32(std::size_t(kResSplit) * kTv * kHidden);
    big_ = f32(TP * 2 * kInter);
    conv_out_ = f32(TP * 10240);
    core_ = f32(TP * 6144);
    qbuf_ = f32(TP * 6144);
    abuf_ = f32(TP * 96);
    partial_ = f32(std::max(TP * 24 * prefill_splits_, std::size_t(kTv) * 24 * max_splits_) * 258);
    logits_ = f32(std::size_t(8) * kVocab);
    tmp_ = f32(TP * kInter);
    score_out_ = f32(8);
    xn_ = b16(TP * kHidden);
    act_ = b16(TP * kInter);
    xfinal_ = b16(TP * kHidden);
    cat_ = b16(TP * 2 * kHidden);
    dq_ = b16(std::size_t(kInter) * kHidden);
    vhid_ = b16(std::size_t(kTv) * kHidden);
    mtp_hin_ = b16(TP * kHidden);
    mtp_h_ = b16(TP * kHidden);
    pend_h_ = b16(kHidden);
    tok_ = i32(TP);
    pos_ = i32(4);
    check(cudaMemset(pos_, 0, 16), "pos init");
    mtok_ = i32(TP);
    mpos_ = i32(kTv);
    pending_ = i32(1);
    prev_n_ = i32(1);
    prev_tok_ = i32(kTv);
    targets_ = i32(8);
    check(cudaMallocHost(&host_ints_, 64), "pinned ints");

    // Device speculative round (crit Forge device-speculative-round).
    const int K = cfg_.draft_tokens;
    dr::RoundSteps steps{
        [this](cudaStream_t, const dr::RoundView& v) { draft_step(v.control, v.drafts, int(v.draft_tokens)); },
        [this](cudaStream_t, const dr::RoundView& v) { verify_step(v.control, v.drafts, v.predictions, int(v.draft_tokens)); },
        [this](cudaStream_t, const dr::RoundView& v) { commit_step(v.control, v.result, int(v.draft_tokens)); }};
    dr::RoundConfig rc;
    rc.draft_tokens = std::uint32_t(K);
    rc.vocabulary = kVocab;
    rc.context = std::uint32_t(ctx);
    rc.stop_tokens = {248046, 248044};
    rc.max_stops = 16;
    rc.graphs = cfg_.graphs;
    round_ = std::make_unique<dr::DeviceSpeculativeRound>(backend_, rc, std::move(steps));
    check(cudaStreamSynchronize(stream_), "load sync");
}

Model::~Model() {
    round_.reset();
    cudaStreamSynchronize(stream_);
    if(cublas_) cublasDestroy(cublas_);
    if(host_ints_) cudaFreeHost(host_ints_);
}

void Model::set_device_int(int* dst, int value) {
    check(cudaStreamSynchronize(stream_), "sync before int upload");
    host_ints_[0] = value;
    check(cudaMemcpyAsync(dst, host_ints_, 4, cudaMemcpyHostToDevice, stream_), "int upload");
    check(cudaStreamSynchronize(stream_), "int upload sync");
}

void Model::proj(const QW& w, const bf16* x, int ldx, int T, float* y, int ldy, bool accumulate) {
    if(T <= kTv) {
        GemvSeg s{w, y, ldy, accumulate ? 1 : 0};
        gemv(s, nullptr, x, ldx, T, stream_);
        return;
    }
    const bf16* wb = reinterpret_cast<const bf16*>(w.w);
    if(w.fmt != W_BF16) { dequant_bf16(w, dq_, stream_); wb = dq_; }
    const float one = 1.f, zero = 0.f;
    checkb(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, w.N, T, w.K, &one, wb, CUDA_R_16BF, w.K, x, CUDA_R_16BF, ldx,
                        &zero, tmp_, CUDA_R_32F, w.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "prefill GEMM");
    row_scale_epilogue(w, tmp_, y, ldy, T, accumulate ? 1 : 0, stream_);
}
void Model::proj2(const QW& a, float* ya, const QW& b, float* yb, int ldy, const bf16* x, int ldx, int T) {
    if(T <= kTv) {
        GemvSeg sa{a, ya, ldy, 0}, sb{b, yb, ldy, 0};
        gemv(sa, &sb, x, ldx, T, stream_);
        return;
    }
    proj(a, x, ldx, T, ya, ldy, false);
    proj(b, x, ldx, T, yb, ldy, false);
}

// act_ = bf16(silu(gate(xn_)) * up(xn_)); fused into the GEMV epilogue for decode widths.
void Model::mlp_up(const QW& gate, const QW& up, int T) {
    if(T <= kTv) {
        GemvSeg g{gate, nullptr, kInter, 0, up, act_};
        gemv_multi(&g, 1, xn_, kHidden, T, stream_);
        return;
    }
    proj2(gate, big_, up, big_ + kInter, 2 * kInter, xn_, kHidden, T);
    swiglu(big_, 2 * kInter, act_, T, kInter, stream_);
}

void Model::target_forward(int T, const int* tokens, const int* base, int mode, int splits, int split) {
    auto s = stream_;
    mark("start");
    embed(tokens, embed_, resid_, T, kHidden, s);
    mark("embed");
    // Decode widths: residual-producing projections write split-K partials that the
    // following residual_norm adds to the residual in a fixed order before normalizing.
    const bool use_split = T <= kTv;
    auto resid_out = [&](const QW& w, const bf16* x, int ldx, const bf16* next_norm, bf16* norm_out, const char* name) {
        if(use_split) {
            GemvSeg g{w, partial_res_, kHidden, 0};
            g.ksplit = kResSplit;
            g.split_stride = std::size_t(kTv) * kHidden;
            gemv_multi(&g, 1, x, ldx, T, s);
            mark(name);
            residual_norm(resid_, partial_res_, kResSplit, std::size_t(kTv) * kHidden, next_norm, norm_out, T, kHidden, kEps, s);
        } else {
            proj(w, x, ldx, T, resid_, kHidden, true);
            mark(name);
            rmsnorm(resid_, kHidden, next_norm, norm_out, kHidden, T, kHidden, kEps, 1, s);
        }
        mark("norm");
    };
    rmsnorm(resid_, kHidden, layers_[0].in_norm, xn_, kHidden, T, kHidden, kEps, 1, s);
    mark("norm");
    for(int i = 0; i < kLayers; ++i) {
        auto& L = layers_[i];
        if(!L.full) {
            if(T <= kTv) {
                GemvSeg segs[3] = {{L.qkv, big_, kGdnLd, 0}, {L.z, big_ + 10240, kGdnLd, 0}, {L.ab, abuf_, 96, 0}};
                gemv_multi(segs, 3, xn_, kHidden, T, s);
            } else {
                proj2(L.qkv, big_, L.z, big_ + 10240, kGdnLd, xn_, kHidden, T);
                proj(L.ab, xn_, kHidden, T, abuf_, 96, false);
            }
            mark("gdn qkvz+ab");
            if(mode == 1) {
                gdn_conv(L.gdn, big_, kGdnLd, conv_out_, T, mode, pending_, s, 1);
                mark("gdn conv");
                gdn_decode(L.gdn, conv_out_, abuf_, 96, big_ + 10240, kGdnLd, act_, T, pending_, kTv, kEps, s);
                mark("gdn recurrent+norm");
            } else {
                gdn_conv(L.gdn, big_, kGdnLd, conv_out_, T, mode, pending_, s);
                mark("gdn conv");
                gdn_recurrent(L.gdn, conv_out_, abuf_, 96, core_, T, mode, pending_, kTv, s);
                mark("gdn recurrent");
                gdn_gate_norm(core_, big_ + 10240, kGdnLd, L.gdn.norm_w, act_, T, kEps, s);
                mark("gdn gate norm");
            }
            resid_out(L.out, act_, 6144, L.post_norm, xn_, "gdn out fp8");
        } else {
            proj(L.qkv, xn_, kHidden, T, big_, L.qkv.N, false);
            mark("attn qkv fp8");
            attn_prep(big_, L.qkv.N, L.q_norm, L.k_norm, base, T, qbuf_, L.kc, L.vc, kEps, s);
            mark("attn prep");
            attention(qbuf_, big_, L.qkv.N, L.kc, L.vc, base, T, split, splits, partial_, act_, s);
            mark("attention");
            resid_out(L.out, act_, 6144, L.post_norm, xn_, "attn o fp8");
        }
        mlp_up(L.gate, L.up, T);
        mark(L.gate.fmt == W_NVFP4 ? "mlp gate/up nvfp4" : "mlp gate/up fp8");
        const bool last = i + 1 == kLayers;
        resid_out(L.down, act_, kInter, last ? final_norm_ : layers_[i + 1].in_norm, last ? xfinal_ : xn_,
                  L.down.fmt == W_NVFP4 ? "mlp down nvfp4" : "mlp down fp8");
    }
    mark("norm");
}

void Model::mtp_forward(const MtpW& m, int T, const int* tokens, const bf16* hidden, int ldh, const int* base, int splits, int split, bf16* out) {
    auto s = stream_;
    mtp_inputs(tokens, embed_, hidden, ldh, m.pre_e, m.pre_h, cat_, T, kEps, s);
    proj(m.fc, cat_, 2 * kHidden, T, resid_, kHidden, false);
    rmsnorm(resid_, kHidden, m.in_norm, xn_, kHidden, T, kHidden, kEps, 1, s);
    proj(m.qkv, xn_, kHidden, T, big_, m.qkv.N, false);
    attn_prep(big_, m.qkv.N, m.q_norm, m.k_norm, base, T, qbuf_, m.kc, m.vc, kEps, s);
    attention(qbuf_, big_, m.qkv.N, m.kc, m.vc, base, T, split, splits, partial_, act_, s);
    proj(m.o, act_, 6144, T, resid_, kHidden, true);
    rmsnorm(resid_, kHidden, m.post_norm, xn_, kHidden, T, kHidden, kEps, 1, s);
    mlp_up(m.gate, m.up, T);
    proj(m.down, act_, kInter, T, resid_, kHidden, true);
    rmsnorm(resid_, kHidden, m.norm, out, kHidden, T, kHidden, kEps, 1, s);
}

void Model::mark(const char* name) {
    if(!profiling_) return;
    cudaEvent_t e;
    check(cudaEventCreate(&e), "event");
    check(cudaEventRecord(e, stream_), "record");
    marks_.push_back({name, e});
}

std::string Model::profile(int T, int reps) {
    if(T < 1 || T > kTv) throw std::invalid_argument("profile columns");
    std::map<std::string, double> total;
    std::map<std::string, int> count;
    reset_request();
    set_device_int(pos_, 600);
    check(cudaMemsetAsync(tok_, 0, kTv * 4, stream_), "tokens");
    for(int r = 0; r < reps + 1; ++r) {
        profiling_ = true;
        target_forward(T, tok_, pos_, 1, max_splits_, cfg_.attn_split);
        GemvSeg hs{lm_head_, logits_, kVocab, 0};
        gemv(hs, nullptr, xfinal_, kHidden, T, stream_);
        mark("lm_head fp8");
        argmax(logits_, kVocab, T, kValidVocab, tok_ + 8, stream_);
        mark("argmax");
        if(r) {
            stream_read(lm_head_.w, std::size_t(lm_head_.N) * lm_head_.K, logits_, stream_);
            mark("stream read lm_head bytes");
        }
        if(r && T == 1) {
            QW head = lm_head_;
            if(cfg_.draft_rows) head.N = cfg_.draft_rows;
            mtp_forward(mtp_draft_, 1, tok_, xfinal_, kHidden, pos_, max_splits_, cfg_.attn_split, mtp_hin_);
            mark("mtp layer (bf16)");
            GemvSeg ds{head, logits_, kVocab, 0};
            gemv(ds, nullptr, mtp_hin_, kHidden, 1, stream_);
            mark("draft head");
        }
        profiling_ = false;
        check(cudaStreamSynchronize(stream_), "profile sync");
        for(std::size_t i = 1; i < marks_.size(); ++i) {
            float ms = 0;
            check(cudaEventElapsedTime(&ms, marks_[i - 1].second, marks_[i].second), "elapsed");
            if(r) { total[marks_[i].first] += ms; count[marks_[i].first] += 1; }
        }
        for(auto& [n, e] : marks_) cudaEventDestroy(e);
        marks_.clear();
    }
    std::string out;
    double sum = 0;
    for(auto& [n, ms] : total) sum += ms / reps;
    for(auto& [n, ms] : total) {
        char line[160];
        std::snprintf(line, sizeof line, "%-22s %8.3f ms  %5.1f%%  (%d launches/forward)\n", n.c_str(), ms / reps, 100.0 * ms / reps / sum, count[n] / reps);
        out += line;
    }
    char line[96];
    std::snprintf(line, sizeof line, "total                  %8.3f ms at T=%d\n", sum, T);
    return out + line;
}

std::string Model::micro(int T) {
    std::string out;
    auto bytes_of = [](const QW& w) {
        const double n = double(w.N) * w.K;
        return w.fmt == W_NVFP4 ? n / 2 + n / 16 : w.fmt == W_FP8 ? n + w.N * 2.0 : n * 2;
    };
    auto run = [&](const char* name, const GemvSeg* segs, int nseg, const bf16* x, int ldx, double bytes) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        for(int i = 0; i < 5; ++i) gemv_multi(segs, nseg, x, ldx, T, stream_);
        cudaEventRecord(a, stream_);
        constexpr int reps = 50;
        for(int i = 0; i < reps; ++i) gemv_multi(segs, nseg, x, ldx, T, stream_);
        cudaEventRecord(b, stream_);
        check(cudaEventSynchronize(b), "micro");
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double us = ms * 1000.0 / reps;
        char line[160];
        std::snprintf(line, sizeof line, "%-28s %8.1f us  %6.3f TB/s\n", name, us, bytes / (us * 1e-6) / 1e12);
        out += line;
        cudaEventDestroy(a); cudaEventDestroy(b);
    };
    const auto& L0 = layers_[0];
    const auto& L60 = layers_[60];
    GemvSeg fused{L0.gate, nullptr, kInter, 0, L0.up, act_};
    run("gate/up nvfp4 fused L0", &fused, 1, xn_, kHidden, bytes_of(L0.gate) * 2);
    GemvSeg two[2] = {{L0.gate, big_, 2 * kInter, 0}, {L0.up, big_ + kInter, 2 * kInter, 0}};
    run("gate/up nvfp4 2-seg L0", two, 2, xn_, kHidden, bytes_of(L0.gate) * 2);
    GemvSeg fused8{L60.gate, nullptr, kInter, 0, L60.up, act_};
    run("gate/up fp8 fused L60", &fused8, 1, xn_, kHidden, bytes_of(L60.gate) * 2);
    GemvSeg down{L0.down, tmp_, kHidden, 0};
    run("down nvfp4 L0", &down, 1, act_, kInter, bytes_of(L0.down));
    GemvSeg down8{L60.down, tmp_, kHidden, 0};
    run("down fp8 L60", &down8, 1, act_, kInter, bytes_of(L60.down));
    GemvSeg q3[3] = {{L0.qkv, big_, 16384, 0}, {L0.z, big_ + 10240, 16384, 0}, {L0.ab, abuf_, 96, 0}};
    run("gdn qkv+z+ab L0", q3, 3, xn_, kHidden, bytes_of(L0.qkv) + bytes_of(L0.z) + bytes_of(L0.ab));
    GemvSeg o{L0.out, tmp_, kHidden, 0};
    run("gdn out fp8 L0", &o, 1, act_, 6144, bytes_of(L0.out));
    GemvSeg head{lm_head_, logits_, kVocab, 0};
    run("lm_head fp8", &head, 1, xfinal_, kHidden, bytes_of(lm_head_));
    return out;
}

void Model::reset_request() {
    auto s = stream_;
    for(auto& L : layers_) {
        if(L.full) continue;
        check(cudaMemsetAsync(L.gdn.state, 0, std::size_t(48) * 128 * 128 * 4, s), "reset state");
        check(cudaMemsetAsync(L.gdn.conv_hist, 0, 3 * 10240 * 4, s), "reset conv");
    }
    check(cudaMemsetAsync(pending_, 0, 4, s), "reset pending");
    check(cudaMemsetAsync(prev_n_, 0, 4, s), "reset prev");
    check(cudaMemsetAsync(prev_tok_, 0, kTv * 4, s), "reset prev tokens");
    check(cudaMemsetAsync(vhid_, 0, std::size_t(kTv) * kHidden * 2, s), "reset hidden");
    have_pending_ = false;
    prefill_ms_total = 0;
}

void Model::prefill(const int* host_tokens, int n, int pos0) {
    if(n <= 0) return;
    if(n > cfg_.prefill_chunk) throw std::length_error("Prefill chunk too large");
    auto s = stream_;
    check(cudaStreamSynchronize(s), "prefill sync");
    check(cudaMemcpy(tok_, host_tokens, std::size_t(n) * 4, cudaMemcpyHostToDevice), "prefill tokens");
    set_device_int(pos_, pos0);
    target_forward(n, tok_, pos_, 0, prefill_splits_, 1024);
    // MTP prefill: column (h_{p-1}, x_p) for every materialized p with a known successor.
    int cols = n - 1;
    const bf16* hidden = xfinal_;
    const int* toks = tok_ + 1;
    int mbase = pos0;
    if(have_pending_) {
        check(cudaMemcpyAsync(mtp_hin_, pend_h_, kHidden * 2, cudaMemcpyDeviceToDevice, s), "mtp pending");
        if(n > 1) check(cudaMemcpyAsync(mtp_hin_ + kHidden, xfinal_, std::size_t(n - 1) * kHidden * 2, cudaMemcpyDeviceToDevice, s), "mtp hidden");
        cols = n; hidden = mtp_hin_; toks = tok_; mbase = pos0 - 1;
    }
    if(cols > 0) {
        set_device_int(pos_ + 1, mbase);
        mtp_forward(mtp_, cols, toks, hidden, kHidden, pos_ + 1, prefill_splits_, 1024, mtp_h_);
    }
    check(cudaMemcpyAsync(pend_h_, xfinal_ + std::size_t(n - 1) * kHidden, kHidden * 2, cudaMemcpyDeviceToDevice, s), "pending hidden");
    have_pending_ = true;
}

void Model::finish_prefill(int prompt_len, int last_token) {
    auto s = stream_;
    if(have_pending_) check(cudaMemcpyAsync(vhid_, pend_h_, kHidden * 2, cudaMemcpyDeviceToDevice, s), "seed hidden");
    seed_kernel<<<1, 1, 0, s>>>(pending_, prev_n_, prev_tok_, have_pending_ && prompt_len > 1 ? 1 : 0, last_token);
    check(cudaPeekAtLastError(), "seed");
}

void Model::draft_step(const int* control, int* drafts, int K) {
    auto s = stream_;
    mtp_positions_kernel<<<1, 32, 0, s>>>(control, prev_n_, mpos_, K);
    check(cudaPeekAtLastError(), "mtp positions");
    QW head = lm_head_;
    if(cfg_.draft_rows) head.N = cfg_.draft_rows;
    const int valid = std::min(head.N, kValidVocab);
    // A row-subset head also drafts the chat control tokens at the end of the vocabulary.
    constexpr int kSpecialBegin = 248032, kSpecialRows = 64;
    const bool specials = cfg_.draft_rows && cfg_.draft_rows <= kSpecialBegin;
    QW special = lm_head_;
    special.N = kSpecialRows;
    special.w = lm_head_.w + std::size_t(kSpecialBegin) * lm_head_.K;
    special.rs = lm_head_.rs + kSpecialBegin;
    // Catch up on the previous round's accepted columns, then chain.
    mtp_forward(mtp_draft_, K + 1, prev_tok_, vhid_, kHidden, mpos_, max_splits_, cfg_.attn_split, mtp_h_);
    gather_row_kernel<<<1, 256, 0, s>>>(mtp_h_, prev_n_, mtp_hin_);
    check(cudaPeekAtLastError(), "gather");
    for(int j = 0; j < K; ++j) {
        if(j) mtp_forward(mtp_draft_, 1, drafts + j - 1, mtp_hin_, kHidden, mpos_ + j, max_splits_, cfg_.attn_split, mtp_hin_);
        GemvSeg hs[2] = {{head, logits_, kVocab, 0}, {special, logits_ + kSpecialBegin, kVocab, 0}};
        gemv_multi(hs, specials ? 2 : 1, mtp_hin_, kHidden, 1, s);
        argmax(logits_, kVocab, 1, valid, drafts + j, s, specials ? kSpecialBegin : 0, specials ? kValidVocab : 0);
    }
}

void Model::verify_step(const int* control, const int* drafts, int* predictions, int K) {
    auto s = stream_;
    const int T = K + 1;
    verify_tokens(control, drafts, K, tok_, s);
    target_forward(T, tok_, control + dr::control::frontier, 1, max_splits_, cfg_.attn_split);
    GemvSeg hs{lm_head_, logits_, kVocab, 0};
    gemv(hs, nullptr, xfinal_, kHidden, T, s);
    argmax(logits_, kVocab, T, kValidVocab, predictions, s);
    check(cudaMemcpyAsync(vhid_, xfinal_, std::size_t(T) * kHidden * 2, cudaMemcpyDeviceToDevice, s), "verify hidden");
}

void Model::commit_step(const int* control, const int* result, int K) {
    commit_kernel<<<1, 32, 0, stream_>>>(control, result, K, pending_, prev_n_, prev_tok_);
    check(cudaPeekAtLastError(), "commit");
}

std::vector<float> Model::score(const std::vector<int>& tokens, int first) {
    const int n = int(tokens.size());
    if(first < 1 || first >= n) throw std::invalid_argument("Invalid first target");
    if(n > cfg_.context) throw std::length_error("Score exceeds context");
    reset_request();
    std::vector<float> out;
    std::vector<int> targets(8);
    for(int c0 = 0; c0 < n - 1; c0 += cfg_.prefill_chunk) {
        const int cn = std::min(cfg_.prefill_chunk, n - 1 - c0);
        check(cudaStreamSynchronize(stream_), "score sync");
        check(cudaMemcpy(tok_, tokens.data() + c0, std::size_t(cn) * 4, cudaMemcpyHostToDevice), "score tokens");
        set_device_int(pos_, c0);
        target_forward(cn, tok_, pos_, 0, prefill_splits_, 1024);
        // Column j predicts tokens[c0 + j + 1].
        for(int j0 = std::max(0, first - 1 - c0); j0 < cn; j0 += 8) {
            const int g = std::min(8, cn - j0);
            for(int i = 0; i < g; ++i) targets[i] = tokens[c0 + j0 + i + 1];
            check(cudaStreamSynchronize(stream_), "score sync");
            check(cudaMemcpy(targets_, targets.data(), std::size_t(g) * 4, cudaMemcpyHostToDevice), "targets");
            GemvSeg hs{lm_head_, logits_, kVocab, 0};
            gemv(hs, nullptr, xfinal_ + std::size_t(j0) * kHidden, kHidden, g, stream_);
            log_prob(logits_, kVocab, g, kValidVocab, targets_, score_out_, stream_);
            float host[8];
            check(cudaMemcpyAsync(host, score_out_, std::size_t(g) * 4, cudaMemcpyDeviceToHost, stream_), "score readback");
            check(cudaStreamSynchronize(stream_), "score readback sync");
            out.insert(out.end(), host, host + g);
        }
    }
    return out;
}

// Same targets through the decode path: one column per step in verify mode
// (deferred recurrent replay, split-K attention, tensor-core GEMV head).
std::vector<float> Model::score_decode(const std::vector<int>& tokens, int first) {
    const int n = int(tokens.size());
    if(first < 1 || first >= n) throw std::invalid_argument("Invalid first target");
    reset_request();
    std::vector<float> out;
    check(cudaStreamSynchronize(stream_), "sync");
    check(cudaMemcpy(mtok_, tokens.data(), std::size_t(std::min(n, cfg_.prefill_chunk)) * 4, cudaMemcpyHostToDevice), "tokens");
    for(int p = 0; p + 1 < n; ++p) {
        set_device_int(pos_, p);
        set_device_int(targets_, tokens[p + 1]);
        target_forward(1, mtok_ + p, pos_, 1, max_splits_, cfg_.attn_split);
        set_int(pending_, pos_ + 2, 1, stream_); // pos_[2] is zero: pending = 1 after the first step
        if(p + 1 < first) continue;
        GemvSeg hs{lm_head_, logits_, kVocab, 0};
        gemv(hs, nullptr, xfinal_, kHidden, 1, stream_);
        log_prob(logits_, kVocab, 1, kValidVocab, targets_, score_out_, stream_);
        float v = 0;
        check(cudaMemcpyAsync(&v, score_out_, 4, cudaMemcpyDeviceToHost, stream_), "readback");
        check(cudaStreamSynchronize(stream_), "sync");
        out.push_back(v);
    }
    return out;
}

std::size_t Model::audit(std::string& report) {
    check(cudaStreamSynchronize(stream_), "audit sync");
    std::vector<std::byte> host;
    std::size_t bytes = 0, mismatches = 0;
    constexpr std::size_t slice = std::size_t(64) << 20;
    for(const auto& u : uploads_) {
        const auto src = u.file->bytes(u.name);
        for(std::size_t done = 0; done < u.bytes; done += slice) {
            const std::size_t n = std::min(slice, u.bytes - done);
            host.resize(n);
            check(cudaMemcpy(host.data(), static_cast<const std::byte*>(u.device) + done, n, cudaMemcpyDeviceToHost), "audit readback");
            if(std::memcmp(host.data(), src.data() + done, n) != 0) {
                ++mismatches;
                report += "mismatch: " + u.name + "\n";
            }
            bytes += n;
        }
    }
    report += "tensors " + std::to_string(uploads_.size()) + " bytes " + std::to_string(bytes) + " mismatches " + std::to_string(mismatches) + "\n";
    return mismatches;
}
}
