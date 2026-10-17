// crit Core integration of the fnl network (Model / Session / Backend) and the C API the Python adapters load.
#include "model.h"
#include "score.cuh"
#include <crit/core.h>
#include <crit/cuda.h>
#include <crit/json.h>
#include <crit/sampling.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <sstream>

namespace fnl {
namespace {
using namespace crit::core;

// FNL_TIMING: host-side step timing of the decode loop (advance enqueue, readiness, Core between steps).
struct StepClock {
    bool on = std::getenv("FNL_TIMING") != nullptr;
    std::chrono::steady_clock::time_point last_ready{}, advance_in{}, advance_out{};
    double enqueue = 0, wait = 0, between = 0;
    long steps = 0;
    static double ms(std::chrono::steady_clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
    void in() { if(on) advance_in = std::chrono::steady_clock::now(); }
    void out() { if(on) advance_out = std::chrono::steady_clock::now(); }
    void ready() {
        if(!on) return;
        const auto now = std::chrono::steady_clock::now();
        if(last_ready.time_since_epoch().count()) between += ms(advance_in - last_ready);
        enqueue += ms(advance_out - advance_in);
        wait += ms(now - advance_out);
        last_ready = now;
        if(++steps % 50 == 0)
            std::fprintf(stderr, "steps %ld: enqueue %.2f ms, wait %.2f ms, between %.2f ms per step\n", steps,
                         enqueue / steps, wait / steps, between / (steps - 1));
    }
};
StepClock step_clock;

constexpr std::uint32_t kValidVocab = 248077;   // tokenizer size; rows beyond are padding
const std::vector<Token> kStops = {248044, 248046};

struct Config {
    std::string model;
    int capacity = 262144;
    int chunk = 16384;
    std::size_t slots = 0;
    double vram_margin_gib = 1.0;   // device memory left free at load (~0.97 GiB stays free at peak, nvidia-smi)
    // Sampling defaults (generation_config.json)
    double temperature = 1.0, top_p = 0.95;
    std::uint32_t top_k = 20;
    bool speculative = true;   // greedy requests decode in MTP-drafted rounds of 4 verified tokens
};

class FnlModel;

class FnlSession final : public Session {
public:
    FnlSession(FnlModel& m, std::span<const Token> prompt, const RequestOptions& options);
    Progress advance(WorkBudget budget) override;
    std::span<const Token> ready_tokens() override;
    void commit(std::uint32_t accepted, bool terminal) override;
    void close(Finish) noexcept override {}
private:
    FnlModel& m_;
    std::vector<int> history_;
    std::size_t prompt_ = 0, frontier_ = 0, pending_frontier_ = 0;
    bool decoding_ = false, first_ready_ = false, started_ = false;
    Token pending_ = 0;
    SamplingParams params_;
    TokenHistory th_;
    bool greedy_ = true;
    // speculative rounds
    bool spec_ = false, spec_started_ = false;
    int drafts_[kDrafts] = {0};
    std::vector<Token> round_;      // tokens of the last round (kept drafts, then the model's token)
    std::uint32_t out_limit_ = 1;   // output tokens this step may return
};

class FnlModel final : public Model {
public:
    FnlModel(const Config& c, Services s)
        : config_(c), resources_(s.resources), backend_(dynamic_cast<CudaBackend&>(s.backend)), sampler_(kValidVocab) {
        Options o;
        o.model = c.model;
        o.capacity = c.capacity;
        o.prefill_chunk = c.chunk;
        o.expert_slots = c.slots;
        o.vram_margin = std::size_t(c.vram_margin_gib * double(std::size_t(1) << 30));
        o.mtp = c.speculative;
        net_ = std::make_unique<Net>(o, backend_, resources_);
        auto h = backend_.allocate(Memory::host, kVocab * 4 + 4096);
        host_ = std::move(h);
        token_host_ = static_cast<int*>(host_.data());
        logits_host_ = reinterpret_cast<float*>(static_cast<uint8_t*>(host_.data()) + 4096);
        ready_ = backend_.event();
    }
    ModelInfo info() const override {
        ModelInfo i;
        i.implementation = "fnl (qwen4_exp Flash-Next FP4-Local)";
        i.vocabulary = kVocab;
        i.valid_vocabulary = kValidVocab;
        i.context = std::uint32_t(config_.capacity);
        i.prefill_chunk = std::uint32_t(config_.chunk);
        i.stop_tokens = kStops;
        i.scoring = true;
        i.query_scoring = true;
        i.score_paths = {"prefill", "decode", "verify"};
        i.sampling.temperature = i.sampling.top_p = i.sampling.top_k = i.sampling.seed = true;
        return i;
    }
    std::unique_ptr<Session> open(std::span<const Token> prompt) override { return open_request(prompt, {}); }
    std::unique_ptr<Session> open_request(std::span<const Token> prompt, const RequestOptions& options) override {
        if(prompt.empty()) throw std::invalid_argument("empty prompt");
        if(prompt.size() >= std::size_t(config_.capacity)) throw std::invalid_argument("prompt exceeds the context");
        net_->reset();
        accepted_.clear();
        return std::make_unique<FnlSession>(*this, prompt, options);
    }
    QueryScores score_queries(std::span<const Token> ids, std::uint32_t first, const ScoreQueries& q,
                              const RequestOptions&) override;
    std::vector<float> score(std::span<const Token> ids, std::uint32_t first) override {
        ScoreQueries q;
        return score_queries(ids, first, q, {}).target;
    }

    Net& net() { return *net_; }
    CudaBackend& backend() { return backend_; }
    Sampler& sampler() { return sampler_; }
    const Config& config() const { return config_; }
    int* token_host_ = nullptr;
    float* logits_host_ = nullptr;
    CudaEvent ready_;
    std::vector<int> accepted_;    // drafts kept per speculative round of the last request
private:
    Config config_;
    Resources& resources_;
    CudaBackend& backend_;
    std::unique_ptr<Net> net_;
    CudaBuffer host_;
    Sampler sampler_;
};

FnlSession::FnlSession(FnlModel& m, std::span<const Token> prompt, const RequestOptions& options)
    : m_(m), history_(prompt.begin(), prompt.end()), prompt_(prompt.size()), th_(prompt) {
    SamplingParams defaults;
    defaults.temperature = m.config().temperature;
    defaults.top_p = m.config().top_p;
    defaults.top_k = m.config().top_k;
    defaults.seed = 0x5eedULL;
    params_ = resolve_sampling(options.sampling, defaults, kValidVocab);
    greedy_ = params_.temperature == 0.0;
    spec_ = greedy_ && m.net().has_mtp() && m.config().speculative;
}

Progress FnlSession::advance(WorkBudget budget) {
    struct Out { ~Out() { step_clock.out(); } } stamp_out;
    step_clock.in();
    auto& net = m_.net();
    const auto s = net.stream();
    decoding_ = budget.prefill_complete;
    if(!started_) {
        started_ = true;
        // Speculative rounds may write up to kMaxTokens + 2 positions past the last kept token.
        const std::size_t extra = spec_ ? kMaxTokens + 3 : 1;
        // The drafter seeds at the last prompt position and writes two more; near the end of the context the
        // rounds give way to one-token passes (below).
        if(spec_ && prompt_ + 3 > std::size_t(net.capacity())) spec_ = false;
        const std::size_t need = std::min<std::size_t>(prompt_ + budget.output_tokens + extra, std::size_t(net.capacity()));
        net.begin_request(int(need));
    }
    if(!decoding_) {
        const std::size_t n = std::min<std::size_t>(budget.prefill_tokens, prompt_ - frontier_);
        net.prefill(history_, int(frontier_), int(n), false, spec_);
        pending_frontier_ = frontier_ + n;
        const bool done = pending_frontier_ == prompt_;
        first_ready_ = done;
        m_.ready_.record(s);
        Progress p{std::uint32_t(n), done, {}, &m_.ready_};
        return p;
    }
    if(spec_ && spec_started_ && history_.size() - 1 + kMaxTokens + 2 > std::size_t(net.capacity()))
        spec_ = false;   // a round would write past the KV capacity: the last tokens decode one per pass
    if(spec_) {
        out_limit_ = std::max<std::uint32_t>(1, budget.output_tokens);
        // The first round seeds the drafter with the prompt pass's token; later rounds verify the drafts.
        if(!spec_started_) {
            spec_started_ = true;
            first_ready_ = false;
            net.spec_begin(int(history_.size()) - 1);
            net.spec_begin_prefetch(history_, int(history_.size()) - 1);
        } else {
            net.spec_round(history_, int(history_.size()) - 1, drafts_);
        }
        m_.ready_.record(s);
        Progress p;
        p.prefill_complete = true;
        p.readiness = &m_.ready_;
        p.deferred_tokens = true;
        return p;
    }
    if(first_ready_) {
        first_ready_ = false;   // the prompt pass already produced this position's logits
    } else {
        net.decode(history_, int(history_.size()) - 1);
    }
    if(greedy_) {
        FNL_CUDA(cudaMemcpyAsync(m_.token_host_, net.next_token(), 4, cudaMemcpyDeviceToHost, s));
    } else {
        FNL_CUDA(cudaMemcpyAsync(m_.logits_host_, net.logits(), kValidVocab * 4, cudaMemcpyDeviceToHost, s));
    }
    m_.ready_.record(s);
    Progress p;
    p.prompt_tokens = 0;
    p.prefill_complete = true;
    p.readiness = &m_.ready_;
    p.deferred_tokens = true;
    return p;
}

std::span<const Token> FnlSession::ready_tokens() {
    step_clock.ready();
    if(spec_) {
        const Accept& a = m_.net().spec_result();
        if(history_.size() > prompt_) m_.accepted_.push_back(a.n - 1);
        round_.clear();
        // Kept drafts (the round's inputs after its first token), then the model's token.
        if(history_.size() > prompt_)
            for(int k = 0; k + 1 < a.n; ++k) round_.push_back(Token(drafts_[k]));
        round_.push_back(Token(a.bonus));
        if(round_.size() > out_limit_) round_.resize(out_limit_);
        for(int k = 0; k < kDrafts; ++k) drafts_[k] = a.drafts[k];
        return {round_.data(), round_.size()};
    }
    if(greedy_) {
        pending_ = *m_.token_host_;
        if(spec_started_) m_.accepted_.push_back(0);   // one-token pass after the speculative rounds
    } else {
        th_.rollback();
        const Choice c = m_.sampler().sample(std::span<const float>(m_.logits_host_, kValidVocab), params_, th_);
        pending_ = c.token;
    }
    return {&pending_, 1};
}

void FnlSession::commit(std::uint32_t accepted, bool) {
    if(!decoding_) {
        frontier_ = pending_frontier_;
        return;
    }
    if(spec_) {
        for(std::uint32_t i = 0; i < accepted && i < round_.size(); ++i) history_.push_back(int(round_[i]));
        return;
    }
    if(accepted) {
        history_.push_back(pending_);
        th_.stage({&pending_, 1});
        th_.commit(1);
    }
}

QueryScores FnlModel::score_queries(std::span<const Token> ids_span, std::uint32_t first, const ScoreQueries& q,
                                    const RequestOptions&) {
    const std::vector<int> ids(ids_span.begin(), ids_span.end());
    const int n = int(ids.size());
    if(first < 1 || int(first) > n) throw std::invalid_argument("first target");
    const int targets = n - int(first);
    const int per = int(q.per_target);
    if(q.top_k) throw std::invalid_argument("top_k alternatives are not supported");
    const auto s = net_->stream();
    // Device staging: per target row the target id, its queries and outputs.
    const std::size_t bytes = std::size_t(targets) * (4 + 4 + 4 + per * 8) + 4096;
    auto buf = backend_.allocate(Memory::device, bytes);
    auto* base = static_cast<uint8_t*>(buf.data());
    int* d_t = reinterpret_cast<int*>(base);
    int* d_q = d_t + targets;
    float* d_tlp = reinterpret_cast<float*>(d_q + std::size_t(targets) * per);
    float* d_qlp = d_tlp + targets;
    int* d_am = reinterpret_cast<int*>(d_qlp + std::size_t(targets) * per);
    FNL_CUDA(cudaMemcpyAsync(d_t, ids.data() + first, std::size_t(targets) * 4, cudaMemcpyHostToDevice, s));
    if(per)
        FNL_CUDA(cudaMemcpyAsync(d_q, q.ids.data(), std::size_t(targets) * per * 4, cudaMemcpyHostToDevice, s));
    net_->reset();
    net_->begin_request(n);
    if(q.path.empty() || q.path == "prefill") {
        const int rows_per = Net::kScoreRows;
        float* logits = net_->score_logits();
        for(int c0 = 0; c0 < n; c0 += config_.chunk) {
            const int cn = std::min(config_.chunk, n - c0);
            net_->prefill(ids, c0, cn, true);
            for(int r0 = 0; r0 < cn; r0 += rows_per) {
                const int rn = std::min(rows_per, cn - r0);
                // Row r (position c0 + r0 + r) predicts target index c0 + r0 + r + 1 - first.
                const int pos_first = c0 + r0, pos_last = c0 + r0 + rn - 1;
                const int lo = std::max(pos_first, int(first) - 1), hi = std::min(pos_last, n - 2);
                if(lo > hi) continue;
                net_->head_rows(lo - c0, hi - lo + 1, logits);
                const int t0 = lo + 1 - int(first);
                row_scores(logits, kVocab, kValidVocab, hi - lo + 1, d_t + t0, d_q + std::size_t(t0) * per, per,
                           d_tlp + t0, d_qlp + std::size_t(t0) * per, d_am + t0, s);
            }
        }
        FNL_CUDA(cudaStreamSynchronize(s));
    } else if(q.path == "decode") {
        for(int pos = 0; pos < n - 1; ++pos) {
            net_->decode(ids, pos);
            if(pos + 1 >= int(first)) {
                const int t0 = pos + 1 - int(first);
                row_scores(net_->logits(), kVocab, kValidVocab, 1, d_t + t0, d_q + std::size_t(t0) * per, per,
                           d_tlp + t0, d_qlp + std::size_t(t0) * per, d_am + t0, s);
            }
        }
    } else if(q.path == "verify") {
        // The multi-token passes that verify drafts, here fed the request's own tokens 4 at a time.
        for(int pos = 0; pos < n - 1; pos += kMaxTokens) {
            const int T = std::min(kMaxTokens, n - 1 - pos);
            net_->verify_pass(ids, pos, T);
            for(int r = 0; r < T; ++r) {
                if(pos + r + 1 < int(first)) continue;
                const int t0 = pos + r + 1 - int(first);
                row_scores(net_->logits() + std::size_t(r) * kVocab, kVocab, kValidVocab, 1, d_t + t0,
                           d_q + std::size_t(t0) * per, per, d_tlp + t0, d_qlp + std::size_t(t0) * per, d_am + t0, s);
            }
        }
    } else {
        throw std::invalid_argument("unknown scoring path " + q.path);
    }
    QueryScores out;
    out.target.resize(targets);
    out.queries.resize(std::size_t(targets) * per);
    out.argmax.resize(targets);
    FNL_CUDA(cudaMemcpyAsync(out.target.data(), d_tlp, std::size_t(targets) * 4, cudaMemcpyDeviceToHost, s));
    if(per)
        FNL_CUDA(cudaMemcpyAsync(out.queries.data(), d_qlp, std::size_t(targets) * per * 4, cudaMemcpyDeviceToHost, s));
    FNL_CUDA(cudaMemcpyAsync(out.argmax.data(), d_am, std::size_t(targets) * 4, cudaMemcpyDeviceToHost, s));
    FNL_CUDA(cudaStreamSynchronize(s));
    return out;
}

Config parse_config(const char* json) {
    Config c;
    const auto doc = json::Document::parse(json, "fnl config");
    const auto& r = doc.root();
    if(auto* v = r.find("model")) c.model = std::string(v->string());
    if(auto* v = r.find("capacity")) c.capacity = int(v->i64());
    if(auto* v = r.find("prefill_chunk")) c.chunk = int(v->i64());
    if(auto* v = r.find("expert_slots")) c.slots = std::size_t(v->u64());
    if(auto* v = r.find("vram_margin_gib")) c.vram_margin_gib = v->real();
    if(auto* v = r.find("temperature")) c.temperature = v->real();
    if(auto* v = r.find("top_p")) c.top_p = v->real();
    if(auto* v = r.find("top_k")) c.top_k = std::uint32_t(v->u64());
    if(auto* v = r.find("speculative")) c.speculative = v->boolean();
    if(c.model.empty()) throw std::invalid_argument("config needs model");
    return c;
}

struct EngineHandle {
    Config config;
    std::unique_ptr<Engine> engine;
    FnlModel* model = nullptr;
};
thread_local std::string last_error;
struct Cancelled {};
}  // namespace
}  // namespace fnl

extern "C" {
typedef int (*fnl_token_cb)(void* user, const int32_t* tokens, int32_t n);

const char* fnl_last_error() { return fnl::last_error.c_str(); }

void* fnl_open(const char* config_json) {
    using namespace fnl;
    try {
        auto h = std::make_unique<EngineHandle>();
        h->config = parse_config(config_json);
        crit::core::Profile profile;
        profile.context = std::uint32_t(h->config.capacity);
        profile.prefill_chunk = std::uint32_t(h->config.chunk);
        profile.host_limit = std::size_t(84) << 30;
        profile.device_limit = std::size_t(34) << 30;
        auto backend = std::make_shared<crit::core::CudaBackend>(0);
        EngineHandle* raw = h.get();
        h->engine = std::make_unique<crit::core::Engine>(profile, backend,
            [raw](const crit::core::Profile&, crit::core::Services services) {
                auto m = std::make_unique<FnlModel>(raw->config, services);
                raw->model = m.get();
                return m;
            });
        return h.release();
    } catch(const std::exception& e) {
        last_error = e.what();
        return nullptr;
    }
}

void fnl_close(void* handle) { delete static_cast<fnl::EngineHandle*>(handle); }

// Writes JSON facts about the loaded engine into buf.
int fnl_info(void* handle, char* buf, int len) {
    auto* h = static_cast<fnl::EngineHandle*>(handle);
    std::ostringstream o;
    auto& net = h->model->net();
    const bool spec = net.has_mtp() && h->config.speculative;
    o << "{\"capacity\":" << net.capacity() << ",\"prefill_chunk\":" << net.prefill_chunk()
      << ",\"speculative\":" << (spec ? "true" : "false") << ",\"draft_width\":" << (spec ? fnl::kDrafts : 0)
      << ",\"expert_slots\":" << net.expert_slots() << ",\"expert_lookups\":" << net.expert_lookups()
      << ",\"expert_misses\":" << net.expert_misses() << ",\"prefill_fetched\":" << net.prefill_fetched() << "}";
    const std::string s = o.str();
    std::snprintf(buf, std::size_t(len), "%s", s.c_str());
    return int(s.size());
}

// Generates up to max_new tokens. stops: 0 disables stop tokens, 1 uses the model's. temperature < 0 keeps the
// default sampling; 0 is greedy. Returns the number of tokens, -1 on error (fnl_last_error), -2 when cancelled.
int fnl_generate(void* handle, const int32_t* prompt, int n, int max_new, int stops, double temperature, double top_p,
                 int top_k, unsigned long long seed, int has_seed, fnl_token_cb cb, void* user, int32_t* out,
                 double* timing) {
    using namespace fnl;
    auto* h = static_cast<EngineHandle*>(handle);
    try {
        crit::core::RequestOptions o;
        if(!stops) o.stop_tokens = std::vector<crit::core::Token>{};
        if(temperature >= 0) o.sampling.temperature = temperature;
        if(top_p > 0) o.sampling.top_p = top_p;
        if(top_k >= 0) o.sampling.top_k = std::uint32_t(top_k);
        if(has_seed) o.sampling.seed = seed;
        auto handle2 = h->engine->submit(std::vector<crit::core::Token>(prompt, prompt + n), std::uint32_t(max_new), o);
        int count = 0;
        crit::core::Result r = handle2.wait([&](crit::core::Token t) {
            out[count++] = t;
            if(cb && cb(user, &t, 1)) throw Cancelled{};
        });
        if(timing) {
            timing[0] = double(r.timing.prefill_ns) / 1e6;
            timing[1] = double(r.timing.first_token_ns) / 1e6;
            timing[2] = double(r.timing.decode_ns) / 1e6;
            timing[3] = double(r.timing.total_ns) / 1e6;
            timing[4] = double(r.timing.decode_steps);
            timing[5] = double(int(r.finish));
        }
        return int(r.tokens.size());
    } catch(const Cancelled&) {
        return -2;
    } catch(const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}

// Device milliseconds of the last generate call's target passes (out[0]) and drafter steps (out[1]).
void fnl_pass_timing(void* handle, double* out) {
    auto* h = static_cast<fnl::EngineHandle*>(handle);
    const auto t = h->model->net().pass_timing();
    out[0] = t.first;
    out[1] = t.second;
}

// Drafts kept in each speculative round of the last generate call; returns the number of rounds.
int fnl_accepted(void* handle, int32_t* out, int max) {
    auto* h = static_cast<fnl::EngineHandle*>(handle);
    const auto& a = h->model->accepted_;
    for(int i = 0; i < int(a.size()) && i < max; ++i) out[i] = a[i];
    return int(a.size());
}

// path: 0 prefill, 1 decode, 2 verify (4-token passes). queries: targets x per ids (may be null with per 0). Returns 0 or -1.
int fnl_score(void* handle, const int32_t* ids, int n, int first, int path, const int32_t* queries, int per,
              float* target_lp, float* query_lp, int32_t* argmax) {
    using namespace fnl;
    auto* h = static_cast<EngineHandle*>(handle);
    try {
        crit::core::ScoreQueries q;
        q.per_target = std::uint32_t(per);
        if(per) q.ids.assign(queries, queries + std::size_t(n - first) * per);
        q.path = path == 2 ? "verify" : path == 1 ? "decode" : "prefill";
        auto hd = h->engine->submit_score(std::vector<crit::core::Token>(ids, ids + n), std::uint32_t(first), q);
        crit::core::Result r = hd.wait();
        const int targets = n - first;
        std::memcpy(target_lp, r.scores.data(), std::size_t(targets) * 4);
        if(per) std::memcpy(query_lp, r.query_scores.data(), std::size_t(targets) * per * 4);
        if(argmax && !r.argmax.empty()) std::memcpy(argmax, r.argmax.data(), std::size_t(targets) * 4);
        return 0;
    } catch(const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}
}
