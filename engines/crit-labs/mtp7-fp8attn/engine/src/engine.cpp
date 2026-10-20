// crit Core integration: Model / Session over the Net, plus the C API used by the Python adapters.
#include <crit/core.h>
#include <crit/cuda.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "net.h"

using namespace kestrel;
namespace core = crit::core;

namespace {

constexpr int kStopIm = 248046, kStopEos = 248044;

// Per-request extension: plain (non-speculative) decoding and a forced continuation for replay.
struct RequestExt final : core::RequestContext {
    bool speculative = true;
    std::vector<int> forced;
    std::size_t host_bytes() const noexcept override { return sizeof(*this) + forced.size() * sizeof(int); }
};

struct Adaptive {
    bool on = false;
    double cost_ratio = 0.07;
};

struct Stats {
    int rounds = 0;
    std::vector<int> accepted;  // drafts accepted per round
    double prefill_ms = 0;
};

class KSession final : public core::Session {
public:
    KSession(Net& net, std::span<const core::Token> prompt, const core::RequestOptions& options, Stats& stats, Adaptive adaptive)
        : net_(net), prompt_(prompt), stats_(stats), adaptive_(adaptive) {
        net_.reset();
        // Device stops: the request's effective stop list.
        std::vector<int> stops;
        if(options.stop_tokens) stops.assign(options.stop_tokens->begin(), options.stop_tokens->end());
        else stops = {kStopIm, kStopEos};
        net_.set_stops(stops);
        const auto& sm = options.sampling;
        const float temperature = static_cast<float>(sm.temperature.value_or(1.0));
        net_.set_sampling(temperature, static_cast<float>(sm.top_p.value_or(0.95)), static_cast<int>(sm.top_k.value_or(20)),
                          sm.seed.value_or(0x5eed));
        if(auto* ext = dynamic_cast<const RequestExt*>(options.context.get())) {
            speculative_ = ext->speculative;
            if(!ext->forced.empty()) {
                replay_ = true;
                forced_ = ext->forced;
                net_.set_forced(forced_.data() + 1, static_cast<int>(forced_.size()) - 1);
            }
        }
        stats_ = Stats{};
        t0_ = std::chrono::steady_clock::now();
    }

    core::Progress advance(core::WorkBudget budget) override {
        if(!prefilled_) {
            const std::size_t left = prompt_.size() - frontier_;
            const std::size_t n = std::min<std::size_t>(budget.prefill_tokens ? budget.prefill_tokens : left, left);
            const bool last = frontier_ + n == prompt_.size();
            net_.prefill(prompt_.data() + frontier_, static_cast<int>(n), static_cast<int>(frontier_), last);
            pending_frontier_ = frontier_ + n;
            if(!last) return {static_cast<std::uint32_t>(n), false, {}, nullptr};
            first_ = net_.first_token();
            if(replay_) {
                first_ = forced_[0];
                net_.force_first(first_);
            }
            stats_.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
            emitted_[0] = first_;
            last_round_ = 1;
            prefill_step_ = true;
            return {static_cast<std::uint32_t>(n), true, std::span<const core::Token>(emitted_, 1), nullptr};
        }
        prefill_step_ = false;
        int k = speculative_ ? net_.config().draft : 0;
        if(speculative_ && adaptive_.on && k > 1) {
            // Expected emitted tokens 1 + p + ... + p^K over the round cost 1 + K * ratio.
            const double p = (acc_num_ + 4.0 * 0.8) / (acc_den_ + 4.0);  // prior: 0.8 over 4 trials
            double best = 0.0, sum = 1.0, term = 1.0;
            int arg = 1;
            for(int kk = 1; kk <= net_.config().draft; ++kk) {
                term *= p;
                sum += term;
                const double rate = sum / (1.0 + kk * adaptive_.cost_ratio);
                if(rate > best) { best = rate; arg = kk; }
            }
            k = arg;
        }
        round_k_ = k;
        net_.launch_round(static_cast<int>(budget.output_tokens), replay_, k);
        core::Progress p;
        p.prefill_complete = true;
        p.readiness = &net_.round_event();
        p.deferred_tokens = true;
        return p;
    }

    std::span<const core::Token> ready_tokens() override {
        const int* r = net_.round_result();
        last_round_ = r[0];
        for(int i = 0; i < r[0]; ++i) emitted_[i] = r[1 + i];
        stats_.rounds += 1;
        const int acc = std::max(r[0] - 1, 0);
        stats_.accepted.push_back(acc);
        if(round_k_ > 0) {
            // Decayed counts of accepted drafts over attempted drafts (a rejection ends the attempts of a round).
            acc_num_ = 0.9 * acc_num_ + acc;
            acc_den_ = 0.9 * acc_den_ + std::min(acc + 1, round_k_);
        }
        return std::span<const core::Token>(emitted_, static_cast<std::size_t>(r[0]));
    }

    void commit(std::uint32_t accepted, bool terminal) override {
        if(!prefilled_) {
            frontier_ = pending_frontier_;
            if(frontier_ == prompt_.size()) prefilled_ = true;
            return;
        }
        if(!prefill_step_ && static_cast<int>(accepted) < last_round_ && !terminal) net_.truncate_round(static_cast<int>(accepted));
    }

    void close(core::Finish) noexcept override { prompt_ = {}; }

private:
    Net& net_;
    std::span<const core::Token> prompt_;
    Stats& stats_;
    std::size_t frontier_ = 0, pending_frontier_ = 0;
    bool prefilled_ = false, prefill_step_ = false, replay_ = false, speculative_ = true;
    std::vector<int> forced_;
    int first_ = 0, last_round_ = 0, round_k_ = 0;
    Adaptive adaptive_;
    double acc_num_ = 0.0, acc_den_ = 0.0;
    core::Token emitted_[kMaxCols + 2] = {};
    std::chrono::steady_clock::time_point t0_;
};

class KModel final : public core::Model {
public:
    KModel(std::shared_ptr<core::CudaBackend> backend, const std::string& dir, const Config& cfg, Stats& stats)
        : net_(std::move(backend), dir, cfg), stats_(stats) {}
    Adaptive adaptive;
    Net& net() { return net_; }

    core::ModelInfo info() const override {
        core::ModelInfo m;
        m.implementation = "kestrel";
        m.vocabulary = kVocab;
        m.valid_vocabulary = kVocabValid;
        m.context = static_cast<std::uint32_t>(net_.config().context);
        m.prefill_chunk = static_cast<std::uint32_t>(net_.config().prefill_chunk);
        m.stop_tokens = {kStopIm, kStopEos};
        m.scoring = true;
        m.query_scoring = true;
        m.score_paths = {"prefill", "decode", "verify"};
        m.sampling.temperature = m.sampling.top_p = m.sampling.top_k = m.sampling.seed = true;
        m.request_context = true;
        return m;
    }
    std::unique_ptr<core::Session> open(std::span<const core::Token> prompt) override {
        return open_request(prompt, core::RequestOptions{});
    }
    std::unique_ptr<core::Session> open_request(std::span<const core::Token> prompt, const core::RequestOptions& o) override {
        if(prompt.empty()) throw std::invalid_argument("kestrel: empty prompt");
        return std::make_unique<KSession>(net_, prompt, o, stats_, adaptive);
    }
    std::vector<float> score(std::span<const core::Token> t, std::uint32_t first) override {
        core::ScoreQueries q;
        return score_queries(t, first, q, {}).target;
    }
    core::QueryScores score_queries(std::span<const core::Token> t, std::uint32_t first, const core::ScoreQueries& q,
                                    const core::RequestOptions&) override {
        int path = 0;
        if(q.path == "decode") path = 1;
        else if(q.path == "verify") path = 2;
        else if(!q.path.empty() && q.path != "prefill") throw std::invalid_argument("kestrel: unknown score path " + q.path);
        const int n = static_cast<int>(t.size()), targets = n - static_cast<int>(first);
        core::QueryScores r;
        r.target.resize(targets);
        r.queries.resize(static_cast<std::size_t>(targets) * q.per_target);
        r.argmax.resize(targets);
        std::vector<int> ids(t.begin(), t.end());
        std::vector<int> qs(q.ids.begin(), q.ids.end());
        net_.score(ids.data(), n, static_cast<int>(first), path, qs.data(), static_cast<int>(q.per_target), r.target.data(),
                   r.queries.data(), reinterpret_cast<int*>(r.argmax.data()));
        return r;
    }

private:
    Net net_;
    Stats& stats_;
};

struct Handle {
    std::shared_ptr<core::CudaBackend> backend;
    std::unique_ptr<core::Engine> engine;
    KModel* model = nullptr;
    Stats stats;
    std::string error;
    std::string description;
};

thread_local std::string g_error;
std::string g_gemm_table;

}  // namespace

extern "C" {

const char* ks_error() { return g_error.c_str(); }

// config: context, prefill_chunk, draft, attn_blocks, splits (0 = default).
void* ks_create(const char* model_dir, int context, int prefill_chunk, int draft, int attn_blocks, int splits) {
    try {
        auto h = std::make_unique<Handle>();
        h->backend = std::make_shared<core::CudaBackend>(0);
        Config cfg;
        cfg.context = context;
        if(prefill_chunk > 0) cfg.prefill_chunk = prefill_chunk;
        if(draft >= 0) cfg.draft = draft;
        if(attn_blocks > 0) cfg.attn_blocks = attn_blocks;
        if(splits > 0) cfg.splits = splits;
        cfg.gemm_table = g_gemm_table;
        core::Profile profile;
        profile.context = static_cast<std::uint32_t>(context);
        profile.prefill_chunk = static_cast<std::uint32_t>(cfg.prefill_chunk);
        profile.host_limit = std::size_t(80) << 30;
        profile.device_limit = std::size_t(40) << 30;
        std::string dir(model_dir);
        Handle* raw = h.get();
        h->engine = std::make_unique<core::Engine>(profile, h->backend,
            [raw, dir, cfg](const core::Profile&, core::Services) {
                auto m = std::make_unique<KModel>(raw->backend, dir, cfg, raw->stats);
                raw->model = m.get();
                return m;
            });
        h->description = h->model->net().describe();
        return h.release();
    } catch(const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void ks_close(void* p) {
    try {
        delete static_cast<Handle*>(p);
    } catch(...) {
    }
}

void ks_set_adaptive(void* p, int on, double cost_ratio) {
    auto* h = static_cast<Handle*>(p);
    h->model->adaptive.on = on != 0;
    if(cost_ratio > 0) h->model->adaptive.cost_ratio = cost_ratio;
}

// Config extras before ks_create: path of the recorded cuBLASLt table.
void ks_set_gemm_table(const char* path) { g_gemm_table = path ? path : ""; }

const char* ks_describe(void* p) { return static_cast<Handle*>(p)->description.c_str(); }
long long ks_bytes_per_pass(void* p) { return static_cast<long long>(static_cast<Handle*>(p)->model->net().bytes_per_pass()); }
long long ks_draft_bytes(void* p) { return static_cast<long long>(static_cast<Handle*>(p)->model->net().draft_bytes_per_step()); }

typedef int (*token_cb)(void* user, const int* tokens, int n);  // nonzero cancels
struct Cancelled {};

// Generation through Core. temperature < 0 = greedy with stops disabled (Eval); otherwise chat sampling with the
// model's stop tokens. forced (n_forced > 0) replays a continuation. Returns tokens generated or -1.
int ks_generate(void* p, const int* prompt, int n_prompt, int max_new, float temperature, float top_p, int top_k,
                unsigned long long seed, int speculative, const int* forced, int n_forced, const int* stops, int n_stops,
                token_cb cb, void* user, int* out, int out_cap) {
    try {
        auto* h = static_cast<Handle*>(p);
        std::vector<core::Token> pr(prompt, prompt + n_prompt);
        core::RequestOptions o;
        if(temperature < 0.f) {
            o.sampling.temperature = 0.0;
            o.stop_tokens = std::vector<core::Token>{};
        } else {
            o.sampling.temperature = temperature;
            o.sampling.top_p = top_p;
            o.sampling.top_k = static_cast<std::uint32_t>(top_k);
            o.sampling.seed = seed;
            if(n_stops > 0) o.stop_tokens = std::vector<core::Token>(stops, stops + n_stops);
        }
        auto ext = std::make_shared<RequestExt>();
        ext->speculative = speculative != 0;
        if(n_forced > 0) ext->forced.assign(forced, forced + n_forced);
        o.context = ext;
        auto handle = h->engine->submit(std::move(pr), static_cast<std::uint32_t>(max_new), std::move(o));
        int count = 0;
        try {
            auto result = handle.wait([&](core::Token t) {
                if(count < out_cap) out[count] = t;
                ++count;
                if(cb && cb(user, &t, 1)) throw Cancelled{};
            });
            if(result.finish == core::Finish::failed) throw std::runtime_error("generation failed");
            return static_cast<int>(result.tokens.size());
        } catch(const Cancelled&) {
            return std::min(count, out_cap);
        }
    } catch(const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

// Stats of the last generation: rounds and accepted drafts per round (up to cap).
int ks_stats(void* p, int* accepted, int cap, double* prefill_ms) {
    auto* h = static_cast<Handle*>(p);
    const int n = std::min<int>(cap, static_cast<int>(h->stats.accepted.size()));
    for(int i = 0; i < n; ++i) accepted[i] = h->stats.accepted[i];
    if(prefill_ms) *prefill_ms = h->stats.prefill_ms;
    return h->stats.rounds;
}

int ks_score(void* p, const int* ids, int n, int first, const char* path, const int* queries, int nq, float* target_lp,
             float* query_lp, int* argmax_out) {
    try {
        auto* h = static_cast<Handle*>(p);
        core::ScoreQueries q;
        q.ids.assign(queries, queries + static_cast<std::size_t>(n - first) * nq);
        q.per_target = static_cast<std::uint32_t>(nq);
        q.path = path;
        auto handle = h->engine->submit_score(std::vector<core::Token>(ids, ids + n), static_cast<std::uint32_t>(first), q);
        auto r = handle.wait();
        const int targets = n - first;
        for(int i = 0; i < targets; ++i) target_lp[i] = r.scores[i];
        for(std::size_t i = 0; i < r.query_scores.size(); ++i) query_lp[i] = r.query_scores[i];
        for(std::size_t i = 0; i < r.argmax.size(); ++i) argmax_out[i] = r.argmax[i];
        return 0;
    } catch(const std::exception& e) {
        g_error = e.what();
        return 1;
    }
}

}
