// crit Core integration of the nmh network (Model / Session) exported through crit/native_engine.h.
#include "net.h"
#include <crit/core.h>
#include <crit/cuda.h>
#include <crit/json.h>
#include <crit/native_engine.h>
#include <crit/sampling.h>
#include <cstring>
#include <sstream>

namespace nmh {
namespace {
using namespace crit::core;

const std::vector<Token> kStops = {2, 11};   // generation_config.json eos_token_id: </s>, <|im_end|>

struct Config {
    std::string model;
    int capacity = 1048576;
    int chunk = 32768;
    int moe_sub = 8192;   // prompt MoE sub-chunk (W4A4 grouped GEMMs want >= ~176 rows per expert)
    std::size_t slots = 0;
    double vram_margin_gib = 1.5;
    bool lend_kv = true;
    int attn_blocks = 64;
    int load_threads = 12;
    bool keep_host_weights = false;
    bool speculative = true;   // greedy requests decode in MTP-drafted rounds of drafts + 1 verified tokens
    int drafts = 3;
    bool prefill_fp8 = true;
    bool prompt_fp4 = true;
    bool prompt_attn_fp8 = true;
    int cpu_prompt_max = 384;
    double cpu_prompt_share = 0.35;
    double dt_min = 0.0;   // config.json time_step_limit (0, inf); recon.md
    double cpu_share = 0.6;   // share of a decode pass's expert misses computed by the CPU
    int cpu_min = 2, cpu_threads = 6;
    // Sampling defaults (generation_config.json, recon frozen block)
    double temperature = 1.0, top_p = 0.95;
    std::uint32_t top_k = 0;
};

class NmhModel;

class NmhSession final : public Session {
public:
    NmhSession(NmhModel& m, std::span<const Token> prompt, const RequestOptions& options);
    Progress advance(WorkBudget budget) override;
    std::span<const Token> ready_tokens() override;
    void commit(std::uint32_t accepted, bool terminal) override;
    void close(Finish) noexcept override {}
private:
    NmhModel& m_;
    std::vector<int> history_;
    std::size_t prompt_ = 0, frontier_ = 0, pending_frontier_ = 0;
    bool decoding_ = false, first_ready_ = false, started_ = false;
    Token pending_ = 0;
    SamplingParams params_;
    TokenHistory th_;
    bool greedy_ = true;
    // speculative rounds
    bool spec_ = false, spec_started_ = false, round_pending_ = false;
    int drafts_[kDrafts] = {0};
    std::vector<Token> round_;      // tokens of the last round (kept drafts, then the bonus token)
    std::uint32_t out_limit_ = 1;
    std::vector<int> forced_;       // replayed continuation (empty: none)
    std::size_t emitted_ = 0;       // output tokens committed so far
};

class NmhModel final : public Model {
public:
    NmhModel(const Config& c, Services s)
        : config_(c), backend_(dynamic_cast<CudaBackend&>(s.backend)), sampler_(kVocab) {
        Options o;
        o.model = c.model;
        o.capacity = c.capacity;
        o.prefill_chunk = c.chunk;
        o.moe_sub = c.moe_sub;
        o.expert_slots = c.slots;
        o.vram_margin = std::size_t(c.vram_margin_gib * double(std::size_t(1) << 30));
        o.lend_kv = c.lend_kv;
        o.attn_blocks = c.attn_blocks;
        o.load_threads = c.load_threads;
        o.keep_host_weights = c.keep_host_weights;
        o.mtp = c.speculative;
        o.cpu_share = float(c.cpu_share);
        o.cpu_min = c.cpu_min;
        o.cpu_threads = c.cpu_threads;
        o.drafts = c.drafts;
        o.prefill_fp8 = c.prefill_fp8;
        o.prompt_fp4 = c.prompt_fp4;
        o.prompt_attn_fp8 = c.prompt_attn_fp8;
        o.cpu_prompt_max = c.cpu_prompt_max;
        o.cpu_prompt_share = float(c.cpu_prompt_share);
        o.dt_min = float(c.dt_min);
        net_ = std::make_unique<Net>(o, backend_);
        host_ = backend_.allocate(Memory::host, kVocab * 4 + 4096);
        token_host_ = static_cast<int*>(host_.data());
        logits_host_ = reinterpret_cast<float*>(static_cast<uint8_t*>(host_.data()) + 4096);
        ready_ = backend_.event();
    }
    ModelInfo info() const override {
        ModelInfo i;
        i.implementation = "nmh (nemotron_h Nemotron-3.5-Super NVFP4)";
        i.vocabulary = kVocab;
        i.valid_vocabulary = kVocab;
        i.context = std::uint32_t(config_.capacity);
        i.prefill_chunk = std::uint32_t(config_.chunk);
        i.stop_tokens = kStops;
        i.scoring = true;
        i.query_scoring = true;
        i.score_paths = {"prefill", "decode", "prefill_decode"};
        i.sampling.temperature = i.sampling.top_p = i.sampling.top_k = i.sampling.seed = true;
        i.request_context = true;   // ReplayContinuation and PlainDecoding
        return i;
    }
    std::unique_ptr<Session> open(std::span<const Token> prompt) override { return open_request(prompt, {}); }
    std::unique_ptr<Session> open_request(std::span<const Token> prompt, const RequestOptions& options) override {
        if(prompt.empty()) throw std::invalid_argument("empty prompt");
        if(prompt.size() >= std::size_t(config_.capacity)) throw std::invalid_argument("prompt exceeds the context");
        net_->reset();
        accepted_.clear();
        return std::make_unique<NmhSession>(*this, prompt, options);
    }
    QueryScores score_queries(std::span<const Token> ids, std::uint32_t first, const ScoreQueries& q,
                              const RequestOptions&) override;
    std::vector<float> score(std::span<const Token> ids, std::uint32_t first) override {
        ScoreQueries q;
        return score_queries(ids, first, q, {}).target;
    }
    Net& net() { return *net_; }
    Sampler& sampler() { return sampler_; }
    const Config& config() const { return config_; }
    int* token_host_ = nullptr;
    float* logits_host_ = nullptr;
    CudaEvent ready_;
    std::vector<int> accepted_;   // drafts kept per target pass of the last request (0 for one-token passes)
private:
    Config config_;
    CudaBackend& backend_;
    std::unique_ptr<Net> net_;
    CudaBuffer host_;
    Sampler sampler_;
};

NmhSession::NmhSession(NmhModel& m, std::span<const Token> prompt, const RequestOptions& options)
    : m_(m), history_(prompt.begin(), prompt.end()), prompt_(prompt.size()), th_(prompt) {
    SamplingParams defaults;
    defaults.temperature = m.config().temperature;
    defaults.top_p = m.config().top_p;
    defaults.top_k = m.config().top_k;
    defaults.seed = 0x5eedULL;
    params_ = resolve_sampling(options.sampling, defaults, kVocab);
    greedy_ = params_.temperature == 0.0;
    const bool plain = dynamic_cast<const PlainDecoding*>(options.context.get()) != nullptr;
    if(auto* replay = dynamic_cast<const ReplayContinuation*>(options.context.get()))
        forced_.assign(replay->tokens.begin(), replay->tokens.end());
    spec_ = greedy_ && !plain && m.net().has_mtp() && m.config().speculative;
    if(m.net().has_mtp()) m.net().set_forced(spec_ ? forced_ : std::vector<int>{});
}

Progress NmhSession::advance(WorkBudget budget) {
    auto& net = m_.net();
    const auto s = net.stream();
    decoding_ = budget.prefill_complete;
    if(!started_) {
        started_ = true;
        // A round writes drafts + 1 positions past the last kept token; the drafter seeds at the prompt's last.
        const std::size_t width = std::size_t(net.drafts()) + 1;
        if(spec_ && prompt_ + width > std::size_t(net.capacity())) spec_ = false;
        const std::size_t extra = spec_ ? width + 1 : 1;
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
        return Progress{std::uint32_t(n), done, {}, &m_.ready_};
    }
    // Near the end of the context a round would write past it: the last tokens decode one per pass.
    if(spec_ && spec_started_ && history_.size() + std::size_t(net.drafts()) > std::size_t(net.capacity())) spec_ = false;
    if(spec_) {
        out_limit_ = std::max<std::uint32_t>(1, budget.output_tokens);
        if(!spec_started_) {
            spec_started_ = true;
            first_ready_ = false;
            round_pending_ = false;
            net.spec_begin(int(history_.size()) - 1);
        } else {
            round_pending_ = true;
            net.spec_round(int(history_.size()) - 1, history_.back(), drafts_);
        }
        m_.ready_.record(s);
        Progress p;
        p.prefill_complete = true;
        p.readiness = &m_.ready_;
        p.deferred_tokens = true;
        return p;
    }
    if(first_ready_) first_ready_ = false;   // the prompt pass already produced this position's logits
    else {
        net.decode(history_, int(history_.size()) - 1);
        if(spec_started_) m_.accepted_.push_back(0);   // one-token pass after the speculative rounds
    }
    if(greedy_) NMH_CUDA(cudaMemcpyAsync(m_.token_host_, net.next_token(), 4, cudaMemcpyDeviceToHost, s));
    else NMH_CUDA(cudaMemcpyAsync(m_.logits_host_, net.logits(), kVocab * 4, cudaMemcpyDeviceToHost, s));
    m_.ready_.record(s);
    Progress p;
    p.prefill_complete = true;
    p.readiness = &m_.ready_;
    p.deferred_tokens = true;
    return p;
}

std::span<const Token> NmhSession::ready_tokens() {
    if(spec_) {
        const Accept& a = m_.net().spec_result();
        round_.clear();
        if(round_pending_) {
            m_.accepted_.push_back(a.n - 1);
            for(int k = 0; k + 1 < a.n; ++k) round_.push_back(Token(drafts_[k]));
        }
        round_.push_back(Token(a.bonus));
        if(round_.size() > out_limit_) round_.resize(out_limit_);
        for(int k = 0; k < m_.net().drafts(); ++k) drafts_[k] = a.drafts[k];
        return {round_.data(), round_.size()};
    }
    if(!forced_.empty() && emitted_ < forced_.size()) {
        pending_ = Token(forced_[emitted_]);   // replay: the continuation's token, also past the rounds
    } else if(greedy_) {
        pending_ = *m_.token_host_;
    } else {
        th_.rollback();
        const Choice c = m_.sampler().sample(std::span<const float>(m_.logits_host_, kVocab), params_, th_);
        pending_ = c.token;
    }
    return {&pending_, 1};
}

void NmhSession::commit(std::uint32_t accepted, bool) {
    if(!decoding_) {
        frontier_ = pending_frontier_;
        return;
    }
    if(spec_) {
        for(std::uint32_t i = 0; i < accepted && i < round_.size(); ++i) history_.push_back(int(round_[i]));
        emitted_ += std::min<std::size_t>(accepted, round_.size());
        return;
    }
    if(accepted) {
        ++emitted_;
        history_.push_back(pending_);
        th_.stage({&pending_, 1});
        th_.commit(1);
    }
}

QueryScores NmhModel::score_queries(std::span<const Token> ids_span, std::uint32_t first, const ScoreQueries& q,
                                    const RequestOptions&) {
    const std::vector<int> ids(ids_span.begin(), ids_span.end());
    const int n = int(ids.size());
    if(first < 1 || int(first) > n) throw std::invalid_argument("first target");
    if(n > config_.capacity) throw std::invalid_argument("scoring past the context");
    const int targets = n - int(first);
    const int per = int(q.per_target);
    if(q.top_k) throw std::invalid_argument("top_k alternatives are not supported");
    const auto s = net_->stream();
    const std::size_t bytes = std::size_t(targets) * (4 + 4 + 4 + per * 8) + 4096;
    auto buf = backend_.allocate(Memory::device, bytes);
    auto* base = static_cast<uint8_t*>(buf.data());
    int* d_t = reinterpret_cast<int*>(base);
    int* d_q = d_t + targets;
    float* d_tlp = reinterpret_cast<float*>(d_q + std::size_t(targets) * per);
    float* d_qlp = d_tlp + targets;
    int* d_am = reinterpret_cast<int*>(d_qlp + std::size_t(targets) * per);
    NMH_CUDA(cudaMemcpyAsync(d_t, ids.data() + first, std::size_t(targets) * 4, cudaMemcpyHostToDevice, s));
    if(per) NMH_CUDA(cudaMemcpyAsync(d_q, q.ids.data(), std::size_t(targets) * per * 4, cudaMemcpyHostToDevice, s));
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
                const int pos_first = c0 + r0, pos_last = c0 + r0 + rn - 1;
                const int lo = std::max(pos_first, int(first) - 1), hi = std::min(pos_last, n - 2);
                if(lo > hi) continue;
                net_->head_rows(lo - c0, hi - lo + 1);
                const int t0 = lo + 1 - int(first);
                row_scores(logits, kVocab, kVocab, hi - lo + 1, d_t + t0, d_q + std::size_t(t0) * per, per, d_tlp + t0,
                           d_qlp + std::size_t(t0) * per, d_am + t0, s);
            }
        }
    } else if(q.path == "decode") {
        for(int pos = 0; pos < n - 1; ++pos) {
            net_->decode(ids, pos);
            if(pos + 1 >= int(first)) {
                const int t0 = pos + 1 - int(first);
                row_scores(net_->logits(), kVocab, kVocab, 1, d_t + t0, d_q + std::size_t(t0) * per, per, d_tlp + t0,
                           d_qlp + std::size_t(t0) * per, d_am + t0, s);
            }
        }
    } else if(q.path == "prefill_decode") {
        // ids[:first] prefilled as a request prompt; the target at first from the prefill's last position, each
        // later one from an ordinary one-token decode step fed the id before it.
        for(int c0 = 0; c0 < int(first); c0 += config_.chunk) net_->prefill(ids, c0, std::min(config_.chunk, int(first) - c0));
        row_scores(net_->logits(), kVocab, kVocab, 1, d_t, d_q, per, d_tlp, d_qlp, d_am, s);
        for(int pos = int(first); pos < n - 1; ++pos) {
            net_->decode(ids, pos);
            const int t0 = pos + 1 - int(first);
            row_scores(net_->logits(), kVocab, kVocab, 1, d_t + t0, d_q + std::size_t(t0) * per, per, d_tlp + t0,
                       d_qlp + std::size_t(t0) * per, d_am + t0, s);
        }
    } else {
        throw std::invalid_argument("unknown scoring path " + q.path);
    }
    QueryScores out;
    out.target.resize(targets);
    out.queries.resize(std::size_t(targets) * per);
    out.argmax.resize(targets);
    NMH_CUDA(cudaMemcpyAsync(out.target.data(), d_tlp, std::size_t(targets) * 4, cudaMemcpyDeviceToHost, s));
    if(per) NMH_CUDA(cudaMemcpyAsync(out.queries.data(), d_qlp, std::size_t(targets) * per * 4, cudaMemcpyDeviceToHost, s));
    NMH_CUDA(cudaMemcpyAsync(out.argmax.data(), d_am, std::size_t(targets) * 4, cudaMemcpyDeviceToHost, s));
    NMH_CUDA(cudaStreamSynchronize(s));
    return out;
}

Config parse_config(std::string_view json) {
    Config c;
    const auto doc = json::Document::parse(std::string(json), "nmh config");
    const auto& r = doc.root();
    if(auto* v = r.find("model")) c.model = std::string(v->string());
    if(auto* v = r.find("context_tokens")) c.capacity = int(v->i64());
    if(auto* v = r.find("prefill_chunk")) c.chunk = int(v->i64());
    if(auto* v = r.find("moe_sub")) c.moe_sub = int(v->i64());
    if(auto* v = r.find("expert_slots")) c.slots = std::size_t(v->u64());
    if(auto* v = r.find("vram_margin_gib")) c.vram_margin_gib = v->real();
    if(auto* v = r.find("lend_kv")) c.lend_kv = v->boolean();
    if(auto* v = r.find("attn_blocks")) c.attn_blocks = int(v->i64());
    if(auto* v = r.find("load_threads")) c.load_threads = int(v->i64());
    if(auto* v = r.find("keep_host_weights")) c.keep_host_weights = v->boolean();
    if(auto* v = r.find("speculative")) c.speculative = v->boolean();
    if(auto* v = r.find("cpu_share")) c.cpu_share = v->real();
    if(auto* v = r.find("cpu_min")) c.cpu_min = int(v->i64());
    if(auto* v = r.find("cpu_threads")) c.cpu_threads = int(v->i64());
    if(auto* v = r.find("drafts")) c.drafts = int(v->i64());
    if(auto* v = r.find("prefill_fp8")) c.prefill_fp8 = v->boolean();
    if(auto* v = r.find("prompt_fp4")) c.prompt_fp4 = v->boolean();
    if(auto* v = r.find("prompt_attn_fp8")) c.prompt_attn_fp8 = v->boolean();
    if(auto* v = r.find("cpu_prompt_max")) c.cpu_prompt_max = int(v->i64());
    if(auto* v = r.find("cpu_prompt_share")) c.cpu_prompt_share = v->real();
    if(auto* v = r.find("dt_min")) c.dt_min = v->real();
    if(c.drafts < 1 || c.drafts > kDrafts) throw std::invalid_argument("drafts must be 1..3");
    if(auto* v = r.find("temperature")) c.temperature = v->real();
    if(auto* v = r.find("top_p")) c.top_p = v->real();
    if(auto* v = r.find("top_k")) c.top_k = std::uint32_t(v->u64());
    if(c.model.empty()) throw std::invalid_argument("config needs model");
    return c;
}

// Model bytes read per decode pass (resident weights plus one token's routed experts), for Eval's floors.
std::size_t pass_bytes(const Net& net) {
    return net.dense_bytes_per_pass() + std::size_t(kMoeLayers) * kTopK * kSlotBytes;
}

NativeEngine open_engine(std::string_view configuration) {
    const Config config = parse_config(configuration);
    Profile profile;
    profile.context = std::uint32_t(config.capacity);
    profile.prefill_chunk = std::uint32_t(config.chunk);
    profile.host_limit = std::size_t(84) << 30;
    profile.device_limit = std::size_t(34) << 30;
    auto backend = std::make_shared<CudaBackend>(0);
    auto model = std::make_shared<NmhModel*>(nullptr);
    NativeEngine native;
    native.owner = backend;
    native.engine = std::make_unique<Engine>(profile, backend, [config, model](const Profile&, Services services) {
        auto m = std::make_unique<NmhModel>(config, services);
        *model = m.get();
        return m;
    });
    NmhModel* m = *model;
    native.info = [m] {
        auto& net = m->net();
        std::ostringstream o;
        const bool spec = net.has_mtp();
        o << "{\"name\":\"nmh\",\"speculative\":" << (spec ? "true" : "false")
          << ",\"replay\":" << (spec ? "true" : "false") << ",\"plain\":" << (spec ? "true" : "false");
        if(spec) o << ",\"draft_width\":" << net.drafts() << ",\"draft_bytes_per_step\":" << net.draft_bytes_per_step();
        o << ",\"bytes_per_pass\":" << pass_bytes(net)
          << ",\"kv_bytes_per_token\":" << kAttnLayers * 2 * (kKvDim + kKvHeads * 4)
          << ",\"expert_slots\":" << net.expert_slots() << ",\"own_slots\":" << net.own_slots()
          << ",\"expert_lookups\":" << net.expert_lookups() << ",\"expert_misses\":" << net.expert_misses()
          << ",\"prefill_fetched\":" << net.prefill_fetched() << "}";
        return o.str();
    };
    native.request = [m] {
        std::ostringstream o;
        const auto t = m->net().pass_timing();
        o << "{\"accepted\":[";
        for(std::size_t i = 0; i < m->accepted_.size(); ++i) o << (i ? "," : "") << m->accepted_[i];
        o << "],\"verify_ms\":" << t.first << ",\"draft_ms\":" << t.second
          << ",\"expert_lookups\":" << m->net().expert_lookups() << ",\"expert_misses\":" << m->net().expert_misses()
          << ",\"expert_cpu\":" << m->net().expert_cpu()
          << "}";
        return o.str();
    };
    return native;
}

}  // namespace
}  // namespace nmh

CRIT_EXPORT_ENGINE(nmh::open_engine)
