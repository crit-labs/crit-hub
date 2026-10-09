#include "engine.h"
#include "forge/device_round.h"
#include <stdexcept>

namespace q38 {
namespace core = crit::core;
namespace {
const std::vector<core::Token> kStops = {248046, 248044}; // <|im_end|>, <|endoftext|>

class Session final : public core::Session {
    q38::Model& m_;
    RequestStats& stats_;
    std::vector<core::Token> prompt_;
    std::uint32_t consumed_ = 0;
    bool decoding_ = false;
    std::chrono::steady_clock::time_point t0_{};
public:
    Session(q38::Model& m, RequestStats& stats, std::span<const core::Token> prompt)
        : m_(m), stats_(stats), prompt_(prompt.begin(), prompt.end()) {}
    core::Progress advance(core::WorkBudget budget) override {
        decoding_ = budget.prefill_complete;
        const auto P = std::uint32_t(prompt_.size());
        if(!decoding_) {
            if(consumed_ == 0) t0_ = std::chrono::steady_clock::now();
            const auto count = std::min<std::uint32_t>(budget.prefill_tokens, P - consumed_);
            const auto end = std::min<std::uint32_t>(consumed_ + count, P - 1);
            const auto chunk = std::uint32_t(m_.config().prefill_chunk);
            for(auto i = consumed_; i < end; i += chunk)
                m_.prefill(prompt_.data() + i, int(std::min(chunk, end - i)), int(i));
            consumed_ += count;
            const bool done = consumed_ == P;
            if(done) {
                m_.finish_prefill(int(P), prompt_.back());
                m_.round().start(P - 1, prompt_.back()); // waits for its staged copy: prefill is settled
                stats_.prefill_end = std::chrono::steady_clock::now();
                stats_.prefill_ms = std::chrono::duration<double, std::milli>(stats_.prefill_end - t0_).count();
                stats_.prefill_tokens = P - 1;
            }
            return {count, done, {}, nullptr};
        }
        auto* ready = m_.round().launch(budget.output_tokens);
        return {0, true, {}, ready, true};
    }
    std::span<const core::Token> ready_tokens() override {
        const auto t = m_.round().tokens();
        ++stats_.rounds;
        stats_.emitted += t.size();
        return t;
    }
    std::span<const core::DeviceTiming> device_timings() override {
        return decoding_ ? m_.round().timings() : std::span<const core::DeviceTiming>{};
    }
    void commit(std::uint32_t accepted, bool) override { if(decoding_) m_.round().commit(accepted); }
    void close(core::Finish) noexcept override {
        try { if(m_.round().pending()) m_.round().abandon(); } catch(...) {}
    }
};
}

CoreModel::CoreModel(std::shared_ptr<core::CudaBackend> backend, const Config& cfg)
    : backend_(std::move(backend)), model_(std::make_unique<q38::Model>(backend_, cfg)) {}

core::ModelInfo CoreModel::info() const {
    core::ModelInfo i;
    i.implementation = "qwen3.8-27b-nvfp4-cleanroom";
    i.vocabulary = kVocab;
    i.context = std::uint32_t(model_->config().context);
    i.prefill_chunk = std::uint32_t(model_->config().prefill_chunk);
    i.stop_tokens = kStops;
    i.scoring = true;
    return i;
}
std::unique_ptr<core::Session> CoreModel::open(std::span<const core::Token> prompt) {
    return open_request(prompt, core::RequestOptions{});
}
std::unique_ptr<core::Session> CoreModel::open_request(std::span<const core::Token> prompt, const core::RequestOptions& options) {
    if(prompt.empty()) throw std::invalid_argument("A nonempty prompt is required");
    for(auto t : prompt) if(t < 0 || t >= kVocab) throw std::invalid_argument("Prompt token outside vocabulary");
    if(prompt.size() >= std::size_t(model_->config().context)) throw std::length_error("Prompt exceeds context");
    if(model_->round().pending()) model_->round().abandon();
    model_->reset_request();
    const auto& stops = options.stop_tokens ? *options.stop_tokens : kStops;
    model_->round().set_stops(stops);
    stats = {};
    return std::make_unique<Session>(*model_, stats, prompt);
}
std::vector<core::Token> CoreModel::tokenize(std::string_view) {
    throw std::logic_error("Text is tokenized by the engine's Python frontend");
}
std::string CoreModel::detokenize(std::span<const core::Token>) {
    throw std::logic_error("Text is decoded by the engine's Python frontend");
}
std::vector<float> CoreModel::score(std::span<const core::Token> tokens, std::uint32_t first_target) {
    if(model_->round().pending()) model_->round().abandon();
    return model_->score(std::vector<int>(tokens.begin(), tokens.end()), int(first_target));
}

std::unique_ptr<Runtime> create_runtime(const Config& cfg) {
    auto rt = std::make_unique<Runtime>();
    rt->backend = std::make_shared<core::CudaBackend>(0, core::SyncSchedule::spin);
    core::Profile profile;
    profile.context = std::uint32_t(cfg.context);
    profile.prefill_chunk = std::uint32_t(cfg.prefill_chunk);
    profile.host_limit = std::size_t(8) << 30;
    profile.device_limit = std::size_t(31) << 30;
    auto* slot = &rt->core;
    auto backend = rt->backend;
    rt->engine = std::make_unique<core::Engine>(profile, rt->backend, [slot, backend, cfg](const core::Profile&, core::Services) {
        auto m = std::make_unique<CoreModel>(backend, cfg);
        *slot = m.get();
        return m;
    });
    return rt;
}
}
