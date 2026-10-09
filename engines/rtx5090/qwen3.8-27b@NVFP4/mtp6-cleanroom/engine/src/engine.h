#pragma once
// crit Core adapter: Model/Session over q38::Model with the CUDA backend.
#include "model.h"
#include <crit/core.h>
#include <chrono>
#include <memory>

namespace q38 {
struct RequestStats {
    double prefill_ms = 0;           // prompt materialization, synchronized
    std::uint32_t prefill_tokens = 0;
    std::uint64_t rounds = 0, emitted = 0;
    std::chrono::steady_clock::time_point prefill_end{};
};

class CoreModel final : public crit::core::Model {
public:
    CoreModel(std::shared_ptr<crit::core::CudaBackend>, const Config&);
    crit::core::ModelInfo info() const override;
    std::unique_ptr<crit::core::Session> open(std::span<const crit::core::Token>) override;
    std::unique_ptr<crit::core::Session> open_request(std::span<const crit::core::Token>, const crit::core::RequestOptions&) override;
    std::vector<crit::core::Token> tokenize(std::string_view) override;
    std::string detokenize(std::span<const crit::core::Token>) override;
    std::vector<float> score(std::span<const crit::core::Token>, std::uint32_t first_target) override;
    q38::Model& model() { return *model_; }
    RequestStats stats;
private:
    std::shared_ptr<crit::core::CudaBackend> backend_;
    std::unique_ptr<q38::Model> model_;
};

struct Runtime {
    std::shared_ptr<crit::core::CudaBackend> backend;
    CoreModel* core = nullptr; // owned by engine
    std::unique_ptr<crit::core::Engine> engine;
};
std::unique_ptr<Runtime> create_runtime(const Config&);
}
