// Development entry points: drive a Net directly (no Core runtime) for kernel and model checks.
#include <chrono>
#include <cstdio>
#include <memory>

#include "net.h"

using namespace g4;

namespace {
struct Dev {
    std::shared_ptr<crit::core::Resources> res;
    std::shared_ptr<crit::core::CudaBackend> backend;
    std::unique_ptr<Net> net;
};
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

extern "C" {

void* kd_open(const char* model_dir, const char* draft_dir, int context, int draft, int chunk) {
    try {
        auto d = std::make_unique<Dev>();
        d->res = std::make_shared<crit::core::Resources>(std::size_t(64) << 30, std::size_t(40) << 30);
        d->backend = std::make_shared<crit::core::CudaBackend>(0);
        d->backend->initialize(*d->res);
        Config c;
        c.context = context;
        c.draft = draft;
        c.prefill_chunk = chunk;
        d->net = std::make_unique<Net>(d->backend, model_dir, draft_dir ? draft_dir : "", c);
        std::fprintf(stderr, "%s\n", d->net->describe().c_str());
        return d.release();
    } catch(const std::exception& e) {
        std::fprintf(stderr, "kd_open: %s\n", e.what());
        return nullptr;
    }
}

void kd_close(void* h) { delete static_cast<Dev*>(h); }

// Greedy generation: prompt -> n tokens (out). times[0] = prefill ms (to first token), times[1] = decode ms.
// rounds_out[0] = rounds after the first token; when rounds_out[1] = cap > 0, rounds_out[2 + i] = tokens of round i. Returns tokens produced or -1.
int kd_generate(void* h, const int* prompt, int T, int n, int* out, double* times, int* rounds_out, int replay,
                const int* forced) {
    try {
        auto* d = static_cast<Dev*>(h);
        Net& net = *d->net;
        net.reset();
        if(replay) net.set_forced(forced + 1, n - 1);
        KCHECK(cudaStreamSynchronize(net.stream()));
        const double t0 = now_ms();
        net.prefill(prompt, T, 0, true);
        int first = net.first_token();
        if(replay) {
            first = forced[0];
            net.force_first(first);
        }
        const double t1 = now_ms();
        int produced = 0;
        out[produced++] = first;
        int rounds = 0;
        if(replay) {
            // The replay continuation starts with the first token: re-seed the device's last token.
            (void)0;
        }
        while(produced < n) {
            net.launch_round(n - produced, replay != 0);
            net.round_event().wait();
            const int* r = net.round_result();
            for(int i = 0; i < r[0] && produced < n; ++i) out[produced++] = r[1 + i];
            if(rounds_out[1] > 0 && rounds + 1 < rounds_out[1]) rounds_out[2 + rounds] = r[0];
            ++rounds;
        }
        const double t2 = now_ms();
        times[0] = t1 - t0;
        times[1] = t2 - t1;
        rounds_out[0] = rounds;
        return produced;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "kd_generate: %s\n", e.what());
        return -1;
    }
}

int kd_score(void* h, const int* ids, int n, int first, int path, const int* queries, int nq, float* target_lp,
             float* query_lp, int* argmax_out) {
    try {
        static_cast<Dev*>(h)->net->score(ids, n, first, path, queries, nq, target_lp, query_lp, argmax_out);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "kd_score: %s\n", e.what());
        return 1;
    }
}

}
