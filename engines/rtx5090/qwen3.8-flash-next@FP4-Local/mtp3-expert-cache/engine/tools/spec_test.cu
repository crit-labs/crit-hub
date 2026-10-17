// Development driver: greedy decoding of a prompt with plain single-token passes and with MTP speculative rounds;
// compares the two token sequences and reports speed and acceptance.
//   spec_test MODEL IDS_JSON N [capacity] [plain|spec|both] [chunk]
#include "model.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    try {
        if(argc < 4) {
            std::cerr << "usage: spec_test MODEL IDS_JSON N [capacity] [plain|spec|both] [chunk]\n";
            return 2;
        }
        std::ifstream in(argv[2]);
        std::stringstream ss;
        ss << in.rdbuf();
        std::string text = ss.str();
        std::vector<int> prompt;
        for(char& c : text)
            if(c == '[' || c == ']' || c == ',') c = ' ';
        std::istringstream is(text);
        for(int v; is >> v;) prompt.push_back(v);
        const int N = std::atoi(argv[3]);
        const std::string mode = argc > 5 ? argv[5] : "both";
        fnl::Options opt;
        opt.model = argv[1];
        opt.capacity = argc > 4 ? std::atoi(argv[4]) : 32768;
        if(argc > 6) opt.prefill_chunk = std::atoi(argv[6]);
        crit::core::Resources resources(std::size_t(80) << 30, std::size_t(32) << 30);
        crit::core::CudaBackend backend(0);
        backend.initialize(resources);
        auto t0 = std::chrono::steady_clock::now();
        fnl::Net net(opt, backend, resources);
        FNL_CUDA(cudaDeviceSynchronize());
        std::fprintf(stderr, "loaded in %.1fs, %zu expert slots, mtp %d\n", since(t0), net.expert_slots(),
                     int(net.has_mtp()));
        const int P = int(prompt.size());
        const auto s = net.stream();
        auto prefill_all = [&](std::vector<int>& hist, bool drafter) {
            for(int c0 = 0; c0 < P; c0 += net.prefill_chunk()) {
                const int n = std::min(net.prefill_chunk(), P - c0);
                net.prefill(hist, c0, n, false, drafter);
            }
        };
        std::vector<int> plain, spec;
        for(int rep = 0; rep < 2; ++rep) {
            if(mode == "plain" || mode == "both") {
                std::vector<int> hist = prompt;
                net.reset();
                net.begin_request(P + N + 8);
                auto ta = std::chrono::steady_clock::now();
                prefill_all(hist, false);
                int next;
                FNL_CUDA(cudaMemcpyAsync(&next, net.next_token(), 4, cudaMemcpyDeviceToHost, s));
                FNL_CUDA(cudaStreamSynchronize(s));
                const double ttft = since(ta);
                const auto l0 = net.expert_lookups(), m0 = net.expert_misses();
                auto td = std::chrono::steady_clock::now();
                hist.push_back(next);
                while(int(hist.size()) < P + N) {
                    net.decode(hist, int(hist.size()) - 1);
                    FNL_CUDA(cudaMemcpyAsync(&next, net.next_token(), 4, cudaMemcpyDeviceToHost, s));
                    FNL_CUDA(cudaStreamSynchronize(s));
                    hist.push_back(next);
                }
                const double dt = since(td);
                const double lk = double(net.expert_lookups() - l0), ms = double(net.expert_misses() - m0);
                std::fprintf(stderr, "plain: ttft %.3f s, decode %.2f tok/s (%d tokens), miss rate %.3f\n", ttft,
                             (N - 1) / dt, N - 1, lk > 0 ? ms / lk : 0.0);
                plain.assign(hist.begin() + P, hist.end());
            }
            if(net.has_mtp() && (mode == "spec" || mode == "both")) {
                std::vector<int> hist = prompt;
                net.reset();
                net.begin_request(P + N + 8);
                auto ta = std::chrono::steady_clock::now();
                prefill_all(hist, true);
                net.spec_begin(P - 1);
                net.spec_begin_prefetch(hist, P - 1);
                FNL_CUDA(cudaEventSynchronize(net.spec_done()));
                const double ttft = since(ta);
                fnl::Accept a = net.spec_result();
                hist.push_back(a.bonus);
                int drafts[fnl::kDrafts];
                for(int k = 0; k < fnl::kDrafts; ++k) drafts[k] = a.drafts[k];
                const auto l0 = net.expert_lookups(), m0 = net.expert_misses();
                auto td = std::chrono::steady_clock::now();
                int rounds = 0;
                int hist_n[fnl::kMaxTokens + 1] = {0};
                while(int(hist.size()) < P + N) {
                    const int pos = int(hist.size()) - 1;
                    net.spec_round(hist, pos, drafts);
                    FNL_CUDA(cudaEventSynchronize(net.spec_done()));
                    a = net.spec_result();
                    ++rounds;
                    ++hist_n[a.n];
                    for(int k = 0; k + 1 < a.n; ++k) hist.push_back(drafts[k]);
                    hist.push_back(a.bonus);
                    for(int k = 0; k < fnl::kDrafts; ++k) drafts[k] = a.drafts[k];
                }
                const double dt = since(td);
                const int produced = int(hist.size()) - P - 1;
                const double lk = double(net.expert_lookups() - l0), ms = double(net.expert_misses() - m0);
                std::fprintf(stderr,
                             "spec: ttft %.3f s, decode %.2f tok/s (%d tokens, %d rounds, %.2f tokens/round, %.2f ms/round), "
                             "kept n=1..4: %d %d %d %d, misses/round %.1f\n",
                             ttft, produced / dt, produced, rounds, double(produced) / rounds, 1000.0 * dt / rounds,
                             hist_n[1], hist_n[2], hist_n[3], hist_n[4], ms / rounds);
                (void)lk;
                const auto pt = net.pass_timing();
                std::fprintf(stderr, "device time per round: target pass %.2f ms, drafter %.2f ms\n", pt.first / rounds,
                             pt.second / rounds);
                spec.assign(hist.begin() + P, hist.begin() + std::min<std::size_t>(hist.size(), P + N));
            }
        }
        net.cpu_report();
        if(!plain.empty() && !spec.empty()) {
            int same = 0;
            while(same < int(std::min(plain.size(), spec.size())) && plain[same] == spec[same]) ++same;
            std::fprintf(stderr, "plain and speculative agree on the first %d of %zu tokens\n", same, plain.size());
        }
        std::ostringstream o;
        o << "{\"plain\": [";
        for(std::size_t i = 0; i < plain.size(); ++i) o << (i ? "," : "") << plain[i];
        o << "], \"spec\": [";
        for(std::size_t i = 0; i < spec.size(); ++i) o << (i ? "," : "") << spec[i];
        o << "]}\n";
        std::cout << o.str();
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
