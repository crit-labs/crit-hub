// Development driver: run a token list through the decode path (one position at a time) or the prompt path,
// dump per-layer hidden streams and logits.
//   fnl_test MODEL IDS_JSON OUT_DIR [capacity] [slots] [decode|prefill|both] [chunk]
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
            std::cerr << "usage: fnl_test MODEL IDS_JSON OUT_DIR [capacity] [slots] [decode|prefill|both] [chunk]\n";
            return 2;
        }
        std::ifstream in(argv[2]);
        std::stringstream ss;
        ss << in.rdbuf();
        std::string text = ss.str();
        std::vector<int> ids;
        for(char& c : text)
            if(c == '[' || c == ']' || c == ',') c = ' ';
        std::istringstream is(text);
        for(int v; is >> v;) ids.push_back(v);
        const std::string out = argv[3];
        const std::string mode = argc > 6 ? argv[6] : "decode";
        fnl::Options opt;
        opt.model = argv[1];
        opt.capacity = argc > 4 ? std::atoi(argv[4]) : 4096;
        if(argc > 5) opt.expert_slots = std::size_t(std::atoll(argv[5]));
        if(argc > 7) opt.prefill_chunk = std::atoi(argv[7]);
        crit::core::Resources resources(std::size_t(80) << 30, std::size_t(32) << 30);
        crit::core::CudaBackend backend(0);
        backend.initialize(resources);
        auto t0 = std::chrono::steady_clock::now();
        fnl::Net net(opt, backend, resources);
        FNL_CUDA(cudaDeviceSynchronize());
        std::fprintf(stderr, "loaded in %.1fs, %zu expert slots\n", since(t0), net.expert_slots());
        const int T = int(ids.size());
        std::vector<float> layers(std::size_t(fnl::kLayers) * T * fnl::kHyper);
        if(mode == "decode" || mode == "both") {
            int cur = 0;
            net.on_layer = [&](int l, const float* H, int n) {
                FNL_CUDA(cudaMemcpyAsync(layers.data() + (std::size_t(l) * T + cur) * fnl::kHyper, H,
                                         std::size_t(n) * fnl::kHyper * 4, cudaMemcpyDeviceToHost, net.stream()));
            };
            std::vector<float> logits(std::size_t(T) * fnl::kVocab);
            std::vector<int> argmax(T);
            auto t2 = std::chrono::steady_clock::now();
            for(cur = 0; cur < T; ++cur) {
                net.decode(ids, cur);
                FNL_CUDA(cudaMemcpyAsync(logits.data() + std::size_t(cur) * fnl::kVocab, net.logits(),
                                         fnl::kVocab * 4, cudaMemcpyDeviceToHost, net.stream()));
                FNL_CUDA(cudaMemcpyAsync(&argmax[cur], net.next_token(), 4, cudaMemcpyDeviceToHost, net.stream()));
                FNL_CUDA(cudaStreamSynchronize(net.stream()));
            }
            std::fprintf(stderr, "decoded %d positions in %.3fs; expert lookups %llu misses %llu\n", T, since(t2),
                         net.expert_lookups(), net.expert_misses());
            std::fprintf(stderr, "argmax:");
            for(int v : argmax) std::fprintf(stderr, " %d", v);
            std::fprintf(stderr, "\n");
            std::ofstream(out + "/layers.f32", std::ios::binary)
                .write(reinterpret_cast<const char*>(layers.data()), std::streamsize(layers.size() * 4));
            std::ofstream(out + "/logits.f32", std::ios::binary)
                .write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * 4));
            net.reset();
        }
        if(mode == "prefill" || mode == "both") {
            net.on_layer = [&](int l, const float* H, int n) {
                FNL_CUDA(cudaMemcpyAsync(layers.data() + std::size_t(l) * T * fnl::kHyper, H,
                                         std::size_t(n) * fnl::kHyper * 4, cudaMemcpyDeviceToHost, net.stream()));
            };
            std::vector<float> logits(std::size_t(T) * fnl::kVocab);
            const int rows = fnl::Net::kScoreRows;
            for(int rep = 0; rep < 2; ++rep) {
                net.reset();
                auto t2 = std::chrono::steady_clock::now();
                net.prefill(ids, 0, T, true);
                FNL_CUDA(cudaStreamSynchronize(net.stream()));
                std::fprintf(stderr, "prefill %d tokens in %.3fs\n", T, since(t2));
                for(int r = 0; r < T; r += rows) {
                    const int n = std::min(rows, T - r);
                    net.head_rows(r, n, net.score_logits());
                    FNL_CUDA(cudaStreamSynchronize(net.stream()));
                    FNL_CUDA(cudaMemcpy(logits.data() + std::size_t(r) * fnl::kVocab, net.score_logits(),
                                        std::size_t(n) * fnl::kVocab * 4, cudaMemcpyDeviceToHost));
                }
                std::fprintf(stderr, "prefill %d tokens in %.3fs (fetched %llu experts so far)\n", T, since(t2),
                             net.prefill_fetched());
            }
            std::ofstream(out + "/players.f32", std::ios::binary)
                .write(reinterpret_cast<const char*>(layers.data()), std::streamsize(layers.size() * 4));
            std::ofstream(out + "/plogits.f32", std::ios::binary)
                .write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * 4));
            // A few decode steps after the prompt.
            net.on_layer = nullptr;
            std::vector<int> hist = ids;
            net.reset();
            net.prefill(hist, 0, T);
            auto t3 = std::chrono::steady_clock::now();
            for(int i = 0; i < 16; ++i) {
                int next;
                FNL_CUDA(cudaMemcpyAsync(&next, net.next_token(), 4, cudaMemcpyDeviceToHost, net.stream()));
                FNL_CUDA(cudaStreamSynchronize(net.stream()));
                hist.push_back(next);
                net.decode(hist, int(hist.size()) - 1);
            }
            FNL_CUDA(cudaStreamSynchronize(net.stream()));
            std::fprintf(stderr, "16 decode steps after prefill in %.3fs:", since(t3));
            for(std::size_t i = ids.size(); i < hist.size(); ++i) std::fprintf(stderr, " %d", hist[i]);
            std::fprintf(stderr, "\n");
        }
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
