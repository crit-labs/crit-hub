// Device-resident expert GEMVs back to back (launch cost amortized): expert_bench [entries] [tokens] [reps] [full]
// Rotates over plans with disjoint experts from a pool larger than L2, so every call reads HBM.
#include "experts.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    using namespace fnl;
    const int entries = argc > 1 ? std::atoi(argv[1]) : 30;
    const int tokens = argc > 2 ? std::atoi(argv[2]) : 4;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 40;
    const bool full = argc > 4 && std::atoi(argv[4]) != 0;   // every token routed to every entry
    const int nplans = 8;
    const int pool = entries * nplans;
    uint8_t* dev = nullptr;
    FNL_CUDA(cudaMalloc(&dev, std::size_t(pool) * kSlotBytes));
    // Weights 0x11 (small E2M1 values), scales 0x38 (E4M3 1.0).
    FNL_CUDA(cudaMemset(dev, 0x11, std::size_t(pool) * kSlotBytes));
    for(int e = 0; e < pool; ++e)
        FNL_CUDA(cudaMemset(dev + std::size_t(e) * kSlotBytes + kSlotScales, 0x38, kSlotBytes - kSlotScales));
    std::vector<ExpertPlan> plans(nplans);
    ExpertPlan* dplans;
    FNL_CUDA(cudaMalloc(&dplans, nplans * sizeof(ExpertPlan)));
    for(int p = 0; p < nplans; ++p) {
        ExpertPlan& plan = plans[p];
        plan.n = entries;
        for(int u = 0; u < entries; ++u) {
            const int e = p * entries + u;
            plan.expert[u] = e;
            // A pass of T tokens routes about 1.33 tokens to each union entry (40 routes over ~30 entries).
            plan.mask[u] = full ? (1 << tokens) - 1 : (1 << (u % tokens)) | (u % 3 == 0 ? 1 << ((u + 1) % tokens) : 0);
            for(int t = 0; t < 4; ++t) plan.weight[u][t] = 0.1f;
            plan.s2[u][0] = plan.s2[u][1] = plan.s2[u][2] = 1.f;
            uint8_t* s = dev + std::size_t(e) * kSlotBytes;
            plan.ref[u] = {s, s, 0, 0};
        }
    }
    FNL_CUDA(cudaMemcpy(dplans, plans.data(), nplans * sizeof(ExpertPlan), cudaMemcpyHostToDevice));
    float *x, *act, *sh, *sg, *y;
    FNL_CUDA(cudaMalloc(&x, 4 * kHidden * 4));
    std::vector<float> hx(4 * kHidden);
    for(std::size_t i = 0; i < hx.size(); ++i) hx[i] = 0.01f * float(int(i % 17) - 8);
    FNL_CUDA(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    FNL_CUDA(cudaMalloc(&act, std::size_t(kMaxUnion) * 4 * kInter * 4));
    FNL_CUDA(cudaMalloc(&sh, 4 * kHidden * 4));
    FNL_CUDA(cudaMemset(sh, 0, 4 * kHidden * 4));
    FNL_CUDA(cudaMalloc(&sg, 16));
    FNL_CUDA(cudaMemset(sg, 0, 16));
    FNL_CUDA(cudaMalloc(&y, 4 * kHidden * 4));
    float* part;
    FNL_CUDA(cudaMalloc(&part, std::size_t(kMaxUnion) * kMaxPassTokens * kHidden * 4));
    cudaStream_t s;
    FNL_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for(int which = 0; which < 3; ++which) {
        // warm-up
        for(int r = 0; r < 4; ++r) {
            expert_gate_up(dplans + r % nplans, x, act, tokens, s);
            expert_down(dplans + r % nplans, act, sh, sg, y, tokens, s, part);
        }
        // The calls run as one CUDA graph, as in the engine (launch gaps of a graph, not of the stream).
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        FNL_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for(int r = 0; r < reps; ++r) {
            const ExpertPlan* p = dplans + r % nplans;
            if(which != 2) expert_gate_up(p, x, act, tokens, s);
            if(which != 1) expert_down(p, act, sh, sg, y, tokens, s, part);
        }
        FNL_CUDA(cudaStreamEndCapture(s, &graph));
        FNL_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        FNL_CUDA(cudaGraphLaunch(exec, s));
        FNL_CUDA(cudaEventRecord(a, s));
        FNL_CUDA(cudaGraphLaunch(exec, s));
        FNL_CUDA(cudaEventRecord(b, s));
        FNL_CUDA(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        const double us = 1e3 * ms / reps;
        const double bytes = which == 0 ? kExpertWeightBytes : which == 1 ? 2 * kGateBytes : kDownBytes;
        std::printf("%-8s %2d entries, %d tokens: %7.1f us per call, %6.0f GB/s of weights\n",
                    which == 0 ? "both" : which == 1 ? "gate_up" : "down", entries, tokens, us,
                    entries * bytes / us / 1e3);
    }
    float hy[4];
    FNL_CUDA(cudaMemcpy(hy, y, 16, cudaMemcpyDeviceToHost));
    std::printf("check y[0] %.6f\n", hy[0]);
    return 0;
}
