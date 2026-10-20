// Expert GEMV throughput when every entry is read from mapped host memory (misses) or from device slots (hits).
//   miss_bench [entries] [tokens]
#include "experts.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    using namespace fnl;
    const int entries = argc > 1 ? std::atoi(argv[1]) : 3;
    const int tokens = argc > 2 ? std::atoi(argv[2]) : 1;
    const int pool = 64;
    uint8_t* host = nullptr;
    FNL_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host), std::size_t(pool) * kSlotBytes, cudaHostAllocMapped));
    std::memset(host, 0x11, std::size_t(pool) * kSlotBytes);
    uint8_t* dev = nullptr;
    FNL_CUDA(cudaMalloc(&dev, std::size_t(pool) * kSlotBytes));
    FNL_CUDA(cudaMemset(dev, 0x11, std::size_t(pool) * kSlotBytes));
    uint8_t* dst = nullptr;
    FNL_CUDA(cudaMalloc(&dst, std::size_t(kMaxUnion) * kSlotBytes));
    FNL_CUDA(cudaMemset(dst, 0x30, std::size_t(kMaxUnion) * kSlotBytes));
    ExpertPlan* dplan;
    FNL_CUDA(cudaMalloc(&dplan, sizeof(ExpertPlan)));
    static ExpertPlan hplan;
    ExpertPlan* plan = &hplan;
    float *x, *act, *sh, *sg, *y;
    FNL_CUDA(cudaMalloc(&x, 4 * kHidden * 4));
    FNL_CUDA(cudaMemset(x, 0, 4 * kHidden * 4));
    FNL_CUDA(cudaMalloc(&act, std::size_t(kMaxUnion) * 4 * kInter * 4));
    FNL_CUDA(cudaMalloc(&sh, 4 * kHidden * 4));
    FNL_CUDA(cudaMemset(sh, 0, 4 * kHidden * 4));
    FNL_CUDA(cudaMalloc(&sg, 16));
    FNL_CUDA(cudaMemset(sg, 0, 16));
    FNL_CUDA(cudaMalloc(&y, 4 * kHidden * 4));
    cudaEvent_t a, b, c;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventCreate(&c);
    for(int miss = 0; miss < 2; ++miss) {
        double total = 0, tg = 0, td = 0;
        const int reps = 20;
        for(int r = 0; r < reps + 2; ++r) {
            plan->n = entries;
            for(int u = 0; u < entries; ++u) {
                const int e = (r * entries + u) % pool;
                plan->expert[u] = e;
                plan->mask[u] = (1 << tokens) - 1;
                for(int t = 0; t < 4; ++t) plan->weight[u][t] = 0.1f;
                plan->s2[u][0] = plan->s2[u][1] = plan->s2[u][2] = 1.f;
                if(miss) plan->ref[u] = {host + std::size_t(e) * kSlotBytes, dst + std::size_t(u) * kSlotBytes, 1, 0};
                else plan->ref[u] = {dev + std::size_t(e) * kSlotBytes, dev + std::size_t(e) * kSlotBytes, 0, 0};
            }
            FNL_CUDA(cudaMemcpy(dplan, plan, sizeof(ExpertPlan), cudaMemcpyHostToDevice));
            FNL_CUDA(cudaDeviceSynchronize());
            cudaEventRecord(a);
            expert_gate_up(dplan, x, act, tokens, 0);
            cudaEventRecord(b);
            expert_down(dplan, act, sh, sg, y, tokens, 0);
            cudaEventRecord(c);
            cudaEventSynchronize(c);
            float ms, ms2;
            cudaEventElapsedTime(&ms, a, b);
            cudaEventElapsedTime(&ms2, b, c);
            if(r >= 2) { total += ms + ms2; tg += ms; td += ms2; }
        }
        const double ms = total / reps;
        std::printf("%s: %d entries, %d tokens: %.3f ms (gate/up %.3f, down %.3f), %.1f GB/s of weights\n",
                    miss ? "misses" : "hits", entries, tokens, ms, tg / reps, td / reps,
                    entries * double(kExpertWeightBytes) / ms / 1e6);
    }
    return 0;
}
