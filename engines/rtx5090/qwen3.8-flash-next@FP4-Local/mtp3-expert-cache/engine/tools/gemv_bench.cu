// BF16 decode GEMV shapes: gemv_bench [tokens] [reps]
#include "gemv.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace fnl;
int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 1;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    struct C { int rows, cols; } cs[] = {{320, 10240}, {10240, 320}, {512, 2560}, {12288, 2560}, {2560, 6144},
                                          {2560, 640}, {248320, 2560}};
    float *x, *y;
    cudaMalloc(&x, std::size_t(4) * 10240 * 4);
    std::vector<float> hx(4 * 10240);
    for(std::size_t i = 0; i < hx.size(); ++i) hx[i] = 0.01f * float(int(i % 11) - 5);
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&y, std::size_t(4) * 248320 * 4);
    for(auto c : cs) {
        void* w;
        const std::size_t bytes = std::size_t(c.rows) * c.cols * 2;
        cudaMalloc(&w, bytes);
        cudaMemset(w, 0x3c, bytes);
        Bf16Gemv g{(const __nv_bfloat16*)w, x, y, c.rows, c.cols, Post::none, T};
        bf16_gemv(g, 0);
        cudaDeviceSynchronize();
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        for(int i = 0; i < reps; i++) bf16_gemv(g, 0);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        float hy[2];
        cudaMemcpy(hy, y, 8, cudaMemcpyDeviceToHost);
        std::printf("bf16 gemv %6d x %5d, %d tokens: %8.2f us, %5.0f GB/s (y0 %.6f)\n", c.rows, c.cols, T,
                    1e3 * ms / reps, bytes / (ms / reps) / 1e6, hy[0]);
        cudaFree(w);
    }
}
