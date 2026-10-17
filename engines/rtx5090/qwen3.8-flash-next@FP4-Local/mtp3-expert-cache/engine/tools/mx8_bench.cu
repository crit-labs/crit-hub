// MXFP8 decode GEMV shapes of one layer, back to back: mx8_bench [tokens] [reps]
#include "gemv.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    using namespace fnl;
    const int T = argc > 1 ? std::atoi(argv[1]) : 4;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
    struct Shape { const char* name; int rows[4]; int nseg; int cols; };
    const Shape shapes[] = {
        {"gdn in (qkv,z,a,b)", {kQkvDim, kZDim, kVHeads, kVHeads}, 4, kHidden},
        {"gdn out", {kHidden}, 1, kZDim},
        {"attn in (q,k,v,idx)", {kQProj, kKvHeads * kHeadDim, kKvHeads * kHeadDim, kIdxProj}, 4, kHidden},
        {"attn out", {kHidden}, 1, kAttnOut},
        {"shared gate,up", {kShared, kShared}, 2, kHidden},
        {"shared down", {kHidden}, 1, kShared},
    };
    // Enough copies of each matrix to defeat L2 between calls.
    const int copies = 6;
    float *x, *y;
    FNL_CUDA(cudaMalloc(&x, std::size_t(4) * 8192 * 4));
    std::vector<float> hx(4 * 8192);
    for(std::size_t i = 0; i < hx.size(); ++i) hx[i] = 0.01f * float(int(i % 13) - 6);
    FNL_CUDA(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    FNL_CUDA(cudaMalloc(&y, std::size_t(4) * 16384 * 4 * 4));
    cudaStream_t s;
    FNL_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    double total_us = 0, total_bytes = 0;
    for(const Shape& sh : shapes) {
        int rows = 0;
        for(int i = 0; i < sh.nseg; ++i) rows += sh.rows[i];
        const std::size_t wbytes = std::size_t(rows) * sh.cols, sbytes = std::size_t(rows) * (sh.cols / 32);
        std::vector<Mx8Batch> batches(copies);
        std::vector<void*> mem;
        for(int k = 0; k < copies; ++k) {
            uint8_t *w, *sc;
            FNL_CUDA(cudaMalloc(&w, wbytes));
            FNL_CUDA(cudaMalloc(&sc, sbytes));
            FNL_CUDA(cudaMemset(w, 0x31, wbytes));
            FNL_CUDA(cudaMemset(sc, 0x7f, sbytes));
            mem.push_back(w);
            mem.push_back(sc);
            Mx8Batch& bt = batches[k];
            bt = Mx8Batch{};
            std::size_t ro = 0;
            float* yo = y;
            for(int i = 0; i < sh.nseg; ++i) {
                bt.seg[i] = {w + ro * sh.cols, sc + ro * (sh.cols / 32), yo, sh.rows[i]};
                ro += sh.rows[i];
                yo += std::size_t(T) * sh.rows[i];
            }
            bt.nseg = sh.nseg;
            bt.cols = sh.cols;
            bt.x = x;
            bt.tokens = T;
        }
        for(int r = 0; r < copies; ++r) mx8_gemv(batches[r], s);
        FNL_CUDA(cudaEventRecord(a, s));
        for(int r = 0; r < reps; ++r) mx8_gemv(batches[r % copies], s);
        FNL_CUDA(cudaEventRecord(b, s));
        FNL_CUDA(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        const double us = 1e3 * ms / reps;
        total_us += us;
        total_bytes += double(wbytes + sbytes);
        std::printf("%-22s %6d x %5d, %d tokens: %7.1f us, %6.0f GB/s\n", sh.name, rows, sh.cols, T, us,
                    double(wbytes + sbytes) / us / 1e3);
        for(void* p : mem) cudaFree(p);
    }
    std::printf("all shapes: %.1f us, %.0f GB/s\n", total_us, total_bytes / total_us / 1e3);
    float hy[2];
    FNL_CUDA(cudaMemcpy(hy, y, 8, cudaMemcpyDeviceToHost));
    std::printf("check %.5f %.5f\n", hy[0], hy[1]);
    return 0;
}
