// Unit test of the prompt GEMM against a host FP64 reference on exactly dequantized weights.
#include "gemm.cuh"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace fnl;

static float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
static float e4m3(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return float(v);
}
static float e2m1(int n) {
    const float m[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return (n & 8) ? -m[n & 7] : m[n & 7];
}

int run(WFmt fmt, Epi epi, int M, int N, int K, bool gather) {
    std::mt19937 rng(1234 + int(fmt) * 7 + int(epi));
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    const int Arows = M + 37;
    std::vector<__half> A(std::size_t(Arows) * K);
    for(auto& a : A) a = __float2half(u(rng));
    std::vector<int> rows(M);
    for(int i = 0; i < M; ++i) rows[i] = gather ? (i * 7 + 3) % Arows : i;
    const int WN = epi == Epi::act ? N / 2 : N;   // rows per weight matrix
    const int nmat = epi == Epi::act ? 2 : 1;
    std::vector<std::vector<uint8_t>> W(nmat), S(nmat);
    std::vector<std::vector<float>> Wf(nmat, std::vector<float>(std::size_t(WN) * K));
    for(int m = 0; m < nmat; ++m) {
        if(fmt == WFmt::bf16) {
            W[m].resize(std::size_t(WN) * K * 2);
            auto* w = reinterpret_cast<__nv_bfloat16*>(W[m].data());
            for(std::size_t i = 0; i < std::size_t(WN) * K; ++i) {
                w[i] = __float2bfloat16(u(rng) * 0.05f);
                Wf[m][i] = __bfloat162float(w[i]);
            }
        } else if(fmt == WFmt::mx8) {
            W[m].resize(std::size_t(WN) * K);
            S[m].resize(std::size_t(WN) * K / 32);
            for(auto& s : S[m]) s = uint8_t((epi == Epi::act ? 112 : 120) + rng() % 8);
            for(std::size_t i = 0; i < W[m].size(); ++i) {
                uint8_t b = uint8_t(rng() & 0xff);
                if((b & 0x7f) == 0x7f) b = 0;
                W[m][i] = b;
                const int r = int(i / K), c = int(i % K);
                Wf[m][i] = e4m3(b) * std::ldexp(1.f, int(S[m][r * (K / 32) + c / 32]) - 127);
            }
        } else {
            W[m].resize(std::size_t(WN) * K / 2);
            S[m].resize(std::size_t(WN) * K / 16);
            for(auto& s : S[m]) s = uint8_t((epi == Epi::act ? 20 : 40) + rng() % 40);
            for(std::size_t i = 0; i < W[m].size(); ++i) {
                const uint8_t b = uint8_t(rng() & 0xff);
                W[m][i] = b;
                const int r = int((2 * i) / K), c = int((2 * i) % K);
                const float sc = e4m3(S[m][r * (K / 16) + c / 16]);
                Wf[m][2 * i] = e2m1(b & 15) * sc;
                Wf[m][2 * i + 1] = e2m1(b >> 4) * sc;
            }
        }
    }
    const float alpha = fmt == WFmt::fp4 ? 0.37f : 1.f, alpha2 = 0.81f;
    // reference
    const int outN = epi == Epi::act ? N / 2 : N;
    std::vector<double> ref(std::size_t(M) * outN);
    for(int i = 0; i < M; ++i)
        for(int j = 0; j < outN; ++j) {
            auto dot = [&](int m) {
                double s = 0;
                for(int k = 0; k < K; ++k)
                    s += double(__half2float(A[std::size_t(rows[i]) * K + k])) * double(__half2float(__float2half(Wf[m][std::size_t(j) * K + k])));
                return s;
            };
            double v;
            if(epi == Epi::act) {
                const double g = dot(0) * alpha, uu = dot(1) * alpha2;
                v = g / (1 + std::exp(-g)) * uu;
            } else {
                const double d = dot(0);
                if(epi == Epi::f32) v = d * alpha;
                else if(epi == Epi::f32_sigmoid) v = 1 / (1 + std::exp(-d));
                else if(epi == Epi::bf16_siluq) v = (d / 4) / (1 + std::exp(-d / 4));
                else v = d * alpha;
            }
            ref[std::size_t(i) * outN + j] = v;
        }
    // device
    void *dA, *dW[2] = {nullptr, nullptr}, *dS[2] = {nullptr, nullptr}, *dC;
    int* dR;
    FNL_CUDA(cudaMalloc(&dA, A.size() * 2));
    FNL_CUDA(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    for(int m = 0; m < nmat; ++m) {
        FNL_CUDA(cudaMalloc(&dW[m], W[m].size()));
        FNL_CUDA(cudaMemcpy(dW[m], W[m].data(), W[m].size(), cudaMemcpyHostToDevice));
        if(!S[m].empty()) {
            FNL_CUDA(cudaMalloc(&dS[m], S[m].size()));
            FNL_CUDA(cudaMemcpy(dS[m], S[m].data(), S[m].size(), cudaMemcpyHostToDevice));
        }
    }
    FNL_CUDA(cudaMalloc(&dR, M * 4));
    FNL_CUDA(cudaMemcpy(dR, rows.data(), M * 4, cudaMemcpyHostToDevice));
    FNL_CUDA(cudaMalloc(&dC, std::size_t(M) * outN * 4));
    GemmArgs p;
    p.A = static_cast<__half*>(dA);
    p.lda = K;
    p.a_rows = gather ? dR : nullptr;
    p.M = M;
    p.N = N;
    p.K = K;
    p.W = dW[0];
    p.Ws = static_cast<uint8_t*>(dS[0]);
    p.W2 = dW[1];
    p.Ws2 = static_cast<uint8_t*>(dS[1]);
    p.alpha = alpha;
    p.alpha2 = alpha2;
    const bool bout = epi == Epi::act || epi == Epi::bf16_siluq || epi == Epi::bf16;
    p.C = bout ? nullptr : static_cast<float*>(dC);
    p.Cb = bout ? static_cast<__half*>(dC) : nullptr;
    p.ldc = outN;
    gemm(fmt, epi, p, 0, 0);
    FNL_CUDA(cudaDeviceSynchronize());
    std::vector<float> got(std::size_t(M) * outN);
    if(bout) {
        std::vector<__half> gb(got.size());
        FNL_CUDA(cudaMemcpy(gb.data(), dC, gb.size() * 2, cudaMemcpyDeviceToHost));
        for(std::size_t i = 0; i < got.size(); ++i) got[i] = __half2float(gb[i]);
    } else {
        FNL_CUDA(cudaMemcpy(got.data(), dC, got.size() * 4, cudaMemcpyDeviceToHost));
    }
    double maxerr = 0, maxref = 0;
    for(std::size_t i = 0; i < got.size(); ++i) {
        maxerr = std::max(maxerr, std::abs(got[i] - ref[i]));
        maxref = std::max(maxref, std::abs(ref[i]));
    }
    const double tol = bout ? 1e-2 : 1e-4;
    const bool ok = maxerr <= tol * std::max(1.0, maxref);
    std::printf("fmt %d epi %d M %d N %d K %d gather %d: max err %.3e (max |ref| %.3e) %s\n", int(fmt), int(epi), M,
                N, K, int(gather), maxerr, maxref, ok ? "ok" : "FAIL");
    cudaFree(dA);
    cudaFree(dC);
    cudaFree(dR);
    for(int m = 0; m < 2; ++m) {
        cudaFree(dW[m]);
        cudaFree(dS[m]);
    }
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    fails += run(WFmt::bf16, Epi::f32, 200, 300, 320, false);
    fails += run(WFmt::bf16, Epi::bf16_siluq, 130, 320, 2560, true);
    fails += run(WFmt::bf16, Epi::f32_sigmoid, 77, 257, 320, false);
    fails += run(WFmt::mx8, Epi::f32, 300, 200, 2560, false);
    fails += run(WFmt::mx8, Epi::act, 129, 1280, 2560, true);
    fails += run(WFmt::fp4, Epi::f32, 150, 2560, 640, true);
    fails += run(WFmt::fp4, Epi::act, 64, 1280, 2560, false);
    return fails;
}
