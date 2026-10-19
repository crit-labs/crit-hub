// Prompt GEMMs on cuBLASLt FP8 tensor cores: D[T][N] (row-major, FP32) = Xq[T][K] * W[N][K]^T for E4M3 codes
// (activations quantized per row by the caller, weights as stored with their per-tensor scale applied by the caller's
// epilogue). The algorithm is the first heuristic result for each shape (deterministic, no timing).
#pragma once
#include <cublasLt.h>
#include <cstdint>
#include <map>
#include <tuple>

namespace nmh {

class Lt {
public:
    Lt(void* workspace, std::size_t workspace_bytes);
    ~Lt();
    Lt(const Lt&) = delete;
    Lt& operator=(const Lt&) = delete;
    void fp8(const std::uint8_t* w, int N, int K, const std::uint8_t* x, int T, float* d, int ldd, cudaStream_t s);
private:
    struct Key {
        int N, K, T, ldd;
        bool operator<(const Key& o) const { return std::tie(N, K, T, ldd) < std::tie(o.N, o.K, o.T, o.ldd); }
    };
    struct Plan {
        cublasLtMatmulDesc_t desc = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
        cublasLtMatmulAlgo_t algo{};
    };
    Plan& plan(const Key& key);
    cublasLtHandle_t handle_ = nullptr;
    void* workspace_;
    std::size_t workspace_bytes_;
    std::map<Key, Plan> plans_;
    float* one_ = nullptr;
    float* zero_ = nullptr;
};

}  // namespace nmh
