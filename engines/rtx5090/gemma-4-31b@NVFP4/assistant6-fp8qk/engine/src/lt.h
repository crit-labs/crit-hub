// Prompt GEMMs on cuBLASLt: D[T][M] (row-major, BF16) = X[T][K] * W[M][K]^T for FP8 (E4M3, unscaled) and NVFP4
// (VEC16 UE4M3 block scales, alpha from device memory). Algorithms come from a fixed per-shape choice made once.
#pragma once
#include <cublasLt.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "common.h"

namespace g4 {

class Lt {
public:
    // table: recorded algorithm indices (file of lines "fmt M K T ldd index"), empty = first heuristic.
    Lt(void* workspace, std::size_t workspace_bytes, const std::string& table = {});
    ~Lt();
    Lt(const Lt&) = delete;
    Lt& operator=(const Lt&) = delete;
    // FP8: D = X * W^T (raw codes; row and token scales applied by the caller's epilogue). ldd >= M.
    void fp8(const std::uint8_t* w, int M, int K, const std::uint8_t* x, int T, __nv_bfloat16* d, int ldd, cudaStream_t s);
    // NVFP4: x packed [T][K/2] with swizzled block scales x_sf (T outer); alpha_dev = combined global scale (device).
    void fp4(const Fp4Mat& w, const std::uint8_t* x, const std::uint8_t* x_sf, int T, const float* alpha_dev,
             __nv_bfloat16* d, int ldd, cudaStream_t s);
    // BF16: D (FP32) = X * W^T.
    void bf16(const __nv_bfloat16* w, int M, int K, const __nv_bfloat16* x, int T, float* d, int ldd, cudaStream_t s);
    // Choice index used for (format, M, K, T, ldd); recorded in the engine configuration.
    struct Key {
        int fmt, M, K, T, ldd;
        bool operator<(const Key& o) const { return std::tie(fmt, M, K, T, ldd) < std::tie(o.fmt, o.M, o.K, o.T, o.ldd); }
    };
private:
    struct Plan {
        cublasLtMatmulDesc_t desc = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
        cublasLtMatmulAlgo_t algo{};
        std::vector<cublasLtMatmulAlgo_t> candidates;
        bool tuned = false;
    };
    template <typename Run> void maybe_tune(const Key& key, Plan& p, cudaStream_t s, Run run);
    Plan& plan(const Key& key, const void* a_sf, const void* b_sf);
    cublasLtHandle_t handle_ = nullptr;
    void* workspace_;
    std::size_t workspace_bytes_;
    std::map<Key, Plan> plans_;
    std::map<Key, int> table_;
    std::string tune_out_;
    float* one_ = nullptr;   // device constants for FP8 alpha/beta
    float* zero_ = nullptr;
};

}  // namespace g4
