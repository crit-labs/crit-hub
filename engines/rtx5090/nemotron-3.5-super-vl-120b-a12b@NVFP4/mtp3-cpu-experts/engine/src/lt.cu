// Adapted from the kestrel engine's cuBLASLt wrapper (crit-hub qwen3.8-27b mtp7-fp8attn): FP8 only, FP32 output.
#include "lt.h"
#include "common.cuh"

namespace nmh {
namespace {
void lt_check(cublasStatus_t s, const char* what) {
    if(s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: " + std::to_string(static_cast<int>(s)));
}
}  // namespace

Lt::Lt(void* workspace, std::size_t workspace_bytes) : workspace_(workspace), workspace_bytes_(workspace_bytes) {
    lt_check(cublasLtCreate(&handle_), "create");
    NMH_CUDA(cudaMalloc(&one_, 2 * sizeof(float)));
    zero_ = one_ + 1;
    const float h[2] = {1.f, 0.f};
    NMH_CUDA(cudaMemcpy(one_, h, sizeof(h), cudaMemcpyHostToDevice));
}

Lt::~Lt() {
    for(auto& [k, p] : plans_) {
        cublasLtMatrixLayoutDestroy(p.a);
        cublasLtMatrixLayoutDestroy(p.b);
        cublasLtMatrixLayoutDestroy(p.d);
        cublasLtMatmulDescDestroy(p.desc);
    }
    cudaFree(one_);
    cublasLtDestroy(handle_);
}

Lt::Plan& Lt::plan(const Key& key) {
    auto it = plans_.find(key);
    if(it != plans_.end()) return it->second;
    Plan p;
    lt_check(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    const cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)), "pointer mode");
    // A = W stored [N][K] row-major = column-major K x N, transposed. B = X column-major K x T. D column-major N x T.
    lt_check(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, key.K, key.N, key.K), "layout a");
    lt_check(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, key.K, key.T, key.K), "layout b");
    lt_check(cublasLtMatrixLayoutCreate(&p.d, CUDA_R_32F, key.N, key.T, key.ldd), "layout d");
    cublasLtMatmulPreference_t pref;
    lt_check(cublasLtMatmulPreferenceCreate(&pref), "pref");
    lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
                                                  sizeof(workspace_bytes_)), "pref ws");
    cublasLtMatmulHeuristicResult_t res[4];
    int n = 0;
    lt_check(cublasLtMatmulAlgoGetHeuristic(handle_, p.desc, p.a, p.b, p.d, p.d, pref, 4, res, &n), "heuristic");
    cublasLtMatmulPreferenceDestroy(pref);
    if(n == 0)
        throw std::runtime_error("cuBLASLt: no FP8 algorithm for N=" + std::to_string(key.N) + " K=" +
                                 std::to_string(key.K) + " T=" + std::to_string(key.T));
    p.algo = res[0].algo;
    return plans_.emplace(key, p).first->second;
}

void Lt::fp8(const std::uint8_t* w, int N, int K, const std::uint8_t* x, int T, float* d, int ldd, cudaStream_t s) {
    Plan& p = plan({N, K, T, ldd});
    lt_check(cublasLtMatmul(handle_, p.desc, one_, w, p.a, x, p.b, zero_, d, p.d, d, p.d, &p.algo, workspace_,
                            workspace_bytes_, s), "fp8 matmul");
}

}  // namespace nmh
