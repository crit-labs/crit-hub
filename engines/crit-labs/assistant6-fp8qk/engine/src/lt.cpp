#include "lt.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace g4 {
namespace {
void lt_check(cublasStatus_t s, const char* what) {
    if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: " + std::to_string(static_cast<int>(s)));
}
}  // namespace

Lt::Lt(void* workspace, std::size_t workspace_bytes, const std::string& table)
    : workspace_(workspace), workspace_bytes_(workspace_bytes) {
    if(!table.empty()) {
        std::ifstream f(table);
        if(!f) throw std::runtime_error("cuBLASLt table not readable: " + table);
        Key k{};
        int idx = 0;
        while(f >> k.fmt >> k.M >> k.K >> k.T >> k.ldd >> idx) table_[k] = idx;
    }
    if(const char* t = std::getenv("G4_LT_TUNE")) tune_out_ = t;
    lt_check(cublasLtCreate(&handle_), "create");
    KCHECK(cudaMalloc(&one_, 2 * sizeof(float)));
    zero_ = one_ + 1;
    const float h[2] = {1.f, 0.f};
    KCHECK(cudaMemcpy(one_, h, sizeof(h), cudaMemcpyHostToDevice));
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

Lt::Plan& Lt::plan(const Key& key, const void* a_sf, const void* b_sf) {
    auto it = plans_.find(key);
    if(it != plans_.end()) return it->second;
    Plan p;
    const bool fp4 = key.fmt == 1;
    lt_check(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    const cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)), "pointer mode");
    const cudaDataType_t in = fp4 ? CUDA_R_4F_E2M1 : key.fmt == 2 ? CUDA_R_16BF : CUDA_R_8F_E4M3;
    if(fp4) {
        const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)), "a scale mode");
        lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)), "b scale mode");
        lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sf, sizeof(a_sf)), "a scale");
        lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sf, sizeof(b_sf)), "b scale");
    }
    // A = W stored [M][K] row-major = column-major K x M, transposed. B = X column-major K x T. D column-major M x T.
    lt_check(cublasLtMatrixLayoutCreate(&p.a, in, key.K, key.M, key.K), "layout a");
    lt_check(cublasLtMatrixLayoutCreate(&p.b, in, key.K, key.T, key.K), "layout b");
    lt_check(cublasLtMatrixLayoutCreate(&p.d, key.fmt == 2 ? CUDA_R_32F : CUDA_R_16BF, key.M, key.T, key.ldd), "layout d");
    cublasLtMatmulPreference_t pref;
    lt_check(cublasLtMatmulPreferenceCreate(&pref), "pref");
    lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
                                                  sizeof(workspace_bytes_)), "pref ws");
    cublasLtMatmulHeuristicResult_t res[16];
    int n = 0;
    lt_check(cublasLtMatmulAlgoGetHeuristic(handle_, p.desc, p.a, p.b, p.d, p.d, pref, 16, res, &n), "heuristic");
    cublasLtMatmulPreferenceDestroy(pref);
    if(n == 0) throw std::runtime_error("cuBLASLt: no algorithm for M=" + std::to_string(key.M) + " K=" + std::to_string(key.K) +
                                        " T=" + std::to_string(key.T));
    for(int i = 0; i < n; ++i) p.candidates.push_back(res[i].algo);
    const auto t = table_.find(key);
    p.algo = res[t != table_.end() && t->second < n ? t->second : 0].algo;
    p.tuned = tune_out_.empty();
    return plans_.emplace(key, p).first->second;
}

template <typename Run>
void Lt::maybe_tune(const Key& key, Plan& p, cudaStream_t s, Run run) {
    if(p.tuned) return;
    p.tuned = true;
    cudaEvent_t a, b;
    KCHECK(cudaEventCreate(&a));
    KCHECK(cudaEventCreate(&b));
    int best = 0;
    float best_ms = 1e30f;
    for(std::size_t i = 0; i < p.candidates.size(); ++i) {
        float ms = 0.f;
        bool ok = true;
        for(int r = 0; r < 3 && ok; ++r) {
            KCHECK(cudaEventRecord(a, s));
            ok = run(p.candidates[i]) == CUBLAS_STATUS_SUCCESS;
            KCHECK(cudaEventRecord(b, s));
            KCHECK(cudaEventSynchronize(b));
            float m = 0.f;
            KCHECK(cudaEventElapsedTime(&m, a, b));
            if(r > 0) ms += m;  // first run is a warmup
        }
        if(ok && ms < best_ms) { best_ms = ms; best = static_cast<int>(i); }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    p.algo = p.candidates[best];
    std::ofstream f(tune_out_, std::ios::app);
    f << key.fmt << ' ' << key.M << ' ' << key.K << ' ' << key.T << ' ' << key.ldd << ' ' << best << '\n';
}

void Lt::fp8(const std::uint8_t* w, int M, int K, const std::uint8_t* x, int T, __nv_bfloat16* d, int ldd, cudaStream_t s) {
    const Key key{0, M, K, T, ldd};
    Plan& p = plan(key, nullptr, nullptr);
    maybe_tune(key, p, s, [&](const cublasLtMatmulAlgo_t& al) {
        return cublasLtMatmul(handle_, p.desc, one_, w, p.a, x, p.b, zero_, d, p.d, d, p.d, &al, workspace_, workspace_bytes_, s);
    });
    lt_check(cublasLtMatmul(handle_, p.desc, one_, w, p.a, x, p.b, zero_, d, p.d, d, p.d, &p.algo, workspace_,
                            workspace_bytes_, s), "fp8 matmul");
}

void Lt::bf16(const __nv_bfloat16* w, int M, int K, const __nv_bfloat16* x, int T, float* d, int ldd, cudaStream_t s) {
    const Key key{2, M, K, T, ldd};
    Plan& p = plan(key, nullptr, nullptr);
    maybe_tune(key, p, s, [&](const cublasLtMatmulAlgo_t& al) {
        return cublasLtMatmul(handle_, p.desc, one_, w, p.a, x, p.b, zero_, d, p.d, d, p.d, &al, workspace_, workspace_bytes_, s);
    });
    lt_check(cublasLtMatmul(handle_, p.desc, one_, w, p.a, x, p.b, zero_, d, p.d, d, p.d, &p.algo, workspace_,
                            workspace_bytes_, s), "bf16 matmul");
}

void Lt::fp4(const Fp4Mat& w, const std::uint8_t* x, const std::uint8_t* x_sf, int T, const float* alpha_dev,
             __nv_bfloat16* d, int ldd, cudaStream_t s) {
    Plan& p = plan({1, w.rows, w.cols, T, ldd}, w.sf, x_sf);
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w.sf, sizeof(w.sf)), "a scale");
    lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x_sf, sizeof(x_sf)), "b scale");
    maybe_tune(Key{1, w.rows, w.cols, T, ldd}, p, s, [&](const cublasLtMatmulAlgo_t& al) {
        return cublasLtMatmul(handle_, p.desc, alpha_dev, w.w, p.a, x, p.b, zero_, d, p.d, d, p.d, &al, workspace_, workspace_bytes_, s);
    });
    lt_check(cublasLtMatmul(handle_, p.desc, alpha_dev, w.w, p.a, x, p.b, zero_, d, p.d, d, p.d, &p.algo, workspace_,
                            workspace_bytes_, s), "fp4 matmul");
}

}  // namespace g4
