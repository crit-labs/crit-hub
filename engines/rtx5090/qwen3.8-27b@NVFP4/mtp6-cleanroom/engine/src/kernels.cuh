#pragma once
// Clean-room CUDA kernels for Qwen3.8-27B (Qwen3.5 hybrid architecture) with the
// checkpoint's original NVFP4, row-scaled FP8 and BF16 weight words.
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace q38 {
using bf16 = __nv_bfloat16;

enum WFormat : int { W_FP8 = 0, W_NVFP4 = 1, W_BF16 = 2 };

// One weight matrix [N][K] in its stored encoding.
struct QW {
    int fmt = W_BF16;
    int N = 0, K = 0;
    const std::uint8_t* w = nullptr;  // FP8: N*K bytes, NVFP4: N*K/2 bytes (low nibble = even k), BF16: N*K*2 bytes
    const std::uint8_t* bs = nullptr; // NVFP4 E4M3 block scales [N][K/16]
    const bf16* rs = nullptr;         // FP8 BF16 row scales [N]
    float gs = 1.f;                   // NVFP4: weight_global_scale (divisor)
};

// y[m * ldy + n] = (accumulate ? y : 0) + sum_k W[n][k] x[m * ldx + k] for m < M.
struct GemvSeg {
    QW w;
    float* y = nullptr;
    int ldy = 0;
    int accumulate = 0;
    QW w2{};             // fused SwiGLU: up projection (w is the gate projection)
    bf16* act = nullptr; // fused SwiGLU output act[m * ldy + n] = silu(gate) * up
    int ksplit = 1;          // split-K over blocks: partial s written at y + s * split_stride
    std::size_t split_stride = 0;
};

// Tensor-core GEMV for few columns (any M, eight per launch group). Up to two
// weight segments sharing the same input are executed by one launch.
void gemv(const GemvSeg& a, const GemvSeg* b, const bf16* x, int ldx, int M, cudaStream_t s);
void gemv_multi(const GemvSeg* segs, int nseg, const bf16* x, int ldx, int M, cudaStream_t s);

// Prefill GEMM support: exact BF16 expansion of the stored words (FP8 code or
// NVFP4 code times its E4M3 block scale), then a per-row output scale.
void stream_read(const void* p, std::size_t bytes, void* sink, cudaStream_t s);
void quantize_fp8_rows(const QW& src, uint8_t* q, bf16* rs, cudaStream_t s);
void dequant_bf16(const QW& w, bf16* out, cudaStream_t s);
// y[m*ldy+n] = (acc ? y : 0) + t[m*N+n] * scale(n)
void row_scale_epilogue(const QW& w, const float* t, float* y, int ldy, int M, int accumulate, cudaStream_t s);

void embed(const int* tokens, const bf16* table, float* out, int T, int D, cudaStream_t s);
// out = bf16(rmsnorm(x) * (plus_one ? 1 + w : w))
void rmsnorm(const float* x, int ldx, const bf16* w, bf16* out, int ldo, int T, int D, float eps, int plus_one, cudaStream_t s);
// Same, reading BF16 input.
void rmsnorm_bf(const bf16* x, int ldx, const bf16* w, bf16* out, int ldo, int T, int D, float eps, int plus_one, cudaStream_t s);
void residual_norm(float* resid, const float* partial, int nsplit, std::size_t stride, const bf16* w, bf16* out, int T, int D,
                   float eps, cudaStream_t s);
void swiglu(const float* gu, int ld, bf16* out, int T, int I, cudaStream_t s);

// Gated DeltaNet. mode 0: prefill (commit immediately); mode 1: verify (apply the
// pending accepted count from the previous round, then stash this round's inputs).
struct GdnLayer {
    const bf16* conv_w;   // [10240][4]
    const bf16* A_log;    // [48]
    const bf16* dt_bias;  // [48]
    const bf16* norm_w;   // [128]
    float* conv_hist;     // [3][10240]
    float* conv_stash;    // [Tmax][10240]
    float* state;         // [48][128][128] (k-major)
    float* rec_stash;     // [48*4 warps][Tmax][stash_stride]
};
constexpr int kGdnStash = 164;
void gdn_conv(const GdnLayer& L, const float* qkvz, int ldq, float* out, int T, int mode, const int* pending, cudaStream_t s, int normalize = 0);
// Verify-mode recurrence + gated RMSNorm for T <= 8 columns (q/k already normalized by gdn_conv).
void gdn_decode(const GdnLayer& L, const float* conv_out, const float* ab, int ldab, const float* z, int ldz, bf16* out, int T,
                const int* pending, int tmax, float eps, cudaStream_t s);
void gdn_recurrent(const GdnLayer& L, const float* conv_out, const float* ab, int ldab, float* out, int T, int mode,
                   const int* pending, int tmax, cudaStream_t s);
// out = bf16(rmsnorm(core) * w * silu(z)) per 128-wide value head
void gdn_gate_norm(const float* core, const float* z, int ldz, const bf16* w, bf16* out, int T, float eps, cudaStream_t s);

// Full attention. Positions are *base + t.
void attn_prep(const float* qkv, int ldq, const bf16* qn, const bf16* kn, const int* base, int T,
               float* q_out, bf16* kc, bf16* vc, float eps, cudaStream_t s);
void attention(const float* q, const float* qkv_gate, int ldg, const bf16* kc, const bf16* vc, const int* base, int T,
               int split, int max_splits, float* partial, bf16* out, cudaStream_t s);

void argmax(const float* logits, int ld, int T, int valid, int* out, cudaStream_t s, int eb = 0, int ee = 0);
void log_prob(const float* logits, int ld, int T, int valid, const int* targets, float* out, cudaStream_t s);

// Round helpers.
void verify_tokens(const int* control, const int* drafts, int K, int* tokens, cudaStream_t s);
void set_int(int* dst, const int* src, int add, cudaStream_t s);
void mtp_inputs(const int* tokens, const bf16* embed_table, const bf16* hidden, int ldh, const bf16* we, const bf16* wh,
                bf16* out, int T, float eps, cudaStream_t s);
}
