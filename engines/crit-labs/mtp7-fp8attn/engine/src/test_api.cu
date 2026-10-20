// C entry points for kernel tests driven from Python (torch tensors pass raw device pointers).
#include "gemv.cuh"

using namespace kestrel;
static bool g_sync = true;
extern "C" void kt_set_sync(int v) { g_sync = v != 0; }

extern "C" {

int kt_gemv_fp8(const void* w, const float* scale, int rows, int cols, const void* x, int ldx, int N, int epi,
                float* y, int ldy, void* h, int ldh, int splits, float* partial) {
    try {
        Fp8Mat m{static_cast<const std::uint8_t*>(w), scale, rows, cols};
        GemvOut out{static_cast<GemvEpi>(epi), y, ldy, static_cast<__half*>(h), ldh};
        gemv_fp8(m, static_cast<const __half*>(x), ldx, N, out, splits, partial, 0);
        if(g_sync) KCHECK(cudaDeviceSynchronize());
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

int kt_gemv_fp4(const void* w, const void* sf, float alpha, int rows, int cols, const void* x, int ldx, int N,
                int epi, float* y, int ldy, void* h, int ldh, int splits, float* partial) {
    try {
        Fp4Mat m{static_cast<const std::uint8_t*>(w), static_cast<const std::uint8_t*>(sf), alpha, rows, cols};
        GemvOut out{static_cast<GemvEpi>(epi), y, ldy, static_cast<__half*>(h), ldh};
        gemv_fp4(m, static_cast<const __half*>(x), ldx, N, out, splits, partial, 0);
        if(g_sync) KCHECK(cudaDeviceSynchronize());
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

// Time `iters` launches of a GEMV (events on the default stream); returns ms per launch.
float kt_time_gemv(int fp4, const void* w, const void* scale_or_sf, float alpha, int rows, int cols, const void* x,
                   int N, float* y, int splits, float* partial, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    GemvOut out{GemvEpi::store, y, rows, nullptr, 0};
    auto run = [&] {
        if(fp4) gemv_fp4(Fp4Mat{static_cast<const std::uint8_t*>(w), static_cast<const std::uint8_t*>(scale_or_sf),
                                alpha, rows, cols}, static_cast<const __half*>(x), cols, N, out, splits, partial, 0);
        else gemv_fp8(Fp8Mat{static_cast<const std::uint8_t*>(w), static_cast<const float*>(scale_or_sf), rows, cols},
                      static_cast<const __half*>(x), cols, N, out, splits, partial, 0);
    };
    for(int i = 0; i < 3; ++i) run();
    cudaEventRecord(a);
    for(int i = 0; i < iters; ++i) run();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

}

namespace kestrel { void gemv_set_variant(int); }
extern "C" void kt_set_variant(int v) { kestrel::gemv_set_variant(v); }

#include "fp8_kv_attention.h"
// Time decode attention over nkeys cached positions for T query columns (random cache contents).
#include "attn_q8.cuh"
extern "C" float kt_attn_decode(int nkeys, int T, int blocks, int iters, int q8) {
    using namespace crit::fp8_kv;
    Cache c;
    c.kv_heads = 4;
    c.head_dim = 256;
    const std::size_t n = static_cast<std::size_t>(nkeys + 64) * 4 * 256;
    cudaMalloc(&c.k, n); cudaMalloc(&c.v, n);
    cudaMalloc(&c.k_scale, (nkeys + 64) * 16); cudaMalloc(&c.v_scale, (nkeys + 64) * 16);
    cudaMemset(c.k, 0x30, n); cudaMemset(c.v, 0x30, n);
    cudaMemset(c.k_scale, 0, (nkeys + 64) * 16); cudaMemset(c.v_scale, 0, (nkeys + 64) * 16);
    __nv_bfloat16* q; float* out; float* part;
    cudaMalloc(&q, 8 * 6144 * 2); cudaMemset(q, 0, 8 * 6144 * 2);
    cudaMalloc(&out, 8 * 6144 * 4);
    cudaMalloc(&part, decode_partial_floats(4, 256, blocks) * 4);
    Decode p; p.q = q; p.q_stride = 6144; p.cache = c; p.out = out; p.out_stride = 6144; p.T = T; p.pos0 = nkeys - T;
    p.heads = 24; p.scale = 1.f / 16.f; p.blocks = blocks; p.partial = part;
    std::uint8_t* qc = nullptr; float* qs = nullptr;
    if(q8 == 2) { cudaMalloc(&qc, 8 * 6144); cudaMemset(qc, 0x30, 8 * 6144); cudaMalloc(&qs, 8 * 24 * 4); cudaMemset(qs, 0, 8 * 24 * 4); p.q8 = qc; p.q8_scale = qs; }
    auto run = [&] { if(q8) kestrel::decode_attention_q8(p, 0); else decode(p, 0); };
    for(int i = 0; i < 3; ++i) run();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i = 0; i < iters; ++i) run();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaFree(c.k); cudaFree(c.v); cudaFree(c.k_scale); cudaFree(c.v_scale); cudaFree(q); cudaFree(out); cudaFree(part);
    return ms / iters;
}

// Prompt attention timing: T queries at positions nkeys - T .. nkeys - 1. mode 0 entry, 1 FP8 Q K^T, 2 FP8 Q K^T + P V.
extern "C" float kt_attn_prefill(int nkeys, int T, int iters, int mode) {
    using namespace crit::fp8_kv;
    Cache c;
    c.kv_heads = 4;
    c.head_dim = 256;
    const std::size_t n = static_cast<std::size_t>(nkeys + 64) * 4 * 256;
    cudaMalloc(&c.k, n); cudaMalloc(&c.v, n);
    cudaMalloc(&c.k_scale, (nkeys + 64) * 16); cudaMalloc(&c.v_scale, (nkeys + 64) * 16);
    cudaMemset(c.k, 0x30, n); cudaMemset(c.v, 0x30, n);
    cudaMemset(c.k_scale, 0, (nkeys + 64) * 16); cudaMemset(c.v_scale, 0, (nkeys + 64) * 16);
    __nv_bfloat16* q; float* out;
    cudaMalloc(&q, static_cast<std::size_t>(T) * 6144 * 2); cudaMemset(q, 0, static_cast<std::size_t>(T) * 6144 * 2);
    cudaMalloc(&out, static_cast<std::size_t>(T) * 6144 * 4);
    Prefill p; p.q = q; p.q_stride = 6144; p.cache = c; p.out = out; p.out_stride = 6144; p.T = T; p.pos0 = nkeys - T;
    p.heads = 24; p.scale = 1.f / 16.f;
    auto run = [&] { if(mode) kestrel::prefill_attention_q8(p, 0, mode == 2); else prefill(p, 0); };
    for(int i = 0; i < 2; ++i) run();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i = 0; i < iters; ++i) run();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaFree(c.k); cudaFree(c.v); cudaFree(c.k_scale); cudaFree(c.v_scale); cudaFree(q); cudaFree(out);
    return ms / iters;
}
