// Causal prompt attention over the FP8 KV cache with FP8 tensor cores for S = Q K^T: queries are quantized to E4M3
// per (row, head) in the kernel, keys stay as cache codes (no dequantization), and the product is rescaled by the query
// and per-key scales. P V keeps the BF16 path (values dequantized once per tile). Head dim 256, 128 queries per block
// (8 warps x 16 rows), 32-key tiles prefetched into registers.
#include <cuda_fp8.h>

#include <type_traits>

#include "attn_q8.cuh"
#include "common.h"

namespace kestrel {
namespace kq8 {

constexpr int HD = 256;
constexpr int kQ = 128, kKt = 32, kThreads = 256;
constexpr int kC8 = HD / 16;   // 16-byte FP8 chunks per row
constexpr int kC16 = HD / 8;   // 16-byte BF16 chunks per row

__device__ __forceinline__ int swz8(int row, int chunk) { return row * kC8 + (chunk ^ (row & 7)); }
__device__ __forceinline__ int swz16(int row, int chunk) { return row * kC16 + (chunk ^ (row & 7)); }

__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_fp8(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
                                         std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ std::uint32_t dq_pair(std::uint32_t two, float sc) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&r));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    return *reinterpret_cast<const std::uint32_t*>(&b);
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for(int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

struct Smem8 {
    uint4 q[kQ * kC8];        // FP8 query codes
    uint4 k[kKt * kC8];       // FP8 key codes
    uint4 v[kKt * kC8];       // FP8 value codes
    float qs[kQ];
    float ks[kKt];
    float vs[kKt];
};
struct Smem {
    uint4 q[kQ * kC8];        // FP8 query codes
    uint4 k[kKt * kC8];       // FP8 key codes
    uint4 v[kKt * kC16];      // BF16 values
    float qs[kQ];
    float ks[kKt];
};

// Tile prefetch: 512 FP8 chunks of keys and of values (2 per thread each) plus their scales.
struct Fetch {
    uint4 kr[2], vr[2];
    float kss[2], vss[2];
    __device__ __forceinline__ void load(const crit::fp8_kv::Cache& c, int g, int key0, int nkeys, int tid) {
#pragma unroll
        for(int i = 0; i < 2; ++i) {
            const int e = tid + kThreads * i, key = key0 + e / kC8;
            if(key < nkeys) {
                const std::size_t off = (static_cast<std::size_t>(key) * c.kv_heads + g) * HD + (e % kC8) * 16;
                kr[i] = __ldcg(reinterpret_cast<const uint4*>(c.k + off));
                vr[i] = __ldcg(reinterpret_cast<const uint4*>(c.v + off));
                kss[i] = __ldcg(c.k_scale + static_cast<std::size_t>(key) * c.kv_heads + g);
                vss[i] = __ldcg(c.v_scale + static_cast<std::size_t>(key) * c.kv_heads + g);
            } else {
                kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
                kss[i] = vss[i] = 0.f;
            }
        }
    }
    __device__ __forceinline__ void stage(Smem& S, int tid) const {
#pragma unroll
        for(int i = 0; i < 2; ++i) {
            const int e = tid + kThreads * i, r = e / kC8, cc = e % kC8;
            S.k[swz8(r, cc)] = kr[i];
            S.v[swz16(r, 2 * cc)] = make_uint4(dq_pair(vr[i].x, vss[i]), dq_pair(vr[i].x >> 16, vss[i]), dq_pair(vr[i].y, vss[i]),
                                               dq_pair(vr[i].y >> 16, vss[i]));
            S.v[swz16(r, 2 * cc + 1)] = make_uint4(dq_pair(vr[i].z, vss[i]), dq_pair(vr[i].z >> 16, vss[i]), dq_pair(vr[i].w, vss[i]),
                                                   dq_pair(vr[i].w >> 16, vss[i]));
            if(cc == 0) S.ks[r] = kss[i];
        }
    }
    __device__ __forceinline__ void stage(Smem8& S, int tid) const {
#pragma unroll
        for(int i = 0; i < 2; ++i) {
            const int e = tid + kThreads * i, r = e / kC8, cc = e % kC8;
            S.k[swz8(r, cc)] = kr[i];
            S.v[swz8(r, cc)] = vr[i];
            if(cc == 0) {
                S.ks[r] = kss[i];
                S.vs[r] = vss[i];
            }
        }
    }
};

__device__ __forceinline__ std::uint32_t prmt(std::uint32_t a, std::uint32_t b, std::uint32_t sel) {
    std::uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ std::uint32_t fp8x4(float a, float b, float c, float d) {
    const std::uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const std::uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

template <bool PV8>
__global__ void __launch_bounds__(kThreads, 1) prefill_q8_kernel(crit::fp8_kv::Prefill p) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    using SM = std::conditional_t<PV8, Smem8, Smem>;
    SM& S = *reinterpret_cast<SM*>(smem_raw);
    const int h = blockIdx.y, g = h / (p.heads / p.cache.kv_heads);
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int last_q = min(p.T, qt0 + kQ) - 1;
    const int nkeys = p.pos0 + last_q + 1;
    const int ntiles = (nkeys + kKt - 1) / kKt;
    Fetch f;
    f.load(p.cache, g, 0, nkeys, tid);
    // Quantize this warp's 16 query rows: lane holds 8 of a row's 256 values.
    const int qr0 = warp * 16;
    for(int r = 0; r < 16; ++r) {
        const int row = qr0 + r, t = qt0 + row;
        uint4 u = make_uint4(0u, 0u, 0u, 0u);
        if(t < p.T) u = *reinterpret_cast<const uint4*>(p.q + static_cast<std::size_t>(t) * p.q_stride + static_cast<std::size_t>(h) * HD + lane * 8);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
        float x[8];
        float am = 0.f;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float2 v2 = __bfloat1622float2(b[i]);
            x[2 * i] = v2.x;
            x[2 * i + 1] = v2.y;
            am = fmaxf(am, fmaxf(fabsf(v2.x), fabsf(v2.y)));
        }
        am = warp_max(am);
        const float sc = am > 0.f ? am / 448.f : 1.f;
        std::uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            w0 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
            w1 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[4 + i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
        }
        // Chunk lane / 2 of the row, bytes (lane & 1) * 8.
        uint2* dst = reinterpret_cast<uint2*>(&S.q[swz8(row, lane >> 1)]) + (lane & 1);
        *dst = make_uint2(w0, w1);
        if(lane == 0) S.qs[row] = sc;
    }
    f.stage(S, tid);
    __syncthreads();
    const int qpos0 = p.pos0 + qt0 + qr0 + gl, qpos1 = qpos0 + 8, warp_last = p.pos0 + qt0 + qr0 + 15;
    const float qs0 = S.qs[qr0 + gl] * p.scale, qs1 = S.qs[qr0 + gl + 8] * p.scale;
    float o[HD / 8][4];
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float u = 1.f;  // PV8: o holds the output in units of the current tile's value scale / 448
    for(int j = 0; j < ntiles; ++j) {
        const bool more = j + 1 < ntiles;
        if(more) f.load(p.cache, g, (j + 1) * kKt, nkeys, tid);
        const int kfirst = j * kKt;
        if(kfirst <= warp_last) {
            float s[4][4];
#pragma unroll
            for(int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for(int ks = 0; ks < HD / 32; ++ks) {
                std::uint32_t a[4];
                ldsm_x4(a, &S.q[swz8(qr0 + (lane & 15), ks * 2 + (lane >> 4))]);
#pragma unroll
                for(int np = 0; np < 2; ++np) {
                    std::uint32_t b[4];
                    const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    ldsm_x4(b, &S.k[swz8(key, ks * 2 + ((lane >> 3) & 1))]);
                    mma_fp8(s[2 * np], a, b[0], b[1]);
                    mma_fp8(s[2 * np + 1], a, b[2], b[3]);
                }
            }
#pragma unroll
            for(int n = 0; n < 4; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int kl = n * 8 + 2 * t4 + (i & 1), key = kfirst + kl;
                    s[n][i] = key > ((i >> 1) ? qpos1 : qpos0) ? -INFINITY : s[n][i] * ((i >> 1) ? qs1 : qs0) * S.ks[kl];
                }
            float mx0 = m0, mx1 = m1;
#pragma unroll
            for(int n = 0; n < 4; ++n) { mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1])); mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3])); }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float c0 = mx0 == -INFINITY ? 1.f : __expf(m0 - mx0), c1 = mx1 == -INFINITY ? 1.f : __expf(m1 - mx1);
            m0 = mx0; m1 = mx1;
            if constexpr(PV8) {
                // Tile value scale sigma (largest per-key scale); P is stored as p * vs[key] * 448 / sigma.
                const float sig = warp_max(S.vs[lane]);
                const float un = sig > 0.f ? sig / 448.f : 1.f, f8 = 1.f / un;
                const float ru = u / un;
                u = un;
                float rs0 = 0.f, rs1 = 0.f;
                float e[4][4];
#pragma unroll
                for(int n = 0; n < 4; ++n) {
                    const float vs0 = S.vs[n * 8 + 2 * t4] * f8, vs1 = S.vs[n * 8 + 2 * t4 + 1] * f8;
                    const float e0 = mx0 == -INFINITY ? 0.f : __expf(s[n][0] - mx0), e1 = mx0 == -INFINITY ? 0.f : __expf(s[n][1] - mx0);
                    const float e2 = mx1 == -INFINITY ? 0.f : __expf(s[n][2] - mx1), e3 = mx1 == -INFINITY ? 0.f : __expf(s[n][3] - mx1);
                    rs0 += e0 + e1;
                    rs1 += e2 + e3;
                    e[n][0] = e0 * vs0; e[n][1] = e1 * vs1; e[n][2] = e2 * vs0; e[n][3] = e3 * vs1;
                }
                l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
                const float o0 = c0 * ru, o1 = c1 * ru;
#pragma unroll
                for(int n = 0; n < HD / 8; ++n) { o[n][0] *= o0; o[n][1] *= o0; o[n][2] *= o1; o[n][3] *= o1; }
                // A fragment (k32): k index 4*t4+i <-> tile key {2t4, 2t4+1, 8+2t4, 9+2t4}[i] (+16 for a2/a3).
                std::uint32_t af[4];
                af[0] = fp8x4(e[0][0], e[0][1], e[1][0], e[1][1]);
                af[1] = fp8x4(e[0][2], e[0][3], e[1][2], e[1][3]);
                af[2] = fp8x4(e[2][0], e[2][1], e[3][0], e[3][1]);
                af[3] = fp8x4(e[2][2], e[2][3], e[3][2], e[3][3]);
                const int key = (lane >> 3) * 8 + (lane & 7);
#pragma unroll
                for(int dp = 0; dp < HD / 16; ++dp) {
                    // Transposed 8x8 b16 tiles of keys 0-7 / 8-15 / 16-23 / 24-31 over dims 16dp..16dp+15: register
                    // bytes are (key 2t4: dims 2gl, 2gl+1), (key 2t4+1: dims 2gl, 2gl+1).
                    std::uint32_t r[4];
                    ldsm_x4_t(r, &S.v[swz8(key, dp)]);
                    // Even dims 16dp + 2c (tile 2dp) and odd dims 16dp + 2c + 1 (tile 2dp + 1).
                    mma_fp8(o[2 * dp], af, prmt(r[0], r[1], 0x6420), prmt(r[2], r[3], 0x6420));
                    mma_fp8(o[2 * dp + 1], af, prmt(r[0], r[1], 0x7531), prmt(r[2], r[3], 0x7531));
                }
            } else {
                float rs0 = 0.f, rs1 = 0.f;
                std::uint32_t pf[4][2];
    #pragma unroll
                for(int n = 0; n < 4; ++n) {
                    const float e0 = mx0 == -INFINITY ? 0.f : __expf(s[n][0] - mx0), e1 = mx0 == -INFINITY ? 0.f : __expf(s[n][1] - mx0);
                    const float e2 = mx1 == -INFINITY ? 0.f : __expf(s[n][2] - mx1), e3 = mx1 == -INFINITY ? 0.f : __expf(s[n][3] - mx1);
                    const __nv_bfloat162 p01 = __floats2bfloat162_rn(e0, e1), p23 = __floats2bfloat162_rn(e2, e3);
                    rs0 += __low2float(p01) + __high2float(p01);
                    rs1 += __low2float(p23) + __high2float(p23);
                    pf[n][0] = *reinterpret_cast<const std::uint32_t*>(&p01);
                    pf[n][1] = *reinterpret_cast<const std::uint32_t*>(&p23);
                }
                l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
    #pragma unroll
                for(int n = 0; n < HD / 8; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
    #pragma unroll
                for(int kk = 0; kk < 2; ++kk) {
                    const std::uint32_t a0 = pf[2 * kk][0], a1 = pf[2 * kk][1], a2 = pf[2 * kk + 1][0], a3 = pf[2 * kk + 1][1];
                    const int key = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
    #pragma unroll
                    for(int dp = 0; dp < HD / 16; ++dp) {
                        std::uint32_t b[4];
                        ldsm_x4_t(b, &S.v[swz16(key, dp * 2 + (lane >> 4))]);
                        mma_bf16(o[2 * dp], a0, a1, a2, a3, b[0], b[1]);
                        mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, b[2], b[3]);
                    }
                }
            }
        }
        if(!more) break;
        __syncthreads();
        f.stage(S, tid);
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int t0r = qt0 + qr0 + gl, t1r = t0r + 8;
    if constexpr(PV8) {
        const float j0 = i0 * u, j1 = i1 * u;
#pragma unroll
        for(int dp = 0; dp < HD / 16; ++dp) {
            const int d = dp * 16 + 4 * t4;
            if(t0r < p.T)
                *reinterpret_cast<float4*>(p.out + static_cast<std::size_t>(t0r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                    make_float4(o[2 * dp][0] * j0, o[2 * dp + 1][0] * j0, o[2 * dp][1] * j0, o[2 * dp + 1][1] * j0);
            if(t1r < p.T)
                *reinterpret_cast<float4*>(p.out + static_cast<std::size_t>(t1r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                    make_float4(o[2 * dp][2] * j1, o[2 * dp + 1][2] * j1, o[2 * dp][3] * j1, o[2 * dp + 1][3] * j1);
        }
        return;
    }
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) {
        const int d = n * 8 + 2 * t4;
        if(t0r < p.T)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(t0r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(o[n][0] * i0, o[n][1] * i0);
        if(t1r < p.T)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(t1r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(o[n][2] * i1, o[n][3] * i1);
    }
}


// ------------------------------------------------------------------------------------------------ prompt, FP8 P V
constexpr int kKt2 = 64;
struct Smem64 {
    uint4 q[kQ * kC8];
    uint4 k[2][kKt2 * kC8];
    uint4 v[2][kKt2 * kC8];
    float qs[kQ];
    float ks[2][kKt2];
    float vs[2][kKt2];
};
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp4z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(ok ? 4 : 0));
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// Tile of 64 keys: 1024 FP8 chunks of keys and of values (4 per thread each) and 64 + 64 scales.
__device__ __forceinline__ void fetch64(Smem64& S, int buf, const crit::fp8_kv::Cache& c, int g, int key0, int nkeys, int tid) {
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        const int e = tid + kThreads * i, r = e / kC8, cc = e % kC8, key = key0 + r;
        const bool ok = key < nkeys;
        const std::size_t off = (static_cast<std::size_t>(ok ? key : 0) * c.kv_heads + g) * HD + cc * 16;
        cp16z(&S.k[buf][swz8(r, cc)], c.k + off, ok);
        cp16z(&S.v[buf][swz8(r, cc)], c.v + off, ok);
    }
    if(tid < 2 * kKt2) {
        const int r = tid & (kKt2 - 1), key = key0 + r;
        const bool ok = key < nkeys;
        const std::size_t si = static_cast<std::size_t>(ok ? key : 0) * c.kv_heads + g;
        if(tid < kKt2) cp4z(&S.ks[buf][r], c.k_scale + si, ok);
        else cp4z(&S.vs[buf][r], c.v_scale + si, ok);
    }
    asm volatile("cp.async.commit_group;");
}

__global__ void __launch_bounds__(kThreads, 1) prefill_pv8_kernel(crit::fp8_kv::Prefill p) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem64& S = *reinterpret_cast<Smem64*>(smem_raw);
    const int h = blockIdx.y, g = h / (p.heads / p.cache.kv_heads);
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int last_q = min(p.T, qt0 + kQ) - 1;
    const int nkeys = p.pos0 + last_q + 1;
    const int ntiles = (nkeys + kKt2 - 1) / kKt2;
    fetch64(S, 0, p.cache, g, 0, nkeys, tid);
    const int qr0 = warp * 16;
    for(int r = 0; r < 16; ++r) {
        const int row = qr0 + r, t = qt0 + row;
        uint4 u = make_uint4(0u, 0u, 0u, 0u);
        if(t < p.T) u = *reinterpret_cast<const uint4*>(p.q + static_cast<std::size_t>(t) * p.q_stride + static_cast<std::size_t>(h) * HD + lane * 8);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
        float x[8];
        float am = 0.f;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float2 v2 = __bfloat1622float2(b[i]);
            x[2 * i] = v2.x;
            x[2 * i + 1] = v2.y;
            am = fmaxf(am, fmaxf(fabsf(v2.x), fabsf(v2.y)));
        }
        am = warp_max(am);
        const float sc = am > 0.f ? am / 448.f : 1.f;
        std::uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            w0 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
            w1 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[4 + i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
        }
        uint2* dst = reinterpret_cast<uint2*>(&S.q[swz8(row, lane >> 1)]) + (lane & 1);
        *dst = make_uint2(w0, w1);
        if(lane == 0) S.qs[row] = sc;
    }
    __syncwarp();
    constexpr float kLog2e = 1.4426950408889634f;
    const int qpos0 = p.pos0 + qt0 + qr0 + gl, qpos1 = qpos0 + 8, warp_first = p.pos0 + qt0 + qr0, warp_last = warp_first + 15;
    // Scores in the log2 domain.
    const float qs0 = S.qs[qr0 + gl] * p.scale * kLog2e, qs1 = S.qs[qr0 + gl + 8] * p.scale * kLog2e;
    float o[HD / 8][4];
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float sig = 0.f;  // running maximum value scale; o is in units of sig / 448
    for(int j = 0; j < ntiles; ++j) {
        const int buf = j & 1;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if(j + 1 < ntiles) fetch64(S, buf ^ 1, p.cache, g, (j + 1) * kKt2, nkeys, tid);
        const int kfirst = j * kKt2;
        if(kfirst > warp_last) continue;
        const uint4* K = S.k[buf];
        const uint4* V = S.v[buf];
        const float* ks = S.ks[buf];
        const float* vs = S.vs[buf];
        float s[8][4];
#pragma unroll
        for(int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for(int kc = 0; kc < HD / 32; ++kc) {
            std::uint32_t qa[4];
            ldsm_x4(qa, &S.q[swz8(qr0 + (lane & 15), kc * 2 + (lane >> 4))]);
#pragma unroll
            for(int np = 0; np < 4; ++np) {
                std::uint32_t b[4];
                const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, &K[swz8(key, kc * 2 + ((lane >> 3) & 1))]);
                mma_fp8(s[2 * np], qa, b[0], b[1]);
                mma_fp8(s[2 * np + 1], qa, b[2], b[3]);
            }
        }
        const bool diag = kfirst + kKt2 - 1 > warp_first;
#pragma unroll
        for(int n = 0; n < 8; ++n) {
            const float k0 = ks[n * 8 + 2 * t4], k1 = ks[n * 8 + 2 * t4 + 1];
            s[n][0] *= qs0 * k0; s[n][1] *= qs0 * k1; s[n][2] *= qs1 * k0; s[n][3] *= qs1 * k1;
            if(diag) {
                const int key = kfirst + n * 8 + 2 * t4;
                if(key > qpos0) s[n][0] = -INFINITY;
                if(key + 1 > qpos0) s[n][1] = -INFINITY;
                if(key > qpos1) s[n][2] = -INFINITY;
                if(key + 1 > qpos1) s[n][3] = -INFINITY;
            }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for(int n = 0; n < 8; ++n) { mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1])); mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float tsig = warp_max(fmaxf(vs[lane], vs[lane + 32]));
        const float nsig = fmaxf(sig, tsig);
        const float c0 = mx0 == m0 ? 1.f : (m0 == -INFINITY ? 0.f : ex2(m0 - mx0));
        const float c1 = mx1 == m1 ? 1.f : (m1 == -INFINITY ? 0.f : ex2(m1 - mx1));
        const float ru = nsig == sig ? 1.f : (sig > 0.f ? sig / nsig : 0.f);
        m0 = mx0; m1 = mx1; sig = nsig;
        const float f8 = sig > 0.f ? 448.f / sig : 0.f;
        float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
        for(int n = 0; n < 8; ++n) {
            const float vs0 = vs[n * 8 + 2 * t4] * f8, vs1 = vs[n * 8 + 2 * t4 + 1] * f8;
            const float e0 = mx0 == -INFINITY ? 0.f : ex2(s[n][0] - mx0), e1 = mx0 == -INFINITY ? 0.f : ex2(s[n][1] - mx0);
            const float e2 = mx1 == -INFINITY ? 0.f : ex2(s[n][2] - mx1), e3 = mx1 == -INFINITY ? 0.f : ex2(s[n][3] - mx1);
            rs0 += e0 + e1;
            rs1 += e2 + e3;
            s[n][0] = e0 * vs0; s[n][1] = e1 * vs1; s[n][2] = e2 * vs0; s[n][3] = e3 * vs1;
        }
        l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
        const float o0 = c0 * ru, o1 = c1 * ru;
        if(__any_sync(0xffffffffu, o0 != 1.f || o1 != 1.f)) {
#pragma unroll
            for(int n = 0; n < HD / 8; ++n) { o[n][0] *= o0; o[n][1] *= o0; o[n][2] *= o1; o[n][3] *= o1; }
        }
#pragma unroll
        for(int half = 0; half < 2; ++half) {
            // A fragment (k32): k index 4*t4+i <-> key {2t4, 2t4+1, 8+2t4, 9+2t4}[i] (+16 for a2/a3) of this 32-key half.
            const int b0 = half * 4;
            std::uint32_t af[4];
            af[0] = fp8x4(s[b0][0], s[b0][1], s[b0 + 1][0], s[b0 + 1][1]);
            af[1] = fp8x4(s[b0][2], s[b0][3], s[b0 + 1][2], s[b0 + 1][3]);
            af[2] = fp8x4(s[b0 + 2][0], s[b0 + 2][1], s[b0 + 3][0], s[b0 + 3][1]);
            af[3] = fp8x4(s[b0 + 2][2], s[b0 + 2][3], s[b0 + 3][2], s[b0 + 3][3]);
            const int key = half * 32 + (lane >> 3) * 8 + (lane & 7);
#pragma unroll
            for(int dp = 0; dp < HD / 16; ++dp) {
                std::uint32_t r[4];
                ldsm_x4_t(r, &V[swz8(key, dp)]);
                mma_fp8(o[2 * dp], af, prmt(r[0], r[1], 0x6420), prmt(r[2], r[3], 0x6420));
                mma_fp8(o[2 * dp + 1], af, prmt(r[0], r[1], 0x7531), prmt(r[2], r[3], 0x7531));
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float u = sig / 448.f;
    const float j0 = l0 > 0.f ? u / l0 : 0.f, j1 = l1 > 0.f ? u / l1 : 0.f;
    const int t0r = qt0 + qr0 + gl, t1r = t0r + 8;
#pragma unroll
    for(int dp = 0; dp < HD / 16; ++dp) {
        const int d = dp * 16 + 4 * t4;
        if(t0r < p.T)
            *reinterpret_cast<float4*>(p.out + static_cast<std::size_t>(t0r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float4(o[2 * dp][0] * j0, o[2 * dp + 1][0] * j0, o[2 * dp][1] * j0, o[2 * dp + 1][1] * j0);
        if(t1r < p.T)
            *reinterpret_cast<float4*>(p.out + static_cast<std::size_t>(t1r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float4(o[2 * dp][2] * j1, o[2 * dp + 1][2] * j1, o[2 * dp][3] * j1, o[2 * dp + 1][3] * j1);
    }
}

// ------------------------------------------------------------------------------------------------ decode
// Fixed grid (kv heads x blocks); block b merges 128-key pieces b, b + blocks, ... in key order, like the entry's
// decode, so a column's result does not depend on how many columns run together.
constexpr int kDRows = 64, kDThreads = 128, kPiece = 128;
struct DecSmem {
    uint4 q[kDRows * kC8];
    uint4 k[kKt * kC8];
    uint4 v[kKt * kC16];
    float qs[kDRows];
    float ks[kKt];
};
struct DFetch {
    uint4 kr[4], vr[4];
    float kss[4], vss[4];
    __device__ __forceinline__ void load(const crit::fp8_kv::Cache& c, int g, int key0, int nkeys, int tid) {
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const int e = tid + kDThreads * i, key = key0 + e / kC8;
            if(key < nkeys) {
                const std::size_t off = (static_cast<std::size_t>(key) * c.kv_heads + g) * HD + (e % kC8) * 16;
                kr[i] = __ldcg(reinterpret_cast<const uint4*>(c.k + off));
                vr[i] = __ldcg(reinterpret_cast<const uint4*>(c.v + off));
                kss[i] = __ldcg(c.k_scale + static_cast<std::size_t>(key) * c.kv_heads + g);
                vss[i] = __ldcg(c.v_scale + static_cast<std::size_t>(key) * c.kv_heads + g);
            } else {
                kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
                kss[i] = vss[i] = 0.f;
            }
        }
    }
    __device__ __forceinline__ void stage(DecSmem& S, int tid) const {
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const int e = tid + kDThreads * i, r = e / kC8, cc = e % kC8;
            S.k[swz8(r, cc)] = kr[i];
            S.v[swz16(r, 2 * cc)] = make_uint4(dq_pair(vr[i].x, vss[i]), dq_pair(vr[i].x >> 16, vss[i]), dq_pair(vr[i].y, vss[i]),
                                               dq_pair(vr[i].y >> 16, vss[i]));
            S.v[swz16(r, 2 * cc + 1)] = make_uint4(dq_pair(vr[i].z, vss[i]), dq_pair(vr[i].z >> 16, vss[i]), dq_pair(vr[i].w, vss[i]),
                                                   dq_pair(vr[i].w >> 16, vss[i]));
            if(cc == 0) S.ks[r] = kss[i];
        }
    }
};

__global__ void __launch_bounds__(kDThreads, 2) decode_q8_kernel(crit::fp8_kv::Decode p) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    DecSmem& S = *reinterpret_cast<DecSmem*>(smem_raw);
    const int g = blockIdx.x, blk = blockIdx.y, group = p.heads / p.cache.kv_heads;
    const int pos0 = p.pos0_device ? *p.pos0_device : p.pos0;
    const int nkeys = pos0 + p.T;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    float* part = p.partial + (static_cast<std::size_t>(g) * p.blocks + blk) * kDRows * HD;
    float* ml = p.partial + static_cast<std::size_t>(p.cache.kv_heads) * p.blocks * kDRows * HD +
                (static_cast<std::size_t>(g) * p.blocks + blk) * kDRows * 2;
    auto tile_key = [&](int j) { return (blk + (j >> 2) * p.blocks) * kPiece + (j & 3) * kKt; };
    if(tile_key(0) >= nkeys) {
        for(int r = tid; r < kDRows; r += kDThreads) { ml[r * 2] = -INFINITY; ml[r * 2 + 1] = 0.f; }
        return;
    }
    const int used = p.T * group;
    DFetch f;
    f.load(p.cache, g, tile_key(0), nkeys, tid);
    const int qr0 = warp * 16;
    for(int r = 0; r < 16 && p.q8; ++r) {
        // Pre-quantized queries: lane copies 8 codes of the row.
        const int row = qr0 + r;
        uint2 c = make_uint2(0u, 0u);
        float sc = 1.f;
        if(row < used) {
            const std::size_t qrow = static_cast<std::size_t>(row / group) * p.heads + g * group + row % group;
            c = *reinterpret_cast<const uint2*>(p.q8 + qrow * HD + lane * 8);
            sc = p.q8_scale[qrow];
        }
        *(reinterpret_cast<uint2*>(&S.q[swz8(row, lane >> 1)]) + (lane & 1)) = c;
        if(lane == 0) S.qs[row] = sc;
    }
    for(int r = 0; r < 16 && !p.q8; ++r) {
        const int row = qr0 + r;
        uint4 u = make_uint4(0u, 0u, 0u, 0u);
        if(row < used)
            u = *reinterpret_cast<const uint4*>(p.q + static_cast<std::size_t>(row / group) * p.q_stride +
                                                static_cast<std::size_t>(g * group + row % group) * HD + lane * 8);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
        float x[8];
        float am = 0.f;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float2 v2 = __bfloat1622float2(b[i]);
            x[2 * i] = v2.x;
            x[2 * i + 1] = v2.y;
            am = fmaxf(am, fmaxf(fabsf(v2.x), fabsf(v2.y)));
        }
        am = warp_max(am);
        const float sc = am > 0.f ? am / 448.f : 1.f;
        std::uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            w0 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
            w1 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[4 + i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
        }
        uint2* dst = reinterpret_cast<uint2*>(&S.q[swz8(row, lane >> 1)]) + (lane & 1);
        *dst = make_uint2(w0, w1);
        if(lane == 0) S.qs[row] = sc;
    }
    f.stage(S, tid);
    __syncthreads();
    const int r0 = qr0 + gl, r1 = r0 + 8;
    const int qpos0 = r0 < used ? pos0 + r0 / group : -1, qpos1 = r1 < used ? pos0 + r1 / group : -1;
    const float qs0 = S.qs[r0] * p.scale, qs1 = S.qs[r1] * p.scale;
    float o[HD / 8][4];
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for(int j = 0;; ++j) {
        const int kfirst = tile_key(j), knext = tile_key(j + 1);
        const bool more = knext < nkeys;
        if(more) f.load(p.cache, g, knext, nkeys, tid);
        {
            float s[4][4];
#pragma unroll
            for(int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for(int ks = 0; ks < HD / 32; ++ks) {
                std::uint32_t a[4];
                ldsm_x4(a, &S.q[swz8(qr0 + (lane & 15), ks * 2 + (lane >> 4))]);
#pragma unroll
                for(int np = 0; np < 2; ++np) {
                    std::uint32_t b[4];
                    const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    ldsm_x4(b, &S.k[swz8(key, ks * 2 + ((lane >> 3) & 1))]);
                    mma_fp8(s[2 * np], a, b[0], b[1]);
                    mma_fp8(s[2 * np + 1], a, b[2], b[3]);
                }
            }
#pragma unroll
            for(int n = 0; n < 4; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int kl = n * 8 + 2 * t4 + (i & 1), key = kfirst + kl;
                    s[n][i] = key > ((i >> 1) ? qpos1 : qpos0) ? -INFINITY : s[n][i] * ((i >> 1) ? qs1 : qs0) * S.ks[kl];
                }
            float mx0 = m0, mx1 = m1;
#pragma unroll
            for(int n = 0; n < 4; ++n) { mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1])); mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3])); }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float c0 = mx0 == -INFINITY ? 1.f : __expf(m0 - mx0), c1 = mx1 == -INFINITY ? 1.f : __expf(m1 - mx1);
            m0 = mx0; m1 = mx1;
            float rs0 = 0.f, rs1 = 0.f;
            std::uint32_t pf[4][2];
#pragma unroll
            for(int n = 0; n < 4; ++n) {
                const float e0 = mx0 == -INFINITY ? 0.f : __expf(s[n][0] - mx0), e1 = mx0 == -INFINITY ? 0.f : __expf(s[n][1] - mx0);
                const float e2 = mx1 == -INFINITY ? 0.f : __expf(s[n][2] - mx1), e3 = mx1 == -INFINITY ? 0.f : __expf(s[n][3] - mx1);
                const __nv_bfloat162 p01 = __floats2bfloat162_rn(e0, e1), p23 = __floats2bfloat162_rn(e2, e3);
                rs0 += __low2float(p01) + __high2float(p01);
                rs1 += __low2float(p23) + __high2float(p23);
                pf[n][0] = *reinterpret_cast<const std::uint32_t*>(&p01);
                pf[n][1] = *reinterpret_cast<const std::uint32_t*>(&p23);
            }
            l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
#pragma unroll
            for(int n = 0; n < HD / 8; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
#pragma unroll
            for(int kk = 0; kk < 2; ++kk) {
                const std::uint32_t a0 = pf[2 * kk][0], a1 = pf[2 * kk][1], a2 = pf[2 * kk + 1][0], a3 = pf[2 * kk + 1][1];
                const int key = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
                for(int dp = 0; dp < HD / 16; ++dp) {
                    std::uint32_t b[4];
                    ldsm_x4_t(b, &S.v[swz16(key, dp * 2 + (lane >> 4))]);
                    mma_bf16(o[2 * dp], a0, a1, a2, a3, b[0], b[1]);
                    mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, b[2], b[3]);
                }
            }
        }
        if(!more) break;
        __syncthreads();
        f.stage(S, tid);
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) {
        const int dd = n * 8 + 2 * t4;
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r0) * HD + dd) = make_float2(o[n][0] * i0, o[n][1] * i0);
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r1) * HD + dd) = make_float2(o[n][2] * i1, o[n][3] * i1);
    }
    if(t4 == 0) {
        ml[r0 * 2] = m0; ml[r0 * 2 + 1] = l0;
        ml[r1 * 2] = m1; ml[r1 * 2 + 1] = l1;
    }
}

}  // namespace kq8

void decode_attention_q8(const crit::fp8_kv::Decode& p, cudaStream_t s) {
    if(p.cache.head_dim != kq8::HD || p.T * (p.heads / p.cache.kv_heads) > kq8::kDRows)
        throw std::invalid_argument("decode_attention_q8: head_dim 256 and at most 64 rows");
    const int bytes = static_cast<int>(sizeof(kq8::DecSmem));
    KCHECK(cudaFuncSetAttribute(kq8::decode_q8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    launch_k(kq8::decode_q8_kernel, dim3(p.cache.kv_heads, p.blocks), kq8::kDThreads, bytes, s, p);
    KCHECK(crit::fp8_kv::combine(p, s));
}

void prefill_attention_q8(const crit::fp8_kv::Prefill& p, cudaStream_t s, bool pv8) {
    if(p.cache.head_dim != kq8::HD) throw std::invalid_argument("prefill_attention_q8: head_dim 256 only");
    const dim3 grid((p.T + kq8::kQ - 1) / kq8::kQ, p.heads);
    if(pv8) {
        const int bytes = static_cast<int>(sizeof(kq8::Smem64));
        KCHECK(cudaFuncSetAttribute(kq8::prefill_pv8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        launch_k(kq8::prefill_pv8_kernel, grid, kq8::kThreads, bytes, s, p);
        return;
    }
    const int bytes = static_cast<int>(sizeof(kq8::Smem));
    KCHECK(cudaFuncSetAttribute(kq8::prefill_q8_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    launch_k(kq8::prefill_q8_kernel<false>, grid, kq8::kThreads, bytes, s, p);
}

}  // namespace kestrel
