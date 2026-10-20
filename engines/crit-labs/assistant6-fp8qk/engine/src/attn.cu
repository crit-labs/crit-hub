// g4 attention kernels; see attn.cuh.
#include "attn.cuh"

#include <cuda_fp8.h>

namespace g4 {
namespace kattn {

__device__ __forceinline__ std::uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const std::uint32_t*>(&r);
}
__device__ __forceinline__ float2 fp8x2(std::uint32_t two) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
    return __half22float2(*reinterpret_cast<const __half2*>(&r));
}
__device__ __forceinline__ std::uint32_t pair(std::uint32_t two, float sc) {
    const float2 f = fp8x2(two);
    return pack_bf2(f.x * sc, f.y * sc);
}
// Eight codes (two words, first code in the low byte) times the scale -> eight BF16 values.
__device__ __forceinline__ uint4 dequant8(std::uint32_t w0, std::uint32_t w1, float sc) {
    return make_uint4(pair(w0, sc), pair(w0 >> 16, sc), pair(w1, sc), pair(w1 >> 16, sc));
}
// Sixteen codes -> FP32 values times the scale.
__device__ __forceinline__ void dequant16(const uint4& u, float sc, float (&x)[16]) {
    const std::uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        const float2 a = fp8x2(w[i]), b = fp8x2(w[i] >> 16);
        x[4 * i] = a.x * sc; x[4 * i + 1] = a.y * sc; x[4 * i + 2] = b.x * sc; x[4 * i + 3] = b.y * sc;
    }
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(std::uint32_t (&r)[2], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
                                         std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}
// Chunk c (8 BF16) of row r in a row of C chunks, 16-byte XOR swizzle.
template <int C>
__device__ __forceinline__ int swz(int row, int chunk) { return row * C + (chunk ^ (row & 7)); }

// Allowed keys of a query row: lo < key <= hi. Unused rows have hi = -1.
struct Span {
    int lo0, hi0, lo1, hi1;  // rows g and g + 8 of the warp's 16
};

// Online softmax over one tile of NT8 * 8 keys for a warp's 16 rows, given raw scores s[NT8][4] (fragment rows g
// (i < 2) and g + 8, keys kfirst + n * 8 + 2 * t4 + (i & 1)). Writes BF16 probabilities to pf and rescales o by the
// returned factors. A tile with no allowed key for a row leaves that row's state bit for bit unchanged.
template <int NT8, int ON>
struct Softmax {
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    __device__ __forceinline__ void update(float (&s)[NT8][4], int kfirst, const Span& sp, int lane, float (&o)[ON][4],
                                           std::uint32_t (&pf)[NT8][2]) {
        const int t4 = lane & 3;
#pragma unroll
        for(int n = 0; n < NT8; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int key = kfirst + n * 8 + 2 * t4 + (i & 1);
                const bool ok = (i >> 1) ? (key <= sp.hi1 && key > sp.lo1) : (key <= sp.hi0 && key > sp.lo0);
                if(!ok) s[n][i] = -INFINITY;
            }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for(int n = 0; n < NT8; ++n) { mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1])); mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float c0 = mx0 == -INFINITY ? 1.f : __expf(m0 - mx0), c1 = mx1 == -INFINITY ? 1.f : __expf(m1 - mx1);
        m0 = mx0; m1 = mx1;
        float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
        for(int n = 0; n < NT8; ++n) {
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
        for(int n = 0; n < ON; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
    }
    __device__ __forceinline__ void finish() {
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    }
};

// ================================================================================================ sliding (HD 256)
constexpr int kSK = 32;              // keys per tile
constexpr int kSC = kSHd / 8;        // 32 BF16 chunks per row
constexpr int kSPiece = 4 * kSK;     // keys per decode piece
constexpr int kDecRows = 64;         // decode rows per key head (columns x query heads of the group)

template <int NT>
struct SlideTile {
    static constexpr int CH = kSK * kSHd / 16;  // 16-byte FP8 chunks per tile (512)
    static constexpr int PER = (CH + NT - 1) / NT;
    uint4 kr[PER], vr[PER];
    float ks[PER], vs[PER];
    __device__ __forceinline__ void fetch(const SlideCache& c, int g, int key0, int klo, int khi, int tid) {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i, key = key0 + e / (kSHd / 16);
            if(e < CH && key >= klo && key < khi) {
                const std::size_t slot = static_cast<std::size_t>(key % c.ring);
                const std::size_t off = (slot * kSKv + g) * kSHd + (e % (kSHd / 16)) * 16;
                kr[i] = __ldcg(reinterpret_cast<const uint4*>(c.k + off));
                vr[i] = __ldcg(reinterpret_cast<const uint4*>(c.v + off));
                ks[i] = __ldcg(c.ks + slot * kSKv + g);
                vs[i] = __ldcg(c.vs + slot * kSKv + g);
            } else {
                kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
                ks[i] = vs[i] = 0.f;
            }
        }
    }
    __device__ __forceinline__ void stage(uint4* Ks, uint4* Vs, int tid) const {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i;
            if(e >= CH) break;
            const int r = e / (kSHd / 16), cc = e % (kSHd / 16);
            Ks[swz<kSC>(r, 2 * cc)] = dequant8(kr[i].x, kr[i].y, ks[i]);
            Ks[swz<kSC>(r, 2 * cc + 1)] = dequant8(kr[i].z, kr[i].w, ks[i]);
            Vs[swz<kSC>(r, 2 * cc)] = dequant8(vr[i].x, vr[i].y, vs[i]);
            Vs[swz<kSC>(r, 2 * cc + 1)] = dequant8(vr[i].z, vr[i].w, vs[i]);
        }
    }
};

// A warp's 16 rows against one staged 32-key tile (HD 256 for both QK and PV).
struct SlideRows {
    float o[kSHd / 8][4];
    Softmax<4, kSHd / 8> sm;
    __device__ __forceinline__ SlideRows() {
#pragma unroll
        for(int n = 0; n < kSHd / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    }
    __device__ __forceinline__ void step(const uint4* Qs, const uint4* Ks, const uint4* Vs, int qr0, int kfirst, const Span& sp,
                                         int lane) {
        float s[4][4];
#pragma unroll
        for(int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for(int kq = 0; kq < kSHd / 16; ++kq) {
            std::uint32_t a[4];
            ldsm_x4(a, Qs + swz<kSC>(qr0 + (lane & 15), kq * 2 + (lane >> 4)));
#pragma unroll
            for(int np = 0; np < 2; ++np) {
                std::uint32_t b[4];
                const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, Ks + swz<kSC>(key, kq * 2 + ((lane >> 3) & 1)));
                mma_bf16(s[2 * np], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_bf16(s[2 * np + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
        std::uint32_t pf[4][2];
        sm.update(s, kfirst, sp, lane, o, pf);
#pragma unroll
        for(int kk = 0; kk < 2; ++kk) {
            const std::uint32_t a0 = pf[2 * kk][0], a1 = pf[2 * kk][1], a2 = pf[2 * kk + 1][0], a3 = pf[2 * kk + 1][1];
            const int key = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for(int dp = 0; dp < kSHd / 16; ++dp) {
                std::uint32_t b[4];
                ldsm_x4_t(b, Vs + swz<kSC>(key, dp * 2 + (lane >> 4)));
                mma_bf16(o[2 * dp], a0, a1, a2, a3, b[0], b[1]);
                mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
    }
};

__device__ __forceinline__ Span row_span(int qpos0, int qpos1, int shift, int window) {
    Span sp;
    sp.hi0 = qpos0 < 0 ? -1 : qpos0 - shift;
    sp.hi1 = qpos1 < 0 ? -1 : qpos1 - shift;
    sp.lo0 = window > 0 ? sp.hi0 - window : -1;
    sp.lo1 = window > 0 ? sp.hi1 - window : -1;
    if(qpos0 < 0) sp.lo0 = 0;
    if(qpos1 < 0) sp.lo1 = 0;
    return sp;
}

constexpr int kSQRows = 128, kSPrefillThreads = 256;

__global__ void __launch_bounds__(kSPrefillThreads, 1) slide_prefill_kernel(AttnRows a, SlideCache c) {
    pdl_wait();
    pdl_trigger();
    using T_ = SlideTile<kSPrefillThreads>;
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    uint4* Ks = Qs + kSQRows * kSC;
    uint4* Vs = Ks + kSK * kSC;
    const int h = blockIdx.y, g = h / (kHeads / kSKv);
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kSQRows;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int last_q = min(a.T, qt0 + kSQRows) - 1;
    const int khi = a.pos0 + last_q - a.shift + 1;
    const int klo = max(0, a.pos0 + qt0 - a.shift - kWindow + 1);
    const int kstart = klo / kSK * kSK;
    for(int e = tid; e < kSQRows * kSC; e += kSPrefillThreads) {
        const int r = e / kSC, cc = e % kSC;
        uint4* dst = Qs + swz<kSC>(r, cc);
        if(qt0 + r < a.T) cp_async16(dst, a.q + static_cast<std::size_t>(qt0 + r) * a.q_stride + static_cast<std::size_t>(h) * kSHd + cc * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    T_ tile;
    tile.fetch(c, g, kstart, klo, khi, tid);
    tile.stage(Ks, Vs, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    SlideRows rows;
    const int qr0 = warp * 16;
    const int r0 = qt0 + qr0 + gl, r1 = r0 + 8;
    const Span sp = row_span(r0 < a.T ? a.pos0 + r0 : -1, r1 < a.T ? a.pos0 + r1 : -1, a.shift, kWindow);
    const int warp_hi = a.pos0 + qt0 + qr0 + 15 - a.shift, warp_lo = a.pos0 + qt0 + qr0 - a.shift - kWindow;
    for(int key0 = kstart;; key0 += kSK) {
        const bool more = key0 + kSK < khi;
        if(more) tile.fetch(c, g, key0 + kSK, klo, khi, tid);
        if(key0 <= warp_hi && key0 + kSK - 1 > warp_lo && qt0 + qr0 < a.T) rows.step(Qs, Ks, Vs, qr0, key0, sp, lane);
        if(!more) break;
        __syncthreads();
        tile.stage(Ks, Vs, tid);
        __syncthreads();
    }
    rows.sm.finish();
    const float i0 = rows.sm.l0 > 0.f ? 1.f / rows.sm.l0 : 0.f, i1 = rows.sm.l1 > 0.f ? 1.f / rows.sm.l1 : 0.f;
#pragma unroll
    for(int n = 0; n < kSHd / 8; ++n) {
        const int d = n * 8 + 2 * t4;
        if(r0 < a.T)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(r0) * a.out_stride + static_cast<std::size_t>(h) * kSHd + d) =
                make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        if(r1 < a.T)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(r1) * a.out_stride + static_cast<std::size_t>(h) * kSHd + d) =
                make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
}

constexpr int kSDecThreads = 128;

// Block (key head g, block blk) merges the 128-key pieces p = blk, blk + blocks, ... (absolute key / 128) that hold keys
// in [klo, khi), in key order.
__global__ void __launch_bounds__(kSDecThreads, 1) slide_decode_kernel(AttnRows a, SlideCache c, int blocks, float* partial) {
    pdl_wait();
    pdl_trigger();
    using T_ = SlideTile<kSDecThreads>;
    constexpr int group = kHeads / kSKv;  // 2
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    uint4* Ks = Qs + kDecRows * kSC;
    uint4* Vs = Ks + kSK * kSC;
    const int g = blockIdx.x, blk = blockIdx.y;
    const int pos0 = a.pos0_dev ? *a.pos0_dev : a.pos0;
    const int khi = pos0 + a.T - a.shift;
    const int klo = max(0, pos0 - a.shift - kWindow + 1);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    float* part = partial + (static_cast<std::size_t>(g) * blocks + blk) * kDecRows * kSHd;
    float* ml = partial + static_cast<std::size_t>(kSKv) * blocks * kDecRows * kSHd + (static_cast<std::size_t>(g) * blocks + blk) * kDecRows * 2;
    const int pi0 = klo / kSPiece;
    int first = pi0 - pi0 % blocks + blk;
    if(first < pi0) first += blocks;
    auto tile_key = [&](int j) { return (first + (j >> 2) * blocks) * kSPiece + (j & 3) * kSK; };
    if(tile_key(0) >= khi) {
        for(int r = tid; r < kDecRows; r += kSDecThreads) { ml[r * 2] = -INFINITY; ml[r * 2 + 1] = 0.f; }
        return;
    }
    const int used = a.T * group;
    for(int e = tid; e < kDecRows * kSC; e += kSDecThreads) {
        const int r = e / kSC, cc = e % kSC;
        uint4* dst = Qs + swz<kSC>(r, cc);
        if(r < used) cp_async16(dst, a.q + static_cast<std::size_t>(r / group) * a.q_stride + static_cast<std::size_t>(g * group + r % group) * kSHd + cc * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    T_ tile;
    tile.fetch(c, g, tile_key(0), klo, khi, tid);
    tile.stage(Ks, Vs, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    SlideRows rows;
    const int qr0 = warp * 16, r0 = qr0 + gl, r1 = r0 + 8;
    const Span sp = row_span(r0 < used ? pos0 + r0 / group : -1, r1 < used ? pos0 + r1 / group : -1, a.shift, kWindow);
    for(int j = 0;; ++j) {
        const int kfirst = tile_key(j), knext = tile_key(j + 1);
        const bool more = knext < khi;
        if(more) tile.fetch(c, g, knext, klo, khi, tid);
        if(qr0 < used) rows.step(Qs, Ks, Vs, qr0, kfirst, sp, lane);
        if(!more) break;
        __syncthreads();
        tile.stage(Ks, Vs, tid);
        __syncthreads();
    }
    rows.sm.finish();
    const float i0 = rows.sm.l0 > 0.f ? 1.f / rows.sm.l0 : 0.f, i1 = rows.sm.l1 > 0.f ? 1.f / rows.sm.l1 : 0.f;
#pragma unroll
    for(int n = 0; n < kSHd / 8; ++n) {
        const int dd = n * 8 + 2 * t4;
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r0) * kSHd + dd) = make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r1) * kSHd + dd) = make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
    if(t4 == 0) {
        ml[r0 * 2] = rows.sm.m0; ml[r0 * 2 + 1] = rows.sm.l0;
        ml[r1 * 2] = rows.sm.m1; ml[r1 * 2 + 1] = rows.sm.l1;
    }
}

// Block (column t, query head h), one thread per dimension: block weights l_b exp(m_b - M), summed in block order.
template <int HD, int KV>
__global__ void __launch_bounds__(HD) combine_kernel(AttnRows a, int blocks, const float* __restrict__ partial) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ float wsp[];
    __shared__ float total;
    constexpr int group = kHeads / KV;
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, g = h / group;
    const int r = t * group + h % group;
    const float* part = partial + static_cast<std::size_t>(g) * blocks * kDecRows * HD;
    const float* ml = partial + static_cast<std::size_t>(KV) * blocks * kDecRows * HD + static_cast<std::size_t>(g) * blocks * kDecRows * 2;
    for(int b = d; b < blocks; b += blockDim.x) wsp[b] = ml[(static_cast<std::size_t>(b) * kDecRows + r) * 2];
    __syncthreads();
    if(d == 0) {
        float M = -INFINITY;
        for(int b = 0; b < blocks; ++b) M = fmaxf(M, wsp[b]);
        float den = 0.f;
        for(int b = 0; b < blocks; ++b) {
            const float m = wsp[b], l = ml[(static_cast<std::size_t>(b) * kDecRows + r) * 2 + 1];
            const float w = m == -INFINITY ? 0.f : l * __expf(m - M);
            wsp[b] = w;
            den += w;
        }
        total = den;
    }
    __syncthreads();
    float num = 0.f;
    for(int b = 0; b < blocks; ++b) {
        const float w = wsp[b];
        if(w != 0.f) num = fmaf(w, part[(static_cast<std::size_t>(b) * kDecRows + r) * HD + d], num);
    }
    a.out[static_cast<std::size_t>(t) * a.out_stride + static_cast<std::size_t>(h) * HD + d] = total > 0.f ? num / total : 0.f;
}

// ================================================================================================ global (HD 512)
constexpr int kGK = 16;              // keys per tile
constexpr int kGC = kGHd / 8;        // 64 BF16 chunks per row
constexpr int kGPiece = 4 * kGK;     // keys per decode piece
constexpr int kGThreads = 256;       // 8 warps: 4 row groups x 2 output halves
constexpr int kGRows = 64;

constexpr int kGRc = 2 * kGRopePairs / 8;  // 16 BF16 chunks of rotated key dims per row (dims 0..63 and 256..319)

// One 16-key tile of one key head: thread tid owns key tid / 16 and the dimension chunks c and c + 16 (c = tid % 16),
// 16 codes each, so the RoPE pairs (d, d + 256) of chunks c < 4 meet in one thread. Staged: Vs = u (all 512 dims),
// Kr = RoPE(u * w) of the 128 rotated dims; the other key dims are u * w, and w is folded into the queries there.
struct GlobalTile {
    uint4 lo, hi;
    float sc;
    int key;
    __device__ __forceinline__ void fetch(const GlobalCache& c, int g, int key0, int khi, int tid) {
        key = key0 + tid / 16;
        const int cc = tid % 16;
        if(key < khi) {
            const std::size_t base = (static_cast<std::size_t>(key) * kGKv + g) * kGHd;
            lo = __ldcg(reinterpret_cast<const uint4*>(c.u + base + cc * 16));
            hi = __ldcg(reinterpret_cast<const uint4*>(c.u + base + 256 + cc * 16));
            sc = __ldcg(c.us + static_cast<std::size_t>(key) * kGKv + g);
        } else {
            lo = hi = make_uint4(0u, 0u, 0u, 0u);
            sc = 0.f;
            key = 0;  // a valid table row; the values are zero
        }
    }
    __device__ __forceinline__ void stage(const GlobalCache& c, uint4* Kr, uint4* Vs, int tid) const {
        const int r = tid / 16, cc = tid % 16;
        float xl[16], xh[16];
        dequant16(lo, sc, xl);
        dequant16(hi, sc, xh);
        Vs[swz<kGC>(r, 2 * cc)] = make_uint4(pack_bf2(xl[0], xl[1]), pack_bf2(xl[2], xl[3]), pack_bf2(xl[4], xl[5]), pack_bf2(xl[6], xl[7]));
        Vs[swz<kGC>(r, 2 * cc + 1)] = make_uint4(pack_bf2(xl[8], xl[9]), pack_bf2(xl[10], xl[11]), pack_bf2(xl[12], xl[13]), pack_bf2(xl[14], xl[15]));
        Vs[swz<kGC>(r, 32 + 2 * cc)] = make_uint4(pack_bf2(xh[0], xh[1]), pack_bf2(xh[2], xh[3]), pack_bf2(xh[4], xh[5]), pack_bf2(xh[6], xh[7]));
        Vs[swz<kGC>(r, 32 + 2 * cc + 1)] = make_uint4(pack_bf2(xh[8], xh[9]), pack_bf2(xh[10], xh[11]), pack_bf2(xh[12], xh[13]), pack_bf2(xh[14], xh[15]));
        if(cc >= kGRopePairs / 16) return;
        const float4* w4l = reinterpret_cast<const float4*>(c.kw + cc * 16);
        const float4* w4h = reinterpret_cast<const float4*>(c.kw + 256 + cc * 16);
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float4 a = w4l[i], b = w4h[i];
            xl[4 * i] *= a.x; xl[4 * i + 1] *= a.y; xl[4 * i + 2] *= a.z; xl[4 * i + 3] *= a.w;
            xh[4 * i] *= b.x; xh[4 * i + 1] *= b.y; xh[4 * i + 2] *= b.z; xh[4 * i + 3] *= b.w;
        }
        const float4* t4 = reinterpret_cast<const float4*>(c.rope + static_cast<std::size_t>(key) * kGRopePairs + cc * 16);
#pragma unroll
        for(int j = 0; j < 8; ++j) {
            const float4 t = __ldg(t4 + j);  // (cos, sin) of pairs 2j and 2j + 1
            float a = xl[2 * j], b = xh[2 * j];
            xl[2 * j] = a * t.x - b * t.y;
            xh[2 * j] = b * t.x + a * t.y;
            a = xl[2 * j + 1];
            b = xh[2 * j + 1];
            xl[2 * j + 1] = a * t.z - b * t.w;
            xh[2 * j + 1] = b * t.z + a * t.w;
        }
        Kr[swz<kGRc>(r, 2 * cc)] = make_uint4(pack_bf2(xl[0], xl[1]), pack_bf2(xl[2], xl[3]), pack_bf2(xl[4], xl[5]), pack_bf2(xl[6], xl[7]));
        Kr[swz<kGRc>(r, 2 * cc + 1)] = make_uint4(pack_bf2(xl[8], xl[9]), pack_bf2(xl[10], xl[11]), pack_bf2(xl[12], xl[13]), pack_bf2(xl[14], xl[15]));
        Kr[swz<kGRc>(r, 8 + 2 * cc)] = make_uint4(pack_bf2(xh[0], xh[1]), pack_bf2(xh[2], xh[3]), pack_bf2(xh[4], xh[5]), pack_bf2(xh[6], xh[7]));
        Kr[swz<kGRc>(r, 8 + 2 * cc + 1)] = make_uint4(pack_bf2(xh[8], xh[9]), pack_bf2(xh[10], xh[11]), pack_bf2(xh[12], xh[13]), pack_bf2(xh[14], xh[15]));
    }
};

// A warp pair's 16 rows against one staged 16-key tile. Warp `half` holds its rows' queries for dims half * 256 .. + 255
// in registers (loaded once) and computes the partial S of all 16 keys over those dims; the pair adds its partials
// through shared memory in a fixed order (dims 0..255 first), then both run the same softmax and P V for their 256 dims.
struct GlobalRows {
    float o[16 * 2][4];
    std::uint32_t qa[16][4];
    Softmax<2, 32> sm;
    __device__ __forceinline__ GlobalRows() {
#pragma unroll
        for(int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    }
    __device__ __forceinline__ void load_q(const uint4* Qs, int qr0, int half, int lane) {
#pragma unroll
        for(int kq = 0; kq < 16; ++kq) ldsm_x4(qa[kq], Qs + swz<kGC>(qr0 + (lane & 15), (half * 16 + kq) * 2 + (lane >> 4)));
    }
    __device__ __forceinline__ void step(const uint4* Kr, const uint4* Vs, float4* Sx, int rg, int half, int kfirst,
                                         const Span& sp, int lane) {
        float p[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const int key = (lane & 7) + ((lane >> 4) << 3), sub = (lane >> 3) & 1;
#pragma unroll
        for(int kq = 0; kq < 16; ++kq) {
            const int gq = half * 16 + kq;
            std::uint32_t bb[4];
            const uint4* src;
            if(kq < 4) src = Kr + swz<kGRc>(key, half * 8 + kq * 2 + sub);
            else src = Vs + swz<kGC>(key, gq * 2 + sub);
            ldsm_x4(bb, src);
            mma_bf16(p[0], qa[kq][0], qa[kq][1], qa[kq][2], qa[kq][3], bb[0], bb[1]);
            mma_bf16(p[1], qa[kq][0], qa[kq][1], qa[kq][2], qa[kq][3], bb[2], bb[3]);
        }
        float4* mine = Sx + ((rg * 2 + half) * 32 + lane) * 2;
        mine[0] = make_float4(p[0][0], p[0][1], p[0][2], p[0][3]);
        mine[1] = make_float4(p[1][0], p[1][1], p[1][2], p[1][3]);
        asm volatile("bar.sync %0, 64;" :: "r"(1 + rg) : "memory");
        const float4* other = Sx + ((rg * 2 + (half ^ 1)) * 32 + lane) * 2;
        const float4 o0 = other[0], o1 = other[1];
        const float q0[4] = {o0.x, o0.y, o0.z, o0.w}, q1[4] = {o1.x, o1.y, o1.z, o1.w};
        float s[2][4];
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            s[0][i] = half ? q0[i] + p[0][i] : p[0][i] + q0[i];
            s[1][i] = half ? q1[i] + p[1][i] : p[1][i] + q1[i];
        }
        std::uint32_t pf[2][2];
        sm.update(s, kfirst, sp, lane, o, pf);
        const std::uint32_t a0 = pf[0][0], a1 = pf[0][1], a2 = pf[1][0], a3 = pf[1][1];
        const int vkey = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
        for(int dp = 0; dp < 16; ++dp) {
            std::uint32_t bv[4];
            ldsm_x4_t(bv, Vs + swz<kGC>(vkey, half * 32 + dp * 2 + (lane >> 4)));
            mma_bf16(o[2 * dp], a0, a1, a2, a3, bv[0], bv[1]);
            mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, bv[2], bv[3]);
        }
    }
};

// Prompt attention, one block per (key head g, 8 query positions): rows r = t * 8 + j are position qt0 + t of query head
// g * 8 + j, so every staged key tile serves all 8 query heads of its key head.
static_assert(2 * (kGK * kGRc + kGK * kGC) <= kGRows * kGC, "two tile buffers fit the query area");
__global__ void __launch_bounds__(kGThreads, 1) global_prefill_kernel(AttnRows a, GlobalCache c) {
    pdl_wait();
    pdl_trigger();
    constexpr int group = kHeads / kGKv;  // 8
    constexpr int kPos = kGRows / group;  // 8 positions per block
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    constexpr int kBuf = kGK * kGRc + kGK * kGC;  // one tile: Kr then Vs
    uint4* Kr[2] = {smem, smem + kBuf};
    uint4* Vs[2] = {smem + kGK * kGRc, smem + kBuf + kGK * kGRc};
    float4* Sx = reinterpret_cast<float4*>(smem + kGRows * kGC);
    const int g = blockIdx.y;
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kPos;  // the latest (heaviest) positions first
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int rg = warp & 3, half = warp >> 2;
    const int npos = min(kPos, a.T - qt0);
    const int used = npos * group;
    const int khi = a.pos0 + qt0 + npos - a.shift;
    for(int e = tid; e < kGRows * kGC; e += kGThreads) {
        const int r = e / kGC, cc = e % kGC;
        uint4* dst = Qs + swz<kGC>(r, cc);
        if(r < used)
            cp_async16(dst, a.q + static_cast<std::size_t>(qt0 + r / group) * a.q_stride +
                                static_cast<std::size_t>(g * group + r % group) * kGHd + cc * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    GlobalTile tile;
    tile.fetch(c, g, 0, khi, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    GlobalRows rows;
    const int qr0 = rg * 16, r0 = qr0 + gl, r1 = r0 + 8;
    rows.load_q(Qs, qr0, half, lane);
    __syncthreads();  // the query area becomes the tile buffers
    tile.stage(c, Kr[0], Vs[0], tid);
    __syncthreads();
    const Span sp = row_span(r0 < used ? a.pos0 + qt0 + r0 / group : -1, r1 < used ? a.pos0 + qt0 + r1 / group : -1,
                             a.shift, 0);
    const int warp_hi = a.pos0 + qt0 + min(qr0 + 15, used - 1) / group - a.shift;
    for(int key0 = 0, j = 0;; key0 += kGK, ++j) {
        const bool more = key0 + kGK < khi;
        if(more) tile.fetch(c, g, key0 + kGK, khi, tid);
        if(qr0 < used && key0 <= warp_hi) rows.step(Kr[j & 1], Vs[j & 1], Sx, rg, half, key0, sp, lane);
        if(!more) break;
        tile.stage(c, Kr[(j + 1) & 1], Vs[(j + 1) & 1], tid);
        __syncthreads();
    }
    rows.sm.finish();
    const float i0 = rows.sm.l0 > 0.f ? 1.f / rows.sm.l0 : 0.f, i1 = rows.sm.l1 > 0.f ? 1.f / rows.sm.l1 : 0.f;
#pragma unroll
    for(int n = 0; n < 32; ++n) {
        const int d = half * 256 + n * 8 + 2 * t4;
        if(r0 < used)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(qt0 + r0 / group) * a.out_stride +
                                       static_cast<std::size_t>(g * group + r0 % group) * kGHd + d) =
                make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        if(r1 < used)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(qt0 + r1 / group) * a.out_stride +
                                       static_cast<std::size_t>(g * group + r1 % group) * kGHd + d) =
                make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
}

__global__ void rope_table_kernel(float2* __restrict__ table, const float* __restrict__ inv, int positions) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= static_cast<std::size_t>(positions) * kGRopePairs) return;
    const int p = static_cast<int>(i / kGRopePairs), f = static_cast<int>(i % kGRopePairs);
    float sn, cs;
    sincosf(static_cast<float>(p) * inv[f], &sn, &cs);
    table[i] = make_float2(cs, sn);
}

__global__ void __launch_bounds__(kGThreads, 1) global_decode_kernel(AttnRows a, GlobalCache c, int blocks, float* partial) {
    pdl_wait();
    pdl_trigger();
    constexpr int group = kHeads / kGKv;  // 8
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    constexpr int kBuf = kGK * kGRc + kGK * kGC;  // one tile: Kr then Vs
    uint4* Kr[2] = {smem, smem + kBuf};
    uint4* Vs[2] = {smem + kGK * kGRc, smem + kBuf + kGK * kGRc};
    float4* Sx = reinterpret_cast<float4*>(smem + kGRows * kGC);
    const int g = blockIdx.x, blk = blockIdx.y;
    const int pos0 = a.pos0_dev ? *a.pos0_dev : a.pos0;
    const int khi = pos0 + a.T - a.shift;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int rg = warp & 3, half = warp >> 2;
    float* part = partial + (static_cast<std::size_t>(g) * blocks + blk) * kDecRows * kGHd;
    float* ml = partial + static_cast<std::size_t>(kGKv) * blocks * kDecRows * kGHd + (static_cast<std::size_t>(g) * blocks + blk) * kDecRows * 2;
    auto tile_key = [&](int j) { return (blk + (j >> 2) * blocks) * kGPiece + (j & 3) * kGK; };
    if(tile_key(0) >= khi) {
        for(int r = tid; r < kDecRows; r += kGThreads) { ml[r * 2] = -INFINITY; ml[r * 2 + 1] = 0.f; }
        return;
    }
    const int used = a.T * group;
    for(int e = tid; e < kGRows * kGC; e += kGThreads) {
        const int r = e / kGC, cc = e % kGC;
        uint4* dst = Qs + swz<kGC>(r, cc);
        if(r < used) cp_async16(dst, a.q + static_cast<std::size_t>(r / group) * a.q_stride + static_cast<std::size_t>(g * group + r % group) * kGHd + cc * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    GlobalTile tile;
    tile.fetch(c, g, tile_key(0), khi, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    GlobalRows rows;
    const int qr0 = rg * 16, r0 = qr0 + gl, r1 = r0 + 8;
    rows.load_q(Qs, qr0, half, lane);
    __syncthreads();  // the query area becomes the tile buffers
    tile.stage(c, Kr[0], Vs[0], tid);
    __syncthreads();
    const Span sp = row_span(r0 < used ? pos0 + r0 / group : -1, r1 < used ? pos0 + r1 / group : -1, a.shift, 0);
    for(int j = 0;; ++j) {
        const int kfirst = tile_key(j), knext = tile_key(j + 1);
        const bool more = knext < khi;
        if(more) tile.fetch(c, g, knext, khi, tid);
        if(qr0 < used) rows.step(Kr[j & 1], Vs[j & 1], Sx, rg, half, kfirst, sp, lane);
        if(!more) break;
        tile.stage(c, Kr[(j + 1) & 1], Vs[(j + 1) & 1], tid);
        __syncthreads();
    }
    rows.sm.finish();
    const float i0 = rows.sm.l0 > 0.f ? 1.f / rows.sm.l0 : 0.f, i1 = rows.sm.l1 > 0.f ? 1.f / rows.sm.l1 : 0.f;
#pragma unroll
    for(int n = 0; n < 32; ++n) {
        const int dd = half * 256 + n * 8 + 2 * t4;
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r0) * kGHd + dd) = make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r1) * kGHd + dd) = make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
    if(t4 == 0 && half == 0) {
        ml[r0 * 2] = rows.sm.m0; ml[r0 * 2 + 1] = rows.sm.l0;
        ml[r1 * 2] = rows.sm.m1; ml[r1 * 2 + 1] = rows.sm.l1;
    }
}

// ================================================================================================ full attention, FP8 Q K^T
// Prompt attention with S = Q K^T on FP8 tensor cores (mma.m16n8k32 e4m3): queries quantized per (row, 256-dim half)
// in the kernel, keys read as their cache codes, u (384 dims, the k_norm weight folded into the queries) and the stored
// rotated dims (128). P V keeps BF16 (values dequantized once per tile). Rows, warp roles and the pair exchange are the
// BF16 kernel's (8 positions x 8 query heads, warp halves own 256 dims).
constexpr int kG8C = kGHd / 16;  // 32 16-byte FP8 chunks per u row
constexpr int kG8R = 2 * kGRopePairs / 16;  // 8 16-byte FP8 chunks per rotated-key row

struct Q8Tile {
    uint4 u[2];  // chunks tid and tid + 256 of the 16 keys x 32 chunks
    float su[2]; // their keys' scales
    uint4 r;     // rotated-key chunk tid (threads < 128)
    float us, rs;
    __device__ __forceinline__ void fetch(const GlobalCache& c, int g, int key0, int khi, int tid) {
#pragma unroll
        for(int i = 0; i < 2; ++i) {
            const int e = tid + 256 * i, key = key0 + e / kG8C;
            u[i] = key < khi ? __ldcg(reinterpret_cast<const uint4*>(c.u + (static_cast<std::size_t>(key) * kGKv + g) * kGHd + (e % kG8C) * 16))
                             : make_uint4(0u, 0u, 0u, 0u);
            su[i] = key < khi ? __ldcg(c.us + static_cast<std::size_t>(key) * kGKv + g) : 0.f;
        }
        if(tid < kGK * kG8R) {
            const int key = key0 + tid / kG8R;
            r = key < khi ? __ldcg(reinterpret_cast<const uint4*>(c.kr + (static_cast<std::size_t>(key) * kGKv + g) * (2 * kGRopePairs) + (tid % kG8R) * 16))
                          : make_uint4(0u, 0u, 0u, 0u);
        }
        if(tid < kGK) {
            const int key = key0 + tid;
            us = key < khi ? __ldcg(c.us + static_cast<std::size_t>(key) * kGKv + g) : 0.f;
            rs = key < khi ? __ldcg(c.krs + static_cast<std::size_t>(key) * kGKv + g) : 0.f;
        }
    }
    // Key codes and rotated-key codes as stored, their scales, and the values dequantized to BF16.
    __device__ __forceinline__ void stage(uint4* K8, uint4* R8, uint4* Vb, float* sus, float* srs, int tid) const {
        if(tid < kGK) { sus[tid] = us; srs[tid] = rs; }
#pragma unroll
        for(int i = 0; i < 2; ++i) {
            const int e = tid + 256 * i, row = e / kG8C, cc = e % kG8C;
            K8[swz<kG8C>(row, cc)] = u[i];
            Vb[swz<kGC>(row, 2 * cc)] = dequant8(u[i].x, u[i].y, su[i]);
            Vb[swz<kGC>(row, 2 * cc + 1)] = dequant8(u[i].z, u[i].w, su[i]);
        }
        if(tid < kGK * kG8R) R8[swz<kG8R>(tid / kG8R, tid % kG8R)] = r;
    }
};

__device__ __forceinline__ void mma_fp8(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct Q8Rows {
    float o[32][4];
    std::uint32_t qa[8][4];
    float qs0 = 0.f, qs1 = 0.f;
    Softmax<2, 32> sm;
    __device__ __forceinline__ Q8Rows() {
#pragma unroll
        for(int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    }
    __device__ __forceinline__ void load_q(const uint4* Q8, const float* qs, int qr0, int half, int lane) {
#pragma unroll
        for(int ks = 0; ks < 8; ++ks) ldsm_x4(qa[ks], Q8 + swz<kG8C>(qr0 + (lane & 15), half * 16 + ks * 2 + (lane >> 4)));
        qs0 = qs[(qr0 + (lane >> 2)) * 2 + half];
        qs1 = qs[(qr0 + (lane >> 2) + 8) * 2 + half];
    }
    __device__ __forceinline__ void step(const uint4* K8, const uint4* R8, const uint4* Vb, const float* sus, const float* srs,
                                         float4* Sx, int rg, int half, int kfirst, const Span& sp, int lane) {
        float sn[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, sr[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const int key = (lane & 7) + ((lane >> 4) << 3), sub = (lane >> 3) & 1;
#pragma unroll
        for(int ks = 0; ks < 8; ++ks) {
            std::uint32_t bb[4];
            if(ks < 2) {
                ldsm_x4(bb, R8 + swz<kG8R>(key, half * 4 + ks * 2 + sub));
                mma_fp8(sr[0], qa[ks], bb[0], bb[1]);
                mma_fp8(sr[1], qa[ks], bb[2], bb[3]);
            } else {
                ldsm_x4(bb, K8 + swz<kG8C>(key, half * 16 + ks * 2 + sub));
                mma_fp8(sn[0], qa[ks], bb[0], bb[1]);
                mma_fp8(sn[1], qa[ks], bb[2], bb[3]);
            }
        }
        const int t4 = lane & 3;
        float p[2][4];
#pragma unroll
        for(int n = 0; n < 2; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int kl = n * 8 + 2 * t4 + (i & 1);
                p[n][i] = ((i >> 1) ? qs1 : qs0) * (sn[n][i] * sus[kl] + sr[n][i] * srs[kl]);
            }
        float4* mine = Sx + ((rg * 2 + half) * 32 + lane) * 2;
        mine[0] = make_float4(p[0][0], p[0][1], p[0][2], p[0][3]);
        mine[1] = make_float4(p[1][0], p[1][1], p[1][2], p[1][3]);
        asm volatile("bar.sync %0, 64;" :: "r"(1 + rg) : "memory");
        const float4* other = Sx + ((rg * 2 + (half ^ 1)) * 32 + lane) * 2;
        const float4 o0 = other[0], o1 = other[1];
        const float q0[4] = {o0.x, o0.y, o0.z, o0.w}, q1[4] = {o1.x, o1.y, o1.z, o1.w};
        float s[2][4];
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            s[0][i] = half ? q0[i] + p[0][i] : p[0][i] + q0[i];
            s[1][i] = half ? q1[i] + p[1][i] : p[1][i] + q1[i];
        }
        std::uint32_t pf[2][2];
        sm.update(s, kfirst, sp, lane, o, pf);
        const std::uint32_t a0 = pf[0][0], a1 = pf[0][1], a2 = pf[1][0], a3 = pf[1][1];
        const int vkey = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
        for(int dp = 0; dp < 16; ++dp) {
            std::uint32_t bv[4];
            ldsm_x4_t(bv, Vb + swz<kGC>(vkey, half * 32 + dp * 2 + (lane >> 4)));
            mma_bf16(o[2 * dp], a0, a1, a2, a3, bv[0], bv[1]);
            mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, bv[2], bv[3]);
        }
    }
};

struct Q8Smem {
    uint4 q[kGRows * kG8C];
    float qs[kGRows * 2];
    uint4 k[2][kGK * kG8C];
    uint4 r[2][kGK * kG8R];
    uint4 v[2][kGK * kGC];
    float us[2][kGK], rs[2][kGK];
    float4 x[8 * 32 * 2];
};

__global__ void __launch_bounds__(256, 1) global_prefill_q8_kernel(AttnRows a, GlobalCache c) {
    pdl_wait();
    pdl_trigger();
    constexpr int group = kHeads / kGKv;  // 8
    constexpr int kPos = kGRows / group;  // 8 positions per block
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Q8Smem& S = *reinterpret_cast<Q8Smem*>(smem_raw);
    const int g = blockIdx.y;
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kPos;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int rg = warp & 3, half = warp >> 2;
    const int npos = min(kPos, a.T - qt0);
    const int used = npos * group;
    const int khi = a.pos0 + qt0 + npos - a.shift;
    Q8Tile tile;
    tile.fetch(c, g, 0, khi, tid);
    // Quantize the queries: warp w handles (row, half) tasks w, w + 8, ...; lane holds 8 of the 256 values.
    for(int task = warp; task < kGRows * 2; task += 8) {
        const int row = task >> 1, hh = task & 1;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if(row < used)
            v = *reinterpret_cast<const uint4*>(a.q + static_cast<std::size_t>(qt0 + row / group) * a.q_stride +
                                                static_cast<std::size_t>(g * group + row % group) * kGHd + hh * 256 + lane * 8);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
        float x[8], am = 0.f;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(b[i]);
            x[2 * i] = f.x;
            x[2 * i + 1] = f.y;
            am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
#pragma unroll
        for(int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const float sc = am > 0.f ? am / 448.f : 1.f;
        std::uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            w0 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
            w1 |= static_cast<std::uint32_t>(__nv_cvt_float_to_fp8(x[4 + i] / sc, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
        }
        uint2* dst = reinterpret_cast<uint2*>(&S.q[swz<kG8C>(row, hh * 16 + (lane >> 1))]) + (lane & 1);
        *dst = make_uint2(w0, w1);
        if(lane == 0) S.qs[row * 2 + hh] = sc;
    }
    tile.stage(S.k[0], S.r[0], S.v[0], S.us[0], S.rs[0], tid);
    __syncthreads();
    Q8Rows rows;
    const int qr0 = rg * 16, r0 = qr0 + gl, r1 = r0 + 8;
    rows.load_q(S.q, S.qs, qr0, half, lane);
    __syncthreads();
    const Span sp = row_span(r0 < used ? a.pos0 + qt0 + r0 / group : -1, r1 < used ? a.pos0 + qt0 + r1 / group : -1,
                             a.shift, 0);
    const int warp_hi = a.pos0 + qt0 + min(qr0 + 15, used - 1) / group - a.shift;
    for(int key0 = 0, j = 0;; key0 += kGK, ++j) {
        const int b = j & 1, nb = b ^ 1;
        const bool more = key0 + kGK < khi;
        if(more) tile.fetch(c, g, key0 + kGK, khi, tid);
        if(qr0 < used && key0 <= warp_hi)
            rows.step(S.k[b], S.r[b], S.v[b], S.us[b], S.rs[b], S.x, rg, half, key0, sp, lane);
        if(!more) break;
        tile.stage(S.k[nb], S.r[nb], S.v[nb], S.us[nb], S.rs[nb], tid);
        __syncthreads();
    }
    rows.sm.finish();
    const float i0 = rows.sm.l0 > 0.f ? 1.f / rows.sm.l0 : 0.f, i1 = rows.sm.l1 > 0.f ? 1.f / rows.sm.l1 : 0.f;
#pragma unroll
    for(int n = 0; n < 32; ++n) {
        const int d = half * 256 + n * 8 + 2 * t4;
        if(r0 < used)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(qt0 + r0 / group) * a.out_stride +
                                       static_cast<std::size_t>(g * group + r0 % group) * kGHd + d) =
                make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        if(r1 < used)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(qt0 + r1 / group) * a.out_stride +
                                       static_cast<std::size_t>(g * group + r1 % group) * kGHd + d) =
                make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
}

// ================================================================================================ full attention, BF16 Q x FP8 key codes
// Prompt attention with BF16 queries against the stored key codes: each B fragment of S = Q K^T is one 32-bit shared
// load of 4 E4M3 codes (dims 4t .. 4t + 3 of a 16-dim group) converted exactly to BF16 in registers; the queries are
// staged with the matching dim order (MMA k index 2t + j <-> dim 4t + j, 8 + 2t + j <-> 4t + 2 + j), so S is exact
// in the codes and the key scales multiply S per key in FP32. Same tiles, rows and exchange as the FP8 kernel.
__device__ __forceinline__ int qb_dim(int k) { return k < 8 ? 4 * (k >> 1) + (k & 1) : 4 * ((k - 8) >> 1) + 2 + (k & 1); }
__device__ __forceinline__ std::uint32_t e4m3x2_bf2(std::uint32_t two) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&r));
    return pack_bf2(f.x, f.y);
}

struct QBRows {
    float o[32][4];
    std::uint32_t qa[16][4];
    Softmax<2, 32> sm;
    __device__ __forceinline__ QBRows() {
#pragma unroll
        for(int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    }
    __device__ __forceinline__ void load_q(const uint4* Qs, int qr0, int half, int lane) {
#pragma unroll
        for(int kq = 0; kq < 16; ++kq) ldsm_x4(qa[kq], Qs + swz<kGC>(qr0 + (lane & 15), (half * 16 + kq) * 2 + (lane >> 4)));
    }
    __device__ __forceinline__ void step(const uint4* K8, const uint4* R8, const uint4* Vb, const float* sus, const float* srs,
                                         float4* Sx, int rg, int half, int kfirst, const Span& sp, int lane) {
        float sn[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, sr[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const int g8 = lane >> 2, t4 = lane & 3;
#pragma unroll
        for(int kq = 0; kq < 16; ++kq) {
            // 16-dim group kq of this half: rotated dims for kq < 4 (rotated-key chunk), u codes otherwise.
#pragma unroll
            for(int n = 0; n < 2; ++n) {
                const int key = n * 8 + g8;
                std::uint32_t w;
                if(kq < 4) w = reinterpret_cast<const std::uint32_t*>(R8 + swz<kG8R>(key, half * 4 + kq))[t4];
                else w = reinterpret_cast<const std::uint32_t*>(K8 + swz<kG8C>(key, half * 16 + kq))[t4];
                const std::uint32_t b0 = e4m3x2_bf2(w), b1 = e4m3x2_bf2(w >> 16);
                if(kq < 4) mma_bf16(sr[n], qa[kq][0], qa[kq][1], qa[kq][2], qa[kq][3], b0, b1);
                else mma_bf16(sn[n], qa[kq][0], qa[kq][1], qa[kq][2], qa[kq][3], b0, b1);
            }
        }
        float p[2][4];
#pragma unroll
        for(int n = 0; n < 2; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int kl = n * 8 + 2 * t4 + (i & 1);
                p[n][i] = sn[n][i] * sus[kl] + sr[n][i] * srs[kl];
            }
        float4* mine = Sx + ((rg * 2 + half) * 32 + lane) * 2;
        mine[0] = make_float4(p[0][0], p[0][1], p[0][2], p[0][3]);
        mine[1] = make_float4(p[1][0], p[1][1], p[1][2], p[1][3]);
        asm volatile("bar.sync %0, 64;" :: "r"(1 + rg) : "memory");
        const float4* other = Sx + ((rg * 2 + (half ^ 1)) * 32 + lane) * 2;
        const float4 o0 = other[0], o1 = other[1];
        const float q0[4] = {o0.x, o0.y, o0.z, o0.w}, q1[4] = {o1.x, o1.y, o1.z, o1.w};
        float s[2][4];
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            s[0][i] = half ? q0[i] + p[0][i] : p[0][i] + q0[i];
            s[1][i] = half ? q1[i] + p[1][i] : p[1][i] + q1[i];
        }
        std::uint32_t pf[2][2];
        sm.update(s, kfirst, sp, lane, o, pf);
        const std::uint32_t a0 = pf[0][0], a1 = pf[0][1], a2 = pf[1][0], a3 = pf[1][1];
        const int vkey = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
        for(int dp = 0; dp < 16; ++dp) {
            std::uint32_t bv[4];
            ldsm_x4_t(bv, Vb + swz<kGC>(vkey, half * 32 + dp * 2 + (lane >> 4)));
            mma_bf16(o[2 * dp], a0, a1, a2, a3, bv[0], bv[1]);
            mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, bv[2], bv[3]);
        }
    }
};

struct QBTiles {
    uint4 k[2][kGK * kG8C];
    uint4 r[2][kGK * kG8R];
    uint4 v[2][kGK * kGC];
    float us[2][kGK], rs[2][kGK];
};
struct QBSmem {
    union {
        uint4 q[kGRows * kGC];  // BF16 queries, dims permuted within each 16-dim group (qb_dim), until loaded
        QBTiles t;              // then the double-buffered tiles
    };
    float4 x[8 * 32 * 2];
};

__global__ void __launch_bounds__(256, 1) global_prefill_qb_kernel(AttnRows a, GlobalCache c) {
    pdl_wait();
    pdl_trigger();
    constexpr int group = kHeads / kGKv;
    constexpr int kPos = kGRows / group;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    QBSmem& S = *reinterpret_cast<QBSmem*>(smem_raw);
    const int g = blockIdx.y;
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kPos;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int rg = warp & 3, half = warp >> 2;
    const int npos = min(kPos, a.T - qt0);
    const int used = npos * group;
    const int khi = a.pos0 + qt0 + npos - a.shift;
    Q8Tile tile;
    tile.fetch(c, g, 0, khi, tid);
    // Queries: thread e handles 8 consecutive physical dims (one 16-byte chunk) of a row, gathering the permuted dims.
    for(int e = tid; e < kGRows * kGC; e += 256) {
        const int row = e / kGC, cc = e % kGC;  // chunk cc: dims (cc >> 1) * 16 + (cc & 1) * 8 .. + 7 (physical k)
        uint4 out = make_uint4(0u, 0u, 0u, 0u);
        if(row < used) {
            const __nv_bfloat16* src = a.q + static_cast<std::size_t>(qt0 + row / group) * a.q_stride +
                                       static_cast<std::size_t>(g * group + row % group) * kGHd + (cc >> 1) * 16;
            std::uint16_t h[8];
#pragma unroll
            for(int j = 0; j < 8; ++j) h[j] = reinterpret_cast<const std::uint16_t*>(src)[qb_dim((cc & 1) * 8 + j)];
            out = make_uint4(h[0] | (h[1] << 16), h[2] | (h[3] << 16), h[4] | (h[5] << 16), h[6] | (h[7] << 16));
        }
        S.q[swz<kGC>(row, cc)] = out;
    }
    __syncthreads();
    QBRows rows;
    const int qr0 = rg * 16, r0 = qr0 + gl, r1 = r0 + 8;
    rows.load_q(S.q, qr0, half, lane);
    __syncthreads();
    QBTiles& T = S.t;
    tile.stage(T.k[0], T.r[0], T.v[0], T.us[0], T.rs[0], tid);
    __syncthreads();
    const Span sp = row_span(r0 < used ? a.pos0 + qt0 + r0 / group : -1, r1 < used ? a.pos0 + qt0 + r1 / group : -1,
                             a.shift, 0);
    const int warp_hi = a.pos0 + qt0 + min(qr0 + 15, used - 1) / group - a.shift;
    for(int key0 = 0, j = 0;; key0 += kGK, ++j) {
        const int b = j & 1, nb = b ^ 1;
        const bool more = key0 + kGK < khi;
        if(more) tile.fetch(c, g, key0 + kGK, khi, tid);
        if(qr0 < used && key0 <= warp_hi)
            rows.step(T.k[b], T.r[b], T.v[b], T.us[b], T.rs[b], S.x, rg, half, key0, sp, lane);
        if(!more) break;
        tile.stage(T.k[nb], T.r[nb], T.v[nb], T.us[nb], T.rs[nb], tid);
        __syncthreads();
    }
    rows.sm.finish();
    const float i0 = rows.sm.l0 > 0.f ? 1.f / rows.sm.l0 : 0.f, i1 = rows.sm.l1 > 0.f ? 1.f / rows.sm.l1 : 0.f;
#pragma unroll
    for(int n = 0; n < 32; ++n) {
        const int d = half * 256 + n * 8 + 2 * t4;
        if(r0 < used)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(qt0 + r0 / group) * a.out_stride +
                                       static_cast<std::size_t>(g * group + r0 % group) * kGHd + d) =
                make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        if(r1 < used)
            *reinterpret_cast<float2*>(a.out + static_cast<std::size_t>(qt0 + r1 / group) * a.out_stride +
                                       static_cast<std::size_t>(g * group + r1 % group) * kGHd + d) =
                make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
}

template <typename K>
void set_smem(K kernel, std::size_t bytes) {
    KCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
}

}  // namespace kattn
using namespace kattn;

std::size_t slide_partial_floats(int blocks) { return static_cast<std::size_t>(kSKv) * blocks * kDecRows * (kSHd + 2); }
std::size_t global_partial_floats(int blocks) { return static_cast<std::size_t>(kGKv) * blocks * kDecRows * (kGHd + 2); }

void slide_prefill(const AttnRows& a, const SlideCache& c, cudaStream_t s) {
    const std::size_t bytes = static_cast<std::size_t>(kSQRows + 2 * kSK) * kSHd * 2;
    set_smem(slide_prefill_kernel, bytes);
    launch_k(slide_prefill_kernel, dim3((a.T + kSQRows - 1) / kSQRows, kHeads), kSPrefillThreads, bytes, s, a, c);
    KLAUNCH();
}

void global_prefill(const AttnRows& a, const GlobalCache& c, cudaStream_t s) {
    const std::size_t bytes = static_cast<std::size_t>(kGRows * kGC) * 16 + 8 * 32 * 32;
    set_smem(global_prefill_kernel, bytes);
    launch_k(global_prefill_kernel, dim3((a.T + 7) / 8, kGKv), kGThreads, bytes, s, a, c);
    KLAUNCH();
}

void fill_rope_table(float2* table, const float* inv, int positions, cudaStream_t s) {
    const std::size_t n = static_cast<std::size_t>(positions) * kGRopePairs;
    rope_table_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(table, inv, positions);
    KLAUNCH();
}

void global_prefill_q8(const AttnRows& a, const GlobalCache& c, cudaStream_t s) {
    const std::size_t bytes = sizeof(Q8Smem);
    set_smem(global_prefill_q8_kernel, bytes);
    launch_k(global_prefill_q8_kernel, dim3((a.T + 7) / 8, kGKv), 256, bytes, s, a, c);
    KLAUNCH();
}

void global_prefill_qb(const AttnRows& a, const GlobalCache& c, cudaStream_t s) {
    const std::size_t bytes = sizeof(QBSmem);
    set_smem(global_prefill_qb_kernel, bytes);
    launch_k(global_prefill_qb_kernel, dim3((a.T + 7) / 8, kGKv), 256, bytes, s, a, c);
    KLAUNCH();
}

void slide_decode(const AttnRows& a, const SlideCache& c, int blocks, float* partial, cudaStream_t s) {
    if(a.T < 1 || a.T * (kHeads / kSKv) > kDecRows) throw std::invalid_argument("slide_decode: 1..32 columns");
    const std::size_t bytes = static_cast<std::size_t>(kDecRows + 2 * kSK) * kSHd * 2;
    set_smem(slide_decode_kernel, bytes);
    launch_k(slide_decode_kernel, dim3(kSKv, blocks), kSDecThreads, bytes, s, a, c, blocks, partial);
    launch_k(combine_kernel<kSHd, kSKv>, dim3(a.T, kHeads), kSHd, sizeof(float) * blocks, s, a, blocks,
             static_cast<const float*>(partial));
    KLAUNCH();
}

void global_decode(const AttnRows& a, const GlobalCache& c, int blocks, float* partial, cudaStream_t s) {
    if(a.T < 1 || a.T * (kHeads / kGKv) > kDecRows) throw std::invalid_argument("global_decode: 1..8 columns");
    const std::size_t bytes = static_cast<std::size_t>(kGRows * kGC) * 16 + 8 * 32 * 32;
    set_smem(global_decode_kernel, bytes);
    launch_k(global_decode_kernel, dim3(kGKv, blocks), kGThreads, bytes, s, a, c, blocks, partial);
    launch_k(combine_kernel<kGHd, kGKv>, dim3(a.T, kHeads), kGHd, sizeof(float) * blocks, s, a, blocks,
             static_cast<const float*>(partial));
    KLAUNCH();
}

}  // namespace g4
