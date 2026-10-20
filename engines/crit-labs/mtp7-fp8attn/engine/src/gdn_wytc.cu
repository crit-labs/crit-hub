// Chunked (WY) gated-delta scan on BF16 tensor cores with FP32 accumulation and an FP32 state.
//
// Chunk of C = 64 tokens, inclusive cumulative log decay gam, K normalized keys, Q scaled queries, V values, beta:
//   A_ij = beta_i e^{gam_i - gam_j} k_i.k_j (j < i), T = (I + A)^-1, W = T diag(beta e^gam) K, Ub = T diag(beta) V,
//   Qk_ij = e^{gam_i - gam_j} q_i.k_j (j <= i);   with S0 (keys x values) at the chunk start:
//   U = Ub - W S0,   O = diag(e^gam) Q S0 + Qk U,   S = e^{gam_C} S0 + (diag(e^{gam_C - gam}) K)^T U.
// Pass 1 (tc_prep, one block per chunk and head) builds T in FP32 and writes W, Ub, Qk (BF16) and gam. Pass 2
// (tc_chunk, 4 blocks per head, 32 value columns each) keeps S in mma accumulator registers across the chunks; the
// matrix products take BF16 operands (S is rounded to BF16 only as an operand, never in its FP32 accumulation).
#include "gdn_wy.cuh"

namespace kestrel {
namespace kwytc {

constexpr int C = kWyChunk;  // 64
constexpr int D = kLinDim;   // 128
constexpr int VS = 32;       // value columns per pass-2 block

__device__ __forceinline__ void mma(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ std::uint32_t pack2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const std::uint32_t*>(&v);
}

// BF16 tiles in shared memory with a 16-byte-chunk XOR swizzle on rows of `cols` elements (cols % 64 == 0).
template <int COLS>
__device__ __forceinline__ __nv_bfloat16* at(__nv_bfloat16* base, int row, int col) {
    // 64-byte rows (32 columns) pair rows per 128-byte bank period: swizzle with (row / 2) % 4.
    const int chunk = COLS == 32 ? ((col >> 3) ^ ((row >> 1) & 3)) : ((col >> 3) ^ (row & 7));
    return base + row * COLS + chunk * 8 + (col & 7);
}

// A fragment (16 x 16, row-major) of a [rows][COLS] tile at (r0, k0).
template <int COLS>
__device__ __forceinline__ void frag_a(std::uint32_t (&a)[4], __nv_bfloat16* base, int r0, int k0, int lane) {
    ldsm4(a, at<COLS>(base, r0 + (lane & 15), k0 + ((lane >> 4) << 3)));
}
// B fragments (k 16 x n 16, two n-tiles) of a row-major [k][COLS] tile (B[k][n] = tile[k0 + k][n0 + n]): trans load.
template <int COLS>
__device__ __forceinline__ void frag_b_t(std::uint32_t (&b)[4], __nv_bfloat16* base, int k0, int n0, int lane) {
    ldsm4t(b, at<COLS>(base, k0 + (lane & 7) + (((lane >> 3) & 1) << 3), n0 + ((lane >> 4) << 3)));
}
// B fragments (k 16 x n 16) from a row-major [n][COLS] tile (B[k][n] = tile[n0 + n][k0 + k]): plain load.
template <int COLS>
__device__ __forceinline__ void frag_b_n(std::uint32_t (&b)[4], __nv_bfloat16* base, int k0, int n0, int lane) {
    ldsm4(b, at<COLS>(base, n0 + (lane & 7) + ((lane >> 4) << 3), k0 + (((lane >> 3) & 1) << 3)));
}

// ------------------------------------------------------------------------------------------------ pass 1
struct PrepSmem {
    __nv_bfloat16 k[C * D];   // keys
    __nv_bfloat16 x[C * D];   // queries, then beta e^gam K, then beta V
    float a[C][C + 1];        // A, then T (in place below the diagonal)
    __nv_bfloat16 t[C * C];   // T as BF16 operand
    float gam[C], beta[C];
};

__global__ void __launch_bounds__(128) tc_prep_kernel(GdnPrefillScratch sc, int T, int t_base, WyScratch ws) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PrepSmem& S = *reinterpret_cast<PrepSmem*>(smem_raw);
    const int c = blockIdx.x, h = blockIdx.y, kh = h / 3, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int t0 = t_base + c * C;
    if(tid < C) {
        const int t = t0 + tid;
        S.gam[tid] = t < T ? sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 0] : 0.f;
        S.beta[tid] = t < T ? sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 1] : 0.f;
    }
    for(int e = tid; e < C * D / 4; e += blockDim.x) {
        const int i = e / (D / 4), d = (e % (D / 4)) * 4, t = t0 + i;
        float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), qv = kv;
        if(t < T) {
            kv = *reinterpret_cast<const float4*>(sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 1) * D + d);
            qv = *reinterpret_cast<const float4*>(sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 0) * D + d);
        }
        *reinterpret_cast<uint2*>(at<D>(S.k, i, d)) = make_uint2(pack2(kv.x, kv.y), pack2(kv.z, kv.w));
        *reinterpret_cast<uint2*>(at<D>(S.x, i, d)) = make_uint2(pack2(qv.x, qv.y), pack2(qv.z, qv.w));
    }
    __syncthreads();
    if(tid == 0) {
        float acc = 0.f;
        for(int i = 0; i < C; ++i) { acc += S.gam[i]; S.gam[i] = acc; }
    }
    // KK and QK (64 x 64, K-dim 128): warp w computes rows 16w..16w+15, all 8 n-tiles, for both products.
    float kk[8][4], qk[8][4];
#pragma unroll
    for(int n = 0; n < 8; ++n)
#pragma unroll
        for(int i = 0; i < 4; ++i) kk[n][i] = qk[n][i] = 0.f;
#pragma unroll
    for(int ks = 0; ks < D / 16; ++ks) {
        std::uint32_t ak[4], aq[4];
        frag_a<D>(ak, S.k, warp * 16, ks * 16, lane);
        frag_a<D>(aq, S.x, warp * 16, ks * 16, lane);
#pragma unroll
        for(int np = 0; np < 4; ++np) {
            std::uint32_t b[4];
            frag_b_n<D>(b, S.k, ks * 16, np * 16, lane);
            mma(kk[2 * np], ak, b[0], b[1]);
            mma(kk[2 * np + 1], ak, b[2], b[3]);
            mma(qk[2 * np], aq, b[0], b[1]);
            mma(qk[2 * np + 1], aq, b[2], b[3]);
        }
    }
    __syncthreads();  // gam ready
    __nv_bfloat16* qk_out = reinterpret_cast<__nv_bfloat16*>(ws.Qk) + (static_cast<std::size_t>(c) * kLinVHeads + h) * C * C;
    const int g = lane >> 2, q4 = lane & 3;
#pragma unroll
    for(int n = 0; n < 8; ++n)
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const int r = warp * 16 + g + ((i >> 1) << 3), col = n * 8 + 2 * q4 + (i & 1);
            const float dec = col <= r ? expf(S.gam[r] - S.gam[col]) : 0.f;
            S.a[r][col] = col < r ? S.beta[r] * dec * kk[n][i] : 0.f;
            qk_out[r * C + col] = __float2bfloat16_rn(qk[n][i] * dec);
        }
    __syncthreads();
    // T = (I + A)^-1, row by row (FP32), stored back into a (diagonal implied 1).
    for(int i = 1; i < C; ++i) {
        float v = 0.f;
        if(tid < i) {
            v = -S.a[i][tid];
            for(int m = tid + 1; m < i; ++m) v -= S.a[i][m] * S.a[m][tid];
        }
        __syncthreads();
        if(tid < i) S.a[i][tid] = v;
        __syncthreads();
    }
    for(int e = tid; e < C * C; e += blockDim.x) {
        const int i = e / C, j = e % C;
        *at<C>(S.t, i, j) = __float2bfloat16_rn(j < i ? S.a[i][j] : (j == i ? 1.f : 0.f));
    }
    // x = diag(beta e^gam) K (bf16), then W = T x.
    for(int e = tid; e < C * D; e += blockDim.x) {
        const int i = e / D, d = e % D;
        *at<D>(S.x, i, d) = __float2bfloat16_rn(__bfloat162float(*at<D>(S.k, i, d)) * S.beta[i] * expf(S.gam[i]));
    }
    __syncthreads();
    auto t_times_x = [&](__nv_bfloat16* out) {
        // out [64][128] = T [64][64] * x [64][128]; warp w: rows 16w.., all 16 n-tiles.
        float acc[16][4];
#pragma unroll
        for(int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
        for(int ks = 0; ks < C / 16; ++ks) {
            std::uint32_t a[4];
            frag_a<C>(a, S.t, warp * 16, ks * 16, lane);
#pragma unroll
            for(int np = 0; np < 8; ++np) {
                std::uint32_t b[4];
                frag_b_t<D>(b, S.x, ks * 16, np * 16, lane);
                mma(acc[2 * np], a, b[0], b[1]);
                mma(acc[2 * np + 1], a, b[2], b[3]);
            }
        }
#pragma unroll
        for(int n = 0; n < 16; ++n) {
            const int r = warp * 16 + g, col = n * 8 + 2 * q4;
            *reinterpret_cast<std::uint32_t*>(out + static_cast<std::size_t>(r) * D + col) = pack2(acc[n][0], acc[n][1]);
            *reinterpret_cast<std::uint32_t*>(out + static_cast<std::size_t>(r + 8) * D + col) = pack2(acc[n][2], acc[n][3]);
        }
    };
    const std::size_t base = (static_cast<std::size_t>(c) * kLinVHeads + h) * C * D;
    t_times_x(reinterpret_cast<__nv_bfloat16*>(ws.W) + base);
    __syncthreads();
    // x = diag(beta) V, then Ub = T x.
    for(int e = tid; e < C * D / 4; e += blockDim.x) {
        const int i = e / (D / 4), d = (e % (D / 4)) * 4, t = t0 + i;
        float4 vv = make_float4(0.f, 0.f, 0.f, 0.f);
        if(t < T) vv = *reinterpret_cast<const float4*>(sc.v + (static_cast<std::size_t>(t) * kLinVHeads + h) * D + d);
        const float b = S.beta[i];
        *reinterpret_cast<uint2*>(at<D>(S.x, i, d)) = make_uint2(pack2(vv.x * b, vv.y * b), pack2(vv.z * b, vv.w * b));
    }
    __syncthreads();
    t_times_x(reinterpret_cast<__nv_bfloat16*>(ws.Ub) + base);
    if(tid < C) ws.gam[(static_cast<std::size_t>(c) * kLinVHeads + h) * C + tid] = S.gam[tid];
}

// ------------------------------------------------------------------------------------------------ pass 2
struct ChunkSmem {
    __nv_bfloat16 w[C * D];    // W
    __nv_bfloat16 q[C * D];    // Q
    __nv_bfloat16 k[C * D];    // diag(e^{gam_C - gam}) K
    __nv_bfloat16 s[D * VS];   // S0 slice as an operand [key][value]
    __nv_bfloat16 u[C * VS];   // U [token][value]
    __nv_bfloat16 qk[C * C];
    float ub[C][VS];
    float gam[C];
};

__global__ void __launch_bounds__(128) tc_chunk_kernel(GdnState st, GdnPrefillScratch sc, int T, int t_base, int nch, WyScratch ws) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    ChunkSmem& S = *reinterpret_cast<ChunkSmem*>(smem_raw);
    const int h = blockIdx.x, slice = blockIdx.y, kh = h / 3, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, q4 = lane & 3;
    // State in accumulator layout: warp w owns key rows 32w..32w+31 (m-tiles 2w, 2w+1), 4 n-tiles of 8 values.
    float acc_s[2][4][4];
    // Load from the decode layout [h][v * 4 + quarter][32] (key = quarter * 32 + i).
    auto sidx = [&](int key, int vcol) {
        const int v = slice * VS + vcol;
        return (static_cast<std::size_t>(h) * 512 + v * 4 + (key >> 5)) * 32 + (key & 31);
    };
#pragma unroll
    for(int mt = 0; mt < 2; ++mt)
#pragma unroll
        for(int nt = 0; nt < 4; ++nt)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int key = warp * 32 + mt * 16 + g + ((i >> 1) << 3), vcol = nt * 8 + 2 * q4 + (i & 1);
                acc_s[mt][nt][i] = st.S[sidx(key, vcol)];
            }
    for(int c = 0; c < nch; ++c) {
        const std::size_t base = (static_cast<std::size_t>(c) * kLinVHeads + h) * C * D;
        const __nv_bfloat16* wsrc = reinterpret_cast<const __nv_bfloat16*>(ws.W) + base;
        const __nv_bfloat16* usrc = reinterpret_cast<const __nv_bfloat16*>(ws.Ub) + base;
        const __nv_bfloat16* qksrc = reinterpret_cast<const __nv_bfloat16*>(ws.Qk) + (static_cast<std::size_t>(c) * kLinVHeads + h) * C * C;
        if(tid < C) S.gam[tid] = ws.gam[(static_cast<std::size_t>(c) * kLinVHeads + h) * C + tid];
        for(int e = tid; e < C * D / 8; e += blockDim.x) {
            const int i = e / (D / 8), d = (e % (D / 8)) * 8;
            *reinterpret_cast<uint4*>(at<D>(S.w, i, d)) = *reinterpret_cast<const uint4*>(wsrc + i * D + d);
        }
        for(int e = tid; e < C * VS; e += blockDim.x) {
            const int i = e / VS, vcol = e % VS;
            S.ub[i][vcol] = __bfloat162float(usrc[i * D + slice * VS + vcol]);
        }
        for(int e = tid; e < C * C / 8; e += blockDim.x) {
            const int i = e / (C / 8), j = (e % (C / 8)) * 8;
            *reinterpret_cast<uint4*>(at<C>(S.qk, i, j)) = *reinterpret_cast<const uint4*>(qksrc + i * C + j);
        }
        // S0 as a BF16 operand [key][value].
#pragma unroll
        for(int mt = 0; mt < 2; ++mt)
#pragma unroll
            for(int nt = 0; nt < 4; ++nt) {
                const int key = warp * 32 + mt * 16 + g, vcol = nt * 8 + 2 * q4;
                *reinterpret_cast<std::uint32_t*>(at<VS>(S.s, key, vcol)) = pack2(acc_s[mt][nt][0], acc_s[mt][nt][1]);
                *reinterpret_cast<std::uint32_t*>(at<VS>(S.s, key + 8, vcol)) = pack2(acc_s[mt][nt][2], acc_s[mt][nt][3]);
            }
        __syncthreads();
        // Q and the decayed K of the chunk (FP32 -> BF16).
        const float gc = S.gam[C - 1];
        for(int e = tid; e < C * D / 4; e += blockDim.x) {
            const int i = e / (D / 4), d = (e % (D / 4)) * 4;
            const int t = min(t_base + c * C + i, T - 1);
            const bool live = t_base + c * C + i < T;
            const float4 qv = *reinterpret_cast<const float4*>(sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 0) * D + d);
            const float4 kv = *reinterpret_cast<const float4*>(sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 1) * D + d);
            const float f = live ? expf(gc - S.gam[i]) : 0.f;
            *reinterpret_cast<uint2*>(at<D>(S.q, i, d)) = make_uint2(pack2(qv.x, qv.y), pack2(qv.z, qv.w));
            *reinterpret_cast<uint2*>(at<D>(S.k, i, d)) = make_uint2(pack2(kv.x * f, kv.y * f), pack2(kv.z * f, kv.w * f));
        }
        // U = Ub - W S0 and P = Q S0: warp w rows 16w.., 4 n-tiles, K-dim 128.
        float uacc[4][4], pacc[4][4];
#pragma unroll
        for(int n = 0; n < 4; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) uacc[n][i] = pacc[n][i] = 0.f;
        __syncthreads();
#pragma unroll
        for(int ks = 0; ks < D / 16; ++ks) {
            std::uint32_t aw[4], aq[4];
            frag_a<D>(aw, S.w, warp * 16, ks * 16, lane);
            frag_a<D>(aq, S.q, warp * 16, ks * 16, lane);
#pragma unroll
            for(int np = 0; np < 2; ++np) {
                std::uint32_t b[4];
                frag_b_t<VS>(b, S.s, ks * 16, np * 16, lane);
                mma(uacc[2 * np], aw, b[0], b[1]);
                mma(uacc[2 * np + 1], aw, b[2], b[3]);
                mma(pacc[2 * np], aq, b[0], b[1]);
                mma(pacc[2 * np + 1], aq, b[2], b[3]);
            }
        }
        // U to shared memory (BF16 operand); O = e^gam P + Qk U.
#pragma unroll
        for(int n = 0; n < 4; ++n) {
            const int r = warp * 16 + g, vcol = n * 8 + 2 * q4;
            const float u0 = S.ub[r][vcol] - uacc[n][0], u1 = S.ub[r][vcol + 1] - uacc[n][1];
            const float u2 = S.ub[r + 8][vcol] - uacc[n][2], u3 = S.ub[r + 8][vcol + 1] - uacc[n][3];
            *reinterpret_cast<std::uint32_t*>(at<VS>(S.u, r, vcol)) = pack2(u0, u1);
            *reinterpret_cast<std::uint32_t*>(at<VS>(S.u, r + 8, vcol)) = pack2(u2, u3);
        }
        __syncthreads();
        float oacc[4][4];
#pragma unroll
        for(int n = 0; n < 4; ++n) {
            const int r = warp * 16 + g;
            const float e0 = expf(S.gam[r]), e1 = expf(S.gam[r + 8]);
            oacc[n][0] = pacc[n][0] * e0; oacc[n][1] = pacc[n][1] * e0;
            oacc[n][2] = pacc[n][2] * e1; oacc[n][3] = pacc[n][3] * e1;
        }
#pragma unroll
        for(int ks = 0; ks < C / 16; ++ks) {
            std::uint32_t a[4];
            frag_a<C>(a, S.qk, warp * 16, ks * 16, lane);
#pragma unroll
            for(int np = 0; np < 2; ++np) {
                std::uint32_t b[4];
                frag_b_t<VS>(b, S.u, ks * 16, np * 16, lane);
                mma(oacc[2 * np], a, b[0], b[1]);
                mma(oacc[2 * np + 1], a, b[2], b[3]);
            }
        }
#pragma unroll
        for(int n = 0; n < 4; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int r = warp * 16 + g + ((i >> 1) << 3), vcol = n * 8 + 2 * q4 + (i & 1);
                const int t = t_base + c * C + r;
                if(t < T) sc.o[(static_cast<std::size_t>(t) * kLinVHeads + h) * D + slice * VS + vcol] = oacc[n][i];
            }
        // S = e^{gam_C} S0 + Kd^T U: A = Kd^T [128 keys][64 tokens] (trans load of Kd [token][key]), B = U [token][value].
        const float egc = expf(gc);
#pragma unroll
        for(int mt = 0; mt < 2; ++mt)
#pragma unroll
            for(int nt = 0; nt < 4; ++nt)
#pragma unroll
                for(int i = 0; i < 4; ++i) acc_s[mt][nt][i] *= egc;
#pragma unroll
        for(int ks = 0; ks < C / 16; ++ks) {
#pragma unroll
            for(int mt = 0; mt < 2; ++mt) {
                std::uint32_t a[4];
                // A[m][k] = Kd[k][m]: trans load of the [token][key] tile at (token ks*16, key m0).
                const int m0 = warp * 32 + mt * 16;
                ldsm4t(a, at<D>(S.k, ks * 16 + (lane & 7) + ((lane >> 4) << 3), m0 + (((lane >> 3) & 1) << 3)));
                // ldmatrix.trans of matrices [tok0-7][m0-7], [tok0-7][m8-15], [tok8-15][m0-7], [tok8-15][m8-15] gives
                // a0 (m0-7, k0-7), a1 (m8-15, k0-7), a2 (m0-7, k8-15), a3 (m8-15, k8-15).
#pragma unroll
                for(int np = 0; np < 2; ++np) {
                    std::uint32_t b[4];
                    frag_b_t<VS>(b, S.u, ks * 16, np * 16, lane);
                    mma(acc_s[mt][2 * np], a, b[0], b[1]);
                    mma(acc_s[mt][2 * np + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int mt = 0; mt < 2; ++mt)
#pragma unroll
        for(int nt = 0; nt < 4; ++nt)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int key = warp * 32 + mt * 16 + g + ((i >> 1) << 3), vcol = nt * 8 + 2 * q4 + (i & 1);
                st.S[sidx(key, vcol)] = acc_s[mt][nt][i];
            }
}

}  // namespace kwytc

void gdn_scan_wy_tc(GdnState st, GdnPrefillScratch sc, int T, WyScratch ws, cudaStream_t s) {
    const int prep_bytes = static_cast<int>(sizeof(kwytc::PrepSmem));
    const int chunk_bytes = static_cast<int>(sizeof(kwytc::ChunkSmem));
    KCHECK(cudaFuncSetAttribute(kwytc::tc_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_bytes));
    KCHECK(cudaFuncSetAttribute(kwytc::tc_chunk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, chunk_bytes));
    const int chunks = (T + kWyChunk - 1) / kWyChunk;
    for(int c0 = 0; c0 < chunks; c0 += ws.max_chunks) {
        const int nch = std::min(ws.max_chunks, chunks - c0);
        const int t_base = c0 * kWyChunk;
        launch_k(kwytc::tc_prep_kernel, dim3(nch, kLinVHeads), 128, prep_bytes, s, sc, T, t_base, ws);
        launch_k(kwytc::tc_chunk_kernel, dim3(kLinVHeads, kLinDim / kwytc::VS), 128, chunk_bytes, s, st, sc, T, t_base, nch, ws);
    }
}

}  // namespace kestrel
