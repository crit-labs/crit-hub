// Chunked (WY) gated-delta scan for the prompt path, FP32 throughout.
//
// Within a chunk of C = 64 tokens with inclusive cumulative log decay g (gam), normalized keys K, scaled queries Q,
// values V and gates beta:  A_ij = beta_i e^{gam_i - gam_j} k_i.k_j (j < i),  T = (I + A)^-1,
// W = T diag(beta e^gam) K,  Ub = T diag(beta) V,  Qk_ij = e^{gam_i - gam_j} q_i.k_j (j <= i).
// With the state S0 at the chunk start:  U = Ub - W S0,  O = diag(e^gam) Q S0 + Qk U,
// S = e^{gam_C} S0 + sum_i e^{gam_C - gam_i} k_i U_i^T.
// Pass 1 (wy_prep) computes T, W, Ub, Qk and gam for every (chunk, head) in parallel; pass 2 (wy_chunk) walks the
// chunks of a head in order, 4 blocks per head splitting the value columns, the state held in registers in the
// layout the decode kernel uses ([head][v * 4 + key quarter][32]).
#include "gdn_wy.cuh"

namespace kestrel {
namespace kwy {

constexpr int C = kWyChunk;    // 64
constexpr int D = kLinDim;     // 128
constexpr int RP = D;          // row (floats)
// Pass-2 staging rotates the 16-byte columns of each 32-float key quarter so the four quarters a warp reads at once
// fall in different banks: float4 j of quarter g is stored at quarter g, slot (j + g) % 8.
__device__ __forceinline__ int rot4(int quarter, int j) { return quarter * 8 + ((j + quarter) & 7); }

struct PrepSmem {
    float k[C][RP];
    float q[C][RP];            // queries, then values
    float a[C][C + 1];
    float t[C][C + 1];
    float gam[C], beta[C], g[C];
};

__global__ void __launch_bounds__(256) wy_prep_kernel(GdnPrefillScratch sc, int T, int t_base, WyScratch ws) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PrepSmem& S = *reinterpret_cast<PrepSmem*>(smem_raw);
    const int c = blockIdx.x, h = blockIdx.y, kh = h / 3, tid = threadIdx.x;
    const int t0 = t_base + c * C;
    for(int e = tid; e < C * D / 4; e += blockDim.x) {
        const int i = e / (D / 4), d4 = (e % (D / 4)) * 4, t = t0 + i;
        float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), qv = kv;
        if(t < T) {
            kv = *reinterpret_cast<const float4*>(sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 1) * D + d4);
            qv = *reinterpret_cast<const float4*>(sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + 0) * D + d4);
        }
        *reinterpret_cast<float4*>(&S.k[i][d4]) = kv;
        *reinterpret_cast<float4*>(&S.q[i][d4]) = qv;
    }
    if(tid < C) {
        const int t = t0 + tid;
        S.g[tid] = t < T ? sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 0] : 0.f;
        S.beta[tid] = t < T ? sc.gb[(static_cast<std::size_t>(t) * kLinVHeads + h) * 2 + 1] : 0.f;
    }
    __syncthreads();
    if(tid == 0) {
        float acc = 0.f;
        for(int i = 0; i < C; ++i) {
            acc += S.g[i];
            S.gam[i] = acc;
        }
    }
    __syncthreads();
    if(tid < C) S.g[tid] = S.beta[tid] * expf(S.gam[tid]);  // beta e^gam (g no longer needed)
    // A (strictly lower) and Qk (lower with the diagonal).
    float* qk_out = ws.Qk + (static_cast<std::size_t>(c) * kLinVHeads + h) * C * C;
    const int lane = tid & 31, warp = tid >> 5;
    for(int e = warp; e < C * C; e += 8) {
        const int i = e / C, j = e % C;
        float a = 0.f, qk = 0.f;
        if(j <= i) {
            const float4 ki = *reinterpret_cast<const float4*>(&S.k[i][lane * 4]);
            const float4 qi = *reinterpret_cast<const float4*>(&S.q[i][lane * 4]);
            const float4 kj = *reinterpret_cast<const float4*>(&S.k[j][lane * 4]);
            float dk = ki.x * kj.x + ki.y * kj.y + ki.z * kj.z + ki.w * kj.w;
            float dq = qi.x * kj.x + qi.y * kj.y + qi.z * kj.z + qi.w * kj.w;
#pragma unroll
            for(int o = 16; o; o >>= 1) {
                dk += __shfl_xor_sync(0xffffffffu, dk, o);
                dq += __shfl_xor_sync(0xffffffffu, dq, o);
            }
            const float dec = expf(S.gam[i] - S.gam[j]);
            qk = dq * dec;
            if(j < i) a = S.beta[i] * dec * dk;
        }
        if(lane == 0) {
            S.a[i][j] = a;
            qk_out[e] = qk;
        }
    }
    __syncthreads();
    // T = (I + A)^-1 by forward substitution, one row at a time.
    for(int i = 0; i < C; ++i) {
        if(tid <= i) {
            float v = tid == i ? 1.f : 0.f;
            if(tid < i) {
                for(int m = tid; m < i; ++m) v -= S.a[i][m] * S.t[m][tid];
            }
            S.t[i][tid] = v;
        } else if(tid < C) {
            S.t[i][tid] = 0.f;
        }
        __syncthreads();
    }
    // W = T diag(beta e^gam) K.
    float* w_out = ws.W + (static_cast<std::size_t>(c) * kLinVHeads + h) * C * D;
    for(int e = tid; e < C * D; e += blockDim.x) {
        const int i = e / D, d = e % D;
        float acc = 0.f;
        for(int m = 0; m <= i; ++m) acc = fmaf(S.t[i][m] * S.g[m], S.k[m][d], acc);
        w_out[e] = acc;
    }
    __syncthreads();
    // Values into the query buffer, then Ub = T diag(beta) V.
    for(int e = tid; e < C * D / 4; e += blockDim.x) {
        const int i = e / (D / 4), d4 = (e % (D / 4)) * 4, t = t0 + i;
        float4 vv = make_float4(0.f, 0.f, 0.f, 0.f);
        if(t < T) vv = *reinterpret_cast<const float4*>(sc.v + (static_cast<std::size_t>(t) * kLinVHeads + h) * D + d4);
        *reinterpret_cast<float4*>(&S.q[i][d4]) = vv;
    }
    __syncthreads();
    float* u_out = ws.Ub + (static_cast<std::size_t>(c) * kLinVHeads + h) * C * D;
    for(int e = tid; e < C * D; e += blockDim.x) {
        const int i = e / D, d = e % D;
        float acc = 0.f;
        for(int m = 0; m <= i; ++m) acc = fmaf(S.t[i][m] * S.beta[m], S.q[m][d], acc);
        u_out[e] = acc;
    }
    if(tid < C) ws.gam[(static_cast<std::size_t>(c) * kLinVHeads + h) * C + tid] = S.gam[tid];
}

struct ChunkSmem {
    float m[2][C][RP];         // staged W, Q or K of a chunk (double buffered)
    float u[C][33];
    float o[C][33];
    float qk[C][C + 1];
    float gam[C];
};

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src));
}

__global__ void __launch_bounds__(128) wy_chunk_kernel(GdnState st, GdnPrefillScratch sc, int T, int t_base, int nch,
                                                       WyScratch ws) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    ChunkSmem& S = *reinterpret_cast<ChunkSmem*>(smem_raw);
    const int h = blockIdx.x, slice = blockIdx.y, kh = h / 3, tid = threadIdx.x;
    const int vl = tid >> 2, v = slice * 32 + vl, kg = tid & 3;
    float s[32];
    float4* sp = reinterpret_cast<float4*>(st.S + (static_cast<std::size_t>(h) * 512 + v * 4 + kg) * 32);
#pragma unroll
    for(int i = 0; i < 8; ++i) {
        const float4 t4 = sp[i];
        s[4 * i] = t4.x; s[4 * i + 1] = t4.y; s[4 * i + 2] = t4.z; s[4 * i + 3] = t4.w;
    }
    // Matrix sequence per chunk: W (0), Q (1), K (2); matrix n goes to buffer n % 2.
    auto stage = [&](int n) {
        const int cc = n / 3, which = n % 3, buf = n & 1;
        if(cc >= nch) return;
        for(int e = tid; e < C * D / 4; e += blockDim.x) {
            const int i = e / (D / 4), d4 = (e % (D / 4)) * 4;
            const float* src;
            if(which == 0) {
                src = ws.W + ((static_cast<std::size_t>(cc) * kLinVHeads + h) * C + i) * D + d4;
            } else {
                const int t = min(t_base + cc * C + i, T - 1);
                src = sc.qk + ((static_cast<std::size_t>(t) * kLinKHeads + kh) * 2 + (which == 1 ? 0 : 1)) * D + d4;
            }
            const int c4 = d4 / 4;
            cp16(&S.m[buf][i][rot4(c4 / 8, c4 % 8) * 4], src);
        }
        asm volatile("cp.async.commit_group;");
    };
    stage(0);
    for(int c = 0; c < nch; ++c) {
        const int n0 = c * 3;
        // ---- W: U = Ub - W S0
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        stage(n0 + 1);
        if(tid < C) S.gam[tid] = ws.gam[(static_cast<std::size_t>(c) * kLinVHeads + h) * C + tid];
        {
            const float (*W)[RP] = S.m[n0 & 1];
            const float* ub = ws.Ub + (static_cast<std::size_t>(c) * kLinVHeads + h) * C * D;
            for(int i = 0; i < C; ++i) {
                const float4* wp = reinterpret_cast<const float4*>(&W[i][0]);
                float p = 0.f;
#pragma unroll
                for(int j = 0; j < 8; ++j) {
                    const float4 w4 = wp[rot4(kg, j)];
                    p = fmaf(w4.x, s[4 * j], p);
                    p = fmaf(w4.y, s[4 * j + 1], p);
                    p = fmaf(w4.z, s[4 * j + 2], p);
                    p = fmaf(w4.w, s[4 * j + 3], p);
                }
                p += __shfl_xor_sync(0xffffffffu, p, 1);
                p += __shfl_xor_sync(0xffffffffu, p, 2);
                if(kg == 0) S.u[i][vl] = ub[i * D + v] - p;
            }
        }
        // ---- Q: O = diag(e^gam) Q S0 (+ Qk U below)
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        stage(n0 + 2);
        {
            const float (*Q)[RP] = S.m[(n0 + 1) & 1];
            for(int i = 0; i < C; ++i) {
                const float4* qp = reinterpret_cast<const float4*>(&Q[i][0]);
                float p = 0.f;
#pragma unroll
                for(int j = 0; j < 8; ++j) {
                    const float4 q4 = qp[rot4(kg, j)];
                    p = fmaf(q4.x, s[4 * j], p);
                    p = fmaf(q4.y, s[4 * j + 1], p);
                    p = fmaf(q4.z, s[4 * j + 2], p);
                    p = fmaf(q4.w, s[4 * j + 3], p);
                }
                p += __shfl_xor_sync(0xffffffffu, p, 1);
                p += __shfl_xor_sync(0xffffffffu, p, 2);
                if(kg == 0) S.o[i][vl] = expf(S.gam[i]) * p;
            }
            const float* qk = ws.Qk + (static_cast<std::size_t>(c) * kLinVHeads + h) * C * C;
            for(int e = tid; e < C * C; e += blockDim.x) S.qk[e / C][e % C] = qk[e];
        }
        __syncthreads();
        // Rows i = kg, kg + 4, ...: O[i][v] += sum_{j <= i} Qk[i][j] U[j][v]; then write the outputs.
        for(int i = kg; i < C; i += 4) {
            float acc = S.o[i][vl];
            for(int j = 0; j <= i; ++j) acc = fmaf(S.qk[i][j], S.u[j][vl], acc);
            const int t = t_base + c * C + i;
            if(t < T) sc.o[(static_cast<std::size_t>(t) * kLinVHeads + h) * D + v] = acc;
        }
        // ---- K: S = e^{gam_C} S0 + sum_i e^{gam_C - gam_i} k_i U_i^T
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        stage(n0 + 3);
        {
            const float (*K)[RP] = S.m[(n0 + 2) & 1];
            const float gc = S.gam[C - 1], egc = expf(gc);
#pragma unroll
            for(int j = 0; j < 32; ++j) s[j] *= egc;
            for(int i = 0; i < C; ++i) {
                const float f = expf(gc - S.gam[i]) * S.u[i][vl];
                const float4* kp = reinterpret_cast<const float4*>(&K[i][0]);
#pragma unroll
                for(int j = 0; j < 8; ++j) {
                    const float4 k4 = kp[rot4(kg, j)];
                    s[4 * j] = fmaf(k4.x, f, s[4 * j]);
                    s[4 * j + 1] = fmaf(k4.y, f, s[4 * j + 1]);
                    s[4 * j + 2] = fmaf(k4.z, f, s[4 * j + 2]);
                    s[4 * j + 3] = fmaf(k4.w, f, s[4 * j + 3]);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
    for(int i = 0; i < 8; ++i) sp[i] = make_float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
}

}  // namespace kwy

std::size_t wy_scratch_floats(int max_chunks) {
    return static_cast<std::size_t>(max_chunks) * kLinVHeads * (2 * kWyChunk * kLinDim + kWyChunk * kWyChunk + kWyChunk);
}

WyScratch wy_scratch(float* base, int max_chunks) {
    WyScratch w;
    const std::size_t per = static_cast<std::size_t>(max_chunks) * kLinVHeads;
    w.W = base;
    w.Ub = w.W + per * kWyChunk * kLinDim;
    w.Qk = w.Ub + per * kWyChunk * kLinDim;
    w.gam = w.Qk + per * kWyChunk * kWyChunk;
    w.max_chunks = max_chunks;
    return w;
}

void gdn_scan_wy(GdnState st, GdnPrefillScratch sc, int T, WyScratch ws, cudaStream_t s) {
    const int prep_bytes = static_cast<int>(sizeof(kwy::PrepSmem));
    const int chunk_bytes = static_cast<int>(sizeof(kwy::ChunkSmem));
    KCHECK(cudaFuncSetAttribute(kwy::wy_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_bytes));
    KCHECK(cudaFuncSetAttribute(kwy::wy_chunk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, chunk_bytes));
    const int chunks = (T + kWyChunk - 1) / kWyChunk;
    for(int c0 = 0; c0 < chunks; c0 += ws.max_chunks) {
        const int nch = std::min(ws.max_chunks, chunks - c0);
        const int t_base = c0 * kWyChunk;
        launch_k(kwy::wy_prep_kernel, dim3(nch, kLinVHeads), 256, prep_bytes, s, sc, T, t_base, ws);
        launch_k(kwy::wy_chunk_kernel, dim3(kLinVHeads, 4), 128, chunk_bytes, s, st, sc, T, t_base, nch, ws);
    }
}

}  // namespace kestrel
