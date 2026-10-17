#include "gemm.cuh"

namespace fnl {

namespace kern {
constexpr int BM = 128, BN = 128, BK = 32, PAD = 40;  // smem row stride in f16 elements (80 bytes)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t r[2], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Two floats to an f16 pair, saturating to the finite range.
__device__ __forceinline__ uint32_t pack_h(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// Weight-tile source for one CTA: row pointers resolved per B-tile row.
struct BRow {
    const uint8_t* w;   // start of the weight row (bytes)
    const uint8_t* s;   // start of the scale row (mx8/fp4)
    bool valid;
};

template <WFmt F>
struct BStage {
    uint4 v;       // raw weight bytes for this thread
    uint32_t sc;   // scale byte(s)
};

template <WFmt F, Epi E, bool G>
__global__ void __launch_bounds__(256) gemm_kernel(GemmArgs p) {
    __shared__ __align__(16) __half As[2][BM][PAD];
    __shared__ __align__(16) __half Bs[2][BN][PAD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 1, wn = warp >> 1;
    const int n0 = blockIdx.x * BN;

    // Row range of this CTA.
    int group = 0, row_begin, row_end;
    if constexpr(G) {
        if(int(blockIdx.y) >= *p.ntiles) return;
        const int4 t = p.tiles[blockIdx.y];
        group = t.x;
        row_begin = t.y;
        row_end = t.z;
    } else {
        row_begin = blockIdx.y * BM;
        row_end = min(p.M, row_begin + BM);
    }
    const uint8_t* gbase = G ? p.group_base[group] : nullptr;
    const uint8_t* W1 = G ? gbase + p.w_off : static_cast<const uint8_t*>(p.W);
    const uint8_t* S1 = G ? gbase + p.s_off : p.Ws;
    const uint8_t* W2 = G ? gbase + p.w2_off : static_cast<const uint8_t*>(p.W2);
    const uint8_t* S2 = G ? gbase + p.s2_off : p.Ws2;
    const float alpha = G ? (p.group_alpha ? p.group_alpha[group * 3 + p.alpha_idx] : 1.f) : p.alpha;
    const float alpha2 = G ? (p.group_alpha ? p.group_alpha[group * 3 + p.alpha2_idx] : 1.f) : p.alpha2;

    // A rows owned by this thread for cp.async (2 chunks: rows r0, r0 + 64).
    const int a_chunk_col = (tid & 3) * 8;
    const __half* a_src[2];
    bool a_valid[2];
#pragma unroll
    for(int i = 0; i < 2; ++i) {
        const int r = (tid >> 2) + i * 64;
        const int logical = row_begin + r;
        a_valid[i] = logical < row_end;
        const int arow = a_valid[i] ? (p.a_rows ? p.a_rows[logical] : logical) : 0;
        a_src[i] = p.A + std::size_t(arow) * p.lda + a_chunk_col;
    }
    // B row owned by this thread.
    const int bj = tid >> 1;  // one B row per thread pair, 16 elements each
    auto b_row = [&](int j) -> BRow {
        int n = n0 + j;
        const uint8_t* w = W1;
        const uint8_t* s = S1;
        bool valid;
        int row;
        if constexpr(E == Epi::act) {
            const int grp = j >> 3, within = j & 7;
            row = (n0 >> 1) + (grp >> 1) * 8 + within;
            if(grp & 1) {
                w = W2;
                s = S2;
            }
            valid = row < (p.N >> 1);
        } else {
            row = n;
            valid = n < p.N;
        }
        if(!valid) row = 0;
        const std::size_t wbytes = (F == WFmt::bf16 || F == WFmt::f16) ? std::size_t(p.K) * 2 : F == WFmt::mx8 ? p.K : p.K / 2;
        const std::size_t sbytes = F == WFmt::mx8 ? p.K / 32 : p.K / 16;
        return {w + std::size_t(row) * wbytes, s ? s + std::size_t(row) * sbytes : nullptr, valid};
    };
    BRow brow[1];
    brow[0] = b_row(bj);

    const int KT = p.K / BK;
    float acc[4][4][4];
#pragma unroll
    for(int i = 0; i < 4; ++i)
#pragma unroll
        for(int j = 0; j < 4; ++j)
#pragma unroll
            for(int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

    auto load_a = [&](int kt, int stage) {
#pragma unroll
        for(int i = 0; i < 2; ++i) {
            const int r = (tid >> 2) + i * 64;
            cp_async16(&As[stage][r][a_chunk_col], a_src[i] + kt * BK, a_valid[i]);
        }
    };
    // B through registers: load raw bytes, convert to f16 tiles in shared memory.
    uint4 braw = make_uint4(0, 0, 0, 0), braw2 = make_uint4(0, 0, 0, 0);
    uint32_t bsc = 0;
    auto load_b = [&](int kt, int stage) {
        (void)stage;
        if constexpr(F == WFmt::bf16 || F == WFmt::f16) {
            const int half = tid & 1;
            if(brow[0].valid) {
                const uint4* src = reinterpret_cast<const uint4*>(brow[0].w + (std::size_t(kt) * BK + half * 16) * 2);
                braw = __ldg(src);
                braw2 = __ldg(src + 1);
            } else {
                braw = braw2 = make_uint4(0, 0, 0, 0);
            }
        } else if constexpr(F == WFmt::mx8) {
            const int half = tid & 1;
            if(brow[0].valid) {
                braw = __ldg(reinterpret_cast<const uint4*>(brow[0].w + std::size_t(kt) * BK + half * 16));
                bsc = brow[0].s[kt];
            } else {
                braw = make_uint4(0, 0, 0, 0);
                bsc = 127;
            }
        } else {
            const int half = tid & 1;
            if(brow[0].valid) {
                const uint2 v = __ldg(reinterpret_cast<const uint2*>(brow[0].w + std::size_t(kt) * (BK / 2) + half * 8));
                braw = make_uint4(v.x, v.y, 0, 0);
                bsc = brow[0].s[kt * 2 + half];
            } else {
                braw = make_uint4(0, 0, 0, 0);
                bsc = 0;
            }
        }
    };
    auto store_b = [&](int stage) {
        if constexpr(F == WFmt::f16) {
            uint4* dst = reinterpret_cast<uint4*>(&Bs[stage][bj][(tid & 1) * 16]);
            dst[0] = braw;
            dst[1] = braw2;
        } else if constexpr(F == WFmt::bf16) {
            const uint32_t words[8] = {braw.x, braw.y, braw.z, braw.w, braw2.x, braw2.y, braw2.z, braw2.w};
            uint32_t out[8];
#pragma unroll
            for(int q = 0; q < 8; ++q) out[q] = pack_h(bf16_lo(words[q]), bf16_hi(words[q]));
            uint4* dst = reinterpret_cast<uint4*>(&Bs[stage][bj][(tid & 1) * 16]);
            dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
            dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
        } else if constexpr(F == WFmt::mx8) {
            const float sc = ue8m0f(bsc);
            const uint32_t words[4] = {braw.x, braw.y, braw.z, braw.w};
            uint32_t out[8];
#pragma unroll
            for(int q = 0; q < 4; ++q) {
                const float2 lo = e4m3x2f(words[q] & 0xffff), hi = e4m3x2f(words[q] >> 16);
                out[q * 2] = pack_h(lo.x * sc, lo.y * sc);
                out[q * 2 + 1] = pack_h(hi.x * sc, hi.y * sc);
            }
            uint4* dst = reinterpret_cast<uint4*>(&Bs[stage][bj][(tid & 1) * 16]);
            dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
            dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
        } else if constexpr(F == WFmt::fp4) {
            const float sc = e4m3f(bsc);
            const uint32_t words[2] = {braw.x, braw.y};
            uint32_t out[8];
#pragma unroll
            for(int q = 0; q < 2; ++q)
#pragma unroll
                for(int b = 0; b < 4; ++b) {
                    const float2 v = e2m1x2f((words[q] >> (8 * b)) & 0xff);
                    out[q * 4 + b] = pack_h(v.x * sc, v.y * sc);
                }
            uint4* dst = reinterpret_cast<uint4*>(&Bs[stage][bj][(tid & 1) * 16]);
            dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
            dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
        }
    };
    auto compute = [&](int stage) {
#pragma unroll
        for(int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4], bf[4][2];
#pragma unroll
            for(int mi = 0; mi < 4; ++mi) {
                const int r = wm * 64 + mi * 16 + (lane & 15);
                const int c = kk + (lane >> 4) * 8;
                ldmatrix_x4(af[mi], &As[stage][r][c]);
            }
#pragma unroll
            for(int ni = 0; ni < 4; ++ni) {
                const int r = wn * 32 + ni * 8 + (lane & 7);
                const int c = kk + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[ni], &Bs[stage][r][c]);
            }
#pragma unroll
            for(int mi = 0; mi < 4; ++mi)
#pragma unroll
                for(int ni = 0; ni < 4; ++ni) mma_f16(acc[mi][ni], af[mi], bf[ni]);
        }
    };

    load_a(0, 0);
    load_b(0, 0);
    cp_async_commit();
    store_b(0);
    cp_async_wait_all();
    __syncthreads();
    for(int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if(more) {
            load_a(kt + 1, cur ^ 1);
            load_b(kt + 1, cur ^ 1);
            cp_async_commit();
        }
        compute(cur);
        if(more) {
            store_b(cur ^ 1);
            cp_async_wait_all();
        }
        __syncthreads();
    }

    // Epilogue.
    const int qr = lane >> 2, qc = (lane & 3) * 2;
    if constexpr(E == Epi::idx) {
        // Rows are (query, index head) with 4 heads per query: C[query][col] = sum_h relu(acc) / sqrt(128).
#pragma unroll
        for(int mi = 0; mi < 4; ++mi)
#pragma unroll
            for(int half = 0; half < 2; ++half)
#pragma unroll
                for(int ni = 0; ni < 4; ++ni) {
                    float v0 = fmaxf(acc[mi][ni][half * 2], 0.f), v1 = fmaxf(acc[mi][ni][half * 2 + 1], 0.f);
                    v0 += __shfl_xor_sync(0xffffffffu, v0, 4);
                    v0 += __shfl_xor_sync(0xffffffffu, v0, 8);
                    v1 += __shfl_xor_sync(0xffffffffu, v1, 4);
                    v1 += __shfl_xor_sync(0xffffffffu, v1, 8);
                    const int row = row_begin + wm * 64 + mi * 16 + qr + half * 8;
                    const int c = n0 + wn * 32 + ni * 8 + qc;
                    if((lane & 12) == 0 && row < row_end && c < p.N) {
                        float* dst = p.C + std::size_t(row / 4) * p.ldc + c;
                        dst[0] = v0 * 0.08838834764831845f;
                        if(c + 1 < p.N) dst[1] = v1 * 0.08838834764831845f;
                    }
                }
        return;
    }
#pragma unroll
    for(int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for(int half = 0; half < 2; ++half) {
            const int r = wm * 64 + mi * 16 + qr + half * 8;
            const int row = row_begin + r;
            if(row >= row_end) continue;
            if constexpr(E == Epi::act) {
#pragma unroll
                for(int pi = 0; pi < 2; ++pi) {
                    const int f = (n0 >> 1) + (2 * wn + pi) * 8 + qc;
                    if(f >= (p.N >> 1)) continue;
                    const float g0 = acc[mi][2 * pi][half * 2] * alpha, g1 = acc[mi][2 * pi][half * 2 + 1] * alpha;
                    const float u0 = acc[mi][2 * pi + 1][half * 2] * alpha2;
                    const float u1 = acc[mi][2 * pi + 1][half * 2 + 1] * alpha2;
                    *reinterpret_cast<uint32_t*>(p.Cb + std::size_t(row) * p.ldc + f) =
                        pack_h(siluf(g0) * u0, siluf(g1) * u1);
                }
            } else {
#pragma unroll
                for(int ni = 0; ni < 4; ++ni) {
                    const int c = n0 + wn * 32 + ni * 8 + qc;
                    if(c >= p.N) continue;
                    float v0 = acc[mi][ni][half * 2], v1 = acc[mi][ni][half * 2 + 1];
                    const bool two = c + 1 < p.N;
                    if constexpr(E == Epi::f32) {
                        float* dst = p.C + std::size_t(row) * p.ldc + c;
                        dst[0] = v0 * alpha;
                        if(two) dst[1] = v1 * alpha;
                    } else if constexpr(E == Epi::f32_sigmoid) {
                        float* dst = p.C + std::size_t(row) * p.ldc + c;
                        dst[0] = sigmoidf(v0);
                        if(two) dst[1] = sigmoidf(v1);
                    } else if constexpr(E == Epi::bf16_siluq) {
                        __half* dst = p.Cb + std::size_t(row) * p.ldc + c;
                        dst[0] = __float2half_rn(siluf(v0 / float(kStreams)));
                        if(two) dst[1] = __float2half_rn(siluf(v1 / float(kStreams)));
                    } else {
                        __half* dst = p.Cb + std::size_t(row) * p.ldc + c;
                        dst[0] = __float2half_rn(v0 * alpha);
                        if(two) dst[1] = __float2half_rn(v1 * alpha);
                    }
                }
            }
        }
    }
}

template <WFmt F, Epi E>
void launch(const GemmArgs& p, int grid_tiles, cudaStream_t stream) {
    const int nt = (p.N + BN - 1) / BN;
    if(p.tiles) {
        gemm_kernel<F, E, true><<<dim3(nt, grid_tiles), 256, 0, stream>>>(p);
    } else {
        gemm_kernel<F, E, false><<<dim3(nt, (p.M + BM - 1) / BM), 256, 0, stream>>>(p);
    }
    FNL_CUDA(cudaGetLastError());
}
}  // namespace kern
using namespace kern;

void gemm(WFmt fmt, Epi epi, const GemmArgs& p, int grid_tiles, cudaStream_t stream) {
    if(p.K % BK) throw std::invalid_argument("gemm K must be a multiple of 32");
    if(!p.tiles && p.M == 0) return;
#define FNL_GEMM_CASE(F, E) \
    if(fmt == F && epi == E) return launch<F, E>(p, grid_tiles, stream);
    FNL_GEMM_CASE(WFmt::bf16, Epi::f32)
    FNL_GEMM_CASE(WFmt::bf16, Epi::bf16_siluq)
    FNL_GEMM_CASE(WFmt::bf16, Epi::f32_sigmoid)
    FNL_GEMM_CASE(WFmt::bf16, Epi::bf16)
    FNL_GEMM_CASE(WFmt::f16, Epi::idx)
    FNL_GEMM_CASE(WFmt::mx8, Epi::f32)
    FNL_GEMM_CASE(WFmt::mx8, Epi::act)
    FNL_GEMM_CASE(WFmt::fp4, Epi::f32)
    FNL_GEMM_CASE(WFmt::fp4, Epi::act)
#undef FNL_GEMM_CASE
    throw std::invalid_argument("unsupported gemm format/epilogue");
}

}  // namespace fnl
