// Adapted from the fnl engine's prompt GEMM (crit-hub qwen3.8-flash-next mtp3-expert-cache): 128x128x32 CTA tiles,
// 8 warps of 64x32, A through cp.async, B converted to FP16 through registers, double-buffered shared memory.
#include "gemm.cuh"

namespace nmh {
namespace {
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
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t r[2], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_h(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

template <WFmt F, Epi E, bool G>
__global__ void __launch_bounds__(256) gemm_kernel(GemmArgs p) {
    __shared__ __align__(16) __half As[2][BM][PAD];
    __shared__ __align__(16) __half Bs[2][BN][PAD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 1, wn = warp >> 1;
    const int n0 = blockIdx.x * BN;
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
    const uint8_t* W1 = G ? p.group_base[group] + p.w_off : static_cast<const uint8_t*>(p.W);
    const uint8_t* S1 = G ? p.group_base[group] + p.s_off : p.Ws;
    const float alpha = G ? p.group_alpha[group * 2 + p.alpha_idx] : p.alpha;

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
    const int bj = tid >> 1;   // one B row per thread pair, 16 elements each
    const int bn = n0 + bj;
    const bool bvalid = bn < p.N;
    const std::size_t wbytes = F == WFmt::bf16 ? std::size_t(p.K) * 2 : F == WFmt::e4m3 ? p.K : p.K / 2;
    const uint8_t* brow_w = W1 + std::size_t(bvalid ? bn : 0) * wbytes;
    const uint8_t* brow_s = F == WFmt::fp4 ? S1 + std::size_t(bvalid ? bn : 0) * (p.K / 16) : nullptr;

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
    uint4 braw = make_uint4(0, 0, 0, 0), braw2 = make_uint4(0, 0, 0, 0);
    uint32_t bsc = 0;
    auto load_b = [&](int kt) {
        const int half = tid & 1;
        if constexpr(F == WFmt::bf16) {
            if(bvalid) {
                const uint4* src = reinterpret_cast<const uint4*>(brow_w + (std::size_t(kt) * BK + half * 16) * 2);
                braw = __ldg(src);
                braw2 = __ldg(src + 1);
            } else {
                braw = braw2 = make_uint4(0, 0, 0, 0);
            }
        } else if constexpr(F == WFmt::e4m3) {
            braw = bvalid ? __ldg(reinterpret_cast<const uint4*>(brow_w + std::size_t(kt) * BK + half * 16))
                          : make_uint4(0, 0, 0, 0);
        } else {
            if(bvalid) {
                const uint2 v = __ldg(reinterpret_cast<const uint2*>(brow_w + std::size_t(kt) * (BK / 2) + half * 8));
                braw = make_uint4(v.x, v.y, 0, 0);
                bsc = brow_s[kt * 2 + half];
            } else {
                braw = make_uint4(0, 0, 0, 0);
                bsc = 0;
            }
        }
    };
    auto store_b = [&](int stage) {
        uint32_t out[8];
        if constexpr(F == WFmt::bf16) {
            const uint32_t words[8] = {braw.x, braw.y, braw.z, braw.w, braw2.x, braw2.y, braw2.z, braw2.w};
#pragma unroll
            for(int q = 0; q < 8; ++q) out[q] = pack_h(bf16_lo(words[q]), bf16_hi(words[q]));
        } else if constexpr(F == WFmt::e4m3) {
            const uint32_t words[4] = {braw.x, braw.y, braw.z, braw.w};
#pragma unroll
            for(int q = 0; q < 4; ++q) {
                const float2 lo = e4m3x2f(words[q] & 0xffff), hi = e4m3x2f(words[q] >> 16);
                out[q * 2] = pack_h(lo.x, lo.y);
                out[q * 2 + 1] = pack_h(hi.x, hi.y);
            }
        } else {
            const float sc = e4m3f(bsc);
            const uint32_t words[2] = {braw.x, braw.y};
#pragma unroll
            for(int q = 0; q < 2; ++q)
#pragma unroll
                for(int b = 0; b < 4; ++b) {
                    const float2 v = e2m1x2f((words[q] >> (8 * b)) & 0xff);
                    out[q * 4 + b] = pack_h(v.x * sc, v.y * sc);
                }
        }
        uint4* dst = reinterpret_cast<uint4*>(&Bs[stage][bj][(tid & 1) * 16]);
        dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
        dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
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
    load_b(0);
    cp_async_commit();
    store_b(0);
    cp_async_wait_all();
    __syncthreads();
    for(int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if(more) {
            load_a(kt + 1, cur ^ 1);
            load_b(kt + 1);
            cp_async_commit();
        }
        compute(cur);
        if(more) {
            store_b(cur ^ 1);
            cp_async_wait_all();
        }
        __syncthreads();
    }

    const int qr = lane >> 2, qc = (lane & 3) * 2;
#pragma unroll
    for(int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for(int half = 0; half < 2; ++half) {
            const int row = row_begin + wm * 64 + mi * 16 + qr + half * 8;
            if(row >= row_end) continue;
#pragma unroll
            for(int ni = 0; ni < 4; ++ni) {
                const int c = n0 + wn * 32 + ni * 8 + qc;
                if(c >= p.N) continue;
                const float v0 = acc[mi][ni][half * 2] * alpha, v1 = acc[mi][ni][half * 2 + 1] * alpha;
                const bool two = c + 1 < p.N;
                if constexpr(E == Epi::f32) {
                    float* dst = p.C + std::size_t(row) * p.ldc + c;
                    dst[0] = v0;
                    if(two) dst[1] = v1;
                } else if constexpr(E == Epi::f32_add) {
                    float* dst = p.C + std::size_t(row) * p.ldc + c;
                    dst[0] += v0;
                    if(two) dst[1] += v1;
                } else if constexpr(E == Epi::f16) {
                    __half* dst = p.Cb + std::size_t(row) * p.ldc + c;
                    dst[0] = __float2half_rn(v0);
                    if(two) dst[1] = __float2half_rn(v1);
                } else {
                    __half* dst = p.Cb + std::size_t(row) * p.ldc + c;
                    dst[0] = __float2half_rn(relu2f(v0) * p.out_scale);
                    if(two) dst[1] = __float2half_rn(relu2f(v1) * p.out_scale);
                }
            }
        }
    }
}

template <WFmt F, Epi E>
void launch(const GemmArgs& p, int grid_tiles, cudaStream_t stream) {
    const int nt = (p.N + BN - 1) / BN;
    if(p.tiles) gemm_kernel<F, E, true><<<dim3(nt, grid_tiles), 256, 0, stream>>>(p);
    else gemm_kernel<F, E, false><<<dim3(nt, (p.M + BM - 1) / BM), 256, 0, stream>>>(p);
    NMH_CUDA(cudaGetLastError());
}
}  // namespace

void gemm(WFmt fmt, Epi epi, const GemmArgs& p, int grid_tiles, cudaStream_t stream) {
    if(p.K % BK) throw std::invalid_argument("gemm K must be a multiple of 32");
    if(!p.tiles && p.M == 0) return;
    if(p.tiles && grid_tiles == 0) return;
#define NMH_GEMM_CASE(F, E) \
    if(fmt == F && epi == E) return launch<F, E>(p, grid_tiles, stream);
    NMH_GEMM_CASE(WFmt::bf16, Epi::f32)
    NMH_GEMM_CASE(WFmt::bf16, Epi::f32_add)
    NMH_GEMM_CASE(WFmt::bf16, Epi::f16)
    NMH_GEMM_CASE(WFmt::e4m3, Epi::f32)
    NMH_GEMM_CASE(WFmt::e4m3, Epi::f32_add)
    NMH_GEMM_CASE(WFmt::fp4, Epi::f32)
    NMH_GEMM_CASE(WFmt::fp4, Epi::f32_add)
    NMH_GEMM_CASE(WFmt::fp4, Epi::relu2)
#undef NMH_GEMM_CASE
    throw std::invalid_argument("unsupported gemm format/epilogue");
}

}  // namespace nmh
