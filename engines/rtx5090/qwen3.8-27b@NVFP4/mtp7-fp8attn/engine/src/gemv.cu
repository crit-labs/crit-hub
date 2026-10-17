#include "gemv.cuh"

namespace kestrel {
namespace kgemv {

constexpr int kWarps = 4;  // warps per block

__device__ __forceinline__ void mma16816(float c[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                         std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ std::uint32_t e4m3x2(std::uint32_t v16) {
    std::uint32_t r;
    asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(v16));
    return r;
}
__device__ __forceinline__ std::uint32_t e2m1x2(std::uint32_t v8) {
    std::uint32_t r;
    asm("{ .reg .b8 t; .reg .b16 w; cvt.u16.u32 w, %1; cvt.u8.u16 t, w; cvt.rn.f16x2.e2m1x2 %0, t; }"
        : "=r"(r) : "r"(v8));
    return r;
}
__device__ __forceinline__ std::uint32_t hmul2u(std::uint32_t a, std::uint32_t b) {
    __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<std::uint32_t*>(&r);
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 ldg_keep(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

struct Args {
    const std::uint8_t* w;
    const float* scale;      // FP8 row scales
    const std::uint8_t* sf;  // FP4 swizzled block scales
    float alpha;
    int rows, cols, blocks_padded;
    const __half* x;
    int ldx, N;
    GemvOut out;
    int splits, steps_per_split;
    float* partial;
};

__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

// Epilogue for a finished tile: c[0..1] row g cols 2q,2q+1; c[2..3] row g+8.
__device__ __forceinline__ void store_tile(const Args& a, int row0, int g, int q, const float c[4], float s_lo,
                                           float s_hi) {
    const int n0 = 2 * q;
    if(a.splits > 1) {
        const int split = blockIdx.y;
#pragma unroll
        for(int j = 0; j < 2; ++j) {
            const int n = n0 + j;
            if(n < a.N) {
                float* p = a.partial + (static_cast<std::size_t>(split) * a.N + n) * a.rows;
                p[row0 + g] = c[j];
                p[row0 + g + 8] = c[2 + j];
            }
        }
        return;
    }
    if(a.out.epi == GemvEpi::swiglu) {
#pragma unroll
        for(int j = 0; j < 2; ++j) {
            const int n = n0 + j;
            if(n < a.N) {
                const float gate = c[j] * s_lo, up = c[2 + j] * s_hi;
                a.out.h[static_cast<std::size_t>(n) * a.out.ldh + (row0 >> 1) + g] = __float2half_rn(silu(gate) * up);
            }
        }
        return;
    }
#pragma unroll
    for(int j = 0; j < 2; ++j) {
        const int n = n0 + j;
        if(n < a.N) {
            float* y = a.out.y + static_cast<std::size_t>(n) * a.out.ldy;
            if(a.out.epi == GemvEpi::residual) {
                y[row0 + g] += c[j] * s_lo;
                y[row0 + g + 8] += c[2 + j] * s_hi;
            } else {
                y[row0 + g] = c[j] * s_lo;
                y[row0 + g + 8] = c[2 + j] * s_hi;
            }
        }
    }
}

// One step: thread q of a quad owns CH 16-byte chunks (consecutive) of rows g and g+8 of each tile.
// FP8: a chunk is 16 elements; FP4: a chunk is 32 elements (two scale blocks).
template <bool FP4, int TILES, int CH, int KW>
__global__ void __launch_bounds__((KW > 1 ? KW : kWarps) * 32) gemv_kernel(Args a) {
    pdl_wait();
    pdl_trigger();
    constexpr int kElemsPerChunk = FP4 ? 32 : 16;
    constexpr int kStepElems = 4 * CH * kElemsPerChunk;
    constexpr int kXChunks = CH * kElemsPerChunk / 8;  // uint4 of FP16 activations per step
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tile0 = KW > 1 ? blockIdx.x * TILES : (blockIdx.x * kWarps + warp) * TILES;
    if(tile0 * 16 >= a.rows) return;
    int step0, step1;
    if constexpr(KW > 1) {
        const int total = a.cols / kStepElems, per = (total + KW - 1) / KW;
        step0 = min(warp * per, total);
        step1 = min(step0 + per, total);
    } else {
        step0 = blockIdx.y * a.steps_per_split;
        step1 = min(step0 + a.steps_per_split, a.cols / kStepElems);
    }
    const bool has_x = g < a.N;
    const std::size_t row_bytes = FP4 ? a.cols / 2 : a.cols;
    const std::uint8_t* wrow[TILES][2];
    int rr[TILES][2];
#pragma unroll
    for(int t = 0; t < TILES; ++t) {
        rr[t][0] = (tile0 + t) * 16 + g;
        rr[t][1] = rr[t][0] + 8;
        wrow[t][0] = a.w + static_cast<std::size_t>(rr[t][0]) * row_bytes + q * CH * 16;
        wrow[t][1] = a.w + static_cast<std::size_t>(rr[t][1]) * row_bytes + q * CH * 16;
    }
    const __half* xrow = a.x + static_cast<std::size_t>(has_x ? g : 0) * a.ldx + q * CH * kElemsPerChunk;
    float c[TILES][4] = {};
    uint4 wb[TILES][2][CH], xb[kXChunks];
    std::uint32_t sb[TILES][2][FP4 ? CH : 1];
    auto load = [&](int step, uint4 (&w)[TILES][2][CH], uint4 (&x)[kXChunks], std::uint32_t (&s)[TILES][2][FP4 ? CH : 1]) {
        const std::size_t kb = static_cast<std::size_t>(step) * 4 * CH * 16;  // byte offset in the row
#pragma unroll
        for(int t = 0; t < TILES; ++t)
#pragma unroll
            for(int h = 0; h < 2; ++h) {
#pragma unroll
                for(int ch = 0; ch < CH; ++ch) w[t][h][ch] = ldg_stream(wrow[t][h] + kb + ch * 16);
                if constexpr(FP4) {
                    const int block = (step * kStepElems + q * CH * 32) / 16;
#pragma unroll
                    for(int ch = 0; ch < CH; ++ch)
                        s[t][h][ch] = *reinterpret_cast<const unsigned short*>(
                            a.sf + sf_offset(rr[t][h], block + 2 * ch, a.blocks_padded));
                }
            }
        const __half* xp = xrow + static_cast<std::size_t>(step) * kStepElems;
#pragma unroll
        for(int i = 0; i < kXChunks; ++i) x[i] = has_x ? ldg_keep(xp + 8 * i) : make_uint4(0, 0, 0, 0);
    };
    if(step0 < step1) load(step0, wb, xb, sb);
    for(int step = step0; step < step1; ++step) {
        uint4 w[TILES][2][CH], x[kXChunks];
        std::uint32_t s[TILES][2][FP4 ? CH : 1];
#pragma unroll
        for(int t = 0; t < TILES; ++t)
#pragma unroll
            for(int h = 0; h < 2; ++h) {
#pragma unroll
                for(int ch = 0; ch < CH; ++ch) w[t][h][ch] = wb[t][h][ch];
#pragma unroll
                for(int ch = 0; ch < (FP4 ? CH : 1); ++ch) s[t][h][ch] = sb[t][h][ch];
            }
#pragma unroll
        for(int i = 0; i < kXChunks; ++i) x[i] = xb[i];
        if(step + 1 < step1) load(step + 1, wb, xb, sb);
        const std::uint32_t* xs = reinterpret_cast<const std::uint32_t*>(x);  // FP16 pairs
#pragma unroll
        for(int t = 0; t < TILES; ++t) {
#pragma unroll
            for(int ch = 0; ch < CH; ++ch) {
                const std::uint32_t* lo = reinterpret_cast<const std::uint32_t*>(&w[t][0][ch]);
                const std::uint32_t* hi = reinterpret_cast<const std::uint32_t*>(&w[t][1][ch]);
                if constexpr(!FP4) {
#pragma unroll
                    for(int j = 0; j < 4; ++j) {
                        // Elements 4j..4j+3 of this chunk: pair 2j -> slots (0,1), pair 2j+1 -> slots (2,3).
                        const std::uint32_t l0 = e4m3x2(lo[j] & 0xffff), l1 = e4m3x2(lo[j] >> 16);
                        const std::uint32_t h0 = e4m3x2(hi[j] & 0xffff), h1 = e4m3x2(hi[j] >> 16);
                        const int p = ch * 8 + 2 * j;
                        mma16816(c[t], l0, h0, l1, h1, xs[p], xs[p + 1]);
                    }
                } else {
                    const std::uint32_t slo = e4m3x2(s[t][0][ch]), shi = e4m3x2(s[t][1][ch]);
                    const std::uint32_t sl[2] = {__byte_perm(slo, 0, 0x1010), __byte_perm(slo, 0, 0x3232)};
                    const std::uint32_t sh[2] = {__byte_perm(shi, 0, 0x1010), __byte_perm(shi, 0, 0x3232)};
#pragma unroll
                    for(int j = 0; j < 8; ++j) {
                        // Elements 4j..4j+3: byte 2j holds 4j, 4j+1; byte 2j+1 holds 4j+2, 4j+3.
                        const std::uint32_t wl = lo[j >> 1] >> ((j & 1) * 16), wh = hi[j >> 1] >> ((j & 1) * 16);
                        const std::uint32_t l0 = hmul2u(e2m1x2(wl & 0xff), sl[j >> 2]);
                        const std::uint32_t l1 = hmul2u(e2m1x2((wl >> 8) & 0xff), sl[j >> 2]);
                        const std::uint32_t h0 = hmul2u(e2m1x2(wh & 0xff), sh[j >> 2]);
                        const std::uint32_t h1 = hmul2u(e2m1x2((wh >> 8) & 0xff), sh[j >> 2]);
                        const int p = ch * 16 + 2 * j;
                        mma16816(c[t], l0, h0, l1, h1, xs[p], xs[p + 1]);
                    }
                }
            }
        }
    }
    if constexpr(KW > 1) {
        __shared__ float red[KW][TILES][32][4];
#pragma unroll
        for(int t = 0; t < TILES; ++t)
#pragma unroll
            for(int i = 0; i < 4; ++i) red[warp][t][lane][i] = c[t][i];
        __syncthreads();
        if(warp != 0) return;
        float ss[2] = {0.f, 0.f};
#pragma unroll
        for(int t = 0; t < TILES; ++t) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for(int w = 0; w < KW; ++w)
#pragma unroll
                for(int i = 0; i < 4; ++i) v[i] += red[w][t][lane][i];
            const int row0 = (tile0 + t) * 16;
            const float s_lo = FP4 ? a.alpha : a.scale[row0 + g], s_hi = FP4 ? a.alpha : a.scale[row0 + g + 8];
            if(a.out.epi == GemvEpi::residual_norm || a.out.epi == GemvEpi::residual) {
#pragma unroll
                for(int j = 0; j < 2; ++j) {
                    const int n = 2 * q + j;
                    if(n < a.N) {
                        float* y = a.out.y + static_cast<std::size_t>(n) * a.out.ldy;
                        const float y0 = y[row0 + g] + v[j] * s_lo, y1 = y[row0 + g + 8] + v[2 + j] * s_hi;
                        y[row0 + g] = y0;
                        y[row0 + g + 8] = y1;
                        ss[j] += y0 * y0 + y1 * y1;
                    }
                }
            } else {
                Args b = a;
                b.splits = 1;
                store_tile(b, row0, g, q, v, s_lo, s_hi);
            }
        }
        if(a.out.epi == GemvEpi::residual_norm) {
            // Column sums of squares over this block's rows: lanes with the same q hold the same columns.
#pragma unroll
            for(int j = 0; j < 2; ++j) {
                ss[j] += __shfl_xor_sync(0xffffffffu, ss[j], 4);
                ss[j] += __shfl_xor_sync(0xffffffffu, ss[j], 8);
                ss[j] += __shfl_xor_sync(0xffffffffu, ss[j], 16);
            }
            if(g == 0)
#pragma unroll
                for(int j = 0; j < 2; ++j)
                    if(2 * q + j < a.N) a.out.ssq[static_cast<std::size_t>(2 * q + j) * gridDim.x + blockIdx.x] = ss[j];
        }
        return;
    }
#pragma unroll
    for(int t = 0; t < TILES; ++t) {
        const int row0 = (tile0 + t) * 16;
        if(row0 >= a.rows) break;
        float s_lo, s_hi;
        if constexpr(FP4) {
            s_lo = s_hi = a.splits > 1 ? 1.f : a.alpha;
        } else {
            s_lo = a.splits > 1 ? 1.f : a.scale[row0 + g];
            s_hi = a.splits > 1 ? 1.f : a.scale[row0 + g + 8];
        }
        store_tile(a, row0, g, q, c[t], s_lo, s_hi);
    }
}

// Sum split partials in split order, then apply the row scale and the epilogue.
__global__ void gemv_reduce_kernel(const float* __restrict__ partial, int splits, int N, int rows,
                                   const float* __restrict__ scale, float alpha, GemvOut out) {
    pdl_wait();
    pdl_trigger();
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if(row >= rows || n >= N) return;
    float acc = 0.f;
    for(int s = 0; s < splits; ++s) acc += partial[(static_cast<std::size_t>(s) * N + n) * rows + row];
    const float v = acc * (scale ? scale[row] : alpha);
    float* y = out.y + static_cast<std::size_t>(n) * out.ldy + row;
    if(out.epi == GemvEpi::residual) *y += v;
    else *y = v;
}

// RMSNorm after a residual_norm projection: blocks sum the per-block sums of squares in order, then normalize
// their 256-row slice of each column.
__global__ void __launch_bounds__(256) rows_norm_kernel(const float* __restrict__ y, int ldy, const float* __restrict__ ssq,
                                                        int parts, int rows, const float* __restrict__ w, __half* __restrict__ out,
                                                        int ldo) {
    pdl_wait();
    pdl_trigger();
    __shared__ float r_s;
    const int n = blockIdx.y;
    if(threadIdx.x < 32) {
        float t = 0.f;
        for(int i = threadIdx.x; i < parts; i += 32) t += ssq[static_cast<std::size_t>(n) * parts + i];
#pragma unroll
        for(int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if(threadIdx.x == 0) r_s = rsqrtf(t / rows + kEps);
    }
    __syncthreads();
    const int row = blockIdx.x * 256 + threadIdx.x;
    if(row < rows)
        out[static_cast<std::size_t>(n) * ldo + row] =
            __float2half_rn(y[static_cast<std::size_t>(n) * ldy + row] * r_s * (1.f + w[row]));
}

// Split reduction fused with the residual add and the next RMSNorm: one block per column.
__global__ void __launch_bounds__(1024) gemv_reduce_norm_kernel(const float* __restrict__ partial, int splits, int N, int rows,
                                                                const float* __restrict__ scale, float alpha, GemvOut out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[32];
    const int n = blockIdx.x;
    float* y = out.y + static_cast<std::size_t>(n) * out.ldy;
    constexpr int kPer = 8;
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for(int i = 0; i < kPer; ++i) {
        const int row = threadIdx.x + i * blockDim.x;
        v[i] = 0.f;
        if(row < rows) {
            float acc = 0.f;
            for(int s = 0; s < splits; ++s) acc += partial[(static_cast<std::size_t>(s) * N + n) * rows + row];
            v[i] = y[row] + acc * (scale ? scale[row] : alpha);
            y[row] = v[i];
            ss += v[i] * v[i];
        }
    }
#pragma unroll
    for(int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for(int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) t += red[i];
    const float r = rsqrtf(t / rows + kEps);
#pragma unroll
    for(int i = 0; i < kPer; ++i) {
        const int row = threadIdx.x + i * blockDim.x;
        if(row < rows) out.h[static_cast<std::size_t>(n) * out.ldh + row] = __float2half_rn(v[i] * r * (1.f + out.norm_w[row]));
    }
}

template <bool FP4, int TILES, int CH>
void launch_t(Args a, cudaStream_t stream) {
    constexpr int step_elems = 4 * CH * (FP4 ? 32 : 16);
    if(a.cols % step_elems || a.rows % (16 * TILES)) throw std::invalid_argument("gemv: unsupported shape");
    const int steps = a.cols / step_elems;
    a.steps_per_split = (steps + a.splits - 1) / a.splits;
    const int warps = a.rows / (16 * TILES);
    dim3 grid((warps + kWarps - 1) / kWarps, a.splits);
    launch_k(gemv_kernel<FP4, TILES, CH, 1>, grid, kWarps * 32, 0, stream, a);
    KLAUNCH();
}

// Intra-block K split: 8 warps, one 16-row tile per block.
template <bool FP4, int CH>
void launch_ksplit(Args a, cudaStream_t stream) {
    constexpr int step_elems = 4 * CH * (FP4 ? 32 : 16);
    if(a.cols % step_elems || a.rows % 16) throw std::invalid_argument("gemv: unsupported shape");
    launch_k(gemv_kernel<FP4, 1, CH, 8>, a.rows / 16, 8 * 32, 0, stream, a);
    if(a.out.epi == GemvEpi::residual_norm)
        launch_k(rows_norm_kernel, dim3((a.rows + 255) / 256, a.N), 256, 0, stream, a.out.y, a.out.ldy, a.out.ssq,
                 a.rows / 16, a.rows, a.out.norm_w, a.out.h, a.out.ldh);
}

int g_variant = 0;  // test hook: 0 default

void launch(bool fp4, Args a, cudaStream_t stream) {
    if(a.N < 1 || a.N > kMaxCols) throw std::invalid_argument("gemv: 1..8 columns");
    if(a.splits < 1) a.splits = 1;
    if(a.splits > 1 && a.out.epi == GemvEpi::swiglu) throw std::invalid_argument("gemv: swiglu needs one split");
    if(a.out.epi == GemvEpi::residual_norm && (a.splits < 2 || !a.out.ssq))
        throw std::invalid_argument("gemv: residual_norm needs splits > 1 and ssq scratch");
    if(a.splits > 1) {
        fp4 ? launch_ksplit<true, 2>(a, stream) : launch_ksplit<false, 2>(a, stream);
        return;
    }
    switch(g_variant) {
    case 1: fp4 ? launch_t<true, 1, 2>(a, stream) : launch_t<false, 1, 2>(a, stream); break;
    case 2: fp4 ? launch_t<true, 2, 1>(a, stream) : launch_t<false, 2, 1>(a, stream); break;
    case 3: fp4 ? launch_t<true, 1, 4>(a, stream) : launch_t<false, 1, 4>(a, stream); break;
    case 4: fp4 ? launch_t<true, 2, 2>(a, stream) : launch_t<false, 2, 2>(a, stream); break;
    case 5: fp4 ? launch_t<true, 4, 1>(a, stream) : launch_t<false, 4, 1>(a, stream); break;
    case 6: fp4 ? launch_t<true, 4, 2>(a, stream) : launch_t<false, 4, 2>(a, stream); break;
    default:
        if(fp4) launch_t<true, 2, 2>(a, stream);
        else launch_t<false, 2, 2>(a, stream);
        break;
    }
    if(a.out.epi == GemvEpi::residual_norm) {
        launch_k(gemv_reduce_norm_kernel, a.N, 1024, 0, stream, a.partial, a.splits, a.N, a.rows, fp4 ? nullptr : a.scale,
                 a.alpha, a.out);
    } else if(a.splits > 1) {
        dim3 rg((a.rows + 255) / 256, a.N);
        launch_k(gemv_reduce_kernel, rg, 256, 0, stream, a.partial, a.splits, a.N, a.rows, fp4 ? nullptr : a.scale,
                                                   a.alpha, a.out);
        KLAUNCH();
    }
}

}  // namespace kgemv
using namespace kgemv;

void gemv_set_variant(int v) { g_variant = v; }

void gemv_fp8(const Fp8Mat& m, const __half* x, int ldx, int N, const GemvOut& out, int splits, float* partial,
              cudaStream_t stream) {
    Args a{m.w, m.scale, nullptr, 1.f, m.rows, m.cols, 0, x, ldx, N, out, splits, 0, partial};
    launch(false, a, stream);
}

void gemv_fp4(const Fp4Mat& m, const __half* x, int ldx, int N, const GemvOut& out, int splits, float* partial,
              cudaStream_t stream) {
    Args a{m.w, nullptr, m.sf, m.alpha, m.rows, m.cols, sf_blocks_padded(m.cols), x, ldx, N, out, splits, 0, partial};
    launch(true, a, stream);
}

}  // namespace kestrel
