#include "gemv.cuh"

namespace nmh {
namespace {

constexpr int kWarps = 8;
constexpr std::size_t kStageMax = 96 * 1024;   // activations staged in shared memory up to this size

// Quarter q of the 16-float chunk c in staged activations, rotated so the 8 lanes of a quarter-warp (8
// consecutive chunks) read distinct banks.
__device__ __forceinline__ int sw16(int c, int q) { return c * 16 + 4 * (q ^ ((c >> 1) & 3)); }

template <WFormat F>
struct Chunk {
    uint4 a, b;
    uint32_t s;
};

template <WFormat F>
__device__ __forceinline__ void load_chunk(Chunk<F>& k, const GemvSeg& seg, int row, int cols, int c) {
    if constexpr(F == WFormat::fp8) {
        k.a = __ldg(reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(seg.w) + std::size_t(row) * cols) + c);
    } else if constexpr(F == WFormat::bf16) {
        const uint4* p = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(seg.w) + std::size_t(row) * cols) + 2 * c;
        k.a = __ldg(p);
        k.b = __ldg(p + 1);
    } else {
        const uint2 v = __ldg(reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(seg.w) + std::size_t(row) * (cols / 2)) + c);
        k.a = make_uint4(v.x, v.y, 0, 0);
        k.s = __ldg(seg.s + std::size_t(row) * (cols / 16) + c);
    }
}

// The 16 weights of a chunk as floats.
template <WFormat F>
__device__ __forceinline__ void decode_chunk(const Chunk<F>& k, float* f) {
    if constexpr(F == WFormat::fp8) {
        const uint32_t words[4] = {k.a.x, k.a.y, k.a.z, k.a.w};
#pragma unroll
        for(int q = 0; q < 4; ++q) {
            const float2 lo = e4m3x2f(words[q] & 0xffff), hi = e4m3x2f(words[q] >> 16);
            f[4 * q] = lo.x;
            f[4 * q + 1] = lo.y;
            f[4 * q + 2] = hi.x;
            f[4 * q + 3] = hi.y;
        }
    } else if constexpr(F == WFormat::bf16) {
        const uint32_t words[8] = {k.a.x, k.a.y, k.a.z, k.a.w, k.b.x, k.b.y, k.b.z, k.b.w};
#pragma unroll
        for(int q = 0; q < 8; ++q) {
            f[2 * q] = bf16_lo(words[q]);
            f[2 * q + 1] = bf16_hi(words[q]);
        }
    } else {
        decode16_e2m1(make_uint2(k.a.x, k.a.y), f);
    }
}

// Rows of segment-mapped matrices against T tokens. A warp takes R consecutive rows and issues the loads of the
// next row before the arithmetic of the current one. Chunks c = lane + 32 i, i < CPL.
template <WFormat F, Out O, int T, int CPL, int R, bool STAGED>
__global__ void __launch_bounds__(kWarps * 32) gemv_kernel(Gemv g) {
    extern __shared__ __align__(16) float xs[];
    const int chunks = g.cols / 16;
    if constexpr(STAGED) {
        for(int i = threadIdx.x; i < T * chunks * 4; i += blockDim.x) {
            const int t = i / (chunks * 4);
            const int rem = i - t * chunks * 4;
            const int c = rem >> 2, q = rem & 3;
            *reinterpret_cast<float4*>(xs + t * g.cols + sw16(c, q)) =
                *reinterpret_cast<const float4*>(g.x + std::size_t(t) * g.ldx + c * 16 + q * 4);
        }
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * kWarps + warp) * R;
    Chunk<F> w[2][CPL];
    int srow[2] = {0, 0}, sidx[2] = {-1, -1};
#pragma unroll
    for(int r = 0; r <= R; ++r) {
        if(r < R) {
            const int k = r & 1;
            int row = row0 + r, s = 0;
            while(s < g.nseg && row >= g.seg[s].rows) row -= g.seg[s++].rows;
            sidx[k] = s < g.nseg ? s : -1;
            srow[k] = row;
            if(sidx[k] >= 0) {
                const GemvSeg& seg = g.seg[s];
#pragma unroll
                for(int i = 0; i < CPL; ++i) {
                    const int c = lane + 32 * i;
                    if(c < chunks) load_chunk<F>(w[k][i], seg, row, g.cols, c);
                }
            }
        }
        if(r > 0) {
            const int k = (r - 1) & 1;
            if(sidx[k] >= 0) {
                float acc[T];
#pragma unroll
                for(int t = 0; t < T; ++t) acc[t] = 0.f;
#pragma unroll
                for(int i = 0; i < CPL; ++i) {
                    const int c = lane + 32 * i;
                    if(c < chunks) {
                        float f[16];
                        decode_chunk<F>(w[k][i], f);
                        const float sc = F == WFormat::fp4 ? e4m3f(w[k][i].s) : 1.f;
#pragma unroll
                        for(int t = 0; t < T; ++t) {
                            float d = 0.f;
#pragma unroll
                            for(int q = 0; q < 4; ++q) {
                                float4 v;
                                if constexpr(STAGED) v = *reinterpret_cast<const float4*>(xs + t * g.cols + sw16(c, q));
                                else v = __ldg(reinterpret_cast<const float4*>(g.x + std::size_t(t) * g.ldx + c * 16) + q);
                                d = fmaf(f[4 * q], v.x, d);
                                d = fmaf(f[4 * q + 1], v.y, d);
                                d = fmaf(f[4 * q + 2], v.z, d);
                                d = fmaf(f[4 * q + 3], v.w, d);
                            }
                            if constexpr(F == WFormat::fp4) acc[t] = fmaf(d, sc, acc[t]);
                            else acc[t] += d;
                        }
                    }
                }
                const GemvSeg& seg = g.seg[sidx[k]];
#pragma unroll
                for(int t = 0; t < T; ++t) {
                    const float v = warp_sum(acc[t]) * seg.alpha;
                    if(lane == 0) {
                        float* y = seg.y + std::size_t(t) * seg.ldy + srow[k];
                        if constexpr(O == Out::store) *y = v;
                        else if constexpr(O == Out::add) *y += v;
                        else *y = relu2f(v);
                    }
                }
            }
        }
    }
}

template <WFormat F, Out O, int T, int CPL, int R>
void launch_rows(const Gemv& g, int rows, cudaStream_t stream) {
    const std::size_t smem = std::size_t(T) * g.cols * sizeof(float);
    const int blocks = (rows + kWarps * R - 1) / (kWarps * R);
    if(smem <= kStageMax) {
        auto kernel = gemv_kernel<F, O, T, CPL, R, true>;
        NMH_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kStageMax)));
        kernel<<<blocks, kWarps * 32, smem, stream>>>(g);
    } else {
        gemv_kernel<F, O, T, CPL, R, false><<<blocks, kWarps * 32, 0, stream>>>(g);
    }
    NMH_CUDA(cudaGetLastError());
}

// Rows per warp: as many as keep at least two blocks per SM of a 170-SM device.
template <WFormat F, Out O, int T, int CPL>
void launch_cpl(const Gemv& g, int rows, cudaStream_t stream) {
    if(rows >= kWarps * 4 * 340) launch_rows<F, O, T, CPL, 4>(g, rows, stream);
    else if(rows >= kWarps * 2 * 340) launch_rows<F, O, T, CPL, 2>(g, rows, stream);
    else launch_rows<F, O, T, CPL, 1>(g, rows, stream);
}

template <WFormat F, Out O, int T>
void launch_t(const Gemv& g, int rows, cudaStream_t stream) {
    switch(g.cols) {
        case 1024: return launch_cpl<F, O, T, 2>(g, rows, stream);
        case 4096: return launch_cpl<F, O, T, 8>(g, rows, stream);
        case 5376: return launch_cpl<F, O, T, 11>(g, rows, stream);
        case 8192: return launch_cpl<F, O, T, 16>(g, rows, stream);
        default: throw std::invalid_argument("gemv: unsupported width " + std::to_string(g.cols));
    }
}

template <WFormat F, Out O>
void launch_o(const Gemv& g, int rows, cudaStream_t stream) {
    switch(g.tokens) {
        case 1: return launch_t<F, O, 1>(g, rows, stream);
        case 2: return launch_t<F, O, 2>(g, rows, stream);
        case 3: return launch_t<F, O, 3>(g, rows, stream);
        case 4: return launch_t<F, O, 4>(g, rows, stream);
        default: throw std::invalid_argument("gemv: 1..4 tokens");
    }
}

template <WFormat F>
void launch_f(const Gemv& g, int rows, cudaStream_t stream) {
    switch(g.out) {
        case Out::store: return launch_o<F, Out::store>(g, rows, stream);
        case Out::add: return launch_o<F, Out::add>(g, rows, stream);
        case Out::relu2: return launch_o<F, Out::relu2>(g, rows, stream);
    }
}

}  // namespace

void gemv(const Gemv& in, cudaStream_t stream) {
    Gemv g = in;
    if(g.ldx == 0) g.ldx = g.cols;
    int rows = 0;
    for(int i = 0; i < g.nseg; ++i) {
        if(g.seg[i].ldy == 0) g.seg[i].ldy = g.seg[i].rows;
        rows += g.seg[i].rows;
    }
    switch(g.fmt) {
        case WFormat::fp8: return launch_f<WFormat::fp8>(g, rows, stream);
        case WFormat::bf16: return launch_f<WFormat::bf16>(g, rows, stream);
        case WFormat::fp4: return launch_f<WFormat::fp4>(g, rows, stream);
    }
}

}  // namespace nmh
