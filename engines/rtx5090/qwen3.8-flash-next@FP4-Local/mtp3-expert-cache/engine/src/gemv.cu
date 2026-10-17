#include "gemv.cuh"

namespace fnl {

namespace kern {
constexpr int kWarps = 8;
constexpr std::size_t kStageLimit = 48 * 1024;   // x staged in shared memory up to this size

// Sum of 16 E4M3 bytes times 16 FP32 activations.
__device__ __forceinline__ float dot16_e4m3(uint4 w, const float* x) {
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
    float acc = 0.f;
#pragma unroll
    for(int q = 0; q < 4; ++q) {
        float2 a = e4m3x2f(words[q] & 0xffff);
        float2 b = e4m3x2f(words[q] >> 16);
        const float4 xv = *reinterpret_cast<const float4*>(x + q * 4);
        acc = fmaf(a.x, xv.x, acc);
        acc = fmaf(a.y, xv.y, acc);
        acc = fmaf(b.x, xv.z, acc);
        acc = fmaf(b.y, xv.w, acc);
    }
    return acc;
}

// T tokens: x [T][ldx]; segment s writes y_s[t * ldy_s + row]. One warp per row; each weight chunk is read once.
template <int T>
__global__ void __launch_bounds__(kWarps * 32) mx8_gemv_kernel(Mx8Batch b, bool staged) {
    extern __shared__ float xs[];
    const float* xv = b.x;
    int ldx = b.ldx;
    if(staged) {
        for(int t = 0; t < T; ++t)
            for(int i = threadIdx.x; i < b.cols; i += blockDim.x) xs[t * b.cols + i] = b.x[std::size_t(t) * b.ldx + i];
        __syncthreads();
        xv = xs;
        ldx = b.cols;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * kWarps + warp;
    int s = 0;
    while(s < b.nseg && row >= b.seg[s].rows) row -= b.seg[s++].rows;
    if(s >= b.nseg) return;
    const Mx8Seg seg = b.seg[s];
    const int chunks = b.cols / 16;
    const uint4* w = reinterpret_cast<const uint4*>(seg.w + std::size_t(row) * b.cols);
    const uint8_t* sc = seg.s + std::size_t(row) * (b.cols / 32);
    float acc[T];
#pragma unroll
    for(int t = 0; t < T; ++t) acc[t] = 0.f;
    for(int c = lane; c < chunks; c += 32) {
        const uint4 wv = __ldg(w + c);
        const float scale = ue8m0f(__ldg(sc + (c >> 1)));
        if constexpr(T == 1) {
            acc[0] = fmaf(dot16_e4m3(wv, xv + c * 16), scale, acc[0]);
        } else {
            // Decode the 16 weights once for every token (same summation order as dot16_e4m3).
            float f[16];
            const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for(int q = 0; q < 4; ++q) {
                const float2 a = e4m3x2f(words[q] & 0xffff), b = e4m3x2f(words[q] >> 16);
                f[4 * q] = a.x;
                f[4 * q + 1] = a.y;
                f[4 * q + 2] = b.x;
                f[4 * q + 3] = b.y;
            }
#pragma unroll
            for(int t = 0; t < T; ++t) {
                const float* xt = xv + std::size_t(t) * ldx + c * 16;
                float d = 0.f;
#pragma unroll
                for(int q = 0; q < 4; ++q) {
                    const float4 v = *reinterpret_cast<const float4*>(xt + 4 * q);
                    d = fmaf(f[4 * q], v.x, d);
                    d = fmaf(f[4 * q + 1], v.y, d);
                    d = fmaf(f[4 * q + 2], v.z, d);
                    d = fmaf(f[4 * q + 3], v.w, d);
                }
                acc[t] = fmaf(d, scale, acc[t]);
            }
        }
    }
#pragma unroll
    for(int t = 0; t < T; ++t) {
        const float v = warp_sum(acc[t]);
        if(lane == 0) seg.y[std::size_t(t) * seg.ldy + row] = v;
    }
}

// Quarter q of the 16-float chunk c in staged activations. Quarters rotate with (c >> 1) & 3, so the 8 lanes of
// a quarter-warp (8 consecutive chunks) read distinct banks.
__device__ __forceinline__ int sw16(int c, int q) { return c * 16 + 4 * (q ^ ((c >> 1) & 3)); }

// MXFP8 rows against T staged tokens. A warp takes R consecutive rows and issues the loads of the next row (its
// CPL chunks per lane) before the arithmetic of the current one. Chunk order per lane, products and the warp
// reduction are those of mx8_gemv_kernel, so the results are bitwise the same.
template <int T, int CPL, int R>
__global__ void __launch_bounds__(kWarps * 32) mx8_rows_kernel(Mx8Batch b) {
    extern __shared__ __align__(16) float xs[];
    const int chunks = b.cols / 16;
    const int stage = chunks * 16;
    for(int i = threadIdx.x; i < T * chunks * 4; i += blockDim.x) {
        const int t = i / (chunks * 4);
        const int rem = i - t * chunks * 4;
        const int c = rem >> 2, q = rem & 3;
        *reinterpret_cast<float4*>(xs + t * stage + sw16(c, q)) =
            *reinterpret_cast<const float4*>(b.x + std::size_t(t) * b.ldx + c * 16 + q * 4);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * kWarps + warp) * R;
    uint4 w[2][CPL];
    uint32_t sc[2][CPL];
    float* yr[2];
    int ldy[2];
    bool live[2] = {false, false};
#pragma unroll
    for(int r = 0; r <= R; ++r) {
        if(r < R) {
            const int k = r & 1;
            int row = row0 + r, s = 0;
            while(s < b.nseg && row >= b.seg[s].rows) row -= b.seg[s++].rows;
            live[k] = s < b.nseg;
            if(live[k]) {
                const Mx8Seg seg = b.seg[s];
                const uint4* wr = reinterpret_cast<const uint4*>(seg.w + std::size_t(row) * b.cols);
                const uint8_t* sr = seg.s + std::size_t(row) * (b.cols / 32);
#pragma unroll
                for(int i = 0; i < CPL; ++i) {
                    const int c = lane + 32 * i;
                    if(c < chunks) {
                        w[k][i] = __ldg(wr + c);
                        sc[k][i] = __ldg(sr + (c >> 1));
                    }
                }
                yr[k] = seg.y + row;
                ldy[k] = seg.ldy;
            }
        }
        if(r > 0) {
            const int k = (r - 1) & 1;
            if(live[k]) {
                float acc[T];
#pragma unroll
                for(int t = 0; t < T; ++t) acc[t] = 0.f;
#pragma unroll
                for(int i = 0; i < CPL; ++i) {
                    const int c = lane + 32 * i;
                    if(c < chunks) {
                        const float scale = ue8m0f(sc[k][i]);
                        float f[16];
                        const uint32_t words[4] = {w[k][i].x, w[k][i].y, w[k][i].z, w[k][i].w};
#pragma unroll
                        for(int q = 0; q < 4; ++q) {
                            const float2 a = e4m3x2f(words[q] & 0xffff), bb = e4m3x2f(words[q] >> 16);
                            f[4 * q] = a.x;
                            f[4 * q + 1] = a.y;
                            f[4 * q + 2] = bb.x;
                            f[4 * q + 3] = bb.y;
                        }
#pragma unroll
                        for(int t = 0; t < T; ++t) {
                            const float* xt = xs + t * stage;
                            float d = 0.f;
#pragma unroll
                            for(int q = 0; q < 4; ++q) {
                                const float4 v = *reinterpret_cast<const float4*>(xt + sw16(c, q));
                                d = fmaf(f[4 * q], v.x, d);
                                d = fmaf(f[4 * q + 1], v.y, d);
                                d = fmaf(f[4 * q + 2], v.z, d);
                                d = fmaf(f[4 * q + 3], v.w, d);
                            }
                            acc[t] = fmaf(d, scale, acc[t]);
                        }
                    }
                }
#pragma unroll
                for(int t = 0; t < T; ++t) {
                    const float v = warp_sum(acc[t]);
                    if(lane == 0) yr[k][std::size_t(t) * ldy[k]] = v;
                }
            }
        }
    }
}

template <int T, int CPL, int R>
void mx8_rows_launch(const Mx8Batch& c, int rows, cudaStream_t stream) {
    const std::size_t smem = std::size_t(T) * c.cols * sizeof(float);
    static bool attr = false;
    if(!attr) {
        FNL_CUDA(cudaFuncSetAttribute(mx8_rows_kernel<T, CPL, R>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      int(std::size_t(T) * 6144 * sizeof(float))));
        attr = true;
    }
    const int blocks = (rows + kWarps * R - 1) / (kWarps * R);
    mx8_rows_kernel<T, CPL, R><<<blocks, kWarps * 32, smem, stream>>>(c);
}

// Rows per warp: as many as keep at least two blocks per SM of a 170-SM device.
inline int rows_per_warp(int rows) { return rows >= kWarps * 4 * 340 ? 4 : rows >= kWarps * 2 * 340 ? 2 : 1; }

template <int T, int CPL>
void mx8_rows(const Mx8Batch& c, int rows, cudaStream_t stream) {
    switch(rows_per_warp(rows)) {
        case 4: mx8_rows_launch<T, CPL, 4>(c, rows, stream); break;
        case 2: mx8_rows_launch<T, CPL, 2>(c, rows, stream); break;
        default: mx8_rows_launch<T, CPL, 1>(c, rows, stream); break;
    }
}

template <int T>
bool mx8_rows_t(const Mx8Batch& c, int rows, cudaStream_t stream) {
    switch(c.cols) {
        case 640: mx8_rows<T, 2>(c, rows, stream); return true;
        case 2560: mx8_rows<T, 5>(c, rows, stream); return true;
        case 6144:
            // Wide rows: more than two staged tokens cost more shared-memory traffic than v1 reading them through
            // L1, which blocks on one SM share.
            if(T > 2) return false;
            mx8_rows<T, 12>(c, rows, stream);
            return true;
        default: return false;
    }
}
}  // namespace kern
using namespace kern;

void mx8_gemv(const Mx8Batch& b, cudaStream_t stream) {
    int rows = 0;
    for(int i = 0; i < b.nseg; ++i) rows += b.seg[i].rows;
    const int blocks = (rows + kWarps - 1) / kWarps;
    const int T = b.tokens < 1 ? 1 : b.tokens;
    Mx8Batch c = b;
    if(c.ldx == 0) c.ldx = c.cols;
    for(int i = 0; i < c.nseg; ++i)
        if(c.seg[i].ldy == 0) c.seg[i].ldy = c.seg[i].rows;
    static const bool v1 = std::getenv("FNL_GEMV_V1") != nullptr;
    bool done = false;
    if(!v1) {
        switch(T) {
            case 1: done = mx8_rows_t<1>(c, rows, stream); break;
            case 2: done = mx8_rows_t<2>(c, rows, stream); break;
            case 3: done = mx8_rows_t<3>(c, rows, stream); break;
            case 4: done = mx8_rows_t<4>(c, rows, stream); break;
        }
    }
    if(done) {
        FNL_CUDA(cudaGetLastError());
        return;
    }
    const std::size_t stage = std::size_t(T) * c.cols * sizeof(float);
    const bool staged = stage <= kStageLimit;
    const std::size_t smem = staged ? stage : 0;
    switch(T) {
        case 1: mx8_gemv_kernel<1><<<blocks, kWarps * 32, smem, stream>>>(c, staged); break;
        case 2: mx8_gemv_kernel<2><<<blocks, kWarps * 32, smem, stream>>>(c, staged); break;
        case 3: mx8_gemv_kernel<3><<<blocks, kWarps * 32, smem, stream>>>(c, staged); break;
        case 4: mx8_gemv_kernel<4><<<blocks, kWarps * 32, smem, stream>>>(c, staged); break;
        default: throw std::invalid_argument("mx8_gemv supports 1..4 tokens");
    }
    FNL_CUDA(cudaGetLastError());
}

namespace kern {
__device__ __forceinline__ float post_op(float acc, Post post) {
    switch(post) {
        case Post::silu_quarter: return siluf(acc / float(kStreams));
        case Post::sigmoid: return sigmoidf(acc);
        case Post::inject: return 2.f * sigmoidf(acc / float(kStreams));
        default: return acc;
    }
}
__device__ __forceinline__ float dot8_bf16(uint4 v, const float* x) {
    const float4 a = *reinterpret_cast<const float4*>(x);
    const float4 b = *reinterpret_cast<const float4*>(x + 4);
    float acc = bf16_lo(v.x) * a.x;
    acc = fmaf(bf16_hi(v.x), a.y, acc);
    acc = fmaf(bf16_lo(v.y), a.z, acc);
    acc = fmaf(bf16_hi(v.y), a.w, acc);
    acc = fmaf(bf16_lo(v.z), b.x, acc);
    acc = fmaf(bf16_hi(v.z), b.y, acc);
    acc = fmaf(bf16_lo(v.w), b.z, acc);
    acc = fmaf(bf16_hi(v.w), b.w, acc);
    return acc;
}

// TPR threads per row (8, 32 or 256), 256 threads per block, T tokens (x [T][ldx], y [T][ldy]).
template <int TPR, int T>
__global__ void __launch_bounds__(256) bf16_gemv_kernel(Bf16Gemv g, bool staged) {
    extern __shared__ __align__(16) float xs[];
    const float* xv = g.x;
    int ldx = g.ldx;
    if(staged) {
        for(int t = 0; t < T; ++t)
            for(int i = threadIdx.x; i < g.cols; i += blockDim.x) xs[t * g.cols + i] = g.x[std::size_t(t) * g.ldx + i];
        __syncthreads();
        xv = xs;
        ldx = g.cols;
    }
    constexpr int kRows = 256 / TPR;
    const int sub = threadIdx.x % TPR;
    const int row = blockIdx.x * kRows + threadIdx.x / TPR;
    const int chunks = g.cols / 8;
    float acc[T];
#pragma unroll
    for(int t = 0; t < T; ++t) acc[t] = 0.f;
    if(row < g.rows) {
        const uint4* w = reinterpret_cast<const uint4*>(g.w + std::size_t(row) * g.cols);
        for(int c = sub; c < chunks; c += TPR) {
            const uint4 wv = __ldg(w + c);
#pragma unroll
            for(int t = 0; t < T; ++t) acc[t] += dot8_bf16(wv, xv + std::size_t(t) * ldx + c * 8);
        }
    }
    if constexpr(TPR <= 32) {
#pragma unroll
        for(int t = 0; t < T; ++t) {
            float a = acc[t];
#pragma unroll
            for(int o = TPR / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if(sub == 0 && row < g.rows) g.y[std::size_t(t) * g.ldy + row] = post_op(a, g.post);
        }
    } else {
        __shared__ float red[T][8];
#pragma unroll
        for(int t = 0; t < T; ++t) {
            const float a = warp_sum(acc[t]);
            if((threadIdx.x & 31) == 0) red[t][threadIdx.x >> 5] = a;
        }
        __syncthreads();
        if(threadIdx.x < T && row < g.rows) {
            float s = red[threadIdx.x][0];
#pragma unroll
            for(int w = 1; w < 8; ++w) s += red[threadIdx.x][w];
            g.y[std::size_t(threadIdx.x) * g.ldy + row] = post_op(s, g.post);
        }
    }
}


// Half h of the 8-float chunk c in staged activations; halves swap with (c >> 2) & 1 so that a quarter-warp's 8
// consecutive chunks read distinct banks.
__device__ __forceinline__ int sw8(int c, int h) { return c * 8 + 4 * (h ^ ((c >> 2) & 1)); }

// BF16 rows (one warp per row, as bf16_gemv_kernel<32, T>) with R rows per warp and the next row's loads issued
// before the current row's arithmetic; same per-lane order and reduction, so bitwise the same results.
template <int T, int CPL, int R>
__global__ void __launch_bounds__(256) bf16_rows_kernel(Bf16Gemv g) {
    extern __shared__ __align__(16) float xs[];
    const int chunks = g.cols / 8;
    for(int i = threadIdx.x; i < T * chunks * 2; i += blockDim.x) {
        const int t = i / (chunks * 2);
        const int rem = i - t * chunks * 2;
        const int c = rem >> 1, h = rem & 1;
        *reinterpret_cast<float4*>(xs + t * g.cols + sw8(c, h)) =
            *reinterpret_cast<const float4*>(g.x + std::size_t(t) * g.ldx + c * 8 + h * 4);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    uint4 w[2][CPL];
#pragma unroll
    for(int r = 0; r <= R; ++r) {
        if(r < R && row0 + r < g.rows) {
            const int k = r & 1;
            const uint4* wr = reinterpret_cast<const uint4*>(g.w + std::size_t(row0 + r) * g.cols);
#pragma unroll
            for(int i = 0; i < CPL; ++i) {
                const int c = lane + 32 * i;
                if(c < chunks) w[k][i] = __ldg(wr + c);
            }
        }
        if(r > 0 && row0 + r - 1 < g.rows) {
            const int k = (r - 1) & 1;
            float acc[T];
#pragma unroll
            for(int t = 0; t < T; ++t) acc[t] = 0.f;
#pragma unroll
            for(int i = 0; i < CPL; ++i) {
                const int c = lane + 32 * i;
                if(c < chunks) {
                    const uint4 v = w[k][i];
#pragma unroll
                    for(int t = 0; t < T; ++t) {
                        const float4 a = *reinterpret_cast<const float4*>(xs + t * g.cols + sw8(c, 0));
                        const float4 b = *reinterpret_cast<const float4*>(xs + t * g.cols + sw8(c, 1));
                        float d = bf16_lo(v.x) * a.x;
                        d = fmaf(bf16_hi(v.x), a.y, d);
                        d = fmaf(bf16_lo(v.y), a.z, d);
                        d = fmaf(bf16_hi(v.y), a.w, d);
                        d = fmaf(bf16_lo(v.z), b.x, d);
                        d = fmaf(bf16_hi(v.z), b.y, d);
                        d = fmaf(bf16_lo(v.w), b.z, d);
                        d = fmaf(bf16_hi(v.w), b.w, d);
                        acc[t] += d;
                    }
                }
            }
#pragma unroll
            for(int t = 0; t < T; ++t) {
                float a = acc[t];
#pragma unroll
                for(int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if(lane == 0) g.y[std::size_t(t) * g.ldy + row0 + r - 1] = post_op(a, g.post);
            }
        }
    }
}

template <int T, int CPL, int R>
void bf16_rows_launch(const Bf16Gemv& g, cudaStream_t stream) {
    const std::size_t smem = std::size_t(T) * g.cols * sizeof(float);
    static bool attr = false;
    if(!attr) {
        FNL_CUDA(cudaFuncSetAttribute(bf16_rows_kernel<T, CPL, R>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      int(std::size_t(T) * 6144 * sizeof(float))));
        attr = true;
    }
    bf16_rows_kernel<T, CPL, R><<<(g.rows + 8 * R - 1) / (8 * R), 256, smem, stream>>>(g);
}

template <int T, int CPL>
void bf16_rows(const Bf16Gemv& g, cudaStream_t stream) {
    // Wide rows (24 chunks per lane) keep one row per warp: two rows of loads would not fit in registers.
    switch(CPL > 12 ? 1 : rows_per_warp(g.rows)) {
        case 4: bf16_rows_launch<T, CPL, 4>(g, stream); break;
        case 2: bf16_rows_launch<T, CPL, 2>(g, stream); break;
        default: bf16_rows_launch<T, CPL, 1>(g, stream); break;
    }
}

template <int T>
bool bf16_rows_t(const Bf16Gemv& g, cudaStream_t stream) {
    switch(g.cols) {
        case 640: bf16_rows<T, 3>(g, stream); return true;
        case 2560: bf16_rows<T, 10>(g, stream); return true;
        case 6144:
            if(T > 2) return false;
            bf16_rows<T, 24>(g, stream);
            return true;
        default: return false;
    }
}

template <int T>
void bf16_gemv_t(const Bf16Gemv& g, cudaStream_t stream) {
    const std::size_t stage = std::size_t(T) * g.cols * sizeof(float);
    const bool staged = stage <= kStageLimit;
    const std::size_t smem = staged ? stage : 0;
    if(g.cols <= 512) {
        bf16_gemv_kernel<8, T><<<(g.rows + 31) / 32, 256, smem, stream>>>(g, staged);
    } else if(g.rows <= 1024) {
        bf16_gemv_kernel<256, T><<<g.rows, 256, 0, stream>>>(g, false);
    } else {
        static const bool v1 = std::getenv("FNL_GEMV_V1") != nullptr;
        if(v1 || !bf16_rows_t<T>(g, stream))
            bf16_gemv_kernel<32, T><<<(g.rows + 7) / 8, 256, smem, stream>>>(g, staged);
    }
}
}  // namespace kern
using namespace kern;

void bf16_gemv(const Bf16Gemv& g0, cudaStream_t stream) {
    Bf16Gemv g = g0;
    if(g.ldx == 0) g.ldx = g.cols;
    if(g.ldy == 0) g.ldy = g.rows;
    switch(g.tokens < 1 ? 1 : g.tokens) {
        case 1: bf16_gemv_t<1>(g, stream); break;
        case 2: bf16_gemv_t<2>(g, stream); break;
        case 3: bf16_gemv_t<3>(g, stream); break;
        case 4: bf16_gemv_t<4>(g, stream); break;
        default: throw std::invalid_argument("bf16_gemv supports 1..4 tokens");
    }
    FNL_CUDA(cudaGetLastError());
}

}  // namespace fnl
