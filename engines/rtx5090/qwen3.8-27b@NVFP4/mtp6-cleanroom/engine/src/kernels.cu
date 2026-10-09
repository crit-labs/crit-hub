// Clean-room kernels for the Qwen3.5 hybrid decoder (Gated DeltaNet + gated
// full attention) over the checkpoint's stored NVFP4, FP8 and BF16 words.
#include "kernels.cuh"
#include <cuda_fp8.h>
#include <cfloat>
#include <stdexcept>
#include <string>

namespace q38 {
namespace {
constexpr int kHidden = 5120;
constexpr int kConv = 10240;      // q(2048) + k(2048) + v(6144)
constexpr int kVHeads = 48, kKHeads = 16, kHeadK = 128, kHeadV = 128;
constexpr int kQHeads = 24, kKvHeads = 4, kHeadDim = 256, kRot = 64;

void check(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }

// Programmatic dependent launch: every kernel triggers its dependents at entry
// and waits for its predecessor's completion before touching activations or
// state, so the next kernel's blocks become resident during this kernel's tail.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_begin() { pdl_trigger(); pdl_wait(); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint2 ldg_stream2(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t mul_bf2(uint32_t a, uint32_t b) {
    __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
    __nv_bfloat162 r = __hmul2(x, y);
    return *reinterpret_cast<uint32_t*>(&r);
}
// E4M3 bytes 0,1 (lo) or 2,3 (hi) of w as an exact BF16 pair. The code bits are
// placed in the BF16 exponent/mantissa fields and rescaled by 2^120, which is
// exact for normal and subnormal E4M3 values.
constexpr uint32_t kTwo120 = 0x7B807B80u;
__device__ __forceinline__ uint32_t e4m3x2_lo(uint32_t w) {
    const uint32_t x = __byte_perm(w, 0, 0x1404);
    return mul_bf2((x & 0x80008000u) | ((x & 0x7F007F00u) >> 4), kTwo120);
}
__device__ __forceinline__ uint32_t e4m3x2_hi(uint32_t w) {
    const uint32_t x = __byte_perm(w, 0, 0x3424);
    return mul_bf2((x & 0x80008000u) | ((x & 0x7F007F00u) >> 4), kTwo120);
}
__device__ __forceinline__ float e4m3_float(uint8_t b) {
    const uint32_t r = e4m3x2_lo(uint32_t(b));
    return __bfloat162float(*reinterpret_cast<const bf16*>(&r));
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float e2m1_value(uint32_t n) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = mag[n & 7];
    return (n & 8) ? -v : v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for(int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for(int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---------------------------------------------------------------- GEMV (mma)
// Warp tile: 16 weight rows x 8 activation columns, k split across the block's
// warps. Inside a chunk the k order is permuted identically for A and B so each
// thread loads contiguous bytes: for mma step j, thread (g, t) holds
// k = 16t + 4j + {0..3} (64-wide FP8/BF16 chunks) or 32t + 4j + {0..3}
// (128-wide NVFP4 chunks, whose 32 values span exactly two scale blocks).
// Rows g come from `top`, rows g + 8 from `bot` (the next 8 rows of the same
// matrix, or the matching rows of a second matrix for fused SwiGLU).

// E2M1 byte (two codes) -> BF16 pair scaled by 2^-126: the 3 magnitude bits are
// placed in the low exponent/high mantissa field, which encodes {0, .5, 1, 1.5,
// 2, 3, 4, 6} * 2^-126 exactly (0.5 lands on the BF16 subnormal 2^-127).
__device__ __forceinline__ uint32_t e2m1x2_raw(uint32_t b) {
    return ((b & 0x7u) << 6) | ((b & 0x8u) << 12) | ((b & 0x70u) << 18) | ((b & 0x80u) << 24);
}
// E4M3 block scale times 2^119 as a broadcast BF16 pair (exact: <= 4 significant
// bits and below the BF16 maximum). The product with e2m1x2_raw is v * s * 2^-7.
__device__ __forceinline__ uint32_t nvfp4_scale_pair(uint32_t byte) {
    const bf16 s = __float2bfloat16(e4m3_float(uint8_t(byte)) * 0x1p119f);
    const uint32_t u = *reinterpret_cast<const uint16_t*>(&s);
    return u | (u << 16);
}
constexpr float kNvfp4Unscale = 128.f; // 2^7

// Two 16-row mma tiles per warp share each activation fragment.
constexpr int kRT = 2;
struct TileRows {
    const uint8_t* top[kRT];  // rows (tile r, g)
    const uint8_t* bot[kRT];  // rows (tile r, g + 8)
    const uint8_t* stop[kRT]; // NVFP4 scales of top/bottom rows
    const uint8_t* sbot[kRT];
};

template<int FMT>
__device__ __forceinline__ void gemv_tile(const TileRows& R, int K, const bf16* xr, bool xv, float (&acc)[kRT][4], int warp, int nwarps) {
    if constexpr(FMT == W_NVFP4) {
        const int chunks = K >> 7;
        for(int c = 2 * warp; c < chunks; c += 2 * nwarps) {
            const bool two = c + 1 < chunks;
            uint4 wa[2][kRT], wb[2][kRT], x[2][4];
            uint32_t sa[2][kRT], sb[2][kRT];
#pragma unroll
            for(int u = 0; u < 2; ++u) {
                if(u && !two) break;
                const int cc = c + u;
#pragma unroll
                for(int r = 0; r < kRT; ++r) {
                    wa[u][r] = ldg_stream(R.top[r] + cc * 64);
                    wb[u][r] = ldg_stream(R.bot[r] + cc * 64);
                    sa[u][r] = __ldg(reinterpret_cast<const uint16_t*>(R.stop[r] + cc * 8));
                    sb[u][r] = __ldg(reinterpret_cast<const uint16_t*>(R.sbot[r] + cc * 8));
                }
#pragma unroll
                for(int i = 0; i < 4; ++i)
                    x[u][i] = xv ? __ldg(reinterpret_cast<const uint4*>(xr + cc * 128) + i) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for(int u = 0; u < 2; ++u) {
                if(u && !two) break;
                const uint32_t X[16] = {x[u][0].x, x[u][0].y, x[u][0].z, x[u][0].w, x[u][1].x, x[u][1].y, x[u][1].z, x[u][1].w,
                                        x[u][2].x, x[u][2].y, x[u][2].z, x[u][2].w, x[u][3].x, x[u][3].y, x[u][3].z, x[u][3].w};
#pragma unroll
                for(int r = 0; r < kRT; ++r) {
                    const uint32_t A[4] = {wa[u][r].x, wa[u][r].y, wa[u][r].z, wa[u][r].w};
                    const uint32_t B[4] = {wb[u][r].x, wb[u][r].y, wb[u][r].z, wb[u][r].w};
                    const uint32_t scA[2] = {nvfp4_scale_pair(sa[u][r] & 0xFF), nvfp4_scale_pair(sa[u][r] >> 8)};
                    const uint32_t scB[2] = {nvfp4_scale_pair(sb[u][r] & 0xFF), nvfp4_scale_pair(sb[u][r] >> 8)};
#pragma unroll
                    for(int j = 0; j < 8; ++j) {
                        const uint32_t wA = A[j >> 1] >> (16 * (j & 1)), wB = B[j >> 1] >> (16 * (j & 1));
                        const uint32_t a0 = mul_bf2(e2m1x2_raw(wA), scA[j >> 2]), a2 = mul_bf2(e2m1x2_raw(wA >> 8), scA[j >> 2]);
                        const uint32_t a1 = mul_bf2(e2m1x2_raw(wB), scB[j >> 2]), a3 = mul_bf2(e2m1x2_raw(wB >> 8), scB[j >> 2]);
                        mma16816(acc[r], a0, a1, a2, a3, X[2 * j], X[2 * j + 1]);
                    }
                }
            }
        }
    } else if constexpr(FMT == W_FP8) {
        const int chunks = K >> 6;
        for(int c = 2 * warp; c < chunks; c += 2 * nwarps) {
            const bool two = c + 1 < chunks;
            uint4 wa[2][kRT], wb[2][kRT], x0[2], x1[2];
#pragma unroll
            for(int u = 0; u < 2; ++u) {
                if(u && !two) break;
                const int cc = c + u;
#pragma unroll
                for(int r = 0; r < kRT; ++r) {
                    wa[u][r] = ldg_stream(R.top[r] + cc * 64);
                    wb[u][r] = ldg_stream(R.bot[r] + cc * 64);
                }
                x0[u] = xv ? __ldg(reinterpret_cast<const uint4*>(xr + cc * 64)) : make_uint4(0, 0, 0, 0);
                x1[u] = xv ? __ldg(reinterpret_cast<const uint4*>(xr + cc * 64 + 8)) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for(int u = 0; u < 2; ++u) {
                if(u && !two) break;
                const uint32_t X[8] = {x0[u].x, x0[u].y, x0[u].z, x0[u].w, x1[u].x, x1[u].y, x1[u].z, x1[u].w};
#pragma unroll
                for(int r = 0; r < kRT; ++r) {
                    const uint32_t A[4] = {wa[u][r].x, wa[u][r].y, wa[u][r].z, wa[u][r].w};
                    const uint32_t B[4] = {wb[u][r].x, wb[u][r].y, wb[u][r].z, wb[u][r].w};
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        mma16816(acc[r], e4m3x2_lo(A[j]), e4m3x2_lo(B[j]), e4m3x2_hi(A[j]), e4m3x2_hi(B[j]), X[2 * j], X[2 * j + 1]);
                }
            }
        }
    } else {
        const int chunks = K >> 6;
        for(int c = warp; c < chunks; c += nwarps) {
            const uint4 x0 = xv ? __ldg(reinterpret_cast<const uint4*>(xr + c * 64)) : make_uint4(0, 0, 0, 0);
            const uint4 x1 = xv ? __ldg(reinterpret_cast<const uint4*>(xr + c * 64 + 8)) : make_uint4(0, 0, 0, 0);
            uint4 al[kRT], ah[kRT], bl[kRT], bh[kRT];
#pragma unroll
            for(int r = 0; r < kRT; ++r) {
                al[r] = ldg_stream(R.top[r] + c * 128); ah[r] = ldg_stream(R.top[r] + c * 128 + 16);
                bl[r] = ldg_stream(R.bot[r] + c * 128); bh[r] = ldg_stream(R.bot[r] + c * 128 + 16);
            }
            const uint32_t X[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for(int r = 0; r < kRT; ++r) {
                const uint32_t A[8] = {al[r].x, al[r].y, al[r].z, al[r].w, ah[r].x, ah[r].y, ah[r].z, ah[r].w};
                const uint32_t B[8] = {bl[r].x, bl[r].y, bl[r].z, bl[r].w, bh[r].x, bh[r].y, bh[r].z, bh[r].w};
#pragma unroll
                for(int j = 0; j < 4; ++j) mma16816(acc[r], A[2 * j], B[2 * j], A[2 * j + 1], B[2 * j + 1], X[2 * j], X[2 * j + 1]);
            }
        }
    }
}

__device__ __forceinline__ void row_ptrs(const QW& W, int row, int t, const uint8_t*& data, const uint8_t*& scale) {
    if(W.fmt == W_NVFP4) {
        data = W.w + std::size_t(row) * (W.K >> 1) + 16 * t;
        scale = W.bs + std::size_t(row) * (W.K >> 4) + 2 * t;
    } else if(W.fmt == W_FP8) {
        data = W.w + std::size_t(row) * W.K + 16 * t;
        scale = nullptr;
    } else {
        data = W.w + (std::size_t(row) * W.K + 16 * t) * 2;
        scale = nullptr;
    }
}
__device__ __forceinline__ float out_scale(const QW& W, int row) {
    if(W.fmt == W_FP8) return bf(W.rs[row]);
    if(W.fmt == W_NVFP4) return kNvfp4Unscale / W.gs; // FP32 global divisor
    return W.rs ? bf(W.rs[row]) : 1.f;
}
__device__ __forceinline__ void run_tile(const QW& W, const TileRows& R, const bf16* xg, int ldx, int mcols, float (&acc)[kRT][4],
                                         int warp, int nwarps) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bool xv = g < mcols;
    const bf16* xr = xg + std::size_t(xv ? g : 0) * ldx + (W.fmt == W_NVFP4 ? 32 : 16) * t;
    if(W.fmt == W_NVFP4) gemv_tile<W_NVFP4>(R, W.K, xr, xv, acc, warp, nwarps);
    else if(W.fmt == W_FP8) gemv_tile<W_FP8>(R, W.K, xr, xv, acc, warp, nwarps);
    else gemv_tile<W_BF16>(R, W.K, xr, xv, acc, warp, nwarps);
}

#ifndef Q38_GEMV_WARPS
#define Q38_GEMV_WARPS 8
#endif
constexpr int kGemvWarps = Q38_GEMV_WARPS;
constexpr int kMaxSegs = 3;
struct GemvLaunch {
    GemvSeg seg[kMaxSegs];
    int start[kMaxSegs + 1];   // cumulative block offsets
    const bf16* x;
    int ldx, M;
};
__global__ void __launch_bounds__(kGemvWarps * 32, 16 / kGemvWarps) gemv_kernel(GemvLaunch L) {
    pdl_begin();
    __shared__ float red[kGemvWarps][32][kRT * 4];
    int si = 0;
    while(si + 1 < kMaxSegs && int(blockIdx.x) >= L.start[si + 1]) ++si;
    const GemvSeg& S = L.seg[si];
    const int tile = int(blockIdx.x) - L.start[si];
    const int cg = blockIdx.y;
    const int mcols = min(8, L.M - cg * 8);
    const bf16* xg = L.x + std::size_t(cg) * 8 * L.ldx;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    // Split-K across gridDim.z blocks: global warp index over all splits.
    const int gwarp = int(blockIdx.z) * kGemvWarps + warp, nwarps = kGemvWarps * int(gridDim.z);
    const bool fused = S.act != nullptr;
    // Plain: 32 rows per block (tile r covers rows n0 + 16r + {g, g + 8}).
    // Fused SwiGLU: 16 gate rows and the same 16 up rows (tile r: gate/up rows n0 + 8r + g).
    const int n0 = tile * (fused ? 8 * kRT : 16 * kRT);
    TileRows R;
#pragma unroll
    for(int r = 0; r < kRT; ++r) {
        if(fused) {
            row_ptrs(S.w, n0 + 8 * r + g, t, R.top[r], R.stop[r]);
            row_ptrs(S.w2, n0 + 8 * r + g, t, R.bot[r], R.sbot[r]);
        } else {
            row_ptrs(S.w, n0 + 16 * r + g, t, R.top[r], R.stop[r]);
            row_ptrs(S.w, n0 + 16 * r + g + 8, t, R.bot[r], R.sbot[r]);
        }
    }
    {
        const int seg_bytes = S.w.fmt == W_BF16 ? 128 : 64; // bytes per row per chunk
        const int c0 = S.w.fmt == W_BF16 ? gwarp : 2 * gwarp;
        const int chunks = S.w.fmt == W_NVFP4 ? S.w.K >> 7 : S.w.K >> 6;
#pragma unroll
        for(int r = 0; r < kRT; ++r) {
            if(c0 >= chunks) break;
            prefetch_l2(R.top[r] + c0 * seg_bytes);
            prefetch_l2(R.bot[r] + c0 * seg_bytes);
        }
    }
    pdl_wait();
    float acc[kRT][4] = {};
    run_tile(S.w, R, xg, L.ldx, mcols, acc, gwarp, nwarps);
#pragma unroll
    for(int r = 0; r < kRT; ++r)
#pragma unroll
        for(int i = 0; i < 4; ++i) red[warp][lane][r * 4 + i] = acc[r][i];
    __syncthreads();
    if(warp) return;
#pragma unroll
    for(int r = 0; r < kRT; ++r)
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            float v = 0.f;
#pragma unroll
            for(int w = 0; w < kGemvWarps; ++w) v += red[w][lane][r * 4 + i];
            acc[r][i] = v;
        }
#pragma unroll
    for(int r = 0; r < kRT; ++r) {
        if(fused) {
            const int row = n0 + 8 * r + g;
            const float sg = out_scale(S.w, row), su = out_scale(S.w2, row);
#pragma unroll
            for(int i = 0; i < 2; ++i) {
                const int col = 2 * t + i;
                if(col >= mcols) continue;
                const float gate = acc[r][i] * sg, up = acc[r][2 + i] * su;
                S.act[std::size_t(cg * 8 + col) * S.ldy + row] = __float2bfloat16(gate / (1.f + __expf(-gate)) * up);
            }
            continue;
        }
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const int row = n0 + 16 * r + g + ((i & 2) ? 8 : 0);
            const int col = 2 * t + (i & 1);
            if(col >= mcols) continue;
            float* y = S.y + std::size_t(blockIdx.z) * S.split_stride + std::size_t(cg * 8 + col) * S.ldy + row;
            const float v = acc[r][i] * out_scale(S.w, row);
            *y = S.accumulate ? *y + v : v;
        }
    }
}

// --------------------------------------------------------------- dequant
__global__ void dequant_kernel(QW W, bf16* out) {
    pdl_begin();
    const std::size_t i8 = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    const std::size_t total = std::size_t(W.N) * W.K;
    if(i8 >= total) return;
    uint32_t r[4];
    if(W.fmt == W_FP8) {
        const uint2 w = *reinterpret_cast<const uint2*>(W.w + i8);
        r[0] = e4m3x2_lo(w.x); r[1] = e4m3x2_hi(w.x); r[2] = e4m3x2_lo(w.y); r[3] = e4m3x2_hi(w.y);
    } else {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(W.w + i8 / 2);
        const float s = e4m3_float(W.bs[i8 / 16]);
#pragma unroll
        for(int j = 0; j < 4; ++j) {
            const uint32_t byte = (w >> (8 * j)) & 0xFF;
            const bf16 lo = __float2bfloat16(e2m1_value(byte & 15) * s), hi = __float2bfloat16(e2m1_value(byte >> 4) * s);
            r[j] = uint32_t(*reinterpret_cast<const uint16_t*>(&lo)) | (uint32_t(*reinterpret_cast<const uint16_t*>(&hi)) << 16);
        }
    }
    *reinterpret_cast<uint4*>(out + i8) = make_uint4(r[0], r[1], r[2], r[3]);
}
__global__ void row_scale_kernel(QW W, const float* t, float* y, int ldy, int M, int accumulate) {
    pdl_begin();
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= std::size_t(M) * W.N) return;
    const int m = int(i / W.N), n = int(i % W.N);
    float v = t[i];
    if(W.fmt == W_FP8) v *= bf(W.rs[n]);
    else if(W.fmt == W_NVFP4) v /= W.gs;
    else if(W.rs) v *= bf(W.rs[n]);
    float* o = y + std::size_t(m) * ldy + n;
    *o = accumulate ? *o + v : v;
}

// ------------------------------------------------------------- elementwise
__global__ void embed_kernel(const int* tokens, const bf16* table, float* out, int D) {
    pdl_begin();
    const int t = blockIdx.x;
    const bf16* row = table + std::size_t(tokens[t]) * D;
    for(int d = threadIdx.x; d < D; d += blockDim.x) out[std::size_t(t) * D + d] = bf(row[d]);
}
template<typename In>
__global__ void rmsnorm_kernel(const In* x, int ldx, const bf16* w, bf16* out, int ldo, int D, float eps, int plus_one) {
    pdl_begin();
    __shared__ float part[32];
    const In* row = x + std::size_t(blockIdx.x) * ldx;
    float ss = 0.f;
    for(int d = threadIdx.x; d < D; d += blockDim.x) { const float v = float(row[d]); ss += v * v; }
    ss = warp_sum(ss);
    if((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if(threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if(threadIdx.x == 0) part[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(part[0] / D + eps);
    for(int d = threadIdx.x; d < D; d += blockDim.x) {
        const float wv = bf(w[d]);
        out[std::size_t(blockIdx.x) * ldo + d] = __float2bfloat16(float(row[d]) * r * (plus_one ? 1.f + wv : wv));
    }
}
// resid += sum of split partials (fixed order), then out = bf16(rmsnorm(resid) * (1 + w)).
__global__ void residual_norm_kernel(float* resid, const float* partial, int nsplit, std::size_t stride, const bf16* w, bf16* out,
                                     int D, float eps) {
    pdl_begin();
    __shared__ float part[32];
    float* row = resid + std::size_t(blockIdx.x) * D;
    const float* p = partial + std::size_t(blockIdx.x) * D;
    float ss = 0.f;
    for(int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = row[d];
        for(int s = 0; s < nsplit; ++s) v += p[s * stride + d];
        row[d] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    if((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for(int i = 0; i < int(blockDim.x >> 5); ++i) tot += part[i];
    const float r = rsqrtf(tot / D + eps);
    for(int d = threadIdx.x; d < D; d += blockDim.x)
        out[std::size_t(blockIdx.x) * D + d] = __float2bfloat16(row[d] * r * (1.f + bf(w[d])));
}
__global__ void swiglu_kernel(const float* gu, int ld, bf16* out, int I) {
    pdl_begin();
    const int t = blockIdx.y;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < I; i += gridDim.x * blockDim.x) {
        const float g = gu[std::size_t(t) * ld + i], u = gu[std::size_t(t) * ld + I + i];
        out[std::size_t(t) * I + i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
    }
}

// ----------------------------------------------------------- Gated DeltaNet
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__global__ void gdn_conv_kernel(GdnLayer L, const float* qkvz, int ldq, float* out, int T, int mode, const int* pending, int normalize) {
    pdl_begin();
    __shared__ float part[8];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float h0 = L.conv_hist[c], h1 = L.conv_hist[kConv + c], h2 = L.conv_hist[2 * kConv + c];
    if(mode == 1) {
        const int m = *pending;
        for(int i = 0; i < m; ++i) { h0 = h1; h1 = h2; h2 = L.conv_stash[std::size_t(i) * kConv + c]; }
    }
    const float w0 = bf(L.conv_w[c * 4]), w1 = bf(L.conv_w[c * 4 + 1]), w2 = bf(L.conv_w[c * 4 + 2]), w3 = bf(L.conv_w[c * 4 + 3]);
    // Blocks 0..15 hold q (0..2047) and k (2048..4095): two 128-channel heads per block.
    const bool qk = normalize && blockIdx.x < 4096 / 256;
    float a = h0, b = h1, d = h2;
    for(int t = 0; t < T; ++t) {
        const float x = qkvz[std::size_t(t) * ldq + c];
        float y = silu(w0 * a + w1 * b + w2 * d + w3 * x);
        if(qk) {
            const float ss = warp_sum(y * y);
            if(lane == 0) part[warp] = ss;
            __syncthreads();
            const int g4 = (warp >> 2) * 4;
            const float tot = part[g4] + part[g4 + 1] + part[g4 + 2] + part[g4 + 3];
            __syncthreads();
            y *= rsqrtf(tot + 1e-6f) * (c < 2048 ? rsqrtf(float(kHeadK)) : 1.f);
        }
        out[std::size_t(t) * kConv + c] = y;
        if(mode == 1) L.conv_stash[std::size_t(t) * kConv + c] = x;
        a = b; b = d; d = x;
    }
    if(mode == 0) { h0 = a; h1 = b; h2 = d; }
    L.conv_hist[c] = h0; L.conv_hist[kConv + c] = h1; L.conv_hist[2 * kConv + c] = h2;
}

// Decode/verify Gated DeltaNet (T <= 8 columns, verify mode). The conv kernel has
// already L2-normalized q (with the 1/sqrt(dk) scale) and k. One block per value
// head: 16 warps = 4 value slices (32 columns) x 4 key slices (32 rows), so each
// thread holds 32 state values; partial dots are reduced through shared memory.
// The gated RMSNorm (norm(o) * w * silu(z)) is fused into the epilogue.
constexpr int kKs = 4;
constexpr int kGdnStash2 = 260; // per head and column: k[128], g, beta*delta[128]
__global__ void __launch_bounds__(512) gdn_fused_kernel(GdnLayer L, const float* conv_out, const float* ab, int ldab, const float* z,
                                                        int ldz, bf16* out, int T, const int* pending, int tmax, float eps) {
    pdl_begin();
    __shared__ float red[kKs][kHeadV];
    __shared__ float sq[4];
    const int h = blockIdx.x, kh = h / (kVHeads / kKHeads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int vb = warp & 3, ks = warp >> 2;
    const int v = vb * 32 + lane, k0 = ks * 32;
    float* S = L.state + std::size_t(h) * kHeadK * kHeadV;
    float s[32];
#pragma unroll
    for(int i = 0; i < 32; ++i) s[i] = S[(k0 + i) * kHeadV + v];
    float* stash = L.rec_stash + std::size_t(h) * tmax * kGdnStash2;
    const int m = *pending;
    for(int i = 0; i < m; ++i) {
        const float* st = stash + std::size_t(i) * kGdnStash2;
        const float g = st[kHeadK], d = st[kHeadK + 1 + v];
#pragma unroll
        for(int j = 0; j < 32; ++j) s[j] = fmaf(st[k0 + j], d, s[j] * g);
    }
    if(m > 0) {
#pragma unroll
        for(int i = 0; i < 32; ++i) S[(k0 + i) * kHeadV + v] = s[i];
    }
    const float A = -expf(bf(L.A_log[h])), dtb = bf(L.dt_bias[h]);
    const float wv = bf(L.norm_w[v]);
    for(int t = 0; t < T; ++t) {
        const float* row = conv_out + std::size_t(t) * kConv;
        float kk[32], qq[32];
#pragma unroll
        for(int j = 0; j < 8; ++j) {
            const float4 a4 = *reinterpret_cast<const float4*>(row + 2048 + kh * kHeadK + k0 + 4 * j);
            const float4 b4 = *reinterpret_cast<const float4*>(row + kh * kHeadK + k0 + 4 * j);
            kk[4 * j] = a4.x; kk[4 * j + 1] = a4.y; kk[4 * j + 2] = a4.z; kk[4 * j + 3] = a4.w;
            qq[4 * j] = b4.x; qq[4 * j + 1] = b4.y; qq[4 * j + 2] = b4.z; qq[4 * j + 3] = b4.w;
        }
        const float a = ab[std::size_t(t) * ldab + h], b = ab[std::size_t(t) * ldab + kVHeads + h];
        const float beta = 1.f / (1.f + expf(-b));
        const float g = expf(A * softplus(a + dtb));
        const float vv = row[4096 + h * kHeadV + v];
        float kv = 0.f;
#pragma unroll
        for(int j = 0; j < 32; ++j) { s[j] *= g; kv = fmaf(s[j], kk[j], kv); }
        red[ks][v] = kv;
        __syncthreads();
        const float d = (vv - (red[0][v] + red[1][v] + red[2][v] + red[3][v])) * beta;
        __syncthreads();
        float o = 0.f;
#pragma unroll
        for(int j = 0; j < 32; ++j) { s[j] = fmaf(kk[j], d, s[j]); o = fmaf(s[j], qq[j], o); }
        red[ks][v] = o;
        float* st = stash + std::size_t(t) * kGdnStash2;
        if(vb == 0) st[k0 + lane] = row[2048 + kh * kHeadK + k0 + lane];
        if(ks == 0) st[kHeadK + 1 + v] = d;
        if(threadIdx.x == 0) st[kHeadK] = g;
        __syncthreads();
        float ot = 0.f;
        if(ks == 0) {
            ot = red[0][v] + red[1][v] + red[2][v] + red[3][v];
            const float ss = warp_sum(ot * ot);
            if(lane == 0) sq[vb] = ss;
        }
        __syncthreads();
        if(ks == 0) {
            const float r = rsqrtf((sq[0] + sq[1] + sq[2] + sq[3]) / kHeadV + eps);
            const float gz = z[std::size_t(t) * ldz + h * kHeadV + v];
            out[std::size_t(t) * (kVHeads * kHeadV) + h * kHeadV + v] = __float2bfloat16(ot * r * wv * silu(gz));
        }
    }
}

// One warp owns value columns [32*vb, 32*vb+32) of value head h and the full
// 128-row key extent, so the recurrence needs no cross-warp reduction.
__global__ void __launch_bounds__(32) gdn_rec_kernel(GdnLayer L, const float* conv_out, const float* ab, int ldab, float* out,
                                                    int T, int mode, const int* pending, int tmax) {
    pdl_begin();
    __shared__ __align__(16) float kq[2][kHeadK];
    const int h = blockIdx.x, vb = blockIdx.y, lane = threadIdx.x, kh = h / (kVHeads / kKHeads);
    const int v = vb * 32 + lane;
    float* S = L.state + std::size_t(h) * kHeadK * kHeadV;
    float s[kHeadK];
#pragma unroll
    for(int i = 0; i < kHeadK; ++i) s[i] = S[i * kHeadV + v];
    float* stash = L.rec_stash + std::size_t(h * 4 + vb) * tmax * kGdnStash;
    if(mode == 1) {
        const int m = *pending;
        for(int i = 0; i < m; ++i) {
            const float* st = stash + std::size_t(i) * kGdnStash;
            const float g = st[kHeadK], d = st[kHeadK + 1 + lane];
#pragma unroll
            for(int j = 0; j < kHeadK; ++j) s[j] = fmaf(st[j], d, s[j] * g);
        }
        if(m > 0) {
#pragma unroll
            for(int i = 0; i < kHeadK; ++i) S[i * kHeadV + v] = s[i];
        }
    }
    const float A = -expf(bf(L.A_log[h])), dtb = bf(L.dt_bias[h]);
    const float qscale = rsqrtf(float(kHeadK));
    for(int t = 0; t < T; ++t) {
        const float* row = conv_out + std::size_t(t) * kConv;
        const float4 q4 = *reinterpret_cast<const float4*>(row + kh * kHeadK + lane * 4);
        const float4 k4 = *reinterpret_cast<const float4*>(row + 2048 + kh * kHeadK + lane * 4);
        const float sq = warp_sum(q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w);
        const float sk = warp_sum(k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w);
        const float rq = rsqrtf(sq + 1e-6f) * qscale, rk = rsqrtf(sk + 1e-6f);
        __syncwarp();
        reinterpret_cast<float4*>(kq[0])[lane] = make_float4(k4.x * rk, k4.y * rk, k4.z * rk, k4.w * rk);
        reinterpret_cast<float4*>(kq[1])[lane] = make_float4(q4.x * rq, q4.y * rq, q4.z * rq, q4.w * rq);
        __syncwarp();
        const float a = ab[std::size_t(t) * ldab + h], b = ab[std::size_t(t) * ldab + kVHeads + h];
        const float beta = 1.f / (1.f + expf(-b));
        const float g = expf(A * softplus(a + dtb));
        const float vv = row[4096 + h * kHeadV + v];
        float kv = 0.f;
#pragma unroll
        for(int j = 0; j < kHeadK; ++j) { s[j] *= g; kv = fmaf(s[j], kq[0][j], kv); }
        const float d = (vv - kv) * beta;
        float o = 0.f;
#pragma unroll
        for(int j = 0; j < kHeadK; ++j) { s[j] = fmaf(kq[0][j], d, s[j]); o = fmaf(s[j], kq[1][j], o); }
        out[std::size_t(t) * (kVHeads * kHeadV) + h * kHeadV + v] = o;
        if(mode == 1) {
            float* st = stash + std::size_t(t) * kGdnStash;
            reinterpret_cast<float4*>(st)[lane] = reinterpret_cast<const float4*>(kq[0])[lane];
            if(lane == 0) st[kHeadK] = g;
            st[kHeadK + 1 + lane] = d;
        }
    }
    if(mode == 0) {
#pragma unroll
        for(int i = 0; i < kHeadK; ++i) S[i * kHeadV + v] = s[i];
    }
}

__global__ void gdn_gate_norm_kernel(const float* core, const float* z, int ldz, const bf16* w, bf16* out, float eps) {
    pdl_begin();
    const int t = blockIdx.x, h = blockIdx.y * 16 + threadIdx.y, lane = threadIdx.x;
    const float* x = core + std::size_t(t) * (kVHeads * kHeadV) + h * kHeadV;
    const float4 v = reinterpret_cast<const float4*>(x)[lane];
    const float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    const float r = rsqrtf(ss / kHeadV + eps);
    const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        const float gz = z[std::size_t(t) * ldz + h * kHeadV + d];
        out[std::size_t(t) * (kVHeads * kHeadV) + h * kHeadV + d] = __float2bfloat16(vals[i] * r * bf(w[d]) * silu(gz));
    }
}

// --------------------------------------------------------------- attention
__global__ void attn_prep_kernel(const float* qkv, int ldq, const bf16* qn, const bf16* kn, const int* base,
                                 float* q_out, bf16* kc, bf16* vc, float eps) {
    pdl_begin();
    __shared__ float vals[kHeadDim];
    __shared__ float part[8];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const int pos = *base + t;
    const bool is_q = hh < kQHeads;
    const int kvh = hh - kQHeads;
    const float* src = qkv + std::size_t(t) * ldq + (is_q ? hh * 2 * kHeadDim : kQHeads * 2 * kHeadDim + kvh * kHeadDim);
    const float x = src[d];
    float ss = warp_sum(x * x);
    if((d & 31) == 0) part[d >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) tot += part[i];
    const float r = rsqrtf(tot / kHeadDim + eps);
    const float y = x * r * (1.f + bf(is_q ? qn[d] : kn[d]));
    vals[d] = y;
    __syncthreads();
    float o = y;
    if(d < kRot) {
        const int i = d & 31;
        const float inv = 1.f / powf(10000000.f, float(2 * i) / float(kRot));
        float sn, cs;
        sincosf(float(pos) * inv, &sn, &cs);
        o = d < 32 ? y * cs - vals[d + 32] * sn : y * cs + vals[d - 32] * sn;
    }
    if(is_q) q_out[(std::size_t(t) * kQHeads + hh) * kHeadDim + d] = o;
    else {
        const std::size_t slot = (std::size_t(pos) * kKvHeads + kvh) * kHeadDim + d;
        kc[slot] = __float2bfloat16(o);
        vc[slot] = __float2bfloat16(qkv[std::size_t(t) * ldq + kQHeads * 2 * kHeadDim + kKvHeads * kHeadDim + kvh * kHeadDim + d]);
    }
}

constexpr int kAttnWarps = 8;
// Partial attention of one (head, column) over keys [sp*split, min(len, (sp+1)*split)).
__global__ void __launch_bounds__(kAttnWarps * 32) attn_kernel(const float* q, const bf16* kc, const bf16* vc, const int* base,
                                                               int split, int max_splits, float* partial) {
    pdl_begin();
    __shared__ float sm[kAttnWarps], sl[kAttnWarps];
    __shared__ float sacc[kAttnWarps][kHeadDim];
    const int h = blockIdx.x, t = blockIdx.y, sp = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int len = *base + t + 1;
    float* out = partial + ((std::size_t(t) * kQHeads + h) * max_splits + sp) * (kHeadDim + 2);
    const int k0 = sp * split;
    if(k0 >= len) { if(threadIdx.x == 0) { out[0] = -FLT_MAX; out[1] = 0.f; } return; }
    const int k1 = min(len, k0 + split);
    const int kvh = h / (kQHeads / kKvHeads);
    const float* qr = q + (std::size_t(t) * kQHeads + h) * kHeadDim + lane * 8;
    float qv[8];
#pragma unroll
    for(int i = 0; i < 8; ++i) qv[i] = qr[i] * 0.0625f;
    float m = -FLT_MAX, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr int U = 4;
    for(int kb = k0 + warp * U; kb < k1; kb += kAttnWarps * U) {
        uint4 kr[U], vr[U];
#pragma unroll
        for(int u = 0; u < U; ++u) {
            const int key = kb + u;
            if(key < k1) {
                const std::size_t off = (std::size_t(key) * kKvHeads + kvh) * kHeadDim + lane * 8;
                kr[u] = *reinterpret_cast<const uint4*>(kc + off);
                vr[u] = *reinterpret_cast<const uint4*>(vc + off);
            }
        }
#pragma unroll
        for(int u = 0; u < U; ++u) {
            const int key = kb + u;
            if(key >= k1) break;
            const bf16* kk = reinterpret_cast<const bf16*>(&kr[u]);
            float dot = 0.f;
#pragma unroll
            for(int i = 0; i < 8; ++i) dot = fmaf(qv[i], bf(kk[i]), dot);
            dot = warp_sum(dot);
            const float mn = fmaxf(m, dot);
            const float corr = __expf(m - mn), p = __expf(dot - mn);
            l = l * corr + p;
            const bf16* vv = reinterpret_cast<const bf16*>(&vr[u]);
#pragma unroll
            for(int i = 0; i < 8; ++i) acc[i] = fmaf(p, bf(vv[i]), acc[i] * corr);
            m = mn;
        }
    }
    if(lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for(int i = 0; i < 8; ++i) sacc[warp][lane * 8 + i] = acc[i];
    __syncthreads();
    float M = -FLT_MAX;
#pragma unroll
    for(int w = 0; w < kAttnWarps; ++w) M = fmaxf(M, sm[w]);
    for(int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float a = 0.f;
#pragma unroll
        for(int w = 0; w < kAttnWarps; ++w) if(sl[w] > 0.f) a += sacc[w][d] * __expf(sm[w] - M);
        out[2 + d] = a;
    }
    if(threadIdx.x == 0) {
        float L = 0.f;
        for(int w = 0; w < kAttnWarps; ++w) if(sl[w] > 0.f) L += sl[w] * __expf(sm[w] - M);
        out[0] = M; out[1] = L;
    }
}
__global__ void attn_combine_kernel(const float* partial, const float* gate, int ldg, const int* base, int split, int max_splits, bf16* out) {
    pdl_begin();
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int len = *base + t + 1;
    const int n = min(max_splits, (len + split - 1) / split);
    const float* p = partial + (std::size_t(t) * kQHeads + h) * max_splits * (kHeadDim + 2);
    float M = -FLT_MAX;
    for(int s = 0; s < n; ++s) M = fmaxf(M, p[s * (kHeadDim + 2)]);
    float L = 0.f, A = 0.f;
    for(int s = 0; s < n; ++s) {
        const float* ps = p + s * (kHeadDim + 2);
        if(ps[1] <= 0.f) continue;
        const float e = __expf(ps[0] - M);
        L += ps[1] * e;
        A += ps[2 + d] * e;
    }
    const float g = gate[std::size_t(t) * ldg + h * 2 * kHeadDim + kHeadDim + d];
    out[std::size_t(t) * (kQHeads * kHeadDim) + h * kHeadDim + d] = __float2bfloat16(A / L * (1.f / (1.f + __expf(-g))));
}

// ------------------------------------------------------------------- head
// Lowest-ID argmax over [0, valid) and, when nonempty, the extra range [eb, ee).
__global__ void argmax_kernel(const float* logits, int ld, int valid, int eb, int ee, int* out) {
    pdl_begin();
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* row = logits + std::size_t(blockIdx.x) * ld;
    float best = -FLT_MAX;
    int idx = 0x7fffffff;
    for(int i = threadIdx.x; i < valid; i += blockDim.x) {
        const float v = row[i];
        if(v > best) { best = v; idx = i; }
    }
    for(int i = eb + threadIdx.x; i < ee; i += blockDim.x) {
        const float v = row[i];
        if(v > best) { best = v; idx = i; }
    }
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if(ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if(threadIdx.x < 32) {
        const int nw = blockDim.x >> 5;
        best = threadIdx.x < nw ? bv[threadIdx.x] : -FLT_MAX;
        idx = threadIdx.x < nw ? bi[threadIdx.x] : 0x7fffffff;
#pragma unroll
        for(int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if(ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
        }
        if(threadIdx.x == 0) out[blockIdx.x] = idx;
    }
}
__global__ void log_prob_kernel(const float* logits, int ld, int valid, const int* targets, float* out) {
    pdl_begin();
    __shared__ float red[32];
    const float* row = logits + std::size_t(blockIdx.x) * ld;
    float mx = -FLT_MAX;
    for(int i = threadIdx.x; i < valid; i += blockDim.x) mx = fmaxf(mx, row[i]);
    mx = warp_max(mx);
    if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = -FLT_MAX;
    for(int i = 0; i < int(blockDim.x >> 5); ++i) mx = fmaxf(mx, red[i]);
    __syncthreads();
    float s = 0.f;
    for(int i = threadIdx.x; i < valid; i += blockDim.x) s += expf(row[i] - mx);
    s = warp_sum(s);
    if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if(threadIdx.x == 0) {
        float tot = 0.f;
        for(int i = 0; i < int(blockDim.x >> 5); ++i) tot += red[i];
        out[blockIdx.x] = row[targets[blockIdx.x]] - mx - logf(tot);
    }
}

__global__ void verify_tokens_kernel(const int* control, const int* drafts, int K, int* tokens) {
    pdl_begin();
    const int i = threadIdx.x;
    if(i == 0) tokens[0] = control[1];
    else if(i <= K) tokens[i] = drafts[i - 1];
}
__global__ void set_int_kernel(int* dst, const int* src, int add) { pdl_begin(); *dst = *src + add; }
__global__ void mtp_inputs_kernel(const int* tokens, const bf16* table, const bf16* hidden, int ldh, const bf16* we, const bf16* wh,
                                  bf16* out, float eps) {
    pdl_begin();
    __shared__ float part[32];
    const int t = blockIdx.x, half = blockIdx.y;
    const bf16* x = half ? hidden + std::size_t(t) * ldh : table + std::size_t(tokens[t]) * kHidden;
    const bf16* w = half ? wh : we;
    float ss = 0.f;
    for(int d = threadIdx.x; d < kHidden; d += blockDim.x) { const float v = bf(x[d]); ss += v * v; }
    ss = warp_sum(ss);
    if((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for(int i = 0; i < int(blockDim.x >> 5); ++i) tot += part[i];
    const float r = rsqrtf(tot / kHidden + eps);
    for(int d = threadIdx.x; d < kHidden; d += blockDim.x)
        out[std::size_t(t) * 2 * kHidden + half * kHidden + d] = __float2bfloat16(bf(x[d]) * r * (1.f + bf(w[d])));
}
// Draft-only derived copy: per-row FP8 E4M3 with a BF16 row scale (amax / 448).
__global__ void quant_fp8_rows_kernel(const bf16* w, int K, uint8_t* q, bf16* rs) {
    pdl_begin();
    __shared__ float part[32];
    const bf16* row = w + std::size_t(blockIdx.x) * K;
    float amax = 0.f;
    for(int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(bf(row[k])));
    amax = warp_max(amax);
    if((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = 0.f;
    for(int i = 0; i < int(blockDim.x >> 5); ++i) amax = fmaxf(amax, part[i]);
    const bf16 s = __float2bfloat16(amax > 0.f ? amax / 448.f : 1.f);
    const float inv = 1.f / bf(s);
    uint8_t* out = q + std::size_t(blockIdx.x) * K;
    for(int k = threadIdx.x; k < K; k += blockDim.x)
        out[k] = __nv_cvt_float_to_fp8(bf(row[k]) * inv, __NV_SATFINITE, __NV_E4M3);
    if(threadIdx.x == 0) rs[blockIdx.x] = s;
}
__global__ void stream_read_kernel(const uint4* p, std::size_t n, uint4* sink) {
    pdl_begin();
    uint4 acc = make_uint4(0, 0, 0, 0);
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 v = ldg_stream(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if(acc.x == 0x12345678u) *sink = acc;
}

template<typename... KArgs, typename... Args>
void launch_pdl(const char* what, void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t s, Args... args) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    check(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...), what);
}
} // namespace

void stream_read(const void* p, std::size_t bytes, void* sink, cudaStream_t s) {
    launch_pdl("stream_read_kernel", stream_read_kernel, dim3(170 * 8), dim3(256), s, static_cast<const uint4*>(p), bytes / 16, static_cast<uint4*>(sink));
    check(cudaPeekAtLastError(), "stream read");
}

void quantize_fp8_rows(const QW& src, uint8_t* q, bf16* rs, cudaStream_t s) {
    if(src.fmt != W_BF16) throw std::invalid_argument("FP8 derivation expects BF16 rows");
    launch_pdl("quant_fp8_rows_kernel", quant_fp8_rows_kernel, dim3(src.N), dim3(256), s, reinterpret_cast<const bf16*>(src.w), src.K, q, rs);
    check(cudaPeekAtLastError(), "quantize fp8");
}

void gemv_multi(const GemvSeg* segs, int nseg, const bf16* x, int ldx, int M, cudaStream_t s) {
    if(nseg < 1 || nseg > kMaxSegs) throw std::invalid_argument("gemv segments");
    GemvLaunch L{};
    int blocks = 0;
    for(int i = 0; i < nseg; ++i) {
        const auto& g = segs[i];
        const int kq = g.w.fmt == W_NVFP4 ? 128 : 64;
        const int rows = g.act ? 8 * kRT : 16 * kRT;
        if(g.w.N % rows || g.w.K % kq || g.w.K != segs[0].w.K) throw std::invalid_argument("gemv shape");
        if(g.act && (g.w2.fmt != g.w.fmt || g.w2.N != g.w.N || g.w2.K != g.w.K)) throw std::invalid_argument("fused gemv shape");
        L.seg[i] = g;
        L.start[i] = blocks;
        blocks += g.w.N / rows;
    }
    for(int i = nseg; i <= kMaxSegs; ++i) L.start[i] = blocks;
    L.x = x; L.ldx = ldx; L.M = M;
    const int ks = segs[0].ksplit;
    for(int i = 0; i < nseg; ++i)
        if(segs[i].ksplit != ks || (ks > 1 && (segs[i].act || segs[i].accumulate))) throw std::invalid_argument("gemv split-K");
    launch_pdl("gemv_kernel", gemv_kernel, dim3(dim3(blocks, (M + 7) / 8, ks)), dim3(kGemvWarps * 32), s, L);
    check(cudaPeekAtLastError(), "gemv");
}
void gemv(const GemvSeg& a, const GemvSeg* b, const bf16* x, int ldx, int M, cudaStream_t s) {
    GemvSeg segs[2] = {a, b ? *b : a};
    gemv_multi(segs, b ? 2 : 1, x, ldx, M, s);
}
void dequant_bf16(const QW& w, bf16* out, cudaStream_t s) {
    const std::size_t n8 = std::size_t(w.N) * w.K / 8;
    launch_pdl("dequant_kernel", dequant_kernel, dim3(unsigned((n8 + 255) / 256)), dim3(256), s, w, out);
    check(cudaPeekAtLastError(), "dequant");
}
void row_scale_epilogue(const QW& w, const float* t, float* y, int ldy, int M, int accumulate, cudaStream_t s) {
    const std::size_t n = std::size_t(M) * w.N;
    launch_pdl("row_scale_kernel", row_scale_kernel, dim3(unsigned((n + 255) / 256)), dim3(256), s, w, t, y, ldy, M, accumulate);
    check(cudaPeekAtLastError(), "row scale");
}
void embed(const int* tokens, const bf16* table, float* out, int T, int D, cudaStream_t s) {
    launch_pdl("embed_kernel", embed_kernel, dim3(T), dim3(256), s, tokens, table, out, D);
    check(cudaPeekAtLastError(), "embed");
}
void rmsnorm(const float* x, int ldx, const bf16* w, bf16* out, int ldo, int T, int D, float eps, int plus_one, cudaStream_t s) {
    launch_pdl("rmsnorm_kernel<float>", rmsnorm_kernel<float>, dim3(T), dim3(512), s, x, ldx, w, out, ldo, D, eps, plus_one);
    check(cudaPeekAtLastError(), "rmsnorm");
}
void rmsnorm_bf(const bf16* x, int ldx, const bf16* w, bf16* out, int ldo, int T, int D, float eps, int plus_one, cudaStream_t s) {
    launch_pdl("rmsnorm_kernel<bf16>", rmsnorm_kernel<bf16>, dim3(T), dim3(512), s, x, ldx, w, out, ldo, D, eps, plus_one);
    check(cudaPeekAtLastError(), "rmsnorm");
}
void residual_norm(float* resid, const float* partial, int nsplit, std::size_t stride, const bf16* w, bf16* out, int T, int D,
                   float eps, cudaStream_t s) {
    launch_pdl("residual_norm_kernel", residual_norm_kernel, dim3(T), dim3(512), s, resid, partial, nsplit, stride, w, out, D, eps);
}
void swiglu(const float* gu, int ld, bf16* out, int T, int I, cudaStream_t s) {
    launch_pdl("swiglu_kernel", swiglu_kernel, dim3(dim3(32, T)), dim3(256), s, gu, ld, out, I);
    check(cudaPeekAtLastError(), "swiglu");
}
void gdn_conv(const GdnLayer& L, const float* qkvz, int ldq, float* out, int T, int mode, const int* pending, cudaStream_t s, int normalize) {
    launch_pdl("gdn_conv_kernel", gdn_conv_kernel, dim3(kConv / 256), dim3(256), s, L, qkvz, ldq, out, T, mode, pending, normalize);
    check(cudaPeekAtLastError(), "gdn conv");
}
void gdn_decode(const GdnLayer& L, const float* conv_out, const float* ab, int ldab, const float* z, int ldz, bf16* out, int T,
                const int* pending, int tmax, float eps, cudaStream_t s) {
    if(std::size_t(kVHeads) * tmax * kGdnStash2 > std::size_t(kVHeads * 4) * tmax * kGdnStash) throw std::logic_error("stash size");
    launch_pdl("gdn_fused_kernel", gdn_fused_kernel, dim3(kVHeads), dim3(kKs * 4 * 32), s, L, conv_out, ab, ldab, z, ldz, out, T, pending, tmax, eps);
}
void gdn_recurrent(const GdnLayer& L, const float* conv_out, const float* ab, int ldab, float* out, int T, int mode,
                   const int* pending, int tmax, cudaStream_t s) {
    launch_pdl("gdn_rec_kernel", gdn_rec_kernel, dim3(dim3(kVHeads, kHeadV / 32)), dim3(32), s, L, conv_out, ab, ldab, out, T, mode, pending, tmax);
    check(cudaPeekAtLastError(), "gdn recurrent");
}
void gdn_gate_norm(const float* core, const float* z, int ldz, const bf16* w, bf16* out, int T, float eps, cudaStream_t s) {
    launch_pdl("gdn_gate_norm_kernel", gdn_gate_norm_kernel, dim3(dim3(T, kVHeads / 16)), dim3(dim3(32, 16)), s, core, z, ldz, w, out, eps);
    check(cudaPeekAtLastError(), "gdn gate norm");
}
void attn_prep(const float* qkv, int ldq, const bf16* qn, const bf16* kn, const int* base, int T,
               float* q_out, bf16* kc, bf16* vc, float eps, cudaStream_t s) {
    launch_pdl("attn_prep_kernel", attn_prep_kernel, dim3(dim3(T, kQHeads + kKvHeads)), dim3(kHeadDim), s, qkv, ldq, qn, kn, base, q_out, kc, vc, eps);
    check(cudaPeekAtLastError(), "attn prep");
}
void attention(const float* q, const float* qkv_gate, int ldg, const bf16* kc, const bf16* vc, const int* base, int T,
               int split, int max_splits, float* partial, bf16* out, cudaStream_t s) {
    launch_pdl("attn_kernel", attn_kernel, dim3(dim3(kQHeads, T, max_splits)), dim3(kAttnWarps * 32), s, q, kc, vc, base, split, max_splits, partial);
    check(cudaPeekAtLastError(), "attention");
    launch_pdl("attn_combine_kernel", attn_combine_kernel, dim3(dim3(kQHeads, T)), dim3(kHeadDim), s, partial, qkv_gate, ldg, base, split, max_splits, out);
    check(cudaPeekAtLastError(), "attention combine");
}
void argmax(const float* logits, int ld, int T, int valid, int* out, cudaStream_t s, int eb, int ee) {
    launch_pdl("argmax_kernel", argmax_kernel, dim3(T), dim3(1024), s, logits, ld, valid, eb, ee, out);
    check(cudaPeekAtLastError(), "argmax");
}
void log_prob(const float* logits, int ld, int T, int valid, const int* targets, float* out, cudaStream_t s) {
    launch_pdl("log_prob_kernel", log_prob_kernel, dim3(T), dim3(1024), s, logits, ld, valid, targets, out);
    check(cudaPeekAtLastError(), "log prob");
}
void verify_tokens(const int* control, const int* drafts, int K, int* tokens, cudaStream_t s) {
    launch_pdl("verify_tokens_kernel", verify_tokens_kernel, dim3(1), dim3(32), s, control, drafts, K, tokens);
    check(cudaPeekAtLastError(), "verify tokens");
}
void set_int(int* dst, const int* src, int add, cudaStream_t s) {
    launch_pdl("set_int_kernel", set_int_kernel, dim3(1), dim3(1), s, dst, src, add);
    check(cudaPeekAtLastError(), "set int");
}
void mtp_inputs(const int* tokens, const bf16* embed_table, const bf16* hidden, int ldh, const bf16* we, const bf16* wh,
                bf16* out, int T, float eps, cudaStream_t s) {
    launch_pdl("mtp_inputs_kernel", mtp_inputs_kernel, dim3(dim3(T, 2)), dim3(512), s, tokens, embed_table, hidden, ldh, we, wh, out, eps);
    check(cudaPeekAtLastError(), "mtp inputs");
}
}
