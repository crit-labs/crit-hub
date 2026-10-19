#include "fp4_prompt.cuh"
#include "fp4_grouped.h"

namespace nmh {
namespace {

__device__ __forceinline__ uint8_t to_e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}
// Two values -> one byte of E2M1 codes, `lo` in the low nibble.
__device__ __forceinline__ uint32_t e2m1_pair(float lo, float hi) {
    uint32_t r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// 16 values -> 8 bytes of codes and the block scale code.
__device__ __forceinline__ void fp4_block(const float (&v)[16], float gs, uint2& codes, uint8_t& scode) {
    float bm = 0.f;
#pragma unroll
    for(int i = 0; i < 16; ++i) bm = fmaxf(bm, fabsf(v[i]));
    scode = to_e4m3(bm / 6.f * gs);
    const float sdec = e4m3f(scode);
    const float inv = sdec > 0.f ? gs / sdec : 0.f;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i) lo |= e2m1_pair(v[2 * i] * inv, v[2 * i + 1] * inv) << (8 * i);
#pragma unroll
    for(int i = 0; i < 4; ++i) hi |= e2m1_pair(v[8 + 2 * i] * inv, v[8 + 2 * i + 1] * inv) << (8 * i);
    codes = make_uint2(lo, hi);
}

// grid (kExperts, 84), block 256: 86,016 scale words of an expert, 1,024 per block.
__global__ void swizzle_kernel(const uint8_t* const* group_base, uint8_t* wsf) {
    const int e = blockIdx.x;
    const uint8_t* base = group_base[e];
    if(!base) return;
    uint8_t* dst = wsf + std::size_t(e) * kSfExpertBytes;
    constexpr int kUpWords = kInter * kUpK16 / 4, kDownWords = kLatent * kDownK16 / 4;
    for(int i = threadIdx.x; i < 1024; i += blockDim.x) {
        const int w = blockIdx.y * 1024 + i;
        if(w < kUpWords) {
            const int row = w / (kUpK16 / 4), b = (w % (kUpK16 / 4)) * 4;
            const uint32_t v = *reinterpret_cast<const uint32_t*>(base + kOffUpS + std::size_t(row) * kUpK16 + b);
            *reinterpret_cast<uint32_t*>(dst + sf_offset(row, b, kUpK16)) = v;
        } else if(w < kUpWords + kDownWords) {
            const int x = w - kUpWords, row = x / (kDownK16 / 4), b = (x % (kDownK16 / 4)) * 4;
            const uint32_t v = *reinterpret_cast<const uint32_t*>(base + kOffDownS + std::size_t(row) * kDownK16 + b);
            *reinterpret_cast<uint32_t*>(dst + kSfUpBytes + sf_offset(row, b, kDownK16)) = v;
        }
    }
}

// grid (ntiles), block 256: (row, 16-value block) items of one tile.
__global__ void quant_gather_kernel(const __half* x, const int* route_token, const int4* tiles, const int* offsets,
                                    const int* sfrow, float gs, uint8_t* xq, uint8_t* xsf) {
    const int4 t = tiles[blockIdx.x];
    const int e = t.x, rows = t.z - t.y;
    for(int i = threadIdx.x; i < rows * kUpK16; i += blockDim.x) {
        const int r = t.y + i / kUpK16, b = i % kUpK16;
        const uint4* src = reinterpret_cast<const uint4*>(x + std::size_t(route_token[r]) * kLatent + b * 16);
        const uint4 u0 = src[0], u1 = src[1];
        const __half2* h0 = reinterpret_cast<const __half2*>(&u0);
        const __half2* h1 = reinterpret_cast<const __half2*>(&u1);
        float v[16];
#pragma unroll
        for(int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(h0[j]), c = __half22float2(h1[j]);
            v[2 * j] = a.x;
            v[2 * j + 1] = a.y;
            v[8 + 2 * j] = c.x;
            v[8 + 2 * j + 1] = c.y;
        }
        uint2 codes;
        uint8_t sc;
        fp4_block(v, gs, codes, sc);
        *reinterpret_cast<uint2*>(xq + std::size_t(r) * (kLatent / 2) + b * 8) = codes;
        xsf[sf_offset(sfrow[e] + r - offsets[e], b, kUpK16)] = sc;
    }
}

__global__ void relu2_quant_kernel(const __nv_bfloat16* up, const int4* tiles, const int* offsets, const int* sfrow,
                                   float gs, uint8_t* aq, uint8_t* asf) {
    const int4 t = tiles[blockIdx.x];
    const int e = t.x, rows = t.z - t.y;
    for(int i = threadIdx.x; i < rows * kDownK16; i += blockDim.x) {
        const int r = t.y + i / kDownK16, b = i % kDownK16;
        const uint4* src = reinterpret_cast<const uint4*>(up + std::size_t(r) * kInter + b * 16);
        const uint4 u0 = src[0], u1 = src[1];
        const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
        float v[16];
#pragma unroll
        for(int j = 0; j < 8; ++j) {
            v[2 * j] = relu2f(bf16_lo(w[j]));
            v[2 * j + 1] = relu2f(bf16_hi(w[j]));
        }
        uint2 codes;
        uint8_t sc;
        fp4_block(v, gs, codes, sc);
        *reinterpret_cast<uint2*>(aq + std::size_t(r) * (kInter / 2) + b * 8) = codes;
        asf[sf_offset(sfrow[e] + r - offsets[e], b, kDownK16)] = sc;
    }
}

// grid (ceil(T * K/16 / 256)), block 256: one 16-value block per thread.
__global__ void quant_rows_kernel(const __half* x, int T, int K, float gs, uint8_t* q, uint8_t* sf) {
    const int k16 = K / 16;
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if(i >= std::size_t(T) * k16) return;
    const int r = int(i / k16), b = int(i % k16);
    const uint4* src = reinterpret_cast<const uint4*>(x + std::size_t(r) * K + b * 16);
    const uint4 u0 = src[0], u1 = src[1];
    const __half2* h0 = reinterpret_cast<const __half2*>(&u0);
    const __half2* h1 = reinterpret_cast<const __half2*>(&u1);
    float v[16];
#pragma unroll
    for(int j = 0; j < 4; ++j) {
        const float2 a = __half22float2(h0[j]), c = __half22float2(h1[j]);
        v[2 * j] = a.x;
        v[2 * j + 1] = a.y;
        v[8 + 2 * j] = c.x;
        v[8 + 2 * j + 1] = c.y;
    }
    uint2 codes;
    uint8_t sc;
    fp4_block(v, gs, codes, sc);
    *reinterpret_cast<uint2*>(q + std::size_t(r) * (K / 2) + b * 8) = codes;
    sf[sf_offset(r, b, k16)] = sc;
}

__global__ void relu2_quant_rows_kernel(const __nv_bfloat16* in, int T, int K, float gs, uint8_t* q, uint8_t* sf) {
    const int k16 = K / 16;
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if(i >= std::size_t(T) * k16) return;
    const int r = int(i / k16), b = int(i % k16);
    const uint4* src = reinterpret_cast<const uint4*>(in + std::size_t(r) * K + b * 16);
    const uint4 u0 = src[0], u1 = src[1];
    const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    float v[16];
#pragma unroll
    for(int j = 0; j < 8; ++j) {
        v[2 * j] = relu2f(bf16_lo(w[j]));
        v[2 * j + 1] = relu2f(bf16_hi(w[j]));
    }
    uint2 codes;
    uint8_t sc;
    fp4_block(v, gs, codes, sc);
    *reinterpret_cast<uint2*>(q + std::size_t(r) * (K / 2) + b * 8) = codes;
    sf[sf_offset(r, b, k16)] = sc;
}

__global__ void swizzle_rows_kernel(const uint8_t* src, int rows, int k16, uint8_t* dst) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if(i >= std::size_t(rows) * (k16 / 4)) return;
    const int r = int(i / (k16 / 4)), b = int(i % (k16 / 4)) * 4;
    *reinterpret_cast<uint32_t*>(dst + sf_offset(r, b, k16)) =
        *reinterpret_cast<const uint32_t*>(src + std::size_t(r) * k16 + b);
}

__global__ void add_f32_kernel(float* h, const float* d, std::size_t n4) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if(i >= n4) return;
    float4 a = reinterpret_cast<float4*>(h)[i];
    const float4 b = reinterpret_cast<const float4*>(d)[i];
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    reinterpret_cast<float4*>(h)[i] = a;
}

}  // namespace

void quant_rows_fp4(const __half* x, int T, int K, float gs, uint8_t* q, uint8_t* sf, cudaStream_t stream) {
    const std::size_t n = std::size_t(T) * (K / 16);
    if(!n) return;
    quant_rows_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(x, T, K, gs, q, sf);
    NMH_CUDA(cudaGetLastError());
}

void relu2_quant_rows_fp4(const __nv_bfloat16* in, int T, int K, float gs, uint8_t* q, uint8_t* sf, cudaStream_t stream) {
    const std::size_t n = std::size_t(T) * (K / 16);
    if(!n) return;
    relu2_quant_rows_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(in, T, K, gs, q, sf);
    NMH_CUDA(cudaGetLastError());
}

void swizzle_rows(const uint8_t* src, int rows, int k16, uint8_t* dst, cudaStream_t stream) {
    const std::size_t n = std::size_t(rows) * (k16 / 4);
    swizzle_rows_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(src, rows, k16, dst);
    NMH_CUDA(cudaGetLastError());
}

void add_f32(float* h, const float* d, std::size_t n, cudaStream_t stream) {
    const std::size_t n4 = n / 4;
    if(!n4) return;
    add_f32_kernel<<<unsigned((n4 + 255) / 256), 256, 0, stream>>>(h, d, n4);
    NMH_CUDA(cudaGetLastError());
}

void swizzle_expert_scales(const uint8_t* const* group_base, uint8_t* wsf, cudaStream_t stream) {
    constexpr int words = (kInter * kUpK16 + kLatent * kDownK16) / 4;
    swizzle_kernel<<<dim3(kExperts, (words + 1023) / 1024), 256, 0, stream>>>(group_base, wsf);
    NMH_CUDA(cudaGetLastError());
}

void quant_gather_fp4(const __half* x, const int* route_token, const int4* tiles, int ntiles, const int* offsets,
                      const int* sfrow, float gs, uint8_t* xq, uint8_t* xsf, cudaStream_t stream) {
    if(!ntiles) return;
    quant_gather_kernel<<<ntiles, 256, 0, stream>>>(x, route_token, tiles, offsets, sfrow, gs, xq, xsf);
    NMH_CUDA(cudaGetLastError());
}

void relu2_quant_fp4(const __nv_bfloat16* up, const int4* tiles, int ntiles, const int* offsets, const int* sfrow,
                     float gs, uint8_t* aq, uint8_t* asf, cudaStream_t stream) {
    if(!ntiles) return;
    relu2_quant_kernel<<<ntiles, 256, 0, stream>>>(up, tiles, offsets, sfrow, gs, aq, asf);
    NMH_CUDA(cudaGetLastError());
}

}  // namespace nmh
