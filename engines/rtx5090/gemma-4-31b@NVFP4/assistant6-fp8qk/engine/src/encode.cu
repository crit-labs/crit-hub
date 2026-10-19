// Load-time NVFP4 encoding of draft-only weights (the MTP layer and a copy of the output head used as the draft
// head): per-tensor global scale gs = 448 * 6 / amax, E4M3 block scales of 16 elements (swizzled), E2M1 codes.
#include <cuda_fp8.h>

#include <cstring>

#include "common.h"

namespace g4 {
namespace kenc {

__device__ __forceinline__ float fp8_value(std::uint8_t c) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(c), __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}
__device__ __forceinline__ float load(const __nv_bfloat16* bf, const std::uint8_t* f8, const float* sc, int cols, int r, int c) {
    const std::size_t i = static_cast<std::size_t>(r) * cols + c;
    return bf ? __bfloat162float(bf[i]) : fp8_value(f8[i]) * sc[r];
}
__device__ __forceinline__ std::uint32_t e2m1_pair(float lo, float hi) {
    std::uint32_t r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__global__ void amax_kernel(const __nv_bfloat16* bf, const std::uint8_t* f8, const float* sc, int rows, int cols, unsigned* amax) {
    float m = 0.f;
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    for(std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(load(bf, f8, sc, cols, static_cast<int>(i / cols), static_cast<int>(i % cols))));
    for(int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}

// One thread per 16-element block.
__global__ void nvfp4_kernel(const __nv_bfloat16* bf, const std::uint8_t* f8, const float* sc, int rows, int cols,
                             const unsigned* amax, std::uint8_t* packed, std::uint8_t* sf) {
    const int blocks = cols / 16;
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if(idx >= static_cast<std::size_t>(rows) * blocks) return;
    const int r = static_cast<int>(idx / blocks), b = static_cast<int>(idx % blocks);
    const float am = __uint_as_float(*amax);
    const float gs = am > 0.f ? 448.f * 6.f / am : 1.f;
    float v[16], bm = 0.f;
#pragma unroll
    for(int i = 0; i < 16; ++i) {
        v[i] = load(bf, f8, sc, cols, r, b * 16 + i);
        bm = fmaxf(bm, fabsf(v[i]));
    }
    const std::uint8_t code = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(bm / 6.f * gs, __NV_SATFINITE, __NV_E4M3));
    const float sdec = fp8_value(code);
    const float inv = sdec > 0.f ? gs / sdec : 0.f;
    std::uint32_t lo = 0, hi = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i) lo |= e2m1_pair(v[2 * i] * inv, v[2 * i + 1] * inv) << (8 * i);
#pragma unroll
    for(int i = 0; i < 4; ++i) hi |= e2m1_pair(v[8 + 2 * i] * inv, v[8 + 2 * i + 1] * inv) << (8 * i);
    *reinterpret_cast<uint2*>(packed + static_cast<std::size_t>(r) * (cols / 2) + b * 8) = make_uint2(lo, hi);
    sf[sf_offset(r, b, sf_blocks_padded(cols))] = code;
}

}  // namespace kenc
using namespace kenc;

// Encode a BF16 matrix (bf) or an FP8 matrix with row scales (f8, sc) to NVFP4. packed: rows * cols / 2 bytes,
// sf: sf_bytes(rows, cols) bytes (zeroed here). Returns alpha = 1 / gs. Synchronous.
float encode_nvfp4(const __nv_bfloat16* bf, const std::uint8_t* f8, const float* sc, int rows, int cols,
                   std::uint8_t* packed, std::uint8_t* sf) {
    unsigned* amax = nullptr;
    KCHECK(cudaMalloc(&amax, 4));
    KCHECK(cudaMemset(amax, 0, 4));
    KCHECK(cudaMemset(sf, 0, sf_bytes(rows, cols)));
    amax_kernel<<<1024, 256>>>(bf, f8, sc, rows, cols, amax);
    KLAUNCH();
    const std::size_t blocks = static_cast<std::size_t>(rows) * (cols / 16);
    nvfp4_kernel<<<static_cast<unsigned>((blocks + 255) / 256), 256>>>(bf, f8, sc, rows, cols, amax, packed, sf);
    KLAUNCH();
    unsigned h = 0;
    KCHECK(cudaMemcpy(&h, amax, 4, cudaMemcpyDeviceToHost));
    KCHECK(cudaFree(amax));
    float a;
    std::memcpy(&a, &h, 4);
    return a > 0.f ? a / (448.f * 6.f) : 1.f;
}

}  // namespace g4
