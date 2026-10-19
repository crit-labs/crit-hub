// nmh: crit engine for nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (nemotron_h text path).
// Model dimensions and shared device helpers.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "dims.h"

namespace nmh {

#define NMH_CUDA(call)                                                                                  \
    do {                                                                                                \
        cudaError_t nmh_err_ = (call);                                                                  \
        if(nmh_err_ != cudaSuccess)                                                                     \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(nmh_err_) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__));                        \
    } while(0)

// Layer kinds by index (layers_block_type).
enum class Kind : int { mamba = 0, moe = 1, attn = 2 };

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float relu2f(float x) { const float r = fmaxf(x, 0.f); return r * r; }

__device__ __forceinline__ float e4m3f(uint32_t b) {
    __nv_fp8_e4m3 v;
    v.__x = static_cast<__nv_fp8_storage_t>(b);
    return static_cast<float>(v);
}
// Two E4M3 bytes (low byte first) to two floats (exact).
__device__ __forceinline__ float2 e4m3x2f(uint32_t b2) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(b2 & 0xffff), __NV_E4M3);
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    return __half22float2(hh);
}
// Two E2M1 nibbles of a byte to floats (low nibble first, exact).
__device__ __forceinline__ float2 e2m1x2f(uint32_t byte) {
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte & 0xff), __NV_E2M1);
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    return __half22float2(hh);
}
// 16 E2M1 values (8 bytes) to floats.
__device__ __forceinline__ void decode16_e2m1(uint2 w, float* f) {
    const uint32_t words[2] = {w.x, w.y};
#pragma unroll
    for(int q = 0; q < 2; ++q)
#pragma unroll
        for(int b = 0; b < 4; ++b) {
            const float2 v = e2m1x2f((words[q] >> (8 * b)) & 0xff);
            f[q * 8 + b * 2] = v.x;
            f[q * 8 + b * 2 + 1] = v.y;
        }
}

}  // namespace nmh
