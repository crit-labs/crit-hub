// fnl: crit engine for nvidia/Qwen3.8-Flash-Next-FP4-Local (qwen4_exp). Shared device helpers.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fnl {

#define FNL_CUDA(call)                                                                                  \
    do {                                                                                                \
        cudaError_t fnl_err_ = (call);                                                                  \
        if(fnl_err_ != cudaSuccess)                                                                     \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(fnl_err_) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__));                        \
    } while(0)

// Model dimensions (config.json text_config of the checkpoint; checked at load).
constexpr int kHidden = 2560, kStreams = 4, kHyper = kHidden * kStreams, kLowrank = 320;
constexpr int kLayers = 48, kVocab = 248320;
constexpr int kKHeads = 16, kVHeads = 48, kKDim = 128, kVDim = 128;
constexpr int kQkvDim = 2 * kKHeads * kKDim + kVHeads * kVDim;  // 10240
constexpr int kZDim = kVHeads * kVDim;                           // 6144
constexpr int kQHeads = 24, kKvHeads = 2, kHeadDim = 256, kRotary = 64, kGroup = kQHeads / kKvHeads;
constexpr int kQProj = kQHeads * kHeadDim * 2;                   // 12288 (query and gate per head)
constexpr int kAttnOut = kQHeads * kHeadDim;                     // 6144
constexpr int kIdxHeads = 4, kIdxDim = 128, kIdxProj = (kIdxHeads + 1) * kIdxDim;  // 640
constexpr int kBlockTokens = 4, kTopBlocks = 512;                // indexer_compress_ratio, budget / ratio
constexpr int kExperts = 512, kTopK = 10, kInter = 640, kShared = 640;
constexpr int kPleHeads = 16, kPleDim = 160, kPleLayer = 1, kPleConv = 4, kPleDilation = 3;
constexpr int kPleState = (kPleConv - 1) * kPleDilation;         // 9
constexpr int kConv = 4;
constexpr float kEps = 1e-6f;

// One routed expert in a device slot or in host memory (identical weight layout).
constexpr std::size_t kGateBytes = std::size_t(kInter) * kHidden / 2;          // 819200
constexpr std::size_t kDownBytes = std::size_t(kHidden) * kInter / 2;          // 819200
constexpr std::size_t kExpertWeightBytes = 2 * kGateBytes + kDownBytes;        // 2457600
constexpr std::size_t kGateScales = std::size_t(kInter) * (kHidden / 16);      // 102400
constexpr std::size_t kDownScales = std::size_t(kHidden) * (kInter / 16);      // 102400
constexpr std::size_t kSlotBytes = kExpertWeightBytes + 2 * kGateScales + kDownScales;  // 2764800
constexpr std::size_t kGateCsf = (kInter / 16) * 16 * (1 + kHidden / 32);       // 51840
constexpr std::size_t kDownCsf = (kHidden / 16) * 16 * (1 + kInter / 32);       // 53760
constexpr std::size_t kHostExpertBytes = (kExpertWeightBytes + 2 * kGateCsf + kDownCsf + 255) / 256 * 256;

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
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// E4M3 byte to float through the hardware conversion.
__device__ __forceinline__ float e4m3f(uint32_t b) {
    __nv_fp8_e4m3 v;
    v.__x = static_cast<__nv_fp8_storage_t>(b);
    return static_cast<float>(v);
}
// Two E4M3 bytes (low byte first) to two floats.
__device__ __forceinline__ float2 e4m3x2f(uint32_t b2) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(b2 & 0xffff), __NV_E4M3);
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    return __half22float2(hh);
}
// UE8M0 scale byte to float: 2^(s - 127) (s in 1..254 for every stored scale).
__device__ __forceinline__ float ue8m0f(uint32_t s) { return __uint_as_float(s << 23); }
// Two E2M1 nibbles of a byte to floats (low nibble first), exact.
__device__ __forceinline__ float2 e2m1x2f(uint32_t byte) {
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte & 0xff), __NV_E2M1);
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    return __half22float2(hh);
}
// Two E2M1 nibbles to a bf16 pair (exact).
__device__ __forceinline__ __nv_bfloat162 e2m1x2bf(uint32_t byte) {
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte & 0xff), __NV_E2M1);
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    return __float22bfloat162_rn(__half22float2(hh));
}

}  // namespace fnl
