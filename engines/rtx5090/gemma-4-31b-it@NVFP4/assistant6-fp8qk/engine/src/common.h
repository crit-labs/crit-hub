// g4: shared definitions for the Gemma 4 31B IT NVFP4 engine.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace g4 {

inline void cuda_check(cudaError_t e, const char* what, const char* file, int line) {
    if(e != cudaSuccess)
        throw std::runtime_error(std::string(what) + " failed at " + file + ":" + std::to_string(line) + ": " +
                                 cudaGetErrorString(e));
}
#define KCHECK(x) ::g4::cuda_check((x), #x, __FILE__, __LINE__)
#define KLAUNCH() ::g4::cuda_check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

// Target geometry (Gemma 4 31B text model).
constexpr int kHidden = 5376;
constexpr int kLayers = 60;
constexpr int kInter = 21504;
constexpr int kVocab = 262144;       // rows of the tied embedding / output head
constexpr int kVocabValid = 262144;  // every row is a token
constexpr int kHeads = 32;
// Sliding-window layers: 16 KV heads of 256, full RoPE (theta 1e4), window 1024.
constexpr int kSHd = 256, kSKv = 16, kWindow = 1024;
constexpr int kSQ = kHeads * kSHd;        // 8192
constexpr int kSKvRows = kSKv * kSHd;     // 4096
constexpr int kSQkv = kSQ + 2 * kSKvRows; // 16384
// Full-attention layers: 4 KV heads of 512, keys equal values (one k_proj), proportional RoPE on 128 dims (theta 1e6).
constexpr int kGHd = 512, kGKv = 4;
constexpr int kGQ = kHeads * kGHd;        // 16384
constexpr int kGKvRows = kGKv * kGHd;     // 2048
constexpr int kGQkv = kGQ + kGKvRows;     // 18432
constexpr int kGRopePairs = 64;           // rotated pairs (d, d + 256) for d < 64
constexpr int kMaxAttnOut = kGQ;
constexpr float kEps = 1e-6f;
constexpr float kSoftcap = 30.f;
constexpr int kMaxCols = 8;  // most columns a decode/verify pass handles

inline bool is_global_layer(int layer) { return layer % 6 == 5; }

// Drafter (google/gemma-4-31B-it-assistant): 4 layers of hidden 1024 on the target's KV of layers 58 / 59.
constexpr int kDHidden = 1024, kDLayers = 4, kDInter = 8192;
constexpr int kDSrcSliding = 58, kDSrcGlobal = 59;

// Matrix encodings on the device.
// FP8: row-major E4M3 codes [rows][cols] with an FP32 scale per row.
struct Fp8Mat {
    const std::uint8_t* w = nullptr;
    const float* scale = nullptr;
    int rows = 0, cols = 0;
};
// NVFP4: row-major packed E2M1 [rows][cols / 2] (low nibble first) and E4M3 block scales (16 elements) in the
// cuBLASLt VEC16 swizzled layout (128-row x 4-block tiles). Value = e2m1 * block_scale * alpha.
struct Fp4Mat {
    const std::uint8_t* w = nullptr;
    const std::uint8_t* sf = nullptr;
    float alpha = 1.f;  // weight_scale_2 (ModelOpt) or 1 / global scale
    int rows = 0, cols = 0;
    float igs = 0.f;                       // static activation global scale (1 / ModelOpt input_scale)
    const float* alpha_static = nullptr;   // device: alpha / igs, the GEMM alpha with static activations
    const float* alpha_dev = nullptr;      // device: alpha, the GEMM alpha with per-row activation scales
};

// Offset of the scale of (row, block) in the swizzled VEC16 layout with `blocks` scale columns (padded to 4).
__host__ __device__ inline std::size_t sf_offset(int row, int block, int blocks_padded) {
    const std::size_t tile = static_cast<std::size_t>(row >> 7) * (blocks_padded >> 2) + (block >> 2);
    return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (block & 3);
}
__host__ __device__ inline int sf_blocks_padded(int cols) { return ((cols / 16) + 3) / 4 * 4; }
__host__ __device__ inline std::size_t sf_bytes(int rows, int cols) {
    return static_cast<std::size_t>((rows + 127) / 128 * 128) * sf_blocks_padded(cols);
}

// A projection output: FP32 values, or raw BF16 GEMM results times a per-column weight scale and a per-row
// activation scale (either scale may be null = 1).
struct Proj {
    const float* f32 = nullptr;
    const __nv_bfloat16* bf = nullptr;
    const float* wscale = nullptr;
    const float* ascale = nullptr;
    int ld = 0;
    __device__ __forceinline__ float at(int r, int c) const {
        if(f32) return f32[static_cast<std::size_t>(r) * ld + c];
        float v = __bfloat162float(bf[static_cast<std::size_t>(r) * ld + c]);
        if(wscale) v *= wscale[c];
        if(ascale) v *= ascale[r];
        return v;
    }
    // Columns c..c+7 (c % 8 == 0, 16-byte aligned rows).
    __device__ __forceinline__ void at8(int r, int c, float (&v)[8]) const {
        if(f32) {
            const float4* p = reinterpret_cast<const float4*>(f32 + static_cast<std::size_t>(r) * ld + c);
            const float4 a = p[0], b = p[1];
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
            return;
        }
        const uint4 u = *reinterpret_cast<const uint4*>(bf + static_cast<std::size_t>(r) * ld + c);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            v[2 * i] = f.x;
            v[2 * i + 1] = f.y;
        }
        if(wscale) {
            const float4 a = *reinterpret_cast<const float4*>(wscale + c), b = *reinterpret_cast<const float4*>(wscale + c + 4);
            v[0] *= a.x; v[1] *= a.y; v[2] *= a.z; v[3] *= a.w; v[4] *= b.x; v[5] *= b.y; v[6] *= b.z; v[7] *= b.w;
        }
        if(ascale) {
            const float s = ascale[r];
#pragma unroll
            for(int i = 0; i < 8; ++i) v[i] *= s;
        }
    }
};

// Programmatic dependent launch: kernels wait for their predecessor before touching memory and let their successor
// launch early. launch_k adds the stream-serialization attribute unless G4_NO_PDL=1.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;"); }
bool pdl_enabled();
template <typename... K, typename... A>
inline void launch_k(void (*kernel)(K...), dim3 grid, dim3 block, std::size_t smem, cudaStream_t stream, A&&... args) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl_enabled() ? 1 : 0;
    KCHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<K>(args)...));
}

// Positions: kernels read the first position from *dev when it is set, else use host.
struct Pos {
    const int* dev = nullptr;
    int host = 0;
};

}  // namespace g4
