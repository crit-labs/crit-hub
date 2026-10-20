// kestrel: shared definitions for the Qwen3.8-27B NVFP4 engine.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kestrel {

inline void cuda_check(cudaError_t e, const char* what, const char* file, int line) {
    if(e != cudaSuccess)
        throw std::runtime_error(std::string(what) + " failed at " + file + ":" + std::to_string(line) + ": " +
                                 cudaGetErrorString(e));
}
#define KCHECK(x) ::kestrel::cuda_check((x), #x, __FILE__, __LINE__)
#define KLAUNCH() ::kestrel::cuda_check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

// Model geometry (Qwen3.8-27B text model).
constexpr int kHidden = 5120;
constexpr int kLayers = 64;
constexpr int kInter = 17408;
constexpr int kVocab = 248320;       // padded rows of embedding and head
constexpr int kVocabValid = 248077;  // tokenizer ids
constexpr int kHeads = 24, kKvHeads = 4, kHeadDim = 256, kRotary = 64;
constexpr int kQRows = kHeads * kHeadDim * 2;  // 12288: query | gate interleaved per head
constexpr int kKvRows = kKvHeads * kHeadDim;    // 1024
constexpr int kAttnOut = kHeads * kHeadDim;     // 6144
constexpr int kLinKHeads = 16, kLinVHeads = 48, kLinDim = 128;
constexpr int kLinQkv = 2 * kLinKHeads * kLinDim + kLinVHeads * kLinDim;  // 10240
constexpr int kLinZ = kLinVHeads * kLinDim;                               // 6144
constexpr int kConv = 4;
constexpr float kEps = 1e-6f;
constexpr double kRopeTheta = 1e7;
constexpr int kMaxCols = 8;  // most columns a decode/verify pass handles

inline bool is_attention_layer(int layer) { return layer % 4 == 3; }

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
    float alpha = 1.f;  // 1 / weight_global_scale
    int rows = 0, cols = 0;
    float igs = 0.f;                       // checkpoint input_global_scale (static activation scale)
    const float* alpha_static = nullptr;   // device: alpha / igs, the GEMM alpha with static activations
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
// launch early. launch_k adds the stream-serialization attribute unless KESTREL_NO_PDL=1.
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

}  // namespace kestrel
