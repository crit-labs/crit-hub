// Grouped NVFP4 W4A4 GEMMs on CUTLASS SM120 block-scaled ptr-array kernels (prompt-path routed experts):
//   for each group g: D_g[m][n] = alpha_g * sum_k A_g[m][k] B_g[n][k]
// A_g row-major [M_g][K] packed E2M1 (activations), B_g row-major [N][K] packed E2M1 (expert weights as stored),
// one UE4M3 scale per 16 values of K for both in the Sm1xx block-scaled layout (128 x 4 atoms, rows padded to 128;
// sf_offset below), D_g row-major [M_g][N]. FP32 accumulation; the kernel and tiling do not depend on M_g.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nmh {

struct Fp4Group {
    int M = 0;
    const uint8_t* A = nullptr;
    const uint8_t* SFA = nullptr;
    const uint8_t* B = nullptr;
    const uint8_t* SFB = nullptr;
    const float* alpha = nullptr;   // device scalar
    void* D = nullptr;
};

// Offset of the scale of (row, k16) in the Sm1xx layout of a matrix with `k16` scale columns (multiple of 4).
__host__ __device__ inline std::size_t sf_offset(int row, int block, int k16) {
    const std::size_t tile = std::size_t(row >> 7) * (k16 >> 2) + (block >> 2);
    return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (block & 3);
}
inline std::size_t sf_bytes(int rows, int k16) { return std::size_t((rows + 127) / 128 * 128) * k16; }

class Fp4Grouped {
public:
    // Device and pinned staging for up to max_groups groups per call and `ring` calls in flight.
    Fp4Grouped(int max_groups, void* workspace, std::size_t workspace_bytes);
    ~Fp4Grouped();
    Fp4Grouped(const Fp4Grouped&) = delete;
    Fp4Grouped& operator=(const Fp4Grouped&) = delete;
    // D in BF16 (bf16 = true) or FP32. N multiple of 128, K multiple of 128.
    void run(const std::vector<Fp4Group>& groups, int N, int K, bool bf16, cudaStream_t stream);

private:
    static constexpr int kRing = 8;
    int max_groups_;
    void* workspace_;
    std::size_t workspace_bytes_;
    std::size_t blob_bytes_ = 0;
    uint8_t* host_ = nullptr;     // pinned [kRing][blob]
    uint8_t* device_ = nullptr;   // [kRing][blob]
    cudaEvent_t done_[kRing] = {};
    int next_ = 0;
    int sms_ = 0;
};

}  // namespace nmh
