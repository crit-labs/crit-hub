// Chunked (WY) gated-delta scan for prompts; see gdn_wy.cu. gb[...][0] holds the log decay g (not exp(g)).
#pragma once
#include <algorithm>

#include "ops.cuh"
#include "prefill_ops.cuh"

namespace kestrel {

constexpr int kWyChunk = 64;
struct WyScratch {
    float* W = nullptr;    // [chunks][48][64][128]
    float* Ub = nullptr;   // [chunks][48][64][128]
    float* Qk = nullptr;   // [chunks][48][64][64]
    float* gam = nullptr;  // [chunks][48][64]
    int max_chunks = 0;
};
std::size_t wy_scratch_floats(int max_chunks);
WyScratch wy_scratch(float* base, int max_chunks);
// Scan T tokens from the committed state (sc.qk, sc.v, sc.gb as the prompt conv wrote them), writing sc.o and the state.
void gdn_scan_wy(GdnState st, GdnPrefillScratch sc, int T, WyScratch ws, cudaStream_t s);
// The same on BF16 tensor cores (FP32 accumulation and state); W, Ub, Qk stored as BF16 in the scratch.
void gdn_scan_wy_tc(GdnState st, GdnPrefillScratch sc, int T, WyScratch ws, cudaStream_t s);

}  // namespace kestrel
