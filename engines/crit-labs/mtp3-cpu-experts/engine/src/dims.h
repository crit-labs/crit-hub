// nmh: model dimensions and the expert record layout (host and device code).
#pragma once
#include <cstddef>

namespace nmh {

// config.json llm_config (checked at load).
constexpr int kHidden = 4096, kVocab = 131072, kLayers = 88;
constexpr int kMambaLayers = 40, kMoeLayers = 40, kAttnLayers = 8;
// Mamba-2 mixer
constexpr int kMHeads = 128, kMHeadDim = 64, kMGroups = 8, kMState = 128, kConv = 4;
constexpr int kMInner = kMHeads * kMHeadDim;                    // 8192
constexpr int kConvDim = kMInner + 2 * kMGroups * kMState;     // 10240
constexpr int kInProj = kMInner + kConvDim + kMHeads;          // 18560 (z | xBC | dt)
constexpr int kNormGroup = kMInner / kMGroups;                  // 1024
constexpr float kDtMin = 0.001f;                                 // transformers' dt limit (time_step_min, inf)
// attention (no rotary embedding)
constexpr int kQHeads = 32, kKvHeads = 2, kHeadDim = 128, kGroup = kQHeads / kKvHeads;
constexpr int kQDim = kQHeads * kHeadDim, kKvDim = kKvHeads * kHeadDim;  // 4096, 256
constexpr int kQkv = kQDim + 2 * kKvDim;                         // 4608
// latent MoE
constexpr int kExperts = 512, kTopK = 22, kLatent = 1024, kInter = 2688, kShared = 5376;
constexpr float kRouteScale = 5.0f;
constexpr float kEps = 1e-5f;

// One routed expert, identical layout in pinned host memory and in a device slot:
// up W [kInter][kLatent/2], down W [kLatent][kInter/2] (E2M1, low nibble first), up scales [kInter][kLatent/16],
// down scales [kLatent][kInter/16] (E4M3). Global scales live in a table.
constexpr std::size_t kUpBytes = std::size_t(kInter) * kLatent / 2;        // 1376256
constexpr std::size_t kDownBytes = std::size_t(kLatent) * kInter / 2;      // 1376256
constexpr std::size_t kUpScales = std::size_t(kInter) * (kLatent / 16);    // 172032
constexpr std::size_t kDownScales = std::size_t(kLatent) * (kInter / 16);  // 172032
constexpr std::size_t kOffDown = kUpBytes;
constexpr std::size_t kOffUpS = kUpBytes + kDownBytes;
constexpr std::size_t kOffDownS = kOffUpS + kUpScales;
constexpr std::size_t kSlotBytes = kOffDownS + kDownScales;                 // 3096576

}  // namespace nmh
