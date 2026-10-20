// Attention over g4's FP8 caches.
//
// Sliding-window layers (16 KV heads of 256, window 1024): a ring of `ring` slots per layer (slot = position % ring),
// K = RoPE(k_norm(k)) and V = v_norm(v) as E4M3 codes with one FP32 scale per (slot, head).
//
// Full-attention layers (4 KV heads of 512, attention_k_eq_v): one tensor per position, u = v_norm(k_proj x), stored
// as E4M3 codes with one scale per (position, head). Values are u; keys are RoPE(u * k_norm.weight). Only the 128
// rotated dims of a key are formed (FP32, BF16 in shared memory); on the other 384 the key is u * w, so the queries
// carry w there (global_prep folds the key head's k_norm.weight into them) and the scores read u. Computed in FP32
// when a key tile is staged (the k_norm and v_norm of the reference share rsqrt(mean(k^2) + eps), so u * w is k_norm(k)).
// Proportional RoPE rotates pairs (d, d + 256) for d < 64 with inv_freq[d]; the other dims have frequency 0.
//
// Every kernel stages tiles as BF16 (code * scale rounded once) and runs S = Q K^T and O = P V on BF16 mma.m16n8k16
// with an FP32 online softmax. Scores are not scaled (Gemma 4 normalizes q and k; scaling = 1). A query row at
// position p sees keys k with k <= p - shift and, in a sliding layer, k > p - shift - window (shift 0 for the target,
// 1 for the drafter, which attends the target's keys before its own position). Decode work is split over key pieces
// by absolute position, so a column's result does not depend on the other columns of a pass.
#pragma once
#include "common.h"

namespace g4 {

struct SlideCache {
    std::uint8_t* k = nullptr;  // [ring][16][256]
    std::uint8_t* v = nullptr;
    float* ks = nullptr;        // [ring][16]
    float* vs = nullptr;
    int ring = 0;
};

struct GlobalCache {
    std::uint8_t* u = nullptr;  // [capacity][4][512]
    float* us = nullptr;        // [capacity][4]
    const float* kw = nullptr;  // k_norm.weight [512]
    const float* inv = nullptr; // 64 inverse frequencies
    const float2* rope = nullptr; // [capacity][64] (cos, sin) of position * inv[i], FP32 as the reference computes them
    std::uint8_t* kr = nullptr;   // [capacity][4][128] rotated key dims (0..63, 256..319) as E4M3 codes, for FP8 Q K^T
    float* krs = nullptr;         // [capacity][4] their scales
};

// Fill a rope table for positions 0 .. positions - 1.
void fill_rope_table(float2* table, const float* inv, int positions, cudaStream_t s);

// Queries (BF16, row t at q + t * q_stride, head h at column h * head_dim); out FP32 [t][h * head_dim].
struct AttnRows {
    const __nv_bfloat16* q = nullptr;
    int q_stride = 0;
    float* out = nullptr;
    int out_stride = 0;
    int T = 0;           // query positions
    int pos0 = 0;        // position of query row 0 (host), or *pos0_dev when set
    const int* pos0_dev = nullptr;
    int shift = 0;       // 0: keys <= position; 1: keys < position (drafter)
};

// Prompt attention: rows t = 0..T-1 at positions pos0 + t, all 32 query heads; keys already written.
void slide_prefill(const AttnRows& a, const SlideCache& c, cudaStream_t s);
void global_prefill(const AttnRows& a, const GlobalCache& c, cudaStream_t s);
// The same with S = Q K^T on FP8 tensor cores (queries quantized per row and 256-dim half; needs c.kr).
void global_prefill_q8(const AttnRows& a, const GlobalCache& c, cudaStream_t s);
// The same with BF16 queries against the stored key codes converted exactly to BF16 in registers (needs c.kr).
void global_prefill_qb(const AttnRows& a, const GlobalCache& c, cudaStream_t s);

// Decode / verify attention, T <= 8 columns (T * group <= 64 rows per key head). partial: decode_partial_floats().
std::size_t slide_partial_floats(int blocks);
std::size_t global_partial_floats(int blocks);
void slide_decode(const AttnRows& a, const SlideCache& c, int blocks, float* partial, cudaStream_t s);
void global_decode(const AttnRows& a, const GlobalCache& c, int blocks, float* partial, cudaStream_t s);

}  // namespace g4
