// Prompt attention over the FP8 KV cache with FP8 Q K^T (see attn_q8.cu). Same problem description as
// crit::fp8_kv::prefill; head dim 256.
#pragma once
#include "fp8_kv_attention.h"

namespace kestrel {
// pv8: P V on FP8 tensor cores too (probabilities in E4M3, value codes undequantized).
void prefill_attention_q8(const crit::fp8_kv::Prefill& p, cudaStream_t s, bool pv8 = false);
// Decode / verify attention with FP8 Q K^T, same partial layout and combine as crit::fp8_kv::decode.
void decode_attention_q8(const crit::fp8_kv::Decode& p, cudaStream_t s);
}
