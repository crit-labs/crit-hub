#pragma once
#include "common.cuh"

namespace fnl {
// For each of `rows` logits rows (stride ld): tlp[r] = log p(targets[r]), qlp[r][j] = log p(queries[r][j]),
// amax[r] = argmax (lowest id on ties); any output pointer may be null.
void row_scores(const float* logits, int ld, int valid, int rows, const int* targets, const int* queries, int per,
                float* tlp, float* qlp, int* amax, cudaStream_t stream);
}  // namespace fnl
