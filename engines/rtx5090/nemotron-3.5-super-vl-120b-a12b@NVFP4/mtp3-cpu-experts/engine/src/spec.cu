#include "spec.cuh"

namespace nmh {
namespace {

__global__ void accept_pass_kernel(const int* next, const Step* steps, int drafts, const int* forced,
                                   const int* forced_len, Accept* acc, DraftCtl* d) {
    if(threadIdx.x != 0) return;
    const int e = acc->emitted;
    const int flen = forced_len ? *forced_len : 0;
    if(flen == 0) forced = nullptr;
    int n = 1;
    for(int i = 1; i <= drafts; ++i) {
        const int want = forced ? (e + i - 1 < flen ? forced[e + i - 1] : -1) : next[i - 1];
        if(steps[i].token != want) break;
        ++n;
    }
    int bonus = next[n - 1];
    if(forced && e + n - 1 < flen) bonus = forced[e + n - 1];
    acc->n = n;
    acc->bonus = bonus;
    acc->emitted = e + n;
    const int pos = steps[0].pos;
    for(int i = 0; i <= drafts; ++i) d->rows[i] = Step{pos + i, i < n - 1 ? steps[i + 1].token : bonus};
    d->sel = n - 1;
    d->one = d->rows[n - 1];
}

__global__ void accept_prompt_kernel(const int* next, const Step* pos_step, const int* forced, const int* forced_len, Accept* acc,
                                     DraftCtl* d) {
    if(threadIdx.x != 0) return;
    const int pos = pos_step->pos;
    const int bonus = forced && forced_len && *forced_len > 0 ? forced[0] : next[0];
    acc->n = 1;
    acc->bonus = bonus;
    acc->emitted = 1;
    for(int i = 0; i <= kDrafts; ++i) d->rows[i] = Step{pos, bonus};
    d->sel = 0;
    d->one = d->rows[0];
}

__global__ void chain_step_kernel(const int* draft_token, int k, Accept* acc, DraftCtl* d) {
    if(threadIdx.x != 0) return;
    const int t = *draft_token;
    acc->drafts[k] = t;
    d->one = Step{d->one.pos + 1, t};
}

__global__ void select_row_kernel(const float* src, int ld, const int* row, float* dst, int cols) {
    const float* s = src + std::size_t(*row) * ld;
    for(int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x) dst[c] = s[c];
}
__global__ void select_row_bf16_kernel(const __nv_bfloat16* src, int ld, const int* row, __nv_bfloat16* dst, int cols) {
    const __nv_bfloat16* s = src + std::size_t(*row) * ld;
    for(int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x) dst[c] = s[c];
}

}  // namespace

void accept_pass(const int* next, const Step* steps, int drafts, const int* forced, const int* forced_len, Accept* acc,
                 DraftCtl* d, cudaStream_t stream) {
    accept_pass_kernel<<<1, 32, 0, stream>>>(next, steps, drafts, forced, forced_len, acc, d);
    NMH_CUDA(cudaGetLastError());
}
void accept_prompt(const int* next, const Step* pos_step, const int* forced, const int* forced_len, Accept* acc, DraftCtl* d,
                   cudaStream_t stream) {
    accept_prompt_kernel<<<1, 32, 0, stream>>>(next, pos_step, forced, forced_len, acc, d);
    NMH_CUDA(cudaGetLastError());
}
void chain_step(const int* draft_token, int k, Accept* acc, DraftCtl* d, cudaStream_t stream) {
    chain_step_kernel<<<1, 32, 0, stream>>>(draft_token, k, acc, d);
    NMH_CUDA(cudaGetLastError());
}
void select_row(const float* src, int ld, const int* row, float* dst, int cols, cudaStream_t stream) {
    select_row_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(src, ld, row, dst, cols);
    NMH_CUDA(cudaGetLastError());
}
void select_row_bf16(const __nv_bfloat16* src, int ld, const int* row, __nv_bfloat16* dst, int cols,
                     cudaStream_t stream) {
    select_row_bf16_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(src, ld, row, dst, cols);
    NMH_CUDA(cudaGetLastError());
}

}  // namespace nmh
