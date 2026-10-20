#include "spec.cuh"

namespace fnl {

namespace kern {

__device__ void fill_draft_inputs(const Accept& a, int pos0, const int* ids, DraftInputs d) {
    // Row r covers kept position pos0 + min(r, n - 1): its next token is the next input (or the bonus).
    for(int r = 0; r < kMaxTokens; ++r) {
        const int rr = r < a.n ? r : a.n - 1;
        const int tok = rr + 1 < a.n ? ids[rr + 1] : a.bonus;
        d.ids[r] = tok;
        d.rows[r] = rr;
        d.steps[r] = make_step(pos0 + rr, tok);
    }
}

__global__ void accept_kernel(const int* ids, const int* next, int tokens, const Step* steps, Accept* acc,
                              DraftInputs d) {
    if(threadIdx.x != 0) return;
    int a = 0;
    while(a + 1 < tokens && ids[a + 1] == next[a]) ++a;
    Accept r{};
    r.n = a + 1;
    r.bonus = next[a];
    *acc = r;
    fill_draft_inputs(r, steps[0].pos, ids, d);
}

__global__ void accept_prompt_kernel(const int* next, const Step* step, Accept* acc, DraftInputs d) {
    if(threadIdx.x != 0) return;
    Accept r{};
    r.n = 1;
    r.bonus = next[0];
    *acc = r;
    fill_draft_inputs(r, step->pos, nullptr, d);
}

__global__ void chain_kernel(const Accept* acc, const Step* steps, int k, DraftInputs d) {
    if(threadIdx.x != 0) return;
    const int tok = acc->drafts[k];
    d.ids[0] = tok;
    d.steps[0] = make_step(steps[0].pos + acc->n + k, tok);
}

__global__ void progress_kernel(const Accept* acc, SpecProgress* p, int stage) {
    if(threadIdx.x != 0) return;
    if(stage == 1) {
        p->n = acc->n;
        p->bonus = acc->bonus;
    } else {
        p->drafts[stage - 2] = acc->drafts[stage - 2];
    }
    __threadfence_system();
    p->stage = stage;
    __threadfence_system();
}

// One warp per 32-value block.
__global__ void quantize_mx8_kernel(const __nv_bfloat16* w, std::size_t blocks, uint8_t* q, uint8_t* s) {
    const std::size_t b = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if(b >= blocks) return;
    const float x = __bfloat162float(w[b * 32 + lane]);
    const float amax = warp_max(fabsf(x));
    // Smallest power of two with amax / 2^e <= 448 (E4M3 max).
    int e = -127;
    if(amax > 0.f) {
        e = int(ceilf(log2f(amax / 448.f)));
        if(ldexpf(448.f, e - 1) >= amax) --e;   // guard against log2 rounding
        e = max(-127, min(127, e));
    }
    const float v = ldexpf(x, -e);
    const __nv_fp8_e4m3 f(v);
    q[b * 32 + lane] = f.__x;
    if(lane == 0) s[b] = uint8_t(e + 127);
}

}  // namespace kern
using namespace kern;

void accept_pass(const int* ids, const int* next, int tokens, const Step* pass_steps, Accept* acc,
                 const DraftInputs& d, cudaStream_t stream) {
    accept_kernel<<<1, 32, 0, stream>>>(ids, next, tokens, pass_steps, acc, d);
    FNL_CUDA(cudaGetLastError());
}
void accept_prompt(const int* next, const Step* pos_step, Accept* acc, const DraftInputs& d, cudaStream_t stream) {
    accept_prompt_kernel<<<1, 32, 0, stream>>>(next, pos_step, acc, d);
    FNL_CUDA(cudaGetLastError());
}
void chain_step(const Accept* acc, const Step* pass_steps, int k, DraftInputs d, cudaStream_t stream) {
    chain_kernel<<<1, 32, 0, stream>>>(acc, pass_steps, k, d);
    FNL_CUDA(cudaGetLastError());
}
void progress_note(const Accept* acc, SpecProgress* p, int stage, cudaStream_t stream) {
    progress_kernel<<<1, 32, 0, stream>>>(acc, p, stage);
    FNL_CUDA(cudaGetLastError());
}
void quantize_mx8(const __nv_bfloat16* w, int rows, int cols, uint8_t* q, uint8_t* s, cudaStream_t stream) {
    const std::size_t blocks = std::size_t(rows) * cols / 32;
    quantize_mx8_kernel<<<unsigned((blocks * 32 + 255) / 256), 256, 0, stream>>>(w, blocks, q, s);
    FNL_CUDA(cudaGetLastError());
}

}  // namespace fnl
