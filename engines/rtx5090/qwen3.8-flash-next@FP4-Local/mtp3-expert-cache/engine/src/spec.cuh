// Speculative decoding with the checkpoint's MTP block: acceptance of a verified pass and the drafter's inputs.
#pragma once
#include "common.cuh"
#include "layers.cuh"

namespace fnl {

constexpr int kDrafts = kMaxTokens - 1;   // drafts verified per pass

// Result of one speculative round (device, copied to pinned host at the end of the round).
struct Accept {
    int n;                  // tokens of the verified pass kept (input token + accepted drafts)
    int bonus;              // the model's token after the kept ones
    int drafts[kDrafts];    // drafts for the next pass
    int pad[3];
};

__host__ __device__ inline Step make_step(int pos, int token) {
    Step st{};
    st.pos = pos;
    st.token = token;
    st.nb = (pos + 1) / kBlockTokens;
    st.sparse = st.nb > kTopBlocks;
    st.nsel = st.sparse ? kTopBlocks : st.nb;
    st.tail_start = st.nb * kBlockTokens;
    st.nkeys = st.nsel * kBlockTokens + (pos + 1 - st.tail_start);
    return st;
}

// Inputs of the drafter's extend pass over the kept positions (rows beyond the kept ones repeat the last).
struct DraftInputs {
    int* ids;               // [kMaxTokens] token after each kept position
    int* rows;              // [kMaxTokens] row of the target pass's hidden streams
    Step* steps;            // [kMaxTokens]
};
// Verified pass of `tokens` inputs ids (input token, then drafts) at positions pos0..: compares the drafts with
// the model's tokens next[], fills acc (n, bonus) and the drafter's extend inputs.
void accept_pass(const int* ids, const int* next, int tokens, const Step* pass_steps, Accept* acc,
                 const DraftInputs& d, cudaStream_t stream);
// Start of decoding after the prompt: n = 1, bonus = next[0] at position pos (the prompt's last).
void accept_prompt(const int* next, const Step* pos_step, Accept* acc, const DraftInputs& d, cudaStream_t stream);
// Drafter step k (0-based) after the extend pass: position = pass_steps[0].pos + n + k, token = drafts[k].
void chain_step(const Accept* acc, const Step* pass_steps, int k, DraftInputs d, cudaStream_t stream);

// Progress of a round in mapped pinned memory: stage 1 once n and bonus are known, 2 + k once draft k is.
struct SpecProgress {
    volatile int stage;
    int n, bonus;
    int drafts[kDrafts];
};
void progress_note(const Accept* acc, SpecProgress* p, int stage, cudaStream_t stream);

// MXFP8 re-encoding of a BF16 matrix [rows][cols] (cols % 32 == 0): E4M3 values with a UE8M0 scale per 32.
void quantize_mx8(const __nv_bfloat16* w, int rows, int cols, uint8_t* q, uint8_t* s, cudaStream_t stream);

}  // namespace fnl
