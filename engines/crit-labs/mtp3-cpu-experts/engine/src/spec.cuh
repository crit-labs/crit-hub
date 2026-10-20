// Speculative decoding with the checkpoint's MTP block: acceptance of a verified pass and the drafter's inputs.
#pragma once
#include "common.cuh"
#include "layers.cuh"

namespace nmh {

constexpr int kDrafts = 3;            // most drafts a pass verifies (the pass has drafts + 1 tokens)

// Result of one speculative round, copied to pinned host memory at the end of the round's graph.
struct Accept {
    int n;                  // tokens of the verified pass kept (input token + accepted drafts)
    int bonus;              // the model's (or the replayed continuation's) token after the kept ones
    int drafts[kDrafts];    // drafts for the next pass
    int emitted;            // output tokens emitted so far, this round included (replay index)
    int pad[2];
};

// Device control of the drafter.
struct DraftCtl {
    Step rows[kDrafts + 1];   // extend pass: (position, token after it) per row; rows past the kept ones repeat
    Step one;                 // a single drafter step (selected row or a chained step)
    int sel;                  // row of the extend pass whose output drafts (n - 1)
    int pad[3];
};

// After a verified pass of T = kDrafts + 1 tokens (steps: input token then drafts, at consecutive positions):
// n = 1 + leading drafts equal to the target's argmax next[] (with a replay continuation forced[0 .. forced_len):
// equal to forced[emitted ..] while *forced_len > 0, and the bonus taken from it). Writes acc (n, bonus, emitted) and the drafter's
// extend rows and selected row.
void accept_pass(const int* next, const Step* steps, int drafts, const int* forced, const int* forced_len, Accept* acc,
                 DraftCtl* d, cudaStream_t stream);
// Start of decoding after the prompt (pos_step->pos: the prompt's last position): the first output token is bonus
// (the prompt pass's argmax, or forced[0]); one drafter row at that position with that token; emitted = 1.
void accept_prompt(const int* next, const Step* pos_step, const int* forced, const int* forced_len, Accept* acc, DraftCtl* d,
                   cudaStream_t stream);
// Draft k (0-based) is known in draft_token: store it and prepare chained step k + 1 (position of the selected row
// plus k + 1, token = the draft).
void chain_step(const int* draft_token, int k, Accept* acc, DraftCtl* d, cudaStream_t stream);
// dst[c] = src[*row * ld + c] for c < cols (one row of a multi-row buffer).
void select_row(const float* src, int ld, const int* row, float* dst, int cols, cudaStream_t stream);
void select_row_bf16(const __nv_bfloat16* src, int ld, const int* row, __nv_bfloat16* dst, int cols,
                     cudaStream_t stream);

}  // namespace nmh
