// Prompt pass (chunked), first token and scoring paths.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemv.cuh"
#include "net.h"

namespace g4 {

void launch_sample_first(const float* logits, const float* params, int pos, int* ids, float* probs, int* cnt, int* out,
                         cudaStream_t s);
void launch_prefill_done(int* ctl, const int* pred, int P, cudaStream_t s);

namespace kprompt {
// G4_PROFILE=1: per-chunk GPU time of the prompt pass by part (development only; adds synchronization).
struct Prof {
    bool on = std::getenv("G4_PROFILE") != nullptr;
    std::vector<std::pair<int, cudaEvent_t>> marks;
    double total[4] = {0, 0, 0, 0};  // full attention, sliding attention, GEMMs and the rest, chunk
    int chunks = 0;
    void mark(int part, cudaStream_t s) {
        if(!on) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        marks.emplace_back(part, e);
    }
    void flush() {
        if(!on || marks.empty()) return;
        cudaEventSynchronize(marks.back().second);
        for(std::size_t i = 1; i < marks.size(); ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, marks[i - 1].second, marks[i].second);
            total[marks[i].first] += ms;
        }
        float all = 0;
        cudaEventElapsedTime(&all, marks.front().second, marks.back().second);
        total[3] += all;
        for(auto& m : marks) cudaEventDestroy(m.second);
        marks.clear();
        if(++chunks % 8 == 0)
            std::fprintf(stderr, "g4 prof: %d chunks: full attn %.0f ms, sliding attn %.0f ms, other %.0f ms, total %.0f ms\n", chunks,
                         total[0], total[1], total[2], total[3]);
    }
};
Prof g_prof;
__global__ void copy_row_kernel(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst) {
    pdl_wait();
    pdl_trigger();
    for(int i = threadIdx.x; i < kHidden / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
}
__global__ void set_step_kernel(int* ctl, int* tok, const int* ids, int i, int ncols) {
    pdl_wait();
    pdl_trigger();
    const int t = threadIdx.x;
    if(t < ncols) tok[t] = ids[i + t];
    if(t == 0) ctl[ctl::pos] = i;
}
__global__ void set_int_kernel(int* p, int v) {
    pdl_wait();
    pdl_trigger();
    *p = v;
}
}  // namespace kprompt
using namespace kprompt;

// One prompt chunk already in p_tok_ (n tokens at pos0). Leaves the chunk's final normed hidden rows in p_final_.
void Net::prompt_chunk(int n, int pos0) {
    cudaStream_t s = stream_;
    const int tp = (n + 127) / 128 * 128;
    g_prof.mark(2, s);
    embed(p_tok_, n, embed_, kHidden, embed_scale_, p_hidden_, kHidden, s);
    rmsnorm_quant_fp8(p_hidden_, layers_[0].in_norm, n, p_xq_, p_xs_, s);
    for(int L = 0; L < kLayers; ++L) {
        const auto& lw = layers_[L];
        const int qrows = lw.global ? kGQ : kSQ;
        lt_->fp8(lw.qkv.w, lw.qkv.rows, kHidden, p_xq_, tp, p_raw_, lw.qkv.rows, s);
        const Proj qkv{nullptr, p_raw_, lw.qkv.scale, p_xs_, lw.qkv.rows};
        AttnRows ar;
        ar.q = p_q_; ar.q_stride = qrows; ar.out = p_attn_; ar.out_stride = qrows; ar.T = n; ar.pos0 = pos0;
        if(lw.global) {
            global_prep(qkv, n, Pos{nullptr, pos0}, lw.q_norm, inv_g_, p_q_, qrows, glob_[L], 0, s);
            g_prof.mark(2, s);
            if(cfg_.prefill_q8 && cfg_.prefill_qb) global_prefill_qb(ar, glob_[L], s);
            else if(cfg_.prefill_q8) global_prefill_q8(ar, glob_[L], s);
            else global_prefill(ar, glob_[L], s);
            g_prof.mark(0, s);
        } else {
            slide_prep(qkv, n, Pos{nullptr, pos0}, lw.q_norm, lw.k_norm, inv_s_, p_q_, qrows, slide_[L], 0, s);
            g_prof.mark(2, s);
            slide_prefill(ar, slide_[L], s);
            g_prof.mark(1, s);
        }
        quant_fp8_rows(p_attn_, n, qrows, p_aq_, p_as_, s);
        lt_->fp8(lw.o.w, kHidden, qrows, p_aq_, tp, p_raw2_, kHidden, s);
        RowOut mlp_in;
        mlp_in.packed = p_x4_;
        mlp_in.sf = p_x4sf_;
        mlp_in.gs = lw.gu.igs;
        mlp_in.rscale = cfg_.row_fp4 ? p_rs1_ : nullptr;
        mlp_in.T_pad = tp;
        sandwich_rows(Proj{nullptr, p_raw2_, lw.o.scale, p_as_, kHidden}, n, p_hidden_, lw.post_attn, 1.f, lw.pre_ff, mlp_in, s);
        const bool row = cfg_.row_fp4;
        lt_->fp4(lw.gu, p_x4_, p_x4sf_, tp, row ? lw.gu.alpha_dev : lw.gu.alpha_static, p_raw_, lw.gu.rows, s);
        geglu_fp4(Proj{nullptr, p_raw_, nullptr, row ? p_rs1_ : nullptr, lw.gu.rows}, n, tp, lw.down.igs, row ? p_rs2_ : nullptr,
                  p_hq_, p_h4sf_, s);
        lt_->fp4(lw.down, p_hq_, p_h4sf_, tp, row ? lw.down.alpha_dev : lw.down.alpha_static, p_raw2_, kHidden, s);
        RowOut next;
        if(L + 1 < kLayers) {
            next.codes = p_xq_;
            next.scale = p_xs_;
        } else {
            next.bf = p_final_;
        }
        sandwich_rows(Proj{nullptr, p_raw2_, nullptr, row ? p_rs2_ : nullptr, kHidden}, n, p_hidden_, lw.post_ff, lw.scalar_host,
                      L + 1 < kLayers ? layers_[L + 1].in_norm : final_norm_, next, s);
    }
    g_prof.mark(2, s);
    g_prof.flush();
}

void Net::prefill(const int* tokens, int T, int p0, bool last) {
    cudaStream_t s = stream_;
    const int C = cfg_.prefill_chunk;
    for(int c0 = 0; c0 < T; c0 += C) {
        const int n = std::min(C, T - c0);
        KCHECK(cudaMemcpyAsync(p_tok_, tokens + c0, static_cast<std::size_t>(n) * 4, cudaMemcpyHostToDevice, s));
        prompt_chunk(n, p0 + c0);
        if(c0 + n == T && last) {
            launch_k(copy_row_kernel, 1, 256, 0, s, p_final_ + static_cast<std::size_t>(n - 1) * kHidden, d_final_);
            KLAUNCH();
            head_gemv(d_final_, 1, d_logits_, s);
            if(sampling_) launch_sample_first(d_logits_, d_sampling_, p0 + T, d_cand_ids_, d_cand_p_, d_cand_n_, d_pred_, s);
            else argmax(d_logits_, kVocab, kVocabValid, 1, d_pred_, s);
            launch_prefill_done(ctl_, d_pred_, p0 + T, s);
        }
    }
}

void Net::force_first(int token) {
    launch_k(set_int_kernel, 1, 1, 0, stream_, ctl_ + ctl::last, token);
    KLAUNCH();
}

int Net::first_token() {
    int t = 0;
    KCHECK(cudaMemcpyAsync(&t, ctl_ + ctl::last, 4, cudaMemcpyDeviceToHost, stream_));
    KCHECK(cudaStreamSynchronize(stream_));
    return t;
}

void Net::score(const int* ids, int n, int first, int path, const int* queries, int nq, float* target_lp, float* query_lp,
                int* argmax_out) {
    cudaStream_t s = stream_;
    const int targets = n - first;
    if(first < 1 || targets <= 0) throw std::invalid_argument("score: bad first target");
    int *d_ids, *d_q, *d_am;
    float *d_tl, *d_ql;
    const int qn = std::max(nq, 1);
    KCHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ids), static_cast<std::size_t>(n + 8) * 4, s));
    KCHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_q), static_cast<std::size_t>(targets) * qn * 4, s));
    KCHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_am), static_cast<std::size_t>(targets) * 4, s));
    KCHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_tl), static_cast<std::size_t>(targets) * 4, s));
    KCHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ql), static_cast<std::size_t>(targets) * qn * 4, s));
    KCHECK(cudaMemcpyAsync(d_ids, ids, static_cast<std::size_t>(n) * 4, cudaMemcpyHostToDevice, s));
    if(nq) KCHECK(cudaMemcpyAsync(d_q, queries, static_cast<std::size_t>(targets) * nq * 4, cudaMemcpyHostToDevice, s));
    reset();
    // Logit rows at positions [a, a + rows) predict ids[a + 1 ..]; score the rows in [first - 1, n - 2].
    auto take = [&](const float* logits, int a, int rows) {
        const int lo = std::max(a, first - 1), hi = std::min(a + rows, n - 1);
        if(lo >= hi) return;
        const int t0 = lo - (first - 1);
        logprob_rows(logits + static_cast<std::size_t>(lo - a) * kVocab, kVocab, hi - lo, d_ids + lo + 1,
                     d_q + static_cast<std::size_t>(t0) * qn, nq, d_tl + t0, d_ql + static_cast<std::size_t>(t0) * qn,
                     d_am + t0, s);
    };
    if(path == 0) {
        const int C = cfg_.prefill_chunk;
        for(int c0 = 0; c0 < n - 1; c0 += C) {
            const int m = std::min(C, n - 1 - c0);
            KCHECK(cudaMemcpyAsync(p_tok_, ids + c0, static_cast<std::size_t>(m) * 4, cudaMemcpyHostToDevice, s));
            prompt_chunk(m, c0);
            for(int r = 0; r < m; r += kMaxCols) {
                const int k = std::min(kMaxCols, m - r);
                if(c0 + r + k <= first - 1) continue;
                head_rows(p_final_ + static_cast<std::size_t>(r) * kHidden, k, d_logits_, s);
                take(d_logits_, c0 + r, k);
            }
        }
    } else if(path == 3) {
        // prefill_decode: ids[:first] prefilled as a request prompt; the target at first from the prefill's last
        // position, each later one from an ordinary one-token decode step fed the id before it.
        const int C = cfg_.prefill_chunk;
        for(int c0 = 0; c0 < first; c0 += C) {
            const int m = std::min(C, first - c0);
            KCHECK(cudaMemcpyAsync(p_tok_, ids + c0, static_cast<std::size_t>(m) * 4, cudaMemcpyHostToDevice, s));
            prompt_chunk(m, c0);
            if(c0 + m == first) {
                head_rows(p_final_ + static_cast<std::size_t>(m - 1) * kHidden, 1, d_logits_, s);
                take(d_logits_, first - 1, 1);
            }
        }
        for(int i = first; i < n - 1; ++i) {
            launch_k(set_step_kernel, 1, 32, 0, s, ctl_, d_tok_, d_ids, i, 1);
            KLAUNCH();
            target_pass(1, s);
            take(d_logits_, i, 1);
        }
    } else {
        const int W = path == 1 ? 1 : std::max(cfg_.draft + 1, 2);
        for(int i = 0; i < n - 1; i += W) {
            const int w = std::min(W, n - 1 - i);
            launch_k(set_step_kernel, 1, 32, 0, s, ctl_, d_tok_, d_ids, i, w);
            KLAUNCH();
            target_pass(w, s);
            take(d_logits_, i, w);
        }
    }
    KCHECK(cudaMemcpyAsync(target_lp, d_tl, static_cast<std::size_t>(targets) * 4, cudaMemcpyDeviceToHost, s));
    if(nq) KCHECK(cudaMemcpyAsync(query_lp, d_ql, static_cast<std::size_t>(targets) * nq * 4, cudaMemcpyDeviceToHost, s));
    if(argmax_out) KCHECK(cudaMemcpyAsync(argmax_out, d_am, static_cast<std::size_t>(targets) * 4, cudaMemcpyDeviceToHost, s));
    KCHECK(cudaStreamSynchronize(s));
    KCHECK(cudaFreeAsync(d_ids, s));
    KCHECK(cudaFreeAsync(d_q, s));
    KCHECK(cudaFreeAsync(d_am, s));
    KCHECK(cudaFreeAsync(d_tl, s));
    KCHECK(cudaFreeAsync(d_ql, s));
    KCHECK(cudaStreamSynchronize(s));
    reset();
}

}  // namespace g4
