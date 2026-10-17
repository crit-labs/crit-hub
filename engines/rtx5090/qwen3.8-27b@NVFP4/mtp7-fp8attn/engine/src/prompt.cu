// Prompt pass (chunked), first token and scoring paths.
#include "gemv.cuh"
#include "attn_q8.cuh"
#include "net.h"

namespace kestrel {

void launch_sample_first(const float* logits, const float* params, int pos, int* ids, float* probs, int* cnt, int* out,
                         cudaStream_t s);
void launch_prefill_done(int* ctl, const int* pred, int P, int* mtp_tok, cudaStream_t s);

namespace kprompt {
__global__ void copy_row_kernel(const __half* __restrict__ src, __half* __restrict__ dst) {
    pdl_wait();
    pdl_trigger();
    for(int i = threadIdx.x; i < kHidden / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
}
__global__ void set_step_kernel(int* ctl, int* tok, const int* ids, int i, int ncols, int pending) {
    pdl_wait();
    pdl_trigger();
    const int t = threadIdx.x;
    if(t < ncols) tok[t] = ids[i + t];
    if(t == 0) {
        ctl[ctl::pos] = i;
        ctl[ctl::pending] = pending;
        ctl[ctl::pass] += 1;
    }
}
__global__ void set_int_kernel(int* p, int v) {
    pdl_wait();
    pdl_trigger(); *p = v; }
}  // namespace kprompt
using namespace kprompt;

// One prompt chunk already in p_tok_ (n tokens at pos0, plus the following token when has_next). Leaves the chunk's
// final normed hidden rows in p_final_ and writes the drafter's keys / values for rows whose next token is known.
void Net::prompt_chunk(int n, int pos0, bool has_next) {
    cudaStream_t s = stream_;
    const int tp = (n + 127) / 128 * 128;
    embed(p_tok_, n, embed_, p_hidden_, s);
    int ai = 0, gi = 0;
    Proj mixer_out{}, mlp_out{};
    bool has_mlp = false;
    for(int L = 0; L < kLayers; ++L) {
        const auto& lw = layers_[L];
        if(lw.attn) {
            const auto& kc = kv_[ai++];
            if(has_mlp) residual_norm(mlp_out, n, p_hidden_, lw.in_norm, p_xq_, p_xs_, nullptr, nullptr, s);
            else rmsnorm_quant_fp8(p_hidden_, lw.in_norm, n, kHidden, p_xq_, p_xs_, nullptr, s);
            lt_->fp8(lw.qkv.w, lw.qkv.rows, kHidden, p_xq_, tp, p_raw_, lw.qkv.rows, s);
            Proj qkv{nullptr, p_raw_, lw.qkv.scale, p_xs_, lw.qkv.rows};
            attn_prep(qkv, n, Pos{nullptr, pos0}, lw.q_norm, lw.k_norm, rope_inv_, p_q_, kc, s);
            crit::fp8_kv::Prefill pp;
            pp.q = p_q_;
            pp.q_stride = kAttnOut;
            pp.cache = kc;
            pp.out = p_attn_;
            pp.out_stride = kAttnOut;
            pp.T = n;
            pp.pos0 = pos0;
            pp.heads = kHeads;
            pp.scale = 1.f / 16.f;
            if(cfg_.attn_q8) prefill_attention_q8(pp, s, cfg_.attn_pv8);
            else KCHECK(crit::fp8_kv::prefill(pp, s));
            attn_gate_quant(p_attn_, qkv, n, p_aq_, p_as_, s);
            lt_->fp8(lw.o.w, kHidden, kAttnOut, p_aq_, tp, p_raw2_, kHidden, s);
            mixer_out = Proj{nullptr, p_raw2_, lw.o.scale, p_as_, kHidden};
        } else {
            auto& st = gdn_[gi++];
            if(has_mlp) residual_norm(mlp_out, n, p_hidden_, lw.in_norm, p_xq_, p_xs_, p_xbf_, nullptr, s);
            else rmsnorm_quant_fp8(p_hidden_, lw.in_norm, n, kHidden, p_xq_, p_xs_, p_xbf_, s);
            lt_->fp8(lw.qkvz.w, lw.qkvz.rows, kHidden, p_xq_, tp, p_raw_, lw.qkvz.rows, s);
            Proj qkvz{nullptr, p_raw_, lw.qkvz.scale, p_xs_, lw.qkvz.rows};
            lt_->bf16(lw.gdn.in_a, 2 * kLinVHeads, kHidden, p_xbf_, tp, p_ab_, 2 * kLinVHeads, s);
            gdn_prefill(lw.gdn, st, psc_, qkvz, p_ab_, n, p_aq_, p_as_, cfg_.wy_scan ? &wy_ : nullptr, s);
            lt_->fp8(lw.out.w, kHidden, kLinZ, p_aq_, tp, p_raw2_, kHidden, s);
            mixer_out = Proj{nullptr, p_raw2_, lw.out.scale, p_as_, kHidden};
        }
        if(lw.fp4 && cfg_.static_fp4) {
            residual_norm_fp4(mixer_out, n, tp, p_hidden_, lw.post_norm, lw.gu4.igs, p_x4_, p_x4sf_, s);
            lt_->fp4(lw.gu4, p_x4_, p_x4sf_, tp, lw.gu4.alpha_static, p_raw_, lw.gu4.rows, s);
            swiglu_fp4(Proj{nullptr, p_raw_, nullptr, nullptr, lw.gu4.rows}, n, tp, lw.down4.igs, p_hq_, p_h4sf_, s);
            lt_->fp4(lw.down4, p_hq_, p_h4sf_, tp, lw.down4.alpha_static, p_raw2_, kHidden, s);
            mlp_out = Proj{nullptr, p_raw2_, nullptr, nullptr, kHidden};
            has_mlp = true;
        } else if(lw.fp4) {
            KCHECK(cudaMemsetAsync(p_amax_, 0, 16, s));
            residual_norm(mixer_out, n, p_hidden_, lw.post_norm, nullptr, nullptr, p_xbf_, p_amax_, s);
            quant_fp4(p_xbf_, n, tp, kHidden, p_amax_, lw.gu4.alpha, p_x4_, p_x4sf_, p_alpha_, s);
            lt_->fp4(lw.gu4, p_x4_, p_x4sf_, tp, p_alpha_, p_raw_, lw.gu4.rows, s);
            swiglu_bf16(Proj{nullptr, p_raw_, nullptr, nullptr, lw.gu4.rows}, n, p_h_, p_amax_ + 1, s);
            quant_fp4(p_h_, n, tp, kInter, p_amax_ + 1, lw.down4.alpha, p_hq_, p_h4sf_, p_alpha_ + 1, s);
            lt_->fp4(lw.down4, p_hq_, p_h4sf_, tp, p_alpha_ + 1, p_raw2_, kHidden, s);
            mlp_out = Proj{nullptr, p_raw2_, nullptr, nullptr, kHidden};
            has_mlp = true;
        } else {
            residual_norm(mixer_out, n, p_hidden_, lw.post_norm, p_xq_, p_xs_, nullptr, nullptr, s);
            lt_->fp8(lw.gu8.w, lw.gu8.rows, kHidden, p_xq_, tp, p_raw_, lw.gu8.rows, s);
            KCHECK(cudaMemsetAsync(p_amax_, 0, 16, s));
            swiglu_bf16(Proj{nullptr, p_raw_, lw.gu8.scale, p_xs_, lw.gu8.rows}, n, p_h_, p_amax_, s);
            quant_fp8_rows(p_h_, n, kInter, p_hq_, p_hs_, s);
            lt_->fp8(lw.down8.w, kHidden, kInter, p_hq_, tp, p_raw2_, kHidden, s);
            mlp_out = Proj{nullptr, p_raw2_, lw.down8.scale, p_hs_, kHidden};
            has_mlp = true;
        }
    }
    if(has_mlp) residual_add(mlp_out, n, p_hidden_, s);
    rmsnorm_f16(p_hidden_, kHidden, final_norm_, n, kHidden, p_final_, kHidden, s);
    // Drafter prompt pass: only its keys and values matter for later positions (one layer).
    const int mrows = has_next ? n : n - 1;
    if(cfg_.draft > 0 && mrows > 0) {
        __half* in = reinterpret_cast<__half*>(p_xbf_);  // [rows][10240]
        embed(p_tok_ + 1, mrows, embed_, p_hidden_, s);
        rmsnorm_f16(p_hidden_, kHidden, mtp_.pre_emb, mrows, kHidden, in, 2 * kHidden, s);
        rmsnorm_f16_from_f16(p_final_, kHidden, mtp_.pre_hidden, mrows, kHidden, in + kHidden, 2 * kHidden, s);
        quant_fp8_rows(in, mrows, 2 * kHidden, p_xq_, p_hs_, s);
        lt_->fp8(mtp_.fc8.w, kHidden, 2 * kHidden, p_xq_, tp, p_raw2_, kHidden, s);
        KCHECK(cudaMemsetAsync(p_hidden_, 0, static_cast<std::size_t>(mrows) * kHidden * 4, s));
        residual_add(Proj{nullptr, p_raw2_, mtp_.fc8.scale, p_hs_, kHidden}, mrows, p_hidden_, s);
        rmsnorm_quant_fp8(p_hidden_, mtp_.in_norm, mrows, kHidden, p_xq_, p_xs_, nullptr, s);
        lt_->fp8(mtp_.kv8.w, 2 * kKvRows, kHidden, p_xq_, tp, p_raw_, 2 * kKvRows, s);
        // The key/value projection holds columns kQRows.. of the drafter's qkv: shift both pointers so
        // at(r, kQRows + c) reads column c.
        Proj kv{nullptr, p_raw_ - kQRows, mtp_.kv8.scale - kQRows, p_xs_, 2 * kKvRows};
        attn_prep(kv, mrows, Pos{nullptr, pos0}, mtp_.q_norm, mtp_.k_norm, rope_inv_, nullptr, mtp_kv_, s, true);
    }
}

void Net::prefill(const int* tokens, int T, int p0, bool last) {
    cudaStream_t s = stream_;
    const int C = cfg_.prefill_chunk;
    for(int c0 = 0; c0 < T; c0 += C) {
        const int n = std::min(C, T - c0);
        const bool final_chunk = c0 + n == T;
        // The token after the chunk is known unless this is the prompt's end (then tokens[T] is not read).
        const bool has_next = !final_chunk || !last;
        KCHECK(cudaMemcpyAsync(p_tok_, tokens + c0, static_cast<std::size_t>(n + (has_next ? 1 : 0)) * 4,
                               cudaMemcpyHostToDevice, s));
        prompt_chunk(n, p0 + c0, has_next);
        if(final_chunk && last) {
            launch_k(copy_row_kernel, 1, 256, 0, s, p_final_ + static_cast<std::size_t>(n - 1) * kHidden, d_final_);
            KLAUNCH();
            gemv_fp8(head_, d_final_, kHidden, 1, {GemvEpi::store, d_logits_, kVocab}, 1, nullptr, s);
            if(sampling_) {
                launch_sample_first(d_logits_, d_sampling_, p0 + T, d_cand_ids_, d_cand_p_, d_cand_n_, d_pred_, s);
            } else {
                argmax(d_logits_, kVocab, 1, d_pred_, s);
            }
            launch_prefill_done(ctl_, d_pred_, p0 + T, d_mtp_tok_, s);
        }
    }
}

void Net::force_first(int token) {
    launch_k(set_int_kernel, 1, 1, 0, stream_, ctl_ + ctl::last, token);
    launch_k(set_int_kernel, 1, 1, 0, stream_, d_mtp_tok_, token);
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
            KCHECK(cudaMemcpyAsync(p_tok_, ids + c0, static_cast<std::size_t>(m + 1) * 4, cudaMemcpyHostToDevice, s));
            prompt_chunk(m, c0, true);
            for(int r = 0; r < m; r += kMaxCols) {
                const int k = std::min(kMaxCols, m - r);
                if(c0 + r + k <= first - 1) continue;
                head_rows(p_final_ + static_cast<std::size_t>(r) * kHidden, k, d_logits_, s);
                take(d_logits_, c0 + r, k);
            }
        }
    } else {
        const int W = path == 1 ? 1 : std::max(cfg_.draft + 1, 2);
        int prev = 0;
        for(int i = 0; i < n - 1; i += W) {
            const int w = std::min(W, n - 1 - i);
            launch_k(set_step_kernel, 1, 32, 0, s, ctl_, d_tok_, d_ids, i, w, prev);
            KLAUNCH();
            target_pass(w, s);
            take(d_logits_, i, w);
            prev = w;
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

}  // namespace kestrel
