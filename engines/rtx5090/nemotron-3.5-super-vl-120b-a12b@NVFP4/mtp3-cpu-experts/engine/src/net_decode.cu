// Decode passes, speculative rounds and the MTP drafter (CUDA graphs over device-side state).
#include "net.h"
#include <cstring>

namespace nmh {

namespace {
constexpr float kAttnScale = 0.08838834764831845f;   // 1 / sqrt(128)
}

void Net::moe_decode(const MoeWeights& w, const float* x, float* h, int T) {
    const auto s = stream();
    Gemv r;
    r.fmt = WFormat::bf16;
    r.seg[0] = {w.router, nullptr, 1.f, RL_, kExperts, kExperts};
    r.seg[1] = {w.fc1, nullptr, 1.f, LAT_, kLatent, kLatent};
    r.nseg = 2;
    r.cols = kHidden;
    r.x = x;
    r.tokens = T;
    gemv(r, s);
    route(RL_, w.bias, routes_, T, s);
    Gemv su;
    su.fmt = WFormat::fp4;
    su.out = Out::relu2;
    su.seg[0] = {w.sh_up.w, w.sh_up.s, w.sh_up.scale, SH_, kShared, kShared};
    su.cols = kHidden;
    su.x = x;
    su.tokens = T;
    gemv(su, s);
    Gemv sd;
    sd.fmt = WFormat::fp4;
    sd.out = Out::add;
    sd.seg[0] = {w.sh_down.w, w.sh_down.s, w.sh_down.scale, h, kHidden, kHidden};
    sd.cols = kShared;
    sd.x = SH_;
    sd.tokens = T;
    gemv(sd, s);
    plan_experts(tables_, w.group, routes_, T, plan_, LAT_, s);
    expert_up(plan_, LAT_, ACT_, T, s);
    expert_down(tables_, plan_, ACT_, PART_, LO_, T, s);
    Gemv f2;
    f2.fmt = WFormat::bf16;
    f2.out = Out::add;
    f2.seg[0] = {w.fc2, nullptr, 1.f, h, kHidden, kHidden};
    f2.cols = kLatent;
    f2.x = LO_;
    f2.tokens = T;
    gemv(f2, s);
}

// One target pass of T tokens (steps in step_). With spec, the Mamba layers keep a state snapshot after every token
// and record their conv inputs, so the commit after acceptance can keep any prefix; otherwise they update in place.
void Net::decode_layers(int T, bool spec) {
    const auto s = stream();
    embed_steps(embed_, step_, H_, T, s);
    begin_pass(tables_, s);
    for(int l = 0; l < kLayers; ++l) {
        const LayerWeights& L = layers_[l];
        rmsnorm(H_, L.norm, X_, T, s);
        if(L.kind == Kind::mamba) {
            Gemv g;
            g.fmt = WFormat::fp8;
            g.seg[0] = {L.in_proj.w, nullptr, L.in_proj.scale, P_, kInProj, kInProj};
            g.cols = kHidden;
            g.x = X_;
            g.tokens = T;
            gemv(g, s);
            float* conv = conv_state_ + std::size_t(L.index) * kConvDim * 3;
            float* ring = ssm_ + std::size_t(L.index) * ring_ * kSsmFloats;
            float* rec = spec ? conv_rec_ + std::size_t(L.index) * kMaxTokensRec * kConvDim : nullptr;
            conv_tokens(L.m, P_, kInProj, conv, XC_, rec, T, !spec, s);
            ssm_tokens(L.m, XC_, P_, kInProj, ring, ssm_cur_, spec, Y_, T, s);
            gated_norm(L.m, Y_, P_, kInProj, G_, nullptr, T, s);
            Gemv o;
            o.fmt = WFormat::fp8;
            o.out = Out::add;
            o.seg[0] = {L.out_proj.w, nullptr, L.out_proj.scale, H_, kHidden, kHidden};
            o.cols = kMInner;
            o.x = G_;
            o.tokens = T;
            gemv(o, s);
        } else if(L.kind == Kind::attn) {
            Gemv g;
            g.fmt = WFormat::fp8;
            g.seg[0] = {L.q.w, nullptr, L.q.scale, P_, kQDim, kQkv};
            g.seg[1] = {L.k.w, nullptr, L.k.scale, P_ + kQDim, kKvDim, kQkv};
            g.seg[2] = {L.v.w, nullptr, L.v.scale, P_ + kQDim + kKvDim, kKvDim, kQkv};
            g.nseg = 3;
            g.cols = kHidden;
            g.x = X_;
            g.tokens = T;
            gemv(g, s);
            kv_write_steps(kv_[L.index], P_, kQkv, step_, Q16_, T, s);
            crit::fp8_kv::Decode d;
            d.q = Q16_;
            d.q_stride = kQDim;
            d.cache = kvc_[L.index];
            d.out = AO_;
            d.out_stride = kQDim;
            d.T = T;
            d.heads = kQHeads;
            d.pos0_device = &step_[0].pos;
            d.scale = kAttnScale;
            d.blocks = options_.attn_blocks;
            d.partial = attn_partial_;
            NMH_CUDA(crit::fp8_kv::decode(d, s));
            Gemv o;
            o.fmt = WFormat::fp8;
            o.out = Out::add;
            o.seg[0] = {L.o.w, nullptr, L.o.scale, H_, kHidden, kHidden};
            o.cols = kQDim;
            o.x = AO_;
            o.tokens = T;
            gemv(o, s);
        } else {
            moe_decode(L.moe, X_, H_, T);
        }
    }
    rmsnorm(H_, norm_f_, X_, T, s);
    Gemv h;
    h.fmt = WFormat::fp8;
    h.seg[0] = {lm_head_.w, nullptr, lm_head_.scale, logits_, kVocab, kVocab};
    h.cols = kHidden;
    h.x = X_;
    h.tokens = T;
    gemv(h, s);
    row_scores(logits_, kVocab, kVocab, T, nullptr, nullptr, 0, nullptr, nullptr, next_, s);
}

void Net::decode(const std::vector<int>& history, int pos, int T) {
    if(T < 1 || T > kMaxTokens) throw std::invalid_argument("decode: 1..4 tokens");
    if(pos + T > options_.capacity) throw std::invalid_argument("decode past the context");
    const auto s = stream();
    // The previous pass may still read the pinned step records.
    NMH_CUDA(cudaStreamSynchronize(s));
    if(!workspace_lent_ && tables_.nslots) {
        directory_download();
        lend_workspace(true);
        directory_upload();
    }
    for(int t = 0; t < T; ++t) step_host_[t] = Step{pos + t, history[pos + t]};
    auto it = graphs_.find(T);
    if(it == graphs_.end()) {
        cudaGraph_t graph;
        NMH_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        NMH_CUDA(cudaMemcpyAsync(step_, step_host_, sizeof(Step) * T, cudaMemcpyHostToDevice, s));
        decode_layers(T, false);
        NMH_CUDA(cudaStreamEndCapture(s, &graph));
        cudaGraphExec_t exec;
        NMH_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        cudaGraphDestroy(graph);
        it = graphs_.emplace(T, exec).first;
    }
    NMH_CUDA(cudaGraphLaunch(it->second, s));
}

// ------------------------------------------------------------------------------------------------ drafter
void Net::mtp_rows(int R, const Step* steps, const float* hidden) {
    const auto s = stream();
    embed_steps(embed_, steps, mE_, R, s);
    rmsnorm(mE_, mtp_.enorm, mO_, R, s, 2 * kHidden);
    rmsnorm(hidden, mtp_.hnorm, mO_ + kHidden, R, s, 2 * kHidden);
    Gemv f;
    f.fmt = WFormat::bf16;
    f.seg[0] = {mtp_.eh_proj, nullptr, 1.f, mF_, kHidden, kHidden};
    f.cols = 2 * kHidden;
    f.x = mO_;
    f.tokens = R;
    gemv(f, s);
    rmsnorm(mF_, mtp_.norm0, mX_, R, s);
    Gemv g;
    g.fmt = WFormat::bf16;
    g.seg[0] = {mtp_.q, nullptr, 1.f, mP_, kQDim, kQkv};
    g.seg[1] = {mtp_.k, nullptr, 1.f, mP_ + kQDim, kKvDim, kQkv};
    g.seg[2] = {mtp_.v, nullptr, 1.f, mP_ + kQDim + kKvDim, kKvDim, kQkv};
    g.nseg = 3;
    g.cols = kHidden;
    g.x = mX_;
    g.tokens = R;
    gemv(g, s);
    kv_write_steps(kv_[kAttnLayers], mP_, kQkv, steps, mQ16_, R, s);
}

void Net::mtp_finish(const int* sel, const Step* step) {
    const auto s = stream();
    select_row_bf16(mQ16_, kQDim, sel, mQsel_, kQDim, s);
    select_row(mF_, kHidden, sel, mR_, kHidden, s);
    crit::fp8_kv::Decode d;
    d.q = mQsel_;
    d.q_stride = kQDim;
    d.cache = kvc_[kAttnLayers];
    d.out = AO_;
    d.out_stride = kQDim;
    d.T = 1;
    d.heads = kQHeads;
    d.pos0_device = &step->pos;
    d.scale = kAttnScale;
    d.blocks = options_.attn_blocks;
    d.partial = attn_partial_;
    NMH_CUDA(crit::fp8_kv::decode(d, s));
    Gemv o;
    o.fmt = WFormat::bf16;
    o.out = Out::add;
    o.seg[0] = {mtp_.o, nullptr, 1.f, mR_, kHidden, kHidden};
    o.cols = kQDim;
    o.x = AO_;
    gemv(o, s);
    rmsnorm(mR_, mtp_.norm1, mX_, 1, s);
    moe_decode(mtp_.moe, mX_, mR_, 1);
    rmsnorm(mR_, mtp_.final_norm, mtp_out_, 1, s);
    Gemv h;
    h.fmt = WFormat::fp8;
    h.seg[0] = {lm_head_.w, nullptr, lm_head_.scale, mlogits_, kVocab, kVocab};
    h.cols = kHidden;
    h.x = mtp_out_;
    gemv(h, s);
    row_scores(mlogits_, kVocab, kVocab, 1, nullptr, nullptr, 0, nullptr, nullptr, draft_tok_, s);
}

// Draft 0 is in draft_tok_; drafts 1 .. kDrafts - 1 chain the drafter on its own output.
void Net::draft_chain(int first_k) {
    const auto s = stream();
    for(int k = first_k; k < options_.drafts; ++k) {
        chain_step(draft_tok_, k, acc_, dctl_, s);
        if(k + 1 == options_.drafts) break;
        mtp_rows(1, &dctl_->one, mtp_out_);
        mtp_finish(draft_tok_ + 1, &dctl_->one);   // draft_tok_[1] is 0: the single row
    }
}

void Net::set_forced(const std::vector<int>& forced) {
    if(int(forced.size()) > kMaxForced) throw std::invalid_argument("replayed continuation too long");
    std::vector<int> buf(forced.size() + 1);
    buf[0] = int(forced.size());
    std::copy(forced.begin(), forced.end(), buf.begin() + 1);
    const auto s = stream();
    NMH_CUDA(cudaStreamSynchronize(s));
    NMH_CUDA(cudaMemcpy(forced_, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice));
    forced_len_ = int(forced.size());
}

void Net::spec_begin(int pos) {
    if(!mtp_loaded_) throw std::logic_error("no drafter loaded");
    const auto s = stream();
    NMH_CUDA(cudaStreamSynchronize(s));
    step_host_[0] = Step{pos, 0};
    if(!begin_graph_) {
        cudaGraph_t graph;
        NMH_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        NMH_CUDA(cudaMemcpyAsync(step_, step_host_, sizeof(Step), cudaMemcpyHostToDevice, s));
        accept_prompt(next_, step_, forced_ + 1, forced_, acc_, dctl_, s);
        mtp_rows(1, &dctl_->one, X_);   // X_: the prompt's last position, normed (prefill's head)
        mtp_finish(draft_tok_ + 1, &dctl_->one);
        draft_chain(0);
        NMH_CUDA(cudaMemcpyAsync(spec_host_, acc_, sizeof(Accept), cudaMemcpyDeviceToHost, s));
        NMH_CUDA(cudaStreamEndCapture(s, &graph));
        NMH_CUDA(cudaGraphInstantiate(&begin_graph_, graph, 0));
        cudaGraphDestroy(graph);
    }
    NMH_CUDA(cudaGraphLaunch(begin_graph_, s));
    if(!workspace_lent_ && tables_.nslots) {
        directory_download();   // synchronizes: the drafter has read the prompt pass's outputs
        lend_workspace(true);
        directory_upload();
    }
    NMH_CUDA(cudaEventRecord(spec_done_, s));
}

void Net::spec_round(int pos, int input, const int* drafts) {
    if(!mtp_loaded_) throw std::logic_error("no drafter loaded");
    const int T = options_.drafts + 1;
    if(pos + T > options_.capacity) throw std::invalid_argument("round past the context");
    const auto s = stream();
    NMH_CUDA(cudaStreamSynchronize(s));
    if(marks_pending_) {
        float a = 0, b = 0;
        NMH_CUDA(cudaEventElapsedTime(&a, marks_[0], marks_[1]));
        NMH_CUDA(cudaEventElapsedTime(&b, marks_[1], marks_[2]));
        verify_ms_ += a;
        draft_ms_ += b;
    }
    if(!workspace_lent_ && tables_.nslots) {
        directory_download();
        lend_workspace(true);
        directory_upload();
    }
    step_host_[0] = Step{pos, input};
    for(int k = 0; k + 1 < T; ++k) step_host_[k + 1] = Step{pos + 1 + k, drafts[k]};
    if(!round_graph_) {
        cudaGraph_t graph;
        NMH_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        NMH_CUDA(cudaMemcpyAsync(step_, step_host_, sizeof(Step) * T, cudaMemcpyHostToDevice, s));
        NMH_CUDA(cudaEventRecordWithFlags(marks_[0], s, cudaEventRecordExternal));
        decode_layers(T, true);
        accept_pass(next_, step_, T - 1, forced_ + 1, forced_, acc_, dctl_, s);
        commit_mamba(conv_state_, conv_rec_, kMambaLayers, ssm_cur_, &acc_->n, s);
        NMH_CUDA(cudaEventRecordWithFlags(marks_[1], s, cudaEventRecordExternal));
        mtp_rows(T, dctl_->rows, X_);
        mtp_finish(&dctl_->sel, &dctl_->one);
        draft_chain(0);
        NMH_CUDA(cudaMemcpyAsync(spec_host_, acc_, sizeof(Accept), cudaMemcpyDeviceToHost, s));
        NMH_CUDA(cudaEventRecordWithFlags(marks_[2], s, cudaEventRecordExternal));
        NMH_CUDA(cudaStreamEndCapture(s, &graph));
        NMH_CUDA(cudaGraphInstantiate(&round_graph_, graph, 0));
        cudaGraphDestroy(graph);
    }
    NMH_CUDA(cudaGraphLaunch(round_graph_, s));
    NMH_CUDA(cudaEventRecord(spec_done_, s));
    marks_pending_ = true;
}

std::pair<double, double> Net::pass_timing() {
    const auto s = stream();
    NMH_CUDA(cudaStreamSynchronize(s));
    if(marks_pending_) {
        float a = 0, b = 0;
        NMH_CUDA(cudaEventElapsedTime(&a, marks_[0], marks_[1]));
        NMH_CUDA(cudaEventElapsedTime(&b, marks_[1], marks_[2]));
        verify_ms_ += a;
        draft_ms_ += b;
        marks_pending_ = false;
    }
    return {verify_ms_, draft_ms_};
}

}  // namespace nmh
