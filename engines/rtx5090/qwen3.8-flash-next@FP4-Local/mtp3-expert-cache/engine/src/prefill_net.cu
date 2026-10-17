// The prompt (multi-token) path of Net: BF16 tensor-core GEMMs, host-planned expert residency with DMA copies.
#include "gemm.cuh"
#include "model.h"
#include "prefill.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace fnl {

using crit::core::CudaBuffer;
using crit::core::Memory;
using crit::core::Queue;

namespace {
// Expert weight copies host -> device as one batch (stream ordered on ts).
struct CopyBatch {
    std::vector<void*> dst;
    std::vector<const void*> src;
    void add(void* d, const void* s) {
        dst.push_back(d);
        src.push_back(s);
    }
    void issue(cudaStream_t ts) {
        if(dst.empty()) return;
        std::vector<std::size_t> sizes(dst.size(), kExpertWeightBytes);
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        std::size_t index = 0;
        FNL_CUDA(cudaMemcpyBatchAsync(dst.data(), src.data(), sizes.data(), dst.size(), &attr, &index, 1, ts));
        dst.clear();
        src.clear();
    }
};
int prompt_cpu_routes_last(const std::vector<CpuRowJob>& jobs) {
    int n = 0;
    for(const auto& j : jobs) n += j.tokens;
    return n;
}
bool is_attention(int layer) { return (layer + 1) % 4 == 0; }
int gdn_index(int layer) { return layer - (layer + 1) / 4; }
int attn_index(int layer) { return layer / 4; }

// Bump allocator over the phase scratch.
struct Bump {
    uint8_t* base;
    std::size_t cap, off = 0;
    template <class T> T* get(std::size_t n) {
        const std::size_t bytes = (n * sizeof(T) + 255) / 256 * 256;
        if(off + bytes > cap) throw std::runtime_error("prompt scratch overflow");
        T* p = reinterpret_cast<T*>(base + off);
        off += bytes;
        return p;
    }
};

GemmArgs dense_args(const __half* A, int lda, int M, int N, int K, const void* W, const uint8_t* Ws, float* C,
                    __half* Cb, int ldc) {
    GemmArgs g;
    g.A = A;
    g.lda = lda;
    g.M = M;
    g.N = N;
    g.K = K;
    g.W = W;
    g.Ws = Ws;
    g.C = C;
    g.Cb = Cb;
    g.ldc = ldc;
    return g;
}
__global__ void absmax_kernel(const __half* x, std::size_t n, unsigned* out) {
    float m = 0.f;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(__half2float(x[i])));
    m = warp_max(m);
    if((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
}
// FNL_RANGE: largest magnitude of an f16 activation buffer (diagnostic, synchronizing).
void report_range(const char* what, int layer, const __half* x, std::size_t n, cudaStream_t s) {
    static const bool on = std::getenv("FNL_RANGE") != nullptr;
    if(!on) return;
    unsigned* d;
    FNL_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d), 4, s));
    FNL_CUDA(cudaMemsetAsync(d, 0, 4, s));
    absmax_kernel<<<256, 256, 0, s>>>(x, n, d);
    unsigned h;
    FNL_CUDA(cudaMemcpyAsync(&h, d, 4, cudaMemcpyDeviceToHost, s));
    FNL_CUDA(cudaFreeAsync(d, s));
    FNL_CUDA(cudaStreamSynchronize(s));
    float f;
    std::memcpy(&f, &h, 4);
    std::fprintf(stderr, "range %s layer %d max %.3g\n", what, layer, f);
}
}  // namespace

void Net::allocate_prefill() {
    const std::size_t T = options_.prefill_chunk, cap = options_.capacity;
    const int nsub = int((T + kSub - 1) / kSub);
    struct Item { void** p; std::size_t bytes; };
    std::vector<Item> items;
    auto add = [&](auto** p, std::size_t bytes) { items.push_back({reinterpret_cast<void**>(p), bytes}); };
    add(&pkeys3_, std::size_t(cap / kBlockTokens) * 3 * kIdxDim * 2);
    add(&pscore_logits_, std::size_t(kScoreRows) * kVocab * 4);
    add(&pH_, T * kHyper * 4);
    add(&px_, T * kHidden * 4);
    add(&px_bf_, T * kHidden * 2);
    add(&pinj_, T * kStreams * 4);
    add(&pout_, T * kHidden * 4);
    add(&proutes_, T * sizeof(Route));
    add(&ppos_of_, T * kTopK * 4);
    // phase scratch: the largest of the HC, GDN, attention, MoE and PLE phases per token
    pscratch_bytes_ = T * 152 * 1024;
    add(&pscratch_, pscratch_bytes_);
    add(&pact_, std::size_t(kSub) * kTopK * kInter * 2);
    add(&proute_out_, std::size_t(kSub) * kTopK * kHidden * 4);
    add(&proute_token_, std::size_t(kSub) * kTopK * 4);
    add(&pcounts_, std::size_t(nsub) * kExperts * 4);
    add(&poffsets_, kExperts * 4);
    add(&pcursor_, kExperts * 4);
    add(&ptiles_, (std::size_t(kSub) * kTopK / 128 + kExperts) * sizeof(int4));
    add(&pntiles_, 4);
    add(&pgroup_base_, kExperts * sizeof(void*));
    add(&pjobs_, 1024 * sizeof(ScaleJob));
    add(&pscores_, std::size_t(kQueryGroup) * (cap / kBlockTokens) * 4);
    add(&psel_, std::size_t(kQueryGroup) * kTopBlocks * 4);
    add(&prows_dev_, T * kPleHeads * 90);
    std::size_t total = 0;
    for(auto& it : items) total += (it.bytes + 255) / 256 * 256;
    auto buf = backend_.allocate(Memory::device, total);
    auto* base = static_cast<uint8_t*>(buf.data());
    std::size_t off = 0;
    for(auto& it : items) {
        *it.p = base + off;
        off += (it.bytes + 255) / 256 * 256;
    }
    buffers_.push_back(std::move(buf));
    // Prompt-only buffers lend their memory to the expert cache between prompt passes.
    workspace_lend_ = {{reinterpret_cast<uint8_t*>(pH_), T * kHyper * 4},
                       {pscratch_, std::size_t(reinterpret_cast<uint8_t*>(proute_out_) - pscratch_) +
                                       std::size_t(kSub) * kTopK * kHidden * 4}};
    // pinned staging
    const std::size_t hbytes = std::size_t(nsub) * kExperts * 4 + kExperts * 4 +
                               (std::size_t(kSub) * kTopK / 128 + kExperts) * sizeof(int4) + 64 +
                               kExperts * sizeof(void*) + 1024 * sizeof(ScaleJob) + T * kPleHeads * 90 + 4096;
    auto h = backend_.allocate(Memory::host, hbytes);
    auto* hb = static_cast<uint8_t*>(h.data());
    std::size_t ho = 0;
    auto hadd = [&](auto** p, std::size_t bytes) {
        *p = reinterpret_cast<std::remove_reference_t<decltype(*p)>>(hb + ho);
        ho += (bytes + 255) / 256 * 256;
    };
    hadd(&hcounts_, std::size_t(nsub) * kExperts * 4);
    hadd(&hoffsets_, kExperts * 4);
    hadd(&htiles_, (std::size_t(kSub) * kTopK / 128 + kExperts) * sizeof(int4));
    hadd(&hntiles_, 64);
    hadd(&hgroup_base_, kExperts * sizeof(void*));
    hadd(&hjobs_, 1024 * sizeof(ScaleJob));
    hadd(&hrows_, T * kPleHeads * 90);
    if(ho > hbytes) throw std::logic_error("pinned staging layout");
    buffers_.push_back(std::move(h));
    copies_done_ = backend_.event();
}

void Net::directory_download() {
    const int n = tables_.nslots;
    mslot_of_.assign(std::size_t(kLayers) * kExperts, -1);
    mowner_.assign(n, -1);
    mref_.assign(n, 0);
    if(int(mvalid_.size()) != n) mvalid_.assign(n, 1);
    mstamp_.assign(n, -1);
    // mvalid_ is owned by the host (begin_request, prompt passes); the device copy only mirrors it.
    FNL_CUDA(cudaStreamSynchronize(stream()));
    FNL_CUDA(cudaMemcpy(mslot_of_.data(), tables_.slot_of, mslot_of_.size() * 4, cudaMemcpyDeviceToHost));
    if(n) {
        FNL_CUDA(cudaMemcpy(mowner_.data(), tables_.owner, n * 4, cudaMemcpyDeviceToHost));
        FNL_CUDA(cudaMemcpy(mref_.data(), tables_.ref, n, cudaMemcpyDeviceToHost));
        FNL_CUDA(cudaMemcpy(&mhand_, tables_.hand, 4, cudaMemcpyDeviceToHost));
    }
    mepoch_ = 1;
    prefetched_.clear();
    // The prompt pass writes its workspace: slots lent from it are evicted and unusable until it ends.
    for(int s = 0; s < n; ++s)
        if(slot_kind_[s] == kWorkspaceSlot) evict_slot(s, false);
}

void Net::directory_upload() {
    const int n = tables_.nslots;
    const auto s = stream();
    FNL_CUDA(cudaStreamSynchronize(s));
    FNL_CUDA(cudaMemcpy(tables_.slot_of, mslot_of_.data(), mslot_of_.size() * 4, cudaMemcpyHostToDevice));
    if(n) {
        FNL_CUDA(cudaMemcpy(tables_.owner, mowner_.data(), n * 4, cudaMemcpyHostToDevice));
        FNL_CUDA(cudaMemcpy(tables_.ref, mref_.data(), n, cudaMemcpyHostToDevice));
        FNL_CUDA(cudaMemcpy(tables_.hand, &mhand_, 4, cudaMemcpyHostToDevice));
        FNL_CUDA(cudaMemset(tables_.stamp, 0, std::size_t(n) * 4));
        // Workspace slots come back (empty) for the decode phase.
        for(int s = 0; s < n; ++s)
            if(slot_kind_[s] == kWorkspaceSlot) mvalid_[s] = 1;
        FNL_CUDA(cudaMemcpy(const_cast<uint8_t*>(tables_.valid), mvalid_.data(), n, cudaMemcpyHostToDevice));
    }
}

void Net::evict_slot(int s, bool keep_valid) {
    if(mowner_[s] >= 0) mslot_of_[mowner_[s]] = -1;
    mowner_[s] = -1;
    mref_[s] = 0;
    mvalid_[s] = keep_valid ? 1 : 0;
}

// Lends the KV positions a request cannot reach to the expert cache: KV-region slots below `positions` are
// evicted and disabled, the rest (and the workspace slots) enabled.
void Net::begin_request(int positions) {
    if(positions > options_.capacity) throw std::out_of_range("request beyond the KV capacity");
    const int n = tables_.nslots;
    if(!n) return;
    sync_insertions();
    directory_download();
    for(int s = 0; s < n; ++s) {
        if(slot_kind_[s] == kKvSlot) {
            if(slot_first_pos_[s] < positions) evict_slot(s, false);
            else mvalid_[s] = 1;
        }
    }
    directory_upload();
}

// CLOCK victim among slots not stamped at or after protect_epoch.
int Net::host_victim(int protect_epoch) {
    const int n = tables_.nslots;
    for(int tries = 0; tries < 2 * n + 2; ++tries) {
        const int s = mhand_;
        mhand_ = mhand_ + 1 == n ? 0 : mhand_ + 1;
        if(mstamp_[s] >= protect_epoch || !mvalid_[s]) continue;
        if(mref_[s]) {
            mref_[s] = 0;
            continue;
        }
        return s;
    }
    throw std::runtime_error("expert cache has no evictable slot for a prompt pass");
}

// Makes every expert in want resident for this layer; experts that need a copy are appended to fetch.
void Net::host_plan_layer(int layer, const std::vector<int>& want, std::vector<int>& fetch) {
    const int epoch = 2 * layer + 2;
    for(int e : want) {
        const int s = mslot_of_[layer * kExperts + e];
        if(s >= 0) {
            mref_[s] = 1;
            mstamp_[s] = epoch;
        }
    }
    for(int e : want) {
        if(mslot_of_[layer * kExperts + e] >= 0) continue;
        const int v = host_victim(epoch);   // never evict this layer's own experts
        if(mowner_[v] >= 0) mslot_of_[mowner_[v]] = -1;
        mowner_[v] = layer * kExperts + e;
        mslot_of_[layer * kExperts + e] = v;
        mref_[v] = 1;
        mstamp_[v] = epoch;
        fetch.push_back(e);
    }
}

void Net::hc_pre_batch(const Hc& w, int T, bool inject) {
    const auto s = stream();
    Bump b{pscratch_, pscratch_bytes_};
    float* xn = b.get<float>(std::size_t(T) * kHyper);
    __half* xn_bf = b.get<__half>(std::size_t(T) * kHyper);
    __half* lora = b.get<__half>(std::size_t(T) * kLowrank);
    float* mixw = b.get<float>(std::size_t(T) * kHyper);
    hc_norm(pH_, w.norm1p, xn, T, s);
    to_f16(xn, xn_bf, std::size_t(T) * kHyper, s);
    gemm(WFmt::bf16, Epi::bf16_siluq, dense_args(xn_bf, kHyper, T, kLowrank, kHyper, w.down, nullptr, nullptr, lora,
                                                  kLowrank), 0, s);
    gemm(WFmt::bf16, Epi::f32_sigmoid, dense_args(lora, kLowrank, T, kHyper, kLowrank, w.up, nullptr, mixw, nullptr,
                                                   kHyper), 0, s);
    if(inject) hc_inject(xn, w.inject, pinj_, T, s);
    hc_mix(xn, mixw, px_, T, s);
    to_f16(px_, px_bf_, std::size_t(T) * kHidden, s);
    report_range("block-input", -1, px_bf_, std::size_t(T) * kHidden, s);
    report_range("hc-xn", -1, xn_bf, std::size_t(T) * kHyper, s);
}

void Net::gdn_batch(const LayerWeights& L, int gi, int T) {
    const auto s = stream();
    Bump b{pscratch_, pscratch_bytes_};
    float* qkv = b.get<float>(std::size_t(T) * kQkvDim);
    float* z = b.get<float>(std::size_t(T) * kZDim);
    float* ab = b.get<float>(std::size_t(T) * 2 * kVHeads);
    float* q = b.get<float>(std::size_t(T) * kKHeads * kKDim);
    float* k = b.get<float>(std::size_t(T) * kKHeads * kKDim);
    float* v = b.get<float>(std::size_t(T) * kZDim);
    float* g = b.get<float>(std::size_t(T) * kVHeads);
    float* beta = b.get<float>(std::size_t(T) * kVHeads);
    float* o = b.get<float>(std::size_t(T) * kZDim);
    __half* gn = b.get<__half>(std::size_t(T) * kZDim);
    gemm(WFmt::mx8, Epi::f32, dense_args(px_bf_, kHidden, T, kQkvDim, kHidden, L.qkv.w, L.qkv.s, qkv, nullptr, kQkvDim), 0, s);
    gemm(WFmt::mx8, Epi::f32, dense_args(px_bf_, kHidden, T, kZDim, kHidden, L.z.w, L.z.s, z, nullptr, kZDim), 0, s);
    gemm(WFmt::mx8, Epi::f32, dense_args(px_bf_, kHidden, T, kVHeads, kHidden, L.a.w, L.a.s, ab, nullptr, 2 * kVHeads), 0, s);
    gemm(WFmt::mx8, Epi::f32, dense_args(px_bf_, kHidden, T, kVHeads, kHidden, L.b.w, L.b.s, ab + kVHeads, nullptr,
                                         2 * kVHeads), 0, s);
    GdnStep gs{qkv, ab, L.conv_w, conv_state_ + std::size_t(gi) * kQkvDim * 3, L.neg_a, L.dt_bias, q, k, v, g, beta, T};
    gdn_conv(gs, s);
    gdn_recur(q, k, v, g, beta, gdn_state_ + std::size_t(gi) * kGdnRing * kGdnStateFloats, gdn_cur_, false, o, T,
              s);
    gdn_gate_norm(o, z, L.gnorm, o, T, s);
    to_f16(o, gn, std::size_t(T) * kZDim, s);
    gemm(WFmt::mx8, Epi::f32, dense_args(gn, kZDim, T, kHidden, kZDim, L.out.w, L.out.s, pout_, nullptr, kHidden), 0, s);
}

void Net::attn_batch(const LayerWeights& L, int ai, int pos0, int T) {
    const auto s = stream();
    const std::size_t cap = options_.capacity;
    Bump b{pscratch_, pscratch_bytes_};
    float* proj = b.get<float>(std::size_t(T) * kProjAttn);
    __half* qb = b.get<__half>(std::size_t(T) * kAttnOut);
    float* gate = b.get<float>(std::size_t(T) * kAttnOut);
    float* iq = b.get<float>(std::size_t(T) * kIdxHeads * kIdxDim);
    __half* iq3 = b.get<__half>(std::size_t(T) * kIdxHeads * 3 * kIdxDim);
    float* raw = b.get<float>(std::size_t(T) * kIdxDim);
    __half* ob = b.get<__half>(std::size_t(T) * kAttnOut);
    int col = 0;
    auto proj_gemm = [&](const Mx8& w) {
        gemm(WFmt::mx8, Epi::f32, dense_args(px_bf_, kHidden, T, w.rows, kHidden, w.w, w.s, proj + col, nullptr,
                                             kProjAttn), 0, s);
        col += w.rows;
    };
    proj_gemm(L.q);
    proj_gemm(L.k);
    proj_gemm(L.v);
    proj_gemm(L.idx);
    uint8_t* kc = kcache_ + std::size_t(ai) * kKvHeads * cap * kHeadDim;
    uint8_t* vc = vcache_ + std::size_t(ai) * kKvHeads * cap * kHeadDim;
    float* ks = kscale_ + std::size_t(ai) * kKvHeads * cap;
    float* vs = vscale_ + std::size_t(ai) * kKvHeads * cap;
    float* bk = blockkeys_ + std::size_t(ai) * (cap / kBlockTokens) * kIdxDim;
    AttnPrepBatch ap{proj, L.qn1p, L.kn1p, L.iqn1p, L.ikn1p, inv_freq_, qb, gate, iq, iq3, raw, kc, vc, ks, vs,
                     rawkeys_ + std::size_t(ai) * kRawRing * kIdxDim, bk, options_.capacity, pos0, T};
    attn_prep_batch(ap, s);
    pool_blocks(ap, s);
    const int nb_all = (pos0 + T) / kBlockTokens;
    if(nb_all > kTopBlocks) split_keys(bk, nb_all, pkeys3_, s);
    for(int q0 = 0; q0 < T; q0 += kQueryGroup) {
        const int G = std::min(kQueryGroup, T - q0);
        const int nb_last = (pos0 + q0 + G) / kBlockTokens;
        if(nb_last > kTopBlocks) {
            GemmArgs ga = dense_args(iq3 + std::size_t(q0) * kIdxHeads * 3 * kIdxDim, 3 * kIdxDim, G * kIdxHeads,
                                     nb_last, 3 * kIdxDim, pkeys3_, nullptr, pscores_, nullptr, nb_last);
            gemm(WFmt::f16, Epi::idx, ga, 0, s);
            topk_batch(pscores_, nb_last, pos0, q0, G, psel_, s);
        }
        attn_prefill({qb, gate, kc, vc, ks, vs, psel_, options_.capacity, pos0, q0, G, ob}, s);
    }
    report_range("attn-out", ai, ob, std::size_t(T) * kAttnOut, s);
    gemm(WFmt::mx8, Epi::f32, dense_args(ob, kAttnOut, T, kHidden, kAttnOut, L.o.w, L.o.s, pout_, nullptr, kHidden), 0, s);
}

void Net::moe_batch(const LayerWeights& L, int layer, int T, bool prefetch_next) {
    const auto s = stream();
    const auto ts = backend_.stream(Queue::transfer);
    const int nsub = (T + kSub - 1) / kSub;
    Bump b{pscratch_, pscratch_bytes_};
    float* logits = b.get<float>(std::size_t(T) * kExperts);
    __half* sh_act = b.get<__half>(std::size_t(T) * kShared);
    float* sh_out = b.get<float>(std::size_t(T) * kHidden);
    float* sg = b.get<float>(T);
    // Short chunks: part of the missing experts (routed by at most 4 tokens) is computed on the CPU.
    const bool cpu = prompt_cpu_ && T <= kCpuPromptTokens;
    float* cpu_rows = cpu ? b.get<float>(std::size_t(kCpuPromptRows) * kHidden) : nullptr;
    if(cpu) FNL_CUDA(cudaMemcpyAsync(hx_, px_, std::size_t(T) * kHidden * 4, cudaMemcpyDeviceToHost, s));
    gemm(WFmt::bf16, Epi::f32, dense_args(px_bf_, kHidden, T, kExperts, kHidden, L.router, nullptr, logits, nullptr,
                                          kExperts), 0, s);
    route_batch(logits, proutes_, T, s);
    if(cpu) FNL_CUDA(cudaMemcpyAsync(hroutes_, proutes_, std::size_t(T) * sizeof(Route), cudaMemcpyDeviceToHost, s));
    FNL_CUDA(cudaMemsetAsync(pcounts_, 0, std::size_t(nsub) * kExperts * 4, s));
    count_routes(proutes_, T, kSub, pcounts_, s);
    FNL_CUDA(cudaMemcpyAsync(hcounts_, pcounts_, std::size_t(nsub) * kExperts * 4, cudaMemcpyDeviceToHost, s));
    // Shared expert while the counts travel.
    {
        GemmArgs g = dense_args(px_bf_, kHidden, T, 2 * kShared, kHidden, L.sh_gate.w, L.sh_gate.s, nullptr, sh_act, kShared);
        g.W2 = L.sh_up.w;
        g.Ws2 = L.sh_up.s;
        gemm(WFmt::mx8, Epi::act, g, 0, s);
        report_range("shared-act", layer, sh_act, std::size_t(T) * kShared, s);
        gemm(WFmt::mx8, Epi::f32, dense_args(sh_act, kShared, T, kHidden, kShared, L.sh_down.w, L.sh_down.s, sh_out,
                                             nullptr, kHidden), 0, s);
        shared_gate(px_, L.shared_gate, sg, T, s);
    }
    FNL_CUDA(cudaStreamSynchronize(s));
    // Plan residency for this layer.
    std::vector<int> want;
    std::vector<int> total(kExperts, 0);
    for(int e = 0; e < kExperts; ++e) {
        int c = 0;
        for(int k = 0; k < nsub; ++k) c += hcounts_[k * kExperts + e];
        total[e] = c;
        if(c) want.push_back(e);
    }
    // CPU share: missing experts with the fewest tokens (at most 4), about the share of the misses that the CPU
    // computes while the GPU copies the rest.
    std::vector<int> cpu_experts;
    std::vector<uint8_t> on_cpu(kExperts, 0);
    if(cpu) {
        std::vector<int> missing, cand;
        for(int e : want)
            if(mslot_of_[layer * kExperts + e] < 0) {
                missing.push_back(e);
                if(total[e] <= kCpuMaxTokens) cand.push_back(e);
            }
        std::stable_sort(cand.begin(), cand.end(), [&](int a, int c) { return total[a] < total[c]; });
        const int n = std::min<int>(int(cand.size()), int(prompt_cpu_share_ * float(missing.size()) + 0.5f));
        cpu_experts.assign(cand.begin(), cand.begin() + n);
        for(int e : cpu_experts) on_cpu[e] = 1;
        std::vector<int> rest;
        for(int e : want)
            if(!on_cpu[e]) rest.push_back(e);
        want.swap(rest);
    }
    // Short chunks: missing experts routed by few tokens go to a transient pool in the unused tail of the
    // prompt workspace (pH_ holds only T rows), not into the cache.
    std::vector<std::pair<int, uint8_t*>> transient;
    if(T < kTransientChunk) {
        auto* ws = reinterpret_cast<uint8_t*>(pH_);
        const std::size_t used = (std::size_t(T) * kHyper * 4 + 255) / 256 * 256;
        const std::size_t room = (workspace_lend_[0].second - used) / kSlotBytes;
        std::vector<int> rest;
        for(int e : want) {
            if(mslot_of_[layer * kExperts + e] < 0 && total[e] <= kTransientTokens && transient.size() < room)
                transient.push_back({e, ws + used + transient.size() * kSlotBytes});
            else
                rest.push_back(e);
        }
        want.swap(rest);
        prompt_transient_ += transient.size();
    }
    std::vector<int> fetch;
    host_plan_layer(layer, want, fetch);
    prefill_fetched_ += fetch.size() + transient.size();
    CopyBatch copies;
    for(int e : fetch)
        copies.add(mslot_addr_[mslot_of_[layer * kExperts + e]], host_layers_[layer] + std::size_t(e) * kHostExpertBytes);
    for(const auto& [e, slot] : transient) copies.add(slot, host_layers_[layer] + std::size_t(e) * kHostExpertBytes);
    copies.issue(ts);
    // Scales: fresh copies and earlier prefetches of this layer.
    int njobs = 0;
    for(int e : fetch) hjobs_[njobs++] = {e, mslot_addr_[mslot_of_[layer * kExperts + e]]};
    for(const auto& [e, slot] : transient) hjobs_[njobs++] = {e, slot};
    for(int e : prefetched_)
        if(std::binary_search(want.begin(), want.end(), e) &&
           std::find(fetch.begin(), fetch.end(), e) == fetch.end())
            hjobs_[njobs++] = {e, mslot_addr_[mslot_of_[layer * kExperts + e]]};
    prefetched_.clear();
    copies_done_.record(ts);
    // Prefetch every expert of the next layer that is not resident (long prompts use nearly all of them).
    if(prefetch_next && layer + 1 < kLayers) {
        std::vector<int> all(kExperts);
        for(int e = 0; e < kExperts; ++e) all[e] = e;
        std::vector<int> next_fetch;
        // Protect this layer (stamp 2 layer + 2) and the next layer's own picks (2 layer + 4).
        const int epoch = 2 * (layer + 1) + 2;
        for(int e : all) {
            const int sl = mslot_of_[(layer + 1) * kExperts + e];
            if(sl >= 0) {
                mstamp_[sl] = epoch;
                mref_[sl] = 1;
            }
        }
        for(int e : all) {
            if(mslot_of_[(layer + 1) * kExperts + e] >= 0) continue;
            const int v = host_victim(2 * layer + 2);
            if(mowner_[v] >= 0) mslot_of_[mowner_[v]] = -1;
            mowner_[v] = (layer + 1) * kExperts + e;
            mslot_of_[(layer + 1) * kExperts + e] = v;
            mref_[v] = 1;
            mstamp_[v] = epoch;
            next_fetch.push_back(e);
            copies.add(mslot_addr_[v], host_layers_[layer + 1] + std::size_t(e) * kHostExpertBytes);
        }
        copies.issue(ts);
        prefill_fetched_ += next_fetch.size();
        prefetched_ = next_fetch;
    }
    for(int e = 0; e < kExperts; ++e) {
        const int sl = mslot_of_[layer * kExperts + e];
        hgroup_base_[e] = sl >= 0 ? mslot_addr_[sl] : nullptr;
    }
    for(const auto& [e, slot] : transient) hgroup_base_[e] = slot;
    FNL_CUDA(cudaMemcpyAsync(pgroup_base_, hgroup_base_, kExperts * sizeof(void*), cudaMemcpyHostToDevice, s));
    FNL_CUDA(cudaMemcpyAsync(pjobs_, hjobs_, njobs * sizeof(ScaleJob), cudaMemcpyHostToDevice, s));
    // CPU routes: rows of the CPU results per (token, route), -1 elsewhere.
    std::vector<CpuRowJob> cpu_jobs;
    if(cpu) {
        std::fill(hcpu_row_, hcpu_row_ + std::size_t(T) * kTopK, -1);
        int rows = 0;
        std::vector<int> job_of(kExperts, -1);
        for(int e : cpu_experts) {
            job_of[e] = int(cpu_jobs.size());
            CpuRowJob j{};
            const std::size_t i = (std::size_t(layer) * kExperts + e) * 3;
            j.expert.record = host_layers_[layer] + std::size_t(e) * kHostExpertBytes;
            for(int m = 0; m < 3; ++m) {
                j.expert.exc[m] = exc_pool_h_ + exc_off_h_[i + m];
                j.expert.exc_count[m] = exc_cnt_h_[i + m];
                j.expert.s2[m] = s2_h_[i + m];
            }
            cpu_jobs.push_back(j);
        }
        for(int t = 0; t < T; ++t)
            for(int k = 0; k < kTopK; ++k) {
                const int e = hroutes_[t].expert[k];
                if(job_of[e] < 0) continue;
                CpuRowJob& j = cpu_jobs[job_of[e]];
                j.x[j.tokens] = hx_ + std::size_t(t) * kHidden;
                j.y[j.tokens] = hcpu_rows_ + std::size_t(rows) * kHidden;
                ++j.tokens;
                hcpu_row_[t * kTopK + k] = rows++;
            }
        FNL_CUDA(cudaMemcpyAsync(pcpu_row_, hcpu_row_, std::size_t(T) * kTopK * 4, cudaMemcpyHostToDevice, s));
        prompt_cpu_routes_ += rows;
    }
    copies_done_.order(s);
    // After the copies: decoding beside them competes with them for the host link (c3 measured -9% prefill).
    decode_scales(tables_, layer, pjobs_, njobs, s);
    // Sub-chunks: sort routes by expert, grouped GEMMs, combine.
    const float* s2 = tables_.s2 + std::size_t(layer) * kExperts * 3;
    for(int k = 0; k < nsub; ++k) {
        const int t0 = k * kSub, t1 = std::min(T, t0 + kSub);
        int cnt[kExperts];
        for(int e = 0; e < kExperts; ++e) cnt[e] = on_cpu[e] ? 0 : hcounts_[k * kExperts + e];
        int off = 0, nt = 0;
        int* hoff = hoffsets_ + 0;
        int4* tiles = htiles_;
        for(int e = 0; e < kExperts; ++e) {
            hoff[e] = off;
            for(int r = 0; r < cnt[e]; r += 128) tiles[nt++] = make_int4(e, off + r, off + std::min(cnt[e], r + 128), 0);
            off += cnt[e];
        }
        *hntiles_ = nt;
        // Uploads are ordered on s; the pinned sources are rewritten only after the next synchronization.
        FNL_CUDA(cudaMemcpyAsync(poffsets_, hoff, kExperts * 4, cudaMemcpyHostToDevice, s));
        FNL_CUDA(cudaMemcpyAsync(ptiles_, tiles, nt * sizeof(int4), cudaMemcpyHostToDevice, s));
        FNL_CUDA(cudaMemcpyAsync(pntiles_, hntiles_, 4, cudaMemcpyHostToDevice, s));
        FNL_CUDA(cudaMemsetAsync(pcursor_, 0, kExperts * 4, s));
        scatter_routes(proutes_, t0, t1, poffsets_, pcursor_, proute_token_, ppos_of_, s, cpu ? pcpu_row_ : nullptr);
        // pos_of holds positions relative to this sub-chunk's route rows; route_token holds absolute tokens.
        GemmArgs gu;
        gu.A = px_bf_;
        gu.lda = kHidden;
        gu.a_rows = proute_token_;
        gu.M = off;
        gu.N = 2 * kInter;
        gu.K = kHidden;
        gu.Cb = pact_;
        gu.ldc = kInter;
        gu.tiles = ptiles_;
        gu.ntiles = pntiles_;
        gu.group_base = pgroup_base_;
        gu.w_off = 0;
        gu.w2_off = kGateBytes;
        gu.s_off = kSlotScales;
        gu.s2_off = kSlotScales + kGateScales;
        gu.group_alpha = s2;
        gu.alpha_idx = 0;
        gu.alpha2_idx = 1;
        gemm(WFmt::fp4, Epi::act, gu, nt, s);
        report_range("expert-act", layer, pact_, std::size_t(off) * kInter, s);
        GemmArgs gd;
        gd.A = pact_;
        gd.lda = kInter;
        gd.M = off;
        gd.N = kHidden;
        gd.K = kInter;
        gd.C = proute_out_;
        gd.ldc = kHidden;
        gd.tiles = ptiles_;
        gd.ntiles = pntiles_;
        gd.group_base = pgroup_base_;
        gd.w_off = 2 * kGateBytes;
        gd.s_off = kSlotScales + 2 * kGateScales;
        gd.group_alpha = s2;
        gd.alpha_idx = 2;
        gemm(WFmt::fp4, Epi::f32, gd, nt, s);
        if(cpu && k == 0) {
            // The GPU has this layer's copies and GEMMs queued; the CPU computes its experts meanwhile.
            prompt_cpu_->run_each(cpu_jobs.data(), int(cpu_jobs.size()));
            FNL_CUDA(cudaMemcpyAsync(cpu_rows, hcpu_rows_, std::size_t(std::max(1, int(prompt_cpu_routes_last(cpu_jobs)))) *
                                     kHidden * 4, cudaMemcpyHostToDevice, s));
        }
        moe_combine(proutes_, ppos_of_, proute_out_, sh_out, sg, pout_, t0, t1, s, cpu_rows);
        if(k + 1 < nsub) FNL_CUDA(cudaStreamSynchronize(s));  // pinned offsets/tiles are reused
    }
}

void Net::ple_batch(const std::vector<int>& history, int pos0, int T) {
    const auto s = stream();
    // Gather the n-gram rows on host threads (page faults of the mapped table run in parallel).
    // Page faults of the mapped table dominate (cold rows): many threads keep the disk queue deep.
    const int threads = std::max(1, std::min(48, T / 16 + 1));
    std::vector<std::thread> pool;
    for(int w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            const int a = T * w / threads, b2 = T * (w + 1) / threads;
            if(b2 > a) ple_rows(history.data(), 0, pos0 + a, b2 - a, hrows_ + std::size_t(a) * kPleHeads * 90);
        });
    for(auto& th : pool) th.join();
    FNL_CUDA(cudaMemcpyAsync(prows_dev_, hrows_, std::size_t(T) * kPleHeads * 90, cudaMemcpyHostToDevice, s));
    Bump b{pscratch_, pscratch_bytes_};
    float* e = b.get<float>(std::size_t(T) * kHidden);
    __half* eb = b.get<__half>(std::size_t(T) * kHidden);
    float* key = b.get<float>(std::size_t(T) * kHyper);
    float* value = b.get<float>(std::size_t(T) * kHidden);
    float* gv = b.get<float>(std::size_t(T) * kHyper);
    float* gvn = b.get<float>(std::size_t(T) * kHyper);
    ple_dequant(prows_dev_, 1.f, e, T, s);
    to_f16(e, eb, std::size_t(T) * kHidden, s);
    GemmArgs gk = dense_args(eb, kHidden, T, kHyper, kHidden, ple_.key, nullptr, key, nullptr, kHyper);
    gk.alpha = ple_.s2;
    gemm(WFmt::bf16, Epi::f32, gk, 0, s);
    GemmArgs gvv = dense_args(eb, kHidden, T, kHidden, kHidden, ple_.value, nullptr, value, nullptr, kHidden);
    gvv.alpha = ple_.s2;
    gemm(WFmt::bf16, Epi::f32, gvv, 0, s);
    ple_mix({pH_, key, value, ple_.nk1p, ple_.nq1p, ple_.nc1p, ple_.conv_w, ple_state_, T}, gv, gvn, s);
}

void Net::prefill(const std::vector<int>& history, int pos0, int T, bool keep_all, bool drafter) {
    if(T <= 0 || T > options_.prefill_chunk) throw std::invalid_argument("prompt chunk size");
    if(pos0 + T > options_.capacity) throw std::out_of_range("prompt beyond the KV capacity");
    if(tables_.nslots < 2 * kExperts + 64) throw std::runtime_error("too few expert slots for a prompt pass");
    const auto s = stream();
    const auto entry = std::chrono::steady_clock::now();
    sync_insertions();
    directory_download();
    // Token ids for the embedding gather.
    std::vector<int> ids(history.begin() + pos0, history.begin() + pos0 + T);
    int* dids = reinterpret_cast<int*>(prows_dev_);
    FNL_CUDA(cudaMemcpyAsync(dids, ids.data(), std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
    embed_rows(embed_, dids, pH_, T, s);
    FNL_CUDA(cudaStreamSynchronize(s));
    const bool prefetch = T >= 512;
    // FNL_PROFILE: synchronized wall time per phase (diagnostic; serializes the pass).
    static const bool profile = std::getenv("FNL_PROFILE") != nullptr;
    double phase[6] = {0, 0, 0, 0, 0, 0};
    auto clock = std::chrono::steady_clock::now();
    const double before = std::chrono::duration<double>(clock - entry).count();
    auto tick = [&](int k) {
        if(!profile) return;
        FNL_CUDA(cudaStreamSynchronize(s));
        const auto now = std::chrono::steady_clock::now();
        phase[k] += std::chrono::duration<double>(now - clock).count();
        clock = now;
    };
    for(int l = 0; l < kLayers; ++l) {
        const auto& L = layers_[l];
        if(l == kPleLayer) ple_batch(history, pos0, T);
        tick(0);
        hc_pre_batch(L.ahc, T, true);
        tick(1);
        if(L.attention) attn_batch(L, attn_index(l), pos0, T);
        else gdn_batch(L, gdn_index(l), T);
        tick(L.attention ? 2 : 3);
        hc_combine(pH_, pout_, pinj_, T, s);
        hc_pre_batch(L.mhc, T, true);
        tick(1);
        moe_batch(L, l, T, prefetch);
        tick(4);
        hc_combine(pH_, pout_, pinj_, T, s);
        tick(1);
        if(on_layer) on_layer(l, pH_, T);
        tick(5);
    }
    if(profile)
        std::fprintf(stderr, "prefill %d: ple %.3f hc %.3f attn %.3f gdn %.3f moe %.3f other %.3f s\n", T, phase[0],
                     phase[1], phase[2], phase[3], phase[4], phase[5]);
    // Last position through the decode head (its hidden streams also seed the drafter).
    FNL_CUDA(cudaMemcpyAsync(H_, pH_ + std::size_t(T - 1) * kHyper, kHyper * 4, cudaMemcpyDeviceToDevice, s));
    if(keep_all) {
        // Final mixed hidden of every position stays in px_bf_ for head_rows().
        hc_pre_batch(mixer_, T, false);
    } else if(drafter && mtp_loaded_) {
        mtp_batch(history, pos0, T);
    }
    directory_upload();
    hc_pre(mixer_, H_, 1, nullptr);
    bf16_gemv({lm_head_, x_, logits_, kVocab, kHidden, Post::none}, s);
    argmax(logits_, kVocab, next_, amax_v_, amax_i_, s);
    if(profile) {
        FNL_CUDA(cudaStreamSynchronize(s));
        std::fprintf(stderr, "prefill %d: before-loop %.3f after-loop %.3f s, call total %.3f s\n", T, before,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - entry).count());
    }
}

// The drafter's cache over the chunk: position p takes the token at p + 1 and the hidden streams of p (pH_), so
// positions whose next token is not known yet (the prompt's last) wait for the first decode round.
void Net::mtp_batch(const std::vector<int>& history, int pos0, int T) {
    const int Tm = std::min(T, int(history.size()) - 1 - pos0);
    if(Tm <= 0) return;
    const auto s = stream();
    int* dids = reinterpret_cast<int*>(prows_dev_);
    FNL_CUDA(cudaMemcpyAsync(dids, history.data() + pos0 + 1, std::size_t(Tm) * 4, cudaMemcpyHostToDevice, s));
    {
        Bump b{pscratch_, pscratch_bytes_};
        float* e = b.get<float>(std::size_t(Tm) * kHidden);
        float* en = b.get<float>(std::size_t(Tm) * kHidden);
        __half* enb = b.get<__half>(std::size_t(Tm) * kHidden);
        float* fe = b.get<float>(std::size_t(Tm) * kHidden);
        float* hn = b.get<float>(std::size_t(Tm) * kHyper);
        __half* hnb = b.get<__half>(std::size_t(Tm) * kHyper);
        float* fh = b.get<float>(std::size_t(Tm) * kHyper);
        embed_single(embed_, dids, e, Tm, s);
        rms_rows(e, kHidden, nullptr, mtp_.emb1p, en, Tm, s);
        to_f16(en, enb, std::size_t(Tm) * kHidden, s);
        gemm(WFmt::bf16, Epi::f32, dense_args(enb, kHidden, Tm, kHidden, kHidden, mtp_.fc_emb, nullptr, fe, nullptr,
                                              kHidden), 0, s);
        rms_rows(pH_, kHyper, nullptr, mtp_.hid1p, hn, Tm, s);
        to_f16(hn, hnb, std::size_t(Tm) * kHyper, s);
        gemm(WFmt::bf16, Epi::f32, dense_args(hnb, kHidden, Tm * kStreams, kHidden, kHidden, mtp_.fc_hid, nullptr, fh,
                                              nullptr, kHidden), 0, s);
        mtp_fuse(fe, fh, pH_, Tm, s);
    }
    hc_pre_batch(mtp_.ahc, Tm, true);
    // Only the cache matters for prompt positions: key, value and index projections, then the cache writes.
    Bump b{pscratch_, pscratch_bytes_};
    float* proj = b.get<float>(std::size_t(Tm) * kProjAttn);
    __half* qb = b.get<__half>(std::size_t(Tm) * kAttnOut);
    float* gate = b.get<float>(std::size_t(Tm) * kAttnOut);
    float* iq = b.get<float>(std::size_t(Tm) * kIdxHeads * kIdxDim);
    __half* iq3 = b.get<__half>(std::size_t(Tm) * kIdxHeads * 3 * kIdxDim);
    float* raw = b.get<float>(std::size_t(Tm) * kIdxDim);
    FNL_CUDA(cudaMemsetAsync(proj, 0, std::size_t(Tm) * kProjAttn * 4, s));
    int col = kQProj;
    auto proj_gemm = [&](const __nv_bfloat16* w, int rows) {
        gemm(WFmt::bf16, Epi::f32, dense_args(px_bf_, kHidden, Tm, rows, kHidden, w, nullptr, proj + col, nullptr,
                                              kProjAttn), 0, s);
        col += rows;
    };
    proj_gemm(mtp_.k, kKvHeads * kHeadDim);
    proj_gemm(mtp_.v, kKvHeads * kHeadDim);
    proj_gemm(mtp_.idx, kIdxProj);
    AttnPrepBatch ap{proj, mtp_.qn1p, mtp_.kn1p, mtp_.iqn1p, mtp_.ikn1p, inv_freq_, qb, gate, iq, iq3, raw, mkcache_,
                     mvcache_, mkscale_, mvscale_, mrawkeys_, mblockkeys_, options_.capacity, pos0, Tm};
    attn_prep_batch(ap, s);
    pool_blocks(ap, s);
}

void Net::head_rows(int r0, int n, float* out) {
    gemm(WFmt::bf16, Epi::f32, dense_args(px_bf_ + std::size_t(r0) * kHidden, kHidden, n, kVocab, kHidden, lm_head_,
                                          nullptr, out, nullptr, kVocab), 0, stream());
}

}  // namespace fnl
