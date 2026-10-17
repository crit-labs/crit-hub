#include "layers.cuh"

namespace fnl {

namespace kern {

// Block-wide max of 256 threads; every thread gets the result.
__device__ float block_max256(float v) {
    __shared__ float red[8];
    __shared__ float total;
    v = warp_max(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if(lane == 0) red[warp] = v;
    __syncthreads();
    if(warp == 0) {
        float x = lane < 8 ? red[lane] : 0.f;
        x = warp_max(x);
        if(lane == 0) total = x;
    }
    __syncthreads();
    return total;
}

// Block-wide sum for blockDim.x <= 1024; every thread gets the result.
__device__ float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float total;
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if(lane == 0) red[warp] = v;
    __syncthreads();
    if(warp == 0) {
        float x = lane < int((blockDim.x + 31) >> 5) ? red[lane] : 0.f;
        x = warp_sum(x);
        if(lane == 0) total = x;
    }
    __syncthreads();
    return total;
}

// ---- hyper-connections ----
// grid (kStreams, T), block 512
__global__ void hc_norm_kernel(const float* H, const float* w1p, float* xn) {
    const int s = blockIdx.x;
    const std::size_t base = std::size_t(blockIdx.y) * kHyper + s * kHidden;
    float v[kHidden / 512];
    float ss = 0.f;
#pragma unroll
    for(int j = 0; j < kHidden / 512; ++j) {
        v[j] = H[base + threadIdx.x + j * 512];
        ss = fmaf(v[j], v[j], ss);
    }
    const float r = rsqrtf(block_sum(ss) / float(kHidden) + kEps);
#pragma unroll
    for(int j = 0; j < kHidden / 512; ++j) {
        const int i = threadIdx.x + j * 512;
        xn[base + i] = v[j] * r * w1p[s * kHidden + i];
    }
}

__global__ void hc_mix_kernel(const float* xn, const float* m, float* x, int tokens) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(tokens) * kHidden) return;
    const std::size_t t = idx / kHidden, i = idx % kHidden;
    const float* a = xn + t * kHyper + i;
    const float* b = m + t * kHyper + i;
    float acc = a[0] * b[0];
    acc += a[kHidden] * b[kHidden];
    acc += a[2 * kHidden] * b[2 * kHidden];
    acc += a[3 * kHidden] * b[3 * kHidden];
    x[idx] = acc / float(kStreams);
}

__global__ void hc_combine_kernel(float* H, const float* out, const float* inj, int tokens) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(tokens) * kHyper) return;
    const std::size_t t = idx / kHyper, r = idx % kHyper;
    const int s = int(r / kHidden), i = int(r % kHidden);
    H[idx] = H[idx] + out[t * kHidden + i] * inj[t * kStreams + s];
}

__global__ void embed_kernel(const __nv_bfloat16* table, const int* ids, float* H, int tokens) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(tokens) * kHidden) return;
    const std::size_t t = idx / kHidden, i = idx % kHidden;
    const float v = __bfloat162float(table[std::size_t(ids[t]) * kHidden + i]);
#pragma unroll
    for(int s = 0; s < kStreams; ++s) H[t * kHyper + s * kHidden + i] = v;
}

// ---- gated delta net ----
// grid (80 + 1, T), block 128. Blocks 0..15 q heads, 16..31 k heads, 32..79 v heads, 80 g/beta.
__global__ void gdn_conv_kernel(GdnStep s) {
    const int b = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    if(b == 80) {
        if(i < kVHeads) {
            const float a = s.ab[std::size_t(t) * 2 * kVHeads + i];
            const float bb = s.ab[std::size_t(t) * 2 * kVHeads + kVHeads + i];
            s.g[std::size_t(t) * kVHeads + i] = s.neg_a[i] * softplusf(a + s.dt_bias[i]);
            s.beta[std::size_t(t) * kVHeads + i] = sigmoidf(bb);
        }
        return;
    }
    const int c = b * 128 + i;  // channel
    const float* w = s.conv_w + c * 4;
    // Inputs t-3..t: from this chunk when available, else the state (which holds the 3 inputs before t = 0).
    float in[4];
#pragma unroll
    for(int j = 0; j < 4; ++j) {
        const int tt = t - 3 + j;
        in[j] = tt >= 0 ? s.qkv[std::size_t(tt) * kQkvDim + c] : s.conv_state[c * 3 + (3 + tt)];
    }
    float y = w[0] * in[0];
    y = fmaf(w[1], in[1], y);
    y = fmaf(w[2], in[2], y);
    y = fmaf(w[3], in[3], y);
    y = siluf(y);
    if(b < 32) {
        const float ss = block_sum(y * y);
        y *= rsqrtf(ss + 1e-6f);
        if(b < 16) {
            y *= rsqrtf(float(kKDim));
            s.q[std::size_t(t) * kKHeads * kKDim + c] = y;
        } else {
            s.k[std::size_t(t) * kKHeads * kKDim + (c - kKHeads * kKDim)] = y;
        }
    } else {
        s.v[std::size_t(t) * kVHeads * kVDim + (c - 2 * kKHeads * kKDim)] = y;
    }
}

// After all tokens: state <- last 3 inputs. grid kQkvDim/128.
__global__ void gdn_conv_state_kernel(GdnStep s) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    float in[3];
#pragma unroll
    for(int j = 0; j < 3; ++j) {
        const int tt = s.tokens - 3 + j;
        in[j] = tt >= 0 ? s.qkv[std::size_t(tt) * kQkvDim + c] : s.conv_state[c * 3 + (3 + tt)];
    }
#pragma unroll
    for(int j = 0; j < 3; ++j) s.conv_state[c * 3 + j] = in[j];
}

// grid (kVHeads, kVDim / 32), block 128 (4 warps x 8 columns). Four lanes share one state column
// (head, v): lane quarter q holds rows 32q .. 32q + 31, and the row sums combine over the 4 lanes.
constexpr int kRecurRows = kKDim / 4;
__global__ void __launch_bounds__(128) gdn_recur_kernel(const float* q, const float* k, const float* v,
                                                        const float* g, const float* beta, float* ring,
                                                        const int* cur_p, bool snapshots, float* out, int tokens) {
    const int cur = *cur_p;
    float* state = ring + std::size_t(cur) * kGdnStateFloats;
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.y * 32 + warp * 8 + (lane >> 2);
    const int quarter = lane & 3;
    const int kh = h / (kVHeads / kKHeads);
    __shared__ __align__(16) float qs[4][kKDim], ks[4][kKDim];
    float S[kRecurRows];
    float* st = state + std::size_t(h) * kKDim * kVDim + std::size_t(quarter * kRecurRows) * kVDim + col;
#pragma unroll
    for(int r = 0; r < kRecurRows; ++r) S[r] = st[r * kVDim];
    const float* qw = qs[warp] + quarter * kRecurRows;
    const float* kw = ks[warp] + quarter * kRecurRows;
    for(int t = 0; t < tokens; ++t) {
        const float4 qv = reinterpret_cast<const float4*>(q + (std::size_t(t) * kKHeads + kh) * kKDim)[lane];
        const float4 kv = reinterpret_cast<const float4*>(k + (std::size_t(t) * kKHeads + kh) * kKDim)[lane];
        __syncwarp();
        reinterpret_cast<float4*>(qs[warp])[lane] = qv;
        reinterpret_cast<float4*>(ks[warp])[lane] = kv;
        __syncwarp();
        const float decay = expf(g[std::size_t(t) * kVHeads + h]);
        const float b = beta[std::size_t(t) * kVHeads + h];
        const float vv = v[(std::size_t(t) * kVHeads + h) * kVDim + col];
        float mem = 0.f;
#pragma unroll
        for(int r = 0; r < kRecurRows; ++r) {
            S[r] *= decay;
            mem = fmaf(S[r], kw[r], mem);
        }
        mem += __shfl_xor_sync(0xffffffffu, mem, 1);
        mem += __shfl_xor_sync(0xffffffffu, mem, 2);
        const float delta = (vv - mem) * b;
        float o = 0.f;
#pragma unroll
        for(int r = 0; r < kRecurRows; ++r) {
            S[r] = fmaf(kw[r], delta, S[r]);
            o = fmaf(S[r], qw[r], o);
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if(quarter == 0) out[(std::size_t(t) * kVHeads + h) * kVDim + col] = o;
        if(snapshots) {
            float* sn = st + std::ptrdiff_t((cur + 1 + t) % kGdnRing - cur) * std::ptrdiff_t(kGdnStateFloats);
#pragma unroll
            for(int r = 0; r < kRecurRows; ++r) sn[r * kVDim] = S[r];
        }
    }
    if(!snapshots) {
#pragma unroll
        for(int r = 0; r < kRecurRows; ++r) st[r * kVDim] = S[r];
    }
}

// grid (kQkvDim / 128 + kHyper / 128, layers): conv states of the kept tokens; block (0, 0) also advances cur.
__global__ void commit_kernel(Commit c) {
    const int n = *c.n;
    const int b = blockIdx.x, i = threadIdx.x;
    if(b < kQkvDim / 128) {
        const int ch = b * 128 + i, l = blockIdx.y;
        float* st = c.conv_states + (std::size_t(l) * kQkvDim + ch) * 3;
        const float* in = c.inputs + std::size_t(l) * kMaxTokens * kQkvDim + ch;
        float v[3];
#pragma unroll
        for(int j = 0; j < 3; ++j) {
            const int tt = n - 3 + j;
            v[j] = tt >= 0 ? in[std::size_t(tt) * kQkvDim] : st[3 + tt];
        }
#pragma unroll
        for(int j = 0; j < 3; ++j) st[j] = v[j];
        return;
    }
    if(blockIdx.y != 0) return;
    const int ch = (b - kQkvDim / 128) * 128 + i;
    if(c.ple_state) {
        float* st = c.ple_state + std::size_t(ch) * kPleState;
        float v[kPleState];
#pragma unroll
        for(int j = 0; j < kPleState; ++j) {
            const int tt = n - kPleState + j;
            v[j] = tt >= 0 ? c.ple_gvn[std::size_t(tt) * kHyper + ch] : st[kPleState + tt];
        }
#pragma unroll
        for(int j = 0; j < kPleState; ++j) st[j] = v[j];
    }
    if(ch == 0) *c.cur = (*c.cur + n) % kGdnRing;
}

// grid (kVHeads, T), block 128
__global__ void gdn_gate_norm_kernel(const float* o, const float* z, const float* w, float* out) {
    const std::size_t base = (std::size_t(blockIdx.y) * kVHeads + blockIdx.x) * kVDim;
    const float x = o[base + threadIdx.x];
    const float r = rsqrtf(block_sum(x * x) / float(kVDim) + kEps);
    out[base + threadIdx.x] = w[threadIdx.x] * (x * r) * sigmoidf(z[base + threadIdx.x]);
}

// ---- attention ----
__device__ __forceinline__ void rope_pair(float& a, float& b, int pos, float inv) {
    // a = x[i], b = x[i + 32]; out[i] = a cos - b sin, out[i + 32] = b cos + a sin
    float sn, cs;
    sincosf(float(pos) * inv, &sn, &cs);
    const float na = a * cs - b * sn;
    const float nb = b * cs + a * sn;
    a = na;
    b = nb;
}

// grid (33, T), block 256. 0..23 q heads, 24..25 k heads, 26..27 v heads, 28..31 index q heads, 32 index key.
__global__ void attn_prep_kernel(AttnPrep a) {
    __shared__ float buf[256];
    const int b = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = a.step[t].pos;
    const float* qp = a.proj + std::size_t(t) * a.ldp;
    const float* kp = qp + kQProj;
    const float* vp = kp + kKvHeads * kHeadDim;
    const float* ip = vp + kKvHeads * kHeadDim;
    if(b < kQHeads + kKvHeads) {
        const bool isq = b < kQHeads;
        const float x = isq ? qp[b * 512 + i] : kp[(b - kQHeads) * kHeadDim + i];
        const float r = rsqrtf(block_sum(x * x) / float(kHeadDim) + kEps);
        buf[i] = x * r * (isq ? a.qn1p[i] : a.kn1p[i]);
        __syncthreads();
        if(i < kRotary / 2) {
            float x0 = buf[i], x1 = buf[i + kRotary / 2];
            rope_pair(x0, x1, pos, a.inv_freq[i]);
            buf[i] = x0;
            buf[i + kRotary / 2] = x1;
        }
        __syncthreads();
        if(isq) {
            a.q[(std::size_t(t) * kQHeads + b) * kHeadDim + i] = buf[i];
            a.gate[(std::size_t(t) * kQHeads + b) * kHeadDim + i] = qp[b * 512 + 256 + i];
        } else {
            const int h = b - kQHeads;
            const float amax = block_max256(fabsf(buf[i]));
            const float scale = amax > 0.f ? amax / 448.f : 1.f;
            a.kcache[(std::size_t(h) * a.capacity + pos) * kHeadDim + i] = kv_e4m3(buf[i], scale);
            if(i == 0) a.kscale[std::size_t(h) * a.capacity + pos] = scale;
        }
        return;
    }
    if(b < kQHeads + 2 * kKvHeads) {
        const int h = b - kQHeads - kKvHeads;
        const float x = vp[h * kHeadDim + i];
        const float amax = block_max256(fabsf(x));
        const float scale = amax > 0.f ? amax / 448.f : 1.f;
        a.vcache[(std::size_t(h) * a.capacity + pos) * kHeadDim + i] = kv_e4m3(x, scale);
        if(i == 0) a.vscale[std::size_t(h) * a.capacity + pos] = scale;
        return;
    }
    if(b < kQHeads + 2 * kKvHeads + kIdxHeads) {
        const int h = b - kQHeads - 2 * kKvHeads;
        const float x = i < kIdxDim ? ip[h * kIdxDim + i] : 0.f;
        const float r = rsqrtf(block_sum(x * x) / float(kIdxDim) + kEps);
        if(i < kIdxDim) buf[i] = x * r * a.iqn1p[i];
        __syncthreads();
        if(i < kRotary / 2) {
            float x0 = buf[i], x1 = buf[i + kRotary / 2];
            rope_pair(x0, x1, pos, a.inv_freq[i]);
            buf[i] = x0;
            buf[i + kRotary / 2] = x1;
        }
        __syncthreads();
        if(i < kIdxDim) a.iq[(std::size_t(t) * kIdxHeads + h) * kIdxDim + i] = buf[i];
        return;
    }
    // Index key: the raw key into the ring; pooling runs after every token of the pass stored its key.
    if(i < kIdxDim) a.rawkeys[(pos % kRawRing) * kIdxDim + i] = ip[kIdxHeads * kIdxDim + i];
}

// grid T, block 128: pool the index block a token completes.
__global__ void attn_pool_kernel(AttnPrep a) {
    __shared__ float buf[kIdxDim];
    const int pos = a.step[blockIdx.x].pos, i = threadIdx.x;
    if(pos % kBlockTokens != kBlockTokens - 1) return;
    const int start = pos - (kBlockTokens - 1);
    float s = a.rawkeys[(start % kRawRing) * kIdxDim + i];
    s += a.rawkeys[((start + 1) % kRawRing) * kIdxDim + i];
    s += a.rawkeys[((start + 2) % kRawRing) * kIdxDim + i];
    s += a.rawkeys[((start + 3) % kRawRing) * kIdxDim + i];
    const float m = s / float(kBlockTokens);
    const float r = rsqrtf(block_sum(m * m) / float(kIdxDim) + kEps);
    buf[i] = m * r * a.ikn1p[i];
    __syncthreads();
    if(i < kRotary / 2) {
        float x0 = buf[i], x1 = buf[i + kRotary / 2];
        rope_pair(x0, x1, start, a.inv_freq[i]);
        buf[i] = x0;
        buf[i + kRotary / 2] = x1;
    }
    __syncthreads();
    a.blockkeys[std::size_t(pos / kBlockTokens) * kIdxDim + i] = buf[i];
}

// One warp per block key.
__global__ void idx_scores_kernel(const float* iq, const float* bk, const Step* step, float* scores, int ld) {
    const int t = blockIdx.y;
    if(!step[t].sparse) return;
    const int n = step[t].nb;
    iq += std::size_t(t) * kIdxHeads * kIdxDim;
    scores += std::size_t(t) * ld;
    __shared__ __align__(16) float qs[kIdxHeads * kIdxDim];
    for(int j = threadIdx.x; j < kIdxHeads * kIdxDim; j += blockDim.x) qs[j] = iq[j];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * (blockDim.x >> 5) + warp;
    if(b >= n) return;
    const float4 k = reinterpret_cast<const float4*>(bk + std::size_t(b) * kIdxDim)[lane];
    float total = 0.f;
#pragma unroll
    for(int h = 0; h < kIdxHeads; ++h) {
        const float4 q = reinterpret_cast<const float4*>(qs + h * kIdxDim)[lane];
        float d = q.x * k.x;
        d = fmaf(q.y, k.y, d);
        d = fmaf(q.z, k.z, d);
        d = fmaf(q.w, k.w, d);
        d = warp_sum(d);
        total += fmaxf(d, 0.f);
    }
    if(lane == 0) scores[b] = total / sqrtf(float(kIdxDim));
}

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block of 1024 threads: radix select of the kTopBlocks largest, then an ascending compaction.
__global__ void __launch_bounds__(1024) topk_blocks_kernel(const float* scores, int ld, const Step* step, int* sel) {
    const int t = blockIdx.x;
    if(!step[t].sparse) return;
    const int n = step[t].nb;
    scores += std::size_t(t) * ld;
    sel += std::size_t(t) * kTopBlocks;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t prefix, prefix_mask, want, greater;
    __shared__ int scan[1024];
    if(threadIdx.x == 0) {
        prefix = 0;
        prefix_mask = 0;
        want = kTopBlocks;
        greater = 0;
    }
    for(int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for(int j = threadIdx.x; j < 256; j += blockDim.x) hist[j] = 0;
        __syncthreads();
        const uint32_t p = prefix, pm = prefix_mask;
        for(int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t key = order_key(scores[i]);
            if((key & pm) == p) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            uint32_t cum = 0;
            int bin = 255;
            for(; bin > 0; --bin) {
                if(cum + hist[bin] >= want) break;
                cum += hist[bin];
            }
            want -= cum;
            greater += cum;
            prefix |= uint32_t(bin) << shift;
            prefix_mask |= 255u << shift;
        }
        __syncthreads();
    }
    // Every key > threshold is selected; `want` of the keys == threshold, lowest indices first.
    const uint32_t thr = prefix;
    const int ties = int(want);
    int written = 0, ties_seen = 0;
    for(int base = 0; base < n; base += blockDim.x) {
        const int i = base + threadIdx.x;
        int gt = 0, eq = 0;
        if(i < n) {
            const uint32_t key = order_key(scores[i]);
            gt = key > thr;
            eq = key == thr;
        }
        // Exclusive scan of eq for tie order, then of the final pick.
        scan[threadIdx.x] = eq;
        __syncthreads();
        for(int o = 1; o < 1024; o <<= 1) {
            const int v = threadIdx.x >= o ? scan[threadIdx.x - o] : 0;
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }
        const int eq_before = scan[threadIdx.x] - eq + ties_seen;
        const int pick = gt || (eq && eq_before < ties);
        const int eq_total = scan[1023];
        __syncthreads();
        scan[threadIdx.x] = pick;
        __syncthreads();
        for(int o = 1; o < 1024; o <<= 1) {
            const int v = threadIdx.x >= o ? scan[threadIdx.x - o] : 0;
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }
        if(pick) sel[written + scan[threadIdx.x] - 1] = i;
        written += scan[1023];
        ties_seen += eq_total;
        __syncthreads();
    }
}

// Split-K decode attention. grid (kKvHeads, splits), block 256 (8 warps), 64 keys per split.
constexpr int kSplitKeys = 64;
__device__ __forceinline__ int key_token(const AttnDecode& a, const Step& st, int j) {
    const int nsel_tokens = st.nsel * kBlockTokens;
    if(j < nsel_tokens) {
        const int blk = st.sparse ? a.sel[j / kBlockTokens] : j / kBlockTokens;
        return blk * kBlockTokens + j % kBlockTokens;
    }
    return st.tail_start + (j - nsel_tokens);
}

__global__ void __launch_bounds__(256) attn_decode_kernel(AttnDecode a) {
    const int t = blockIdx.z;
    const Step st = a.step[t];
    a.q += std::size_t(t) * kAttnOut;
    a.sel += std::size_t(t) * kTopBlocks;
    a.partial += std::size_t(t) * kPartialStride;
    const int nkeys = st.nkeys;
    const int g = blockIdx.x, split = blockIdx.y;
    if(split * kSplitKeys >= nkeys) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ __align__(16) float qs[kGroup][kHeadDim];
    __shared__ float sc[kGroup][kSplitKeys];
    __shared__ int tok[kSplitKeys];
    for(int j = threadIdx.x; j < kGroup * kHeadDim; j += 256) qs[j / kHeadDim][j % kHeadDim] = a.q[g * kGroup * kHeadDim + j];
    const int k0 = split * kSplitKeys;
    const int cnt = min(kSplitKeys, nkeys - k0);
    if(threadIdx.x < cnt) tok[threadIdx.x] = key_token(a, st, k0 + threadIdx.x);
    __syncthreads();
    const uint8_t* K = a.kcache + std::size_t(g) * a.capacity * kHeadDim;
    const uint8_t* V = a.vcache + std::size_t(g) * a.capacity * kHeadDim;
    const float* KS = a.kscale + std::size_t(g) * a.capacity;
    const float* VS = a.vscale + std::size_t(g) * a.capacity;
    __shared__ float vsc[kSplitKeys];
    if(threadIdx.x < cnt) vsc[threadIdx.x] = VS[tok[threadIdx.x]];
    // Scores: each warp handles keys warp, warp + 8, ...; lane covers 8 dims.
    for(int j = warp; j < cnt; j += 8) {
        const uint2 kv = reinterpret_cast<const uint2*>(K + std::size_t(tok[j]) * kHeadDim)[lane];
        const float ks = KS[tok[j]];
        float kf[8];
        {
            const uint32_t words[2] = {kv.x, kv.y};
#pragma unroll
            for(int u = 0; u < 2; ++u) {
                const float2 lo = e4m3x2f(words[u] & 0xffff), hi = e4m3x2f(words[u] >> 16);
                kf[4 * u] = lo.x * ks;
                kf[4 * u + 1] = lo.y * ks;
                kf[4 * u + 2] = hi.x * ks;
                kf[4 * u + 3] = hi.y * ks;
            }
        }
#pragma unroll
        for(int h = 0; h < kGroup; ++h) {
            const float* q = qs[h] + lane * 8;
            float d = 0.f;
#pragma unroll
            for(int u = 0; u < 8; ++u) d = fmaf(q[u], kf[u], d);
            d = warp_sum(d);
            if(lane == 0) sc[h][j] = d * (1.f / 16.f);
        }
    }
    __syncthreads();
    // Per head max and exp (warp h for heads h and h + 8).
    __shared__ float mx[kGroup], sum[kGroup];
    for(int h = warp; h < kGroup; h += 8) {
        float m = -INFINITY;
        for(int j = lane; j < cnt; j += 32) m = fmaxf(m, sc[h][j]);
        m = warp_max(m);
        float s = 0.f;
        for(int j = lane; j < cnt; j += 32) {
            const float p = expf(sc[h][j] - m);
            sc[h][j] = p;
            s += p;
        }
        s = warp_sum(s);
        if(lane == 0) {
            mx[h] = m;
            sum[h] = s;
        }
    }
    __syncthreads();
    // Thread d accumulates dim d for all heads.
    const int d = threadIdx.x;
    float o[kGroup];
#pragma unroll
    for(int h = 0; h < kGroup; ++h) o[h] = 0.f;
    for(int j = 0; j < cnt; ++j) {
        const float v = e4m3f(V[std::size_t(tok[j]) * kHeadDim + d]) * vsc[j];
#pragma unroll
        for(int h = 0; h < kGroup; ++h) o[h] = fmaf(sc[h][j], v, o[h]);
    }
    // partial layout: [g][split][kGroup] m, l, then [g][split][kGroup][256] o
    const int splits = kMaxSplits;
    float* pm = a.partial;
    float* pl = pm + kKvHeads * splits * kGroup;
    float* po = pl + kKvHeads * splits * kGroup;
    if(threadIdx.x < kGroup) {
        pm[(g * splits + split) * kGroup + threadIdx.x] = mx[threadIdx.x];
        pl[(g * splits + split) * kGroup + threadIdx.x] = sum[threadIdx.x];
    }
#pragma unroll
    for(int h = 0; h < kGroup; ++h) po[((std::size_t(g) * splits + split) * kGroup + h) * kHeadDim + d] = o[h];
}

// grid (kQHeads), block 256: merge splits, apply the output gate.
__global__ void attn_merge_kernel(AttnDecode a) {
    const int t = blockIdx.y;
    a.partial += std::size_t(t) * kPartialStride;
    a.gate += std::size_t(t) * kAttnOut;
    a.out += std::size_t(t) * kAttnOut;
    const int used = (a.step[t].nkeys + kSplitKeys - 1) / kSplitKeys;
    constexpr int splits = kMaxSplits;
    const int qh = blockIdx.x, g = qh / kGroup, h = qh % kGroup, d = threadIdx.x;
    const float* pm = a.partial;
    const float* pl = pm + kKvHeads * splits * kGroup;
    const float* po = pl + kKvHeads * splits * kGroup;
    float M = -INFINITY;
    for(int s = 0; s < used; ++s) M = fmaxf(M, pm[(g * splits + s) * kGroup + h]);
    float L = 0.f, O = 0.f;
    for(int s = 0; s < used; ++s) {
        const float w = expf(pm[(g * splits + s) * kGroup + h] - M);
        L = fmaf(w, pl[(g * splits + s) * kGroup + h], L);
        O = fmaf(w, po[((std::size_t(g) * splits + s) * kGroup + h) * kHeadDim + d], O);
    }
    const float val = O / L;
    a.out[qh * kHeadDim + d] = val * sigmoidf(a.gate[qh * kHeadDim + d]);
}

__global__ void silu_mul_kernel(const float* g, const float* u, float* act, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) act[i] = siluf(g[i]) * u[i];
}

// rows: [T][kPleHeads][90] (80 packed bytes, 10 E4M3 scales). grid T, block 256.
__global__ void ple_dequant_kernel(const uint8_t* rows, float s2, float* e) {
    const uint8_t* r = rows + std::size_t(blockIdx.x) * kPleHeads * 90;
    for(int j = threadIdx.x; j < kPleHeads * 80; j += blockDim.x) {
        const int h = j / 80, b = j % 80;
        const uint8_t* row = r + h * 90;
        const float sc = e4m3f(row[80 + b / 8]) * s2;
        const float2 v = e2m1x2f(row[b]);
        e[std::size_t(blockIdx.x) * kHidden + h * kPleDim + 2 * b] = v.x * sc;
        e[std::size_t(blockIdx.x) * kHidden + h * kPleDim + 2 * b + 1] = v.y * sc;
    }
}

// grid (kStreams, T), block 512: gated values and their normalized form.
__global__ void ple_gate_kernel(PleMix p, float* gv, float* gvn) {
    const int s = blockIdx.x, t = blockIdx.y;
    const std::size_t hb = std::size_t(t) * kHyper + s * kHidden;
    constexpr int per = kHidden / 512;
    float kk[per], qq[per];
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for(int j = 0; j < per; ++j) {
        const int i = threadIdx.x + j * 512;
        kk[j] = p.key[hb + i];
        qq[j] = p.H[hb + i];
        sk = fmaf(kk[j], kk[j], sk);
        sq = fmaf(qq[j], qq[j], sq);
    }
    const float rk = rsqrtf(block_sum(sk) / float(kHidden) + kEps);
    const float rq = rsqrtf(block_sum(sq) / float(kHidden) + kEps);
    float dot = 0.f;
#pragma unroll
    for(int j = 0; j < per; ++j) {
        const int i = threadIdx.x + j * 512;
        dot = fmaf(kk[j] * rk * p.nk1p[s * kHidden + i], qq[j] * rq * p.nq1p[s * kHidden + i], dot);
    }
    float gate = block_sum(dot) / sqrtf(float(kHidden));
    const float sgn = gate > 0.f ? 1.f : gate < 0.f ? -1.f : 0.f;
    gate = sqrtf(fmaxf(fabsf(gate), 1e-6f)) * sgn;
    const float sg = sigmoidf(gate);
    float vv[per];
    float sv = 0.f;
#pragma unroll
    for(int j = 0; j < per; ++j) {
        vv[j] = sg * p.value[std::size_t(t) * kHidden + threadIdx.x + j * 512];
        sv = fmaf(vv[j], vv[j], sv);
    }
    const float rv = rsqrtf(block_sum(sv) / float(kHidden) + kEps);
#pragma unroll
    for(int j = 0; j < per; ++j) {
        const int i = threadIdx.x + j * 512;
        gv[hb + i] = vv[j];
        gvn[hb + i] = vv[j] * rv * p.nc1p[s * kHidden + i];
    }
}

// H += gv + silu(dilated conv of gvn). grid ceil(T*kHyper/256).
__global__ void ple_conv_kernel(PleMix p, const float* gv, const float* gvn) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(p.tokens) * kHyper) return;
    const int t = int(idx / kHyper), c = int(idx % kHyper);
    const float* w = p.conv_w + c * 4;
    float y = 0.f;
#pragma unroll
    for(int j = 0; j < kPleConv; ++j) {
        const int tt = t - (kPleConv - 1 - j) * kPleDilation;
        const float x = tt >= 0 ? gvn[std::size_t(tt) * kHyper + c] : p.conv_state[c * kPleState + (kPleState + tt)];
        y = j == 0 ? w[0] * x : fmaf(w[j], x, y);
    }
    p.H[idx] = p.H[idx] + (gv[idx] + siluf(y));
}

__global__ void ple_state_kernel(PleMix p, const float* gvn) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= kHyper) return;
    float in[kPleState];
#pragma unroll
    for(int j = 0; j < kPleState; ++j) {
        const int tt = p.tokens - kPleState + j;
        in[j] = tt >= 0 ? gvn[std::size_t(tt) * kHyper + c] : p.conv_state[c * kPleState + (kPleState + tt)];
    }
#pragma unroll
    for(int j = 0; j < kPleState; ++j) p.conv_state[c * kPleState + j] = in[j];
}

__global__ void argmax_partial_kernel(const float* x, int n, float* bv, int* bi, std::size_t ld) {
    x += blockIdx.y * ld;
    bv += blockIdx.y * gridDim.x;
    bi += blockIdx.y * gridDim.x;
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float y = x[i];
        if(y > v || (y == v && i < idx)) { v = y; idx = i; }
    }
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if(ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if(threadIdx.x < 32) {
        v = threadIdx.x < (blockDim.x >> 5) ? sv[threadIdx.x] : -INFINITY;
        idx = threadIdx.x < (blockDim.x >> 5) ? si[threadIdx.x] : 0x7fffffff;
#pragma unroll
        for(int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if(ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if(threadIdx.x == 0) { bv[blockIdx.x] = v; bi[blockIdx.x] = idx; }
    }
}

// One block of 256 over the 256 partial results.
__global__ void argmax_final_kernel(const float* bv, const int* bi, int* out) {
    bv += blockIdx.x * 256;
    bi += blockIdx.x * 256;
    out += blockIdx.x;
    float v = bv[threadIdx.x];
    int idx = bi[threadIdx.x];
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if(ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int w = 1; w < 8; ++w)
            if(sv[w] > v || (sv[w] == v && si[w] < idx)) { v = sv[w]; idx = si[w]; }
        v = sv[0];
        idx = si[0];
        for(int w = 1; w < 8; ++w)
            if(sv[w] > v || (sv[w] == v && si[w] < idx)) { v = sv[w]; idx = si[w]; }
        *out = idx;
    }
}

}  // namespace kern
using namespace kern;

void hc_norm(const float* H, const float* w1p, float* xn, int tokens, cudaStream_t stream) {
    hc_norm_kernel<<<dim3(kStreams, tokens), 512, 0, stream>>>(H, w1p, xn);
    FNL_CUDA(cudaGetLastError());
}
void hc_mix(const float* xn, const float* m, float* x, int tokens, cudaStream_t stream) {
    const std::size_t n = std::size_t(tokens) * kHidden;
    hc_mix_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(xn, m, x, tokens);
    FNL_CUDA(cudaGetLastError());
}
void hc_combine(float* H, const float* out, const float* inj, int tokens, cudaStream_t stream) {
    const std::size_t n = std::size_t(tokens) * kHyper;
    hc_combine_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(H, out, inj, tokens);
    FNL_CUDA(cudaGetLastError());
}
void embed_rows(const __nv_bfloat16* table, const int* ids, float* H, int tokens, cudaStream_t stream) {
    const std::size_t n = std::size_t(tokens) * kHidden;
    embed_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(table, ids, H, tokens);
    FNL_CUDA(cudaGetLastError());
}
void gdn_conv(const GdnStep& s, cudaStream_t stream) {
    gdn_conv_kernel<<<dim3(81, s.tokens), 128, 0, stream>>>(s);
    FNL_CUDA(cudaGetLastError());
    if(!s.update_state) return;
    gdn_conv_state_kernel<<<kQkvDim / 128, 128, 0, stream>>>(s);
    FNL_CUDA(cudaGetLastError());
}
void gdn_recur(const float* q, const float* k, const float* v, const float* g, const float* beta, float* ring,
               const int* cur, bool snapshots, float* out, int tokens, cudaStream_t stream) {
    if(snapshots && tokens > kGdnRing) throw std::invalid_argument("gdn_recur snapshots cover at most 4 tokens");
    gdn_recur_kernel<<<dim3(kVHeads, kVDim / 32), 128, 0, stream>>>(q, k, v, g, beta, ring, cur, snapshots, out,
                                                                    tokens);
    FNL_CUDA(cudaGetLastError());
}
void commit_states(const Commit& c, cudaStream_t stream) {
    commit_kernel<<<dim3(kQkvDim / 128 + kHyper / 128, c.layers), 128, 0, stream>>>(c);
    FNL_CUDA(cudaGetLastError());
}
void gdn_gate_norm(const float* o, const float* z, const float* w, float* out, int tokens, cudaStream_t stream) {
    gdn_gate_norm_kernel<<<dim3(kVHeads, tokens), 128, 0, stream>>>(o, z, w, out);
    FNL_CUDA(cudaGetLastError());
}
void attn_prep(const AttnPrep& a, cudaStream_t stream) {
    attn_prep_kernel<<<dim3(kQHeads + 2 * kKvHeads + kIdxHeads + 1, a.tokens), 256, 0, stream>>>(a);
    FNL_CUDA(cudaGetLastError());
    attn_pool_kernel<<<a.tokens, kIdxDim, 0, stream>>>(a);
    FNL_CUDA(cudaGetLastError());
}
void idx_scores(const float* iq, const float* blockkeys, const Step* step, int max_blocks, float* scores, int ld,
                int tokens, cudaStream_t stream) {
    idx_scores_kernel<<<dim3((max_blocks + 7) / 8, tokens), 256, 0, stream>>>(iq, blockkeys, step, scores, ld);
    FNL_CUDA(cudaGetLastError());
}
void topk_blocks(const float* scores, int ld, const Step* step, int* sel, int tokens, cudaStream_t stream) {
    topk_blocks_kernel<<<tokens, 1024, 0, stream>>>(scores, ld, step, sel);
    FNL_CUDA(cudaGetLastError());
}
void attn_decode(const AttnDecode& a, cudaStream_t stream) {
    attn_decode_kernel<<<dim3(kKvHeads, kMaxSplits, a.tokens), 256, 0, stream>>>(a);
    FNL_CUDA(cudaGetLastError());
    attn_merge_kernel<<<dim3(kQHeads, a.tokens), 256, 0, stream>>>(a);
    FNL_CUDA(cudaGetLastError());
}
void silu_mul(const float* g, const float* u, float* act, int n, cudaStream_t stream) {
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(g, u, act, n);
    FNL_CUDA(cudaGetLastError());
}
void ple_dequant(const uint8_t* rows, float s2, float* e, int tokens, cudaStream_t stream) {
    ple_dequant_kernel<<<tokens, 256, 0, stream>>>(rows, s2, e);
    FNL_CUDA(cudaGetLastError());
}
void ple_mix(const PleMix& p, float* gv, float* gvn, cudaStream_t stream, bool update_state) {
    ple_gate_kernel<<<dim3(kStreams, p.tokens), 512, 0, stream>>>(p, gv, gvn);
    FNL_CUDA(cudaGetLastError());
    const std::size_t n = std::size_t(p.tokens) * kHyper;
    ple_conv_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(p, gv, gvn);
    FNL_CUDA(cudaGetLastError());
    if(!update_state) return;
    ple_state_kernel<<<(kHyper + 255) / 256, 256, 0, stream>>>(p, gvn);
    FNL_CUDA(cudaGetLastError());
}
void argmax(const float* logits, int n, int* out, float* scratch, int* scratch_idx, cudaStream_t stream, int rows,
            std::size_t ld) {
    argmax_partial_kernel<<<dim3(256, rows), 256, 0, stream>>>(logits, n, scratch, scratch_idx, ld ? ld : n);
    FNL_CUDA(cudaGetLastError());
    argmax_final_kernel<<<rows, 256, 0, stream>>>(scratch, scratch_idx, out);
    FNL_CUDA(cudaGetLastError());
}

namespace kern {
// grid count, block 512.
__global__ void rms_rows_kernel(const float* x, int width, const int* rows, const float* w1p, float* y) {
    const int r = blockIdx.x;
    const float* in = x + std::size_t(rows ? rows[r] : r) * width;
    float ss = 0.f;
    for(int i = threadIdx.x; i < width; i += blockDim.x) ss = fmaf(in[i], in[i], ss);
    const float rs = rsqrtf(block_sum(ss) / float(width) + kEps);
    for(int i = threadIdx.x; i < width; i += blockDim.x) y[std::size_t(r) * width + i] = in[i] * rs * w1p[i];
}
__global__ void embed_single_kernel(const __nv_bfloat16* table, const int* ids, float* out, int count) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(count) * kHidden) return;
    const std::size_t t = idx / kHidden, i = idx % kHidden;
    out[idx] = __bfloat162float(table[std::size_t(ids[t]) * kHidden + i]);
}
__global__ void mtp_fuse_kernel(const float* fe, const float* fh, float* H, int count) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(count) * kHyper) return;
    const std::size_t t = idx / kHyper, i = idx % kHidden;
    H[idx] = fe[t * kHidden + i] + fh[idx];
}
}  // namespace kern

void rms_rows(const float* x, int width, const int* rows, const float* w1p, float* y, int count, cudaStream_t stream) {
    if(count <= 0) return;
    rms_rows_kernel<<<count, 512, 0, stream>>>(x, width, rows, w1p, y);
    FNL_CUDA(cudaGetLastError());
}
void embed_single(const __nv_bfloat16* table, const int* ids, float* out, int count, cudaStream_t stream) {
    const std::size_t n = std::size_t(count) * kHidden;
    embed_single_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(table, ids, out, count);
    FNL_CUDA(cudaGetLastError());
}
void mtp_fuse(const float* fe, const float* fh, float* H, int count, cudaStream_t stream) {
    const std::size_t n = std::size_t(count) * kHyper;
    mtp_fuse_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(fe, fh, H, count);
    FNL_CUDA(cudaGetLastError());
}

}  // namespace fnl
