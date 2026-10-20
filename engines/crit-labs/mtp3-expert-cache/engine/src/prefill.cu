#include "prefill.cuh"

namespace fnl {

namespace kern {

__device__ float block_max256p(float v) {
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
__device__ float block_sum256(float v) {
    __shared__ float red[8];
    __shared__ float total;
    v = warp_sum(v);
    __syncthreads();
    if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if(threadIdx.x < 32) {
        float x = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        x = warp_sum(x);
        if(threadIdx.x == 0) total = x;
    }
    __syncthreads();
    return total;
}

__global__ void to_f16_kernel(const float* x, __half* y, std::size_t n) {
    const std::size_t i = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if(i + 3 < n) {
        const float4 v = *reinterpret_cast<const float4*>(x + i);
        *reinterpret_cast<__half2*>(y + i) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(y + i + 2) = __floats2half2_rn(v.z, v.w);
    } else {
        for(std::size_t j = i; j < n; ++j) y[j] = __float2half_rn(x[j]);
    }
}

__global__ void to_bf16_kernel(const float* x, __nv_bfloat16* y, std::size_t n) {
    const std::size_t i = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if(i + 3 < n) {
        const float4 v = *reinterpret_cast<const float4*>(x + i);
        __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
        *reinterpret_cast<__nv_bfloat162*>(y + i) = a;
        *reinterpret_cast<__nv_bfloat162*>(y + i + 2) = b;
    } else {
        for(std::size_t j = i; j < n; ++j) y[j] = __float2bfloat16(x[j]);
    }
}

// grid T, block 128 (4 warps = 4 streams' inject rows)
__global__ void hc_inject_kernel(const float* xn, const __nv_bfloat16* w, float* inj) {
    const int t = blockIdx.x, s = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* x = xn + std::size_t(t) * kHyper;
    const uint4* wr = reinterpret_cast<const uint4*>(w + std::size_t(s) * kHyper);
    float acc = 0.f;
    for(int c = lane; c < kHyper / 8; c += 32) {
        const uint4 v = __ldg(wr + c);
        const float4 a = *reinterpret_cast<const float4*>(x + c * 8);
        const float4 b = *reinterpret_cast<const float4*>(x + c * 8 + 4);
        acc = fmaf(bf16_lo(v.x), a.x, acc);
        acc = fmaf(bf16_hi(v.x), a.y, acc);
        acc = fmaf(bf16_lo(v.y), a.z, acc);
        acc = fmaf(bf16_hi(v.y), a.w, acc);
        acc = fmaf(bf16_lo(v.z), b.x, acc);
        acc = fmaf(bf16_hi(v.z), b.y, acc);
        acc = fmaf(bf16_lo(v.w), b.z, acc);
        acc = fmaf(bf16_hi(v.w), b.w, acc);
    }
    acc = warp_sum(acc);
    if(lane == 0) inj[t * kStreams + s] = 2.f * sigmoidf(acc / float(kStreams));
}

// grid T, block 32
__global__ void shared_gate_kernel(const float* x, const __nv_bfloat16* w, float* sg) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* xr = x + std::size_t(t) * kHidden;
    const uint4* wr = reinterpret_cast<const uint4*>(w);
    float acc = 0.f;
    for(int c = lane; c < kHidden / 8; c += 32) {
        const uint4 v = __ldg(wr + c);
        const float* a = xr + c * 8;
        acc = fmaf(bf16_lo(v.x), a[0], acc);
        acc = fmaf(bf16_hi(v.x), a[1], acc);
        acc = fmaf(bf16_lo(v.y), a[2], acc);
        acc = fmaf(bf16_hi(v.y), a[3], acc);
        acc = fmaf(bf16_lo(v.z), a[4], acc);
        acc = fmaf(bf16_hi(v.z), a[5], acc);
        acc = fmaf(bf16_lo(v.w), a[6], acc);
        acc = fmaf(bf16_hi(v.w), a[7], acc);
    }
    acc = warp_sum(acc);
    if(lane == 0) sg[t] = sigmoidf(acc);
}

__global__ void count_routes_kernel(const Route* routes, int tokens, int sub, int* counts) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= tokens) return;
    const int s = t / sub;
#pragma unroll
    for(int k = 0; k < kTopK; ++k) atomicAdd(&counts[s * kExperts + routes[t].expert[k]], 1);
}

__global__ void scatter_routes_kernel(const Route* routes, int t0, int t1, const int* offsets, int* cursor,
                                      int* route_token, int* pos_of, const int* cpu_row) {
    const int t = t0 + blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= t1) return;
#pragma unroll
    for(int k = 0; k < kTopK; ++k) {
        if(cpu_row && cpu_row[t * kTopK + k] >= 0) {
            pos_of[t * kTopK + k] = -1 - cpu_row[t * kTopK + k];   // computed on the CPU
            continue;
        }
        const int e = routes[t].expert[k];
        const int p = offsets[e] + atomicAdd(&cursor[e], 1);
        route_token[p] = t;
        pos_of[t * kTopK + k] = p;
    }
}

// grid (t1 - t0), block 256
__global__ void moe_combine_kernel(const Route* routes, const int* pos_of, const float* route_out,
                                   const float* shared, const float* sg, float* out, int t0, const float* cpu_rows) {
    const int t = t0 + blockIdx.x;
    const Route& r = routes[t];
    __shared__ int pos[kTopK];
    __shared__ float w[kTopK];
    if(threadIdx.x < kTopK) {
        pos[threadIdx.x] = pos_of[t * kTopK + threadIdx.x];
        w[threadIdx.x] = r.weight[threadIdx.x];
    }
    __syncthreads();
    const float g = sg[t];
    for(int i = threadIdx.x; i < kHidden; i += blockDim.x) {
        float acc = 0.f;
#pragma unroll
        for(int k = 0; k < kTopK; ++k) {
            const float v = pos[k] >= 0 ? route_out[std::size_t(pos[k]) * kHidden + i]
                                        : cpu_rows[std::size_t(-1 - pos[k]) * kHidden + i];
            acc = fmaf(w[k], v, acc);
        }
        out[std::size_t(t) * kHidden + i] = acc + g * shared[std::size_t(t) * kHidden + i];
    }
}

__device__ __forceinline__ void rope_pair(float& a, float& b, int pos, float inv) {
    float sn, cs;
    sincosf(float(pos) * inv, &sn, &cs);
    const float na = a * cs - b * sn;
    const float nb = b * cs + a * sn;
    a = na;
    b = nb;
}

// grid (33, T), block 256 (as attn_prep_kernel, without pooling).
__global__ void attn_prep_batch_kernel(AttnPrepBatch a) {
    __shared__ float buf[256];
    const int b = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = a.pos0 + t;
    const float* qp = a.proj + std::size_t(t) * kProjAttn;
    const float* kp = qp + kQProj;
    const float* vp = kp + kKvHeads * kHeadDim;
    const float* ip = vp + kKvHeads * kHeadDim;
    if(b < kQHeads + kKvHeads) {
        const bool isq = b < kQHeads;
        const float x = isq ? qp[b * 512 + i] : kp[(b - kQHeads) * kHeadDim + i];
        const float r = rsqrtf(block_sum256(x * x) / float(kHeadDim) + kEps);
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
            a.q[(std::size_t(t) * kQHeads + b) * kHeadDim + i] = __float2half_rn(buf[i]);
            a.gate[(std::size_t(t) * kQHeads + b) * kHeadDim + i] = qp[b * 512 + 256 + i];
        } else {
            const int h = b - kQHeads;
            const float amax = block_max256p(fabsf(buf[i]));
            const float scale = amax > 0.f ? amax / 448.f : 1.f;
            a.kcache[(std::size_t(h) * a.capacity + pos) * kHeadDim + i] = kv_e4m3(buf[i], scale);
            if(i == 0) a.kscale[std::size_t(h) * a.capacity + pos] = scale;
        }
        return;
    }
    if(b < kQHeads + 2 * kKvHeads) {
        const int h = b - kQHeads - kKvHeads;
        const float x = vp[h * kHeadDim + i];
        const float amax = block_max256p(fabsf(x));
        const float scale = amax > 0.f ? amax / 448.f : 1.f;
        a.vcache[(std::size_t(h) * a.capacity + pos) * kHeadDim + i] = kv_e4m3(x, scale);
        if(i == 0) a.vscale[std::size_t(h) * a.capacity + pos] = scale;
        return;
    }
    if(b < kQHeads + 2 * kKvHeads + kIdxHeads) {
        const int h = b - kQHeads - 2 * kKvHeads;
        const float x = i < kIdxDim ? ip[h * kIdxDim + i] : 0.f;
        const float r = rsqrtf(block_sum256(x * x) / float(kIdxDim) + kEps);
        if(i < kIdxDim) buf[i] = x * r * a.iqn1p[i];
        __syncthreads();
        if(i < kRotary / 2) {
            float x0 = buf[i], x1 = buf[i + kRotary / 2];
            rope_pair(x0, x1, pos, a.inv_freq[i]);
            buf[i] = x0;
            buf[i + kRotary / 2] = x1;
        }
        __syncthreads();
        if(i < kIdxDim) {
            a.iq[(std::size_t(t) * kIdxHeads + h) * kIdxDim + i] = buf[i];
            const __half hi = __float2half_rn(buf[i]);
            const __half lo = __float2half_rn(buf[i] - __half2float(hi));
            __half* row = a.iq3 + (std::size_t(t) * kIdxHeads + h) * 3 * kIdxDim;
            row[i] = hi;
            row[kIdxDim + i] = hi;
            row[2 * kIdxDim + i] = lo;
        }
        return;
    }
    if(i < kIdxDim) a.raw[std::size_t(t) * kIdxDim + i] = ip[kIdxHeads * kIdxDim + i];
}

// One block (128 threads) per block completed inside the chunk.
__global__ void pool_blocks_kernel(AttnPrepBatch a, int first_block) {
    __shared__ float buf[kIdxDim];
    const int blk = first_block + blockIdx.x, i = threadIdx.x;
    float s = 0.f;
    for(int j = 0; j < kBlockTokens; ++j) {
        const int p = blk * kBlockTokens + j;
        const float v = p >= a.pos0 ? a.raw[std::size_t(p - a.pos0) * kIdxDim + i] : a.rawkeys[(p % kRawRing) * kIdxDim + i];
        s = j == 0 ? v : s + v;
    }
    const float m = s / float(kBlockTokens);
    const float r = rsqrtf(block_sum256(m * m) / float(kIdxDim) + kEps);
    buf[i] = m * r * a.ikn1p[i];
    __syncthreads();
    if(i < kRotary / 2) {
        float x0 = buf[i], x1 = buf[i + kRotary / 2];
        rope_pair(x0, x1, blk * kBlockTokens, a.inv_freq[i]);
        buf[i] = x0;
        buf[i + kRotary / 2] = x1;
    }
    __syncthreads();
    a.blockkeys[std::size_t(blk) * kIdxDim + i] = buf[i];
}

__global__ void tail_keys_kernel(AttnPrepBatch a, int from) {
    const int p = from + blockIdx.x, i = threadIdx.x;
    a.rawkeys[(p % kRawRing) * kIdxDim + i] = a.raw[std::size_t(p - a.pos0) * kIdxDim + i];
}

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block of 1024 per query row; see topk_blocks_kernel in layers.cu.
__global__ void __launch_bounds__(1024) topk_batch_kernel(const float* scores_all, int ld, int pos0, int q0,
                                                          int* sel_all) {
    const int qi = blockIdx.x;
    const int n = (pos0 + q0 + qi + 1) / kBlockTokens;
    if(n <= kTopBlocks) return;
    const float* scores = scores_all + std::size_t(qi) * ld;
    int* sel = sel_all + std::size_t(qi) * kTopBlocks;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t prefix, prefix_mask, want;
    __shared__ int scan[1024];
    if(threadIdx.x == 0) {
        prefix = 0;
        prefix_mask = 0;
        want = kTopBlocks;
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
            prefix |= uint32_t(bin) << shift;
            prefix_mask |= 255u << shift;
        }
        __syncthreads();
    }
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

// ---- gathered attention on tensor cores ----
constexpr int kQS = 264;      // smem row stride (bf16) for 256-wide rows
constexpr int kChunk = 32;    // keys per chunk

__device__ __forceinline__ void ldsm_x4(uint32_t r[4], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
__device__ __forceinline__ void ldsm_x2(uint32_t r[2], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t r[2], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp16(void* smem, const void* g) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g));
}

// grid (G, kKvHeads), block 128.
__global__ void __launch_bounds__(128) attn_prefill_kernel(AttnPrefill a) {
    __shared__ __align__(16) __half Qs[16][kQS];
    __shared__ __align__(16) __half Ks[kChunk][kQS];
    __shared__ __align__(16) __half Vs[kChunk][kQS];
    __shared__ float Ss[16][kChunk + 1];
    __shared__ __align__(16) __half Ps[16][40];
    __shared__ float rm[16], rl[16], rs[16];
    __shared__ int toks[kChunk];
    const int qi = blockIdx.x, g = blockIdx.y;
    const int t = a.q0 + qi, pos = a.pos0 + t;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nb = (pos + 1) / kBlockTokens;
    const bool sparse = nb > kTopBlocks;
    const int nsel = sparse ? kTopBlocks : nb;
    const int* sel = sparse ? a.sel + std::size_t(qi) * kTopBlocks : nullptr;
    const int tail = nb * kBlockTokens;
    const int nkeys = nsel * kBlockTokens + (pos + 1 - tail);
    // Q: 12 heads of this group, rows 12..15 zero.
    for(int j = tid; j < 16 * 32; j += 128) {
        const int r = j >> 5, c = (j & 31) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if(r < kGroup) v = *reinterpret_cast<const uint4*>(a.q + ((std::size_t(t) * kQHeads + g * kGroup + r) * kHeadDim + c));
        *reinterpret_cast<uint4*>(&Qs[r][c]) = v;
    }
    if(tid < 16) {
        rm[tid] = -INFINITY;
        rl[tid] = 0.f;
    }
    const uint8_t* K = a.kcache + std::size_t(g) * a.capacity * kHeadDim;
    const uint8_t* V = a.vcache + std::size_t(g) * a.capacity * kHeadDim;
    const float* KS = a.kscale + std::size_t(g) * a.capacity;
    const float* VS = a.vscale + std::size_t(g) * a.capacity;
    float o[8][4];
#pragma unroll
    for(int i = 0; i < 8; ++i)
#pragma unroll
        for(int j = 0; j < 4; ++j) o[i][j] = 0.f;
    const int qr = lane >> 2, qc = (lane & 3) * 2;
    for(int k0 = 0; k0 < nkeys; k0 += kChunk) {
        const int cnt = min(kChunk, nkeys - k0);
        __syncthreads();
        if(tid < kChunk) {
            int tok = 0;
            const int j = k0 + tid;
            if(j < nkeys) {
                if(j < nsel * kBlockTokens) {
                    const int blk = sel ? sel[j / kBlockTokens] : j / kBlockTokens;
                    tok = blk * kBlockTokens + (j % kBlockTokens);
                } else {
                    tok = tail + (j - nsel * kBlockTokens);
                }
            } else {
                tok = -1;
            }
            toks[tid] = tok;
        }
        __syncthreads();
        // Load K and V rows: thread -> key tid/4, 64-dim quarter tid%4.
        {
            const int key = tid >> 2, part = (tid & 3) * 64;
            const int tok = toks[key];
            if(tok >= 0) {
                // E4M3 rows to FP16 tiles, each scaled by its vector's scale.
                const float ks = KS[tok], vs = VS[tok];
#pragma unroll
                for(int c = 0; c < 64; c += 16) {
                    const uint4 kq = *reinterpret_cast<const uint4*>(K + std::size_t(tok) * kHeadDim + part + c);
                    const uint4 vq = *reinterpret_cast<const uint4*>(V + std::size_t(tok) * kHeadDim + part + c);
                    const uint32_t kw[4] = {kq.x, kq.y, kq.z, kq.w}, vw[4] = {vq.x, vq.y, vq.z, vq.w};
                    __half2 kh[8], vh[8];
#pragma unroll
                    for(int u = 0; u < 4; ++u) {
                        const float2 k0 = e4m3x2f(kw[u] & 0xffff), k1 = e4m3x2f(kw[u] >> 16);
                        const float2 v0 = e4m3x2f(vw[u] & 0xffff), v1 = e4m3x2f(vw[u] >> 16);
                        kh[2 * u] = __floats2half2_rn(k0.x * ks, k0.y * ks);
                        kh[2 * u + 1] = __floats2half2_rn(k1.x * ks, k1.y * ks);
                        vh[2 * u] = __floats2half2_rn(v0.x * vs, v0.y * vs);
                        vh[2 * u + 1] = __floats2half2_rn(v1.x * vs, v1.y * vs);
                    }
                    *reinterpret_cast<uint4*>(&Ks[key][part + c]) = *reinterpret_cast<const uint4*>(kh);
                    *reinterpret_cast<uint4*>(&Ks[key][part + c + 8]) = *reinterpret_cast<const uint4*>(kh + 4);
                    *reinterpret_cast<uint4*>(&Vs[key][part + c]) = *reinterpret_cast<const uint4*>(vh);
                    *reinterpret_cast<uint4*>(&Vs[key][part + c + 8]) = *reinterpret_cast<const uint4*>(vh + 4);
                }
            } else {
#pragma unroll
                for(int c = 0; c < 64; c += 8) {
                    *reinterpret_cast<uint4*>(&Ks[key][part + c]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&Vs[key][part + c]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        // S = Q K^T for keys 8*warp .. 8*warp+7.
        {
            float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for(int kk = 0; kk < kHeadDim; kk += 16) {
                uint32_t af[4], bf[2];
                ldsm_x4(af, &Qs[lane & 15][kk + (lane >> 4) * 8]);
                ldsm_x2(bf, &Ks[warp * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
                mma16816(s, af, bf);
            }
            const int c0 = warp * 8 + qc;
#pragma unroll
            for(int h = 0; h < 2; ++h) {
                const int r = qr + h * 8;
                Ss[r][c0] = c0 < cnt ? s[h * 2] * (1.f / 16.f) : -INFINITY;
                Ss[r][c0 + 1] = c0 + 1 < cnt ? s[h * 2 + 1] * (1.f / 16.f) : -INFINITY;
            }
        }
        __syncthreads();
        // Online softmax: 8 threads per row, 4 columns each.
        {
            const int r = tid >> 3, c = (tid & 7) * 4;
            float v[4], m = -INFINITY;
#pragma unroll
            for(int j = 0; j < 4; ++j) {
                v[j] = Ss[r][c + j];
                m = fmaxf(m, v[j]);
            }
#pragma unroll
            for(int off = 1; off < 8; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
            const float mold = rm[r];
            const float mnew = fmaxf(mold, m);
            float l = 0.f;
#pragma unroll
            for(int j = 0; j < 4; ++j) {
                const float p = expf(v[j] - mnew);
                v[j] = p;
                l += p;
            }
#pragma unroll
            for(int off = 1; off < 8; off <<= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
            Ps[r][c] = __float2half_rn(v[0]);
            Ps[r][c + 1] = __float2half_rn(v[1]);
            Ps[r][c + 2] = __float2half_rn(v[2]);
            Ps[r][c + 3] = __float2half_rn(v[3]);
            __syncwarp();
            if((tid & 7) == 0) {
                const float scale = mold == -INFINITY ? 0.f : expf(mold - mnew);
                rs[r] = scale;
                rm[r] = mnew;
                rl[r] = rl[r] * scale + l;
            }
        }
        __syncthreads();
        // O = O * scale + P V, warp handles dims 64*warp .. +63.
        {
            const float s0 = rs[qr], s1 = rs[qr + 8];
#pragma unroll
            for(int ni = 0; ni < 8; ++ni) {
                o[ni][0] *= s0;
                o[ni][1] *= s0;
                o[ni][2] *= s1;
                o[ni][3] *= s1;
            }
#pragma unroll
            for(int kk = 0; kk < kChunk; kk += 16) {
                uint32_t af[4];
                ldsm_x4(af, &Ps[lane & 15][kk + (lane >> 4) * 8]);
#pragma unroll
                for(int ni = 0; ni < 8; ++ni) {
                    uint32_t bf[2];
                    ldsm_x2_t(bf, &Vs[kk + (lane & 15)][warp * 64 + ni * 8]);
                    mma16816(o[ni], af, bf);
                }
            }
        }
    }
    __syncthreads();
    // Output: rows < 12 are heads; gate and store bf16.
#pragma unroll
    for(int h2 = 0; h2 < 2; ++h2) {
        const int r = qr + h2 * 8;
        if(r >= kGroup) continue;
        const float inv = 1.f / rl[r];
        const int head = g * kGroup + r;
#pragma unroll
        for(int ni = 0; ni < 8; ++ni) {
            const int d = warp * 64 + ni * 8 + qc;
            const std::size_t idx = (std::size_t(t) * kQHeads + head) * kHeadDim + d;
            const float v0 = o[ni][h2 * 2] * inv * sigmoidf(a.gate[idx]);
            const float v1 = o[ni][h2 * 2 + 1] * inv * sigmoidf(a.gate[idx + 1]);
            *reinterpret_cast<__half2*>(a.out + idx) = __floats2half2_rn(v0, v1);
        }
    }
}

// keys3[b] = [hi(k), lo(k), hi(k)] of the FP32 pooled keys of blocks 0..n-1.
__global__ void split_keys_kernel(const float* bk, int n, __half* keys3) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(n) * kIdxDim) return;
    const std::size_t b = idx / kIdxDim, i = idx % kIdxDim;
    const float x = bk[idx];
    const __half hi = __float2half_rn(x);
    const __half lo = __float2half_rn(x - __half2float(hi));
    __half* row = keys3 + b * 3 * kIdxDim;
    row[i] = hi;
    row[kIdxDim + i] = lo;
    row[2 * kIdxDim + i] = hi;
}

}  // namespace kern
using namespace kern;

void split_keys(const float* bk, int n, __half* keys3, cudaStream_t stream) {
    const std::size_t total = std::size_t(n) * kIdxDim;
    if(!total) return;
    split_keys_kernel<<<unsigned((total + 255) / 256), 256, 0, stream>>>(bk, n, keys3);
    FNL_CUDA(cudaGetLastError());
}

void to_f16(const float* x, __half* y, std::size_t n, cudaStream_t stream) {
    const std::size_t threads = (n + 3) / 4;
    to_f16_kernel<<<unsigned((threads + 255) / 256), 256, 0, stream>>>(x, y, n);
    FNL_CUDA(cudaGetLastError());
}
void to_bf16(const float* x, __nv_bfloat16* y, std::size_t n, cudaStream_t stream) {
    const std::size_t threads = (n + 3) / 4;
    to_bf16_kernel<<<unsigned((threads + 255) / 256), 256, 0, stream>>>(x, y, n);
    FNL_CUDA(cudaGetLastError());
}
void hc_inject(const float* xn, const __nv_bfloat16* w, float* inj, int tokens, cudaStream_t stream) {
    hc_inject_kernel<<<tokens, 128, 0, stream>>>(xn, w, inj);
    FNL_CUDA(cudaGetLastError());
}
void shared_gate(const float* x, const __nv_bfloat16* w, float* sg, int tokens, cudaStream_t stream) {
    shared_gate_kernel<<<tokens, 32, 0, stream>>>(x, w, sg);
    FNL_CUDA(cudaGetLastError());
}
void count_routes(const Route* routes, int tokens, int sub, int* counts, cudaStream_t stream) {
    count_routes_kernel<<<(tokens + 255) / 256, 256, 0, stream>>>(routes, tokens, sub, counts);
    FNL_CUDA(cudaGetLastError());
}
void scatter_routes(const Route* routes, int t0, int t1, const int* offsets, int* cursor, int* route_token,
                    int* pos_of, cudaStream_t stream, const int* cpu_row) {
    scatter_routes_kernel<<<(t1 - t0 + 255) / 256, 256, 0, stream>>>(routes, t0, t1, offsets, cursor, route_token,
                                                                      pos_of, cpu_row);
    FNL_CUDA(cudaGetLastError());
}
void moe_combine(const Route* routes, const int* pos_of, const float* route_out, const float* shared, const float* sg,
                 float* out, int t0, int t1, cudaStream_t stream, const float* cpu_rows) {
    moe_combine_kernel<<<t1 - t0, 256, 0, stream>>>(routes, pos_of, route_out, shared, sg, out, t0, cpu_rows);
    FNL_CUDA(cudaGetLastError());
}
void attn_prep_batch(const AttnPrepBatch& a, cudaStream_t stream) {
    attn_prep_batch_kernel<<<dim3(kQHeads + 2 * kKvHeads + kIdxHeads + 1, a.tokens), 256, 0, stream>>>(a);
    FNL_CUDA(cudaGetLastError());
}
void pool_blocks(const AttnPrepBatch& a, cudaStream_t stream) {
    const int first = a.pos0 / kBlockTokens, last = (a.pos0 + a.tokens) / kBlockTokens;  // [first, last)
    if(last > first) {
        pool_blocks_kernel<<<last - first, 128, 0, stream>>>(a, first);
        FNL_CUDA(cudaGetLastError());
    }
    const int from = std::max(a.pos0, last * kBlockTokens), to = a.pos0 + a.tokens;
    if(to > from) {
        tail_keys_kernel<<<to - from, 128, 0, stream>>>(a, from);
        FNL_CUDA(cudaGetLastError());
    }
}
void topk_batch(const float* scores, int ld, int pos0, int q0, int G, int* sel, cudaStream_t stream) {
    topk_batch_kernel<<<G, 1024, 0, stream>>>(scores, ld, pos0, q0, sel);
    FNL_CUDA(cudaGetLastError());
}
void attn_prefill(const AttnPrefill& a, cudaStream_t stream) {
    attn_prefill_kernel<<<dim3(a.G, kKvHeads), 128, 0, stream>>>(a);
    FNL_CUDA(cudaGetLastError());
}

}  // namespace fnl
