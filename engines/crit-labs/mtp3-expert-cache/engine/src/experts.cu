#include "experts.cuh"
#include "prefill.cuh"

namespace fnl {

namespace kern {

__global__ void route_topk_kernel(const float* logits, Route* route) {
    logits += std::size_t(blockIdx.x) * kExperts;
    route += blockIdx.x;
    __shared__ float prob[kExperts];
    __shared__ float red[32];
    __shared__ int redi[32];
    const int t = threadIdx.x;  // kExperts threads
    const float l = logits[t];
    // max
    float m = warp_max(l);
    if((t & 31) == 0) red[t >> 5] = m;
    __syncthreads();
    if(t < 32) {
        float v = t < kExperts / 32 ? red[t] : -INFINITY;
        v = warp_max(v);
        if(t == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    const float e = expf(l - m);
    float s = warp_sum(e);
    if((t & 31) == 0) red[t >> 5] = s;
    __syncthreads();
    if(t < 32) {
        float v = t < kExperts / 32 ? red[t] : 0.f;
        v = warp_sum(v);
        if(t == 0) red[0] = v;
    }
    __syncthreads();
    prob[t] = e / red[0];
    __syncthreads();
    // Top-k by repeated argmax; ties resolve to the lower expert index.
    float picked[kTopK];
    int ids[kTopK];
    for(int k = 0; k < kTopK; ++k) {
        float v = prob[t];
        int idx = t;
#pragma unroll
        for(int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if(ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if((t & 31) == 0) { red[t >> 5] = v; redi[t >> 5] = idx; }
        __syncthreads();
        if(t < 32) {
            v = t < kExperts / 32 ? red[t] : -INFINITY;
            idx = t < kExperts / 32 ? redi[t] : 1 << 30;
#pragma unroll
            for(int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if(ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if(t == 0) { red[0] = v; redi[0] = idx; }
        }
        __syncthreads();
        picked[k] = red[0];
        ids[k] = redi[0];
        if(t == ids[k]) prob[t] = -INFINITY;
        __syncthreads();
    }
    if(t == 0) {
        float sum = 0.f;
        for(int k = 0; k < kTopK; ++k) sum += picked[k];
        for(int k = 0; k < kTopK; ++k) {
            route->expert[k] = ids[k];
            route->weight[k] = picked[k] / sum;
        }
    }
}

// One block of 64 threads: union of the T tokens' routes, hits in parallel, misses by one warp running CLOCK
// 32 slots at a time (same victims as a sequential CLOCK hand).
__global__ void plan_experts_kernel(ExpertTables t, int layer, const Route* routes, int tokens, ExpertPlan* plan,
                                    const float* x) {
    __shared__ int n;
    __shared__ int ue[kMaxUnion], um[kMaxUnion], us[kMaxUnion];
    __shared__ float uw[kMaxUnion][kMaxPassTokens];
    __shared__ uint32_t step;
    __shared__ int nmiss, ncpu;
    __shared__ int miss_list[kMaxUnion];
    const int tid = threadIdx.x;
    const bool fixed = layer == kMtpLayer;
    // Union of the routes in order of first occurrence over (token, k): route i finds the first route with its
    // expert, the first ones are numbered in route order, and each route ORs its token into its entry.
    __shared__ int re[kMaxUnion], first_of[kMaxUnion], pos[kMaxUnion];
    const int nr = tokens * kTopK;
    if(tid < nr) re[tid] = routes[tid / kTopK].expert[tid % kTopK];
    if(tid < kMaxUnion) {
        um[tid] = 0;
        for(int j = 0; j < kMaxPassTokens; ++j) uw[tid][j] = 0.f;
    }
    __syncthreads();
    if(tid < nr) {
        int f = tid;
        for(int j = 0; j < tid; ++j)
            if(re[j] == re[tid]) {
                f = j;
                break;
            }
        first_of[tid] = f;
    }
    __syncthreads();
    if(tid == 0) {
        int c = 0;
        for(int i = 0; i < nr; ++i)
            if(first_of[i] == i) pos[i] = c++;
        n = c;
        nmiss = 0;
        step = fixed ? 0u : ++(*t.step);
    }
    __syncthreads();
    if(tid < nr) {
        const int u = pos[first_of[tid]], tk = tid / kTopK;
        if(first_of[tid] == tid) ue[u] = re[tid];
        atomicOr(&um[u], 1 << tk);
        uw[u][tk] = routes[tk].weight[tid % kTopK];
    }
    __syncthreads();
    if(tid < n) {
        const int e = ue[tid];
        plan->expert[tid] = e;
        plan->mask[tid] = um[tid];
        for(int j = 0; j < kMaxPassTokens; ++j) plan->weight[tid][j] = uw[tid][j];
        const float* s2 = fixed ? t.fixed_s2 + std::size_t(e) * 3 : t.s2 + (std::size_t(layer) * kExperts + e) * 3;
        plan->s2[tid][0] = s2[0];
        plan->s2[tid][1] = s2[1];
        plan->s2[tid][2] = s2[2];
        if(fixed) {
            uint8_t* p = t.fixed_base + std::size_t(e) * kSlotBytes;
            plan->ref[tid] = {p, p, 0, 0};
            us[tid] = -2;
        } else {
            const int s = t.nslots ? t.slot_of[layer * kExperts + e] : -1;
            us[tid] = s;
            if(s >= 0) {
                t.ref[s] = 1;
                t.stamp[s] = step;
                uint8_t* p = t.slot_addr[s];
                plan->ref[tid] = {p, p, 0, 0};
            }
        }
    }
    if(tid == 0) plan->n = n;
    if(fixed) {
        if(tid == 0 && t.cpu_state) t.cpu_state[1] = 0;
        return;
    }
    __syncthreads();
    if(tid == 0) {
        int m = 0;
        for(int u = 0; u < n; ++u)
            if(us[u] < 0) miss_list[m++] = u;
        nmiss = m;
        ncpu = t.job ? min(m, int(float(m) * t.cpu_share + 0.5f)) : 0;
        if(t.stats) {
            t.stats[0] += tokens * kTopK;
            t.stats[1] += m;
        }
    }
    __syncthreads();
    // CLOCK victims for every miss (one warp). Slots touched earlier in this pass stay protected: their
    // background insertions may still be in flight.
    __shared__ uint8_t* vslot[kMaxUnion];
    const uint32_t first = t.pass_first ? *t.pass_first : step;
    if(tid < 32) {
        const int lane = tid;
        int h = t.nslots ? *t.hand : 0;
        for(int i = 0; i < nmiss; ++i) {
            const int u = miss_list[i];
            const int e = ue[u];
            const uint8_t* host = t.host_layer[layer] + std::size_t(e) * kHostExpertBytes;
            int victim = -1;
            if(t.nslots) {
                // Windows of 32 slots from the hand; eligible slots with ref set before the victim are cleared.
                for(int scanned = 0; scanned < 2 * t.nslots + 64 && victim < 0; scanned += 32) {
                    int s = h + lane;
                    while(s >= t.nslots) s -= t.nslots;
                    const bool in_range = lane < t.nslots;
                    const bool eligible = in_range && t.stamp[s] < first && t.valid[s];
                    const bool refd = eligible && t.ref[s];
                    const unsigned pick = __ballot_sync(0xffffffffu, eligible && !refd);
                    const int stop = pick ? __ffs(pick) - 1 : 32;   // lanes before stop are passed over
                    if(refd && lane < stop) t.ref[s] = 0;
                    if(pick) victim = __shfl_sync(0xffffffffu, s, stop);
                    h += pick ? stop + 1 : 32;
                    while(h >= t.nslots) h -= t.nslots;
                    __syncwarp();
                }
            }
            if(lane == 0) {
                uint8_t* slot = nullptr;
                if(victim >= 0) {
                    const int old = t.owner[victim];
                    if(old >= 0) t.slot_of[old] = -1;
                    t.owner[victim] = layer * kExperts + e;
                    t.slot_of[layer * kExperts + e] = victim;
                    t.ref[victim] = 1;
                    t.stamp[victim] = step;
                    slot = t.slot_addr[victim];
                }
                ExpertRef r;
                if(i < ncpu) {
                    // Computed by the CPU; the slot is filled in the background.
                    r = {host, slot, 2, 0};
                    vslot[i] = slot;
                } else if(slot) {
                    r = {host, slot, 1, 1};
                } else {
                    r = {host, t.scratch + std::size_t(u) * kSlotBytes, 1, 0};
                }
                plan->ref[u] = r;
            }
            __syncwarp();
        }
        if(lane == 0 && t.nslots) *t.hand = h;
    }
    __syncthreads();
    // The CPU's share: the first ncpu misses, posted to the mailbox.
    if(t.job) {
        CpuJob* job = t.job;
        if(tid < ncpu) {
            const int u = miss_list[tid];
            job->expert[tid] = ue[u];
            job->mask[tid] = um[u];
            job->slot[tid] = reinterpret_cast<unsigned long long>(vslot[tid]);
            for(int j = 0; j < kMaxPassTokens; ++j) job->weight[tid][j] = uw[u][j];
        }
        if(ncpu > 0) {
            const float4* xs = reinterpret_cast<const float4*>(x);
            float4* xd = reinterpret_cast<float4*>(&job->x[0][0]);
            for(int i = tid; i < tokens * kHidden / 4; i += blockDim.x) xd[i] = xs[i];
        }
        __threadfence_system();
        __syncthreads();
        if(tid == 0) {
            if(ncpu > 0) {
                job->layer = layer;
                job->tokens = tokens;
                job->count = ncpu;
                __threadfence_system();
                const int seq = t.cpu_state[0] + 1;
                t.cpu_state[0] = seq;
                job->seq = uint32_t(seq);
                __threadfence_system();
            }
            t.cpu_state[1] = ncpu > 0;
        }
    }
}


// First expert-table step of a decode pass (slots stamped from it on are protected until the pass ends).
__global__ void begin_pass_kernel(ExpertTables t) {
    if(threadIdx.x == 0 && t.pass_first) *t.pass_first = *t.step + 1;
}

constexpr int kScaleSplit = 8;
// Slabs part, first, first + step, ... of the CSF fixed planes of matrix m (host record rec) into slot dst.
__device__ void csf_decode_slabs(int m, const uint8_t* rec, uint8_t* dst, int first, int step);
// Exception words of matrix m of expert e into slot dst (after its slabs).
__device__ void csf_exceptions(const ExpertTables& t, int layer, int e, int m, uint8_t* dst) {
    uint8_t* out = dst + kSlotScales + std::size_t(m) * kGateScales;
    const std::size_t idx = (std::size_t(layer) * kExperts + e) * 3 + m;
    const uint32_t off = t.exc_off[idx], cnt = t.exc_cnt[idx];
    for(uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        const uint32_t w = t.exc_pool[off + i];
        out[w & 0xffffffu] = uint8_t(w >> 24);
    }
}
__device__ void csf_decode_slabs(int m, const uint8_t* rec, uint8_t* dst, int first, int step) {
    constexpr int kMaxSlab = 16 * (1 + kHidden / 32);   // 1296 bytes (gate/up); down slabs are 336
    __shared__ __align__(16) uint8_t stage[8][kMaxSlab];
    const int rows = m < 2 ? kInter : kHidden;
    const int cols = m < 2 ? kHidden / 16 : kInter / 16;
    const int half_cols = cols / 2;
    const int slab_bytes = 16 * (1 + half_cols);
    const uint8_t* fixed = rec + kHostCsf + (m == 0 ? 0 : m == 1 ? kGateCsf : 2 * kGateCsf);
    uint8_t* out = dst + kSlotScales + std::size_t(m) * kGateScales;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    uint8_t* sb = stage[warp];
    for(int slab = first + warp * step; slab < rows / 16; slab += warps * step) {
        const uint4* src = reinterpret_cast<const uint4*>(fixed + std::size_t(slab) * slab_bytes);
        for(int c = lane; c < slab_bytes / 16; c += 32) reinterpret_cast<uint4*>(sb)[c] = src[c];
        __syncwarp();
        for(int i = lane; i < 16 * half_cols / 2; i += 32) {
            const int rr = (2 * i) / half_cols, cp = (2 * i) % half_cols;
            const uint32_t base = sb[rr];
            const uint32_t n0 = sb[16 + rr * half_cols + cp], n1 = sb[16 + rr * half_cols + cp + 1];
            const uint32_t v = (base + (n0 & 15)) | ((base + (n0 >> 4)) << 8) | ((base + (n1 & 15)) << 16) |
                               ((base + (n1 >> 4)) << 24);
            *reinterpret_cast<uint32_t*>(out + std::size_t(slab * 16 + rr) * cols + 2 * cp) = v;
        }
        __syncwarp();
    }
}

// CSF scales of matrix m of expert e (host record rec) into slot dst. Whole block; ends synchronized.
__device__ void csf_decode(const ExpertTables& t, int layer, int e, int m, const uint8_t* rec, uint8_t* dst) {
    // Each warp stages one 16-row slab of the (mapped host) fixed plane in shared memory with 16-byte loads,
    // then expands it: scale = base[row] + nibble. Exceptions follow once every slab is written.
    constexpr int kMaxSlab = 16 * (1 + kHidden / 32);   // 1296 bytes (gate/up); down slabs are 336
    __shared__ __align__(16) uint8_t stage[8][kMaxSlab];
    const int rows = m < 2 ? kInter : kHidden;
    const int cols = m < 2 ? kHidden / 16 : kInter / 16;
    const int half_cols = cols / 2;
    const int slab_bytes = 16 * (1 + half_cols);
    const uint8_t* fixed = rec + kHostCsf + (m == 0 ? 0 : m == 1 ? kGateCsf : 2 * kGateCsf);
    uint8_t* out = dst + kSlotScales + std::size_t(m) * kGateScales;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    uint8_t* sb = stage[warp];
    for(int slab = warp; slab < rows / 16; slab += warps) {
        const uint4* src = reinterpret_cast<const uint4*>(fixed + std::size_t(slab) * slab_bytes);
        for(int c = lane; c < slab_bytes / 16; c += 32) reinterpret_cast<uint4*>(sb)[c] = src[c];
        __syncwarp();
        // 16 rows x half_cols nibble bytes; each lane writes 4 scale bytes (2 nibble bytes) per step.
        for(int i = lane; i < 16 * half_cols / 2; i += 32) {
            const int rr = (2 * i) / half_cols, cp = (2 * i) % half_cols;
            const uint32_t base = sb[rr];
            const uint32_t n0 = sb[16 + rr * half_cols + cp], n1 = sb[16 + rr * half_cols + cp + 1];
            const uint32_t v = (base + (n0 & 15)) | ((base + (n0 >> 4)) << 8) | ((base + (n1 & 15)) << 16) |
                               ((base + (n1 >> 4)) << 24);
            *reinterpret_cast<uint32_t*>(out + std::size_t(slab * 16 + rr) * cols + 2 * cp) = v;
        }
        __syncwarp();
    }
    __syncthreads();
    const std::size_t idx = (std::size_t(layer) * kExperts + e) * 3 + m;
    const uint32_t off = t.exc_off[idx], cnt = t.exc_cnt[idx];
    for(uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        const uint32_t w = t.exc_pool[off + i];
        out[w & 0xffffffu] = uint8_t(w >> 24);
    }
}

// grid (T * kTopK, 3, kScaleSplit), block 256: base + nibble planes of slabs z, z + kScaleSplit, ...
__global__ void plan_miss_slabs_kernel(ExpertTables t, int layer, const ExpertPlan* plan) {
    if(int(blockIdx.x) >= plan->n) return;
    const ExpertRef r = plan->ref[blockIdx.x];
    if(r.miss != 1) return;
    csf_decode_slabs(blockIdx.y, r.src, r.dst, blockIdx.z, kScaleSplit);
}
// grid (T * kTopK, 3), block 256: exception words after every slab is written.
__global__ void plan_miss_exceptions_kernel(ExpertTables t, int layer, const ExpertPlan* plan) {
    if(int(blockIdx.x) >= plan->n) return;
    const ExpertRef r = plan->ref[blockIdx.x];
    if(r.miss != 1) return;
    csf_exceptions(t, layer, plan->expert[blockIdx.x], blockIdx.y, r.dst);
}

// grid (n, 3), block 256.
// grid (n, 3, kScaleSplit)
__global__ void decode_slabs_kernel(ExpertTables t, int layer, const ScaleJob* jobs) {
    const ScaleJob j = jobs[blockIdx.x];
    const uint8_t* rec = t.host_layer[layer] + std::size_t(j.expert) * kHostExpertBytes;
    csf_decode_slabs(blockIdx.y, rec, j.slot, blockIdx.z, kScaleSplit);
}
// grid (n, 3)
__global__ void decode_exceptions_kernel(ExpertTables t, int layer, const ScaleJob* jobs) {
    const ScaleJob j = jobs[blockIdx.x];
    csf_exceptions(t, layer, j.expert, blockIdx.y, j.slot);
}

// 16 E2M1 values (8 bytes) dotted with 16 FP32 activations.
__device__ __forceinline__ float dot16_e2m1(uint2 w, const float* x) {
    const uint32_t words[2] = {w.x, w.y};
    float acc = 0.f;
#pragma unroll
    for(int q = 0; q < 2; ++q) {
#pragma unroll
        for(int b = 0; b < 4; ++b) {
            const float2 v = e2m1x2f((words[q] >> (8 * b)) & 0xff);
            acc = fmaf(v.x, x[q * 8 + b * 2], acc);
            acc = fmaf(v.y, x[q * 8 + b * 2 + 1], acc);
        }
    }
    return acc;
}

constexpr int kWarps = 8;
constexpr int kRowsPerWarp = 2;

// 16 E2M1 values (8 bytes) to floats.
__device__ __forceinline__ void decode16_e2m1(uint2 w, float* f) {
    const uint32_t words[2] = {w.x, w.y};
#pragma unroll
    for(int q = 0; q < 2; ++q) {
#pragma unroll
        for(int b = 0; b < 4; ++b) {
            const float2 v = e2m1x2f((words[q] >> (8 * b)) & 0xff);
            f[q * 8 + b * 2] = v.x;
            f[q * 8 + b * 2 + 1] = v.y;
        }
    }
}
// Sequential dot of 16 decoded weights with 16 activations (read through the read-only cache).
__device__ __forceinline__ float dot16_f(const float* w, const float* x) {
    const float4* xp = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
#pragma unroll
    for(int q = 0; q < 4; ++q) {
        const float4 v = __ldg(xp + q);
        acc = fmaf(w[4 * q], v.x, acc);
        acc = fmaf(w[4 * q + 1], v.y, acc);
        acc = fmaf(w[4 * q + 2], v.z, acc);
        acc = fmaf(w[4 * q + 3], v.w, acc);
    }
    return acc;
}

// grid (kInter / (kWarps * kRowsPerWarp), T * kTopK), block 256. Entry u = blockIdx.y; tokens outside its mask
// are skipped. A lane loads all its weight groups of a row before decoding them once for every token.
template <int T>
__global__ void __launch_bounds__(kWarps * 32) expert_gate_up_kernel(const ExpertPlan* plan, const float* x,
                                                                     float* act) {
    const int u = blockIdx.y;
    if(u >= plan->n) return;
    const ExpertRef r = plan->ref[u];
    if(r.miss == 2) return;
    const int mask = plan->mask[u];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float s2g = plan->s2[u][0], s2u = plan->s2[u][1];
    constexpr int groups = kHidden / 16;   // 160 per row
    constexpr int per = groups / 32;       // 5 per lane
#pragma unroll 1
    for(int rr = 0; rr < kRowsPerWarp; ++rr) {
        const int row = (blockIdx.x * kWarps + warp) * kRowsPerWarp + rr;
        const uint2* gw = reinterpret_cast<const uint2*>(r.src + std::size_t(row) * (kHidden / 2));
        const uint2* uw = reinterpret_cast<const uint2*>(r.src + kGateBytes + std::size_t(row) * (kHidden / 2));
        const uint8_t* gs = r.dst + kSlotScales + std::size_t(row) * groups;
        const uint8_t* us = r.dst + kSlotScales + kGateScales + std::size_t(row) * groups;
        uint2 wg[per], wu[per];
        uint8_t sgb[per], sub[per];
        if(r.miss) {
#pragma unroll
            for(int i = 0; i < per; ++i) {
                wg[i] = gw[lane + 32 * i];
                wu[i] = uw[lane + 32 * i];
            }
        } else {
#pragma unroll
            for(int i = 0; i < per; ++i) {
                wg[i] = __ldg(gw + lane + 32 * i);
                wu[i] = __ldg(uw + lane + 32 * i);
            }
        }
#pragma unroll
        for(int i = 0; i < per; ++i) {
            sgb[i] = gs[lane + 32 * i];
            sub[i] = us[lane + 32 * i];
        }
        if(r.write_through) {
            uint2* gd = reinterpret_cast<uint2*>(r.dst + std::size_t(row) * (kHidden / 2));
            uint2* ud = reinterpret_cast<uint2*>(r.dst + kGateBytes + std::size_t(row) * (kHidden / 2));
#pragma unroll
            for(int i = 0; i < per; ++i) {
                gd[lane + 32 * i] = wg[i];
                ud[lane + 32 * i] = wu[i];
            }
        }
        float ag[T], au[T];
#pragma unroll
        for(int tk = 0; tk < T; ++tk) ag[tk] = au[tk] = 0.f;
#pragma unroll
        for(int i = 0; i < per; ++i) {
            const int g = lane + 32 * i;
            const float sg = e4m3f(sgb[i]), su = e4m3f(sub[i]);
            float fg[16], fu[16];
            decode16_e2m1(wg[i], fg);
            decode16_e2m1(wu[i], fu);
#pragma unroll
            for(int tk = 0; tk < T; ++tk)
                if(mask & (1 << tk)) {
                    const float* xt = x + std::size_t(tk) * kHidden + g * 16;
                    ag[tk] = fmaf(dot16_f(fg, xt), sg, ag[tk]);
                    au[tk] = fmaf(dot16_f(fu, xt), su, au[tk]);
                }
        }
#pragma unroll
        for(int tk = 0; tk < T; ++tk)
            if(mask & (1 << tk)) {
                const float a = warp_sum(ag[tk]), b = warp_sum(au[tk]);
                if(lane == 0) act[(std::size_t(u) * kMaxPassTokens + tk) * kInter + row] = siluf(a * s2g) * (b * s2u);
            }
    }
}

// grid kHidden / kWarps, block 256: one output row per warp, entries in plan order, 4 entries' weights loaded
// before their arithmetic; activations read through the read-only cache.
template <int T>
__global__ void __launch_bounds__(kWarps * 32) expert_down_kernel(const ExpertPlan* plan, const float* act,
                                                                  const float* shared, const float* shared_gate,
                                                                  float* y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + warp;
    constexpr int groups = kInter / 16;  // 40: lane takes group lane and, below 8, lane + 32
    constexpr int kAhead = 4;
    const int n = plan->n;
    float out[T];
#pragma unroll
    for(int tk = 0; tk < T; ++tk) out[tk] = 0.f;
    for(int u0 = 0; u0 < n; u0 += kAhead) {
        uint2 w[kAhead][2];
        uint8_t sb[kAhead][2];
        bool live[kAhead];
#pragma unroll
        for(int j = 0; j < kAhead; ++j) {
            const int u = u0 + j;
            live[j] = u < n && plan->ref[u].miss != 2;
            w[j][0] = w[j][1] = make_uint2(0, 0);
            sb[j][0] = sb[j][1] = 0;
            if(!live[j]) continue;
            const ExpertRef r = plan->ref[u];
            const uint2* dw = reinterpret_cast<const uint2*>(r.src + 2 * kGateBytes + std::size_t(row) * (kInter / 2));
            const uint8_t* ds = r.dst + kSlotScales + 2 * kGateScales + std::size_t(row) * groups;
            if(r.miss) {
                w[j][0] = dw[lane];
                if(lane + 32 < groups) w[j][1] = dw[lane + 32];
            } else {
                w[j][0] = __ldg(dw + lane);
                if(lane + 32 < groups) w[j][1] = __ldg(dw + lane + 32);
            }
            sb[j][0] = ds[lane];
            if(lane + 32 < groups) sb[j][1] = ds[lane + 32];
            if(r.write_through) {
                uint2* dd = reinterpret_cast<uint2*>(r.dst + 2 * kGateBytes + std::size_t(row) * (kInter / 2));
                dd[lane] = w[j][0];
                if(lane + 32 < groups) dd[lane + 32] = w[j][1];
            }
        }
#pragma unroll
        for(int j = 0; j < kAhead; ++j) {
            if(!live[j]) continue;
            const int u = u0 + j;
            const int mask = plan->mask[u];
            float acc[T];
#pragma unroll
            for(int tk = 0; tk < T; ++tk) acc[tk] = 0.f;
#pragma unroll
            for(int h = 0; h < 2; ++h) {
                const int g = lane + 32 * h;
                if(g >= groups) continue;
                const float sc = e4m3f(sb[j][h]);
                float f[16];
                decode16_e2m1(w[j][h], f);
#pragma unroll
                for(int tk = 0; tk < T; ++tk)
                    if(mask & (1 << tk))
                        acc[tk] = fmaf(dot16_f(f, act + (std::size_t(u) * kMaxPassTokens + tk) * kInter + g * 16), sc,
                                       acc[tk]);
            }
            const float s2 = plan->s2[u][2];
#pragma unroll
            for(int tk = 0; tk < T; ++tk)
                if(mask & (1 << tk)) {
                    const float a = warp_sum(acc[tk]);
                    out[tk] = fmaf(plan->weight[u][tk], a * s2, out[tk]);
                }
        }
    }
    if(lane == 0) {
#pragma unroll
        for(int tk = 0; tk < T; ++tk)
            y[std::size_t(tk) * kHidden + row] = out[tk] + shared_gate[tk] * shared[std::size_t(tk) * kHidden + row];
    }
}

}  // namespace kern
using namespace kern;

void route_topk(const float* logits, Route* route, cudaStream_t stream) {
    route_topk_kernel<<<1, kExperts, 0, stream>>>(logits, route);
    FNL_CUDA(cudaGetLastError());
}
void route_batch(const float* logits, Route* routes, int tokens, cudaStream_t stream) {
    route_topk_kernel<<<tokens, kExperts, 0, stream>>>(logits, routes);
    FNL_CUDA(cudaGetLastError());
}
void decode_scales(const ExpertTables& t, int layer, const ScaleJob* jobs, int n, cudaStream_t stream) {
    if(n == 0) return;
    decode_slabs_kernel<<<dim3(n, 3, kScaleSplit), 256, 0, stream>>>(t, layer, jobs);
    FNL_CUDA(cudaGetLastError());
    decode_exceptions_kernel<<<dim3(n, 3), 256, 0, stream>>>(t, layer, jobs);
    FNL_CUDA(cudaGetLastError());
}
void plan_experts(const ExpertTables& t, int layer, const Route* routes, int tokens, ExpertPlan* plan,
                  const float* x, cudaStream_t stream) {
    plan_experts_kernel<<<1, 64, 0, stream>>>(t, layer, routes, tokens, plan, x);
    FNL_CUDA(cudaGetLastError());
}

namespace kern {
__global__ void cpu_wait_kernel(const int* cpu_state, const CpuReply* reply) {
    if(threadIdx.x != 0 || !cpu_state[1]) return;
    const uint32_t seq = uint32_t(cpu_state[0]);
    const volatile uint32_t* p = &reply->seq;
    for(long long i = 0; *p != seq && i < (1ll << 32); ++i) __nanosleep(64);
    __threadfence_system();
}
__global__ void hc_combine_cpu_kernel(float* H, const float* out, const float* inj, int tokens, const CpuReply* reply,
                                      const int* cpu_state) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= std::size_t(tokens) * kHyper) return;
    const std::size_t t = idx / kHyper, r = idx % kHyper;
    const int s = int(r / kHidden), i = int(r % kHidden);
    float o = out[t * kHidden + i];
    if(cpu_state[1]) o += reply->y[t][i];
    H[idx] = H[idx] + o * inj[t * kStreams + s];
}
}  // namespace kern

void cpu_wait(const ExpertTables& t, const CpuReply* reply, cudaStream_t stream) {
    cpu_wait_kernel<<<1, 32, 0, stream>>>(t.cpu_state, reply);
    FNL_CUDA(cudaGetLastError());
}
void hc_combine_cpu(float* H, const float* out, const float* inj, int tokens, const CpuReply* reply,
                    const int* cpu_state, cudaStream_t stream) {
    const std::size_t n = std::size_t(tokens) * kHyper;
    hc_combine_cpu_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(H, out, inj, tokens, reply, cpu_state);
    FNL_CUDA(cudaGetLastError());
}

// Quarter q of the 16-float group g in staged activations, rotated with (g >> 1) & 3 so that the 8 lanes of a
// quarter-warp read distinct banks.
__device__ __forceinline__ int sw16e(int g, int q) { return g * 16 + 4 * (q ^ ((g >> 1) & 3)); }

constexpr int kDownRowsPerWarp = 4;
constexpr int kDownRowsPerBlock = kWarps * kDownRowsPerWarp;   // 64

// grid (kHidden / 64, T * kTopK), block 256: entry u = blockIdx.y, 8 rows per warp whose weights are all loaded
// before the activations are staged; part[u][t][row] = s2 * (row . act[u][t]) for the tokens routed to u. The
// per-lane group order, products and warp reduction are those of expert_down_kernel.
template <int T>
__global__ void __launch_bounds__(kWarps * 32) expert_down_part_kernel(const ExpertPlan* plan, const float* act,
                                                                       float* part) {
    __shared__ __align__(16) float as[T][kInter];
    const int u = blockIdx.y;
    if(u >= plan->n) return;
    const ExpertRef r = plan->ref[u];
    if(r.miss == 2) return;
    const int mask = plan->mask[u];
    constexpr int groups = kInter / 16;  // 40: lane takes group lane and, below 8, lane + 32
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * kDownRowsPerBlock + warp * kDownRowsPerWarp;
    uint2 w[kDownRowsPerWarp][2];
    uint8_t sb[kDownRowsPerWarp][2];
#pragma unroll
    for(int j = 0; j < kDownRowsPerWarp; ++j) {
        const int row = row0 + j;
        const uint2* dw = reinterpret_cast<const uint2*>(r.src + 2 * kGateBytes + std::size_t(row) * (kInter / 2));
        const uint8_t* ds = r.dst + kSlotScales + 2 * kGateScales + std::size_t(row) * groups;
        w[j][1] = make_uint2(0, 0);
        sb[j][1] = 0;
        if(r.miss) {
            w[j][0] = dw[lane];
            if(lane + 32 < groups) w[j][1] = dw[lane + 32];
        } else {
            w[j][0] = __ldg(dw + lane);
            if(lane + 32 < groups) w[j][1] = __ldg(dw + lane + 32);
        }
        sb[j][0] = ds[lane];
        if(lane + 32 < groups) sb[j][1] = ds[lane + 32];
    }
    // Activations staged while the weight loads are in flight.
    for(int i = threadIdx.x; i < T * groups * 4; i += blockDim.x) {
        const int t = i / (groups * 4);
        if(!(mask & (1 << t))) continue;
        const int rem = i - t * groups * 4;
        const int g = rem >> 2, q = rem & 3;
        *reinterpret_cast<float4*>(&as[t][sw16e(g, q)]) =
            *reinterpret_cast<const float4*>(act + (std::size_t(u) * kMaxPassTokens + t) * kInter + g * 16 + q * 4);
    }
    __syncthreads();
    if(r.write_through) {
#pragma unroll
        for(int j = 0; j < kDownRowsPerWarp; ++j) {
            uint2* dd = reinterpret_cast<uint2*>(r.dst + 2 * kGateBytes + std::size_t(row0 + j) * (kInter / 2));
            dd[lane] = w[j][0];
            if(lane + 32 < groups) dd[lane + 32] = w[j][1];
        }
    }
    const float s2 = plan->s2[u][2];
#pragma unroll
    for(int j = 0; j < kDownRowsPerWarp; ++j) {
        float acc[T];
#pragma unroll
        for(int tk = 0; tk < T; ++tk) acc[tk] = 0.f;
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            const int g = lane + 32 * h;
            if(g >= groups) continue;
            const float sc = e4m3f(sb[j][h]);
            float f[16];
            decode16_e2m1(w[j][h], f);
#pragma unroll
            for(int tk = 0; tk < T; ++tk)
                if(mask & (1 << tk)) {
                    float d = 0.f;
#pragma unroll
                    for(int q = 0; q < 4; ++q) {
                        const float4 v = *reinterpret_cast<const float4*>(&as[tk][sw16e(g, q)]);
                        d = fmaf(f[4 * q], v.x, d);
                        d = fmaf(f[4 * q + 1], v.y, d);
                        d = fmaf(f[4 * q + 2], v.z, d);
                        d = fmaf(f[4 * q + 3], v.w, d);
                    }
                    acc[tk] = fmaf(d, sc, acc[tk]);
                }
        }
#pragma unroll
        for(int tk = 0; tk < T; ++tk)
            if(mask & (1 << tk)) {
                const float a = warp_sum(acc[tk]);
                if(lane == 0) part[(std::size_t(u) * kMaxPassTokens + tk) * kHidden + row0 + j] = a * s2;
            }
    }
}

// y[t][row] = sum over entries in plan order of w_ut * part[u][t][row], plus the gated shared expert. The plan's
// masks and weights are staged in shared memory and 4 entries' partials are loaded ahead of their sums.
template <int T>
__global__ void expert_down_sum_kernel(const ExpertPlan* plan, const float* part, const float* shared,
                                       const float* shared_gate, float* y) {
    __shared__ float ws[kMaxUnion][T];
    __shared__ int ms[kMaxUnion];
    const int n = plan->n;
    if(threadIdx.x < kMaxUnion) {
        const int u = threadIdx.x;
        ms[u] = u < n && plan->ref[u].miss != 2 ? plan->mask[u] : 0;
#pragma unroll
        for(int tk = 0; tk < T; ++tk) ws[u][tk] = plan->weight[u][tk];
    }
    __syncthreads();
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= kHidden) return;
    float out[T];
#pragma unroll
    for(int tk = 0; tk < T; ++tk) out[tk] = 0.f;
    for(int u0 = 0; u0 < n; u0 += 4) {
        float p[4][T];
#pragma unroll
        for(int j = 0; j < 4; ++j) {
            const int m = u0 + j < n ? ms[u0 + j] : 0;
#pragma unroll
            for(int tk = 0; tk < T; ++tk)
                p[j][tk] = m & (1 << tk) ? part[(std::size_t(u0 + j) * kMaxPassTokens + tk) * kHidden + row] : 0.f;
        }
#pragma unroll
        for(int j = 0; j < 4; ++j) {
            const int m = u0 + j < n ? ms[u0 + j] : 0;
#pragma unroll
            for(int tk = 0; tk < T; ++tk)
                if(m & (1 << tk)) out[tk] = fmaf(ws[u0 + j][tk], p[j][tk], out[tk]);
        }
    }
#pragma unroll
    for(int tk = 0; tk < T; ++tk)
        y[std::size_t(tk) * kHidden + row] = out[tk] + shared_gate[tk] * shared[std::size_t(tk) * kHidden + row];
}

void begin_pass(const ExpertTables& t, cudaStream_t stream) {
    begin_pass_kernel<<<1, 32, 0, stream>>>(t);
    FNL_CUDA(cudaGetLastError());
}
void plan_miss_scales(const ExpertTables& t, int layer, const ExpertPlan* plan, int tokens, cudaStream_t stream) {
    if(layer == kMtpLayer) return;
    plan_miss_slabs_kernel<<<dim3(tokens * kTopK, 3, kScaleSplit), 256, 0, stream>>>(t, layer, plan);
    FNL_CUDA(cudaGetLastError());
    plan_miss_exceptions_kernel<<<dim3(tokens * kTopK, 3), 256, 0, stream>>>(t, layer, plan);
    FNL_CUDA(cudaGetLastError());
}
void expert_gate_up(const ExpertPlan* plan, const float* x, float* act, int tokens, cudaStream_t stream) {
    const dim3 grid(kInter / (kWarps * kRowsPerWarp), tokens * kTopK);
    switch(tokens) {
        case 1: expert_gate_up_kernel<1><<<grid, kWarps * 32, 0, stream>>>(plan, x, act); break;
        case 2: expert_gate_up_kernel<2><<<grid, kWarps * 32, 0, stream>>>(plan, x, act); break;
        case 3: expert_gate_up_kernel<3><<<grid, kWarps * 32, 0, stream>>>(plan, x, act); break;
        case 4: expert_gate_up_kernel<4><<<grid, kWarps * 32, 0, stream>>>(plan, x, act); break;
        default: throw std::invalid_argument("expert_gate_up supports 1..4 tokens");
    }
    FNL_CUDA(cudaGetLastError());
}
void expert_down(const ExpertPlan* plan, const float* act, const float* shared, const float* shared_gate, float* y,
                 int tokens, cudaStream_t stream, float* part) {
    static const bool v1 = std::getenv("FNL_DOWN_V1") != nullptr;
    if(part && !v1) {
        const dim3 grid(kHidden / kDownRowsPerBlock, tokens * kTopK);
        switch(tokens) {
            case 1: expert_down_part_kernel<1><<<grid, kWarps * 32, 0, stream>>>(plan, act, part); break;
            case 2: expert_down_part_kernel<2><<<grid, kWarps * 32, 0, stream>>>(plan, act, part); break;
            case 3: expert_down_part_kernel<3><<<grid, kWarps * 32, 0, stream>>>(plan, act, part); break;
            case 4: expert_down_part_kernel<4><<<grid, kWarps * 32, 0, stream>>>(plan, act, part); break;
            default: throw std::invalid_argument("expert_down supports 1..4 tokens");
        }
        FNL_CUDA(cudaGetLastError());
        switch(tokens) {
            case 1: expert_down_sum_kernel<1><<<kHidden / 128, 128, 0, stream>>>(plan, part, shared, shared_gate, y); break;
            case 2: expert_down_sum_kernel<2><<<kHidden / 128, 128, 0, stream>>>(plan, part, shared, shared_gate, y); break;
            case 3: expert_down_sum_kernel<3><<<kHidden / 128, 128, 0, stream>>>(plan, part, shared, shared_gate, y); break;
            case 4: expert_down_sum_kernel<4><<<kHidden / 128, 128, 0, stream>>>(plan, part, shared, shared_gate, y); break;
        }
        FNL_CUDA(cudaGetLastError());
        return;
    }
    const int blocks = kHidden / kWarps;
    switch(tokens) {
        case 1: expert_down_kernel<1><<<blocks, kWarps * 32, 0, stream>>>(plan, act, shared, shared_gate, y); break;
        case 2: expert_down_kernel<2><<<blocks, kWarps * 32, 0, stream>>>(plan, act, shared, shared_gate, y); break;
        case 3: expert_down_kernel<3><<<blocks, kWarps * 32, 0, stream>>>(plan, act, shared, shared_gate, y); break;
        case 4: expert_down_kernel<4><<<blocks, kWarps * 32, 0, stream>>>(plan, act, shared, shared_gate, y); break;
        default: throw std::invalid_argument("expert_down supports 1..4 tokens");
    }
    FNL_CUDA(cudaGetLastError());
}

}  // namespace fnl
