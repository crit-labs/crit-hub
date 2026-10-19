#include "experts.cuh"

namespace nmh {
namespace {

// One block of 512 threads per token: scores = sigmoid(logits), choice = scores + bias, top-22 of choice by
// repeated argmax (ties to the lower index), weights = scores / sum * 5.
__global__ void __launch_bounds__(kExperts) route_kernel(const float* logits, const float* bias, Route* routes) {
    logits += std::size_t(blockIdx.x) * kExperts;
    Route* route = routes + blockIdx.x;
    __shared__ float choice[kExperts];
    __shared__ float red[32];
    __shared__ int redi[32];
    __shared__ int ids[kTopK];
    const int t = threadIdx.x;
    const float score = sigmoidf_(logits[t]);
    choice[t] = score + bias[t];
    __syncthreads();
    for(int k = 0; k < kTopK; ++k) {
        float v = choice[t];
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
            if(t == 0) ids[k] = idx;
        }
        __syncthreads();
        if(t == ids[k]) choice[t] = -INFINITY;
        __syncthreads();
    }
    // Selected scores in selection order (sum in that order).
    __shared__ float picked[kTopK];
    for(int k = 0; k < kTopK; ++k)
        if(t == ids[k]) picked[k] = score;
    __syncthreads();
    if(t == 0) {
        float sum = 0.f;
        for(int k = 0; k < kTopK; ++k) sum += picked[k];
        const float den = sum + 1e-20f;
        for(int k = 0; k < kTopK; ++k) {
            route->expert[k] = ids[k];
            route->weight[k] = picked[k] / den * kRouteScale;
        }
    }
}

__global__ void begin_pass_kernel(ExpertTables t) {
    if(threadIdx.x == 0 && t.pass_first) *t.pass_first = *t.step + 1;
}

// One block of 128 threads: union of the T tokens' routes in order of first occurrence, hits in parallel; the CPU
// takes the first misses of the pass when they are many enough, the rest go through CLOCK in one warp, 32 slots at a
// time (the same victims as a sequential hand).
__global__ void __launch_bounds__(128) plan_kernel(ExpertTables t, int group, const Route* routes, int tokens,
                                                   ExpertPlan* plan, const float* x) {
    __shared__ int n, nmiss, ncpu;
    __shared__ uint32_t step;
    __shared__ int ue[kMaxUnion], um[kMaxUnion], us[kMaxUnion], miss_list[kMaxUnion];
    __shared__ float uw[kMaxUnion][kMaxPassTokens];
    __shared__ int re[kMaxUnion], first_of[kMaxUnion], pos[kMaxUnion];
    const int tid = threadIdx.x;
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
            if(re[j] == re[tid]) { f = j; break; }
        first_of[tid] = f;
    }
    __syncthreads();
    if(tid == 0) {
        int c = 0;
        for(int i = 0; i < nr; ++i)
            if(first_of[i] == i) pos[i] = c++;
        n = c;
        step = ++(*t.step);
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
        const float* s2 = t.s2 + (std::size_t(group) * kExperts + e) * 2;
        plan->s2[tid][0] = s2[0];
        plan->s2[tid][1] = s2[1];
        const int s = t.nslots ? t.slot_of[group * kExperts + e] : -1;
        us[tid] = s;
        if(s >= 0) {
            t.ref[s] = 1;
            t.stamp[s] = step;
            plan->ref[tid] = {t.slot_addr[s], nullptr, 0, 0, 0, 0};
        }
    }
    if(tid == 0) plan->n = n;
    __syncthreads();
    if(tid == 0) {
        int m = 0;
        for(int u = 0; u < n; ++u)
            if(us[u] < 0) miss_list[m++] = u;
        nmiss = m;
        int c = 0;
        if(t.job && m >= t.cpu_min) c = min(kCpuMaxEntries, int(float(m) * t.cpu_share + 0.5f));
        ncpu = c;
        if(t.stats) {
            t.stats[0] += n;
            t.stats[1] += m;
            t.stats[2] += c;
        }
    }
    __syncthreads();
    // The CPU's share: entries and inputs into the mailbox, then the sequence number.
    if(t.job) {
        CpuJob* job = t.job;
        if(tid < ncpu) {
            const int u = miss_list[tid];
            job->expert[tid] = ue[u];
            job->mask[tid] = um[u];
            const float* s2 = t.s2 + (std::size_t(group) * kExperts + ue[u]) * 2;
            job->s2[tid][0] = s2[0];
            job->s2[tid][1] = s2[1];
            plan->ref[u] = {t.host + (std::size_t(group) * kExperts + ue[u]) * kSlotBytes, nullptr, 2, 0, tid, 0};
        }
        if(ncpu > 0) {
            const float4* xs = reinterpret_cast<const float4*>(x);
            float4* xd = reinterpret_cast<float4*>(&job->x[0][0]);
            for(int i = tid; i < tokens * kLatent / 4; i += blockDim.x) xd[i] = xs[i];
        }
        __threadfence_system();
        __syncthreads();
        if(tid == 0) {
            if(ncpu > 0) {
                job->group = group;
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
    // CLOCK victims for the device's misses (one warp). Slots used earlier in this pass stay protected.
    const uint32_t first = t.pass_first ? *t.pass_first : step;
    if(tid < 32) {
        const int lane = tid;
        int h = t.nslots ? *t.hand : 0;
        for(int i = ncpu; i < nmiss; ++i) {
            const int u = miss_list[i];
            const int e = ue[u];
            const uint8_t* host = t.host + (std::size_t(group) * kExperts + e) * kSlotBytes;
            int victim = -1;
            if(t.nslots) {
                for(int scanned = 0; scanned < 2 * t.nslots + 64 && victim < 0; scanned += 32) {
                    int s = h + lane;
                    while(s >= t.nslots) s -= t.nslots;
                    const bool in_range = lane < t.nslots;
                    const bool eligible = in_range && t.stamp[s] < first && t.valid[s];
                    const bool refd = eligible && t.ref[s];
                    const unsigned pick = __ballot_sync(0xffffffffu, eligible && !refd);
                    const int stop = pick ? __ffs(pick) - 1 : 32;
                    if(refd && lane < stop) t.ref[s] = 0;
                    if(pick) victim = __shfl_sync(0xffffffffu, s, stop);
                    h += pick ? stop + 1 : 32;
                    while(h >= t.nslots) h -= t.nslots;
                    __syncwarp();
                }
            }
            if(lane == 0) {
                ExpertRef r{host, nullptr, 1, 0, 0, 0};
                if(victim >= 0) {
                    const int old = t.owner[victim];
                    if(old >= 0) t.slot_of[old] = -1;
                    t.owner[victim] = group * kExperts + e;
                    t.slot_of[group * kExperts + e] = victim;
                    t.ref[victim] = 1;
                    t.stamp[victim] = step;
                    r.dst = t.slot_addr[victim];
                    r.write_through = 1;
                }
                plan->ref[u] = r;
            }
            __syncwarp();
        }
        if(lane == 0 && t.nslots) *t.hand = h;
    }
}

constexpr int kWarps = 8;
constexpr int kRowsPerBlock = kWarps * 16;   // two threads per row

// The canonical per-row arithmetic shared with the CPU (cpu_experts.cpp): thread h of a row holds accumulators of
// elements 8h .. 8h + 7 of every group, p[e] = fma(w * s, x, p[e]) over the groups in order; the pair then sums the
// 16 accumulators with the fixed tree (p[i] + p[i + 8]) -> + 4 -> + 2 -> + 1.
//
// Weights reach the block in stages of S groups per row: 16-byte chunks loaded row-major (8 or 12 consecutive threads
// per row segment, coalesced also for experts read from mapped host memory) into padded shared memory, written
// through to the new slot when the expert is a miss, then consumed by the row threads.
template <int T, int G, int S>
__device__ __forceinline__ void staged_rows(const uint8_t* w, const uint8_t* sc, uint8_t* wt, uint8_t* st,
                                            const float* xs, int ldx, int mask, float* total, uint4* wsm,
                                            uint32_t* ssm) {
    constexpr int kRowBytes = S * 8;              // weight bytes of a row per stage
    constexpr int kChunks = kRowBytes / 16;       // 16-byte chunks of a row per stage
    constexpr int kStride = kRowBytes / 16 + 1;   // padded row stride in 16-byte units
    constexpr int kScaleWords = S / 4;            // scale words of a row per stage
    constexpr int kPer = kRowsPerBlock * kChunks / (kWarps * 32);
    constexpr int kSPer = (kRowsPerBlock * kScaleWords + kWarps * 32 - 1) / (kWarps * 32);
    static_assert(G % S == 0 && S % 4 == 0 && (kRowsPerBlock * kChunks) % (kWarps * 32) == 0, "stage shape");
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r = warp * 16 + (lane >> 1), h = lane & 1;
    float p[T][8];
#pragma unroll
    for(int t = 0; t < T; ++t)
#pragma unroll
        for(int e = 0; e < 8; ++e) p[t][e] = 0.f;
    for(int g0 = 0; g0 < G; g0 += S) {
        __syncthreads();
#pragma unroll
        for(int i = 0; i < kPer; ++i) {
            const int c = tid + i * kWarps * 32, row = c / kChunks, col = c % kChunks;
            const std::size_t off = std::size_t(row) * (G * 8) + std::size_t(g0) * 8 + col * 16;
            const uint4 v = *reinterpret_cast<const uint4*>(w + off);
            wsm[row * kStride + col] = v;
            if(wt) *reinterpret_cast<uint4*>(wt + off) = v;
        }
#pragma unroll
        for(int i = 0; i < kSPer; ++i) {
            const int c = tid + i * kWarps * 32;
            if(c < kRowsPerBlock * kScaleWords) {
                const int row = c / kScaleWords, col = c % kScaleWords;
                const std::size_t off = std::size_t(row) * G + g0 + col * 4;
                const uint32_t v = *reinterpret_cast<const uint32_t*>(sc + off);
                ssm[row * (kScaleWords + 1) + col] = v;
                if(st) *reinterpret_cast<uint32_t*>(st + off) = v;
            }
        }
        __syncthreads();
        const uint32_t* wrow = reinterpret_cast<const uint32_t*>(wsm + r * kStride);
        const uint8_t* srow = reinterpret_cast<const uint8_t*>(ssm + r * (kScaleWords + 1));
#pragma unroll 4
        for(int g = 0; g < S; ++g) {
            const uint32_t w4 = wrow[g * 2 + h];
            const float scl = e4m3f(srow[g]);
            float ws[8];
#pragma unroll
            for(int b = 0; b < 4; ++b) {
                const float2 v = e2m1x2f((w4 >> (8 * b)) & 0xff);
                ws[2 * b] = v.x * scl;
                ws[2 * b + 1] = v.y * scl;
            }
#pragma unroll
            for(int t = 0; t < T; ++t)
                if(mask & (1 << t)) {
                    const float* xt = xs + t * ldx + (g0 + g) * 16 + 8 * h;
                    const float4 a = *reinterpret_cast<const float4*>(xt);
                    const float4 b = *reinterpret_cast<const float4*>(xt + 4);
                    p[t][0] = fmaf(ws[0], a.x, p[t][0]);
                    p[t][1] = fmaf(ws[1], a.y, p[t][1]);
                    p[t][2] = fmaf(ws[2], a.z, p[t][2]);
                    p[t][3] = fmaf(ws[3], a.w, p[t][3]);
                    p[t][4] = fmaf(ws[4], b.x, p[t][4]);
                    p[t][5] = fmaf(ws[5], b.y, p[t][5]);
                    p[t][6] = fmaf(ws[6], b.z, p[t][6]);
                    p[t][7] = fmaf(ws[7], b.w, p[t][7]);
                }
        }
    }
#pragma unroll
    for(int t = 0; t < T; ++t) {
        float q8[8];
#pragma unroll
        for(int e = 0; e < 8; ++e) {
            const float other = __shfl_xor_sync(0xffffffffu, p[t][e], 1);
            q8[e] = h == 0 ? p[t][e] + other : other + p[t][e];   // p[e] + p[e + 8]
        }
        const float q4[4] = {q8[0] + q8[4], q8[1] + q8[5], q8[2] + q8[6], q8[3] + q8[7]};
        const float q2[2] = {q4[0] + q4[2], q4[1] + q4[3]};
        total[t] = q2[0] + q2[1];
    }
}

constexpr int kUpStage = 16;     // 4 stages of the up rows' 64 groups
constexpr int kDownStage = 24;   // 7 stages of the down rows' 168 groups

// grid (kInter / 128, kMaxUnion), block 256: entry u = blockIdx.y, two threads per row.
template <int T>
__global__ void __launch_bounds__(kWarps * 32) expert_up_kernel(const ExpertPlan* plan, const float* x, float* act) {
    __shared__ __align__(16) float xs[T][kLatent];
    __shared__ uint4 wsm[kRowsPerBlock * (kUpStage * 8 / 16 + 1)];
    __shared__ uint32_t ssm[kRowsPerBlock * (kUpStage / 4 + 1)];
    const int u = blockIdx.y;
    if(u >= plan->n) return;
    const ExpertRef r = plan->ref[u];
    if(r.miss == 2) return;
    const int mask = plan->mask[u];
    for(int i = threadIdx.x; i < T * kLatent / 4; i += blockDim.x)
        reinterpret_cast<float4*>(&xs[0][0])[i] = reinterpret_cast<const float4*>(x)[i];
    const int row0 = blockIdx.x * kRowsPerBlock;
    const uint8_t* w = r.src + std::size_t(row0) * (kLatent / 2);
    const uint8_t* s = r.src + kOffUpS + std::size_t(row0) * (kLatent / 16);
    uint8_t* wt = r.write_through ? r.dst + std::size_t(row0) * (kLatent / 2) : nullptr;
    uint8_t* st = r.write_through ? r.dst + kOffUpS + std::size_t(row0) * (kLatent / 16) : nullptr;
    float total[T];
    staged_rows<T, kLatent / 16, kUpStage>(w, s, wt, st, &xs[0][0], kLatent, mask, total, wsm, ssm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if((lane & 1) == 0) {
        const int row = row0 + warp * 16 + (lane >> 1);
        const float s2 = plan->s2[u][0];
#pragma unroll
        for(int t = 0; t < T; ++t)
            if(mask & (1 << t)) act[(std::size_t(u) * kMaxPassTokens + t) * kInter + row] = relu2f(total[t] * s2);
    }
}

// grid (kLatent / 128, kMaxUnion), block 256: part[u][t][row] = s2 * (down row . act[u][t]).
template <int T>
__global__ void __launch_bounds__(kWarps * 32) expert_down_part_kernel(const ExpertPlan* plan, const float* act,
                                                                       float* part) {
    extern __shared__ __align__(16) uint4 dyn[];
    float* as = reinterpret_cast<float*>(dyn);                       // [T][kInter]
    uint4* wsm = dyn + T * kInter / 4;                               // [128][kDownStage * 8 / 16 + 1]
    uint32_t* ssm = reinterpret_cast<uint32_t*>(wsm + kRowsPerBlock * (kDownStage * 8 / 16 + 1));
    const int u = blockIdx.y;
    if(u >= plan->n) return;
    const ExpertRef r = plan->ref[u];
    if(r.miss == 2) return;
    const int mask = plan->mask[u];
    for(int i = threadIdx.x; i < T * kInter / 4; i += blockDim.x) {
        const int t = i / (kInter / 4), c = i % (kInter / 4);
        if(mask & (1 << t))
            reinterpret_cast<float4*>(as + t * kInter)[c] =
                reinterpret_cast<const float4*>(act + (std::size_t(u) * kMaxPassTokens + t) * kInter)[c];
    }
    const int row0 = blockIdx.x * kRowsPerBlock;
    const uint8_t* w = r.src + kOffDown + std::size_t(row0) * (kInter / 2);
    const uint8_t* s = r.src + kOffDownS + std::size_t(row0) * (kInter / 16);
    uint8_t* wt = r.write_through ? r.dst + kOffDown + std::size_t(row0) * (kInter / 2) : nullptr;
    uint8_t* st = r.write_through ? r.dst + kOffDownS + std::size_t(row0) * (kInter / 16) : nullptr;
    float total[T];
    staged_rows<T, kInter / 16, kDownStage>(w, s, wt, st, as, kInter, mask, total, wsm, ssm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if((lane & 1) == 0) {
        const int row = row0 + warp * 16 + (lane >> 1);
        const float s2 = plan->s2[u][1];
#pragma unroll
        for(int t = 0; t < T; ++t)
            if(mask & (1 << t)) part[(std::size_t(u) * kMaxPassTokens + t) * kLatent + row] = total[t] * s2;
    }
}

template <int T>
constexpr std::size_t down_smem() {
    return std::size_t(T) * kInter * 4 + std::size_t(kRowsPerBlock) * (kDownStage * 8 / 16 + 1) * 16 +
           std::size_t(kRowsPerBlock) * (kDownStage / 4 + 1) * 4;
}

// Waits for the CPU's answer to this layer's job, if one was posted.
__global__ void cpu_wait_kernel(const int* cpu_state, const CpuReply* reply) {
    if(threadIdx.x != 0 || !cpu_state[1]) return;
    const uint32_t seq = uint32_t(cpu_state[0]);
    const volatile uint32_t* p = &reply->seq;
    for(long long i = 0; *p != seq && i < (1ll << 34); ++i) __nanosleep(32);
    __threadfence_system();
}

// y[t][row] = sum over entries in plan order of w_ut * part[u][t][row] (CPU entries from the reply).
template <int T>
__global__ void expert_sum_kernel(const ExpertPlan* plan, const float* part, const CpuReply* reply, float* y) {
    __shared__ float ws[kMaxUnion][T];
    __shared__ int ms[kMaxUnion], cpu[kMaxUnion];
    const int n = plan->n;
    if(threadIdx.x < kMaxUnion) {
        const int u = threadIdx.x;
        ms[u] = u < n ? plan->mask[u] : 0;
        cpu[u] = u < n && plan->ref[u].miss == 2 ? plan->ref[u].cpu : -1;
#pragma unroll
        for(int tk = 0; tk < T; ++tk) ws[u][tk] = plan->weight[u][tk];
    }
    __syncthreads();
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= kLatent) return;
    float out[T];
#pragma unroll
    for(int tk = 0; tk < T; ++tk) out[tk] = 0.f;
    for(int u = 0; u < n; ++u) {
        const int m = ms[u];
        const int k = cpu[u];
#pragma unroll
        for(int tk = 0; tk < T; ++tk)
            if(m & (1 << tk)) {
                const float v = k >= 0 ? *reinterpret_cast<const volatile float*>(&reply->part[k][tk][row])
                                       : part[(std::size_t(u) * kMaxPassTokens + tk) * kLatent + row];
                out[tk] = fmaf(ws[u][tk], v, out[tk]);
            }
    }
#pragma unroll
    for(int tk = 0; tk < T; ++tk) y[std::size_t(tk) * kLatent + row] = out[tk];
}

}  // namespace

void route(const float* logits, const float* bias, Route* routes, int tokens, cudaStream_t stream) {
    route_kernel<<<tokens, kExperts, 0, stream>>>(logits, bias, routes);
    NMH_CUDA(cudaGetLastError());
}
void begin_pass(const ExpertTables& t, cudaStream_t stream) {
    begin_pass_kernel<<<1, 32, 0, stream>>>(t);
    NMH_CUDA(cudaGetLastError());
}
void plan_experts(const ExpertTables& t, int group, const Route* routes, int tokens, ExpertPlan* plan,
                  const float* x, cudaStream_t stream) {
    plan_kernel<<<1, 128, 0, stream>>>(t, group, routes, tokens, plan, x);
    NMH_CUDA(cudaGetLastError());
}

#define NMH_T_SWITCH(T_, CALL)                                         \
    switch(T_) {                                                       \
        case 1: { constexpr int TT = 1; CALL; } break;                \
        case 2: { constexpr int TT = 2; CALL; } break;                \
        case 3: { constexpr int TT = 3; CALL; } break;                \
        case 4: { constexpr int TT = 4; CALL; } break;                \
        default: throw std::invalid_argument("experts: 1..4 tokens");  \
    }

void expert_up(const ExpertPlan* plan, const float* x, float* act, int tokens, cudaStream_t stream) {
    const dim3 grid(kInter / kRowsPerBlock, tokens * kTopK);
    NMH_T_SWITCH(tokens, (expert_up_kernel<TT><<<grid, kWarps * 32, 0, stream>>>(plan, x, act)));
    NMH_CUDA(cudaGetLastError());
}
void expert_down(const ExpertTables& t, const ExpertPlan* plan, const float* act, float* part, float* y, int tokens,
                 cudaStream_t stream) {
    const dim3 grid(kLatent / kRowsPerBlock, tokens * kTopK);
    NMH_T_SWITCH(tokens, ({
        NMH_CUDA(cudaFuncSetAttribute(expert_down_part_kernel<TT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      int(down_smem<TT>())));
        expert_down_part_kernel<TT><<<grid, kWarps * 32, down_smem<TT>(), stream>>>(plan, act, part);
    }));
    NMH_CUDA(cudaGetLastError());
    if(t.job) {
        cpu_wait_kernel<<<1, 32, 0, stream>>>(t.cpu_state, t.reply);
        NMH_CUDA(cudaGetLastError());
    }
    NMH_T_SWITCH(tokens, (expert_sum_kernel<TT><<<kLatent / 128, 128, 0, stream>>>(plan, part, t.reply, y)));
    NMH_CUDA(cudaGetLastError());
}

}  // namespace nmh
