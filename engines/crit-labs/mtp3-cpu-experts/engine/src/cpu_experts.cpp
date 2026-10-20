#include "cpu_experts.h"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>

namespace nmh {
namespace {

alignas(64) const float kE2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                     -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

struct E4m3Table {
    float v[256];
    E4m3Table() {
        for(int b = 0; b < 256; ++b) {
            const int sign = b >> 7, exp = (b >> 3) & 15, man = b & 7;
            float x;
            if(exp == 0) x = std::ldexp(float(man) / 8.f, -6);
            else if(exp == 15 && man == 7) x = 0.f;   // NaN encoding; never a stored scale
            else x = std::ldexp(1.f + float(man) / 8.f, exp - 7);
            v[b] = sign ? -x : x;
        }
    }
};
const E4m3Table kE4m3;

// 16 E2M1 values of one group (8 bytes, low nibble first) times the group's scale.
inline __m512 group16(const uint8_t* p, __m512 lut, float scale) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i lo = _mm_and_si128(b, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
    const __m512i idx = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi));
    return _mm512_mul_ps(_mm512_permutexvar_ps(idx, lut), _mm512_set1_ps(scale));
}

// The fixed tree over the 16 accumulators: (p[i] + p[i+8]), then + 4, + 2, + 1.
inline float tree16(__m512 p) {
    const __m256 q8 = _mm256_add_ps(_mm512_castps512_ps256(p), _mm512_extractf32x8_ps(p, 1));
    const __m128 q4 = _mm_add_ps(_mm256_castps256_ps128(q8), _mm256_extractf128_ps(q8, 1));
    const __m128 q2 = _mm_add_ps(q4, _mm_movehl_ps(q4, q4));
    return _mm_cvtss_f32(_mm_add_ss(q2, _mm_shuffle_ps(q2, q2, 1)));
}

inline float relu2(float x) {
    const float r = x > 0.f ? x : 0.f;
    return r * r;
}

// rows [row0, row1) of a matrix with `groups` groups of 16 per row: out[t][r] = f(tree(p_t) * s2). Four rows run
// interleaved so their accumulator chains overlap; each row's operations and order are unchanged.
template <int T, bool Relu>
void rows_t(const uint8_t* w, const uint8_t* s, int groups, float s2, const float* x, int ldx, int mask, int row0,
            int row1, float* out, int ldo) {
    constexpr int R = 4;
    const __m512 lut = _mm512_load_ps(kE2m1);
    int r = row0;
    for(; r + R <= row1; r += R) {
        const uint8_t* wr[R];
        const uint8_t* sr[R];
        __m512 p[R][T];
        for(int i = 0; i < R; ++i) {
            wr[i] = w + std::size_t(r + i) * groups * 8;
            sr[i] = s + std::size_t(r + i) * groups;
            for(int t = 0; t < T; ++t) p[i][t] = _mm512_setzero_ps();
        }
        for(int g = 0; g < groups; ++g) {
            __m512 xv[T];
            for(int t = 0; t < T; ++t) xv[t] = _mm512_loadu_ps(x + std::size_t(t) * ldx + g * 16);
            for(int i = 0; i < R; ++i) {
                const __m512 ws = group16(wr[i] + g * 8, lut, kE4m3.v[sr[i][g]]);
                for(int t = 0; t < T; ++t)
                    if(mask & (1 << t)) p[i][t] = _mm512_fmadd_ps(ws, xv[t], p[i][t]);
            }
        }
        for(int i = 0; i < R; ++i)
            for(int t = 0; t < T; ++t)
                if(mask & (1 << t)) {
                    const float v = tree16(p[i][t]) * s2;
                    out[std::size_t(t) * ldo + r + i] = Relu ? relu2(v) : v;
                }
    }
    for(; r < row1; ++r) {
        const uint8_t* wr = w + std::size_t(r) * groups * 8;
        const uint8_t* sr = s + std::size_t(r) * groups;
        __m512 p[T];
        for(int t = 0; t < T; ++t) p[t] = _mm512_setzero_ps();
        for(int g = 0; g < groups; ++g) {
            const __m512 ws = group16(wr + g * 8, lut, kE4m3.v[sr[g]]);
            for(int t = 0; t < T; ++t)
                if(mask & (1 << t)) p[t] = _mm512_fmadd_ps(ws, _mm512_loadu_ps(x + std::size_t(t) * ldx + g * 16), p[t]);
        }
        for(int t = 0; t < T; ++t)
            if(mask & (1 << t)) {
                const float v = tree16(p[t]) * s2;
                out[std::size_t(t) * ldo + r] = Relu ? relu2(v) : v;
            }
    }
}

template <bool Relu>
void rows(const uint8_t* w, const uint8_t* s, int groups, float s2, const float* x, int ldx, int tokens, int mask,
          int row0, int row1, float* out, int ldo) {
    switch(tokens) {
        case 1: rows_t<1, Relu>(w, s, groups, s2, x, ldx, mask, row0, row1, out, ldo); break;
        case 2: rows_t<2, Relu>(w, s, groups, s2, x, ldx, mask, row0, row1, out, ldo); break;
        case 3: rows_t<3, Relu>(w, s, groups, s2, x, ldx, mask, row0, row1, out, ldo); break;
        default: rows_t<4, Relu>(w, s, groups, s2, x, ldx, mask, row0, row1, out, ldo); break;
    }
}

inline void cpu_pause() { _mm_pause(); }

}  // namespace

void cpu_expert_up_rows(const uint8_t* record, float s2_up, const float* x, int tokens, int mask, int row0, int row1,
                        float* act) {
    rows<true>(record, record + kOffUpS, kLatent / 16, s2_up, x, kLatent, tokens, mask, row0, row1, act, kInter);
}
void cpu_expert_down_rows(const uint8_t* record, float s2_down, const float* act, int tokens, int mask, int row0,
                          int row1, float* part) {
    rows<false>(record + kOffDown, record + kOffDownS, kInter / 16, s2_down, act, kInter, tokens, mask, row0, row1,
                part, kLatent);
}

namespace {
constexpr int kUpBlock = 336;    // 8 up blocks of an expert's 2688 rows
constexpr int kDownBlock = 128;  // 8 down blocks of its 1024 rows
constexpr int kUpBlocks = kInter / kUpBlock, kDownBlocks = kLatent / kDownBlock;
}  // namespace

namespace {
float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if(exp == 0) {
        if(man == 0) {
            bits = sign;
        } else {
            exp = 113;   // normalize the subnormal
            while(!(man & 0x400)) {
                man <<= 1;
                --exp;
            }
            bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    } else if(exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
}  // namespace

void cpu_prompt_experts(const uint8_t* records, int group, const float* s2, const std::vector<CpuPromptExpert>& list,
                        const int* row_token, const uint16_t* x, float* out, int threads) {
    std::atomic<int> next{0};
    auto work = [&] {
        std::vector<float> xs(std::size_t(kCpuMaxTokens) * kLatent), act(std::size_t(kCpuMaxTokens) * kInter),
            part(std::size_t(kCpuMaxTokens) * kLatent);
        for(int i; (i = next.fetch_add(1)) < int(list.size());) {
            const CpuPromptExpert& e = list[i];
            const std::size_t key = std::size_t(group) * kExperts + e.expert;
            const uint8_t* rec = records + key * kSlotBytes;
            for(int t0 = 0; t0 < e.count; t0 += kCpuMaxTokens) {
                const int n = std::min(kCpuMaxTokens, e.count - t0);
                for(int t = 0; t < n; ++t) {
                    const uint16_t* src = x + std::size_t(row_token[e.first + t0 + t]) * kLatent;
                    for(int c = 0; c < kLatent; ++c) xs[std::size_t(t) * kLatent + c] = half_to_float(src[c]);
                }
                const int mask = (1 << n) - 1;
                cpu_expert_up_rows(rec, s2[key * 2], xs.data(), n, mask, 0, kInter, act.data());
                cpu_expert_down_rows(rec, s2[key * 2 + 1], act.data(), n, mask, 0, kLatent, part.data());
                for(int t = 0; t < n; ++t)
                    std::memcpy(out + std::size_t(e.first + t0 + t) * kLatent, part.data() + std::size_t(t) * kLatent,
                                sizeof(float) * kLatent);
            }
        }
    };
    std::vector<std::thread> pool;
    for(int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for(auto& t : pool) t.join();
}

CpuMissServer::CpuMissServer(CpuJob* job, CpuReply* reply, const uint8_t* records, int threads)
    : job_(job), reply_(reply), records_(records), threads_(threads), snap_(std::make_unique<CpuJob>()),
      act_(std::size_t(kCpuMaxEntries) * kCpuMaxTokens * kInter) {
    for(int i = 0; i < threads_; ++i) pool_.emplace_back([this, i] { run(i); });
}

CpuMissServer::~CpuMissServer() {
    stop_ = true;
    for(auto& t : pool_) t.join();
}

// All threads meet. arrived_ only grows: barrier `round` of job generation g (from 1) waits for (3 (g - 1) + round) arrivals of
// every thread, so a thread still leaving one barrier never sees a counter reset.
void CpuMissServer::barrier(int gen, int round) {
    const long long target = (3LL * (gen - 1) + round) * threads_;
    arrived_.fetch_add(1);
    while(arrived_.load(std::memory_order_acquire) < target) cpu_pause();
}

void CpuMissServer::run(int id) {
    uint32_t seen = 0;
    int idle = 0;
    // The generation this thread last worked on, kept locally: re-reading generation_ here could already see the next
    // job thread 0 posted, and the thread would then wait for the one after it.
    int gen = 0;
    for(;;) {
        // Thread 0 polls the mailbox and publishes a snapshot; the others wait for the generation to change.
        if(id == 0) {
            const uint32_t seq = job_->seq;
            if(seq == seen) {
                if(stop_) {
                    generation_.store(-1, std::memory_order_release);
                    return;
                }
                if(++idle > 20000) {
                    timespec ts{0, 20000};
                    nanosleep(&ts, nullptr);
                } else {
                    cpu_pause();
                }
                continue;
            }
            idle = 0;
            std::atomic_thread_fence(std::memory_order_acquire);
            CpuJob& j = *snap_;
            j.group = job_->group;
            j.tokens = job_->tokens;
            j.count = job_->count;
            for(int k = 0; k < j.count; ++k) {
                j.expert[k] = job_->expert[k];
                j.mask[k] = job_->mask[k];
                j.s2[k][0] = job_->s2[k][0];
                j.s2[k][1] = job_->s2[k][1];
            }
            std::memcpy(j.x, const_cast<const float*>(job_->x[0]), sizeof(float) * kLatent * j.tokens);
            j.seq = seq;
            seen = seq;
            next_item_.store(0, std::memory_order_relaxed);
            posted_.store(seq, std::memory_order_relaxed);
            generation_.store(gen + 1, std::memory_order_release);
            gen = gen + 1;
        } else {
            int g;
            int spins = 0;
            while((g = generation_.load(std::memory_order_acquire)) == gen) {
                if(++spins > 20000) {
                    timespec ts{0, 20000};
                    nanosleep(&ts, nullptr);
                } else {
                    cpu_pause();
                }
            }
            if(g < 0) return;
            gen = g;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const CpuJob& j = *snap_;
        const int up_items = j.count * kUpBlocks;
        for(int it; (it = next_item_.fetch_add(1)) < up_items;) {
            const int k = it / kUpBlocks, b = it % kUpBlocks;
            const uint8_t* rec = records_ + (std::size_t(j.group) * kExperts + j.expert[k]) * kSlotBytes;
            cpu_expert_up_rows(rec, j.s2[k][0], j.x[0], j.tokens, j.mask[k], b * kUpBlock, (b + 1) * kUpBlock,
                               act_.data() + std::size_t(k) * kCpuMaxTokens * kInter);
        }
        barrier(gen, 1);
        if(id == 0) next_item_.store(0, std::memory_order_release);
        barrier(gen, 2);
        const int down_items = j.count * kDownBlocks;
        for(int it; (it = next_item_.fetch_add(1)) < down_items;) {
            const int k = it / kDownBlocks, b = it % kDownBlocks;
            const uint8_t* rec = records_ + (std::size_t(j.group) * kExperts + j.expert[k]) * kSlotBytes;
            cpu_expert_down_rows(rec, j.s2[k][1], act_.data() + std::size_t(k) * kCpuMaxTokens * kInter, j.tokens,
                                 j.mask[k], b * kDownBlock, (b + 1) * kDownBlock, reply_->part[k][0]);
        }
        barrier(gen, 3);
        if(id == 0) {
            _mm_sfence();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            reply_->seq = j.seq;
            jobs_ += 1;
            experts_ += j.count;
            busy_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        }
    }
}

}  // namespace nmh
