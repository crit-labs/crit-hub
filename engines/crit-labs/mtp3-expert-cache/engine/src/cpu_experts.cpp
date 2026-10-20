#include "cpu_experts.h"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace fnl {

namespace {
constexpr int kHid = 2560, kInt = 640;
constexpr std::size_t kGateW = std::size_t(kInt) * kHid / 2, kDownW = std::size_t(kHid) * kInt / 2;
constexpr std::size_t kCsfOffset = 2 * kGateW + kDownW;
constexpr std::size_t kGateCsfBytes = (kInt / 16) * 16 * (1 + kHid / 32);

struct Tables {
    float e4m3[256];
    alignas(64) int8_t e2m1x2[64];   // doubled E2M1 values for every nibble, repeated over 4 lanes of 16
    Tables() {
        for(int b = 0; b < 256; ++b) {
            const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
            float v;
            if(e == 15 && m == 7) v = NAN;
            else if(e == 0) v = std::ldexp(float(m) / 8.f, -6);
            else v = std::ldexp(1.f + float(m) / 8.f, e - 7);
            e4m3[b] = s ? -v : v;
        }
        const int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        for(int i = 0; i < 64; ++i) {
            const int n = i & 15;
            e2m1x2[i] = int8_t((n & 8) ? -mag[n & 7] : mag[n & 7]);
        }
    }
};
const Tables& tables() {
    static const Tables t;
    return t;
}

// Decoded E4M3 scale values (times 0.5, undoing the doubled E2M1 table) of rows [r0, r1) of one matrix.
// Half of the E4M3 values of 16 bytes held in dword lanes (no NaN among stored scales).
inline __m512 e4m3_half(__m512i b) {
    const __m512i e = _mm512_and_si512(_mm512_srli_epi32(b, 3), _mm512_set1_epi32(15));
    const __m512i m = _mm512_and_si512(b, _mm512_set1_epi32(7));
    const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(128)), 24);
    // normal: 2^(e - 7) (1 + m / 8) / 2 -> exponent field e + 119
    const __m512i nbits = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(e, _mm512_set1_epi32(119)), 23),
                                                          _mm512_slli_epi32(m, 20)), sign);
    // subnormal: m 2^-9 / 2
    const __m512 sub = _mm512_castsi512_ps(_mm512_or_si512(
        _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(m), _mm512_set1_ps(1.f / 1024.f))), sign));
    const __mmask16 is_sub = _mm512_cmpeq_epi32_mask(e, _mm512_setzero_si512());
    return _mm512_mask_blend_ps(is_sub, _mm512_castsi512_ps(nbits), sub);
}

void row_scales(const uint8_t* fixed, const uint32_t* exc, uint32_t nexc, int cols, int r0, int r1, float* out) {
    const auto& T = tables();
    const int half = cols / 2, slab_bytes = 16 * (1 + half);
    const __m512i lo_idx = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    const __m512i hi_idx = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
    const __m512i k15 = _mm512_set1_epi32(15), k255 = _mm512_set1_epi32(255);
    for(int r = r0; r < r1; ++r) {
        const uint8_t* slab = fixed + std::size_t(r / 16) * slab_bytes;
        const __m512i base = _mm512_set1_epi32(slab[r % 16]);
        const uint8_t* nib = slab + 16 + (r % 16) * half;
        float* o = out + std::size_t(r - r0) * cols;
        for(int c = 0; c < half; c += 16) {
            const int n = std::min(16, half - c);
            const __mmask16 k = __mmask16((1u << n) - 1);
            const __m512i d = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(k, nib + c));
            const __m512 lo = e4m3_half(_mm512_and_si512(_mm512_add_epi32(_mm512_and_si512(d, k15), base), k255));
            const __m512 hi = e4m3_half(_mm512_and_si512(_mm512_add_epi32(_mm512_srli_epi32(d, 4), base), k255));
            // o[2c'] = lo[c'], o[2c' + 1] = hi[c']
            const __m512 a = _mm512_permutex2var_ps(lo, lo_idx, hi);
            const __m512 b = _mm512_permutex2var_ps(lo, hi_idx, hi);
            if(n == 16) {
                _mm512_storeu_ps(o + 2 * c, a);
                _mm512_storeu_ps(o + 2 * c + 16, b);
            } else {
                const int m2 = 2 * n;
                _mm512_mask_storeu_ps(o + 2 * c, __mmask16(m2 >= 16 ? 0xffff : (1u << m2) - 1), a);
                if(m2 > 16) _mm512_mask_storeu_ps(o + 2 * c + 16, __mmask16((1u << (m2 - 16)) - 1), b);
            }
        }
    }
    const uint32_t lo = uint32_t(r0) * uint32_t(cols), hi = uint32_t(r1) * uint32_t(cols);
    const uint32_t* first = std::lower_bound(exc, exc + nexc, lo, [](uint32_t w, uint32_t pos) {
        return (w & 0xffffffu) < pos;
    });
    for(const uint32_t* p = first; p < exc + nexc && (*p & 0xffffffu) < hi; ++p)
        out[(*p & 0xffffffu) - lo] = 0.5f * T.e4m3[*p >> 24];
}

inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

// Dots of one NVFP4 row (K/2 bytes, low nibble = even element) with the masked tokens' x (even and odd halves);
// scales per 16. The row is decoded once for every token.
template <int T>
inline void row_dots_t(const uint8_t* w, const float* scales, int K, const float (*xe)[1280], const float (*xo)[1280],
                       int mask, float* out) {
    // Doubled E2M1 values as floats; vpermps reads only the low 4 bits of each index lane, so a dword holding a
    // whole weight byte selects its low nibble, and the byte shifted right by 4 its high nibble.
    static const __m512 lut = _mm512_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, -0.f, -1.f, -2.f, -3.f, -4.f,
                                             -6.f, -8.f, -12.f);
    __m512 acc[T], acc2[T];
#pragma GCC unroll 4
    for(int t = 0; t < T; ++t) acc[t] = acc2[t] = _mm512_setzero_ps();
    const int chunks = K / 32;   // 16 weight bytes = 32 elements = 2 scale groups
    __m512 S = _mm512_setzero_ps();
    for(int c = 0; c < chunks; ++c) {
        const int k = c & 7;
        if(k == 0) {
            // Scales of the next 8 chunks (16 groups).
            const int g = 2 * c, n = std::min(16, K / 16 - g);
            S = _mm512_maskz_loadu_ps(__mmask16(n >= 16 ? 0xffff : (1u << n) - 1), scales + g);
        }
        const int j = 16 * c;   // byte index of lane 0: elements 2j (low) and 2j + 1 (high)
        const __m512i d = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)));
        const __m512 lf = _mm512_permutexvar_ps(d, lut);
        const __m512 hf = _mm512_permutexvar_ps(_mm512_srli_epi32(d, 4), lut);
        // lanes 0..7 take group 2k of S, lanes 8..15 group 2k + 1
        const __m512i si = _mm512_add_epi32(_mm512_set1_epi32(2 * k),
                                            _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0));
        const __m512 sv = _mm512_permutexvar_ps(si, S);
        const __m512 lfs = _mm512_mul_ps(lf, sv), hfs = _mm512_mul_ps(hf, sv);
#pragma GCC unroll 4
        for(int t = 0; t < T; ++t) {
            if(T > 1 && !(mask & (1 << t))) continue;
            acc[t] = _mm512_fmadd_ps(lfs, _mm512_loadu_ps(xe[t] + j), acc[t]);
            acc2[t] = _mm512_fmadd_ps(hfs, _mm512_loadu_ps(xo[t] + j), acc2[t]);
        }
    }
#pragma GCC unroll 4
    for(int t = 0; t < T; ++t) acc[t] = _mm512_add_ps(acc[t], acc2[t]);
#pragma GCC unroll 4
    for(int t = 0; t < T; ++t) out[t] = (mask & (1 << t)) ? hsum(acc[t]) : 0.f;
}
inline void row_dots(const uint8_t* w, const float* scales, int K, const float (*xe)[1280], const float (*xo)[1280],
                     int tokens, int mask, float* out) {
    switch(tokens) {
        case 1: row_dots_t<1>(w, scales, K, xe, xo, mask, out); break;
        case 2: row_dots_t<2>(w, scales, K, xe, xo, mask, out); break;
        case 3: row_dots_t<3>(w, scales, K, xe, xo, mask, out); break;
        default: row_dots_t<4>(w, scales, K, xe, xo, mask, out); break;
    }
}

inline float silu(float v) { return v / (1.f + std::exp(-v)); }
}  // namespace

CpuExperts::CpuExperts(int threads) {
    for(int i = 1; i < threads; ++i) workers_.emplace_back([this, i] { worker(i); });
}

CpuExperts::~CpuExperts() {
    {
        std::lock_guard<std::mutex> g(mutex_);
        stop_ = true;
        ++generation_;
    }
    wake_.notify_all();
    for(auto& t : workers_) t.join();
}

void CpuExperts::barrier() {
    const std::uint64_t phase = phase_.load(std::memory_order_acquire);
    if(arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == threads()) {
        arrived_.store(0, std::memory_order_relaxed);
        phase_.store(phase + 1, std::memory_order_release);
    } else {
        while(phase_.load(std::memory_order_acquire) == phase) _mm_pause();
    }
}

void CpuExperts::worker(int id) {
    std::uint64_t seen = 0;
    for(;;) {
        {
            // Spin briefly for the next job (decode passes post one per layer), then sleep.
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(2000);
            bool got = false;
            while(std::chrono::steady_clock::now() < until) {
                if(__atomic_load_n(&generation_, __ATOMIC_ACQUIRE) != seen) {
                    got = true;
                    break;
                }
                for(int i = 0; i < 64; ++i) _mm_pause();
            }
            std::unique_lock<std::mutex> lock(mutex_);
            if(!got) wake_.wait(lock, [&] { return generation_ != seen; });
            if(stop_) return;
            seen = generation_;
        }
        work(id);
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void CpuExperts::run(const float* x, int tokens, const CpuExpert* experts, int count, float* y) {
    if(count <= 0) {
        std::memset(y, 0, std::size_t(tokens) * kHid * sizeof(float));
        return;
    }
    experts_ = experts;
    count_ = count;
    tokens_ = tokens;
    y_ = y;
    for(int t = 0; t < tokens; ++t)
        for(int i = 0; i < kHid / 2; ++i) {
            xe_[t][i] = x[std::size_t(t) * kHid + 2 * i];
            xo_[t][i] = x[std::size_t(t) * kHid + 2 * i + 1];
        }
    if(act_.size() < std::size_t(count) * tokens * kInt) act_.resize(std::size_t(count) * tokens * kInt);
    done_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> g(mutex_);
        __atomic_add_fetch(&generation_, 1, __ATOMIC_RELEASE);
    }
    wake_.notify_all();
    work(0);
    while(done_.load(std::memory_order_acquire) != int(workers_.size())) _mm_pause();
}

void CpuExperts::run_each(const CpuRowJob* jobs, int count) {
    if(count <= 0) return;
    row_jobs_ = jobs;
    row_count_ = count;
    row_next_.store(0, std::memory_order_relaxed);
    count_ = -1;   // marks a row-job generation for work()
    done_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> g(mutex_);
        __atomic_add_fetch(&generation_, 1, __ATOMIC_RELEASE);
    }
    wake_.notify_all();
    work_rows();
    while(done_.load(std::memory_order_acquire) != int(workers_.size())) _mm_pause();
}

// One thread computes a whole expert for its job's tokens.
void CpuExperts::work_rows() {
    thread_local std::vector<float> sc, act;
    alignas(64) static thread_local float xe[kCpuMaxTokens][1280], xo[kCpuMaxTokens][1280];
    alignas(64) static thread_local float ae[kCpuMaxTokens][1280], ao[kCpuMaxTokens][1280];
    sc.resize(std::size_t(kInt) * (kHid / 16) * 2 + std::size_t(kHid) * (kInt / 16));
    for(;;) {
        const int j = row_next_.fetch_add(1, std::memory_order_acq_rel);
        if(j >= row_count_) return;
        const CpuRowJob& job = row_jobs_[j];
        const CpuExpert& e = job.expert;
        const int T = std::min(job.tokens, kCpuMaxTokens);
        const int mask = (1 << T) - 1;
        for(int t = 0; t < T; ++t)
            for(int i = 0; i < kHid / 2; ++i) {
                xe[t][i] = job.x[t][2 * i];
                xo[t][i] = job.x[t][2 * i + 1];
            }
        float* sg = sc.data();
        float* su = sg + std::size_t(kInt) * (kHid / 16);
        float* sd = su + std::size_t(kInt) * (kHid / 16);
        row_scales(e.record + kCsfOffset, e.exc[0], e.exc_count[0], kHid / 16, 0, kInt, sg);
        row_scales(e.record + kCsfOffset + kGateCsfBytes, e.exc[1], e.exc_count[1], kHid / 16, 0, kInt, su);
        for(int r = 0; r < kInt; ++r) {
            float g[kCpuMaxTokens], u[kCpuMaxTokens];
            row_dots(e.record + std::size_t(r) * (kHid / 2), sg + std::size_t(r) * (kHid / 16), kHid, xe, xo, T, mask, g);
            row_dots(e.record + kGateW + std::size_t(r) * (kHid / 2), su + std::size_t(r) * (kHid / 16), kHid, xe, xo, T,
                     mask, u);
            for(int t = 0; t < T; ++t) {
                const float a = silu(g[t] * e.s2[0]) * (u[t] * e.s2[1]);
                if(r & 1) ao[t][r / 2] = a;
                else ae[t][r / 2] = a;
            }
        }
        row_scales(e.record + kCsfOffset + 2 * kGateCsfBytes, e.exc[2], e.exc_count[2], kInt / 16, 0, kHid, sd);
        for(int r = 0; r < kHid; ++r) {
            float d[kCpuMaxTokens];
            row_dots(e.record + 2 * kGateW + std::size_t(r) * (kInt / 2), sd + std::size_t(r) * (kInt / 16), kInt, ae, ao,
                     T, mask, d);
            for(int t = 0; t < T; ++t) job.y[t][r] = d[t] * e.s2[2];
        }
    }
}

void CpuExperts::work(int id) {
    if(count_ < 0) {
        work_rows();
        return;
    }
    const int nt = threads();
    const int T = tokens_;
    thread_local std::vector<float> sc;
    // Phase 1: gate and up rows of every expert -> activations (even/odd halves for phase 2).
    const int g0 = kInt * id / nt, g1 = kInt * (id + 1) / nt;
    sc.resize(std::size_t(std::max(g1 - g0, 1)) * (kHid / 16) * 2 + std::size_t(kHid) * (kInt / 16));
    for(int j = 0; j < count_; ++j) {
        const CpuExpert& e = experts_[j];
        float* sg = sc.data();
        float* su = sg + std::size_t(g1 - g0) * (kHid / 16);
        row_scales(e.record + kCsfOffset, e.exc[0], e.exc_count[0], kHid / 16, g0, g1, sg);
        row_scales(e.record + kCsfOffset + kGateCsfBytes, e.exc[1], e.exc_count[1], kHid / 16, g0, g1, su);
        for(int r = g0; r < g1; ++r) {
            float g[kCpuMaxTokens], u[kCpuMaxTokens];
            row_dots(e.record + std::size_t(r) * (kHid / 2), sg + std::size_t(r - g0) * (kHid / 16), kHid, xe_, xo_, T,
                     e.mask, g);
            row_dots(e.record + kGateW + std::size_t(r) * (kHid / 2), su + std::size_t(r - g0) * (kHid / 16), kHid, xe_,
                     xo_, T, e.mask, u);
            for(int t = 0; t < T; ++t) {
                if(!(e.mask & (1 << t))) continue;
                // Stored as even/odd halves: element i at (i & 1) * kInt / 2 + i / 2.
                float* act = act_.data() + (std::size_t(j) * T + t) * kInt;
                act[(r & 1) * (kInt / 2) + r / 2] = silu(g[t] * e.s2[0]) * (u[t] * e.s2[1]);
            }
        }
    }
    barrier();
    // Phase 2: down rows owned by this thread, experts summed in list order.
    const int d0 = kHid * id / nt, d1 = kHid * (id + 1) / nt;
    float* sd = sc.data();
    for(int t = 0; t < T; ++t)
        for(int r = d0; r < d1; ++r) y_[std::size_t(t) * kHid + r] = 0.f;
    for(int j = 0; j < count_; ++j) {
        const CpuExpert& e = experts_[j];
        row_scales(e.record + kCsfOffset + 2 * kGateCsfBytes, e.exc[2], e.exc_count[2], kInt / 16, d0, d1, sd);
        // Activations of this expert as [token][even | odd]; row_dots wants per-token even/odd arrays.
        float xa_e[kCpuMaxTokens][1280], xa_o[kCpuMaxTokens][1280];
        for(int t = 0; t < T; ++t) {
            if(!(e.mask & (1 << t))) continue;
            const float* act = act_.data() + (std::size_t(j) * T + t) * kInt;
            std::memcpy(xa_e[t], act, (kInt / 2) * sizeof(float));
            std::memcpy(xa_o[t], act + kInt / 2, (kInt / 2) * sizeof(float));
        }
        for(int r = d0; r < d1; ++r) {
            float d[kCpuMaxTokens];
            row_dots(e.record + 2 * kGateW + std::size_t(r) * (kInt / 2), sd + std::size_t(r - d0) * (kInt / 16), kInt,
                     xa_e, xa_o, T, e.mask, d);
            for(int t = 0; t < T; ++t)
                if(e.mask & (1 << t))
                    y_[std::size_t(t) * kHid + r] = std::fma(e.weight[t] * e.s2[2], d[t], y_[std::size_t(t) * kHid + r]);
        }
    }
}

CpuMissServer::CpuMissServer(CpuJob* job, CpuReply* reply, Describe describe, Insert insert, int threads)
    : job_(job), reply_(reply), describe_(std::move(describe)), insert_(std::move(insert)), pool_(threads) {
    snapshot_ = std::make_unique<CpuJob>();
    thread_ = std::thread([this] { serve(); });
}

CpuMissServer::~CpuMissServer() {
    stop_ = true;
    thread_.join();
}

void CpuMissServer::serve() {
    uint32_t last = job_->seq;
    std::vector<CpuExpert> list(kCpuMaxEntries);
    auto idle_since = std::chrono::steady_clock::now();
    while(!stop_.load(std::memory_order_relaxed)) {
        const uint32_t seq = job_->seq;
        if(seq == last) {
            // Spin while decode passes keep coming; back off after a quiet second.
            if(std::chrono::steady_clock::now() - idle_since > std::chrono::seconds(1))
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            else
                _mm_pause();
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto c0 = std::chrono::steady_clock::now();
        const int n = std::min(job_->count, kCpuMaxEntries);
        const int T = std::min(job_->tokens, kCpuMaxTokens);
        for(int i = 0; i < n; ++i) {
            describe_(job_->layer, job_->expert[i], list[i]);
            list[i].mask = job_->mask[i];
            for(int t = 0; t < kCpuMaxTokens; ++t) list[i].weight[t] = job_->weight[i][t];
        }
        pool_.run(&job_->x[0][0], T, list.data(), n, &reply_->y[0][0]);
        // The device posts its next job as soon as this reply lands: keep what the insertion needs first.
        CpuJob& snap = *snapshot_;
        snap.layer = job_->layer;
        snap.count = n;
        for(int i = 0; i < n; ++i) {
            snap.expert[i] = job_->expert[i];
            snap.slot[i] = job_->slot[i];
        }
        std::atomic_thread_fence(std::memory_order_release);
        reply_->seq = seq;
        last = seq;
        busy_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - c0).count(),
                           std::memory_order_relaxed);
        const auto i0 = std::chrono::steady_clock::now();
        if(insert_) insert_(snap);
        insert_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - i0).count(),
                             std::memory_order_relaxed);
        issued_.store(seq, std::memory_order_release);
        jobs_.fetch_add(1, std::memory_order_relaxed);
        experts_.fetch_add(n, std::memory_order_relaxed);
        idle_since = std::chrono::steady_clock::now();
    }
}

}  // namespace fnl
