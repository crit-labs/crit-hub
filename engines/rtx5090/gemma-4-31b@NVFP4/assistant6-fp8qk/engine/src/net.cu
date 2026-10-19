#include "net.h"

#include <crit/safetensors.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "gemv.cuh"

namespace g4 {
float encode_nvfp4(const __nv_bfloat16* bf, const std::uint8_t* f8, const float* sc, int rows, int cols,
                   std::uint8_t* packed, std::uint8_t* sf);
namespace knet {

using crit::core::Checkpoint;
using crit::core::Memory;

float bf16_to_float(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ------------------------------------------------------------------------------------------------ small kernels
// BF16 rows -> E4M3 codes with scale = row amax / 448 (1 for a zero row), round to nearest even, saturating.
__global__ void requant_fp8_rows_kernel(const __nv_bfloat16* __restrict__ in, int cols, std::uint8_t* __restrict__ codes,
                                        float* __restrict__ scale) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    const __nv_bfloat16* row = in + static_cast<std::size_t>(r) * cols;
    float am = 0.f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) am = fmaxf(am, fabsf(__bfloat162float(row[i])));
    for(int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    float m = 0.f;
    for(int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) m = fmaxf(m, red[i]);
    const float sc = m > 0.f ? m / 448.f : 1.f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x)
        codes[static_cast<std::size_t>(r) * cols + i] =
            static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(__bfloat162float(row[i]) / sc, __NV_SATFINITE, __NV_E4M3));
    if(threadIdx.x == 0) scale[r] = sc;
}

// Drafter input: hidden of the column the drafter continues from (ctl[hcol] - 1) of the last pass.
__global__ void gather_hidden_kernel(const int* ctl, const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst) {
    pdl_wait();
    pdl_trigger();
    const int col = max(ctl[ctl::hcol] - 1, 0);
    for(int i = threadIdx.x; i < kHidden / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src + static_cast<std::size_t>(col) * kHidden)[i];
}
__global__ void verify_tokens_kernel(int* ctl, const int* drafts, int K, int* tok) {
    pdl_wait();
    pdl_trigger();
    const int i = threadIdx.x;
    if(i == 0) tok[0] = ctl[ctl::last];
    else if(i <= K) tok[i] = drafts[i - 1];
}

// Greedy (or replayed) acceptance.
__global__ void accept_kernel(int* ctl, const int* drafts, const int* pred, int K, const int* stops, const int* forced,
                              int forced_n, int* result) {
    pdl_wait();
    pdl_trigger();
    if(threadIdx.x != 0) return;
    const int budget = ctl[ctl::budget];
    const int nstop = stops[16];
    int fa = ctl[ctl::forced_at];
    int out = 0, matched = 0;
    bool stopped = false;
    int e[kMaxCols + 1];
    auto is_stop = [&](int t) {
        for(int i = 0; i < nstop; ++i)
            if(stops[i] == t) return true;
        return false;
    };
    auto target = [&](int i) { return forced ? (fa + i < forced_n ? forced[fa + i] : -1) : pred[i]; };
    while(matched < K && out < budget && drafts[matched] == target(matched)) {
        e[out++] = drafts[matched++];
        if(is_stop(e[out - 1])) { stopped = true; break; }
    }
    if(out < budget && !stopped) {
        e[out++] = target(matched);
        if(is_stop(e[out - 1])) stopped = true;
    }
    result[0] = out;
    for(int i = 0; i <= kMaxCols; ++i) result[1 + i] = i < out ? e[i] : -1;
    ctl[ctl::pos] += out;
    ctl[ctl::last] = out ? e[out - 1] : ctl[ctl::last];
    ctl[ctl::emitted] = out;
    ctl[ctl::hcol] = out;
    if(forced) ctl[ctl::forced_at] = fa + out;
    ctl[ctl::stopped] = stopped ? 1 : 0;
}

__global__ void truncate_kernel(int* ctl, int* result, int accepted) {
    pdl_wait();
    pdl_trigger();
    const int out = result[0];
    if(accepted >= out) return;
    ctl[ctl::pos] -= out - accepted;
    ctl[ctl::hcol] = accepted;
    ctl[ctl::last] = accepted ? result[accepted] : ctl[ctl::last];
    result[0] = accepted;
}

__global__ void prefill_done_kernel(int* ctl, const int* pred, int P) {
    pdl_wait();
    pdl_trigger();
    ctl[ctl::pos] = P;
    ctl[ctl::last] = pred[0];
    ctl[ctl::hcol] = 1;
    ctl[ctl::forced_at] = 0;
    ctl[ctl::stopped] = 0;
}

// Temperature / top-k / top-p sampling (Philox-style counter from seed and position).
constexpr int kTopK = 64, kLocal = 8;
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float uniform01(unsigned long long seed, int pos, int stream) {
    const unsigned long long z = mix64(seed + 0x9E3779B97F4A7C15ull * (static_cast<unsigned long long>(pos) * 4 + stream + 1));
    return static_cast<float>(z >> 40) * (1.f / 16777216.f);
}
__device__ __forceinline__ unsigned long long sampling_seed(const float* params) {
    return (static_cast<unsigned long long>(__float_as_uint(params[4])) << 32) | __float_as_uint(params[3]);
}

// Top-k (<= 64) candidates of one logits row per block with temperature, top-p and renormalization.
__global__ void __launch_bounds__(256) topk_probs_kernel(const float* __restrict__ logits, int ld, const float* __restrict__ params,
                                                         int* __restrict__ ids, float* __restrict__ probs, int* __restrict__ cnt) {
    pdl_wait();
    pdl_trigger();
    __shared__ float cv[256 * kLocal];
    __shared__ int ci[256 * kLocal];
    __shared__ float tv[kTopK];
    __shared__ int ti[kTopK];
    __shared__ float sv[8];
    __shared__ int si[8];
    __shared__ int need_full;
    __shared__ int sp[8];
    const int col = blockIdx.x;
    const float* row = logits + static_cast<std::size_t>(col) * ld;
    const float temperature = params[0], top_p = params[1];
    const int k = min(max(static_cast<int>(params[2]), 1), kTopK);
    float lv[kLocal];
    int li[kLocal];
#pragma unroll
    for(int j = 0; j < kLocal; ++j) { lv[j] = -INFINITY; li[j] = 0x7fffffff; }
    for(int i = threadIdx.x; i < kVocabValid; i += blockDim.x) {
        const float v = row[i];
        if(v > lv[kLocal - 1]) {
            int j = kLocal - 1;
            while(j > 0 && v > lv[j - 1]) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j; }
            lv[j] = v; li[j] = i;
        }
    }
#pragma unroll
    for(int j = 0; j < kLocal; ++j) { cv[threadIdx.x * kLocal + j] = lv[j]; ci[threadIdx.x * kLocal + j] = li[j]; }
    if(threadIdx.x == 0) need_full = 0;
    __syncthreads();
    auto select = [&](const float* vals, const int* idx, int n, int r) {
        float best = -INFINITY;
        int bi = 0x7fffffff, bp = -1;
        for(int e = threadIdx.x; e < n; e += blockDim.x) {
            const float v = vals[e];
            const int id = idx[e];
            if(v > best || (v == best && id < bi)) { best = v; bi = id; bp = e; }
        }
        for(int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const int op = __shfl_xor_sync(0xffffffffu, bp, o);
            if(ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; bp = op; }
        }
        if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bi; sp[threadIdx.x >> 5] = bp; }
        __syncthreads();
        if(threadIdx.x == 0) {
            int w0 = 0;
            for(int w = 1; w < 8; ++w)
                if(sv[w] > sv[w0] || (sv[w] == sv[w0] && si[w] < si[w0])) w0 = w;
            tv[r] = sv[w0];
            ti[r] = si[w0];
            if(sp[w0] >= 0) cv[sp[w0]] = -INFINITY;
        }
        __syncthreads();
    };
    for(int r = 0; r < k; ++r) select(cv, ci, 256 * kLocal, r);
    if(lv[kLocal - 1] > tv[k - 1] || (lv[kLocal - 1] == tv[k - 1] && li[kLocal - 1] < ti[k - 1])) need_full = 1;
    __syncthreads();
    if(need_full) {
        for(int r = 0; r < k; ++r) {
            float best = -INFINITY;
            int bi = 0x7fffffff;
            for(int i = threadIdx.x; i < kVocabValid; i += blockDim.x) {
                const float v = row[i];
                bool taken = false;
                for(int j = 0; j < r; ++j) taken |= ti[j] == i;
                if(!taken && (v > best || (v == best && i < bi))) { best = v; bi = i; }
            }
            for(int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if(ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bi; }
            __syncthreads();
            if(threadIdx.x == 0) {
                int w0 = 0;
                for(int w = 1; w < 8; ++w)
                    if(sv[w] > sv[w0] || (sv[w] == sv[w0] && si[w] < si[w0])) w0 = w;
                tv[r] = sv[w0];
                ti[r] = si[w0];
            }
            __syncthreads();
        }
    }
    if(threadIdx.x != 0) return;
    int* oid = ids + col * kTopK;
    float* op = probs + col * kTopK;
    if(temperature <= 0.f) {
        oid[0] = ti[0];
        op[0] = 1.f;
        cnt[col] = 1;
        return;
    }
    float p[kTopK], sum = 0.f;
    for(int r = 0; r < k; ++r) { p[r] = expf((tv[r] - tv[0]) / temperature); sum += p[r]; }
    float cum = 0.f;
    int keep = k;
    for(int r = 0; r < k; ++r) {
        cum += p[r] / sum;
        if(cum >= top_p) { keep = r + 1; break; }
    }
    float ksum = 0.f;
    for(int r = 0; r < keep; ++r) ksum += p[r];
    for(int r = 0; r < keep; ++r) { oid[r] = ti[r]; op[r] = p[r] / ksum; }
    cnt[col] = keep;
}

__device__ int draw(const int* ids, const float* probs, int n, int exclude, float u) {
    float total = 0.f;
    for(int r = 0; r < n; ++r)
        if(ids[r] != exclude) total += probs[r];
    if(total <= 0.f) return ids[0];
    float acc = 0.f, target = u * total;
    int last = ids[0];
    for(int r = 0; r < n; ++r) {
        if(ids[r] == exclude) continue;
        last = ids[r];
        acc += probs[r];
        if(target < acc) return ids[r];
    }
    return last;
}

__global__ void sample_first_kernel(const int* ids, const float* probs, const int* cnt, const float* params, int pos, int* out) {
    pdl_wait();
    pdl_trigger();
    *out = draw(ids, probs, cnt[0], -1, uniform01(sampling_seed(params), pos, 1));
}

// Sampled acceptance with point-mass drafts: draft i is accepted with probability p_i(d_i); on rejection the column is
// resampled without d_i; after all K a bonus is drawn from column K.
__global__ void accept_sampled_kernel(int* ctl, const int* drafts, int K, const int* ids, const float* probs, const int* cnt,
                                      const float* params, const int* stops, int* result) {
    pdl_wait();
    pdl_trigger();
    if(threadIdx.x != 0) return;
    const int budget = ctl[ctl::budget], nstop = stops[16], pos = ctl[ctl::pos];
    const unsigned long long seed = sampling_seed(params);
    auto is_stop = [&](int t) {
        for(int i = 0; i < nstop; ++i)
            if(stops[i] == t) return true;
        return false;
    };
    int e[kMaxCols + 1], out = 0, i = 0;
    bool stopped = false, rejected = false;
    for(; i < K && out < budget; ++i) {
        const int* cid = ids + i * kTopK;
        const float* cp = probs + i * kTopK;
        float pd = 0.f;
        for(int r = 0; r < cnt[i]; ++r)
            if(cid[r] == drafts[i]) pd = cp[r];
        if(uniform01(seed, pos + out, 0) < pd) {
            e[out++] = drafts[i];
            if(is_stop(drafts[i])) { stopped = true; break; }
        } else {
            e[out++] = draw(cid, cp, cnt[i], drafts[i], uniform01(seed, pos + out, 1));
            rejected = true;
            if(is_stop(e[out - 1])) stopped = true;
            break;
        }
    }
    if(!rejected && !stopped && out < budget) {
        e[out++] = draw(ids + i * kTopK, probs + i * kTopK, cnt[i], -1, uniform01(seed, pos + out, 1));
        if(is_stop(e[out - 1])) stopped = true;
    }
    result[0] = out;
    for(int j = 0; j <= kMaxCols; ++j) result[1 + j] = j < out ? e[j] : -1;
    ctl[ctl::pos] += out;
    ctl[ctl::last] = out ? e[out - 1] : ctl[ctl::last];
    ctl[ctl::emitted] = out;
    ctl[ctl::hcol] = out;
    ctl[ctl::stopped] = stopped ? 1 : 0;
}

}  // namespace knet
using namespace knet;

void launch_sample_first(const float* logits, const float* params, int pos, int* ids, float* probs, int* cnt, int* out,
                         cudaStream_t s) {
    launch_k(topk_probs_kernel, 1, 256, 0, s, logits, kVocab, params, ids, probs, cnt);
    launch_k(sample_first_kernel, 1, 1, 0, s, ids, probs, cnt, params, pos, out);
}
void launch_prefill_done(int* ctl, const int* pred, int P, cudaStream_t s) {
    launch_k(prefill_done_kernel, 1, 1, 0, s, ctl, pred, P);
    KLAUNCH();
}

bool pdl_enabled() {
    static const bool on = !std::getenv("G4_NO_PDL");
    return on;
}

// ------------------------------------------------------------------------------------------------ construction
Net::Net(std::shared_ptr<crit::core::CudaBackend> backend, const std::string& model_dir, const std::string& draft_dir,
         const Config& config)
    : backend_(std::move(backend)), cfg_(config) {
    if(cfg_.draft < 0 || cfg_.draft > kMaxCols - 1) throw std::invalid_argument("draft must be 0..7");
    if(cfg_.prefill_chunk < 128 || cfg_.prefill_chunk + kWindow + kMaxCols > cfg_.ring)
        throw std::invalid_argument("prefill_chunk must be >= 128 and fit the sliding ring with the window");
    if(const char* e = std::getenv("G4_ROW_FP4")) cfg_.row_fp4 = e[0] == '1';
    if(const char* e = std::getenv("G4_HEAD_FP4")) cfg_.head_fp4 = e[0] == '1';
    if(const char* e = std::getenv("G4_PREFILL_Q8")) cfg_.prefill_q8 = e[0] == '1';
    if(const char* e = std::getenv("G4_PREFILL_QB")) cfg_.prefill_qb = e[0] == '1';
    stream_ = backend_->stream(crit::core::Queue::compute);
    round_done_ = backend_->event();
    load(model_dir, draft_dir);
    alloc_buffers();
    reset();
    KCHECK(cudaStreamSynchronize(stream_));
}

Net::~Net() {
    drop_graphs();
    if(stream_) cudaStreamSynchronize(stream_);
}

void* Net::dalloc(std::size_t bytes) {
    bytes = (bytes + 255) / 256 * 256;
    owned_.push_back(backend_->allocate(Memory::device, bytes));
    device_bytes_ += bytes;
    return owned_.back().data();
}

void Net::load(const std::string& dir, const std::string& draft_dir) {
    crit::core::CheckpointOptions opt;
    const Checkpoint ck = Checkpoint::open(dir, opt);
    auto upload = [&](void* dst, const void* src, std::size_t n) { KCHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)); };
    auto value = [&](const Checkpoint& c, const std::string& name) -> float {
        const auto& t = c.at(name);
        const auto v = c.map(t);
        if(t.dtype == crit::core::DType::f32) return *reinterpret_cast<const float*>(v.bytes.data());
        if(t.dtype == crit::core::DType::bf16) return bf16_to_float(*reinterpret_cast<const std::uint16_t*>(v.bytes.data()));
        throw std::runtime_error("unexpected scalar dtype: " + name);
    };
    auto f32 = [&](const Checkpoint& c, const std::string& name) -> const float* {
        const auto& t = c.at(name);
        const auto v = c.map(t);
        const std::size_t n = t.elements();
        std::vector<float> h(n);
        const auto* b = reinterpret_cast<const std::uint16_t*>(v.bytes.data());
        for(std::size_t i = 0; i < n; ++i)
            h[i] = t.dtype == crit::core::DType::f32 ? reinterpret_cast<const float*>(v.bytes.data())[i] : bf16_to_float(b[i]);
        float* d = static_cast<float*>(dalloc(n * 4));
        upload(d, h.data(), n * 4);
        return d;
    };
    // Device staging of BF16 rows.
    __nv_bfloat16* tmp = nullptr;
    std::size_t tmp_bytes = 0;
    auto stage = [&](std::size_t nbytes) {
        if(nbytes > tmp_bytes) {
            if(tmp) KCHECK(cudaFree(tmp));
            KCHECK(cudaMalloc(&tmp, nbytes));
            tmp_bytes = nbytes;
        }
        return tmp;
    };
    // BF16 matrices concatenated by rows -> FP8 per row.
    auto fp8_cat = [&](const Checkpoint& c, const std::vector<std::string>& names) {
        int rows = 0, cols = 0;
        for(auto& n : names) {
            const auto& t = c.at(n + ".weight");
            if(t.dtype != crit::core::DType::bf16) throw std::runtime_error("expected BF16: " + n);
            rows += static_cast<int>(t.shape[0]);
            cols = static_cast<int>(t.shape[1]);
        }
        __nv_bfloat16* st = stage(static_cast<std::size_t>(rows) * cols * 2);
        std::size_t off = 0;
        for(auto& n : names) {
            const auto v = c.map(c.at(n + ".weight"));
            upload(reinterpret_cast<std::uint8_t*>(st) + off, v.bytes.data(), v.bytes.size());
            off += v.bytes.size();
        }
        auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(rows) * cols));
        auto* s = static_cast<float*>(dalloc(static_cast<std::size_t>(rows) * 4));
        requant_fp8_rows_kernel<<<rows, 256>>>(st, cols, w, s);
        KLAUNCH();
        KCHECK(cudaDeviceSynchronize());
        bytes_per_pass_ += static_cast<std::size_t>(rows) * cols + static_cast<std::size_t>(rows) * 4;
        return Fp8Mat{w, s, rows, cols};
    };
    auto interleave = [&](const std::uint8_t* a, const std::uint8_t* b, std::size_t R, std::size_t row_bytes, std::uint8_t* out) {
        for(std::size_t blk = 0; blk < R / 8; ++blk) {
            std::memcpy(out + (blk * 16) * row_bytes, a + blk * 8 * row_bytes, 8 * row_bytes);
            std::memcpy(out + (blk * 16 + 8) * row_bytes, b + blk * 8 * row_bytes, 8 * row_bytes);
        }
    };
    auto swizzle = [&](const std::uint8_t* sf, int rows, int cols, std::uint8_t* dev) {
        const int blocks = cols / 16, bp = sf_blocks_padded(cols);
        std::vector<std::uint8_t> h(sf_bytes(rows, cols), 0);
        for(int r = 0; r < rows; ++r)
            for(int cc = 0; cc < blocks; ++cc) h[sf_offset(r, cc, bp)] = sf[static_cast<std::size_t>(r) * blocks + cc];
        upload(dev, h.data(), h.size());
    };
    // ModelOpt NVFP4 projection (b empty) or gate/up pair interleaved in 8-row blocks.
    auto fp4_mat = [&](const std::string& a, const std::string& b) {
        const auto& ta = ck.at(a + ".weight");
        if(ta.dtype != crit::core::DType::u8) throw std::runtime_error("expected packed NVFP4: " + a);
        const auto wa = ck.map(ta);
        const auto sa = ck.map(ck.at(a + ".weight_scale"));
        const float ws2 = value(ck, a + ".weight_scale_2"), is = value(ck, a + ".input_scale");
        int rows = static_cast<int>(ta.shape[0]);
        const int cols = static_cast<int>(ta.shape[1]) * 2;
        std::vector<std::uint8_t> packed, sf;
        const std::uint8_t* pw = reinterpret_cast<const std::uint8_t*>(wa.bytes.data());
        const std::uint8_t* ps = reinterpret_cast<const std::uint8_t*>(sa.bytes.data());
        if(!b.empty()) {
            const auto wb = ck.map(ck.at(b + ".weight"));
            const auto sb = ck.map(ck.at(b + ".weight_scale"));
            if(value(ck, b + ".weight_scale_2") != ws2 || value(ck, b + ".input_scale") != is)
                throw std::runtime_error("gate/up scales differ: " + a);
            packed.resize(static_cast<std::size_t>(rows) * 2 * cols / 2);
            sf.resize(static_cast<std::size_t>(rows) * 2 * cols / 16);
            interleave(pw, reinterpret_cast<const std::uint8_t*>(wb.bytes.data()), rows, cols / 2, packed.data());
            interleave(ps, reinterpret_cast<const std::uint8_t*>(sb.bytes.data()), rows, cols / 16, sf.data());
            rows *= 2;
            pw = packed.data();
            ps = sf.data();
        }
        auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(rows) * cols / 2));
        upload(w, pw, static_cast<std::size_t>(rows) * cols / 2);
        auto* s = static_cast<std::uint8_t*>(dalloc(sf_bytes(rows, cols)));
        swizzle(ps, rows, cols, s);
        bytes_per_pass_ += static_cast<std::size_t>(rows) * cols / 2 + static_cast<std::size_t>(rows) * cols / 16;
        Fp4Mat m{w, s, ws2, rows, cols};
        m.igs = 1.f / is;
        const float as = m.alpha / m.igs;
        auto* ad = static_cast<float*>(dalloc(8));
        upload(ad, &as, 4);
        upload(ad + 1, &m.alpha, 4);
        m.alpha_static = ad;
        m.alpha_dev = ad + 1;
        return m;
    };
    auto scalar_dev = [&](float v) {
        auto* d = static_cast<float*>(dalloc(4));
        upload(d, &v, 4);
        return static_cast<const float*>(d);
    };

    const std::string P = "model.language_model.";
    layers_.resize(kLayers);
    for(int L = 0; L < kLayers; ++L) {
        auto& lw = layers_[L];
        const std::string p = P + "layers." + std::to_string(L) + ".";
        lw.global = is_global_layer(L);
        lw.in_norm = f32(ck, p + "input_layernorm.weight");
        lw.post_attn = f32(ck, p + "post_attention_layernorm.weight");
        lw.pre_ff = f32(ck, p + "pre_feedforward_layernorm.weight");
        lw.post_ff = f32(ck, p + "post_feedforward_layernorm.weight");
        lw.q_norm = f32(ck, p + "self_attn.q_norm.weight");
        lw.k_norm = f32(ck, p + "self_attn.k_norm.weight");
        lw.scalar_host = value(ck, p + "layer_scalar");
        lw.scalar = scalar_dev(lw.scalar_host);
        if(lw.global) lw.qkv = fp8_cat(ck, {p + "self_attn.q_proj", p + "self_attn.k_proj"});
        else lw.qkv = fp8_cat(ck, {p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"});
        lw.o = fp8_cat(ck, {p + "self_attn.o_proj"});
        lw.gu = fp4_mat(p + "mlp.gate_proj", p + "mlp.up_proj");
        lw.down = fp4_mat(p + "mlp.down_proj", "");
    }
    final_norm_ = f32(ck, P + "norm.weight");
    embed_scale_ = std::sqrt(static_cast<float>(kHidden));
    {
        // Tied output head: FP8 per-row re-encoding in row chunks; the BF16 table stays in mapped host memory for lookups.
        const auto& t = ck.at(P + "embed_tokens.weight");
        const auto v = ck.map(t);
        owned_.push_back(backend_->allocate(Memory::host, v.bytes.size()));
        std::memcpy(owned_.back().data(), v.bytes.data(), v.bytes.size());
        embed_ = static_cast<const __nv_bfloat16*>(owned_.back().device_address());
        if(cfg_.head_fp4) {
            // NVFP4 (per-tensor global scale, E4M3 block scales of 16) from the whole BF16 table on the device.
            __nv_bfloat16* st = stage(v.bytes.size());
            upload(st, v.bytes.data(), v.bytes.size());
            auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(kVocab) * kHidden / 2));
            auto* sf = static_cast<std::uint8_t*>(dalloc(sf_bytes(kVocab, kHidden)));
            const float alpha = encode_nvfp4(st, nullptr, nullptr, kVocab, kHidden, w, sf);
            head4_ = Fp4Mat{w, sf, alpha, kVocab, kHidden};
            bytes_per_pass_ += static_cast<std::size_t>(kVocab) * kHidden / 2 + static_cast<std::size_t>(kVocab) * kHidden / 16;
            KCHECK(cudaFree(tmp));
            tmp = nullptr;
            tmp_bytes = 0;
        } else {
            auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(kVocab) * kHidden));
            auto* s = static_cast<float*>(dalloc(static_cast<std::size_t>(kVocab) * 4));
            constexpr int kRows = 16384;
            __nv_bfloat16* st = stage(static_cast<std::size_t>(kRows) * kHidden * 2);
            for(int r0 = 0; r0 < kVocab; r0 += kRows) {
                upload(st, reinterpret_cast<const std::uint8_t*>(v.bytes.data()) + static_cast<std::size_t>(r0) * kHidden * 2,
                       static_cast<std::size_t>(kRows) * kHidden * 2);
                requant_fp8_rows_kernel<<<kRows, 256>>>(st, kHidden, w + static_cast<std::size_t>(r0) * kHidden, s + r0);
                KLAUNCH();
                KCHECK(cudaDeviceSynchronize());
            }
            head_ = Fp8Mat{w, s, kVocab, kHidden};
            bytes_per_pass_ += static_cast<std::size_t>(kVocab) * kHidden + static_cast<std::size_t>(kVocab) * 4;
        }
    }

    // Drafter (google/gemma-4-31B-it-assistant, as released): BF16 matrices re-encoded to NVFP4 for drafting.
    if(cfg_.draft > 0) {
        const Checkpoint dk = Checkpoint::open(draft_dir, opt);
        auto drafter4 = [&](const std::vector<std::string>& names, bool gu) {
            int rows = 0, cols = 0;
            std::vector<Checkpoint::View> parts;
            for(auto& n : names) {
                const auto& t = dk.at(n + ".weight");
                parts.push_back(dk.map(t));
                rows += static_cast<int>(t.shape[0]);
                cols = static_cast<int>(t.shape[1]);
            }
            const std::size_t nbytes = static_cast<std::size_t>(rows) * cols * 2;
            __nv_bfloat16* st = stage(nbytes);
            if(gu) {
                std::vector<std::uint8_t> host(nbytes);
                interleave(reinterpret_cast<const std::uint8_t*>(parts[0].bytes.data()),
                           reinterpret_cast<const std::uint8_t*>(parts[1].bytes.data()), rows / 2, static_cast<std::size_t>(cols) * 2,
                           host.data());
                upload(st, host.data(), nbytes);
            } else {
                std::size_t off = 0;
                for(auto& v : parts) {
                    upload(reinterpret_cast<std::uint8_t*>(st) + off, v.bytes.data(), v.bytes.size());
                    off += v.bytes.size();
                }
            }
            auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(rows) * cols / 2));
            auto* sf = static_cast<std::uint8_t*>(dalloc(sf_bytes(rows, cols)));
            const float alpha = encode_nvfp4(st, nullptr, nullptr, rows, cols, w, sf);
            draft_bytes_ += static_cast<std::size_t>(rows) * cols / 2 + static_cast<std::size_t>(rows) * cols / 16;
            return Fp4Mat{w, sf, alpha, rows, cols};
        };
        dw_.pre = drafter4({"pre_projection"}, false);
        dw_.post = drafter4({"post_projection"}, false);
        dw_.head = drafter4({"model.embed_tokens"}, false);
        for(int l = 0; l < kDLayers; ++l) {
            auto& d = dw_.layers[l];
            const std::string p = "model.layers." + std::to_string(l) + ".";
            d.global = dk.at(p + "self_attn.q_norm.weight").shape[0] == static_cast<std::uint64_t>(kGHd);
            d.in_norm = f32(dk, p + "input_layernorm.weight");
            d.post_attn = f32(dk, p + "post_attention_layernorm.weight");
            d.pre_ff = f32(dk, p + "pre_feedforward_layernorm.weight");
            d.post_ff = f32(dk, p + "post_feedforward_layernorm.weight");
            d.q_norm = f32(dk, p + "self_attn.q_norm.weight");
            d.scalar = scalar_dev(value(dk, p + "layer_scalar"));
            d.q = drafter4({p + "self_attn.q_proj"}, false);
            d.o = drafter4({p + "self_attn.o_proj"}, false);
            d.gu = drafter4({p + "mlp.gate_proj", p + "mlp.up_proj"}, true);
            d.down = drafter4({p + "mlp.down_proj"}, false);
        }
        dw_.norm = f32(dk, "model.norm.weight");
        if(!dw_.layers[kDLayers - 1].global || dw_.layers[0].global) throw std::runtime_error("unexpected drafter layer types");
    }
    if(tmp) KCHECK(cudaFree(tmp));

    // RoPE inverse frequencies as the reference computes them in FP32: 1 / base ** (arange(0, 2n, 2) / head_dim).
    std::vector<float> is(kSHd / 2), ig(kGRopePairs);
    for(int i = 0; i < kSHd / 2; ++i) is[i] = 1.f / std::pow(10000.f, static_cast<float>(2 * i) / static_cast<float>(kSHd));
    for(int i = 0; i < kGRopePairs; ++i) ig[i] = 1.f / std::pow(1000000.f, static_cast<float>(2 * i) / static_cast<float>(kGHd));
    inv_s_ = static_cast<float*>(dalloc(is.size() * 4));
    upload(inv_s_, is.data(), is.size() * 4);
    inv_g_ = static_cast<float*>(dalloc(ig.size() * 4));
    upload(inv_g_, ig.data(), ig.size() * 4);
}

SlideCache Net::slide_cache(int layer) const { return slide_[layer]; }
GlobalCache Net::global_cache(int layer) const { return glob_[layer]; }

void Net::alloc_buffers() {
    const std::size_t ctx = static_cast<std::size_t>(cfg_.context) + 64;
    // cos / sin of every position's full-attention RoPE angles, shared by the 10 full-attention layers.
    rope_ = static_cast<float2*>(dalloc(ctx * kGRopePairs * sizeof(float2)));
    fill_rope_table(rope_, inv_g_, static_cast<int>(ctx), stream_);
    slide_.assign(kLayers, SlideCache{});
    glob_.assign(kLayers, GlobalCache{});
    for(int L = 0; L < kLayers; ++L) {
        if(layers_[L].global) {
            auto& g = glob_[L];
            g.u = static_cast<std::uint8_t*>(dalloc(ctx * kGKvRows));
            g.us = static_cast<float*>(dalloc(ctx * kGKv * 4));
            g.kw = layers_[L].k_norm;
            g.inv = inv_g_;
            g.rope = rope_;
            if(cfg_.prefill_q8) {
                g.kr = static_cast<std::uint8_t*>(dalloc(ctx * kGKv * 2 * kGRopePairs));
                g.krs = static_cast<float*>(dalloc(ctx * kGKv * 4));
            }
        } else {
            auto& c = slide_[L];
            const std::size_t R = cfg_.ring;
            c.ring = cfg_.ring;
            c.k = static_cast<std::uint8_t*>(dalloc(R * kSKvRows));
            c.v = static_cast<std::uint8_t*>(dalloc(R * kSKvRows));
            c.ks = static_cast<float*>(dalloc(R * kSKv * 4));
            c.vs = static_cast<float*>(dalloc(R * kSKv * 4));
        }
    }
    ctl_ = static_cast<int*>(dalloc(ctl::fields * 4));
    const int C = kMaxCols;
    d_hidden_ = static_cast<float*>(dalloc(C * kHidden * 4));
    d_xn_ = static_cast<__nv_bfloat16*>(dalloc(C * kHidden * 2));
    d_proj_ = static_cast<float*>(dalloc(C * kGQkv * 4));
    d_q_ = static_cast<__nv_bfloat16*>(dalloc(C * kGQ * 2));
    d_attn_ = static_cast<float*>(dalloc(C * kGQ * 4));
    d_attn_h_ = static_cast<__nv_bfloat16*>(dalloc(C * kGQ * 2));
    d_mlp_ = static_cast<__nv_bfloat16*>(dalloc(C * kInter * 2));
    d_v_ = static_cast<float*>(dalloc(C * kHidden * 4));
    d_partial_ = static_cast<float*>(dalloc(std::max(slide_partial_floats(cfg_.slide_blocks), global_partial_floats(cfg_.global_blocks)) * 4));
    d_logits_ = static_cast<float*>(dalloc(static_cast<std::size_t>(C) * kVocab * 4));
    d_final_ = static_cast<__nv_bfloat16*>(dalloc(C * kHidden * 2));
    d_tok_ = static_cast<int*>(dalloc(64 * 4));
    d_amx_ = static_cast<float2*>(dalloc(kMaxCols * kArgParts * sizeof(float2)));
    d_pred_ = static_cast<int*>(dalloc(64 * 4));
    d_draft_ = static_cast<int*>(dalloc(64 * 4));
    d_result_ = static_cast<int*>(dalloc(64 * 4));
    d_stops_ = static_cast<int*>(dalloc(32 * 4));
    d_sampling_ = static_cast<float*>(dalloc(8 * 4));
    d_cand_ids_ = static_cast<int*>(dalloc(kMaxCols * 64 * 4));
    d_cand_p_ = static_cast<float*>(dalloc(kMaxCols * 64 * 4));
    d_cand_n_ = static_cast<int*>(dalloc(kMaxCols * 4));
    d_demb_ = static_cast<float*>(dalloc(kHidden * 4));
    d_din_ = static_cast<__nv_bfloat16*>(dalloc(2 * kHidden * 2));
    d_dx_ = static_cast<float*>(dalloc(kDHidden * 4));
    d_dxn_ = static_cast<__nv_bfloat16*>(dalloc(kDHidden * 2));
    d_dh_ = static_cast<float*>(dalloc(kHidden * 4));
    result_pinned_ = backend_->allocate(Memory::host, 64 * 4);
    result_host_ = static_cast<int*>(result_pinned_.data());
    KCHECK(cudaMemset(d_stops_, 0, 32 * 4));

    // Prompt buffers.
    const int T = (cfg_.prefill_chunk + 127) / 128 * 128;
    chunk_pad_ = T;
    const std::size_t tt = static_cast<std::size_t>(T);
    p_tok_ = static_cast<int*>(dalloc((tt + 1) * 4));
    p_hidden_ = static_cast<float*>(dalloc(tt * kHidden * 4));
    p_xq_ = static_cast<std::uint8_t*>(dalloc(tt * kHidden));
    p_xs_ = static_cast<float*>(dalloc(tt * 4));
    p_raw_ = static_cast<__nv_bfloat16*>(dalloc(tt * 2 * kInter * 2));
    p_raw2_ = static_cast<__nv_bfloat16*>(dalloc(tt * kHidden * 2));
    p_q_ = static_cast<__nv_bfloat16*>(dalloc(tt * kGQ * 2));
    p_attn_ = static_cast<float*>(dalloc(tt * kGQ * 4));
    p_aq_ = static_cast<std::uint8_t*>(dalloc(tt * kGQ));
    p_as_ = static_cast<float*>(dalloc(tt * 4));
    p_x4_ = static_cast<std::uint8_t*>(dalloc(tt * kHidden / 2));
    p_x4sf_ = static_cast<std::uint8_t*>(dalloc(sf_bytes(T, kHidden)));
    p_hq_ = static_cast<std::uint8_t*>(dalloc(tt * kInter / 2));
    p_rs1_ = static_cast<float*>(dalloc(tt * 4));
    p_rs2_ = static_cast<float*>(dalloc(tt * 4));
    p_h4sf_ = static_cast<std::uint8_t*>(dalloc(sf_bytes(T, kInter)));
    p_final_ = static_cast<__nv_bfloat16*>(dalloc(tt * kHidden * 2));
    const std::size_t ws = 64ull << 20;
    lt_ = std::make_unique<Lt>(dalloc(ws), ws, cfg_.gemm_table);
}

void Net::reset() { KCHECK(cudaMemsetAsync(ctl_, 0, ctl::fields * 4, stream_)); }

// ------------------------------------------------------------------------------------------------ target pass
void Net::target_pass(int N, cudaStream_t s) {
    const Pos pos{ctl_ + ctl::pos, 0};
    embed(d_tok_, N, embed_, kHidden, embed_scale_, d_hidden_, kHidden, s);
    rmsnorm_bf16(d_hidden_, kHidden, layers_[0].in_norm, N, kHidden, d_xn_, kHidden, s);
    for(int L = 0; L < kLayers; ++L) {
        const auto& lw = layers_[L];
        const int qrows = lw.global ? kGQ : kSQ;
        gemv_fp8(lw.qkv, d_xn_, kHidden, N, {GemvEpi::store, d_proj_, lw.qkv.rows}, 1, nullptr, s);
        const Proj qkv{d_proj_, nullptr, nullptr, nullptr, lw.qkv.rows};
        AttnRows ar;
        ar.q = d_q_; ar.q_stride = qrows; ar.out = d_attn_; ar.out_stride = qrows; ar.T = N; ar.pos0_dev = pos.dev;
        if(lw.global) {
            global_prep(qkv, N, pos, lw.q_norm, inv_g_, d_q_, qrows, glob_[L], 0, s);
            global_decode(ar, glob_[L], cfg_.global_blocks, d_partial_, s);
        } else {
            slide_prep(qkv, N, pos, lw.q_norm, lw.k_norm, inv_s_, d_q_, qrows, slide_[L], 0, s);
            slide_decode(ar, slide_[L], cfg_.slide_blocks, d_partial_, s);
        }
        to_bf16(d_attn_, N * qrows, d_attn_h_, s);
        gemv_fp8(lw.o, d_attn_h_, qrows, N, {GemvEpi::store, d_v_, kHidden}, cfg_.splits, nullptr, s);
        sandwich(d_v_, kHidden, d_hidden_, kHidden, lw.post_attn, nullptr, lw.pre_ff, d_xn_, kHidden, N, kHidden, s);
        gemv_fp4(lw.gu, d_xn_, kHidden, N, {GemvEpi::geglu, nullptr, 0, d_mlp_, kInter}, 1, nullptr, s);
        gemv_fp4(lw.down, d_mlp_, kInter, N, {GemvEpi::store, d_v_, kHidden}, cfg_.splits, nullptr, s);
        const bool last = L + 1 == kLayers;
        sandwich(d_v_, kHidden, d_hidden_, kHidden, lw.post_ff, lw.scalar, last ? final_norm_ : layers_[L + 1].in_norm,
                 last ? d_final_ : d_xn_, kHidden, N, kHidden, s);
    }
    head_gemv(d_final_, N, d_logits_, s);
    argmax_wide(d_logits_, kVocab, kVocabValid, N, d_pred_, d_amx_, s);
}

// ------------------------------------------------------------------------------------------------ drafter
void Net::draft_step(const int* tok, int* out, cudaStream_t s) {
    const Pos pos{ctl_ + ctl::pos, 0};
    embed(tok, 1, embed_, kHidden, embed_scale_, d_demb_, kHidden, s);
    to_bf16(d_demb_, kHidden, d_din_, s);
    gemv_fp4(dw_.pre, d_din_, 2 * kHidden, 1, {GemvEpi::store, d_dx_, kDHidden}, cfg_.splits, nullptr, s);
    rmsnorm_bf16(d_dx_, kDHidden, dw_.layers[0].in_norm, 1, kDHidden, d_dxn_, kDHidden, s);
    for(int l = 0; l < kDLayers; ++l) {
        const auto& d = dw_.layers[l];
        const int qrows = d.global ? kGQ : kSQ;
        gemv_fp4(d.q, d_dxn_, kDHidden, 1, {GemvEpi::store, d_proj_, qrows}, 1, nullptr, s);
        const Proj qp{d_proj_, nullptr, nullptr, nullptr, qrows};
        AttnRows ar;
        ar.q = d_q_; ar.q_stride = qrows; ar.out = d_attn_; ar.out_stride = qrows; ar.T = 1; ar.pos0_dev = pos.dev; ar.shift = 1;
        if(d.global) {
            global_prep(qp, 1, pos, d.q_norm, inv_g_, d_q_, qrows, glob_[kDSrcGlobal], 1, s);
            global_decode(ar, glob_[kDSrcGlobal], cfg_.global_blocks, d_partial_, s);
        } else {
            slide_prep(qp, 1, pos, d.q_norm, nullptr, inv_s_, d_q_, qrows, slide_[kDSrcSliding], 1, s);
            slide_decode(ar, slide_[kDSrcSliding], cfg_.slide_blocks, d_partial_, s);
        }
        to_bf16(d_attn_, qrows, d_attn_h_, s);
        gemv_fp4(d.o, d_attn_h_, qrows, 1, {GemvEpi::store, d_v_, kDHidden}, cfg_.splits, nullptr, s);
        sandwich(d_v_, kDHidden, d_dx_, kDHidden, d.post_attn, nullptr, d.pre_ff, d_dxn_, kDHidden, 1, kDHidden, s);
        gemv_fp4(d.gu, d_dxn_, kDHidden, 1, {GemvEpi::geglu, nullptr, 0, d_mlp_, kDInter}, 1, nullptr, s);
        gemv_fp4(d.down, d_mlp_, kDInter, 1, {GemvEpi::store, d_v_, kDHidden}, cfg_.splits, nullptr, s);
        const bool last = l + 1 == kDLayers;
        sandwich(d_v_, kDHidden, d_dx_, kDHidden, d.post_ff, d.scalar, last ? dw_.norm : dw_.layers[l + 1].in_norm, d_dxn_,
                 kDHidden, 1, kDHidden, s);
    }
    gemv_fp4(dw_.head, d_dxn_, kDHidden, 1, {GemvEpi::store, d_logits_, kVocab}, 1, nullptr, s);
    argmax_wide(d_logits_, kVocab, kVocabValid, 1, out, d_amx_, s);
    gemv_fp4(dw_.post, d_dxn_, kDHidden, 1, {GemvEpi::store, d_dh_, kHidden}, 1, nullptr, s);
    to_bf16(d_dh_, kHidden, d_din_ + kHidden, s);
}

void Net::head_gemv(const __nv_bfloat16* x, int N, float* logits, cudaStream_t s) {
    if(cfg_.head_fp4) gemv_fp4(head4_, x, kHidden, N, {GemvEpi::softcap, logits, kVocab}, 1, nullptr, s);
    else gemv_fp8(head_, x, kHidden, N, {GemvEpi::softcap, logits, kVocab}, 1, nullptr, s);
}

void Net::head_rows(const __nv_bfloat16* xn, int rows, float* logits, cudaStream_t s) {
    for(int r = 0; r < rows; r += kMaxCols) {
        const int n = std::min(kMaxCols, rows - r);
        head_gemv(xn + static_cast<std::size_t>(r) * kHidden, n, logits + static_cast<std::size_t>(r) * kVocab, s);
    }
}

// ------------------------------------------------------------------------------------------------ round
cudaGraphExec_t Net::build_round_graph(int K, bool replay, bool sampling) {
    cudaStream_t s = stream_;
    cudaGraph_t graph;
    KCHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    if(K > 0) {
        launch_k(gather_hidden_kernel, 1, 256, 0, s, ctl_, d_final_, d_din_ + kHidden);
        for(int k = 1; k <= K; ++k) draft_step(k == 1 ? ctl_ + ctl::last : d_draft_ + (k - 2), d_draft_ + (k - 1), s);
    }
    launch_k(verify_tokens_kernel, 1, 32, 0, s, ctl_, d_draft_, K, d_tok_);
    target_pass(K + 1, s);
    if(sampling) {
        launch_k(topk_probs_kernel, K + 1, 256, 0, s, d_logits_, kVocab, d_sampling_, d_cand_ids_, d_cand_p_, d_cand_n_);
        launch_k(accept_sampled_kernel, 1, 32, 0, s, ctl_, d_draft_, K, d_cand_ids_, d_cand_p_, d_cand_n_, d_sampling_, d_stops_,
                 d_result_);
    } else launch_k(accept_kernel, 1, 32, 0, s, ctl_, d_draft_, d_pred_, K, d_stops_, replay ? d_forced_ : nullptr, forced_cap_,
                    d_result_);
    KCHECK(cudaStreamEndCapture(s, &graph));
    cudaGraphExec_t exec = nullptr;
    KCHECK(cudaGraphInstantiate(&exec, graph, 0));
    KCHECK(cudaGraphDestroy(graph));
    return exec;
}

void Net::drop_graphs() {
    for(auto& [k, g] : graphs_) cudaGraphExecDestroy(g);
    graphs_.clear();
}

void Net::launch_round(int budget, bool replay, int draft) {
    if(draft < 0) draft = cfg_.draft;
    const int key = draft * 4 + (replay ? 1 : 0) + (sampling_ ? 2 : 0);
    auto it = graphs_.find(key);
    if(it == graphs_.end()) it = graphs_.emplace(key, build_round_graph(draft, replay, sampling_)).first;
    KCHECK(cudaMemcpyAsync(ctl_ + ctl::budget, &budget, 4, cudaMemcpyHostToDevice, stream_));
    KCHECK(cudaGraphLaunch(it->second, stream_));
    KCHECK(cudaMemcpyAsync(result_host_, d_result_, (kMaxCols + 2) * 4, cudaMemcpyDeviceToHost, stream_));
    round_done_.record(stream_);
}

void Net::set_forced(const int* tokens, int n) {
    if(n > forced_cap_) {
        d_forced_ = static_cast<int*>(dalloc(static_cast<std::size_t>(n + 64) * 4));
        forced_cap_ = n + 64;
        drop_graphs();
    }
    KCHECK(cudaMemcpyAsync(d_forced_, tokens, static_cast<std::size_t>(n) * 4, cudaMemcpyHostToDevice, stream_));
    std::vector<int> pad(forced_cap_ - n, -1);
    KCHECK(cudaMemcpyAsync(d_forced_ + n, pad.data(), pad.size() * 4, cudaMemcpyHostToDevice, stream_));
    KCHECK(cudaStreamSynchronize(stream_));
}

void Net::truncate_round(int accepted) {
    launch_k(truncate_kernel, 1, 1, 0, stream_, ctl_, d_result_, accepted);
    KLAUNCH();
}

void Net::set_stops(const std::vector<int>& stops) {
    int h[17] = {};
    const int n = std::min<int>(16, static_cast<int>(stops.size()));
    for(int i = 0; i < n; ++i) h[i] = stops[i];
    h[16] = n;
    KCHECK(cudaMemcpyAsync(d_stops_, h, sizeof(h), cudaMemcpyHostToDevice, stream_));
    KCHECK(cudaStreamSynchronize(stream_));
}

void Net::set_sampling(float temperature, float top_p, int top_k, std::uint64_t seed) {
    const bool want = temperature > 0.f;
    float h[8] = {temperature, top_p, static_cast<float>(top_k), 0, 0, 0, 0, 0};
    const std::uint32_t lo = static_cast<std::uint32_t>(seed), hi = static_cast<std::uint32_t>(seed >> 32);
    std::memcpy(&h[3], &lo, 4);
    std::memcpy(&h[4], &hi, 4);
    KCHECK(cudaMemcpyAsync(d_sampling_, h, sizeof(h), cudaMemcpyHostToDevice, stream_));
    KCHECK(cudaStreamSynchronize(stream_));
    sampling_ = want;
}

std::string Net::describe() const {
    std::ostringstream o;
    o << "g4 prefill_q8=" << cfg_.prefill_q8 << " prefill_qb=" << cfg_.prefill_qb << " head_fp4=" << cfg_.head_fp4 << " row_fp4=" << cfg_.row_fp4 << " context=" << cfg_.context << " chunk=" << cfg_.prefill_chunk << " ring=" << cfg_.ring << " draft=" << cfg_.draft
      << " bytes_per_pass=" << bytes_per_pass_ << " draft_bytes=" << draft_bytes_ << " device_bytes=" << device_bytes_;
    return o.str();
}

}  // namespace g4
