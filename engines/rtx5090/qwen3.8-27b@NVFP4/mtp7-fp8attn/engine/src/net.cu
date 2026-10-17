#include "net.h"

#include <crit/safetensors.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "attn_q8.cuh"
#include "gemv.cuh"

namespace kestrel {
float encode_nvfp4(const __nv_bfloat16* bf, const std::uint8_t* f8, const float* sc, int rows, int cols,
                   std::uint8_t* packed, std::uint8_t* sf);
namespace knet {

using crit::core::Checkpoint;
using crit::core::CudaBuffer;
using crit::core::Memory;

float bf16_to_float(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ------------------------------------------------------------------------------------------------ small kernels
__global__ void requant_fp8_rows_kernel(const __nv_bfloat16* __restrict__ in, int cols, std::uint8_t* __restrict__ codes,
                                        float* __restrict__ scale) {
    pdl_wait();
    pdl_trigger();
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

// Round start: positions for the MTP catch-up and draft steps, verify input tokens are filled later.
// pos layout: [0] catch-up base, [1 + k] draft step k (k = 1..K-1), [15] verify base.
__global__ void round_begin_kernel(int* ctl, int* pos) {
    pdl_wait();
    pdl_trigger();
    const int p = ctl[ctl::pos], m = ctl[ctl::mtp_cols];
    pos[0] = p - m;
    for(int k = 1; k < 14; ++k) pos[1 + k] = p + k - 1;
    pos[15] = p;
}
// Copy column ctl[mtp_cols] - 1 of src (FP16 rows of 5120) into dst.
__global__ void gather_col_kernel(const int* ctl, const __half* __restrict__ src, __half* __restrict__ dst) {
    pdl_wait();
    pdl_trigger();
    const int col = max(ctl[ctl::mtp_cols] - 1, 0);
    for(int i = threadIdx.x; i < kHidden / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src + static_cast<std::size_t>(col) * kHidden)[i];
}
__global__ void verify_tokens_kernel(int* ctl, const int* drafts, int K, int* tok) {
    pdl_wait();
    pdl_trigger();
    const int i = threadIdx.x;
    if(i == 0) {
        tok[0] = ctl[ctl::last];
        ctl[ctl::pass] += 1;
    }
    else if(i <= K) tok[i] = drafts[i - 1];
}

// Greedy (or replayed) acceptance; see net.h ctl fields.
__global__ void accept_kernel(int* ctl, const int* drafts, const int* pred, int K, const int* stops, const int* forced,
                              int forced_n, int* result, int* mtp_tok) {
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
    for(int i = 0; i < out; ++i) mtp_tok[i] = e[i];
    ctl[ctl::pos] += out;
    ctl[ctl::pending] = out;
    ctl[ctl::last] = out ? e[out - 1] : ctl[ctl::last];
    ctl[ctl::emitted] = out;
    ctl[ctl::mtp_cols] = out;
    if(forced) ctl[ctl::forced_at] = fa + out;
    ctl[ctl::stopped] = stopped ? 1 : 0;
}

__global__ void truncate_kernel(int* ctl, int* result, int accepted) {
    pdl_wait();
    pdl_trigger();
    const int out = result[0];
    if(accepted >= out) return;
    ctl[ctl::pos] -= out - accepted;
    ctl[ctl::pending] = accepted;
    ctl[ctl::mtp_cols] = accepted;
    ctl[ctl::last] = accepted ? result[accepted] : ctl[ctl::last];
    result[0] = accepted;
}

// Prefill finish: first token and the round state (catch-up of the last prompt position).
__global__ void prefill_done_kernel(int* ctl, const int* pred, int P, int* mtp_tok) {
    pdl_wait();
    pdl_trigger();
    ctl[ctl::pos] = P;
    ctl[ctl::pending] = 0;
    ctl[ctl::last] = pred[0];
    ctl[ctl::mtp_cols] = 1;
    ctl[ctl::forced_at] = 0;
    ctl[ctl::stopped] = 0;
    ctl[ctl::pass] = 0;
    mtp_tok[0] = pred[0];
}

// Temperature / top-k / top-p sampling of column 0 (Philox counter from seed and position).
// Top-k (<= 64) candidates of one logits row per block with temperature, top-p and renormalization. Each thread keeps
// its 8 best in registers; the 20th best of the block's candidates is checked against every thread's 8th, and a full
// iterative scan runs in the (rare) case a thread may hold more than 8 of the top k.
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
        if(v > lv[kLocal - 1]) {  // insertion into the sorted local list (ties keep the lower id first)
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
        // r-th best of vals (value desc, id asc) into tv[r]/ti[r]; taken entries are marked -inf in a private copy.
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
    // A thread whose 8th candidate reaches the k-th best may hold more of the top k.
    if(lv[kLocal - 1] > tv[k - 1] || (lv[kLocal - 1] == tv[k - 1] && li[kLocal - 1] < ti[k - 1])) need_full = 1;
    __syncthreads();
    if(need_full) {
        // Exact fallback: repeated block argmax over the whole row, excluding ids already taken.
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

// Draw from a column's candidates, optionally excluding one id (renormalized).
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

// Sampled acceptance with point-mass drafts: draft i is accepted with probability p_i(d_i) (column i's filtered
// distribution); on rejection the column is resampled without d_i; after all K a bonus is drawn from column K.
__global__ void accept_sampled_kernel(int* ctl, const int* drafts, int K, const int* ids, const float* probs, const int* cnt,
                                      const float* params, const int* stops, int* result, int* mtp_tok) {
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
        const int* ci = ids + i * kTopK;
        const float* cp = probs + i * kTopK;
        float pd = 0.f;
        for(int r = 0; r < cnt[i]; ++r)
            if(ci[r] == drafts[i]) pd = cp[r];
        if(uniform01(seed, pos + out, 0) < pd) {
            e[out++] = drafts[i];
            if(is_stop(drafts[i])) { stopped = true; break; }
        } else {
            e[out++] = draw(ci, cp, cnt[i], drafts[i], uniform01(seed, pos + out, 1));
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
    for(int j = 0; j < out; ++j) mtp_tok[j] = e[j];
    ctl[ctl::pos] += out;
    ctl[ctl::pending] = out;
    ctl[ctl::last] = out ? e[out - 1] : ctl[ctl::last];
    ctl[ctl::emitted] = out;
    ctl[ctl::mtp_cols] = out;
    ctl[ctl::stopped] = stopped ? 1 : 0;
}


}  // namespace knet
using namespace knet;

void launch_sample_first(const float* logits, const float* params, int pos, int* ids, float* probs, int* cnt, int* out,
                         cudaStream_t s) {
    launch_k(topk_probs_kernel, 1, 256, 0, s, logits, kVocab, params, ids, probs, cnt);
    launch_k(sample_first_kernel, 1, 1, 0, s, ids, probs, cnt, params, pos, out);
}
void launch_prefill_done(int* ctl, const int* pred, int P, int* mtp_tok, cudaStream_t s) {
    launch_k(prefill_done_kernel, 1, 1, 0, s, ctl, pred, P, mtp_tok);
    KLAUNCH();
}

bool pdl_enabled() {
    static const bool on = !std::getenv("KESTREL_NO_PDL");
    return on;
}

// ------------------------------------------------------------------------------------------------ construction
Net::Net(std::shared_ptr<crit::core::CudaBackend> backend, const std::string& model_dir, const Config& config)
    : backend_(std::move(backend)), cfg_(config) {
    if(cfg_.draft < 0 || cfg_.draft > kMaxCols - 1) throw std::invalid_argument("draft must be 0..7");
    if(const char* e = std::getenv("KESTREL_ATTN_Q8")) cfg_.attn_q8 = e[0] == '1';
    if(const char* e = std::getenv("KESTREL_PV8")) cfg_.attn_pv8 = e[0] == '1';
    if(const char* e = std::getenv("KESTREL_SCAN3")) set_scan3(e[0] == '1');
    if(const char* e = std::getenv("KESTREL_STATIC_FP4")) cfg_.static_fp4 = e[0] == '1';
    if(const char* e = std::getenv("KESTREL_WY")) cfg_.wy_scan = e[0] == '1';
    if(const char* e = std::getenv("KESTREL_DEC_Q8")) cfg_.decode_q8 = e[0] == '1';
    stream_ = backend_->stream(crit::core::Queue::compute);
    round_done_ = backend_->event();
    load(model_dir);
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
    return owned_.back().data();
}

void Net::load(const std::string& dir) {
    crit::core::CheckpointOptions opt;
    const Checkpoint ck = Checkpoint::open(dir, opt);
    auto bytes = [&](const std::string& name) {
        const auto& t = ck.at(name);
        return std::make_pair(ck.map(t), &t);
    };
    // Host staging for layout transforms.
    std::vector<std::uint8_t> host;
    auto upload = [&](void* dst, const void* src, std::size_t n) {
        KCHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
    };
    auto f32 = [&](const std::string& name, float add = 0.f, bool neg_exp = false) -> const float* {
        auto [v, t] = bytes(name);
        const std::size_t n = t->elements();
        std::vector<float> h(n);
        const auto* b = reinterpret_cast<const std::uint16_t*>(v.bytes.data());
        for(std::size_t i = 0; i < n; ++i) {
            float x = t->dtype == crit::core::DType::f32 ? reinterpret_cast<const float*>(v.bytes.data())[i] : bf16_to_float(b[i]);
            h[i] = neg_exp ? -std::exp(x) : x + add;
        }
        float* d = static_cast<float*>(dalloc(n * 4));
        upload(d, h.data(), n * 4);
        return d;
    };
    auto scalar = [&](const std::string& name) {
        auto [v, t] = bytes(name);
        return t->dtype == crit::core::DType::f32 ? *reinterpret_cast<const float*>(v.bytes.data())
                                                  : bf16_to_float(*reinterpret_cast<const std::uint16_t*>(v.bytes.data()));
    };
    // FP8 matrices concatenated by rows (stored codes, BF16 per-row scales -> FP32).
    auto fp8_cat = [&](const std::vector<std::string>& names) {
        int rows = 0, cols = 0;
        for(auto& n : names) {
            const auto& t = ck.at(n + ".weight");
            rows += static_cast<int>(t.shape[0]);
            cols = static_cast<int>(t.shape[1]);
        }
        auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(rows) * cols));
        auto* s = static_cast<float*>(dalloc(static_cast<std::size_t>(rows) * 4));
        std::size_t r0 = 0;
        for(auto& n : names) {
            auto [wv, wt] = bytes(n + ".weight");
            upload(w + r0 * cols, wv.bytes.data(), wv.bytes.size());
            auto [sv, st] = bytes(n + ".weight_scale");
            const std::size_t r = st->shape[0];
            std::vector<float> h(r);
            for(std::size_t i = 0; i < r; ++i) h[i] = bf16_to_float(reinterpret_cast<const std::uint16_t*>(sv.bytes.data())[i]);
            upload(s + r0, h.data(), r * 4);
            r0 += r;
        }
        bytes_per_pass_ += static_cast<std::size_t>(rows) * cols + rows * 4;
        return Fp8Mat{w, s, rows, cols};
    };
    // Interleave two equal [R][row_bytes] matrices in 8-row blocks (8 of a, then 8 of b).
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
            for(int c = 0; c < blocks; ++c) h[sf_offset(r, c, bp)] = sf[static_cast<std::size_t>(r) * blocks + c];
        upload(dev, h.data(), h.size());
    };
    auto fp4_mat = [&](const std::string& a, const std::string& b) {
        // b empty: one matrix; otherwise gate/up interleaved.
        auto [wa, ta] = bytes(a + ".weight_packed");
        auto [sa, tsa] = bytes(a + ".weight_scale");
        const float gs = scalar(a + ".weight_global_scale");
        int rows = static_cast<int>(ta->shape[0]);
        const int cols = static_cast<int>(ta->shape[1]) * 2;
        std::vector<std::uint8_t> packed, sf;
        const std::uint8_t* pw = reinterpret_cast<const std::uint8_t*>(wa.bytes.data());
        const std::uint8_t* ps = reinterpret_cast<const std::uint8_t*>(sa.bytes.data());
        if(!b.empty()) {
            auto [wb, tb] = bytes(b + ".weight_packed");
            auto [sb, tsb] = bytes(b + ".weight_scale");
            if(scalar(b + ".weight_global_scale") != gs) throw std::runtime_error("gate/up global scales differ: " + a);
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
        Fp4Mat m{w, s, 1.f / gs, rows, cols};
        m.igs = scalar(a + ".input_global_scale");
        const float as = m.alpha / m.igs;
        auto* ad = static_cast<float*>(dalloc(4));
        upload(ad, &as, 4);
        m.alpha_static = ad;
        return m;
    };
    auto fp8_gu = [&](const std::string& a, const std::string& b) {
        auto [wa, ta] = bytes(a + ".weight");
        auto [wb, tb] = bytes(b + ".weight");
        const int R = static_cast<int>(ta->shape[0]), cols = static_cast<int>(ta->shape[1]);
        std::vector<std::uint8_t> codes(static_cast<std::size_t>(R) * 2 * cols);
        interleave(reinterpret_cast<const std::uint8_t*>(wa.bytes.data()), reinterpret_cast<const std::uint8_t*>(wb.bytes.data()), R,
                   cols, codes.data());
        auto [sa, tsa] = bytes(a + ".weight_scale");
        auto [sb, tsb] = bytes(b + ".weight_scale");
        std::vector<float> s(static_cast<std::size_t>(R) * 2);
        for(int blk = 0; blk < R / 8; ++blk)
            for(int j = 0; j < 8; ++j) {
                s[blk * 16 + j] = bf16_to_float(reinterpret_cast<const std::uint16_t*>(sa.bytes.data())[blk * 8 + j]);
                s[blk * 16 + 8 + j] = bf16_to_float(reinterpret_cast<const std::uint16_t*>(sb.bytes.data())[blk * 8 + j]);
            }
        auto* w = static_cast<std::uint8_t*>(dalloc(codes.size()));
        upload(w, codes.data(), codes.size());
        auto* sd = static_cast<float*>(dalloc(s.size() * 4));
        upload(sd, s.data(), s.size() * 4);
        bytes_per_pass_ += codes.size() + s.size() * 4;
        return Fp8Mat{w, sd, 2 * R, cols};
    };
    // Drafter BF16 matrices on the device (names concatenated, or gate/up interleaved), then re-encoded.
    __nv_bfloat16* tmp = nullptr;
    std::size_t tmp_bytes = 0;
    auto stage_bf16 = [&](const std::vector<std::string>& names, bool gu, int& rows, int& cols) {
        rows = 0;
        cols = 0;
        std::vector<std::pair<Checkpoint::View, const crit::core::TensorInfo*>> parts;
        for(auto& n : names) {
            parts.push_back(bytes(n + ".weight"));
            rows += static_cast<int>(parts.back().second->shape[0]);
            cols = static_cast<int>(parts.back().second->shape[1]);
        }
        const std::size_t nbytes = static_cast<std::size_t>(rows) * cols * 2;
        if(nbytes > tmp_bytes) {
            if(tmp) KCHECK(cudaFree(tmp));
            KCHECK(cudaMalloc(&tmp, nbytes));
            tmp_bytes = nbytes;
        }
        if(gu) {
            host.resize(nbytes);
            interleave(reinterpret_cast<const std::uint8_t*>(parts[0].first.bytes.data()),
                       reinterpret_cast<const std::uint8_t*>(parts[1].first.bytes.data()), rows / 2,
                       static_cast<std::size_t>(cols) * 2, host.data());
            upload(tmp, host.data(), nbytes);
        } else {
            std::size_t off = 0;
            for(auto& p : parts) {
                upload(reinterpret_cast<std::uint8_t*>(tmp) + off, p.first.bytes.data(), p.first.bytes.size());
                off += p.first.bytes.size();
            }
        }
    };
    auto to_fp8 = [&](const __nv_bfloat16* src, int rows, int cols) {
        auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(rows) * cols));
        auto* s = static_cast<float*>(dalloc(static_cast<std::size_t>(rows) * 4));
        launch_k(requant_fp8_rows_kernel, rows, 256, 0, 0, src, cols, w, s);
        KLAUNCH();
        KCHECK(cudaDeviceSynchronize());
        return Fp8Mat{w, s, rows, cols};
    };
    auto to_fp4 = [&](const __nv_bfloat16* bf, const std::uint8_t* f8, const float* sc, int rows, int cols) {
        auto* w = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(rows) * cols / 2));
        auto* sf = static_cast<std::uint8_t*>(dalloc(sf_bytes(rows, cols)));
        const float alpha = encode_nvfp4(bf, f8, sc, rows, cols, w, sf);
        draft_bytes_ += static_cast<std::size_t>(rows) * cols / 2 + static_cast<std::size_t>(rows) * cols / 16;
        return Fp4Mat{w, sf, alpha, rows, cols};
    };
    auto drafter4 = [&](const std::vector<std::string>& names, bool gu) {
        int rows, cols;
        stage_bf16(names, gu, rows, cols);
        return to_fp4(tmp, nullptr, nullptr, rows, cols);
    };

    const std::string P = "model.language_model.";
    layers_.resize(kLayers);
    for(int L = 0; L < kLayers; ++L) {
        auto& lw = layers_[L];
        const std::string p = P + "layers." + std::to_string(L) + ".";
        lw.attn = is_attention_layer(L);
        lw.in_norm = f32(p + "input_layernorm.weight");
        lw.post_norm = f32(p + "post_attention_layernorm.weight");
        if(lw.attn) {
            lw.qkv = fp8_cat({p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"});
            lw.o = fp8_cat({p + "self_attn.o_proj"});
            lw.q_norm = f32(p + "self_attn.q_norm.weight");
            lw.k_norm = f32(p + "self_attn.k_norm.weight");
        } else {
            lw.qkvz = fp8_cat({p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z"});
            lw.out = fp8_cat({p + "linear_attn.out_proj"});
            lw.gdn.conv_w = f32(p + "linear_attn.conv1d.weight");
            {
                // in_proj_a then in_proj_b, contiguous: one [96][5120] BF16 matrix for the prompt GEMM.
                auto [va, ta] = bytes(p + "linear_attn.in_proj_a.weight");
                auto [vb, tb] = bytes(p + "linear_attn.in_proj_b.weight");
                auto* d = static_cast<__nv_bfloat16*>(dalloc(va.bytes.size() + vb.bytes.size()));
                upload(d, va.bytes.data(), va.bytes.size());
                upload(reinterpret_cast<std::uint8_t*>(d) + va.bytes.size(), vb.bytes.data(), vb.bytes.size());
                lw.gdn.in_a = d;
                lw.gdn.in_b = d + static_cast<std::size_t>(kLinVHeads) * kHidden;
                bytes_per_pass_ += va.bytes.size() + vb.bytes.size();
            }
            lw.gdn.neg_a = f32(p + "linear_attn.A_log", 0.f, true);
            lw.gdn.dt_bias = f32(p + "linear_attn.dt_bias");
            lw.gdn.norm_w = f32(p + "linear_attn.norm.weight");
        }
        lw.fp4 = ck.find(p + "mlp.gate_proj.weight_packed") != nullptr;
        if(lw.fp4) {
            lw.gu4 = fp4_mat(p + "mlp.gate_proj", p + "mlp.up_proj");
            lw.down4 = fp4_mat(p + "mlp.down_proj", "");
        } else {
            lw.gu8 = fp8_gu(p + "mlp.gate_proj", p + "mlp.up_proj");
            lw.down8 = fp8_cat({p + "mlp.down_proj"});
        }
    }
    head_ = fp8_cat({"lm_head"});
    final_norm_ = f32(P + "norm.weight");
    {
        auto [v, t] = bytes(P + "embed_tokens.weight");
        owned_.push_back(backend_->allocate(Memory::host, v.bytes.size()));
        std::memcpy(owned_.back().data(), v.bytes.data(), v.bytes.size());
        embed_ = static_cast<const __nv_bfloat16*>(owned_.back().device_address());
    }
    // MTP drafter: NVFP4 for drafting, FP8 copies of fc and the key/value rows for the prompt pass.
    const std::string m = "mtp.";
    {
        int rows, cols;
        stage_bf16({m + "fc"}, false, rows, cols);
        mtp_.fc8 = to_fp8(tmp, rows, cols);
        mtp_.fc = to_fp4(tmp, nullptr, nullptr, rows, cols);
        stage_bf16({m + "layers.0.self_attn.q_proj", m + "layers.0.self_attn.k_proj", m + "layers.0.self_attn.v_proj"},
                   false, rows, cols);
        mtp_.kv8 = to_fp8(tmp + static_cast<std::size_t>(kQRows) * cols, 2 * kKvRows, cols);
        mtp_.qkv = to_fp4(tmp, nullptr, nullptr, rows, cols);
    }
    mtp_.o = drafter4({m + "layers.0.self_attn.o_proj"}, false);
    mtp_.gu = drafter4({m + "layers.0.mlp.gate_proj", m + "layers.0.mlp.up_proj"}, true);
    mtp_.down = drafter4({m + "layers.0.mlp.down_proj"}, false);
    mtp_.head = to_fp4(nullptr, head_.w, head_.scale, head_.rows, head_.cols);
    mtp_.in_norm = f32(m + "layers.0.input_layernorm.weight");
    mtp_.post_norm = f32(m + "layers.0.post_attention_layernorm.weight");
    mtp_.q_norm = f32(m + "layers.0.self_attn.q_norm.weight");
    mtp_.k_norm = f32(m + "layers.0.self_attn.k_norm.weight");
    mtp_.norm = f32(m + "norm.weight");
    mtp_.pre_emb = f32(m + "pre_fc_norm_embedding.weight");
    mtp_.pre_hidden = f32(m + "pre_fc_norm_hidden.weight");
    if(tmp) KCHECK(cudaFree(tmp));
    // RoPE inverse frequencies, as the reference computes them in FP32.
    std::vector<float> inv(32);
    for(int i = 0; i < 32; ++i) inv[i] = static_cast<float>(1.0 / std::pow(kRopeTheta, static_cast<double>(2 * i) / kRotary));
    for(int i = 0; i < 32; ++i) {
        // torch: 1.0 / (base ** (arange(0, dim, 2, float) / dim)) in FP32.
        const float e = static_cast<float>(2 * i) / static_cast<float>(kRotary);
        inv[i] = 1.f / std::pow(static_cast<float>(kRopeTheta), e);
    }
    rope_inv_ = static_cast<float*>(dalloc(32 * 4));
    upload(rope_inv_, inv.data(), 32 * 4);
}

void Net::alloc_buffers() {
    const std::size_t ctx = static_cast<std::size_t>(cfg_.context) + 64;
    auto cache = [&](crit::fp8_kv::Cache& c) {
        c.kv_heads = kKvHeads;
        c.head_dim = kHeadDim;
        c.k = static_cast<std::uint8_t*>(dalloc(ctx * kKvRows));
        c.v = static_cast<std::uint8_t*>(dalloc(ctx * kKvRows));
        c.k_scale = static_cast<float*>(dalloc(ctx * kKvHeads * 4));
        c.v_scale = static_cast<float*>(dalloc(ctx * kKvHeads * 4));
    };
    kv_.resize(kLayers / 4);
    for(auto& c : kv_) cache(c);
    cache(mtp_kv_);
    gdn_.resize(kLayers - kLayers / 4);
    for(auto& g : gdn_) {
        g.conv = static_cast<float*>(dalloc(2 * 3 * kLinQkv * 4));
        g.conv_stash = static_cast<float*>(dalloc(2 * kMaxCols * kLinQkv * 4));
        g.S = static_cast<float*>(dalloc(static_cast<std::size_t>(kLinVHeads) * kLinDim * kLinDim * 4));
        g.stash_k = static_cast<float*>(dalloc(kMaxCols * kLinVHeads * kLinDim * 4));
        g.stash_decay = static_cast<float*>(dalloc(kMaxCols * kLinVHeads * 4));
        g.stash_delta = static_cast<float*>(dalloc(kMaxCols * kLinVHeads * kLinDim * 4));
    }
    ctl_ = static_cast<int*>(dalloc(ctl::fields * 4 + 16 * 4));
    const int C = kMaxCols;
    d_hidden_ = static_cast<float*>(dalloc(C * kHidden * 4));
    d_xn_ = static_cast<__half*>(dalloc(C * kHidden * 2));
    d_proj_ = static_cast<float*>(dalloc(C * 2 * kInter * 4));
    d_q_ = static_cast<__nv_bfloat16*>(dalloc(C * kAttnOut * 2));
    d_attn_ = static_cast<float*>(dalloc(C * kAttnOut * 4));
    d_gated_ = static_cast<__half*>(dalloc(C * kAttnOut * 2));
    d_mlp_ = static_cast<__half*>(dalloc(C * kInter * 2));
    d_partial_ = static_cast<float*>(dalloc(static_cast<std::size_t>(cfg_.splits) * C * kHidden * 4));
    d_ssq_ = static_cast<float*>(dalloc(kMaxCols * (kHidden / 16) * 4));
    d_q8_ = static_cast<std::uint8_t*>(dalloc(kMaxCols * kAttnOut));
    d_q8s_ = static_cast<float*>(dalloc(kMaxCols * kHeads * 4));
    d_attn_partial_ = static_cast<float*>(dalloc(crit::fp8_kv::decode_partial_floats(kKvHeads, kHeadDim, cfg_.attn_blocks) * 4));
    d_logits_ = static_cast<float*>(dalloc(static_cast<std::size_t>(C) * kVocab * 4));
    d_final_ = static_cast<__half*>(dalloc(C * kHidden * 2));
    d_mtp_in_ = static_cast<__half*>(dalloc(C * 2 * kHidden * 2));
    d_mtp_h_ = static_cast<__half*>(dalloc((C + 1) * kHidden * 2));
    d_tok_ = static_cast<int*>(dalloc(64 * 4));
    d_pred_ = static_cast<int*>(dalloc(64 * 4));
    d_draft_ = static_cast<int*>(dalloc(64 * 4));
    d_mtp_tok_ = static_cast<int*>(dalloc(64 * 4));
    d_result_ = static_cast<int*>(dalloc(64 * 4));
    d_stops_ = static_cast<int*>(dalloc(32 * 4));
    d_sampling_ = static_cast<float*>(dalloc(8 * 4));
    d_cand_ids_ = static_cast<int*>(dalloc(kMaxCols * 64 * 4));
    d_cand_p_ = static_cast<float*>(dalloc(kMaxCols * 64 * 4));
    d_cand_n_ = static_cast<int*>(dalloc(kMaxCols * 4));
    gsc_.qk = static_cast<float*>(dalloc(C * kLinKHeads * 2 * kLinDim * 4));
    gsc_.v = static_cast<float*>(dalloc(C * kLinVHeads * kLinDim * 4));
    gsc_.gb = static_cast<float*>(dalloc(C * kLinVHeads * 2 * 4));
    result_pinned_ = backend_->allocate(Memory::host, 64 * 4);
    result_host_ = static_cast<int*>(result_pinned_.data());
    KCHECK(cudaMemset(d_stops_, 0, 32 * 4));

    // Prompt buffers.
    const int T = (cfg_.prefill_chunk + 127) / 128 * 128;
    chunk_pad_ = T;
    p_tok_ = static_cast<int*>(dalloc((T + 1) * 4));
    p_hidden_ = static_cast<float*>(dalloc(static_cast<std::size_t>(T) * kHidden * 4));
    p_xq_ = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(T) * 2 * kHidden));
    p_xs_ = static_cast<float*>(dalloc(T * 4));
    p_xbf_ = static_cast<__nv_bfloat16*>(dalloc(static_cast<std::size_t>(T) * 2 * kHidden * 2));
    p_raw_ = static_cast<__nv_bfloat16*>(dalloc(static_cast<std::size_t>(T) * 2 * kInter * 2));
    p_raw2_ = static_cast<__nv_bfloat16*>(dalloc(static_cast<std::size_t>(T) * kHidden * 2));
    {
        const std::size_t tt = static_cast<std::size_t>(T);
        const std::size_t gdn = tt * kLinKHeads * 2 * kLinDim * 4 + 2 * tt * kLinVHeads * kLinDim * 4 + tt * kLinVHeads * 2 * 4;
        const std::size_t att = tt * kAttnOut * 2 + tt * kAttnOut * 4;
        const std::size_t mlp = tt * kInter * 2 + tt * kInter;
        auto* region = static_cast<std::uint8_t*>(dalloc(std::max({gdn, att, mlp}) + 4096));
        psc_.qk = reinterpret_cast<float*>(region);
        psc_.v = psc_.qk + tt * kLinKHeads * 2 * kLinDim;
        psc_.o = psc_.v + tt * kLinVHeads * kLinDim;
        psc_.gb = psc_.o + tt * kLinVHeads * kLinDim;
        p_q_ = reinterpret_cast<__nv_bfloat16*>(region);
        p_attn_ = reinterpret_cast<float*>(region + tt * kAttnOut * 2);
        p_h_ = reinterpret_cast<__nv_bfloat16*>(region);
        p_hq_ = region + tt * kInter * 2;
    }
    p_hs_ = static_cast<float*>(dalloc(T * 4));
    p_ab_ = static_cast<float*>(dalloc(static_cast<std::size_t>(T) * 2 * kLinVHeads * 4));
    p_x4_ = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(T) * kHidden / 2));
    p_x4sf_ = static_cast<std::uint8_t*>(dalloc(sf_bytes(T, kHidden)));
    p_h4sf_ = static_cast<std::uint8_t*>(dalloc(sf_bytes(T, kInter)));
    p_amax_ = static_cast<unsigned*>(dalloc(16));
    p_alpha_ = static_cast<float*>(dalloc(16));
    p_aq_ = static_cast<std::uint8_t*>(dalloc(static_cast<std::size_t>(T) * kAttnOut));
    p_as_ = static_cast<float*>(dalloc(T * 4));
    if(cfg_.wy_scan) wy_ = wy_scratch(static_cast<float*>(dalloc(wy_scratch_floats(32) * 4)), 32);
    p_final_ = static_cast<__half*>(dalloc(static_cast<std::size_t>(T) * kHidden * 2));
    const std::size_t ws = 64ull << 20;
    lt_ = std::make_unique<Lt>(dalloc(ws), ws, cfg_.gemm_table);
}

void Net::reset() {
    for(auto& g : gdn_) {
        KCHECK(cudaMemsetAsync(g.conv, 0, 2 * 3 * kLinQkv * 4, stream_));
        KCHECK(cudaMemsetAsync(g.S, 0, static_cast<std::size_t>(kLinVHeads) * kLinDim * kLinDim * 4, stream_));
    }
    KCHECK(cudaMemsetAsync(ctl_, 0, ctl::fields * 4, stream_));
}

// ------------------------------------------------------------------------------------------------ target pass
void Net::target_pass(int N, cudaStream_t s) {
    const Pos pos{ctl_ + ctl::pos, 0};
    const int* pending = ctl_ + ctl::pending;
    const int* pass = ctl_ + ctl::pass;
    embed(d_tok_, N, embed_, d_hidden_, s);
    rmsnorm_f16(d_hidden_, kHidden, layers_[0].in_norm, N, kHidden, d_xn_, kHidden, s);
    int ai = 0, gi = 0;
    for(int L = 0; L < kLayers; ++L) {
        const auto& lw = layers_[L];
        // Mixer output projection: split reduce + residual + post-attention RMSNorm into d_xn_.
        const GemvOut mix_out{GemvEpi::residual_norm, d_hidden_, kHidden, d_xn_, kHidden, lw.post_norm, d_ssq_};
        if(lw.attn) {
            const auto& c = kv_[ai++];
            gemv_fp8(lw.qkv, d_xn_, kHidden, N, {GemvEpi::store, d_proj_, lw.qkv.rows}, 1, nullptr, s);
            Proj qkv{d_proj_, nullptr, nullptr, nullptr, lw.qkv.rows};
            attn_prep(qkv, N, pos, lw.q_norm, lw.k_norm, rope_inv_, d_q_, c, s, false, cfg_.decode_q8 ? d_q8_ : nullptr, d_q8s_);
            crit::fp8_kv::Decode dp;
            dp.q = d_q_; dp.q_stride = kAttnOut; dp.cache = c; dp.out = d_attn_; dp.out_stride = kAttnOut;
            dp.T = N; dp.heads = kHeads; dp.pos0_device = pos.dev; dp.scale = 1.f / 16.f; dp.blocks = cfg_.attn_blocks;
            dp.partial = d_attn_partial_;
            dp.q8 = cfg_.decode_q8 ? d_q8_ : nullptr;
            dp.q8_scale = d_q8s_;
            if(cfg_.decode_q8) decode_attention_q8(dp, s);
            else KCHECK(crit::fp8_kv::decode(dp, s));
            attn_gate(d_attn_, qkv, N, d_gated_, s);
            gemv_fp8(lw.o, d_gated_, kAttnOut, N, mix_out, cfg_.splits, d_partial_, s);
        } else {
            auto& st = gdn_[gi++];
            gemv_fp8(lw.qkvz, d_xn_, kHidden, N, {GemvEpi::store, d_proj_, lw.qkvz.rows}, 1, nullptr, s);
            gdn_fused(lw.gdn, st, d_proj_, d_xn_, N, pending, pass, d_gated_, s);
            gemv_fp8(lw.out, d_gated_, kLinZ, N, mix_out, cfg_.splits, d_partial_, s);
        }
        // MLP: down projection reduce + residual + the next layer's input RMSNorm (or the final norm).
        const bool last = L + 1 == kLayers;
        const GemvOut mlp_out{GemvEpi::residual_norm, d_hidden_, kHidden, last ? d_final_ : d_xn_, kHidden,
                              last ? final_norm_ : layers_[L + 1].in_norm, d_ssq_};
        if(lw.fp4) {
            gemv_fp4(lw.gu4, d_xn_, kHidden, N, {GemvEpi::swiglu, nullptr, 0, d_mlp_, kInter}, 1, nullptr, s);
            gemv_fp4(lw.down4, d_mlp_, kInter, N, mlp_out, cfg_.splits, d_partial_, s);
        } else {
            gemv_fp8(lw.gu8, d_xn_, kHidden, N, {GemvEpi::swiglu, nullptr, 0, d_mlp_, kInter}, 1, nullptr, s);
            gemv_fp8(lw.down8, d_mlp_, kInter, N, mlp_out, cfg_.splits, d_partial_, s);
        }
    }
    gemv_fp8(head_, d_final_, kHidden, N, {GemvEpi::store, d_logits_, kVocab}, 1, nullptr, s);
    argmax(d_logits_, kVocab, N, d_pred_, s);
}

// ------------------------------------------------------------------------------------------------ MTP
void Net::mtp_pass(int N, const int* tok_dev, const __half* hidden_dev, int hidden_ld, Pos pos, cudaStream_t s) {
    embed(tok_dev, N, embed_, d_hidden_, s);
    rmsnorm_f16(d_hidden_, kHidden, mtp_.pre_emb, N, kHidden, d_mtp_in_, 2 * kHidden, s);
    rmsnorm_f16_from_f16(hidden_dev, hidden_ld, mtp_.pre_hidden, N, kHidden, d_mtp_in_ + kHidden, 2 * kHidden, s);
    gemv_fp4(mtp_.fc, d_mtp_in_, 2 * kHidden, N, {GemvEpi::store, d_hidden_, kHidden}, cfg_.splits, d_partial_, s);
    rmsnorm_f16(d_hidden_, kHidden, mtp_.in_norm, N, kHidden, d_xn_, kHidden, s);
    gemv_fp4(mtp_.qkv, d_xn_, kHidden, N, {GemvEpi::store, d_proj_, mtp_.qkv.rows}, 1, nullptr, s);
    Proj qkv{d_proj_, nullptr, nullptr, nullptr, mtp_.qkv.rows};
    attn_prep(qkv, N, pos, mtp_.q_norm, mtp_.k_norm, rope_inv_, d_q_, mtp_kv_, s, false, cfg_.decode_q8 ? d_q8_ : nullptr, d_q8s_);
    crit::fp8_kv::Decode dp;
    dp.q = d_q_; dp.q_stride = kAttnOut; dp.cache = mtp_kv_; dp.out = d_attn_; dp.out_stride = kAttnOut;
    dp.T = N; dp.heads = kHeads; dp.pos0_device = pos.dev; dp.scale = 1.f / 16.f; dp.blocks = cfg_.attn_blocks;
    dp.partial = d_attn_partial_;
    dp.q8 = cfg_.decode_q8 ? d_q8_ : nullptr;
    dp.q8_scale = d_q8s_;
    if(cfg_.decode_q8) decode_attention_q8(dp, s);
    else KCHECK(crit::fp8_kv::decode(dp, s));
    attn_gate(d_attn_, qkv, N, d_gated_, s);
    gemv_fp4(mtp_.o, d_gated_, kAttnOut, N, {GemvEpi::residual_norm, d_hidden_, kHidden, d_xn_, kHidden, mtp_.post_norm, d_ssq_},
             cfg_.splits, d_partial_, s);
    gemv_fp4(mtp_.gu, d_xn_, kHidden, N, {GemvEpi::swiglu, nullptr, 0, d_mlp_, kInter}, 1, nullptr, s);
    gemv_fp4(mtp_.down, d_mlp_, kInter, N, {GemvEpi::residual_norm, d_hidden_, kHidden, d_mtp_h_, kHidden, mtp_.norm, d_ssq_},
             cfg_.splits, d_partial_, s);
}

void Net::head_rows(const __half* xn, int rows, float* logits, cudaStream_t s) {
    for(int r = 0; r < rows; r += kMaxCols) {
        const int n = std::min(kMaxCols, rows - r);
        gemv_fp8(head_, xn + static_cast<std::size_t>(r) * kHidden, kHidden, n,
                 {GemvEpi::store, logits + static_cast<std::size_t>(r) * kVocab, kVocab}, 1, nullptr, s);
    }
}

// ------------------------------------------------------------------------------------------------ round
cudaGraphExec_t Net::build_round_graph(int K, bool replay, bool sampling) {
    int* posbuf = ctl_ + ctl::fields;
    cudaStream_t s = stream_;
    cudaGraph_t graph;
    KCHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    if(K > 0) {
        launch_k(round_begin_kernel, 1, 1, 0, s, ctl_, posbuf);
        // Catch-up over the last round's materialized columns (fixed width K + 1; extra columns are masked by position).
        mtp_pass(kMaxCols, d_mtp_tok_, d_final_, kHidden, Pos{posbuf + 0, 0}, s);
        __half* sel = d_mtp_h_ + static_cast<std::size_t>(kMaxCols) * kHidden;
        launch_k(gather_col_kernel, 1, 256, 0, s, ctl_, d_mtp_h_, sel);
        gemv_fp4(mtp_.head, sel, kHidden, 1, {GemvEpi::store, d_logits_, kVocab}, 1, nullptr, s);
        argmax(d_logits_, kVocab, 1, d_draft_, s);
        for(int k = 2; k <= K; ++k) {
            mtp_pass(1, d_draft_ + (k - 2), sel, kHidden, Pos{posbuf + k, 0}, s);
            cudaMemcpyAsync(sel, d_mtp_h_, kHidden * 2, cudaMemcpyDeviceToDevice, s);
            gemv_fp4(mtp_.head, sel, kHidden, 1, {GemvEpi::store, d_logits_, kVocab}, 1, nullptr, s);
            argmax(d_logits_, kVocab, 1, d_draft_ + (k - 1), s);
        }
    }
    launch_k(verify_tokens_kernel, 1, 32, 0, s, ctl_, d_draft_, K, d_tok_);
    target_pass(K + 1, s);
    if(sampling) {
        launch_k(topk_probs_kernel, K + 1, 256, 0, s, d_logits_, kVocab, d_sampling_, d_cand_ids_, d_cand_p_, d_cand_n_);
        launch_k(accept_sampled_kernel, 1, 32, 0, s, ctl_, d_draft_, K, d_cand_ids_, d_cand_p_, d_cand_n_, d_sampling_, d_stops_,
                 d_result_, d_mtp_tok_);
    } else launch_k(accept_kernel, 1, 32, 0, s, ctl_, d_draft_, d_pred_, K, d_stops_, replay ? d_forced_ : nullptr, forced_cap_,
                                   d_result_, d_mtp_tok_);
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
    // Positions past n hold -1 so no draft matches them.
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
    o << "kestrel context=" << cfg_.context << " chunk=" << cfg_.prefill_chunk << " draft=" << cfg_.draft
      << " bytes_per_pass=" << bytes_per_pass_ << " draft_bytes=" << draft_bytes_;
    return o.str();
}

}  // namespace kestrel
