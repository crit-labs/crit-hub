#include "layers.cuh"

namespace nmh {
namespace {

// grid T, block 512: 8 values per thread.
__global__ void embed_kernel(const __nv_bfloat16* table, const int* ids, const Step* steps, float* out) {
    const int t = blockIdx.x;
    const int id = ids ? ids[t] : steps[t].token;
    const uint4 v = reinterpret_cast<const uint4*>(table + std::size_t(id) * kHidden)[threadIdx.x];
    float* o = out + std::size_t(t) * kHidden + threadIdx.x * 8;
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for(int i = 0; i < 4; ++i) {
        o[2 * i] = bf16_lo(w[i]);
        o[2 * i + 1] = bf16_hi(w[i]);
    }
}

// grid T, block 1024: 4 values per thread.
template <int M>
__global__ void __launch_bounds__(1024) rmsnorm_kernel(const float* x, const float* w, void* y, int ldy) {
    __shared__ float red[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float4 v = reinterpret_cast<const float4*>(x + std::size_t(t) * kHidden)[tid];
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    ss = warp_sum(ss);
    if((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    if(tid < 32) {
        float s = red[tid];
        s = warp_sum(s);
        if(tid == 0) red[0] = s;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / float(kHidden) + kEps);
    const float4 wv = reinterpret_cast<const float4*>(w)[tid];
    const float o0 = v.x * r * wv.x, o1 = v.y * r * wv.y, o2 = v.z * r * wv.z, o3 = v.w * r * wv.w;
    const std::size_t base = std::size_t(t) * ldy + tid * 4;
    if constexpr(M == 0) {
        *reinterpret_cast<float4*>(static_cast<float*>(y) + base) = make_float4(o0, o1, o2, o3);
    } else if constexpr(M == 1) {
        __half* h = static_cast<__half*>(y) + base;
        h[0] = __float2half_rn(o0); h[1] = __float2half_rn(o1); h[2] = __float2half_rn(o2); h[3] = __float2half_rn(o3);
    } else {
        __nv_bfloat16* h = static_cast<__nv_bfloat16*>(y) + base;
        h[0] = __float2bfloat16_rn(o0); h[1] = __float2bfloat16_rn(o1);
        h[2] = __float2bfloat16_rn(o2); h[3] = __float2bfloat16_rn(o3);
    }
}

__global__ void to_bf16_kernel(const float* x, __nv_bfloat16* y, std::size_t n) {
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}
__global__ void to_f16_kernel(const float* x, __half* y, std::size_t n) {
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        y[i] = __float2half_rn(x[i]);
}
__global__ void add_kernel(float* y, const float* x, std::size_t n) {
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        y[i] += x[i];
}

// grid (T, kKvHeads), block kHeadDim: the same E4M3 encoding as crit::fp8_kv::write (scale = max |x| / 448 over the
// head vector, BF16-rounded inputs). Threads also convert the token's queries to BF16.
__global__ void __launch_bounds__(kHeadDim) kv_write_kernel(KvCache c, const float* proj, int ld, const Step* steps,
                                                            __nv_bfloat16* q16) {
    __shared__ float red[2][kHeadDim / 32];
    const int t = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
    const float* row = proj + std::size_t(t) * ld;
    for(int i = g * kHeadDim + d; i < kQDim; i += kKvHeads * kHeadDim)
        q16[std::size_t(t) * kQDim + i] = __float2bfloat16_rn(row[i]);
    const float kx = __bfloat162float(__float2bfloat16_rn(row[kQDim + g * kHeadDim + d]));
    const float vx = __bfloat162float(__float2bfloat16_rn(row[kQDim + kKvDim + g * kHeadDim + d]));
    const float ak = warp_max(fabsf(kx)), av = warp_max(fabsf(vx));
    if((d & 31) == 0) { red[0][d >> 5] = ak; red[1][d >> 5] = av; }
    __syncthreads();
    float mk = 0.f, mv = 0.f;
#pragma unroll
    for(int i = 0; i < kHeadDim / 32; ++i) { mk = fmaxf(mk, red[0][i]); mv = fmaxf(mv, red[1][i]); }
    const float sk = mk > 0.f ? mk / 448.f : 1.f, sv = mv > 0.f ? mv / 448.f : 1.f;
    const std::size_t pos = std::size_t(steps[t].pos), off = (pos * kKvHeads + g) * kHeadDim + d;
    c.k[off] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(kx / sk, __NV_SATFINITE, __NV_E4M3));
    c.v[off] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(vx / sv, __NV_SATFINITE, __NV_E4M3));
    if(d == 0) {
        c.ks[pos * kKvHeads + g] = sk;
        c.vs[pos * kKvHeads + g] = sv;
    }
}

// grid n rows, block 1024.
__global__ void row_scores_kernel(const float* logits, int ld, int valid, const int* targets, const int* queries,
                                  int per, float* tlp, float* qlp, int* amax) {
    const int r = blockIdx.x;
    const float* x = logits + std::size_t(r) * ld;
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ double sd[32];
    __shared__ float mx;
    __shared__ int mi;
    __shared__ double lse;
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for(int i = threadIdx.x; i < valid; i += blockDim.x) {
        const float y = x[i];
        if(y > v || (y == v && i < idx)) { v = y; idx = i; }
    }
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if(ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if(threadIdx.x == 0) {
        float bv = sv[0];
        int bi = si[0];
        for(int w = 1; w < int(blockDim.x >> 5); ++w)
            if(sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        mx = bv;
        mi = bi;
    }
    __syncthreads();
    if(!tlp && !qlp) {
        if(threadIdx.x == 0 && amax) amax[r] = mi;
        return;
    }
    const float m = mx;
    double s = 0.0;
    for(int i = threadIdx.x; i < valid; i += blockDim.x) s += double(expf(x[i] - m));
#pragma unroll
    for(int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if((threadIdx.x & 31) == 0) sd[threadIdx.x >> 5] = s;
    __syncthreads();
    if(threadIdx.x == 0) {
        double t = 0.0;
        for(int w = 0; w < int(blockDim.x >> 5); ++w) t += sd[w];
        lse = double(m) + log(t);
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        if(amax) amax[r] = mi;
        if(tlp) {
            const int t = targets[r];
            tlp[r] = (t >= 0 && t < valid) ? float(double(x[t]) - lse) : -INFINITY;
        }
    }
    for(int j = threadIdx.x; j < per; j += blockDim.x) {
        const int q = queries[std::size_t(r) * per + j];
        qlp[std::size_t(r) * per + j] = (q >= 0 && q < valid) ? float(double(x[q]) - lse) : -INFINITY;
    }
}

// grid T, block 256.
__global__ void quant_rows_kernel(const float* x, int ldx, int K, uint8_t* q, float* rscale) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    const float* row = x + std::size_t(t) * ldx;
    float m = 0.f;
    for(int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, fabsf(row[k]));
    m = warp_max(m);
    if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    float amax = 0.f;
#pragma unroll
    for(int i = 0; i < 8; ++i) amax = fmaxf(amax, red[i]);
    const float scale = amax > 0.f ? amax / 448.f : 1.f;
    uint8_t* qr = q + std::size_t(t) * K;
    for(int k = threadIdx.x; k < K; k += blockDim.x)
        qr[k] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(row[k] / scale, __NV_SATFINITE, __NV_E4M3));
    if(threadIdx.x == 0) rscale[t] = scale;
}

__global__ void scale_rows_kernel(const float* d, int ldd, const float* rscale, float alpha, float* out, int ldo, int N,
                                  int T, int add) {
    const std::size_t total = std::size_t(T) * N;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += std::size_t(gridDim.x) * blockDim.x) {
        const std::size_t t = i / N, n = i % N;
        const float v = d[t * ldd + n] * (rscale[t] * alpha);
        if(add) out[t * ldo + n] += v;
        else out[t * ldo + n] = v;
    }
}

unsigned grid_for(std::size_t n) { return unsigned(std::min<std::size_t>((n + 255) / 256, 4096)); }

}  // namespace

void embed_rows(const __nv_bfloat16* table, const int* ids, float* out, int tokens, cudaStream_t stream) {
    embed_kernel<<<tokens, kHidden / 8, 0, stream>>>(table, ids, nullptr, out);
    NMH_CUDA(cudaGetLastError());
}
void embed_steps(const __nv_bfloat16* table, const Step* steps, float* out, int tokens, cudaStream_t stream) {
    embed_kernel<<<tokens, kHidden / 8, 0, stream>>>(table, nullptr, steps, out);
    NMH_CUDA(cudaGetLastError());
}
void rmsnorm(const float* x, const float* w, float* y, int tokens, cudaStream_t stream, int ldy) {
    rmsnorm_kernel<0><<<tokens, 1024, 0, stream>>>(x, w, y, ldy);
    NMH_CUDA(cudaGetLastError());
}
void rmsnorm_h(const float* x, const float* w, __half* y, int tokens, cudaStream_t stream, int ldy) {
    rmsnorm_kernel<1><<<tokens, 1024, 0, stream>>>(x, w, y, ldy);
    NMH_CUDA(cudaGetLastError());
}
void rmsnorm_bf(const float* x, const float* w, __nv_bfloat16* y, int tokens, cudaStream_t stream) {
    rmsnorm_kernel<2><<<tokens, 1024, 0, stream>>>(x, w, y, kHidden);
    NMH_CUDA(cudaGetLastError());
}
void to_bf16(const float* x, __nv_bfloat16* y, std::size_t n, cudaStream_t stream) {
    to_bf16_kernel<<<grid_for(n), 256, 0, stream>>>(x, y, n);
    NMH_CUDA(cudaGetLastError());
}
void to_f16(const float* x, __half* y, std::size_t n, cudaStream_t stream) {
    to_f16_kernel<<<grid_for(n), 256, 0, stream>>>(x, y, n);
    NMH_CUDA(cudaGetLastError());
}
void add_inplace(float* y, const float* x, std::size_t n, cudaStream_t stream) {
    add_kernel<<<grid_for(n), 256, 0, stream>>>(y, x, n);
    NMH_CUDA(cudaGetLastError());
}
void kv_write_steps(const KvCache& c, const float* proj, int ld, const Step* steps, __nv_bfloat16* q16, int tokens,
                    cudaStream_t stream) {
    kv_write_kernel<<<dim3(tokens, kKvHeads), kHeadDim, 0, stream>>>(c, proj, ld, steps, q16);
    NMH_CUDA(cudaGetLastError());
}
void quant_rows_fp8(const float* x, int ldx, int K, int T, uint8_t* q, float* rscale, cudaStream_t stream) {
    quant_rows_kernel<<<T, 256, 0, stream>>>(x, ldx, K, q, rscale);
    NMH_CUDA(cudaGetLastError());
}
void scale_rows(const float* d, int ldd, const float* rscale, float alpha, float* out, int ldo, int N, int T, bool add,
                cudaStream_t stream) {
    scale_rows_kernel<<<grid_for(std::size_t(T) * N), 256, 0, stream>>>(d, ldd, rscale, alpha, out, ldo, N, T, add ? 1 : 0);
    NMH_CUDA(cudaGetLastError());
}
void row_scores(const float* logits, int ld, int valid, int rows, const int* targets, const int* queries, int per,
                float* tlp, float* qlp, int* amax, cudaStream_t stream) {
    if(rows <= 0) return;
    row_scores_kernel<<<rows, 1024, 0, stream>>>(logits, ld, valid, targets, queries, per, tlp, qlp, amax);
    NMH_CUDA(cudaGetLastError());
}

}  // namespace nmh
