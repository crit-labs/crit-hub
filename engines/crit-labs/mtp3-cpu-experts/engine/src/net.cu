#include "net.h"
#include <crit/json.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <sys/mman.h>
#include <sys/prctl.h>

namespace nmh {

using crit::core::CudaBuffer;
using crit::core::Memory;
using crit::core::Queue;

// Pinned host memory filled before it is pinned: anonymous pages are written by the loading threads (page faults
// in parallel), then registered as mapped memory in one call. cudaMallocHost of 60 GB takes about 90 s under WSL2;
// this way the registration takes a few seconds.
struct HostBlock {
    void* data = nullptr;
    std::size_t bytes = 0;
    bool registered = false;
    explicit HostBlock(std::size_t n) : bytes(n) {
        data = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(data == MAP_FAILED) throw std::runtime_error("host allocation of " + std::to_string(bytes) + " bytes failed");
    }
    void pin() {
        NMH_CUDA(cudaHostRegister(data, bytes, cudaHostRegisterMapped));
        registered = true;
    }
    // Reads one word of every page with `threads` threads. The first CPU request after loading otherwise paid about
    // 10 s of page-table work on the records the CPU miss path reads (seen only with a fresh block).
    void touch(int threads) const {
        const std::size_t pages = bytes / 4096;
        std::atomic<unsigned long long> sink{0};
        std::vector<std::thread> pool;
        for(int i = 0; i < threads; ++i)
            pool.emplace_back([&, i] {
                unsigned long long acc = 0;
                const auto* p = static_cast<const volatile unsigned long long*>(data);
                for(std::size_t k = pages * i / threads; k < pages * (i + 1) / threads; ++k) acc += p[k * 512];
                sink += acc;
            });
        for(auto& t : pool) t.join();
    }
    void* device_address() const {
        void* d = nullptr;
        NMH_CUDA(cudaHostGetDevicePointer(&d, data, 0));
        return d;
    }
    ~HostBlock() {
        if(registered) cudaHostUnregister(data);
        if(data) ::munmap(data, bytes);
    }
};

namespace {
const std::string P = "language_model.backbone.";

std::vector<float> floats_of(const Tensors::Entry& e) {
    if(e.dtype == "F32") {
        std::vector<float> out(e.bytes.size() / 4);
        std::memcpy(out.data(), e.bytes.data(), e.bytes.size());
        return out;
    }
    if(e.dtype != "BF16") throw std::runtime_error("expected F32/BF16 tensor");
    std::vector<float> out(e.bytes.size() / 2);
    const auto* p = reinterpret_cast<const uint16_t*>(e.bytes.data());
    for(std::size_t i = 0; i < out.size(); ++i) {
        const uint32_t u = uint32_t(p[i]) << 16;
        std::memcpy(&out[i], &u, 4);
    }
    return out;
}
float scalar_of(const Tensors::Entry& e) {
    const auto v = floats_of(e);
    if(v.size() != 1) throw std::runtime_error("expected a scalar tensor");
    return v[0];
}
void expect_shape(const Tensors::Entry& e, std::initializer_list<std::uint64_t> shape, const std::string& name) {
    if(e.shape != std::vector<std::uint64_t>(shape)) throw std::runtime_error("unexpected shape of " + name);
}

// Collects device tensors, then allocates one buffer and uploads them.
class Packer {
public:
    template <class T> void add(const T** dst, std::span<const std::byte> src) {
        items_.push_back({reinterpret_cast<const void**>(dst), src, {}});
    }
    template <class T> void add(const T** dst, const std::vector<float>& owned) {
        Item it{reinterpret_cast<const void**>(dst), {}, {}};
        it.owned.resize(owned.size() * 4);
        std::memcpy(it.owned.data(), owned.data(), it.owned.size());
        items_.push_back(std::move(it));
    }
    std::size_t bytes() const {
        std::size_t total = 0;
        for(auto& it : items_) total += (size(it) + 255) / 256 * 256;
        return total;
    }
    CudaBuffer finish(crit::core::CudaBackend& backend) {
        CudaBuffer buffer = backend.allocate(Memory::device, bytes());
        std::size_t off = 0;
        auto* base = static_cast<uint8_t*>(buffer.data());
        for(auto& it : items_) {
            const auto src = it.owned.empty() ? it.src : std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(it.owned.data()), it.owned.size());
            buffer.upload(src, off);
            *it.dst = base + off;
            off += (size(it) + 255) / 256 * 256;
        }
        items_.clear();
        return buffer;
    }
private:
    struct Item {
        const void** dst;
        std::span<const std::byte> src;
        std::vector<uint8_t> owned;
    };
    static std::size_t size(const Item& it) { return it.owned.empty() ? it.src.size() : it.owned.size(); }
    std::vector<Item> items_;
};

// Host-side weights kept across engines of one process when Options::keep_host_weights is set.
struct HostWeights {
    std::string model;
    std::shared_ptr<HostBlock> experts, embed;
    std::vector<float> s2;
};
std::shared_ptr<HostWeights>& host_cache() {
    static std::shared_ptr<HostWeights> cache;
    return cache;
}

template <class T> T* carve(uint8_t*& p, std::size_t count) {
    T* r = reinterpret_cast<T*>(p);
    p += (count * sizeof(T) + 255) / 256 * 256;
    return r;
}
template <class T> std::size_t carve_size(std::size_t count) { return (count * sizeof(T) + 255) / 256 * 256; }
}  // namespace

// ------------------------------------------------------------------------------------------------ Tensors
Tensors::Tensors(const std::filesystem::path& model) {
    const auto index = crit::core::json::Document::read_file(model / "model.safetensors.index.json");
    const auto* wm = index.root().find("weight_map");
    if(!wm) throw std::runtime_error("index without weight_map");
    std::map<std::string, std::size_t> file_of;
    for(const auto& m : wm->object()) {
        const std::string rel(m.value.string());
        if(!file_of.count(rel)) {
            file_of[rel] = files_.size();
            files_.push_back(crit::core::SafetensorsFile::open(model / rel));
        }
    }
    for(std::size_t fi = 0; fi < files_.size(); ++fi) {
        auto& f = files_[fi];
        const auto whole = f.file().map();
        owners_.push_back(f.file().mapping_owner());
        for(const auto& t : f.tensors()) {
            Entry e;
            e.bytes = whole.subspan(t.begin, t.end - t.begin);
            e.dtype = std::string(crit::core::dtype_name(t.dtype));
            e.shape = t.shape;
            e.file = std::uint32_t(fi);
            e.offset = t.begin;
            map_[t.name] = std::move(e);
        }
    }
}

const Tensors::Entry& Tensors::at(const std::string& name) const {
    auto it = map_.find(name);
    if(it == map_.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
}

void Tensors::gather(std::vector<Copy> copies, int threads) const {
    std::vector<std::vector<Copy>> per_file(files_.size());
    for(const auto& c : copies) per_file[c.src->file].push_back(c);
    for(auto& v : per_file)
        std::sort(v.begin(), v.end(), [](const Copy& a, const Copy& b) { return a.src->offset < b.src->offset; });
    std::atomic<std::size_t> next{0};
    std::string error;
    std::mutex error_mutex;
    auto worker = [&] {
        for(;;) {
            const std::size_t fi = next.fetch_add(1);
            if(fi >= per_file.size()) return;
            const auto& list = per_file[fi];
            try {
                std::size_t i = 0;
                while(i < list.size()) {
                    // A run of sources with gaps below 32 MiB, read as one sequential range.
                    std::size_t j = i + 1;
                    std::uint64_t end = list[i].src->offset + list[i].src->bytes.size();
                    while(j < list.size() && list[j].src->offset < end + (std::uint64_t(32) << 20)) {
                        end = std::max<std::uint64_t>(end, list[j].src->offset + list[j].src->bytes.size());
                        ++j;
                    }
                    const std::uint64_t begin = list[i].src->offset;
                    std::size_t cursor = i;
                    crit::core::ReadOptions ro;
                    ro.chunk_bytes = std::size_t(64) << 20;
                    ro.direct = true;
                    files_[fi].file().read_chunks(begin, end - begin, [&](std::span<const std::byte> bytes,
                                                                          std::uint64_t off) {
                        const std::uint64_t cend = off + bytes.size();
                        while(cursor < j && list[cursor].src->offset + list[cursor].src->bytes.size() <= off) ++cursor;
                        for(std::size_t k = cursor; k < j && list[k].src->offset < cend; ++k) {
                            const std::uint64_t s = list[k].src->offset, e = s + list[k].src->bytes.size();
                            const std::uint64_t lo = std::max(s, off), hi = std::min(e, cend);
                            if(lo < hi)
                                std::memcpy(static_cast<std::byte*>(list[k].dst) + (lo - s), bytes.data() + (lo - off),
                                            hi - lo);
                        }
                    }, ro);
                    i = j;
                }
            } catch(const std::exception& ex) {
                std::lock_guard<std::mutex> g(error_mutex);
                error = ex.what();
            }
        }
    };
    std::vector<std::thread> pool;
    for(int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for(auto& t : pool) t.join();
    if(!error.empty()) throw std::runtime_error(error);
}

// ------------------------------------------------------------------------------------------------ Net
Net::Net(const Options& options, crit::core::CudaBackend& backend) : options_(options), backend_(backend) {
    // A core dump of this process would hold tens of GB of pinned expert records (WSL writes it to Windows temp).
    ::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
    if(options_.capacity < 1 || options_.prefill_chunk < 1 || options_.prefill_chunk % options_.moe_sub)
        throw std::invalid_argument("bad capacity / prefill_chunk / moe_sub");
    const auto config = crit::core::json::Document::read_file(options_.model / "config.json");
    const auto* text = config.root().find("llm_config");
    if(!text) throw std::runtime_error("config.json without llm_config");
    auto num = [&](const char* key) { return text->find(key)->i64(); };
    if(num("hidden_size") != kHidden || num("vocab_size") != kVocab || num("mamba_num_heads") != kMHeads ||
       num("mamba_head_dim") != kMHeadDim || num("n_groups") != kMGroups || num("ssm_state_size") != kMState ||
       num("conv_kernel") != kConv || num("num_attention_heads") != kQHeads || num("num_key_value_heads") != kKvHeads ||
       num("head_dim") != kHeadDim || num("n_routed_experts") != kExperts || num("num_experts_per_tok") != kTopK ||
       num("moe_latent_size") != kLatent || num("moe_intermediate_size") != kInter ||
       num("moe_shared_expert_intermediate_size") != kShared || num("n_group") != 1)
        throw std::runtime_error("config.json does not match the dimensions this engine is built for");
    const auto& types = text->find("layers_block_type")->array();
    if(types.size() != std::size_t(kLayers)) throw std::runtime_error("expected 88 layers");
    layers_.resize(kLayers);
    int counts[3] = {0, 0, 0};
    for(int l = 0; l < kLayers; ++l) {
        const std::string t(types[l].string());
        const Kind k = t == "mamba" ? Kind::mamba : t == "moe" ? Kind::moe : t == "attention" ? Kind::attn
                     : throw std::runtime_error("unknown layer type " + t);
        layers_[l].kind = k;
        layers_[l].index = counts[int(k)]++;
    }
    if(counts[0] != kMambaLayers || counts[1] != kMoeLayers || counts[2] != kAttnLayers)
        throw std::runtime_error("unexpected layer mix");

    const auto t0 = std::chrono::steady_clock::now();
    auto mark = [&](const char* what) {
        std::fprintf(stderr, "nmh: %s at %.1f s\n", what,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    };
    Tensors tensors(options_.model);
    load_dense(tensors);
    if(options_.mtp) load_mtp(tensors);
    mark("dense weights");
    load_experts(tensors);
    mark("experts");
    allocate_state();
    allocate_prefill();
    allocate_slots();
    if(options_.cpu_share > 0.f) {
        // Mailbox for the CPU's share of decode misses (mapped pinned memory) and its server threads.
        mailbox_ = backend_.allocate(Memory::host, sizeof(CpuJob) + sizeof(CpuReply) + 4096);
        std::memset(mailbox_.data(), 0, mailbox_.size());
        auto* job = static_cast<CpuJob*>(mailbox_.data());
        auto* reply = reinterpret_cast<CpuReply*>(static_cast<uint8_t*>(mailbox_.data()) + (sizeof(CpuJob) + 4095) / 4096 * 4096);
        auto* djob = static_cast<CpuJob*>(mailbox_.device_address());
        auto* dreply = reinterpret_cast<const CpuReply*>(static_cast<uint8_t*>(mailbox_.device_address()) +
                                                         (sizeof(CpuJob) + 4095) / 4096 * 4096);
        tables_.job = djob;
        tables_.reply = dreply;
        tables_.cpu_share = options_.cpu_share;
        tables_.cpu_min = options_.cpu_min;
        cpu_server_ = std::make_unique<CpuMissServer>(job, reply, host_experts_, options_.cpu_threads);
    }
    reset();
    mark("state");
}

Net::~Net() {
    for(auto& kv : graphs_)
        if(kv.second) cudaGraphExecDestroy(kv.second);
    if(round_graph_) cudaGraphExecDestroy(round_graph_);
    if(begin_graph_) cudaGraphExecDestroy(begin_graph_);
    if(spec_done_) cudaEventDestroy(spec_done_);
    for(auto e : marks_)
        if(e) cudaEventDestroy(e);
    if(prefetch_ready_) cudaEventDestroy(prefetch_ready_);
    if(moe_done_) cudaEventDestroy(moe_done_);
}

cudaStream_t Net::stream() const { return backend_.stream(Queue::compute); }

void Net::load_dense(const Tensors& t) {
    Packer pk;
    auto fp8 = [&](Fp8W& m, const std::string& base, int rows, int cols) {
        const auto& w = t.at(base + ".weight");
        expect_shape(w, {std::uint64_t(rows), std::uint64_t(cols)}, base);
        if(w.dtype != "F8_E4M3") throw std::runtime_error("expected FP8 " + base);
        m.scale = scalar_of(t.at(base + ".weight_scale"));
        pk.add(&m.w, w.bytes);
    };
    auto fp4 = [&](Fp4W& m, const std::string& base, int rows, int cols) {
        const auto& w = t.at(base + ".weight");
        const auto& s = t.at(base + ".weight_scale");
        expect_shape(w, {std::uint64_t(rows), std::uint64_t(cols / 2)}, base);
        expect_shape(s, {std::uint64_t(rows), std::uint64_t(cols / 16)}, base);
        if(w.dtype != "U8" || s.dtype != "F8_E4M3") throw std::runtime_error("expected NVFP4 " + base);
        m.scale = scalar_of(t.at(base + ".weight_scale_2"));
        m.in_scale = scalar_of(t.at(base + ".input_scale"));
        pk.add(&m.w, w.bytes);
        pk.add(&m.s, s.bytes);
    };
    auto bf16 = [&](const __nv_bfloat16** dst, const std::string& name, std::initializer_list<std::uint64_t> shape) {
        const auto& e = t.at(name);
        expect_shape(e, shape, name);
        if(e.dtype != "BF16") throw std::runtime_error("expected BF16 " + name);
        pk.add(dst, e.bytes);
    };
    for(int l = 0; l < kLayers; ++l) {
        auto& L = layers_[l];
        const std::string lp = P + "layers." + std::to_string(l) + ".";
        pk.add(&L.norm, floats_of(t.at(lp + "norm.weight")));
        const std::string mx = lp + "mixer.";
        if(L.kind == Kind::mamba) {
            fp8(L.in_proj, mx + "in_proj", kInProj, kHidden);
            fp8(L.out_proj, mx + "out_proj", kHidden, kMInner);
            pk.add(&L.m.conv_w, floats_of(t.at(mx + "conv1d.weight")));
            pk.add(&L.m.conv_b, floats_of(t.at(mx + "conv1d.bias")));
            pk.add(&L.m.dt_bias, floats_of(t.at(mx + "dt_bias")));
            auto alog = floats_of(t.at(mx + "A_log"));
            for(float& x : alog) x = -std::exp(x);
            pk.add(&L.m.neg_a, alog);
            pk.add(&L.m.d, floats_of(t.at(mx + "D")));
            pk.add(&L.m.norm_w, floats_of(t.at(mx + "norm.weight")));
            L.m.dt_min = options_.dt_min;
        } else if(L.kind == Kind::attn) {
            fp8(L.q, mx + "q_proj", kQDim, kHidden);
            fp8(L.k, mx + "k_proj", kKvDim, kHidden);
            fp8(L.v, mx + "v_proj", kKvDim, kHidden);
            fp8(L.o, mx + "o_proj", kHidden, kQDim);
        } else {
            L.moe.group = L.index;
            bf16(&L.moe.router, mx + "gate.weight", {kExperts, kHidden});
            pk.add(&L.moe.bias, floats_of(t.at(mx + "gate.e_score_correction_bias")));
            bf16(&L.moe.fc1, mx + "fc1_latent_proj.weight", {kLatent, kHidden});
            bf16(&L.moe.fc2, mx + "fc2_latent_proj.weight", {kHidden, kLatent});
            fp4(L.moe.sh_up, mx + "shared_experts.up_proj", kShared, kHidden);
            fp4(L.moe.sh_down, mx + "shared_experts.down_proj", kHidden, kShared);
        }
    }
    pk.add(&norm_f_, floats_of(t.at(P + "norm_f.weight")));
    fp8(lm_head_, "language_model.lm_head", kVocab, kHidden);
    dense_bytes_ = pk.bytes();
    buffers_.push_back(pk.finish(backend_));
    if(options_.prompt_fp4) {
        // Shared experts' weight scales in the Sm1xx layout and their W4A4 alphas, for the prompt path.
        std::vector<Fp4W*> mats;
        for(auto& L : layers_)
            if(L.kind == Kind::moe) {
                mats.push_back(&L.moe.sh_up);
                mats.push_back(&L.moe.sh_down);
            }
        const std::size_t per = std::size_t(kShared) * (kHidden / 16) + std::size_t(kHidden) * (kShared / 16);
        auto buf = backend_.allocate(Memory::device, mats.size() / 2 * per + 256 + mats.size() * 4);
        uint8_t* p = static_cast<uint8_t*>(buf.data());
        float* alphas = reinterpret_cast<float*>(p + mats.size() / 2 * per + 256);
        std::vector<float> ha(mats.size());
        for(std::size_t i = 0; i < mats.size(); ++i) {
            Fp4W& m = *mats[i];
            const bool up = i % 2 == 0;
            const int rows = up ? kShared : kHidden, k16 = (up ? kHidden : kShared) / 16;
            swizzle_rows(m.s, rows, k16, p, stream());
            m.sf = p;
            p += std::size_t(rows) * k16;
            m.alpha = alphas + i;
            ha[i] = m.in_scale * m.scale;
        }
        NMH_CUDA(cudaMemcpy(alphas, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
        NMH_CUDA(cudaStreamSynchronize(stream()));
        buffers_.push_back(std::move(buf));
    }
    // Token embeddings: pinned host, read through the device mapping.
    auto& cache = host_cache();
    if(cache && cache->model == options_.model.string()) {
        embed_block_ = cache->embed;
        embed_ = static_cast<const __nv_bfloat16*>(embed_block_->device_address());
        return;
    }
    const auto& emb = t.at(P + "embeddings.weight");
    expect_shape(emb, {kVocab, kHidden}, "embeddings");
    auto host = std::make_shared<HostBlock>(emb.bytes.size());
    t.gather({{&emb, host->data}}, 1);
    host->pin();
    embed_ = static_cast<const __nv_bfloat16*>(host->device_address());
    if(options_.keep_host_weights) {
        cache = std::make_shared<HostWeights>();
        cache->model = options_.model.string();
        cache->embed = host;
    }
    embed_block_ = host;
}

void Net::load_mtp(const Tensors& t) {
    const std::string mp = "language_model.mtp.layers.";
    Packer pk;
    auto bf16 = [&](const __nv_bfloat16** dst, const std::string& name, std::initializer_list<std::uint64_t> shape) {
        const auto& e = t.at(name);
        expect_shape(e, shape, name);
        if(e.dtype != "BF16") throw std::runtime_error("expected BF16 " + name);
        pk.add(dst, e.bytes);
    };
    auto fp4 = [&](Fp4W& m, const std::string& base, int rows, int cols) {
        const auto& w = t.at(base + ".weight");
        const auto& sc = t.at(base + ".weight_scale");
        expect_shape(w, {std::uint64_t(rows), std::uint64_t(cols / 2)}, base);
        expect_shape(sc, {std::uint64_t(rows), std::uint64_t(cols / 16)}, base);
        m.scale = scalar_of(t.at(base + ".weight_scale_2"));
        pk.add(&m.w, w.bytes);
        pk.add(&m.s, sc.bytes);
    };
    const std::string a = mp + "0.", m = mp + "1.";
    pk.add(&mtp_.enorm, floats_of(t.at(a + "enorm.weight")));
    pk.add(&mtp_.hnorm, floats_of(t.at(a + "hnorm.weight")));
    bf16(&mtp_.eh_proj, a + "eh_proj.weight", {kHidden, 2 * kHidden});
    pk.add(&mtp_.norm0, floats_of(t.at(a + "norm.weight")));
    bf16(&mtp_.q, a + "mixer.q_proj.weight", {kQDim, kHidden});
    bf16(&mtp_.k, a + "mixer.k_proj.weight", {kKvDim, kHidden});
    bf16(&mtp_.v, a + "mixer.v_proj.weight", {kKvDim, kHidden});
    bf16(&mtp_.o, a + "mixer.o_proj.weight", {kHidden, kQDim});
    pk.add(&mtp_.norm1, floats_of(t.at(m + "norm.weight")));
    mtp_.moe.group = kMtpGroup;
    bf16(&mtp_.moe.router, m + "mixer.gate.weight", {kExperts, kHidden});
    pk.add(&mtp_.moe.bias, floats_of(t.at(m + "mixer.gate.e_score_correction_bias")));
    bf16(&mtp_.moe.fc1, m + "mixer.fc1_latent_proj.weight", {kLatent, kHidden});
    bf16(&mtp_.moe.fc2, m + "mixer.fc2_latent_proj.weight", {kHidden, kLatent});
    fp4(mtp_.moe.sh_up, m + "mixer.shared_experts.up_proj", kShared, kHidden);
    fp4(mtp_.moe.sh_down, m + "mixer.shared_experts.down_proj", kHidden, kShared);
    pk.add(&mtp_.final_norm, floats_of(t.at(m + "final_layernorm.weight")));
    // A draft step reads the drafter's dense weights, one token's routed experts and the shared FP8 head.
    draft_bytes_ = pk.bytes() + std::size_t(kTopK) * kSlotBytes + std::size_t(kVocab) * kHidden;
    buffers_.push_back(pk.finish(backend_));
    mtp_loaded_ = true;
}

void Net::load_experts(const Tensors& t) {
    auto& cache = host_cache();
    const bool reuse = cache && cache->model == options_.model.string() && cache->experts;
    const int groups = mtp_loaded_ ? kGroups : kMoeLayers;
    const std::size_t bytes = std::size_t(groups) * kExperts * kSlotBytes;
    const bool reuse_ok = reuse && cache->experts->bytes >= bytes;
    // A cached block too small for this engine (e.g. kept by an engine without the MTP group) is released first:
    // keeping it while allocating the new one would hold two copies of the experts (OOM at ~124 GB).
    if(reuse && !reuse_ok) cache->experts.reset();
    const auto ta = std::chrono::steady_clock::now();
    std::shared_ptr<HostBlock> host = reuse_ok ? cache->experts : std::make_shared<HostBlock>(bytes);
    std::fprintf(stderr, "nmh: host allocation %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count());
    host_experts_ = static_cast<uint8_t*>(host->data);
    if(reuse_ok) {
        s2_host_ = cache->s2;
        std::fprintf(stderr, "nmh: expert records reused from the previous engine of this process\n");
    } else {
    const auto t0 = std::chrono::steady_clock::now();
    s2_host_.assign(std::size_t(kGroups) * kExperts * 2, 0.f);
    std::vector<Tensors::Copy> copies;
    copies.reserve(std::size_t(groups) * kExperts * 4);
    std::vector<std::pair<int, std::string>> sources;   // (group, prefix of its experts)
    for(int l = 0; l < kLayers; ++l)
        if(layers_[l].kind == Kind::moe)
            sources.push_back({layers_[l].index, P + "layers." + std::to_string(l) + ".mixer.experts."});
    if(mtp_loaded_) sources.push_back({kMtpGroup, "language_model.mtp.layers.1.mixer.experts."});
    for(const auto& [mi, prefix] : sources) {
        for(int e = 0; e < kExperts; ++e) {
            const std::string base = prefix + std::to_string(e) + ".";
            uint8_t* rec = host_experts_ + (std::size_t(mi) * kExperts + e) * kSlotBytes;
            const auto& uw = t.at(base + "up_proj.weight");
            const auto& dw = t.at(base + "down_proj.weight");
            const auto& us = t.at(base + "up_proj.weight_scale");
            const auto& ds = t.at(base + "down_proj.weight_scale");
            if(uw.bytes.size() != kUpBytes || dw.bytes.size() != kDownBytes || us.bytes.size() != kUpScales ||
               ds.bytes.size() != kDownScales || uw.dtype != "U8" || us.dtype != "F8_E4M3")
                throw std::runtime_error("unexpected expert layout " + base);
            copies.push_back({&uw, rec});
            copies.push_back({&dw, rec + kOffDown});
            copies.push_back({&us, rec + kOffUpS});
            copies.push_back({&ds, rec + kOffDownS});
            s2_host_[(std::size_t(mi) * kExperts + e) * 2] = scalar_of(t.at(base + "up_proj.weight_scale_2"));
            s2_host_[(std::size_t(mi) * kExperts + e) * 2 + 1] = scalar_of(t.at(base + "down_proj.weight_scale_2"));
        }
    }
    std::fprintf(stderr, "nmh: expert list %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const auto tg = std::chrono::steady_clock::now();
    t.gather(std::move(copies), options_.load_threads);
    std::fprintf(stderr, "nmh: read %.1f GiB of experts in %.1f s\n", double(bytes) / double(1ull << 30),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count());
    const auto tp = std::chrono::steady_clock::now();
    host->pin();
    std::fprintf(stderr, "nmh: pinned the expert records in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
    if(options_.cpu_share > 0.f) {
        const auto tt = std::chrono::steady_clock::now();
        host->touch(options_.load_threads);
        std::fprintf(stderr, "nmh: touched the expert records in %.1f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - tt).count());
    }
    if(options_.keep_host_weights && cache && cache->model == options_.model.string()) {
        cache->experts = host;
        cache->s2 = s2_host_;
    }
    }
    tables_.host = static_cast<const uint8_t*>(host->device_address());
    experts_block_ = host;
    // Routed experts' activation scales: ModelOpt calibrates one input_scale per layer and projection, the same
    // for every expert (checked here).
    in_scale_.assign(std::size_t(kGroups) * 2, 0.f);
    {
        std::vector<std::pair<int, std::string>> sources;
        for(int l = 0; l < kLayers; ++l)
            if(layers_[l].kind == Kind::moe)
                sources.push_back({layers_[l].index, P + "layers." + std::to_string(l) + ".mixer.experts."});
        if(mtp_loaded_) sources.push_back({kMtpGroup, "language_model.mtp.layers.1.mixer.experts."});
        for(const auto& [mi, prefix] : sources)
            for(int j = 0; j < 2; ++j) {
                const char* proj = j ? "down_proj" : "up_proj";
                const float v = scalar_of(t.at(prefix + "0." + proj + ".input_scale"));
                for(int e = 1; e < kExperts; ++e)
                    if(scalar_of(t.at(prefix + std::to_string(e) + "." + proj + ".input_scale")) != v)
                        throw std::runtime_error("routed experts with different input_scale: " + prefix);
                in_scale_[std::size_t(mi) * 2 + j] = v;
            }
    }
    // Global scales on the device; prompt-path alphas (down scaled back from the FP16 activation scale).
    std::vector<float> palpha(std::size_t(kGroups) * kExperts * 2);
    for(std::size_t i = 0; i < palpha.size(); i += 2) {
        palpha[i] = s2_host_[i];
        palpha[i + 1] = s2_host_[i + 1] / kActScale;
    }
    std::vector<float> palpha4(palpha.size());
    for(std::size_t i = 0; i < palpha4.size(); ++i) palpha4[i] = in_scale_[(i / (2 * kExperts)) * 2 + i % 2] * s2_host_[i];
    const std::size_t s2b = s2_host_.size() * 4, pab = palpha.size() * 4;
    auto buf = backend_.allocate(Memory::device, s2b + 256 + pab + 256 + pab);
    buf.upload(std::as_bytes(std::span(s2_host_)), 0);
    buf.upload(std::as_bytes(std::span(palpha)), s2b + 256);
    buf.upload(std::as_bytes(std::span(palpha4)), s2b + 256 + pab + 256);
    tables_.s2 = static_cast<const float*>(buf.data());
    palpha_ = reinterpret_cast<float*>(static_cast<uint8_t*>(buf.data()) + s2b + 256);
    palpha4_ = reinterpret_cast<float*>(static_cast<uint8_t*>(buf.data()) + s2b + 256 + pab + 256);
    buffers_.push_back(std::move(buf));
}

void Net::allocate_state() {
    const std::size_t cap = options_.capacity;
    ring_ = mtp_loaded_ ? kStateRing : 1;
    const int kv_layers = kAttnLayers + (mtp_loaded_ ? 1 : 0);
    std::size_t total = carve_size<float>(std::size_t(kMambaLayers) * kConvDim * 3) +
                        carve_size<float>(std::size_t(kMambaLayers) * kMaxTokensRec * kConvDim) +
                        carve_size<float>(std::size_t(kMambaLayers) * ring_ * kSsmFloats) + carve_size<int>(1);
    const std::size_t codes = cap * kKvDim, scales = cap * kKvHeads;
    total += kv_layers * 2 * (carve_size<uint8_t>(codes) + carve_size<float>(scales));
    // speculative rounds
    total += carve_size<Accept>(1) + carve_size<DraftCtl>(1) + carve_size<int>(kMaxForced + 1) +
             carve_size<float>(kMaxTokens * kHidden) * 3 + carve_size<float>(kMaxTokens * 2 * kHidden) +
             carve_size<float>(kMaxTokens * kQkv) + carve_size<float>(kHidden) * 2 + carve_size<float>(kVocab) +
             carve_size<__nv_bfloat16>(kMaxTokens * kQDim) + carve_size<__nv_bfloat16>(kQDim) + carve_size<int>(4) +
             carve_size<int>(4);
    // decode workspace
    const std::size_t T = kMaxTokens;
    const std::size_t partial = crit::fp8_kv::decode_partial_floats(kKvHeads, kHeadDim, options_.attn_blocks);
    total += carve_size<Step>(T) + carve_size<float>(T * kHidden) * 2 + carve_size<float>(T * kInProj) +
             carve_size<float>(T * kConvDim) + carve_size<float>(T * kMInner) * 2 + carve_size<float>(T * kQDim) +
             carve_size<float>(T * kExperts) + carve_size<float>(T * kLatent) * 2 + carve_size<float>(T * kShared) +
             carve_size<float>(std::size_t(kMaxUnion) * kMaxPassTokens * kInter) +
             carve_size<float>(std::size_t(kMaxUnion) * kMaxPassTokens * kLatent) + carve_size<float>(T * kVocab) +
             carve_size<float>(partial) + carve_size<__nv_bfloat16>(T * kQDim) + carve_size<int>(T) +
             carve_size<Route>(T) + carve_size<ExpertPlan>(1) + carve_size<unsigned long long>(4);
    auto buf = backend_.allocate(Memory::device, total);
    NMH_CUDA(cudaMemset(buf.data(), 0, total));
    uint8_t* p = static_cast<uint8_t*>(buf.data());
    conv_state_ = carve<float>(p, std::size_t(kMambaLayers) * kConvDim * 3);
    conv_rec_ = carve<float>(p, std::size_t(kMambaLayers) * kMaxTokensRec * kConvDim);
    ssm_ = carve<float>(p, std::size_t(kMambaLayers) * ring_ * kSsmFloats);
    ssm_cur_ = carve<int>(p, 1);
    kv_.resize(kv_layers);
    kvc_.resize(kv_layers);
    for(int a = 0; a < kv_layers; ++a) {
        kv_[a].k = carve<uint8_t>(p, codes);
        kv_[a].v = carve<uint8_t>(p, codes);
        kv_[a].ks = carve<float>(p, scales);
        kv_[a].vs = carve<float>(p, scales);
        kvc_[a] = crit::fp8_kv::Cache{kv_[a].k, kv_[a].v, kv_[a].ks, kv_[a].vs, kKvHeads, kHeadDim};
    }
    step_ = carve<Step>(p, T);
    H_ = carve<float>(p, T * kHidden);
    X_ = carve<float>(p, T * kHidden);
    P_ = carve<float>(p, T * kInProj);
    XC_ = carve<float>(p, T * kConvDim);
    Y_ = carve<float>(p, T * kMInner);
    G_ = carve<float>(p, T * kMInner);
    AO_ = carve<float>(p, T * kQDim);
    RL_ = carve<float>(p, T * kExperts);
    LAT_ = carve<float>(p, T * kLatent);
    LO_ = carve<float>(p, T * kLatent);
    SH_ = carve<float>(p, T * kShared);
    ACT_ = carve<float>(p, std::size_t(kMaxUnion) * kMaxPassTokens * kInter);
    PART_ = carve<float>(p, std::size_t(kMaxUnion) * kMaxPassTokens * kLatent);
    logits_ = carve<float>(p, T * kVocab);
    attn_partial_ = carve<float>(p, partial);
    Q16_ = carve<__nv_bfloat16>(p, T * kQDim);
    next_ = carve<int>(p, T);
    routes_ = carve<Route>(p, T);
    plan_ = carve<ExpertPlan>(p, 1);
    stats_ = carve<unsigned long long>(p, 4);
    acc_ = carve<Accept>(p, 1);
    dctl_ = carve<DraftCtl>(p, 1);
    forced_ = carve<int>(p, kMaxForced + 1);   // [0]: length, then the tokens
    mE_ = carve<float>(p, kMaxTokens * kHidden);
    mF_ = carve<float>(p, kMaxTokens * kHidden);
    mX_ = carve<float>(p, kMaxTokens * kHidden);
    mO_ = carve<float>(p, kMaxTokens * 2 * kHidden);   // [R][8192]: enorm(embedding) | hnorm(hidden)
    mP_ = carve<float>(p, kMaxTokens * kQkv);
    mR_ = carve<float>(p, kHidden);
    mtp_out_ = carve<float>(p, kHidden);
    mlogits_ = carve<float>(p, kVocab);
    mQ16_ = carve<__nv_bfloat16>(p, kMaxTokens * kQDim);
    mQsel_ = carve<__nv_bfloat16>(p, kQDim);
    draft_tok_ = carve<int>(p, 4);   // [0] draft, [1] zero (row index of a single-row pass)
    tables_.cpu_state = carve<int>(p, 4);
    buffers_.push_back(std::move(buf));
    auto host = backend_.allocate(Memory::host, 4096);
    step_host_ = static_cast<Step*>(host.data());
    spec_host_ = reinterpret_cast<Accept*>(static_cast<uint8_t*>(host.data()) + 1024);
    buffers_.push_back(std::move(host));
    tables_.stats = stats_;
    NMH_CUDA(cudaEventCreateWithFlags(&spec_done_, cudaEventDisableTiming));
    NMH_CUDA(cudaEventCreateWithFlags(&prefetch_ready_, cudaEventDisableTiming));
    NMH_CUDA(cudaEventCreateWithFlags(&moe_done_, cudaEventDisableTiming));
    for(auto& e : marks_) NMH_CUDA(cudaEventCreate(&e));
}

void Net::allocate_prefill() {
    const std::size_t C = options_.prefill_chunk, S = options_.moe_sub, R = S * kTopK;
    const int nsub = int(C / S);
    max_tiles_ = int(R / 128 + kExperts);
    std::size_t total = carve_size<int>(C + 1) + carve_size<float>(C * kHidden) + carve_size<__half>(C * kMInner) +
                        carve_size<float>(C * kInProj) + carve_size<float>(C * kConvDim) +
                        carve_size<float>(C * kMInner) + carve_size<__nv_bfloat16>(C * kQkv) +
                        carve_size<__half>(C * kLatent) + carve_size<__half>(C * kShared) +
                        carve_size<float>(C * kExperts) + carve_size<Route>(C) + carve_size<__half>(R * kInter) +
                        carve_size<float>(R * kLatent) + carve_size<__half>(S * kLatent) + carve_size<int>(R) * 2 +
                        carve_size<int>(std::size_t(nsub) * kExperts) * 2 + carve_size<int>(kExperts) +
                        carve_size<int4>(std::size_t(nsub) * max_tiles_) + carve_size<int>(nsub) +
                        carve_size<const uint8_t*>(kExperts) + carve_size<float>(std::size_t(kScoreRows) * kVocab) +
                        carve_size<uint8_t>(C * kMInner) + carve_size<float>(C) + carve_size<uint8_t>(std::size_t(32) << 20);
    const std::size_t sf_rows = R + std::size_t(kExperts) * 127;
    if(options_.prompt_fp4)
        total += carve_size<uint8_t>(R * kLatent / 2) + carve_size<uint8_t>(sf_rows * kUpK16) +
                 carve_size<uint8_t>(R * kInter / 2) + carve_size<uint8_t>(sf_rows * kDownK16) +
                 carve_size<uint8_t>(std::size_t(kExperts) * kSfExpertBytes) +
                 carve_size<int>(std::size_t(nsub) * kExperts) + carve_size<uint8_t>(C * kHidden / 2) +
                 carve_size<uint8_t>((C + 127) * (kHidden / 16)) + carve_size<uint8_t>(C * kShared / 2) +
                 carve_size<uint8_t>((C + 127) * (kShared / 16));
    auto buf = backend_.allocate(Memory::device, total);
    uint8_t* p = static_cast<uint8_t*>(buf.data());
    // The big arrays first: between prompts they lend their memory to the expert cache.
    uint8_t* lend_begin = p;
    pH_ = carve<float>(p, C * kHidden);
    pX_ = carve<__half>(p, C * kMInner);
    pP_ = carve<float>(p, C * kInProj);
    pXC_ = carve<float>(p, C * kConvDim);
    pY_ = carve<float>(p, C * kMInner);
    pQKV_ = carve<__nv_bfloat16>(p, C * kQkv);
    pLAT_ = carve<__half>(p, C * kLatent);
    pSH_ = carve<__half>(p, C * kShared);
    pRL_ = carve<float>(p, C * kExperts);
    proutes_ = carve<Route>(p, C);
    pACT_ = carve<__half>(p, R * kInter);
    pROUT_ = carve<float>(p, R * kLatent);
    rout_bytes_ = R * kLatent * sizeof(float);
    pLO_ = carve<__half>(p, S * kLatent);
    if(options_.prompt_fp4) {
        pXq_ = carve<uint8_t>(p, R * kLatent / 2);
        pXsf_ = carve<uint8_t>(p, sf_rows * kUpK16);
        pAq_ = carve<uint8_t>(p, R * kInter / 2);
        pAsf_ = carve<uint8_t>(p, sf_rows * kDownK16);
        pWsf_ = carve<uint8_t>(p, std::size_t(kExperts) * kSfExpertBytes);
        pShq_ = carve<uint8_t>(p, C * kHidden / 2);
        pShsf_ = carve<uint8_t>(p, (C + 127) * (kHidden / 16));
        pShaq_ = carve<uint8_t>(p, C * kShared / 2);
        pShasf_ = carve<uint8_t>(p, (C + 127) * (kShared / 16));
    }
    if(options_.lend_workspace) workspace_lend_.push_back({lend_begin, std::size_t(p - lend_begin)});
    pids_ = carve<int>(p, C + 1);
    ptoken_ = carve<int>(p, R);
    ppos_ = carve<int>(p, R);
    pcounts_ = carve<int>(p, std::size_t(nsub) * kExperts);
    poffsets_ = carve<int>(p, std::size_t(nsub) * kExperts);
    pcursor_ = carve<int>(p, kExperts);
    ptiles_ = carve<int4>(p, std::size_t(nsub) * max_tiles_);
    pntiles_ = carve<int>(p, nsub);
    pgroup_ = carve<const uint8_t*>(p, kExperts);
    pscore_ = carve<float>(p, std::size_t(kScoreRows) * kVocab);
    pq_ = carve<uint8_t>(p, C * kMInner);
    prs_ = carve<float>(p, C);
    uint8_t* lt_ws = carve<uint8_t>(p, std::size_t(32) << 20);
    lt_ = std::make_unique<Lt>(lt_ws, std::size_t(32) << 20);
    if(options_.prompt_fp4) {
        psfrow_ = carve<int>(p, std::size_t(nsub) * kExperts);
        // Both share the 32 MiB workspace: their calls are ordered on the one prompt stream.
        fp4g_ = std::make_unique<Fp4Grouped>(kExperts, lt_ws, std::size_t(32) << 20);
        fp4_groups_.reserve(kExperts);
    }
    buffers_.push_back(std::move(buf));
    const std::size_t hb = carve_size<int>(C + 1) + carve_size<int>(std::size_t(nsub) * kExperts) * 2 +
                           carve_size<int4>(std::size_t(nsub) * max_tiles_) + carve_size<int>(nsub) +
                           carve_size<const uint8_t*>(kExperts) + carve_size<int>(std::size_t(nsub) * kExperts);
    auto host = backend_.allocate(Memory::host, hb);
    uint8_t* h = static_cast<uint8_t*>(host.data());
    pids_host_ = carve<int>(h, C + 1);
    hcounts_ = carve<int>(h, std::size_t(nsub) * kExperts);
    hoffsets_ = carve<int>(h, std::size_t(nsub) * kExperts);
    htiles_ = carve<int4>(h, std::size_t(nsub) * max_tiles_);
    hntiles_ = carve<int>(h, nsub);
    hgroup_ = carve<const uint8_t*>(h, kExperts);
    hsfrow_ = carve<int>(h, std::size_t(nsub) * kExperts);
    buffers_.push_back(std::move(host));
    if(options_.cpu_prompt_share > 0.f && options_.cpu_prompt_max > 0) {
        const std::size_t m = std::min<std::size_t>(options_.cpu_prompt_max, S);
        auto hb2 = backend_.allocate(Memory::host, carve_size<Route>(m) + carve_size<uint16_t>(m * kLatent) +
                                                       carve_size<int>(m * kTopK) +
                                                       carve_size<float>(m * kTopK * kLatent));
        uint8_t* q = static_cast<uint8_t*>(hb2.data());
        hroutes_ = carve<Route>(q, m);
        hlat_ = carve<uint16_t>(q, m * kLatent);
        hfixed_ = carve<int>(q, m * kTopK);
        hcpu_out_ = carve<float>(q, m * kTopK * kLatent);
        buffers_.push_back(std::move(hb2));
        auto db = backend_.allocate(Memory::device, carve_size<int>(m * kTopK));
        pfixed_ = static_cast<int*>(db.data());
        buffers_.push_back(std::move(db));
        options_.cpu_prompt_max = int(m);
    }
}

void Net::allocate_slots() {
    const std::size_t spare = backend_.spare_device_bytes(options_.vram_margin);
    std::size_t own = spare / kSlotBytes;
    if(options_.expert_slots) own = std::min(own, options_.expert_slots);
    own = std::min<std::size_t>(own, std::size_t(kGroups) * kExperts);
    std::vector<uint8_t*> addr;
    std::vector<int> first;
    if(!options_.expert_slots)
        for(const auto& [ptr, bytes] : workspace_lend_)
            for(std::size_t j = 0; (j + 1) * kSlotBytes <= bytes; ++j) {
                addr.push_back(ptr + j * kSlotBytes);
                first.push_back(-2);
            }
    const std::size_t lent_workspace = addr.size();
    if(options_.lend_kv && !options_.expert_slots) {
        // The tail of every KV code region (layer, K/V): positions a request can't reach hold slots.
        const std::size_t region = std::size_t(options_.capacity) * kKvDim;
        for(std::size_t a = 0; a < kv_.size(); ++a)
            for(uint8_t* base : {kv_[a].k, kv_[a].v})
                for(std::size_t j = 1; j * kSlotBytes <= region; ++j) {
                    const std::size_t off = region - j * kSlotBytes;
                    addr.push_back(base + off);
                    first.push_back(int(off / kKvDim));
                }
    }
    const std::size_t n = own + addr.size();
    own_slots_ = int(own);
    tables_.nslots = int(n);
    std::fprintf(stderr, "nmh: %zu own expert slots (%.1f GiB), %zu lent by the prompt workspace, %zu from KV\n", own,
                 double(own * kSlotBytes) / double(1ull << 30), lent_workspace, addr.size() - lent_workspace);
    if(n == 0) return;
    const std::size_t meta = carve_size<int>(std::size_t(kGroups) * kExperts) + carve_size<int>(n) +
                             carve_size<uint8_t>(n) * 2 + carve_size<uint32_t>(n) + carve_size<uint8_t*>(n) +
                             carve_size<int>(4);
    auto buf = backend_.allocate(Memory::device, own * kSlotBytes + meta);
    uint8_t* p = static_cast<uint8_t*>(buf.data());
    uint8_t* own_base = p;
    p += own * kSlotBytes;
    tables_.slot_of = carve<int>(p, std::size_t(kGroups) * kExperts);
    tables_.owner = carve<int>(p, n);
    tables_.ref = carve<uint8_t>(p, n);
    uint8_t* valid = carve<uint8_t>(p, n);
    tables_.valid = valid;
    tables_.stamp = carve<uint32_t>(p, n);
    uint8_t** slot_addr = carve<uint8_t*>(p, n);
    tables_.slot_addr = slot_addr;
    int* misc = carve<int>(p, 4);
    tables_.hand = misc;
    tables_.step = reinterpret_cast<uint32_t*>(misc + 1);
    tables_.pass_first = reinterpret_cast<uint32_t*>(misc + 2);
    mslot_addr_.resize(n);
    slot_first_pos_.assign(n, -1);
    for(std::size_t s = 0; s < own; ++s) mslot_addr_[s] = own_base + s * kSlotBytes;
    for(std::size_t i = 0; i < addr.size(); ++i) {
        mslot_addr_[own + i] = addr[i];
        slot_first_pos_[own + i] = first[i];
    }
    mslot_of_.assign(std::size_t(kGroups) * kExperts, -1);
    mowner_.assign(n, -1);
    mref_.assign(n, 0);
    mvalid_.assign(n, 1);
    mstamp_.assign(n, 0);
    buf.upload(std::as_bytes(std::span(mslot_addr_)), reinterpret_cast<uint8_t*>(slot_addr) - static_cast<uint8_t*>(buf.data()));
    NMH_CUDA(cudaMemset(misc, 0, 16));
    buffers_.push_back(std::move(buf));
    workspace_lent_ = true;
    directory_upload();
}

// The prompt workspace's slots: usable between prompts, evicted before a prompt chunk writes the workspace.
// Works on the host mirror (between directory_download and directory_upload).
void Net::lend_workspace(bool lend) {
    if(lend == workspace_lent_) return;
    for(int s = own_slots_; s < tables_.nslots; ++s) {
        if(slot_first_pos_[s] != -2) continue;
        if(!lend) {
            if(mowner_[s] >= 0) mslot_of_[mowner_[s]] = -1;
            mowner_[s] = -1;
            mref_[s] = 0;
        }
        mvalid_[s] = lend ? 1 : 0;
    }
    workspace_lent_ = lend;
}

void Net::directory_download() {
    if(!tables_.nslots) return;
    const auto s = stream();
    NMH_CUDA(cudaStreamSynchronize(s));
    const std::size_t n = std::size_t(tables_.nslots);
    std::vector<uint32_t> stamp(n);
    NMH_CUDA(cudaMemcpy(mslot_of_.data(), tables_.slot_of, mslot_of_.size() * 4, cudaMemcpyDeviceToHost));
    NMH_CUDA(cudaMemcpy(mowner_.data(), tables_.owner, n * 4, cudaMemcpyDeviceToHost));
    NMH_CUDA(cudaMemcpy(mref_.data(), tables_.ref, n, cudaMemcpyDeviceToHost));
    NMH_CUDA(cudaMemcpy(mvalid_.data(), tables_.valid, n, cudaMemcpyDeviceToHost));
    NMH_CUDA(cudaMemcpy(&mhand_, tables_.hand, 4, cudaMemcpyDeviceToHost));
    std::fill(mstamp_.begin(), mstamp_.end(), 0);
    mepoch_ = 0;
}

void Net::directory_upload() {
    if(!tables_.nslots) return;
    const auto s = stream();
    NMH_CUDA(cudaStreamSynchronize(backend_.stream(Queue::transfer)));
    NMH_CUDA(cudaStreamSynchronize(s));
    const std::size_t n = std::size_t(tables_.nslots);
    NMH_CUDA(cudaMemcpy(tables_.slot_of, mslot_of_.data(), mslot_of_.size() * 4, cudaMemcpyHostToDevice));
    NMH_CUDA(cudaMemcpy(tables_.owner, mowner_.data(), n * 4, cudaMemcpyHostToDevice));
    NMH_CUDA(cudaMemcpy(tables_.ref, mref_.data(), n, cudaMemcpyHostToDevice));
    NMH_CUDA(cudaMemcpy(const_cast<uint8_t*>(tables_.valid), mvalid_.data(), n, cudaMemcpyHostToDevice));
    NMH_CUDA(cudaMemcpy(tables_.hand, &mhand_, 4, cudaMemcpyHostToDevice));
    // Decode-pass stamps restart: every slot is unprotected at the next pass.
    NMH_CUDA(cudaMemset(tables_.stamp, 0, n * 4));
    NMH_CUDA(cudaMemset(tables_.step, 0, 8));
}

// CLOCK victim on the host mirror; slots stamped with protect_epoch hold experts of the current layer.
int Net::host_victim(int protect_epoch) {
    const int n = tables_.nslots;
    for(int scanned = 0; scanned < 2 * n + 1; ++scanned) {
        const int s = mhand_;
        mhand_ = mhand_ + 1 == n ? 0 : mhand_ + 1;
        if(!mvalid_[s] || mstamp_[s] == protect_epoch) continue;
        if(mref_[s]) { mref_[s] = 0; continue; }
        return s;
    }
    return -1;
}

void Net::reset() {
    const auto s = stream();
    NMH_CUDA(cudaMemsetAsync(conv_state_, 0, std::size_t(kMambaLayers) * kConvDim * 3 * 4, s));
    NMH_CUDA(cudaMemsetAsync(ssm_, 0, std::size_t(kMambaLayers) * ring_ * kSsmFloats * 4, s));
    NMH_CUDA(cudaMemsetAsync(ssm_cur_, 0, 4, s));
    verify_ms_ = draft_ms_ = 0;
    marks_pending_ = false;
}

void Net::begin_request(int positions) {
    if(!tables_.nslots || own_slots_ == tables_.nslots) return;
    directory_download();
    bool changed = false;
    for(int s = own_slots_; s < tables_.nslots; ++s) {
        if(slot_first_pos_[s] < 0) continue;   // workspace slots follow the prompt passes
        const bool usable = slot_first_pos_[s] >= positions;
        if(!usable && mvalid_[s]) {
            if(mowner_[s] >= 0) mslot_of_[mowner_[s]] = -1;
            mowner_[s] = -1;
            mref_[s] = 0;
            mvalid_[s] = 0;
            changed = true;
        } else if(usable && !mvalid_[s]) {
            mvalid_[s] = 1;
            changed = true;
        }
    }
    if(changed) directory_upload();
}

unsigned long long Net::expert_lookups() const {
    unsigned long long v = 0;
    cudaMemcpy(&v, stats_, 8, cudaMemcpyDeviceToHost);
    return v;
}
unsigned long long Net::expert_cpu() const {
    unsigned long long v = 0;
    cudaMemcpy(&v, stats_ + 2, 8, cudaMemcpyDeviceToHost);
    return v;
}
unsigned long long Net::expert_misses() const {
    unsigned long long v = 0;
    cudaMemcpy(&v, stats_ + 1, 8, cudaMemcpyDeviceToHost);
    return v;
}

// ------------------------------------------------------------------------------------------------ prompt
void Net::moe_prefill(const MoeWeights& w, const __half* x, float* h, int T) {
    const auto s = stream();
    const int S = options_.moe_sub;
    const int nsub = (T + S - 1) / S;
    GemmArgs a;
    a.A = x;
    a.lda = kHidden;
    a.M = T;
    a.K = kHidden;
    // router logits, latent input
    a.N = kExperts;
    a.W = w.router;
    a.C = pRL_;
    a.ldc = kExperts;
    gemm(WFmt::bf16, Epi::f32, a, 0, s);
    route(pRL_, w.bias, proutes_, T, s);
    a.N = kLatent;
    a.W = w.fc1;
    a.C = nullptr;
    a.Cb = pLAT_;
    a.ldc = kLatent;
    gemm(WFmt::bf16, Epi::f16, a, 0, s);
    // shared expert
    if(options_.prompt_fp4) {
        shared_fp4(w, x, h, T, s);
    } else {
    a.N = kShared;
    a.W = w.sh_up.w;
    a.Ws = w.sh_up.s;
    a.alpha = w.sh_up.scale;
    a.out_scale = kActScale;
    a.Cb = pSH_;
    a.ldc = kShared;
    gemm(WFmt::fp4, Epi::relu2, a, 0, s);
    GemmArgs d;
    d.A = pSH_;
    d.lda = kShared;
    d.M = T;
    d.K = kShared;
    d.N = kHidden;
    d.W = w.sh_down.w;
    d.Ws = w.sh_down.s;
    d.alpha = w.sh_down.scale / kActScale;
    d.C = h;
    d.ldc = kHidden;
    gemm(WFmt::fp4, Epi::f32_add, d, 0, s);
    }
    // routes per expert and sub-chunk
    NMH_CUDA(cudaMemsetAsync(pcounts_, 0, std::size_t(nsub) * kExperts * 4, s));
    count_routes(proutes_, T, S, pcounts_, s);
    NMH_CUDA(cudaMemcpyAsync(hcounts_, pcounts_, std::size_t(nsub) * kExperts * 4, cudaMemcpyDeviceToHost, s));
    // Short prompt chunks: a share of the experts, chosen by expert id only (so a token's result never depends on
    // the cache state), runs on the CPU concurrently with the device's experts and skips the host-to-device copy.
    const bool cpu = hcpu_out_ && nsub == 1 && T <= options_.cpu_prompt_max;
    if(cpu) {
        NMH_CUDA(cudaMemcpyAsync(hroutes_, proutes_, std::size_t(T) * sizeof(Route), cudaMemcpyDeviceToHost, s));
        NMH_CUDA(cudaMemcpyAsync(hlat_, pLAT_, std::size_t(T) * kLatent * 2, cudaMemcpyDeviceToHost, s));
    }
    NMH_CUDA(cudaStreamSynchronize(s));
    std::thread cpu_thread;
    int cpu_rows = 0, gpu_rows = 0;
    if(cpu) {
        const uint32_t limit = uint32_t(options_.cpu_prompt_share * 1000.f);
        std::vector<int> first(kExperts, -1), rank(kExperts, 0);
        cpu_list_.clear();
        for(int e = 0; e < kExperts; ++e) {
            const int c = hcounts_[e];
            if(!c) continue;
            if((uint32_t(e) * 2654435761u ^ uint32_t(w.group) * 40503u) % 1000u < limit) {
                first[e] = cpu_rows;
                cpu_list_.push_back({e, cpu_rows, c});
                cpu_rows += c;
                hcounts_[e] = 0;
            } else {
                gpu_rows += c;
            }
        }
        cpu_row_token_.assign(std::size_t(cpu_rows), 0);
        for(int t = 0; t < T; ++t)
            for(int k = 0; k < kTopK; ++k) {
                const int e = hroutes_[t].expert[k];
                int& f = hfixed_[t * kTopK + k];
                if(first[e] < 0) {
                    f = -1;
                } else {
                    const int row = first[e] + rank[e]++;
                    cpu_row_token_[std::size_t(row)] = t;
                    f = gpu_rows + row;
                }
            }
        NMH_CUDA(cudaMemcpyAsync(pfixed_, hfixed_, std::size_t(T) * kTopK * 4, cudaMemcpyHostToDevice, s));
        if(!cpu_list_.empty())
            cpu_thread = std::thread([this, g = w.group] {
                cpu_prompt_experts(host_experts_, g, s2_host_.data(), cpu_list_, cpu_row_token_.data(), hlat_,
                                   hcpu_out_, options_.cpu_threads);
            });
    }
    // Residency: every expert this chunk routes to gets a slot (cached, prefetched, or copied from host into a CLOCK
    // victim now).
    const int group = w.group;
    if(prefetch_group_ == group) {
        mepoch_ = prefetch_epoch_;   // the prefetched slots stay protected with this layer's
        NMH_CUDA(cudaStreamWaitEvent(s, prefetch_ready_, 0));
        prefetch_group_ = -1;
    } else {
        ++mepoch_;
    }
    for(int e = 0; e < kExperts; ++e) {
        int total = 0;
        for(int k = 0; k < nsub; ++k) total += hcounts_[k * kExperts + e];
        if(!total) {
            hgroup_[e] = nullptr;
            continue;
        }
        const int key = group * kExperts + e;
        int slot = tables_.nslots ? mslot_of_[key] : -1;
        const uint8_t* src = host_experts_ + std::size_t(key) * kSlotBytes;
        if(slot < 0 && tables_.nslots) {
            slot = host_victim(mepoch_);
            if(slot >= 0) {
                if(mowner_[slot] >= 0) mslot_of_[mowner_[slot]] = -1;
                mowner_[slot] = key;
                mslot_of_[key] = slot;
                NMH_CUDA(cudaMemcpyAsync(mslot_addr_[slot], src, kSlotBytes, cudaMemcpyHostToDevice, s));
                ++prefill_fetched_;
            }
        }
        if(slot >= 0) {
            mref_[slot] = 1;
            mstamp_[slot] = mepoch_;
            hgroup_[e] = mslot_addr_[slot];
        } else {
            // No slot to spare: the grouped GEMM reads the expert from mapped host memory.
            hgroup_[e] = tables_.host + std::size_t(key) * kSlotBytes;
        }
    }
    // The W4A4 path needs every routed expert in device memory (TMA loads); otherwise this layer uses FP16.
    bool fp4 = options_.prompt_fp4;
    for(int e = 0; e < kExperts && fp4; ++e)
        if(hgroup_[e] && !(tables_.nslots && mslot_of_[group * kExperts + e] >= 0)) fp4 = false;
    // Offsets, tiles and (W4A4) padded scale rows per sub-chunk.
    for(int k = 0; k < nsub; ++k) {
        int off = 0, nt = 0, sfr = 0;
        int* offs = hoffsets_ + k * kExperts;
        int* sfrow = hsfrow_ + k * kExperts;
        int4* tiles = htiles_ + std::size_t(k) * max_tiles_;
        for(int e = 0; e < kExperts; ++e) {
            const int c = hcounts_[k * kExperts + e];
            offs[e] = off;
            sfrow[e] = sfr;
            for(int r = 0; r < c; r += 128) tiles[nt++] = make_int4(e, off + r, off + std::min(c, r + 128), 0);
            off += c;
            sfr += (c + 127) / 128 * 128;
        }
        hntiles_[k] = nt;
    }
    NMH_CUDA(cudaMemcpyAsync(pgroup_, hgroup_, kExperts * sizeof(void*), cudaMemcpyHostToDevice, s));
    NMH_CUDA(cudaMemcpyAsync(poffsets_, hoffsets_, std::size_t(nsub) * kExperts * 4, cudaMemcpyHostToDevice, s));
    NMH_CUDA(cudaMemcpyAsync(ptiles_, htiles_, std::size_t(nsub) * max_tiles_ * sizeof(int4), cudaMemcpyHostToDevice, s));
    NMH_CUDA(cudaMemcpyAsync(pntiles_, hntiles_, std::size_t(nsub) * 4, cudaMemcpyHostToDevice, s));
    if(fp4) {
        NMH_CUDA(cudaMemcpyAsync(psfrow_, hsfrow_, std::size_t(nsub) * kExperts * 4, cudaMemcpyHostToDevice, s));
        swizzle_expert_scales(pgroup_, pWsf_, s);
    }
    for(int k = 0; k < nsub; ++k) {
        const int t0 = k * S, t1 = std::min(T, t0 + S);
        NMH_CUDA(cudaMemsetAsync(pcursor_, 0, kExperts * 4, s));
        scatter_routes(proutes_, t0, t1, poffsets_ + k * kExperts, pcursor_, ptoken_, ppos_, s, cpu ? pfixed_ : nullptr);
        if(fp4) {
            moe_experts_fp4(group, k, t0, s);
        } else {
        GemmArgs u;
        u.A = pLAT_ + std::size_t(t0) * kLatent;
        u.lda = kLatent;
        u.a_rows = ptoken_;
        u.N = kInter;
        u.K = kLatent;
        u.out_scale = kActScale;
        u.Cb = pACT_;
        u.ldc = kInter;
        u.tiles = ptiles_ + std::size_t(k) * max_tiles_;
        u.ntiles = pntiles_ + k;
        u.group_base = pgroup_;
        u.w_off = 0;
        u.s_off = kOffUpS;
        u.group_alpha = palpha_ + std::size_t(group) * kExperts * 2;
        u.alpha_idx = 0;
        gemm(WFmt::fp4, Epi::relu2, u, hntiles_[k], s);
        GemmArgs dn = u;
        dn.A = pACT_;
        dn.lda = kInter;
        dn.a_rows = nullptr;
        dn.N = kLatent;
        dn.K = kInter;
        dn.Cb = nullptr;
        dn.C = pROUT_;
        dn.ldc = kLatent;
        dn.w_off = kOffDown;
        dn.s_off = kOffDownS;
        dn.alpha_idx = 1;
        gemm(WFmt::fp4, Epi::f32, dn, hntiles_[k], s);
        }
        if(cpu_thread.joinable()) {
            cpu_thread.join();
            NMH_CUDA(cudaMemcpyAsync(pROUT_ + std::size_t(gpu_rows) * kLatent, hcpu_out_,
                                     std::size_t(cpu_rows) * kLatent * 4, cudaMemcpyHostToDevice, s));
        }
        moe_combine(proutes_, ppos_, pROUT_, pLO_, t0, t1, s);
        if(k + 1 == nsub) {
            // The layer's expert GEMMs are enqueued: the next MoE layer's copies may overwrite its slots after them.
            NMH_CUDA(cudaEventRecord(moe_done_, s));
            if(T >= kPrefetchTokens && group + 1 < kMoeLayers) prefetch_group(group + 1);
        }
        GemmArgs f;
        f.A = pLO_;
        f.lda = kLatent;
        f.M = t1 - t0;
        f.K = kLatent;
        f.N = kHidden;
        f.W = w.fc2;
        f.C = h + std::size_t(t0) * kHidden;
        f.ldc = kHidden;
        gemm(WFmt::bf16, Epi::f32_add, f, 0, s);
    }
}

// Shared expert as NVFP4 W4A4: x quantized with its up input_scale, up GEMM into pSH_ (BF16), relu2 quantized with
// the down input_scale, down GEMM in row pieces through pROUT_ (FP32) added into h.
void Net::shared_fp4(const MoeWeights& w, const __half* x, float* h, int T, cudaStream_t s) {
    quant_rows_fp4(x, T, kHidden, 1.f / w.sh_up.in_scale, pShq_, pShsf_, s);
    auto* up = reinterpret_cast<__nv_bfloat16*>(pSH_);
    fp4_groups_.assign(1, Fp4Group{});
    Fp4Group& g = fp4_groups_[0];
    g.M = T;
    g.A = pShq_;
    g.SFA = pShsf_;
    g.B = w.sh_up.w;
    g.SFB = w.sh_up.sf;
    g.alpha = w.sh_up.alpha;
    g.D = up;
    fp4g_->run(fp4_groups_, kShared, kHidden, true, s);
    relu2_quant_rows_fp4(up, T, kShared, 1.f / w.sh_down.in_scale, pShaq_, pShasf_, s);
    const int piece = int(rout_bytes_ / (std::size_t(kHidden) * 4) / 128 * 128);
    for(int r0 = 0; r0 < T; r0 += piece) {
        const int n = std::min(piece, T - r0);
        g.M = n;
        g.A = pShaq_ + std::size_t(r0) * (kShared / 2);
        g.SFA = pShasf_ + std::size_t(r0) * (kShared / 16);
        g.B = w.sh_down.w;
        g.SFB = w.sh_down.sf;
        g.alpha = w.sh_down.alpha;
        g.D = pROUT_;
        fp4g_->run(fp4_groups_, kHidden, kShared, false, s);
        add_f32(h + std::size_t(r0) * kHidden, pROUT_, std::size_t(n) * kHidden, s);
    }
}

// Routed experts of sub-chunk k as NVFP4 W4A4: gathered latent rows quantized with the layer's up input_scale,
// grouped up GEMM (BF16 out), relu2 quantized with the down input_scale, grouped down GEMM into pROUT_ (FP32).
void Net::moe_experts_fp4(int group, int k, int t0, cudaStream_t s) {
    const int* offs = hoffsets_ + k * kExperts;
    const int* sfrow = hsfrow_ + k * kExperts;
    const int* counts = hcounts_ + k * kExperts;
    const int4* tiles = ptiles_ + std::size_t(k) * max_tiles_;
    const int nt = hntiles_[k];
    const float gs_up = 1.f / in_scale_[std::size_t(group) * 2], gs_down = 1.f / in_scale_[std::size_t(group) * 2 + 1];
    quant_gather_fp4(pLAT_ + std::size_t(t0) * kLatent, ptoken_, tiles, nt, poffsets_ + k * kExperts,
                     psfrow_ + k * kExperts, gs_up, pXq_, pXsf_, s);
    auto* up = reinterpret_cast<__nv_bfloat16*>(pACT_);
    const float* alpha = palpha4_ + std::size_t(group) * kExperts * 2;
    fp4_groups_.clear();
    for(int e = 0; e < kExperts; ++e) {
        if(!counts[e]) continue;
        Fp4Group g;
        g.M = counts[e];
        g.A = pXq_ + std::size_t(offs[e]) * (kLatent / 2);
        g.SFA = pXsf_ + std::size_t(sfrow[e]) * kUpK16;
        g.B = hgroup_[e];
        g.SFB = pWsf_ + std::size_t(e) * kSfExpertBytes;
        g.alpha = alpha + std::size_t(e) * 2;
        g.D = up + std::size_t(offs[e]) * kInter;
        fp4_groups_.push_back(g);
    }
    fp4g_->run(fp4_groups_, kInter, kLatent, true, s);
    relu2_quant_fp4(up, tiles, nt, poffsets_ + k * kExperts, psfrow_ + k * kExperts, gs_down, pAq_, pAsf_, s);
    std::size_t i = 0;
    for(int e = 0; e < kExperts; ++e) {
        if(!counts[e]) continue;
        Fp4Group& g = fp4_groups_[i++];
        g.A = pAq_ + std::size_t(offs[e]) * (kInter / 2);
        g.SFA = pAsf_ + std::size_t(sfrow[e]) * kDownK16;
        g.B = hgroup_[e] + kOffDown;
        g.SFB = pWsf_ + std::size_t(e) * kSfExpertBytes + kSfUpBytes;
        g.alpha = alpha + std::size_t(e) * 2 + 1;
        g.D = pROUT_ + std::size_t(offs[e]) * kLatent;
    }
    fp4g_->run(fp4_groups_, kLatent, kInter, false, s);
}

void Net::dense_fp8(const Fp8W& w, int N, int K, const float* x32, int ldx, int T, float* out, int ldo, bool add,
                    float* tmp) {
    const auto s = stream();
    quant_rows_fp8(x32, ldx, K, T, pq_, prs_, s);
    float* d = add ? tmp : out;
    const int ldd = add ? N : ldo;
    lt_->fp8(w.w, N, K, pq_, T, d, ldd, s);
    scale_rows(d, ldd, prs_, w.scale, out, ldo, N, T, add, s);
}

// Copies every expert of `group` that is not resident into CLOCK victims on the transfer stream, after the compute
// stream's last recorded MoE layer (moe_done_). The host mirror marks them resident at once; the layer's GEMMs wait
// for prefetch_ready_.
void Net::prefetch_group(int group) {
    if(!tables_.nslots) return;
    const auto tr = backend_.stream(Queue::transfer);
    NMH_CUDA(cudaStreamWaitEvent(tr, moe_done_, 0));
    const int epoch = ++mepoch_;
    for(int e = 0; e < kExperts; ++e) {
        const int key = group * kExperts + e;
        int slot = mslot_of_[key];
        if(slot < 0) {
            slot = host_victim(epoch);
            if(slot < 0) continue;
            if(mowner_[slot] >= 0) mslot_of_[mowner_[slot]] = -1;
            mowner_[slot] = key;
            mslot_of_[key] = slot;
            NMH_CUDA(cudaMemcpyAsync(mslot_addr_[slot], host_experts_ + std::size_t(key) * kSlotBytes, kSlotBytes,
                                     cudaMemcpyHostToDevice, tr));
            ++prefill_fetched_;
        }
        mref_[slot] = 1;
        mstamp_[slot] = epoch;
    }
    NMH_CUDA(cudaEventRecord(prefetch_ready_, tr));
    prefetch_group_ = group;
    prefetch_epoch_ = epoch;
}

void Net::prefill(const std::vector<int>& history, int pos0, int count, bool keep_all, bool drafter) {
    if(count < 1 || count > options_.prefill_chunk) throw std::invalid_argument("prefill chunk size");
    if(pos0 + count > options_.capacity) throw std::invalid_argument("prefill past the context");
    const auto s = stream();
    const int T = count;
    NMH_CUDA(cudaStreamSynchronize(s));
    // The chunk's ids and, when the history has it, the token after the chunk (the drafter's last row).
    const int known = std::min<int>(T + 1, int(history.size()) - pos0);
    std::memcpy(pids_host_, history.data() + pos0, std::size_t(known) * 4);
    NMH_CUDA(cudaMemcpyAsync(pids_, pids_host_, std::size_t(known) * 4, cudaMemcpyHostToDevice, s));
    directory_download();
    lend_workspace(false);   // the chunk writes the workspace
    prefetch_group_ = -1;
    if(T >= kPrefetchTokens) {
        NMH_CUDA(cudaEventRecord(moe_done_, s));
        prefetch_group(0);
    }
    embed_rows(embed_, pids_, pH_, T, s);
    for(int l = 0; l < kLayers; ++l) {
        const LayerWeights& L = layers_[l];
        const bool fp8 = options_.prefill_fp8 && L.kind != Kind::moe;
        if(fp8) rmsnorm(pH_, L.norm, pXC_, T, s);
        else rmsnorm_h(pH_, L.norm, pX_, T, s);
        if(L.kind == Kind::mamba && fp8) {
            dense_fp8(L.in_proj, kInProj, kHidden, pXC_, kHidden, T, pP_, kInProj, false, nullptr);
            float* conv = conv_state_ + std::size_t(L.index) * kConvDim * 3;
            float* ring = ssm_ + std::size_t(L.index) * ring_ * kSsmFloats;
            conv_tokens(L.m, pP_, kInProj, conv, pXC_, nullptr, T, true, s);
            ssm_tokens(L.m, pXC_, pP_, kInProj, ring, ssm_cur_, false, pY_, T, s);
            gated_norm(L.m, pY_, pP_, kInProj, pXC_, nullptr, T, s);
            dense_fp8(L.out_proj, kHidden, kMInner, pXC_, kMInner, T, pH_, kHidden, true, pY_);
        } else if(L.kind == Kind::attn && fp8) {
            const Fp8W* w3[3] = {&L.q, &L.k, &L.v};
            const int rows[3] = {kQDim, kKvDim, kKvDim}, cols0[3] = {0, kQDim, kQDim + kKvDim};
            for(int i = 0; i < 3; ++i) dense_fp8(*w3[i], rows[i], kHidden, pXC_, kHidden, T, pP_ + cols0[i], kQkv, false, nullptr);
            to_bf16(pP_, pQKV_, std::size_t(T) * kQkv, s);
            const auto& c = kvc_[L.index];
            NMH_CUDA(crit::fp8_kv::write(c, pQKV_ + kQDim, pQKV_ + kQDim + kKvDim, kQkv, T, pos0, s));
            crit::fp8_kv::Prefill pf;
            pf.q = pQKV_;
            pf.q_stride = kQkv;
            pf.cache = c;
            pf.out = pY_;
            pf.out_stride = kQDim;
            pf.T = T;
            pf.pos0 = pos0;
            pf.heads = kQHeads;
            pf.scale = 0.08838834764831845f;
            NMH_CUDA(options_.prompt_attn_fp8 ? crit::fp8_kv::prefill_fp8(pf, s) : crit::fp8_kv::prefill(pf, s));
            dense_fp8(L.o, kHidden, kQDim, pY_, kQDim, T, pH_, kHidden, true, pXC_);
        } else if(L.kind == Kind::mamba) {
            GemmArgs a;
            a.A = pX_;
            a.lda = kHidden;
            a.M = T;
            a.K = kHidden;
            a.N = kInProj;
            a.W = L.in_proj.w;
            a.alpha = L.in_proj.scale;
            a.C = pP_;
            a.ldc = kInProj;
            gemm(WFmt::e4m3, Epi::f32, a, 0, s);
            float* conv = conv_state_ + std::size_t(L.index) * kConvDim * 3;
            float* ring = ssm_ + std::size_t(L.index) * ring_ * kSsmFloats;
            conv_tokens(L.m, pP_, kInProj, conv, pXC_, nullptr, T, true, s);
            ssm_tokens(L.m, pXC_, pP_, kInProj, ring, ssm_cur_, false, pY_, T, s);
            gated_norm(L.m, pY_, pP_, kInProj, nullptr, pX_, T, s);
            GemmArgs o;
            o.A = pX_;
            o.lda = kMInner;
            o.M = T;
            o.K = kMInner;
            o.N = kHidden;
            o.W = L.out_proj.w;
            o.alpha = L.out_proj.scale;
            o.C = pH_;
            o.ldc = kHidden;
            gemm(WFmt::e4m3, Epi::f32_add, o, 0, s);
        } else if(L.kind == Kind::attn) {
            GemmArgs a;
            a.A = pX_;
            a.lda = kHidden;
            a.M = T;
            a.K = kHidden;
            a.C = pP_;
            a.ldc = kQkv;
            const Fp8W* w3[3] = {&L.q, &L.k, &L.v};
            const int rows[3] = {kQDim, kKvDim, kKvDim}, cols0[3] = {0, kQDim, kQDim + kKvDim};
            for(int i = 0; i < 3; ++i) {
                a.N = rows[i];
                a.W = w3[i]->w;
                a.alpha = w3[i]->scale;
                a.C = pP_ + cols0[i];
                gemm(WFmt::e4m3, Epi::f32, a, 0, s);
            }
            to_bf16(pP_, pQKV_, std::size_t(T) * kQkv, s);
            const auto& c = kvc_[L.index];
            NMH_CUDA(crit::fp8_kv::write(c, pQKV_ + kQDim, pQKV_ + kQDim + kKvDim, kQkv, T, pos0, s));
            crit::fp8_kv::Prefill pf;
            pf.q = pQKV_;
            pf.q_stride = kQkv;
            pf.cache = c;
            pf.out = pY_;
            pf.out_stride = kQDim;
            pf.T = T;
            pf.pos0 = pos0;
            pf.heads = kQHeads;
            pf.scale = 0.08838834764831845f;
            NMH_CUDA(options_.prompt_attn_fp8 ? crit::fp8_kv::prefill_fp8(pf, s) : crit::fp8_kv::prefill(pf, s));
            to_f16(pY_, pX_, std::size_t(T) * kQDim, s);
            GemmArgs o;
            o.A = pX_;
            o.lda = kQDim;
            o.M = T;
            o.K = kQDim;
            o.N = kHidden;
            o.W = L.o.w;
            o.alpha = L.o.scale;
            o.C = pH_;
            o.ldc = kHidden;
            gemm(WFmt::e4m3, Epi::f32_add, o, 0, s);
        } else {
            moe_prefill(L.moe, pX_, pH_, T);
        }
        if(const char* dump = std::getenv("NMH_DUMP")) {
            // Debug: the residual stream after each layer, raw FP32 [T][kHidden].
            std::vector<float> h(std::size_t(T) * kHidden);
            NMH_CUDA(cudaMemcpyAsync(h.data(), pH_, h.size() * 4, cudaMemcpyDeviceToHost, s));
            NMH_CUDA(cudaStreamSynchronize(s));
            char name[512];
            std::snprintf(name, sizeof(name), "%s/layer%02d.bin", dump, l);
            if(FILE* f = std::fopen(name, "wb")) {
                std::fwrite(h.data(), 4, h.size(), f);
                std::fclose(f);
            }
        }
    }
    if(drafter && mtp_loaded_) mtp_prefill(history, pos0, known - 1);
    directory_upload();
    last_count_ = T;
    if(keep_all) rmsnorm_h(pH_, norm_f_, pX_, T, s);
    // The last position's logits through the decode head (FP32 activations).
    rmsnorm(pH_ + std::size_t(T - 1) * kHidden, norm_f_, X_, 1, s);
    Gemv h;
    h.fmt = WFormat::fp8;
    h.seg[0] = {lm_head_.w, nullptr, lm_head_.scale, logits_, kVocab, kVocab};
    h.cols = kHidden;
    h.x = X_;
    gemv(h, s);
    row_scores(logits_, kVocab, kVocab, 1, nullptr, nullptr, 0, nullptr, nullptr, next_, s);
}

// The drafter's KV cache for chunk rows 0 .. rows - 1 (positions pos0 + r): row r fuses the embedding of the token
// at pos0 + r + 1 with the target's normed final hidden state at pos0 + r. Only the attention layer's keys and values
// are needed for later drafts, so the rest of the block does not run here.
void Net::mtp_prefill(const std::vector<int>& history, int pos0, int rows) {
    (void)history;
    if(rows < 1) return;
    const auto s = stream();
    // hidden: normed target output (FP32 in pY_), then hnorm -> second half of the FP16 fused input rows
    rmsnorm(pH_, norm_f_, pY_, rows, s);
    rmsnorm_h(pY_, mtp_.hnorm, pX_ + kHidden, rows, s, 2 * kHidden);
    embed_rows(embed_, pids_ + 1, pP_, rows, s);
    rmsnorm_h(pP_, mtp_.enorm, pX_, rows, s, 2 * kHidden);
    GemmArgs f;
    f.A = pX_;
    f.lda = 2 * kHidden;
    f.M = rows;
    f.K = 2 * kHidden;
    f.N = kHidden;
    f.W = mtp_.eh_proj;
    f.C = pXC_;
    f.ldc = kHidden;
    gemm(WFmt::bf16, Epi::f32, f, 0, s);
    rmsnorm_h(pXC_, mtp_.norm0, pX_, rows, s);
    GemmArgs a;
    a.A = pX_;
    a.lda = kHidden;
    a.M = rows;
    a.K = kHidden;
    a.ldc = kQkv;
    a.N = kKvDim;
    a.W = mtp_.k;
    a.C = pP_ + kQDim;
    gemm(WFmt::bf16, Epi::f32, a, 0, s);
    a.W = mtp_.v;
    a.C = pP_ + kQDim + kKvDim;
    gemm(WFmt::bf16, Epi::f32, a, 0, s);
    to_bf16(pP_, pQKV_, std::size_t(rows) * kQkv, s);
    NMH_CUDA(crit::fp8_kv::write(kvc_[kAttnLayers], pQKV_ + kQDim, pQKV_ + kQDim + kKvDim, kQkv, rows, pos0, s));
}

void Net::head_rows(int r0, int n) {
    if(n < 1 || n > kScoreRows || r0 + n > last_count_) throw std::invalid_argument("head_rows range");
    GemmArgs a;
    a.A = pX_ + std::size_t(r0) * kHidden;
    a.lda = kHidden;
    a.M = n;
    a.K = kHidden;
    a.N = kVocab;
    a.W = lm_head_.w;
    a.alpha = lm_head_.scale;
    a.C = pscore_;
    a.ldc = kVocab;
    gemm(WFmt::e4m3, Epi::f32, a, 0, stream());
}

}  // namespace nmh
