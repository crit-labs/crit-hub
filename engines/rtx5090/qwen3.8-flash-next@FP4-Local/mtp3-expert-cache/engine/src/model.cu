#include "model.h"
#include <crit/json.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <sys/mman.h>
#include <sys/prctl.h>

namespace fnl {

using crit::core::CudaBuffer;
using crit::core::Memory;
using crit::core::Queue;

namespace {
constexpr int kEosToken = 248044;
const std::string P = "model.language_model.";

bool is_attention(int layer) { return (layer + 1) % 4 == 0; }   // full_attention_interval = 4 (checked at load)
int gdn_index(int layer) { return layer - (layer + 1) / 4; }
int attn_index(int layer) { return layer / 4; }

std::vector<float> bf16_floats(const Tensors::Entry& e) {
    std::vector<float> out(e.bytes.size() / 2);
    const auto* p = reinterpret_cast<const uint16_t*>(e.bytes.data());
    for(std::size_t i = 0; i < out.size(); ++i) {
        uint32_t u = uint32_t(p[i]) << 16;
        std::memcpy(&out[i], &u, 4);
    }
    return out;
}
std::vector<float> any_floats(const Tensors::Entry& e) {
    if(e.dtype == "BF16") return bf16_floats(e);
    if(e.dtype != "F32") throw std::runtime_error("expected F32/BF16 tensor");
    std::vector<float> out(e.bytes.size() / 4);
    std::memcpy(out.data(), e.bytes.data(), e.bytes.size());
    return out;
}
std::vector<float> one_plus(std::vector<float> v) {
    for(float& x : v) x += 1.f;
    return v;
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
    template <class T> void add(const T** dst, std::vector<float> owned) {
        Item it{reinterpret_cast<const void**>(dst), {}, {}};
        it.owned.resize(owned.size() * 4);
        std::memcpy(it.owned.data(), owned.data(), it.owned.size());
        items_.push_back(std::move(it));
    }
    CudaBuffer finish(crit::core::CudaBackend& backend) {
        std::size_t total = 0;
        for(auto& it : items_) total += (size(it) + 255) / 256 * 256;
        CudaBuffer buffer = backend.allocate(Memory::device, total);
        std::size_t off = 0;
        auto* base = static_cast<uint8_t*>(buffer.data());
        for(auto& it : items_) {
            const auto bytes = it.owned.empty() ? it.src : std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(it.owned.data()), it.owned.size());
            buffer.upload(bytes, off);
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
}  // namespace

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
const Tensors::Entry& Tensors::at(const std::string& name) const {
    auto it = map_.find(name);
    if(it == map_.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
}

Net::Net(const Options& options, crit::core::CudaBackend& backend, crit::core::Resources& resources)
    : options_(options), backend_(backend), resources_(resources) {
    // A core dump of this process would hold tens of GB of pinned expert records (WSL writes it to Windows temp).
    ::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
    if(options_.capacity % kBlockTokens) throw std::invalid_argument("capacity must be a multiple of 4");
    // Check the dimensions this engine is built for.
    const auto config = crit::core::json::Document::read_file(options_.model / "config.json");
    const auto* text = config.root().find("text_config");
    auto num = [&](const char* key) { return text->find(key)->i64(); };
    if(num("hidden_size") != kHidden || num("num_hidden_layers") != kLayers || num("hc_count") != kStreams ||
       num("num_experts") != kExperts || num("num_experts_per_tok") != kTopK || num("vocab_size") != kVocab ||
       num("full_attention_interval") != 4 || num("indexer_budget") != kTopBlocks * kBlockTokens)
        throw std::runtime_error("config.json does not match the fnl engine's dimensions");
    const bool verbose = std::getenv("FNL_VERBOSE") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    auto mark = [&](const char* what) {
        if(verbose)
            std::fprintf(stderr, "fnl: %s at %.1fs\n", what,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    };
    tensors_ = std::make_unique<Tensors>(options_.model);
    const Tensors& t = *tensors_;
    mark("headers");
    load_dense(t);
    mark("dense weights");
    load_experts(t);
    mark("experts");
    if(options_.mtp && t.has("mtp.fc_embedding.weight")) {
        load_mtp(t);
        mark("mtp");
    }
    allocate_state();
    allocate_prefill();
    gather_pool_ = std::make_unique<CopyPool>(16);
    {
        // Off by default: which experts the CPU takes depends on the cache, so short prompts' outputs would not
        // repeat between runs (CPU FP32 vs device FP16 prompt arithmetic).
        int threads = 0;
        if(const char* v = std::getenv("FNL_PROMPT_CPU_THREADS")) threads = std::atoi(v);
        if(const char* v = std::getenv("FNL_PROMPT_CPU_SHARE")) prompt_cpu_share_ = float(std::atof(v));
        if(threads > 0 && prompt_cpu_share_ > 0.f) {
            const std::size_t bytes = std::size_t(kCpuPromptTokens) * kHidden * 4 + std::size_t(kCpuPromptTokens) * sizeof(Route) +
                                      std::size_t(kCpuPromptRows) * kHidden * 4 + std::size_t(kCpuPromptRows) * 4 + 4096;
            buffers_.push_back(backend_.allocate(Memory::host, bytes));
            auto* p = static_cast<uint8_t*>(buffers_.back().data());
            hx_ = reinterpret_cast<float*>(p);
            p += (std::size_t(kCpuPromptTokens) * kHidden * 4 + 255) / 256 * 256;
            hroutes_ = reinterpret_cast<Route*>(p);
            p += (std::size_t(kCpuPromptTokens) * sizeof(Route) + 255) / 256 * 256;
            hcpu_rows_ = reinterpret_cast<float*>(p);
            p += (std::size_t(kCpuPromptRows) * kHidden * 4 + 255) / 256 * 256;
            hcpu_row_ = reinterpret_cast<int*>(p);
            buffers_.push_back(backend_.allocate(Memory::device, std::size_t(kCpuPromptRows) * 4));
            pcpu_row_ = static_cast<int*>(buffers_.back().data());
            prompt_cpu_ = std::make_unique<CpuExperts>(threads);
        }
    }
    {
        int threads = options_.cpu_threads;
        float share = options_.cpu_share;
        if(const char* v = std::getenv("FNL_CPU_THREADS")) threads = std::atoi(v);
        if(const char* v = std::getenv("FNL_CPU_SHARE")) share = float(std::atof(v));
        if(threads > 0 && share > 0.f) {
            buffers_.push_back(backend_.allocate(Memory::host, sizeof(CpuJob) + sizeof(CpuReply) + 256));
            auto* p = static_cast<uint8_t*>(buffers_.back().data());
            std::memset(p, 0, sizeof(CpuJob) + sizeof(CpuReply) + 256);
            auto* job = reinterpret_cast<CpuJob*>(p);
            cpu_reply_ = reinterpret_cast<CpuReply*>(p + (sizeof(CpuJob) + 127) / 128 * 128);
            auto describe = [this](int layer, int e, CpuExpert& x) {
                x.record = host_layers_[layer] + std::size_t(e) * kHostExpertBytes;
                for(int m = 0; m < 3; ++m) {
                    const std::size_t i = (std::size_t(layer) * kExperts + e) * 3 + m;
                    x.exc[m] = exc_pool_h_ + exc_off_h_[i];
                    x.exc_count[m] = exc_cnt_h_[i];
                    x.s2[m] = s2_h_[i];
                }
            };
            FNL_CUDA(cudaStreamCreateWithFlags(&insert_stream_, cudaStreamNonBlocking));
            FNL_CUDA(cudaEventCreateWithFlags(&insert_done_, cudaEventDisableTiming));
            buffers_.push_back(backend_.allocate(Memory::host, kInsertJobs * kCpuMaxEntries * sizeof(ScaleJob)));
            insert_jobs_ = static_cast<ScaleJob*>(buffers_.back().data());
            auto insert = [this](const CpuJob& j) {
                // Weights by DMA, scales decoded on the device from the mapped CSF planes.
                ScaleJob* sj = insert_jobs_ + std::size_t(insert_next_) * kCpuMaxEntries;
                insert_next_ = (insert_next_ + 1) % kInsertJobs;
                int n = 0;
                for(int i = 0; i < j.count; ++i) {
                    if(!j.slot[i]) continue;
                    auto* slot = reinterpret_cast<uint8_t*>(j.slot[i]);
                    FNL_CUDA(cudaMemcpyAsync(slot, host_layers_[j.layer] + std::size_t(j.expert[i]) * kHostExpertBytes,
                                             kExpertWeightBytes, cudaMemcpyHostToDevice, insert_stream_));
                    sj[n++] = {j.expert[i], slot};
                }
                if(n) decode_scales(tables_, j.layer, sj, n, insert_stream_);
            };
            cpu_server_ = std::make_unique<CpuMissServer>(job, cpu_reply_, describe, insert, threads);
            tables_.job = job;
            tables_.cpu_share = share;
        }
    }
    allocate_slots();
    reset();
    mark("state");
}

Net::~Net() {
    if(step_done_) cudaEventDestroy(step_done_);
    for(auto e : pass_marks_)
        if(e) cudaEventDestroy(e);
    for(auto& kv : graphs_) {
        if(kv.second.head) cudaGraphExecDestroy(kv.second.head);
        if(kv.second.tail) cudaGraphExecDestroy(kv.second.tail);
        if(kv.second.draft) cudaGraphExecDestroy(kv.second.draft);
    }
}

cudaStream_t Net::stream() const { return backend_.stream(Queue::compute); }

void Net::load_dense(const Tensors& t) {
    Packer pk;
    auto mx8 = [&](Mx8& m, const std::string& base, int rows, int cols) {
        const auto& w = t.at(base + ".weight");
        const auto& s = t.at(base + ".weight_scale");
        expect_shape(w, {std::uint64_t(rows), std::uint64_t(cols)}, base);
        expect_shape(s, {std::uint64_t(rows), std::uint64_t(cols / 32)}, base);
        if(w.dtype != "F8_E4M3" || s.dtype != "U8") throw std::runtime_error("expected MXFP8 " + base);
        m.rows = rows;
        m.cols = cols;
        pk.add(&m.w, w.bytes);
        pk.add(&m.s, s.bytes);
    };
    auto bf16 = [&](const __nv_bfloat16** dst, const std::string& name, std::initializer_list<std::uint64_t> shape) {
        const auto& e = t.at(name);
        expect_shape(e, shape, name);
        if(e.dtype != "BF16") throw std::runtime_error("expected BF16 " + name);
        pk.add(dst, e.bytes);
    };
    auto hc = [&](Hc& h, const std::string& base, bool inject) {
        pk.add(&h.norm1p, one_plus(any_floats(t.at(base + ".hc_norm.weight"))));
        bf16(&h.down, base + ".input_mix_weight_down.weight", {kLowrank, kHyper});
        bf16(&h.up, base + ".input_mix_weight_up.weight", {kHyper, kLowrank});
        if(inject) bf16(&h.inject, base + ".block_inject_weight.weight", {kStreams, kHyper});
    };
    layers_.resize(kLayers);
    for(int l = 0; l < kLayers; ++l) {
        auto& L = layers_[l];
        const std::string lp = P + "layers." + std::to_string(l) + ".";
        L.attention = is_attention(l);
        hc(L.ahc, lp + "attn_hyper_connection", true);
        hc(L.mhc, lp + "mlp_hyper_connection", true);
        if(L.attention) {
            mx8(L.q, lp + "self_attn.q_proj", kQProj, kHidden);
            mx8(L.k, lp + "self_attn.k_proj", kKvHeads * kHeadDim, kHidden);
            mx8(L.v, lp + "self_attn.v_proj", kKvHeads * kHeadDim, kHidden);
            mx8(L.o, lp + "self_attn.o_proj", kHidden, kAttnOut);
            mx8(L.idx, lp + "self_attn.indexer.index_qk_proj", kIdxProj, kHidden);
            pk.add(&L.qn1p, one_plus(any_floats(t.at(lp + "self_attn.q_norm.weight"))));
            pk.add(&L.kn1p, one_plus(any_floats(t.at(lp + "self_attn.k_norm.weight"))));
            pk.add(&L.iqn1p, one_plus(any_floats(t.at(lp + "self_attn.indexer.q_layernorm.weight"))));
            pk.add(&L.ikn1p, one_plus(any_floats(t.at(lp + "self_attn.indexer.k_layernorm.weight"))));
        } else {
            mx8(L.qkv, lp + "linear_attn.in_proj_qkv", kQkvDim, kHidden);
            mx8(L.z, lp + "linear_attn.in_proj_z", kZDim, kHidden);
            mx8(L.a, lp + "linear_attn.in_proj_a", kVHeads, kHidden);
            mx8(L.b, lp + "linear_attn.in_proj_b", kVHeads, kHidden);
            mx8(L.out, lp + "linear_attn.out_proj", kHidden, kZDim);
            pk.add(&L.conv_w, any_floats(t.at(lp + "linear_attn.conv1d.weight")));
            auto alog = any_floats(t.at(lp + "linear_attn.A_log"));
            for(float& x : alog) x = -std::exp(x);
            pk.add(&L.neg_a, alog);
            pk.add(&L.dt_bias, any_floats(t.at(lp + "linear_attn.dt_bias")));
            pk.add(&L.gnorm, any_floats(t.at(lp + "linear_attn.norm.weight")));
        }
        bf16(&L.router, lp + "mlp.gate.weight", {kExperts, kHidden});
        bf16(&L.shared_gate, lp + "mlp.shared_expert_gate.weight", {1, kHidden});
        mx8(L.sh_gate, lp + "mlp.shared_expert.gate_proj", kShared, kHidden);
        mx8(L.sh_up, lp + "mlp.shared_expert.up_proj", kShared, kHidden);
        mx8(L.sh_down, lp + "mlp.shared_expert.down_proj", kHidden, kShared);
    }
    // PLE
    {
        const std::string pp = P + "layers." + std::to_string(kPleLayer) + ".ple.";
        bf16(&ple_.key, pp + "key_proj.weight", {kHyper, kHidden});
        bf16(&ple_.value, pp + "value_proj.weight", {kHidden, kHidden});
        pk.add(&ple_.nk1p, one_plus(any_floats(t.at(pp + "norm_key.weight"))));
        pk.add(&ple_.nq1p, one_plus(any_floats(t.at(pp + "norm_query.weight"))));
        pk.add(&ple_.nc1p, one_plus(any_floats(t.at(pp + "norm_conv.weight"))));
        pk.add(&ple_.conv_w, any_floats(t.at(pp + "conv1d.weight")));
        const std::string ep = pp + "ple_embedding.";
        ple_.s2 = any_floats(t.at(ep + "ngram_embedding.weight_scale_2"))[0];
        auto ints = [&](const std::string& n) {
            const auto& e = t.at(n);
            std::vector<int64_t> v(e.bytes.size() / 8);
            std::memcpy(v.data(), e.bytes.data(), e.bytes.size());
            return v;
        };
        ple_.sizes = ints(ep + "ngram_heads_vocab_sizes");
        ple_.offsets = ints(ep + "ngram_heads_offsets");
        ple_.mult = ints(ep + "layer_multipliers");
        if(ple_.sizes.size() != kPleHeads || ple_.mult.size() != 3) throw std::runtime_error("unexpected PLE tables");
        for(int s = 0;; ++s) {
            const std::string w = ep + "ngram_embedding.shard_" + std::to_string(s) + ".weight";
            if(!t.has(w)) break;
            const auto& we = t.at(w);
            const auto& se = t.at(ep + "ngram_embedding.shard_" + std::to_string(s) + ".weight_scale");
            if(we.shape.size() != 2 || we.shape[1] != 80 || se.shape[1] != 10) throw std::runtime_error("PLE shard shape");
            ple_.rows_per_shard = int(we.shape[0]);
            ple_.shard_w.push_back(reinterpret_cast<const uint8_t*>(we.bytes.data()));
            ple_.shard_s.push_back(reinterpret_cast<const uint8_t*>(se.bytes.data()));
            // Rows are read at random: no read-ahead around each touched page.
            for(const auto* e : {&we, &se}) {
                const auto a = reinterpret_cast<std::uintptr_t>(e->bytes.data()) & ~std::uintptr_t(4095);
                const auto b = reinterpret_cast<std::uintptr_t>(e->bytes.data() + e->bytes.size());
                ::madvise(reinterpret_cast<void*>(a), b - a, MADV_RANDOM);
            }
        }
        const int64_t rows = int64_t(ple_.rows_per_shard) * int64_t(ple_.shard_w.size());
        if(ple_.offsets.back() + ple_.sizes.back() > rows) throw std::runtime_error("PLE table too small");
    }
    hc(mixer_, P + "hyper_connection_mixer", false);
    bf16(&lm_head_, "lm_head.weight", {kVocab, kHidden});
    std::vector<float> inv(kRotary / 2);
    for(int i = 0; i < kRotary / 2; ++i) inv[i] = 1.0f / std::pow(10000000.0f, float(2 * i) / float(kRotary));
    pk.add(&inv_freq_, inv);
    buffers_.push_back(pk.finish(backend_));

    // Embedding table in pinned host memory, read by the device through its mapping.
    const auto& emb = t.at(P + "embed_tokens.weight");
    expect_shape(emb, {kVocab, kHidden}, "embed_tokens");
    auto host = backend_.allocate(Memory::host, emb.bytes.size());
    std::memcpy(host.data(), emb.bytes.data(), emb.bytes.size());
    embed_ = static_cast<const __nv_bfloat16*>(host.device_address());
    buffers_.push_back(std::move(host));
}

void Net::load_experts(const Tensors& t) {
    const std::size_t per_layer = std::size_t(kExperts) * kHostExpertBytes;
    // Exception offsets first (sizes come from the headers).
    std::vector<uint32_t> off(std::size_t(kLayers) * kExperts * 3), cnt(off.size());
    const char* mats[3] = {"gate_proj", "up_proj", "down_proj"};
    std::size_t words = 0;
    for(int l = 0; l < kLayers; ++l)
        for(int e = 0; e < kExperts; ++e)
            for(int m = 0; m < 3; ++m) {
                const auto& x = t.at(P + "layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." +
                                     mats[m] + ".weight_scale.nvfp4_csf_exceptions");
                const std::size_t i = (std::size_t(l) * kExperts + e) * 3 + m;
                off[i] = uint32_t(words);
                cnt[i] = uint32_t(x.bytes.size() / 4);
                words += cnt[i];
            }
    auto pool = backend_.allocate(Memory::host, std::max<std::size_t>(words, 1) * 4);
    std::vector<float> s2(off.size());
    host_layers_.resize(kLayers);
    for(int l = 0; l < kLayers; ++l) {
        buffers_.push_back(backend_.allocate(Memory::host, per_layer));
        host_layers_[l] = static_cast<uint8_t*>(buffers_.back().data());
    }
    std::vector<Tensors::Copy> copies;
    copies.reserve(std::size_t(kLayers) * kExperts * 12);
    for(int l = 0; l < kLayers; ++l)
        for(int e = 0; e < kExperts; ++e) {
            const std::string base = P + "layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
            uint8_t* rec = host_layers_[l] + std::size_t(e) * kHostExpertBytes;
            std::size_t woff = 0, coff = kHostCsf;
            for(int m = 0; m < 3; ++m) {
                const auto& w = t.at(base + mats[m] + ".weight");
                const auto& f = t.at(base + mats[m] + ".weight_scale.nvfp4_csf_fixed");
                const auto& x = t.at(base + mats[m] + ".weight_scale.nvfp4_csf_exceptions");
                const auto& g = t.at(base + mats[m] + ".weight_scale_2");
                const std::size_t want_w = m < 2 ? kGateBytes : kDownBytes;
                const std::size_t want_f = m < 2 ? kGateCsf : kDownCsf;
                if(w.bytes.size() != want_w || f.bytes.size() != want_f || w.dtype != "U8" || g.bytes.size() != 4)
                    throw std::runtime_error("unexpected expert tensor layout in " + base);
                const std::size_t i = (std::size_t(l) * kExperts + e) * 3 + m;
                copies.push_back({&w, rec + woff});
                copies.push_back({&f, rec + coff});
                if(x.bytes.size()) copies.push_back({&x, static_cast<uint32_t*>(pool.data()) + off[i]});
                copies.push_back({&g, &s2[i]});
                woff += want_w;
                coff += want_f;
            }
        }
    t.gather(std::move(copies), options_.load_threads);

    Packer pk;
    pk.add(&tables_.s2, s2);
    std::vector<float> offf(off.size()), cntf(cnt.size());
    std::memcpy(offf.data(), off.data(), off.size() * 4);
    std::memcpy(cntf.data(), cnt.data(), cnt.size() * 4);
    pk.add(&tables_.exc_off, offf);
    pk.add(&tables_.exc_cnt, cntf);
    exc_off_h_ = off;
    exc_cnt_h_ = cnt;
    s2_h_ = s2;
    exc_pool_h_ = static_cast<const uint32_t*>(pool.data());
    std::vector<float> ptrs(kLayers * 2);
    for(int l = 0; l < kLayers; ++l) {
        const void* p = host_layers_[l];  // pinned memory is device-addressable at the same address (UVA)
        std::memcpy(&ptrs[2 * l], &p, sizeof(p));
    }
    pk.add(reinterpret_cast<const float**>(&tables_.host_layer), ptrs);
    buffers_.push_back(pk.finish(backend_));
    tables_.exc_pool = static_cast<const uint32_t*>(pool.device_address());
    buffers_.push_back(std::move(pool));
}

void Net::load_mtp(const Tensors& t) {
    Packer pk;
    auto bf16 = [&](const __nv_bfloat16** dst, const std::string& name, std::initializer_list<std::uint64_t> shape) {
        const auto& e = t.at(name);
        expect_shape(e, shape, name);
        if(e.dtype != "BF16") throw std::runtime_error("expected BF16 " + name);
        pk.add(dst, e.bytes);
    };
    auto hc = [&](Hc& h, const std::string& base, bool inject) {
        pk.add(&h.norm1p, one_plus(any_floats(t.at(base + ".hc_norm.weight"))));
        bf16(&h.down, base + ".input_mix_weight_down.weight", {kLowrank, kHyper});
        bf16(&h.up, base + ".input_mix_weight_up.weight", {kHyper, kLowrank});
        if(inject) bf16(&h.inject, base + ".block_inject_weight.weight", {kStreams, kHyper});
    };
    const std::string M = "mtp.", L0 = "mtp.layers.0.";
    pk.add(&mtp_.emb1p, one_plus(any_floats(t.at(M + "pre_fc_norm_embedding.weight"))));
    pk.add(&mtp_.hid1p, one_plus(any_floats(t.at(M + "pre_fc_norm_hidden.weight"))));
    bf16(&mtp_.fc_emb, M + "fc_embedding.weight", {kHidden, kHidden});
    bf16(&mtp_.fc_hid, M + "fc_hidden.weight", {kHidden, kHidden});
    hc(mtp_.ahc, L0 + "attn_hyper_connection", true);
    hc(mtp_.mhc, L0 + "mlp_hyper_connection", true);
    hc(mtp_.mixer, M + "hyper_connection_mixer", false);
    bf16(&mtp_.q, L0 + "self_attn.q_proj.weight", {kQProj, kHidden});
    bf16(&mtp_.k, L0 + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden});
    bf16(&mtp_.v, L0 + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden});
    bf16(&mtp_.o, L0 + "self_attn.o_proj.weight", {kHidden, kAttnOut});
    bf16(&mtp_.idx, L0 + "self_attn.indexer.index_qk_proj.weight", {kIdxProj, kHidden});
    pk.add(&mtp_.qn1p, one_plus(any_floats(t.at(L0 + "self_attn.q_norm.weight"))));
    pk.add(&mtp_.kn1p, one_plus(any_floats(t.at(L0 + "self_attn.k_norm.weight"))));
    pk.add(&mtp_.iqn1p, one_plus(any_floats(t.at(L0 + "self_attn.indexer.q_layernorm.weight"))));
    pk.add(&mtp_.ikn1p, one_plus(any_floats(t.at(L0 + "self_attn.indexer.k_layernorm.weight"))));
    bf16(&mtp_.router, L0 + "mlp.gate.weight", {kExperts, kHidden});
    bf16(&mtp_.shared_gate, L0 + "mlp.shared_expert_gate.weight", {1, kHidden});
    bf16(&mtp_.sh_gate, L0 + "mlp.shared_expert.gate_proj.weight", {kShared, kHidden});
    bf16(&mtp_.sh_up, L0 + "mlp.shared_expert.up_proj.weight", {kShared, kHidden});
    bf16(&mtp_.sh_down, L0 + "mlp.shared_expert.down_proj.weight", {kHidden, kShared});

    // Routed experts in device slot layout: weights as stored, E4M3 block scales as stored.
    const char* mats[3] = {"gate_proj", "up_proj", "down_proj"};
    std::vector<float> s2(std::size_t(kExperts) * 3);
    buffers_.push_back(backend_.allocate(Memory::device, std::size_t(kExperts) * kSlotBytes));
    mtp_experts_ = static_cast<uint8_t*>(buffers_.back().data());
    constexpr int kBatch = 32;
    std::vector<uint8_t> stage(std::size_t(kBatch) * kSlotBytes);
    for(int e0 = 0; e0 < kExperts; e0 += kBatch) {
        std::vector<Tensors::Copy> copies;
        for(int e = e0; e < e0 + kBatch; ++e) {
            uint8_t* slot = stage.data() + std::size_t(e - e0) * kSlotBytes;
            const std::string base = L0 + "mlp.experts." + std::to_string(e) + ".";
            for(int m = 0; m < 3; ++m) {
                const auto& w = t.at(base + mats[m] + ".weight");
                const auto& sc = t.at(base + mats[m] + ".weight_scale");
                const auto& g = t.at(base + mats[m] + ".weight_scale_2");
                const std::size_t want_w = m < 2 ? kGateBytes : kDownBytes;
                const std::size_t want_s = m < 2 ? kGateScales : kDownScales;
                if(w.bytes.size() != want_w || sc.bytes.size() != want_s || w.dtype != "U8" || sc.dtype != "F8_E4M3" ||
                   g.bytes.size() != 4)
                    throw std::runtime_error("unexpected MTP expert tensor layout in " + base);
                copies.push_back({&w, slot + std::size_t(m) * kGateBytes});
                copies.push_back({&sc, slot + kSlotScales + std::size_t(m) * kGateScales});
                copies.push_back({&g, &s2[std::size_t(e) * 3 + m]});
            }
        }
        t.gather(std::move(copies), options_.load_threads);
        FNL_CUDA(cudaMemcpy(mtp_experts_ + std::size_t(e0) * kSlotBytes, stage.data(), stage.size(),
                            cudaMemcpyHostToDevice));
    }
    pk.add(&tables_.fixed_s2, s2);
    buffers_.push_back(pk.finish(backend_));
    tables_.fixed_base = mtp_experts_;

    // Drafting head: lm_head re-encoded as MXFP8 (the verified pass keeps the BF16 head).
    const std::size_t qbytes = std::size_t(kVocab) * kHidden, sbytes = qbytes / 32;
    buffers_.push_back(backend_.allocate(Memory::device, qbytes + sbytes));
    auto* hb = static_cast<uint8_t*>(buffers_.back().data());
    quantize_mx8(lm_head_, kVocab, kHidden, hb, hb + qbytes, stream());
    FNL_CUDA(cudaStreamSynchronize(stream()));
    mtp_.head = {hb, hb + qbytes, kVocab, kHidden};
    mtp_loaded_ = true;
}

void Net::allocate_state() {
    const std::size_t cap = options_.capacity;
    constexpr std::size_t T = kMaxTokens;
    std::size_t total = 0;
    struct Slot { void** p; std::size_t bytes; };
    std::vector<Slot> slots;
    auto f = [&](float** p, std::size_t n) { slots.push_back({reinterpret_cast<void**>(p), n * 4}); };
    auto i32 = [&](int** p, std::size_t n) { slots.push_back({reinterpret_cast<void**>(p), n * 4}); };
    auto raw = [&](auto** p, std::size_t bytes) { slots.push_back({reinterpret_cast<void**>(p), bytes}); };
    f(&gdn_state_, std::size_t(36) * kGdnRing * kGdnStateFloats);
    f(&conv_state_, std::size_t(36) * kQkvDim * 3);
    f(&conv_in_, std::size_t(36) * T * kQkvDim);
    raw(&kcache_, std::size_t(12) * kKvHeads * cap * kHeadDim);
    raw(&vcache_, std::size_t(12) * kKvHeads * cap * kHeadDim);
    f(&kscale_, std::size_t(12) * kKvHeads * cap);
    f(&vscale_, std::size_t(12) * kKvHeads * cap);
    f(&rawkeys_, std::size_t(12) * kRawRing * kIdxDim);
    f(&blockkeys_, std::size_t(12) * (cap / kBlockTokens) * kIdxDim);
    f(&ple_state_, std::size_t(kHyper) * kPleState);
    // decode workspace
    f(&H_, T * kHyper); f(&xn_, T * kHyper); f(&lora_, T * kLowrank); f(&mixw_, T * kHyper); f(&inj_, T * kStreams);
    f(&x_, T * kHidden); f(&proj_, T * kProjAttn); f(&z_, T * kZDim); f(&ab_, T * 2 * kVHeads);
    f(&q_, T * kKHeads * kKDim); f(&k_, T * kKHeads * kKDim); f(&v_, T * kZDim); f(&g_, T * kVHeads);
    f(&beta_, T * kVHeads); f(&o_, T * kZDim); f(&gn_, T * kZDim); f(&out_, T * kHidden);
    f(&aq_, T * kAttnOut); f(&agate_, T * kAttnOut); f(&aiq_, T * kIdxHeads * kIdxDim);
    f(&scores_, T * (cap / kBlockTokens)); f(&partial_, T * kPartialStride);
    f(&rlogits_, T * kExperts); f(&act_, std::size_t(kMaxUnion) * kMaxPassTokens * kInter);
    f(&shg_, T * kShared); f(&shu_, T * kShared); f(&sha_, T * kShared); f(&sho_, T * kHidden); f(&sg_, T);
    f(&logits_, T * kVocab); f(&amax_v_, T * 256);
    f(&ple_e_, T * kHidden); f(&ple_k_, T * kHyper); f(&ple_v_, T * kHidden); f(&ple_gv_, T * kHyper);
    f(&ple_gvn_, T * kHyper);
    i32(&sel_, T * kTopBlocks); i32(&next_, T); i32(&amax_i_, T * 256); i32(&ids_, T); i32(&one_, 1);
    i32(&kept_, T + 1);
    i32(&gdn_cur_, 1);
    i32(&tables_.cpu_state, 2);
    raw(&tables_.pass_first, 4);
    raw(&step_, T * sizeof(Step));
    raw(&ple_rows_dev_, T * kPleHeads * 90);
    raw(&routes_, T * sizeof(Route));
    f(&down_part_, std::size_t(kMaxUnion) * kMaxPassTokens * kHidden);
    raw(&plan_, sizeof(ExpertPlan));
    // MTP block state and workspace
    if(mtp_loaded_) {
        raw(&mkcache_, std::size_t(kKvHeads) * cap * kHeadDim);
        raw(&mvcache_, std::size_t(kKvHeads) * cap * kHeadDim);
        f(&mkscale_, std::size_t(kKvHeads) * cap);
        f(&mvscale_, std::size_t(kKvHeads) * cap);
        f(&mrawkeys_, std::size_t(kRawRing) * kIdxDim);
        f(&mblockkeys_, (cap / kBlockTokens) * kIdxDim);
        f(&me_, T * kHidden); f(&men_, T * kHidden); f(&fe_, T * kHidden); f(&hn_, T * kHyper); f(&fh_, T * kHyper);
        f(&mH_, T * kHyper); f(&dlogits_, kVocab);
        raw(&acc_, sizeof(Accept));
        i32(&dinp_.ids, T); i32(&dinp_.rows, T); raw(&dinp_.steps, T * sizeof(Step)); i32(&rows_last_, 2);
    }
    // expert directory
    raw(&tables_.slot_of, std::size_t(kLayers) * kExperts * 4);
    raw(&tables_.hand, 4);
    raw(&tables_.step, 4);
    raw(&tables_.stats, 16);
    raw(&tables_.scratch, std::size_t(kMaxUnion) * kSlotBytes);
    for(auto& s : slots) total += (s.bytes + 255) / 256 * 256;
    auto buffer = backend_.allocate(Memory::device, total);
    auto* base = static_cast<uint8_t*>(buffer.data());
    std::size_t off = 0;
    for(auto& s : slots) {
        *s.p = base + off;
        off += (s.bytes + 255) / 256 * 256;
    }
    buffers_.push_back(std::move(buffer));
    const std::size_t pin_rows = T * kPleHeads * 90;
    auto pinned = backend_.allocate(Memory::host, pin_rows + 1024);
    ple_rows_host_ = static_cast<uint8_t*>(pinned.data());
    ids_host_ = reinterpret_cast<int*>(ple_rows_host_ + pin_rows + 64);
    step_host_ = reinterpret_cast<Step*>(ple_rows_host_ + pin_rows + 128);
    spec_host_ = reinterpret_cast<Accept*>(ple_rows_host_ + pin_rows + 512);
    buffers_.push_back(std::move(pinned));
    stats_ = tables_.stats;
    FNL_CUDA(cudaMemset(tables_.slot_of, 0xff, std::size_t(kLayers) * kExperts * 4));
    FNL_CUDA(cudaMemset(tables_.hand, 0, 4));
    FNL_CUDA(cudaMemset(tables_.step, 0, 4));
    FNL_CUDA(cudaMemset(tables_.stats, 0, 16));
    FNL_CUDA(cudaMemset(tables_.cpu_state, 0, 8));
    FNL_CUDA(cudaMemset(tables_.pass_first, 0, 4));
    const int one = 1;
    FNL_CUDA(cudaMemcpy(one_, &one, 4, cudaMemcpyHostToDevice));
    const int kept[kMaxTokens + 1] = {0, 1, 2, 3, 4};
    FNL_CUDA(cudaMemcpy(kept_, kept, sizeof(kept), cudaMemcpyHostToDevice));
    if(mtp_loaded_) {
        const int rows[2] = {kMaxTokens - 1, 0};
        FNL_CUDA(cudaMemcpy(rows_last_, rows, 8, cudaMemcpyHostToDevice));
        buffers_.push_back(backend_.allocate(Memory::host, 256 + std::size_t(kMaxTokens) * kPleHeads * 90));
        auto* p = static_cast<uint8_t*>(buffers_.back().data());
        std::memset(p, 0, 256);
        progress_ = reinterpret_cast<SpecProgress*>(p);
        next_rows_ = p + 256;
    }
}

void Net::allocate_slots() {
    std::size_t spare = backend_.spare_device_bytes(options_.vram_margin);
    std::size_t own = spare / kSlotBytes;
    if(options_.expert_slots) own = std::min(own, options_.expert_slots);
    own = std::min<std::size_t>(own, std::size_t(kLayers) * kExperts);
    // Lendable slots: the tail of every KV (layer, head, K/V) region and the prompt workspace.
    std::vector<uint8_t*> addr;
    std::vector<int> kind, first;
    const std::size_t cap = options_.capacity;
    const std::size_t region = cap * kHeadDim;   // one E4M3 byte per element
    if(!options_.expert_slots) {
        std::vector<uint8_t*> regions;
        for(int r = 0; r < 12 * kKvHeads; ++r) regions.push_back(reinterpret_cast<uint8_t*>(kcache_) + r * region);
        for(int r = 0; r < 12 * kKvHeads; ++r) regions.push_back(reinterpret_cast<uint8_t*>(vcache_) + r * region);
        if(mtp_loaded_)
            for(int r = 0; r < kKvHeads; ++r) {
                regions.push_back(reinterpret_cast<uint8_t*>(mkcache_) + r * region);
                regions.push_back(reinterpret_cast<uint8_t*>(mvcache_) + r * region);
            }
        for(uint8_t* base : regions)
            for(std::size_t j = 1; j * kSlotBytes <= region; ++j) {
                const std::size_t off = region - j * kSlotBytes;
                addr.push_back(base + off);
                kind.push_back(kKvSlot);
                first.push_back(int(off / kHeadDim));
            }
        for(const auto& [ptr, bytes] : workspace_lend_)
            for(std::size_t j = 0; (j + 1) * kSlotBytes <= bytes; ++j) {
                addr.push_back(ptr + j * kSlotBytes);
                kind.push_back(kWorkspaceSlot);
                first.push_back(0);
            }
    }
    const std::size_t n = own + addr.size();
    tables_.nslots = int(n);
    if(n == 0) return;
    std::size_t meta = (n * 4 + 255) / 256 * 256 * 2 + (n + 255) / 256 * 256 * 2 + (n * 8 + 255) / 256 * 256;
    auto buf = backend_.allocate(Memory::device, own * kSlotBytes + meta);
    auto* base = static_cast<uint8_t*>(buf.data());
    std::size_t off = own * kSlotBytes;
    tables_.owner = reinterpret_cast<int*>(base + off); off += (n * 4 + 255) / 256 * 256;
    tables_.stamp = reinterpret_cast<uint32_t*>(base + off); off += (n * 4 + 255) / 256 * 256;
    tables_.ref = base + off; off += (n + 255) / 256 * 256;
    tables_.valid = base + off; off += (n + 255) / 256 * 256;
    tables_.slot_addr = reinterpret_cast<uint8_t* const*>(base + off);
    buffers_.push_back(std::move(buf));
    mslot_addr_.resize(n);
    slot_kind_.assign(n, kOwnSlot);
    slot_first_pos_.assign(n, 0);
    for(std::size_t s = 0; s < own; ++s) mslot_addr_[s] = base + s * kSlotBytes;
    for(std::size_t j = 0; j < addr.size(); ++j) {
        mslot_addr_[own + j] = addr[j];
        slot_kind_[own + j] = kind[j];
        slot_first_pos_[own + j] = first[j];
    }
    // Lent slots start disabled; begin_request() enables what a request leaves unused.
    mvalid_.assign(n, 0);
    for(std::size_t s = 0; s < own; ++s) mvalid_[s] = 1;
    FNL_CUDA(cudaMemcpy(const_cast<uint8_t**>(tables_.slot_addr), mslot_addr_.data(), n * 8, cudaMemcpyHostToDevice));
    FNL_CUDA(cudaMemcpy(const_cast<uint8_t*>(tables_.valid), mvalid_.data(), n, cudaMemcpyHostToDevice));
    FNL_CUDA(cudaMemset(tables_.owner, 0xff, n * 4));
    FNL_CUDA(cudaMemset(tables_.stamp, 0, n * 4));
    FNL_CUDA(cudaMemset(tables_.ref, 0, n));
    own_slots_ = int(own);
}

void Net::reset() {
    const auto s = stream();
    FNL_CUDA(cudaMemsetAsync(gdn_state_, 0, std::size_t(36) * kGdnRing * kGdnStateFloats * 4, s));
    FNL_CUDA(cudaMemsetAsync(gdn_cur_, 0, 4, s));
    FNL_CUDA(cudaMemsetAsync(conv_state_, 0, std::size_t(36) * kQkvDim * 3 * 4, s));
    FNL_CUDA(cudaMemsetAsync(ple_state_, 0, std::size_t(kHyper) * kPleState * 4, s));
    collect_pass_timing();
    target_ms_ = draft_ms_ = 0;
}

void Net::collect_pass_timing() {
    if(!pass_pending_) return;
    pass_pending_ = false;
    FNL_CUDA(cudaEventSynchronize(pass_marks_[2]));
    float a = 0, b = 0;
    FNL_CUDA(cudaEventElapsedTime(&a, pass_marks_[0], pass_marks_[1]));
    target_ms_ += a;
    if(pass_drafted_) {
        FNL_CUDA(cudaEventElapsedTime(&b, pass_marks_[1], pass_marks_[2]));
        draft_ms_ += b;
    }
}

std::pair<double, double> Net::pass_timing() {
    collect_pass_timing();
    return {target_ms_, draft_ms_};
}

unsigned long long Net::expert_lookups() const {
    unsigned long long v[2];
    FNL_CUDA(cudaMemcpy(v, stats_, 16, cudaMemcpyDeviceToHost));
    return v[0];
}
unsigned long long Net::expert_misses() const {
    unsigned long long v[2];
    FNL_CUDA(cudaMemcpy(v, stats_, 16, cudaMemcpyDeviceToHost));
    return v[1];
}
void Net::hc_pre(const Hc& w, float* H, int tokens, float* inj) {
    const auto s = stream();
    hc_norm(H, w.norm1p, xn_, tokens, s);
    bf16_gemv({w.down, xn_, lora_, kLowrank, kHyper, Post::silu_quarter, tokens}, s);
    bf16_gemv({w.up, lora_, mixw_, kHyper, kLowrank, Post::sigmoid, tokens}, s);
    if(w.inject) bf16_gemv({w.inject, xn_, inj, kStreams, kHyper, Post::inject, tokens}, s);
    hc_mix(xn_, mixw_, x_, tokens, s);
}

// tokens[p - base] is the token at position p (base <= first - 2 or 0).
void Net::ple_rows(const int* h, int base, int first, int count, uint8_t* out, bool parallel) const {
    std::vector<RowCopy> copies;
    copies.reserve(std::size_t(count) * kPleHeads * 2);
    for(int t = first; t < first + count; ++t) {
        int prev_eos = -1;
        for(int j = t - 1; j >= std::max(0, t - 2); --j)
            if(h[j - base] == kEosToken) { prev_eos = j; break; }
        int64_t tok[3];
        for(int shift = 0; shift < 3; ++shift) {
            const bool valid = shift <= t - prev_eos - 1 && t - shift >= 0;
            tok[shift] = valid ? h[t - shift - base] : kEosToken;
        }
        for(int n = 2; n <= 3; ++n) {
            uint64_t mixed = uint64_t(tok[0]) * uint64_t(ple_.mult[0]);
            for(int p = 1; p < n; ++p) mixed ^= uint64_t(tok[p]) * uint64_t(ple_.mult[p]);
            const int64_t sm = int64_t(mixed);
            for(int hh = 0; hh < 8; ++hh) {
                const int head = (n - 2) * 8 + hh;
                int64_t r = sm % ple_.sizes[head];
                if(r < 0) r += ple_.sizes[head];
                const int64_t row = r + ple_.offsets[head];
                const int shard = int(row / ple_.rows_per_shard);
                const int64_t in = row % ple_.rows_per_shard;
                uint8_t* dst = out + (std::size_t(t - first) * kPleHeads + head) * 90;
                copies.push_back({dst, ple_.shard_w[shard] + in * 80, 80});
                copies.push_back({dst + 80, ple_.shard_s[shard] + in * 10, 10});
            }
        }
    }
    if(parallel && gather_pool_) {
        gather_pool_->run(copies.data(), int(copies.size()));
    } else {
        for(const auto& c : copies) std::memcpy(c.dst, c.src, std::size_t(c.bytes));
    }
}

void Net::decode_layer(int l, int T) {
    const auto s = stream();
    const auto& L = layers_[l];
    if(l == kPleLayer) {
        FNL_CUDA(cudaMemcpyAsync(ple_rows_dev_, ple_rows_host_, std::size_t(T) * kPleHeads * 90,
                                 cudaMemcpyHostToDevice, s));
        ple_dequant(ple_rows_dev_, ple_.s2, ple_e_, T, s);
        bf16_gemv({ple_.key, ple_e_, ple_k_, kHyper, kHidden, Post::none, T}, s);
        bf16_gemv({ple_.value, ple_e_, ple_v_, kHidden, kHidden, Post::none, T}, s);
        ple_mix({H_, ple_k_, ple_v_, ple_.nk1p, ple_.nq1p, ple_.nc1p, ple_.conv_w, ple_state_, T}, ple_gv_,
                ple_gvn_, s, false);
    }
    hc_pre(L.ahc, H_, T, inj_);
    if(L.attention) {
        const int ai = attn_index(l);
        Mx8Batch b{};
        b.seg[0] = {L.q.w, L.q.s, proj_, kQProj, kProjAttn};
        b.seg[1] = {L.k.w, L.k.s, proj_ + kQProj, kKvHeads * kHeadDim, kProjAttn};
        b.seg[2] = {L.v.w, L.v.s, proj_ + kQProj + kKvHeads * kHeadDim, kKvHeads * kHeadDim, kProjAttn};
        b.seg[3] = {L.idx.w, L.idx.s, proj_ + kQProj + 2 * kKvHeads * kHeadDim, kIdxProj, kProjAttn};
        b.nseg = 4;
        b.cols = kHidden;
        b.x = x_;
        b.tokens = T;
        mx8_gemv(b, s);
        const std::size_t cap = options_.capacity;
        uint8_t* kc = kcache_ + std::size_t(ai) * kKvHeads * cap * kHeadDim;
        uint8_t* vc = vcache_ + std::size_t(ai) * kKvHeads * cap * kHeadDim;
        float* ks = kscale_ + std::size_t(ai) * kKvHeads * cap;
        float* vs = vscale_ + std::size_t(ai) * kKvHeads * cap;
        float* bk = blockkeys_ + std::size_t(ai) * (cap / kBlockTokens) * kIdxDim;
        AttnPrep ap{proj_, L.qn1p, L.kn1p, L.iqn1p, L.ikn1p, inv_freq_, aq_, agate_, aiq_, kc, vc, ks, vs,
                    rawkeys_ + std::size_t(ai) * kRawRing * kIdxDim, bk, options_.capacity, step_, T, kProjAttn};
        attn_prep(ap, s);
        if(graph_blocks_ > kTopBlocks) {
            idx_scores(aiq_, bk, step_, graph_blocks_, scores_, options_.capacity / kBlockTokens, T, s);
            topk_blocks(scores_, options_.capacity / kBlockTokens, step_, sel_, T, s);
        }
        attn_decode({aq_, kc, vc, ks, vs, sel_, step_, options_.capacity, partial_, agate_, gn_, T}, s);
        Mx8Batch ob{};
        ob.seg[0] = {L.o.w, L.o.s, out_, kHidden};
        ob.nseg = 1;
        ob.cols = kAttnOut;
        ob.x = gn_;
        ob.tokens = T;
        mx8_gemv(ob, s);
    } else {
        const int gi = gdn_index(l);
        float* qkv = conv_in_ + std::size_t(gi) * kMaxTokens * kQkvDim;
        Mx8Batch b{};
        b.seg[0] = {L.qkv.w, L.qkv.s, qkv, kQkvDim};
        b.seg[1] = {L.z.w, L.z.s, z_, kZDim};
        b.seg[2] = {L.a.w, L.a.s, ab_, kVHeads, 2 * kVHeads};
        b.seg[3] = {L.b.w, L.b.s, ab_ + kVHeads, kVHeads, 2 * kVHeads};
        b.nseg = 4;
        b.cols = kHidden;
        b.x = x_;
        b.tokens = T;
        mx8_gemv(b, s);
        GdnStep gs{qkv, ab_, L.conv_w, conv_state_ + std::size_t(gi) * kQkvDim * 3, L.neg_a, L.dt_bias, q_, k_, v_, g_,
                   beta_, T, false};
        gdn_conv(gs, s);
        gdn_recur(q_, k_, v_, g_, beta_, gdn_state_ + std::size_t(gi) * kGdnRing * kGdnStateFloats, gdn_cur_, true, o_,
                  T, s);
        gdn_gate_norm(o_, z_, L.gnorm, gn_, T, s);
        Mx8Batch ob{};
        ob.seg[0] = {L.out.w, L.out.s, out_, kHidden};
        ob.nseg = 1;
        ob.cols = kZDim;
        ob.x = gn_;
        ob.tokens = T;
        mx8_gemv(ob, s);
    }
    hc_combine(H_, out_, inj_, T, s);
    hc_pre(L.mhc, H_, T, inj_);
    // MoE
    bf16_gemv({L.router, x_, rlogits_, kExperts, kHidden, Post::none, T}, s);
    route_batch(rlogits_, routes_, T, s);
    plan_experts(tables_, l, routes_, T, plan_, x_, s);
    plan_miss_scales(tables_, l, plan_, T, s);
    expert_gate_up(plan_, x_, act_, T, s);
    Mx8Batch sb{};
    sb.seg[0] = {L.sh_gate.w, L.sh_gate.s, shg_, kShared};
    sb.seg[1] = {L.sh_up.w, L.sh_up.s, shu_, kShared};
    sb.nseg = 2;
    sb.cols = kHidden;
    sb.x = x_;
    sb.tokens = T;
    mx8_gemv(sb, s);
    silu_mul(shg_, shu_, sha_, T * kShared, s);
    Mx8Batch db{};
    db.seg[0] = {L.sh_down.w, L.sh_down.s, sho_, kHidden};
    db.nseg = 1;
    db.cols = kShared;
    db.x = sha_;
    db.tokens = T;
    mx8_gemv(db, s);
    bf16_gemv({L.shared_gate, x_, sg_, 1, kHidden, Post::sigmoid, T}, s);
    expert_down(plan_, act_, sho_, sg_, out_, T, s, down_part_);
    if(tables_.job) {
        cpu_wait(tables_, cpu_reply_, s);
        hc_combine_cpu(H_, out_, inj_, T, cpu_reply_, tables_.cpu_state, s);
    } else {
        hc_combine(H_, out_, inj_, T, s);
    }
    if(on_layer) on_layer(l, H_, T);
}

// Background slot insertions of the previous pass are enqueued and ordered before what follows on the stream.
void Net::sync_insertions() {
    if(!cpu_server_) return;
    const uint32_t posted = tables_.job->seq;
    while(cpu_server_->issued() != posted) std::this_thread::yield();
    FNL_CUDA(cudaEventRecord(insert_done_, insert_stream_));
    FNL_CUDA(cudaStreamWaitEvent(stream(), insert_done_, 0));
}

// Head of a pass: step facts and token ids to the device, embedding, layer 0 (which needs no n-gram rows).
void Net::decode_head(int T) {
    const auto s = stream();
    begin_pass(tables_, s);
    FNL_CUDA(cudaMemcpyAsync(step_, step_host_, std::size_t(T) * sizeof(Step), cudaMemcpyHostToDevice, s));
    FNL_CUDA(cudaMemcpyAsync(ids_, ids_host_, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
    embed_rows(embed_, ids_, H_, T, s);
    decode_layer(0, T);
}

// Tail of a pass: layers 1..47 (the PLE layer copies the host-gathered rows), mixer, head, argmax per token.
void Net::decode_tail(int T) {
    const auto s = stream();
    for(int l = 1; l < kLayers; ++l) decode_layer(l, T);
    hc_pre(mixer_, H_, T, nullptr);
    bf16_gemv({lm_head_, x_, logits_, kVocab, kHidden, Post::none, T}, s);
    argmax(logits_, kVocab, next_, amax_v_, amax_i_, s, T, kVocab);
}

// The drafter over R rows: fused input (token embedding, hidden streams), its attention layer and experts.
void Net::mtp_pass(int R, const int* ids, const float* hidden, const int* rows, const Step* steps) {
    const auto s = stream();
    embed_single(embed_, ids, me_, R, s);
    rms_rows(me_, kHidden, nullptr, mtp_.emb1p, men_, R, s);
    bf16_gemv({mtp_.fc_emb, men_, fe_, kHidden, kHidden, Post::none, R}, s);
    rms_rows(hidden, kHyper, rows, mtp_.hid1p, hn_, R, s);
    for(int r = 0; r < R; ++r)
        bf16_gemv({mtp_.fc_hid, hn_ + std::size_t(r) * kHyper, fh_ + std::size_t(r) * kHyper, kHidden, kHidden,
                   Post::none, kStreams, kHidden, kHidden}, s);
    mtp_fuse(fe_, fh_, mH_, R, s);
    hc_pre(mtp_.ahc, mH_, R, inj_);
    bf16_gemv({mtp_.q, x_, proj_, kQProj, kHidden, Post::none, R, 0, kProjAttn}, s);
    bf16_gemv({mtp_.k, x_, proj_ + kQProj, kKvHeads * kHeadDim, kHidden, Post::none, R, 0, kProjAttn}, s);
    bf16_gemv({mtp_.v, x_, proj_ + kQProj + kKvHeads * kHeadDim, kKvHeads * kHeadDim, kHidden, Post::none, R, 0,
               kProjAttn}, s);
    bf16_gemv({mtp_.idx, x_, proj_ + kQProj + 2 * kKvHeads * kHeadDim, kIdxProj, kHidden, Post::none, R, 0,
               kProjAttn}, s);
    AttnPrep ap{proj_, mtp_.qn1p, mtp_.kn1p, mtp_.iqn1p, mtp_.ikn1p, inv_freq_, aq_, agate_, aiq_, mkcache_, mvcache_,
                mkscale_, mvscale_, mrawkeys_, mblockkeys_, options_.capacity, steps, R, kProjAttn};
    attn_prep(ap, s);
    if(graph_blocks_ > kTopBlocks) {
        idx_scores(aiq_, mblockkeys_, steps, graph_blocks_, scores_, options_.capacity / kBlockTokens, R, s);
        topk_blocks(scores_, options_.capacity / kBlockTokens, steps, sel_, R, s);
    }
    attn_decode({aq_, mkcache_, mvcache_, mkscale_, mvscale_, sel_, steps, options_.capacity, partial_, agate_, gn_, R},
                s);
    bf16_gemv({mtp_.o, gn_, out_, kHidden, kAttnOut, Post::none, R}, s);
    hc_combine(mH_, out_, inj_, R, s);
    hc_pre(mtp_.mhc, mH_, R, inj_);
    bf16_gemv({mtp_.router, x_, rlogits_, kExperts, kHidden, Post::none, R}, s);
    route_batch(rlogits_, routes_, R, s);
    plan_experts(tables_, kMtpLayer, routes_, R, plan_, x_, s);
    expert_gate_up(plan_, x_, act_, R, s);
    bf16_gemv({mtp_.sh_gate, x_, shg_, kShared, kHidden, Post::none, R}, s);
    bf16_gemv({mtp_.sh_up, x_, shu_, kShared, kHidden, Post::none, R}, s);
    silu_mul(shg_, shu_, sha_, R * kShared, s);
    bf16_gemv({mtp_.sh_down, sha_, sho_, kHidden, kShared, Post::none, R}, s);
    bf16_gemv({mtp_.shared_gate, x_, sg_, 1, kHidden, Post::sigmoid, R}, s);
    expert_down(plan_, act_, sho_, sg_, out_, R, s, down_part_);
    hc_combine(mH_, out_, inj_, R, s);
}

// The drafter's token from its last row: final mixer, MXFP8 head, argmax into acc_->drafts[k].
void Net::mtp_draft(int R, int k) {
    const auto s = stream();
    hc_pre(mtp_.mixer, mH_ + std::size_t(R - 1) * kHyper, 1, nullptr);
    Mx8Batch hb{};
    hb.seg[0] = {mtp_.head.w, mtp_.head.s, dlogits_, kVocab};
    hb.nseg = 1;
    hb.cols = kHidden;
    hb.x = x_;
    mx8_gemv(hb, s);
    argmax(dlogits_, kVocab, acc_->drafts + k, amax_v_, amax_i_, s);
}

// After a verified pass: acceptance and state commit for the kept tokens (spec_drafts follows).
void Net::spec_accept() {
    const auto s = stream();
    accept_pass(ids_, next_, kMaxTokens, step_, acc_, dinp_, s);
    progress_note(acc_, progress_, 1, s);
    commit_states({conv_state_, conv_in_, 36, ple_state_, ple_gvn_, gdn_cur_, &acc_->n}, s);
}

// Drafter extend pass over the kept positions, chained drafts, result to pinned memory.
void Net::spec_drafts() {
    const auto s = stream();
    mtp_pass(kMaxTokens, dinp_.ids, H_, dinp_.rows, dinp_.steps);
    mtp_draft(kMaxTokens, 0);
    progress_note(acc_, progress_, 2, s);
    for(int k = 0; k + 1 < kDrafts; ++k) {
        chain_step(acc_, step_, k, dinp_, s);
        mtp_pass(1, dinp_.ids, mH_, rows_last_ + (k == 0 ? 0 : 1), dinp_.steps);
        mtp_draft(1, k + 1);
        progress_note(acc_, progress_, k + 3, s);
    }
    FNL_CUDA(cudaMemcpyAsync(spec_host_, acc_, sizeof(Accept), cudaMemcpyDeviceToHost, s));
}

void Net::set_steps(const std::vector<int>& history, int pos, int T, const int* tokens) {
    for(int t = 0; t < T; ++t) {
        step_host_[t] = make_step(pos + t, tokens[t]);
        ids_host_[t] = tokens[t];
    }
    (void)history;
}

namespace {
int index_bucket(int max_pos, int capacity) {
    const int nb = (max_pos + 1) / kBlockTokens;
    if(nb <= kTopBlocks) return 0;
    int bucket = 1024;
    while(bucket < nb) bucket <<= 1;
    return std::min(bucket, capacity / kBlockTokens);
}
}  // namespace

// One pass of T tokens at positions pos..pos+T-1: tokens history[pos] then (spec) the drafts in step_host_.
void Net::launch_pass(const std::vector<int>& history, int pos, int T, bool spec, bool have_rows) {
    const int* keep = T == 1 ? one_ : kept_ + T;   // tokens a plain pass keeps
    const auto s = stream();
    // Index-score grids come in power-of-two block buckets; kernels bound themselves by the device steps.
    const int bucket = index_bucket(pos + T - 1 + (spec ? 2 : 0), options_.capacity);
    static const bool no_graphs = std::getenv("FNL_NO_GRAPHS") != nullptr;
    const bool eager = no_graphs || on_layer;
    graph_blocks_ = bucket;
    // n-gram rows: the window of tokens before pos plus the pass's tokens.
    const int base = std::max(0, pos - 2);
    int win[2 + kMaxTokens];
    for(int p = base; p < pos; ++p) win[p - base] = history[p];
    for(int t = 0; t < T; ++t) win[pos - base + t] = ids_host_[t];
    auto gather = [&] {
        if(have_rows) return;
        static const bool timing = std::getenv("FNL_TIMING") != nullptr;
        static double total = 0;
        static long calls = 0;
        const auto g0 = std::chrono::steady_clock::now();
        ple_rows(win, base, pos, T, ple_rows_host_, true);
        if(timing) {
            total += std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
            if(++calls % 50 == 0)
                std::fprintf(stderr, "ple gather: %.3f ms per pass over %ld passes\n", 1e3 * total / calls, calls);
        }
    };
    // Device time of the pass and of its drafts, read when the next pass is launched (its caller waited for
    // this one) or by pass_timing().
    collect_pass_timing();
    if(!pass_marks_[0])
        for(auto& e : pass_marks_) FNL_CUDA(cudaEventCreate(&e));
    FNL_CUDA(cudaEventRecord(pass_marks_[0], s));
    pass_pending_ = true;
    pass_drafted_ = spec;
    if(eager) {
        decode_head(T);
        gather();
        decode_tail(T);
        if(spec) spec_accept();
        else commit_states({conv_state_, conv_in_, 36, ple_state_, ple_gvn_, gdn_cur_, keep}, s);
        FNL_CUDA(cudaEventRecord(pass_marks_[1], s));
        if(spec) spec_drafts();
        FNL_CUDA(cudaEventRecord(pass_marks_[2], s));
        FNL_CUDA(cudaEventRecord(step_done_, s));
        return;
    }
    auto& g = graphs_[{T * 2 + (spec ? 1 : 0), bucket}];
    if(!g.head) {
        cudaGraph_t graph;
        FNL_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        decode_head(T);
        FNL_CUDA(cudaStreamEndCapture(s, &graph));
        FNL_CUDA(cudaGraphInstantiate(&g.head, graph, 0));
        FNL_CUDA(cudaGraphDestroy(graph));
        FNL_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        decode_tail(T);
        if(spec) spec_accept();
        else commit_states({conv_state_, conv_in_, 36, ple_state_, ple_gvn_, gdn_cur_, keep}, s);
        FNL_CUDA(cudaStreamEndCapture(s, &graph));
        FNL_CUDA(cudaGraphInstantiate(&g.tail, graph, 0));
        FNL_CUDA(cudaGraphDestroy(graph));
        if(spec) {
            FNL_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            spec_drafts();
            FNL_CUDA(cudaStreamEndCapture(s, &graph));
            FNL_CUDA(cudaGraphInstantiate(&g.draft, graph, 0));
            FNL_CUDA(cudaGraphDestroy(graph));
        }
    }
    FNL_CUDA(cudaGraphLaunch(g.head, s));
    // The n-gram rows are gathered while the head (embedding and layer 0) runs.
    gather();
    FNL_CUDA(cudaGraphLaunch(g.tail, s));
    FNL_CUDA(cudaEventRecord(pass_marks_[1], s));
    if(spec) FNL_CUDA(cudaGraphLaunch(g.draft, s));
    FNL_CUDA(cudaEventRecord(pass_marks_[2], s));
    FNL_CUDA(cudaEventRecord(step_done_, s));
}

void Net::decode(const std::vector<int>& history, int pos) {
    if(pos >= options_.capacity) throw std::out_of_range("position beyond the KV capacity");
    // The previous pass must have read the pinned step facts and n-gram rows before they are rewritten.
    if(!step_done_) FNL_CUDA(cudaEventCreateWithFlags(&step_done_, cudaEventDisableTiming));
    else FNL_CUDA(cudaEventSynchronize(step_done_));
    const int tok = history[pos];
    set_steps(history, pos, 1, &tok);
    sync_insertions();
    launch_pass(history, pos, 1, false);
}

void Net::verify_pass(const std::vector<int>& history, int pos, int T) {
    if(T < 1 || T > kMaxTokens) throw std::invalid_argument("verify pass of 1..4 tokens");
    if(pos + T > options_.capacity) throw std::out_of_range("position beyond the KV capacity");
    if(!step_done_) FNL_CUDA(cudaEventCreateWithFlags(&step_done_, cudaEventDisableTiming));
    else FNL_CUDA(cudaEventSynchronize(step_done_));
    int toks[kMaxTokens];
    for(int t = 0; t < T; ++t) toks[t] = history[pos + t];
    set_steps(history, pos, T, toks);
    sync_insertions();
    launch_pass(history, pos, T, false);
}

void Net::spec_round(const std::vector<int>& history, int pos, const int* drafts) {
    if(!mtp_loaded_) throw std::logic_error("speculative decoding needs the MTP block");
    if(pos + kMaxTokens + 2 > options_.capacity) throw std::out_of_range("position beyond the KV capacity");
    if(!step_done_) FNL_CUDA(cudaEventCreateWithFlags(&step_done_, cudaEventDisableTiming));
    else FNL_CUDA(cudaEventSynchronize(step_done_));
    int toks[kMaxTokens];
    toks[0] = history[pos];
    for(int k = 0; k < kDrafts; ++k) toks[k + 1] = drafts[k];
    set_steps(history, pos, kMaxTokens, toks);
    sync_insertions();
    // Rows gathered during the previous round, when its prediction of this pass holds.
    bool have = ahead_.valid && ahead_.pos == pos;
    for(int t = 0; have && t < kMaxTokens; ++t) have = ahead_.tokens[t] == toks[t];
    if(have) std::memcpy(ple_rows_host_, next_rows_, std::size_t(kMaxTokens) * kPleHeads * 90);
    ahead_.valid = false;
    progress_->stage = 0;
    launch_pass(history, pos, kMaxTokens, true, have);
    // Next round: tokens before pos, this pass's inputs; kept ones follow from the device's progress.
    const int base = std::max(0, pos - 2);
    int window[2 + 2 * kMaxTokens];
    for(int p = base; p < pos; ++p) window[p - base] = history[p];
    prefetch_next(window, base, pos - 1, toks, kMaxTokens);
}

// The next pass starts at pos_last + n with tokens bonus, d1, d2, d3; the window holds the tokens from base up
// to pos_last, then the inputs of this round's pass (its first n are kept).
void Net::prefetch_next(const int* window, int base, int pos_last, const int* inputs, int ninputs) {
    static const bool off = std::getenv("FNL_NO_PREFETCH") != nullptr;
    if(off || !mtp_loaded_) return;
    auto wait_stage = [&](int stage) {
        const auto t0 = std::chrono::steady_clock::now();
        while(progress_->stage < stage) {
            _mm_pause();
            if(std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) return false;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        return true;
    };
    if(!wait_stage(1)) return;
    const int n = progress_->n;
    if(n < 1 || n > ninputs) return;
    // Positions: inputs[k] sits at pos_last + 1 + k. Kept tokens are inputs[0 .. n-1]; the next pass starts at
    // pos_last + 1 + n with the bonus.
    int tokens[2 + 2 * kMaxTokens];
    int tb = std::max(0, pos_last + 1 + n - 2);   // first position the next pass's rows can read
    int cnt = 0;
    auto token_at = [&](int p) -> int {
        if(p <= pos_last) return window[p - base];
        return inputs[p - pos_last - 1];
    };
    for(int p = tb; p < pos_last + 1 + n; ++p) tokens[cnt++] = token_at(p);
    const int next_pos = pos_last + 1 + n;
    int next_tokens[kMaxTokens];
    next_tokens[0] = progress_->bonus;
    tokens[cnt++] = next_tokens[0];
    ple_rows(tokens, tb, next_pos, 1, next_rows_, true);
    for(int k = 0; k < kDrafts; ++k) {
        if(!wait_stage(k + 2)) return;
        next_tokens[k + 1] = progress_->drafts[k];
        tokens[cnt++] = next_tokens[k + 1];
        ple_rows(tokens, tb, next_pos + k + 1, 1, next_rows_ + std::size_t(k + 1) * kPleHeads * 90, true);
    }
    ahead_.valid = true;
    ahead_.pos = next_pos;
    for(int t = 0; t < kMaxTokens; ++t) ahead_.tokens[t] = next_tokens[t];
}

void Net::spec_begin(int pos) {
    if(!mtp_loaded_) throw std::logic_error("speculative decoding needs the MTP block");
    const auto s = stream();
    if(!step_done_) FNL_CUDA(cudaEventCreateWithFlags(&step_done_, cudaEventDisableTiming));
    else FNL_CUDA(cudaEventSynchronize(step_done_));
    step_host_[0] = make_step(pos, 0);
    FNL_CUDA(cudaMemcpyAsync(step_, step_host_, sizeof(Step), cudaMemcpyHostToDevice, s));
    graph_blocks_ = index_bucket(pos + kMaxTokens, options_.capacity);
    ahead_.valid = false;
    progress_->stage = 0;
    accept_prompt(next_, step_, acc_, dinp_, s);
    progress_note(acc_, progress_, 1, s);
    spec_drafts();
    FNL_CUDA(cudaEventRecord(step_done_, s));
}

void Net::spec_begin_prefetch(const std::vector<int>& history, int pos) {
    const int base = std::max(0, pos - 2);
    int window[2 + 2 * kMaxTokens];
    for(int p = base; p <= pos; ++p) window[p - base] = history[p];
    // The prompt's last token acts as the round's only input: its successor is the bonus.
    int inputs[1] = {history[pos]};
    prefetch_next(window, base, pos - 1, inputs, 1);
}

}  // namespace fnl
