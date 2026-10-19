// Grouped NVFP4 GEMMs on CUTLASS SM120 block-scaled ptr-array collectives (CUTLASS v4.7.1 headers, retained in
// third_party/kernels/cutlass). Configuration: 128x128x128 tiles, 1x1x1 cluster, cooperative warp-specialized
// mainloop and ptr-array TMA epilogue, per-group alpha from device scalars, no source matrix.
#include "fp4_grouped.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace nmh {
namespace {
using namespace cute;

void check(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string("fp4_grouped ") + what + ": " + cudaGetErrorString(e));
}

template <class ElementD>
struct Grouped {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using ElementPair = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using Tile = Shape<_128, _128, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*, kAlignD, ElementD, LayoutD*,
        kAlignD, cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementPair, LayoutA*, 32, ElementPair,
        LayoutB*, 32, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideD = typename Kernel::InternalStrideD;
    using LayoutSFA = typename Main::InternalLayoutSFA;
    using LayoutSFB = typename Main::InternalLayoutSFB;
    using Cfg = typename Main::Sm1xxBlkScaledConfig;
    using Shape3 = typename ProblemShape::UnderlyingProblemShape;
    using EA = typename Main::ElementA;
    using ESF = typename Main::ElementSF;
};

// Sections of one call's metadata blob, each [max_groups] entries, 16-byte aligned.
struct Blob {
    std::size_t shapes, a, b, sfa, sfb, d, alpha, sa, sb, sd, lsfa, lsfb, bytes;
};
template <class G>
Blob blob_layout(int n) {
    Blob b{};
    std::size_t off = 0;
    auto take = [&](std::size_t per) {
        const std::size_t o = off;
        off += (per * n + 15) / 16 * 16;
        return o;
    };
    b.shapes = take(sizeof(typename G::Shape3));
    b.a = take(sizeof(void*));
    b.b = take(sizeof(void*));
    b.sfa = take(sizeof(void*));
    b.sfb = take(sizeof(void*));
    b.d = take(sizeof(void*));
    b.alpha = take(sizeof(void*));
    b.sa = take(sizeof(typename G::StrideA));
    b.sb = take(sizeof(typename G::StrideB));
    b.sd = take(sizeof(typename G::StrideD));
    b.lsfa = take(sizeof(typename G::LayoutSFA));
    b.lsfb = take(sizeof(typename G::LayoutSFB));
    b.bytes = off;
    return b;
}

template <class G>
void run_grouped(const std::vector<Fp4Group>& groups, int N, int K, uint8_t* host, uint8_t* dev, const Blob& L,
                 void* ws, std::size_t ws_bytes, int sms, cudaStream_t stream) {
    const int n = int(groups.size());
    auto* shapes = reinterpret_cast<typename G::Shape3*>(host + L.shapes);
    auto* pa = reinterpret_cast<const void**>(host + L.a);
    auto* pb = reinterpret_cast<const void**>(host + L.b);
    auto* psfa = reinterpret_cast<const void**>(host + L.sfa);
    auto* psfb = reinterpret_cast<const void**>(host + L.sfb);
    auto* pd = reinterpret_cast<void**>(host + L.d);
    auto* palpha = reinterpret_cast<const float**>(host + L.alpha);
    auto* sa = reinterpret_cast<typename G::StrideA*>(host + L.sa);
    auto* sb = reinterpret_cast<typename G::StrideB*>(host + L.sb);
    auto* sd = reinterpret_cast<typename G::StrideD*>(host + L.sd);
    auto* lsfa = reinterpret_cast<typename G::LayoutSFA*>(host + L.lsfa);
    auto* lsfb = reinterpret_cast<typename G::LayoutSFB*>(host + L.lsfb);
    for(int g = 0; g < n; ++g) {
        const Fp4Group& q = groups[g];
        shapes[g] = typename G::Shape3(q.M, N, K);
        pa[g] = q.A;
        pb[g] = q.B;
        psfa[g] = q.SFA;
        psfb[g] = q.SFB;
        pd[g] = q.D;
        palpha[g] = q.alpha;
        sa[g] = cutlass::make_cute_packed_stride(typename G::StrideA{}, make_shape(q.M, K, 1));
        sb[g] = cutlass::make_cute_packed_stride(typename G::StrideB{}, make_shape(N, K, 1));
        sd[g] = cutlass::make_cute_packed_stride(typename G::StrideD{}, make_shape(q.M, N, 1));
        lsfa[g] = G::Cfg::tile_atom_to_shape_SFA(make_shape(q.M, N, K, 1));
        lsfb[g] = G::Cfg::tile_atom_to_shape_SFB(make_shape(q.M, N, K, 1));
    }
    check(cudaMemcpyAsync(dev, host, L.bytes, cudaMemcpyHostToDevice, stream), "metadata copy");
    using ElementD = typename G::Kernel::ElementD;
    typename G::Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape = {n, reinterpret_cast<typename G::Shape3*>(dev + L.shapes), shapes};
    args.mainloop.ptr_A = reinterpret_cast<const typename G::EA**>(dev + L.a);
    args.mainloop.dA = reinterpret_cast<typename G::StrideA*>(dev + L.sa);
    args.mainloop.ptr_B = reinterpret_cast<const typename G::EA**>(dev + L.b);
    args.mainloop.dB = reinterpret_cast<typename G::StrideB*>(dev + L.sb);
    args.mainloop.ptr_SFA = reinterpret_cast<const typename G::ESF**>(dev + L.sfa);
    args.mainloop.layout_SFA = reinterpret_cast<typename G::LayoutSFA*>(dev + L.lsfa);
    args.mainloop.ptr_SFB = reinterpret_cast<const typename G::ESF**>(dev + L.sfb);
    args.mainloop.layout_SFB = reinterpret_cast<typename G::LayoutSFB*>(dev + L.lsfb);
    auto& f = args.epilogue.thread;
    f.alpha = 1.f;
    f.beta = 0.f;
    f.alpha_ptr = nullptr;
    f.beta_ptr = nullptr;
    f.alpha_ptr_array = reinterpret_cast<const float**>(dev + L.alpha);
    f.beta_ptr_array = nullptr;
    f.dAlpha = {_0{}, _0{}, 1};
    f.dBeta = {_0{}, _0{}, 0};
    args.epilogue.ptr_C = nullptr;
    args.epilogue.dC = nullptr;
    args.epilogue.ptr_D = reinterpret_cast<ElementD**>(dev + L.d);
    args.epilogue.dD = reinterpret_cast<typename G::StrideD*>(dev + L.sd);
    args.hw_info.device_id = 0;
    args.hw_info.sm_count = sms;
    typename G::Gemm gemm;
    if(gemm.can_implement(args) != cutlass::Status::kSuccess) throw std::runtime_error("fp4_grouped: can_implement");
    if(G::Gemm::get_workspace_size(args) > ws_bytes) throw std::runtime_error("fp4_grouped: workspace too small");
    if(gemm.initialize(args, ws, stream) != cutlass::Status::kSuccess) throw std::runtime_error("fp4_grouped: initialize");
    if(gemm.run(stream) != cutlass::Status::kSuccess) throw std::runtime_error("fp4_grouped: run");
}

}  // namespace

Fp4Grouped::Fp4Grouped(int max_groups, void* workspace, std::size_t workspace_bytes)
    : max_groups_(max_groups), workspace_(workspace), workspace_bytes_(workspace_bytes) {
    const Blob a = blob_layout<Grouped<cutlass::bfloat16_t>>(max_groups);
    const Blob b = blob_layout<Grouped<float>>(max_groups);
    blob_bytes_ = std::max(a.bytes, b.bytes);
    check(cudaMallocHost(&host_, blob_bytes_ * kRing), "host staging");
    check(cudaMalloc(&device_, blob_bytes_ * kRing), "device metadata");
    for(auto& e : done_) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    int dev = 0;
    check(cudaGetDevice(&dev), "device");
    check(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, dev), "sm count");
}

Fp4Grouped::~Fp4Grouped() {
    for(auto& e : done_) if(e) cudaEventDestroy(e);
    if(host_) cudaFreeHost(host_);
    if(device_) cudaFree(device_);
}

void Fp4Grouped::run(const std::vector<Fp4Group>& groups, int N, int K, bool bf16, cudaStream_t stream) {
    if(groups.empty()) return;
    if(int(groups.size()) > max_groups_) throw std::runtime_error("fp4_grouped: too many groups");
    const int r = next_;
    next_ = (next_ + 1) % kRing;
    // The staging slot's previous copy must have left the host buffer.
    check(cudaEventSynchronize(done_[r]), "staging wait");
    uint8_t* host = host_ + std::size_t(r) * blob_bytes_;
    uint8_t* dev = device_ + std::size_t(r) * blob_bytes_;
    if(bf16)
        run_grouped<Grouped<cutlass::bfloat16_t>>(groups, N, K, host, dev, blob_layout<Grouped<cutlass::bfloat16_t>>(max_groups_),
                                                  workspace_, workspace_bytes_, sms_, stream);
    else
        run_grouped<Grouped<float>>(groups, N, K, host, dev, blob_layout<Grouped<float>>(max_groups_), workspace_,
                                    workspace_bytes_, sms_, stream);
    check(cudaEventRecord(done_[r], stream), "staging event");
}

}  // namespace nmh
