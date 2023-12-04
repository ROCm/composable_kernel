#include <cstring>
#include <ostream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"

#include "reference_batched_gemm.hpp"
#include "reference_batched_softmax.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "fmha_fwd_epilogue.hpp"
#include "arg_parser.hpp"
#include <tuple>

using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;

//                                                 M0   N0  K0   N1  K1  K0L
// using FmhaShape = ck::tile_program::TileFmhaShape<128,  64, 64, 128, 64>;
// using FmhaShape = ck::tile_program::TileFmhaShape<128, 256, 32, 128, 32>;
using VLayout = ck::tensor_layout::gemm::RowMajor; // (bs, nhead) seqlen * hdim
// using VLayout = ck::tensor_layout::gemm::ColumnMajor; // (bs, nhead) hdim * seqlen

using FmhaBlockTileHdim64  = ck::Sequence<128, 64, 32, 64, 32, 64>;
using FmhaBlockTileHdim128 = ck::Sequence<128, 128, 32, 128, 32, 128>;
using FmhaBlockWarps       = ck::Sequence<4, 1, 1>;
using FmhaWarpTile         = ck::Sequence<32, 32, 16>;
using FmhaShapeHDim64      = ck::tile_program::TileFmhaShape<FmhaBlockTileHdim64,
                                                        FmhaBlockWarps,
                                                        FmhaWarpTile,
                                                        FmhaBlockWarps,
                                                        FmhaWarpTile,
                                                        3, // occupancy
                                                        VLayout>;
using FmhaShapeHDim128     = ck::tile_program::TileFmhaShape<FmhaBlockTileHdim128,
                                                         FmhaBlockWarps,
                                                         FmhaWarpTile,
                                                         FmhaBlockWarps,
                                                         FmhaWarpTile,
                                                         2, // occupancy
                                                         VLayout>;

using FmhaTilePartitionerHDim64  = FmhaFwdTilePartitioner<FmhaShapeHDim64>;
using FmhaTilePartitionerHDim128 = FmhaFwdTilePartitioner<FmhaShapeHDim128>;
using FmhaPipelineProblemHDim64 =
    ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                      KDataType,
                                                      VDataType,
                                                      SaccDataType,
                                                      SMPLComputeDataType,
                                                      PDataType,
                                                      OaccDataType,
                                                      ODataType,
                                                      256, // BlockSize
                                                      FmhaShapeHDim64>;
using FmhaPipelineProblemHDim128 =
    ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                      KDataType,
                                                      VDataType,
                                                      SaccDataType,
                                                      SMPLComputeDataType,
                                                      PDataType,
                                                      OaccDataType,
                                                      ODataType,
                                                      256, // BlockSize
                                                      FmhaShapeHDim128>;
// using FmhaPipeline        = ck::tile_program::block::BlockFmhaPipelineQKVS<FmhaPipelineProblem>;
using FmhaPipelineHDim64 =
    ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblemHDim64>;
using FmhaPipelineHDim128 =
    ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblemHDim128>;

using FmhaEpilogue     = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;
using FmhaKernelHDim64 = FmhaFwdKernel<FmhaTilePartitionerHDim64, FmhaPipelineHDim64, FmhaEpilogue>;
using FmhaKernelHDim128 =
    FmhaFwdKernel<FmhaTilePartitionerHDim128, FmhaPipelineHDim128, FmhaEpilogue>;

template <typename FmhaKernel>
float invoker_fmha_kernel(const void* q_ptr,
                          const void* k_ptr,
                          const void* v_ptr,
                          void* o_ptr,
                          ck::index_t batch,
                          ck::index_t nhead,
                          ck::index_t nhead_k,
                          ck::index_t seqlen_q,
                          ck::index_t seqlen_k,
                          ck::index_t hdim_q,
                          ck::index_t hdim_v,
                          float scale,
                          bool i_perm,
                          bool o_perm,
                          StreamConfig stream_config)
{
    dim3 kGridSize            = FmhaKernel::GridSize(batch, nhead, seqlen_q, hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    constexpr ck::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernel::VLayout, ck::tensor_layout::gemm::RowMajor>;

    assert(nhead % nhead_k == 0);
    // batch * nhead * seqlen * hdim or batch * seqlen * nhead * hdim
    auto kargs = FmhaKernel::MakeKargs(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        seqlen_q, // seqlen_q
        seqlen_k, // seqlen_k
        hdim_q,   // hdim_q
        hdim_v,   // hdim_v
        nhead / nhead_k,
        scale,
        i_perm ? hdim_q : nhead * hdim_q,   // stride_q
        i_perm ? hdim_q : nhead_k * hdim_q, // stride_k
        [&]() {
            if constexpr(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? seqlen_k : nhead_k * seqlen_k;
        }(),                                 // stride_v
        o_perm ? hdim_v : nhead * hdim_v,    // stride_o
        i_perm ? seqlen_q * hdim_q : hdim_q, // nhead_stride_q
        i_perm ? seqlen_k * hdim_q : hdim_q, // nhead_stride_k
        [&]() {
            if constexpr(is_v_rowmajor)
                return i_perm ? seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_k : seqlen_k;
        }(),                                 // nhead_stride_v
        o_perm ? seqlen_q * hdim_v : hdim_v, // nhead_stride_o
        nhead * seqlen_q * hdim_q,           // batch_stride_q
        nhead_k * seqlen_k * hdim_q,         // batch_stride_k
        nhead_k * hdim_v * seqlen_k,         // batch_stride_v
        nhead * seqlen_q * hdim_v);          // batch_stride_o

    float ave_time = launch_kernel<kBlockSize.x, kBlockPerCu>(stream_config,
                                                              FmhaKernel{},
                                                              kGridSize,
                                                              kBlockSize,
                                                              0,
                                                              kargs); // BatchStrideO
    return ave_time;
}

static inline int env_get_int(const char* var_name, int default_int)
{
    char* v = getenv(var_name);
    int r   = default_int;
    if(v)
        r = atoi(v);
    return r;
}

auto create_args(int argc, char* argv[])
{
    ArgParser arg_parser;
    arg_parser.insert("v", "1", "weather do cpu validation or not")
        .insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert("h_k",
                "0",
                "num of head, for k/v, 0 means equal to h\n"
                "if not equal to h, then this is GQA/MQA case")
        .insert("s", "3328", "seqlen_q")
        .insert("s_k", "0", "seqlen_k, 0 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "0", "head dim for v, 0 means equal to d")
        .insert("scale", "0", "scale factor. 0 means equal to 1/sqrt(seqlen)")
        .insert("iperm",
                "1",
                "permute input\n"
                "if true, will be b*h*s*d, else b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("init", "1", "init method. 0:random int, 1:random float, 2:trig float");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    int do_validation   = arg_parser.get_int("v");
    ck::index_t batch   = arg_parser.get_int("b");
    ck::index_t nhead   = arg_parser.get_int("h");
    ck::index_t nhead_k = arg_parser.get_int("h_k");
    if(nhead_k == 0)
        nhead_k = nhead;

    if(nhead % nhead_k != 0)
    {
        std::cout << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return -1;
    }

    ck::index_t seqlen_q = arg_parser.get_int("s");
    ck::index_t seqlen_k = arg_parser.get_int("s_k");
    if(seqlen_k == 0)
        seqlen_k = seqlen_q;
    ck::index_t hdim_q = arg_parser.get_int("d");
    ck::index_t hdim_v = arg_parser.get_int("d_v");
    if(hdim_v == 0)
        hdim_v = hdim_q;

    int i_perm = arg_parser.get_int("iperm"); // if true, will be batch * nhead * seqlen * hdim
    int o_perm = arg_parser.get_int("operm"); // if false, will be batch * seqlen * nhead * hdim

    float scale = arg_parser.get_float("scale");
    if(scale == .0f)
        scale = 1.0 / ck::math::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

    int init_method = arg_parser.get_int("init");

    int stream_warmup = env_get_int("CK_WARMUP", 5);
    int stream_repeat = env_get_int("CK_REPEAT", 20);

    StreamConfig stream_config{nullptr, true, 0, stream_warmup, stream_repeat};

    auto get_lengths = [&](int permute,
                           ck::index_t b /*batch*/,
                           ck::index_t h /*nhead*/,
                           ck::index_t s /*seqlen*/,
                           ck::index_t d /*hdim*/) {
        if(permute)
            return std::array<ck::index_t, 4>{b, h, s, d};
        else
            return std::array<ck::index_t, 4>{b, s, h, d};
    };

    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernelHDim64::VLayout, ck::tensor_layout::gemm::RowMajor>;

    // host verify
    Tensor<QDataType> q_host(get_lengths(i_perm, batch, nhead, seqlen_q, hdim_q));
    Tensor<KDataType> k_host(get_lengths(i_perm, batch, nhead_k, seqlen_k, hdim_q));
    Tensor<VDataType> v_host(is_v_rowmajor ? get_lengths(i_perm, batch, nhead_k, seqlen_k, hdim_v)
                                           : get_lengths(i_perm, batch, nhead_k, hdim_v, seqlen_k));
    Tensor<ODataType> o_host(get_lengths(o_perm, batch, nhead, seqlen_q, hdim_v));

    if(init_method == 0)
    {
        ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
        ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
        ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
    }
    else if(init_method == 1)
    {
        ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
        ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
        ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
    }
    else if(init_method == 2)
    {
        ck::utils::FillTrigValue<QDataType>{}(q_host);
        ck::utils::FillTrigValue<KDataType>{}(k_host);
        ck::utils::FillTrigValue<VDataType>{}(v_host);
    }

    DeviceMem q_buf(sizeof(QDataType) * q_host.GetElementSpaceSize());
    DeviceMem k_buf(sizeof(KDataType) * k_host.GetElementSpaceSize());
    DeviceMem v_buf(sizeof(VDataType) * v_host.GetElementSpaceSize());
    DeviceMem o_buf(sizeof(ODataType) * o_host.GetElementSpaceSize());

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());

    // clang-format off
    auto layout_str = [&](int permute){
        if (permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    // clang-format on

    std::cout << "b:" << batch << ", h:" << nhead << ", h_k:" << nhead_k << ", s:" << seqlen_q
              << ", s_k:" << seqlen_k << ", d:" << hdim_q << ", d_v:" << hdim_v
              << ", scale:" << scale << ", i:" << layout_str(i_perm) << ", o:" << layout_str(o_perm)
              << ", v:" << std::string(FmhaKernelHDim64::VLayout::name)[0] << std::flush;

    float ave_time = 0;
    if(hdim_q == hdim_v && hdim_q == 64)
        ave_time = invoker_fmha_kernel<FmhaKernelHDim64>(q_buf.GetDeviceBuffer(),
                                                         k_buf.GetDeviceBuffer(),
                                                         v_buf.GetDeviceBuffer(),
                                                         o_buf.GetDeviceBuffer(),
                                                         batch,
                                                         nhead,
                                                         nhead_k,
                                                         seqlen_q,
                                                         seqlen_k,
                                                         hdim_q,
                                                         hdim_v,
                                                         scale,
                                                         i_perm,
                                                         o_perm,
                                                         stream_config);
    else if(hdim_q == hdim_v && hdim_q == 128)
        ave_time = invoker_fmha_kernel<FmhaKernelHDim128>(q_buf.GetDeviceBuffer(),
                                                          k_buf.GetDeviceBuffer(),
                                                          v_buf.GetDeviceBuffer(),
                                                          o_buf.GetDeviceBuffer(),
                                                          batch,
                                                          nhead,
                                                          nhead_k,
                                                          seqlen_q,
                                                          seqlen_k,
                                                          hdim_q,
                                                          hdim_v,
                                                          scale,
                                                          i_perm,
                                                          o_perm,
                                                          stream_config);
    else
    {
        std::cout << "not support hdim, will not run" << std::endl;
        return -1;
    }

    std::size_t flop = std::size_t(2) * batch * nhead * seqlen_q * seqlen_k * hdim_q +
                       std::size_t(2) * batch * nhead * seqlen_q * hdim_v * seqlen_k;

    // TODO: MQA/GQA case nhead is smaller, do we need to change this formular?
    std::size_t num_btype = sizeof(QDataType) * batch * nhead * seqlen_q * hdim_q +
                            sizeof(KDataType) * batch * nhead * seqlen_k * hdim_q +
                            sizeof(VDataType) * batch * nhead * hdim_v * seqlen_k +
                            sizeof(ODataType) * batch * nhead * seqlen_q * hdim_v;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << ", " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::flush << std::endl;

    if(do_validation)
    {
        Tensor<QDataType> q_host_ref({batch * nhead, seqlen_q, hdim_q});
        Tensor<KDataType> k_host_ref(
            {batch * nhead, seqlen_k, hdim_q}); // NOTE: expand nhead the same as q
        const auto v_lengths = std::array<ck::index_t, 3>{batch * nhead, hdim_v, seqlen_k};
        const auto v_strides = is_v_rowmajor
                                   ? std::array<ck::index_t, 3>{hdim_v * seqlen_k, 1, hdim_v}
                                   : std::array<ck::index_t, 3>{hdim_v * seqlen_k, seqlen_k, 1};
        Tensor<VDataType> v_host_ref(v_lengths, v_strides);
        Tensor<ODataType> o_host_ref({batch * nhead, seqlen_q, hdim_v});
        Tensor<ODataType> o_host_result_ref(get_lengths(o_perm, batch, nhead, seqlen_q, hdim_v));

        Tensor<SMPLComputeDataType> s_host_ref({batch * nhead, seqlen_q, seqlen_k});
        Tensor<PDataType> p_host_ref({batch * nhead, seqlen_q, seqlen_k});

        ck::index_t nr = nhead / nhead_k;

#define EACH_R for(ck::index_t r = 0; r < nr; r++)
        // clang-format off
        // permute
        if(i_perm) q_host.ForEach([&](auto& self, auto i) { q_host_ref(i[0] * nhead + i[1], i[2], i[3]) = self(i); });
        else       q_host.ForEach([&](auto& self, auto i) { q_host_ref(i[0] * nhead + i[2], i[1], i[3]) = self(i); });

        if(i_perm) k_host.ForEach([&](auto& self, auto i) { EACH_R k_host_ref(i[0] * nhead + i[1] * nr + r, i[2], i[3]) = self(i); });
        else       k_host.ForEach([&](auto& self, auto i) { EACH_R k_host_ref(i[0] * nhead + i[2] * nr + r, i[1], i[3]) = self(i); });

        if constexpr (is_v_rowmajor) {
            //                              v_host ：b, h, s, d, v_host_ref : batch*hdim*seq
            if(i_perm) v_host.ForEach([&](auto& self, auto i) { EACH_R v_host_ref(i[0] * nhead + i[1] * nr + r, i[3], i[2]) = self(i); });
            //                              v_host : b, s, h, d, v_host_ref : batch*hdim*seq
            else       v_host.ForEach([&](auto& self, auto i) { EACH_R v_host_ref(i[0] * nhead + i[2] * nr + r, i[3], i[1]) = self(i); });
        }
        else {
            if(i_perm) v_host.ForEach([&](auto& self, auto i) { EACH_R v_host_ref(i[0] * nhead + i[1] * nr + r, i[2], i[3]) = self(i); });
            else       v_host.ForEach([&](auto& self, auto i) { EACH_R v_host_ref(i[0] * nhead + i[2] * nr + r, i[1], i[3]) = self(i); });
        }
#undef EACH_R

        // reference
        reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
            q_host_ref, k_host_ref, s_host_ref,
            [](const QDataType& x) { return x; },
            [](const KDataType& x) { return x; },
            [&scale](const SaccDataType& x) { return scale * x; });
        reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref,
                                                                                       p_host_ref);
        reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
            p_host_ref, v_host_ref, o_host_ref);

        // permute
        if(o_perm) o_host_result_ref.ForEach([&](auto& self, auto i) { self(i) = o_host_ref(i[0] * nhead + i[1], i[2], i[3]); });
        else       o_host_result_ref.ForEach([&](auto& self, auto i) { self(i) = o_host_ref(i[0] * nhead + i[2], i[1], i[3]); });
        // clang-format on

        o_buf.FromDevice(o_host.mData.data());
        return !ck::utils::check_err(o_host, o_host_result_ref);
    }
    else
    {
        return 0;
    }
}
