// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <array>
#include <cstring>
#include <numeric>
#include <ostream>
#include <tuple>

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
#include "ck/library/utility/literals.hpp"

#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_masking_specialization.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"
#include "ck/tile_program/tile/tile_fmha_traits.hpp"

#include "reference_batched_elementwise.hpp"
#include "reference_batched_gemm.hpp"
#include "reference_batched_masking.hpp"
#include "reference_batched_softmax.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "fmha_fwd_epilogue.hpp"
#include "fmha_utils.hpp"
#include "arg_parser.hpp"

#if 1
using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using BiasDataType        = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;
#else
using QDataType           = ck::bhalf_t;
using KDataType           = ck::bhalf_t;
using VDataType           = ck::bhalf_t;
using BiasDataType        = ck::bhalf_t;
using SaccDataType        = float;       // data type for first gemm accumulation
using SMPLComputeDataType = float;       // data type for reduction, softmax
using PDataType           = ck::bhalf_t; // data type for A matrix of second gemm
using OaccDataType        = float;       // data type for second gemm accumulation
using ODataType           = ck::bhalf_t;
#endif

//                                                 M0   N0  K0   N1  K1  K0L
// using FmhaShape = ck::tile_program::TileFmhaShape<128,  64, 64, 128, 64>;
// using FmhaShape = ck::tile_program::TileFmhaShape<128, 256, 32, 128, 32>;
using VLayout = ck::tensor_layout::gemm::RowMajor; // (bs, nhead) seqlen * hdim
// using VLayout = ck::tensor_layout::gemm::ColumnMajor; // (bs, nhead) hdim * seqlen

template <ck::index_t HDim>
struct FmhaBlockTile;

template <>
struct FmhaBlockTile</* HDim = */ 64> : ck::Sequence<128, 64, 32, 64, 32, 64>
{
};
template <>
struct FmhaBlockTile</* HDim = */ 128> : ck::Sequence<128, 128, 32, 128, 32, 128>
{
};
using FmhaBlockWarps = ck::Sequence<4, 1, 1>;
using FmhaWarpTile   = ck::Sequence<32, 32, 16>;

template <ck::index_t HDim>
struct FmhaShape;

template <>
struct FmhaShape</* HDim = */ 64> : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 64>,
                                                                    FmhaBlockWarps,
                                                                    FmhaWarpTile,
                                                                    FmhaBlockWarps,
                                                                    FmhaWarpTile,
                                                                    VLayout>
{
};
template <>
struct FmhaShape</* HDim = */ 128>
    : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 128>,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      VLayout>
{
};

// using FmhaMask = ck::tile_program::block::MaskUpperTriangleFromTopLeftPredicate;
// using FmhaMask = ck::tile_program::block::MaskUpperTriangleFromBottomRightPredicate;
using FmhaMask = ck::tile_program::block::MaskDisabledPredicate;

inline constexpr bool kM0NeedPadding   = false;
inline constexpr bool kN0K1NeedPadding = false;
template <ck::index_t HDim, bool kHasBias>
using FmhaTraits = ck::tile_program::TileFmhaTraits<kM0NeedPadding,
                                                    kN0K1NeedPadding,
                                                    kHasBias,
                                                    HDim == 64 ? /* occupancy = */ 3 : 2>;

template <ck::index_t HDim>
using FmhaTilePartitioner = FmhaFwdTilePartitioner<FmhaShape<HDim>>;

template <ck::index_t HDim, bool kIsGroupMode, bool kHasBias>
using FmhaPipelineProblem =
    ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                      KDataType,
                                                      VDataType,
                                                      SaccDataType,
                                                      SMPLComputeDataType,
                                                      BiasDataType,
                                                      PDataType,
                                                      OaccDataType,
                                                      ODataType,
                                                      /* BlockSize = */ 256,
                                                      FmhaShape<HDim>,
                                                      kIsGroupMode,
                                                      FmhaMask,
                                                      FmhaTraits<HDim, kHasBias>>;

template <ck::index_t HDim, bool kIsGroupMode, bool kHasBias>
using FmhaPipeline = ck::tile_program::block::BlockFmhaPipelineQRKSVSAsync<
    FmhaPipelineProblem<HDim, kIsGroupMode, kHasBias>>;

using FmhaEpilogue = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;

template <ck::index_t HDim, bool kIsGroupMode, bool kHasBias>
using FmhaKernel = FmhaFwdKernel<FmhaTilePartitioner<HDim>,
                                 FmhaPipeline<HDim, kIsGroupMode, kHasBias>,
                                 FmhaEpilogue>;

template <typename FmhaKernel_>
float invoker_fmha_kernel(const void* q_ptr,
                          const void* k_ptr,
                          const void* v_ptr,
                          const void* bias_ptr,
                          void* o_ptr,
                          const void* seqstart_q_ptr,
                          const void* seqstart_k_ptr,
                          const void* seqlen_k_ptr,
                          ck::index_t batch,
                          ck::index_t nhead,
                          ck::index_t nhead_k,
                          ck::index_t seqlen_q,
                          ck::index_t seqlen_k,
                          ck::index_t hdim_q,
                          ck::index_t hdim_v,
                          ck::index_t max_seqlen_q,
                          float scale,
                          bool i_perm,
                          bool o_perm,
                          StreamConfig stream_config)
{
    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernel_::VLayout, ck::tensor_layout::gemm::RowMajor>;

    assert(nhead % nhead_k == 0);
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' & 'nhead_stride_bias'
    ///       are 0.
    // setup stride_* arguments
    const ck::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
    const ck::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
    const ck::index_t stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? hdim_v : nhead_k * hdim_v;
        else
            return i_perm ? seqlen_k : nhead_k * seqlen_k;
    }();
    const ck::index_t stride_bias = (i_perm ? seqlen_k : 1 * seqlen_k);
    const ck::index_t stride_o    = (o_perm ? hdim_v : nhead * hdim_v);
    // setup nhead_stride_* arguments
    const ck::index_t nhead_stride_q = (i_perm ? seqlen_q * hdim_q : hdim_q);
    const ck::index_t nhead_stride_k = (i_perm ? seqlen_k * hdim_q : hdim_q);
    const ck::index_t nhead_stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? seqlen_k * hdim_v : hdim_v;
        else
            return i_perm ? hdim_v * seqlen_k : seqlen_k;
    }();
    const ck::index_t nhead_stride_bias = (i_perm ? 0 * seqlen_q * seqlen_k : 0 * seqlen_k);
    const ck::index_t nhead_stride_o    = (o_perm ? seqlen_q * hdim_v : hdim_v);
    // setup batch_stride_* arguments
    const ck::index_t batch_stride_q    = (nhead * seqlen_q * hdim_q);
    const ck::index_t batch_stride_k    = (nhead_k * seqlen_k * hdim_q);
    const ck::index_t batch_stride_v    = (nhead_k * hdim_v * seqlen_k);
    const ck::index_t batch_stride_bias = (0 * nhead * seqlen_q * seqlen_k);
    const ck::index_t batch_stride_o    = (nhead * seqlen_q * hdim_v);

    const auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel_::kIsGroupMode)
        {
            return FmhaKernel_::MakeKargs(q_ptr,
                                          k_ptr,
                                          v_ptr,
                                          bias_ptr,
                                          o_ptr,
                                          seqstart_q_ptr,
                                          seqstart_k_ptr,
                                          seqlen_k_ptr,
                                          hdim_q,
                                          hdim_v,
                                          nhead / nhead_k,
                                          scale,
                                          stride_q,
                                          stride_k,
                                          stride_v,
                                          stride_bias,
                                          stride_o,
                                          nhead_stride_q,
                                          nhead_stride_k,
                                          nhead_stride_v,
                                          nhead_stride_bias,
                                          nhead_stride_o);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel_::MakeKargs(q_ptr,
                                          k_ptr,
                                          v_ptr,
                                          bias_ptr,
                                          o_ptr,
                                          seqlen_q,
                                          seqlen_k,
                                          hdim_q,
                                          hdim_v,
                                          nhead / nhead_k,
                                          scale,
                                          stride_q,
                                          stride_k,
                                          stride_v,
                                          stride_bias,
                                          stride_o,
                                          nhead_stride_q,
                                          nhead_stride_k,
                                          nhead_stride_v,
                                          nhead_stride_bias,
                                          nhead_stride_o,
                                          batch_stride_q,
                                          batch_stride_k,
                                          batch_stride_v,
                                          batch_stride_bias,
                                          batch_stride_o);
        }
    }();

    const dim3 kGridSize      = FmhaKernel_::GridSize(batch, nhead, max_seqlen_q, hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel_::BlockSize();

    constexpr ck::index_t kBlockPerCu = FmhaKernel_::kBlockPerCu;

    return launch_kernel<kBlockSize.x, kBlockPerCu>(stream_config,
                                                    FmhaKernel_{},
                                                    kGridSize,
                                                    kBlockSize,
                                                    0,
                                                    kargs); // BatchStrideO
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
        .insert("mode", "0", "kernel mode. 0:batch, 1:group")
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
        .insert("bias", "0", "add bias or not")
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
    auto mode           = static_cast<Mode>(arg_parser.get_uint32("mode"));
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

    bool use_bias = arg_parser.get_uint32("bias");

    int init_method = arg_parser.get_int("init");

    int stream_warmup = env_get_int("CK_WARMUP", 5);
    int stream_repeat = env_get_int("CK_REPEAT", 20);

    StreamConfig stream_config{nullptr, true, 0, stream_warmup, stream_repeat};

    const std::vector<int32_t> seqstart_q_host = generate_seqstarts(mode, batch, seqlen_q);
    const std::vector<int32_t> seqstart_k_host = generate_seqstarts(mode, batch, seqlen_k);

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    {
        for(ck::index_t wb = 0; wb < batch; ++wb)
        {
            const int32_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            const int32_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

            if(max_seqlen_q < real_seqlen_q)
            {
                max_seqlen_q = real_seqlen_q;
            }

            using namespace ck::literals;

            flop += nhead * (2_uz * real_seqlen_q * real_seqlen_k * hdim_q +
                             2_uz * real_seqlen_q * hdim_v * real_seqlen_k);

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VDataType) * hdim_v * real_seqlen_k +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v);
        }
    }

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

    constexpr bool is_v_rowmajor = ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>;

    // host memory for storing all the tensor elements
    const ck::index_t shape_batch    = (mode == Mode::Batch ? batch : 1);
    const ck::index_t shape_seqlen_q = (mode == Mode::Batch ? seqlen_q : seqstart_q_host.back());
    const ck::index_t shape_seqlen_k = (mode == Mode::Batch ? seqlen_k : seqstart_k_host.back());

    Tensor<QDataType> q_host(get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    Tensor<KDataType> k_host(get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    Tensor<VDataType> v_host(
        is_v_rowmajor ? get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v)
                      : get_lengths(i_perm, shape_batch, nhead_k, hdim_v, shape_seqlen_k));
    // use bias shape = [1, 1, shape_seqlen_q, shape_seqlen_k]. if use_bias=false, the bias_host
    // will not be used for verification at all (but will be copied to device anyway).
    Tensor<KDataType> bias_host(
        use_bias ? get_lengths(i_perm, 1, 1, shape_seqlen_q, shape_seqlen_k)
                 : std::array<ck::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    Tensor<ODataType> o_host(get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));

    if(init_method == 0)
    {
        ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
        ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
        ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
        ck::utils::FillUniformDistributionIntegerValue<BiasDataType>{-2.f, 2.f}(bias_host);
    }
    else if(init_method == 1)
    {
        ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
        ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
        ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
        ck::utils::FillUniformDistribution<BiasDataType>{0.f, 1.f}(bias_host);
    }
    else if(init_method == 2)
    {
        ck::utils::FillTrigValue<QDataType>{}(q_host);
        ck::utils::FillTrigValue<KDataType>{}(k_host);
        ck::utils::FillTrigValue<VDataType>{}(v_host);
        ck::utils::FillTrigValue<BiasDataType>{}(bias_host);
    }

    DeviceMem q_buf(q_host.GetElementSpaceSizeInBytes());
    DeviceMem k_buf(k_host.GetElementSpaceSizeInBytes());
    DeviceMem v_buf(v_host.GetElementSpaceSizeInBytes());
    DeviceMem bias_buf(bias_host.GetElementSpaceSizeInBytes());
    DeviceMem o_buf(o_host.GetElementSpaceSizeInBytes());
    DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    bias_buf.ToDevice(bias_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());

    // clang-format off
    auto layout_str = [&](int permute){
        if (permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    // clang-format on

    std::cout << "[" << mode << "] b:" << batch << ", h:" << nhead << ", h_k:" << nhead_k
              << ", s:" << seqlen_q << ", s_k:" << seqlen_k << ", d:" << hdim_q
              << ", d_v:" << hdim_v << ", scale:" << scale << ", i:" << layout_str(i_perm)
              << ", o:" << layout_str(o_perm) << ", bias:" << use_bias
              << ", v:" << std::string(VLayout::name)[0] << std::flush;

    float ave_time = 0;
    if(hdim_q == hdim_v && hdim_q == 64)
    {
        BOOL_SWITCH_2(mode == Mode::Group, kIsGroupMode, use_bias, kHasBias, [&] {
            using Kernel = FmhaKernel</* HDim = */ 64, kIsGroupMode, kHasBias>;

            ave_time = invoker_fmha_kernel<Kernel>(q_buf.GetDeviceBuffer(),
                                                   k_buf.GetDeviceBuffer(),
                                                   v_buf.GetDeviceBuffer(),
                                                   bias_buf.GetDeviceBuffer(),
                                                   o_buf.GetDeviceBuffer(),
                                                   seqstart_q.GetDeviceBuffer(),
                                                   seqstart_k.GetDeviceBuffer(),
                                                   nullptr,
                                                   batch,
                                                   nhead,
                                                   nhead_k,
                                                   shape_seqlen_q,
                                                   shape_seqlen_k,
                                                   hdim_q,
                                                   hdim_v,
                                                   max_seqlen_q,
                                                   scale,
                                                   i_perm,
                                                   o_perm,
                                                   stream_config);
        });
    }
    else if(hdim_q == hdim_v && hdim_q == 128)
    {
        BOOL_SWITCH_2(mode == Mode::Group, kIsGroupMode, use_bias, kHasBias, [&] {
            using Kernel = FmhaKernel</* HDim = */ 128, kIsGroupMode, kHasBias>;

            ave_time = invoker_fmha_kernel<Kernel>(q_buf.GetDeviceBuffer(),
                                                   k_buf.GetDeviceBuffer(),
                                                   v_buf.GetDeviceBuffer(),
                                                   bias_buf.GetDeviceBuffer(),
                                                   o_buf.GetDeviceBuffer(),
                                                   seqstart_q.GetDeviceBuffer(),
                                                   seqstart_k.GetDeviceBuffer(),
                                                   nullptr,
                                                   batch,
                                                   nhead,
                                                   nhead_k,
                                                   shape_seqlen_q,
                                                   shape_seqlen_k,
                                                   hdim_q,
                                                   hdim_v,
                                                   max_seqlen_q,
                                                   scale,
                                                   i_perm,
                                                   o_perm,
                                                   stream_config);
        });
    }
    else
    {
        std::cerr << "not support hdim, will not run" << std::endl;
        return -1;
    }

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
              << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2) << gb_per_sec
              << " GB/s" << std::flush << std::endl;

    if(do_validation)
    {
        o_buf.FromDevice(o_host.data());

        for(ck::index_t wb = 0; wb < batch; ++wb)
        {
            const ck::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            const ck::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

            // adjust matrix index according to the mode
            const ck::index_t b            = (mode == Mode::Batch ? wb : 0);
            const ck::index_t query_offset = (mode == Mode::Batch ? 0 : seqstart_q_host[wb]);
            const ck::index_t key_offset   = (mode == Mode::Batch ? 0 : seqstart_k_host[wb]);

            const auto v_host_ref_lengths =
                std::array<ck::index_t, 3>{nhead, hdim_v, real_seqlen_k};
            const auto v_host_ref_strides =
                is_v_rowmajor
                    ? std::array<ck::index_t, 3>{hdim_v * real_seqlen_k, 1, hdim_v}
                    : std::array<ck::index_t, 3>{hdim_v * real_seqlen_k, real_seqlen_k, 1};

            Tensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q});
            Tensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q});
            Tensor<VDataType> v_host_ref(v_host_ref_lengths, v_host_ref_strides);
            Tensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v});

            Tensor<SMPLComputeDataType> s_host_ref({nhead, real_seqlen_q, real_seqlen_k});
            Tensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});

            ck::index_t nr = nhead / nhead_k;

            // clang-format off
            // permute
            if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[0], i[1] + query_offset, i[2]); });
            else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[1] + query_offset, i[0], i[2]); });

            if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[0] / nr, i[1] + key_offset, i[2]); });
            else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[1] + key_offset, i[0] / nr, i[2]); });

            if constexpr (is_v_rowmajor) {
                //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d] 
                if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[2] + key_offset, i[1]); });
                //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
                else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[2] + key_offset, i[0] / nr, i[1]); });
            }
            else {
                if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[1], i[2] + key_offset); });
                else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[1], i[0] / nr, i[2] + key_offset); });
            }

            // reference
            reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
                q_host_ref, k_host_ref, s_host_ref,
                ck::identity{}, ck::identity{},
                [&](SaccDataType x) { return scale * x; });

            if(use_bias)
            {
                Tensor<BiasDataType> bias_host_ref({1, real_seqlen_q, real_seqlen_k});
                if(i_perm)
                    bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, 0, i[1] + query_offset, i[2] + key_offset); });
                else
                    bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, i[1] + query_offset, 0, i[2] + key_offset); });
                
                // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q, real_seqlen_k]
                reference_batched_elementwise<SMPLComputeDataType, BiasDataType, SMPLComputeDataType, SMPLComputeDataType>(
                    s_host_ref, bias_host_ref, s_host_ref);
            }

            reference_batched_masking<SaccDataType, FmhaMask>(s_host_ref);
            reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref, p_host_ref);
            reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(p_host_ref, v_host_ref, o_host_ref);
            
            Tensor<ODataType> o_host_result({nhead, real_seqlen_q, hdim_v});
            // permute
            if(o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[0], idx[1] + query_offset, idx[2]); });
            else       o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[1] + query_offset, idx[0], idx[2]); });
            // clang-format on

            if(!ck::utils::check_err(o_host_result, o_host_ref))
            {
                std::cerr << "mismatch found at batch: " << wb << std::endl
                          << "\tseqlen_q: " << real_seqlen_q << std::endl
                          << "\tseqlen_k: " << real_seqlen_k << std::endl
                          << "\tseqstart_q: " << seqstart_q_host << std::endl
                          << "\tseqstart_k: " << seqstart_k_host << std::endl;

                return -1;
            }
        }
    }
    else
    {
        return 0;
    }
}
