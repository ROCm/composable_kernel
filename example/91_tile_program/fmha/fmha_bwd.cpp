// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <array>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/common_header.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "common/arg_parser.hpp"
#include "fmha_bwd.hpp"
#include "mask.hpp"
#include "reference/reference_batched_elementwise.hpp"
#include "reference/reference_batched_gemm.hpp"
#include "reference/reference_batched_masking.hpp"
#include "reference/reference_batched_softmax.hpp"
#include "utils.hpp"

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
        .insert("prec", "fp16", "data type. fp16 or bf16")
        .insert("mask",
                "0",
                "0: no mask, 1: top-left, 2:bottom-right\n"
                "'t:l,r', top-left local-attn with left right size\n"
                "'b:l,r', bottom-r local-attn with left right size\n"
                "'g:y,x', generic attention mask coordinate with y/x size\n")
        .insert("init", "1", "init method. 0:random int, 1:random float, 2:trig float");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <ck::index_t HDim_, typename DataType_>
struct fmha_bwd_dot_do_o_kernel_invoker
{
    static constexpr ck::index_t HDim = HDim_;
    using DataType                    = DataType_;
    // this arg is used to select kernel.
    // args that may passed as karg shoule use operator()
    mode_enum mode;

    fmha_bwd_dot_do_o_kernel_invoker(mode_enum mode_) : mode(mode_) {}

    template <typename... Args>
    float operator()(const StreamConfig& stream, Args&&... args)
    {
        float ave_time;
        BOOL_SWITCH(mode == mode_enum::group, kIsGroupMode, [&] {
            using Kernel = FmhaBwdOGradDotOKernelSelector<HDim, DataType, kIsGroupMode>;

            auto [kargs, grids] =
                fmha_bwd_dot_do_o_create_kargs_and_grids<Kernel>(std::forward<Args>(args)...);
            ave_time = fmha_bwd_dot_do_o_run<Kernel>(stream, kargs, grids);
        });
        return ave_time;
    }
};

template <ck::index_t HDim_, typename DataType_>
struct fmha_bwd_kernel_invoker
{
    static constexpr ck::index_t HDim = HDim_;
    using DataType                    = DataType_;
    // these args are used to select kernel.
    // args that may passed as karg shoule use operator()
    mode_enum mode;
    bool use_bias;
    mask_info mask;

    fmha_bwd_kernel_invoker(mode_enum mode_, bool use_bias_, mask_info mask_)
        : mode(mode_), use_bias(use_bias_), mask(mask_)
    {
    }

    template <typename... Args>
    float operator()(const StreamConfig& stream, Args&&... args)
    {
        float ave_time;
        BOOL_SWITCH_2(mode == mode_enum::group, kIsGroupMode, use_bias, kHasBias, [&] {
            if(mask.type == mask_enum::no_mask)
            {
                using FmhaMask = FmhaMasks::NoMask;
                using Kernel =
                    FmhaBwdKernelSelector<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>;

                auto [kargs, grids] =
                    fmha_bwd_create_kargs_and_grids<Kernel>(std::forward<Args>(args)...);
                ave_time = fmha_bwd_run<Kernel>(stream, kargs, grids);
            }
            else
            {
                BOOL_SWITCH(mask.type == mask_enum::window_generic, kIsLocal, [&]() {
                    using FmhaMask = ck::tile_program::block::GenericAttentionMask<true, kIsLocal>;
                    using Kernel =
                        FmhaBwdKernelSelector<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>;

                    auto [kargs, grids] =
                        fmha_bwd_create_kargs_and_grids<Kernel>(std::forward<Args>(args)...);
                    ave_time = fmha_bwd_run<Kernel>(stream, kargs, grids);
                });
            }
        });
        return ave_time;
    }
};

// different threshold for different dtype
template <typename DataType>
auto get_elimit(int /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck::bhalf_t>(int init_method)
{
    if(init_method == 0)
    {
        double rtol = 1e-2;
        double atol = 1e-2;
        return ck::make_tuple(rtol, atol);
    }
    else
    {
        double rtol = 1e-2;
        double atol = 1e-2;
        return ck::make_tuple(rtol, atol);
    }
}

template <typename DataType>
bool run(const ArgParser& arg_parser)
{
    int do_validation   = arg_parser.get_int("v");
    auto mode           = static_cast<mode_enum>(arg_parser.get_uint32("mode"));
    ck::index_t batch   = arg_parser.get_int("b");
    ck::index_t nhead   = arg_parser.get_int("h");
    ck::index_t nhead_k = arg_parser.get_int("h_k");
    if(nhead_k == 0)
        nhead_k = nhead;

    if(nhead % nhead_k != 0)
    {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return false;
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
        scale = 1.0 / ck::math::sqrt(static_cast<float>(hdim_q));

    bool use_bias = arg_parser.get_uint32("bias");

    mask_info mask = decode_mask_info(arg_parser.get_str("mask"), seqlen_q, seqlen_k);

    int init_method = arg_parser.get_int("init");

    int stream_warmup = env_get_int("CK_WARMUP", 5);
    int stream_repeat = env_get_int("CK_REPEAT", 20);

    StreamConfig stream_config{nullptr, true, 0, stream_warmup, stream_repeat};
    StreamConfig stream_vconfig{nullptr, false, 0, 0, 0};

    const auto [seqlens_q, seqstart_q_host] = generate_seqlens_seqstarts_q(mode, batch, seqlen_q);
    const std::vector<int32_t> seqstart_k_host =
        generate_seqstarts_k(mode, batch, seqlen_k, seqlens_q, seqlen_q);

    using TypeConfig = FmhaBwdTypeConfig<DataType>;

    using QDataType    = typename TypeConfig::QDataType;
    using KDataType    = typename TypeConfig::KDataType;
    using VDataType    = typename TypeConfig::VDataType;
    using GemmDataType = typename TypeConfig::GemmDataType;
    using BiasDataType = typename TypeConfig::BiasDataType;
    using LSEDataType  = typename TypeConfig::LSEDataType;
    using AccDataType  = typename TypeConfig::AccDataType;
    using DDataType    = typename TypeConfig::DDataType;
    // using ZDataType        = typename TypeConfig::ZDataType;
    using ODataType        = typename TypeConfig::ODataType;
    using OGradDataType    = typename TypeConfig::OGradDataType;
    using QGradDataType    = typename TypeConfig::QGradDataType;
    using KGradDataType    = typename TypeConfig::KGradDataType;
    using VGradDataType    = typename TypeConfig::VGradDataType;
    using BiasGradDataType = typename TypeConfig::BiasGradDataType;

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    auto max_seqlen_k =
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

            if(max_seqlen_k < real_seqlen_k)
            {
                max_seqlen_k = real_seqlen_k;
            }

            using namespace ck::literals;

            flop += nhead *
                    (3_uz * 2_uz * real_seqlen_q * real_seqlen_k * hdim_q + // Q@K/dS^T@Q^T/dS@K^T
                     2_uz * 2_uz * real_seqlen_q * real_seqlen_k * hdim_v); // dO@V/P^T@dO^T

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VDataType) * real_seqlen_k * hdim_v +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v +
                                 sizeof(OGradDataType) * real_seqlen_q * hdim_v +
                                 sizeof(QGradDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KGradDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VGradDataType) * real_seqlen_k * hdim_v +
                                 sizeof(LSEDataType) * real_seqlen_q);
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

    // host memory for storing all the tensor elements
    const ck::index_t shape_batch = (mode == mode_enum::batch ? batch : 1);
    const ck::index_t shape_seqlen_q =
        (mode == mode_enum::batch ? seqlen_q : seqstart_q_host.back());
    const ck::index_t shape_seqlen_k =
        (mode == mode_enum::batch ? seqlen_k : seqstart_k_host.back());

    Tensor<QDataType> q_host(get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    Tensor<KDataType> k_host(get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    Tensor<VDataType> v_host(get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v));
    // use bias shape = [1, 1, shape_seqlen_q, shape_seqlen_k]. if use_bias=false, the bias_host
    // will not be used for verification at all (but will be copied to device anyway).
    Tensor<BiasDataType> bias_host(
        use_bias ? get_lengths(i_perm, 1, 1, shape_seqlen_q, shape_seqlen_k)
                 : std::array<ck::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    Tensor<ODataType> o_host(get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));
    Tensor<LSEDataType> lse_host(std::array<ck::index_t, 3>{shape_batch, nhead, shape_seqlen_q});
    Tensor<DDataType> d_host(std::array<ck::index_t, 3>{shape_batch, nhead, shape_seqlen_q});
    // Tensor<ZDataType> z_host(
    //     std::array<ck::index_t, 4>{shape_batch, nhead, shape_seqlen_q, shape_seqlen_k});
    Tensor<QGradDataType> dq_host(get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    Tensor<KGradDataType> dk_host(get_lengths(i_perm, shape_batch, nhead, shape_seqlen_k, hdim_q));
    Tensor<VGradDataType> dv_host(get_lengths(i_perm, shape_batch, nhead, shape_seqlen_k, hdim_v));
    Tensor<OGradDataType> do_host(get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));
    Tensor<BiasGradDataType> dbias_host(
        use_bias ? get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, shape_seqlen_k)
                 : std::array<ck::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);

    if(init_method == 0)
    {
        ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
        ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
        ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
        ck::utils::FillUniformDistributionIntegerValue<BiasDataType>{-2.f, 2.f}(bias_host);
        ck::utils::FillUniformDistributionIntegerValue<OGradDataType>{-2.f, 2.f}(do_host);
    }
    else if(init_method == 1)
    {
        ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
        ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
        ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
        ck::utils::FillUniformDistribution<BiasDataType>{0.f, 1.f}(bias_host);
        ck::utils::FillUniformDistribution<OGradDataType>{-.5f, .5f}(do_host);
    }
    else if(init_method == 2)
    {
        ck::utils::FillTrigValue<QDataType>{}(q_host);
        ck::utils::FillTrigValue<KDataType>{}(k_host);
        ck::utils::FillTrigValue<VDataType>{}(v_host);
        ck::utils::FillTrigValue<BiasDataType>{}(bias_host);
        ck::utils::FillTrigValue<OGradDataType>{}(do_host);
    }

    DeviceMem q_buf(q_host.GetElementSpaceSizeInBytes());
    DeviceMem k_buf(k_host.GetElementSpaceSizeInBytes());
    DeviceMem v_buf(v_host.GetElementSpaceSizeInBytes());
    DeviceMem bias_buf(bias_host.GetElementSpaceSizeInBytes());
    DeviceMem o_buf(o_host.GetElementSpaceSizeInBytes());
    DeviceMem lse_buf(sizeof(LSEDataType) * lse_host.GetElementSpaceSize());
    DeviceMem d_buf(sizeof(DDataType) * d_host.GetElementSpaceSize());
    // DeviceMem z_buf(sizeof(ZDataType) * z_host.GetElementSpaceSize());
    DeviceMem dq_buf(sizeof(QGradDataType) * dq_host.GetElementSpaceSize());
    DeviceMem dk_buf(sizeof(KGradDataType) * dk_host.GetElementSpaceSize());
    DeviceMem dv_buf(sizeof(VGradDataType) * dv_host.GetElementSpaceSize());
    DeviceMem do_buf(sizeof(OGradDataType) * do_host.GetElementSpaceSize());
    DeviceMem dbias_buf(sizeof(BiasGradDataType) * dbias_host.GetElementSpaceSize());
    DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    bias_buf.ToDevice(bias_host.data());
    do_buf.ToDevice(do_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());

    // clang-format off
    auto layout_str = [&](int permute){
        if (permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    auto io_layout = [&](int iperm_, int operm_) {
        if (iperm_ == operm_) return layout_str(iperm_);
        else return layout_str(iperm_) + std::string("-") + layout_str(operm_);
    };
    // clang-format on
    const std::string prec = arg_parser.get_str("prec");

    std::cout << "[" << prec << "|" << mode << "|" << io_layout(i_perm, o_perm) << "] b:" << batch
              << ", h:" << nhead << "/" << nhead_k << ", s:" << seqlen_q << "/" << seqlen_k
              << ", d:" << hdim_q << "/" << hdim_v << ", scale:" << scale << ", bias:" << use_bias
              << ", mask:" << mask << std::flush;

#define INVOKE_FMHA_BWD_DOT_DO_O_KERNEL(hdim_)                                            \
    fmha_bwd_dot_do_o_kernel_invoker<hdim_, DataType>{mode}(stream_config,                \
                                                            o_buf.GetDeviceBuffer(),      \
                                                            do_buf.GetDeviceBuffer(),     \
                                                            d_buf.GetDeviceBuffer(),      \
                                                            seqstart_q.GetDeviceBuffer(), \
                                                            batch,                        \
                                                            nhead,                        \
                                                            shape_seqlen_q,               \
                                                            hdim_v,                       \
                                                            max_seqlen_q,                 \
                                                            o_perm)

#define INVOKE_FMHA_BWD_KERNEL(hdim_)                                                            \
    fmha_bwd_kernel_invoker<hdim_, DataType>{mode, use_bias, mask}(stream_config,                \
                                                                   q_buf.GetDeviceBuffer(),      \
                                                                   k_buf.GetDeviceBuffer(),      \
                                                                   v_buf.GetDeviceBuffer(),      \
                                                                   bias_buf.GetDeviceBuffer(),   \
                                                                   lse_buf.GetDeviceBuffer(),    \
                                                                   do_buf.GetDeviceBuffer(),     \
                                                                   d_buf.GetDeviceBuffer(),      \
                                                                   dq_buf.GetDeviceBuffer(),     \
                                                                   dk_buf.GetDeviceBuffer(),     \
                                                                   dv_buf.GetDeviceBuffer(),     \
                                                                   dbias_buf.GetDeviceBuffer(),  \
                                                                   seqstart_q.GetDeviceBuffer(), \
                                                                   seqstart_k.GetDeviceBuffer(), \
                                                                   nullptr,                      \
                                                                   batch,                        \
                                                                   nhead,                        \
                                                                   nhead_k,                      \
                                                                   shape_seqlen_q,               \
                                                                   shape_seqlen_k,               \
                                                                   hdim_q,                       \
                                                                   hdim_v,                       \
                                                                   max_seqlen_k,                 \
                                                                   scale,                        \
                                                                   i_perm,                       \
                                                                   o_perm,                       \
                                                                   mask.y,                       \
                                                                   mask.x)

#define INVOKE_FMHA_BWD_KERNEL_V(hdim_)                                                          \
    fmha_bwd_kernel_invoker<hdim_, DataType>{mode, use_bias, mask}(stream_vconfig,               \
                                                                   q_buf.GetDeviceBuffer(),      \
                                                                   k_buf.GetDeviceBuffer(),      \
                                                                   v_buf.GetDeviceBuffer(),      \
                                                                   bias_buf.GetDeviceBuffer(),   \
                                                                   lse_buf.GetDeviceBuffer(),    \
                                                                   do_buf.GetDeviceBuffer(),     \
                                                                   d_buf.GetDeviceBuffer(),      \
                                                                   dq_buf.GetDeviceBuffer(),     \
                                                                   dk_buf.GetDeviceBuffer(),     \
                                                                   dv_buf.GetDeviceBuffer(),     \
                                                                   dbias_buf.GetDeviceBuffer(),  \
                                                                   seqstart_q.GetDeviceBuffer(), \
                                                                   seqstart_k.GetDeviceBuffer(), \
                                                                   nullptr,                      \
                                                                   batch,                        \
                                                                   nhead,                        \
                                                                   nhead_k,                      \
                                                                   shape_seqlen_q,               \
                                                                   shape_seqlen_k,               \
                                                                   hdim_q,                       \
                                                                   hdim_v,                       \
                                                                   max_seqlen_k,                 \
                                                                   scale,                        \
                                                                   i_perm,                       \
                                                                   o_perm,                       \
                                                                   mask.y,                       \
                                                                   mask.x)

    float ave_time         = 0;
    const auto check_hdims = [](ck::index_t hdim_q_, ck::index_t hdim_v_, ck::index_t threshold) {
        const auto compare =
            std::conditional_t<kK0N1NeedPadding, std::less_equal<>, std::equal_to<>>{};
        return compare(hdim_q_, threshold) && compare(hdim_v_, threshold);
    };

    if(check_hdims(hdim_q, hdim_v, 32))
    {
        ave_time = INVOKE_FMHA_BWD_DOT_DO_O_KERNEL(32);
        ave_time += INVOKE_FMHA_BWD_KERNEL(32);
    }
    else if(check_hdims(hdim_q, hdim_v, 64))
    {
        ave_time = INVOKE_FMHA_BWD_DOT_DO_O_KERNEL(64);
        ave_time += INVOKE_FMHA_BWD_KERNEL(64);
    }
    else if(check_hdims(hdim_q, hdim_v, 128))
    {
        ave_time = INVOKE_FMHA_BWD_DOT_DO_O_KERNEL(128);
        ave_time += INVOKE_FMHA_BWD_KERNEL(128);
    }
    else
    {
        std::cerr << "not support hdim, will not run" << std::endl;
        return false;
    }

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
              << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2) << gb_per_sec
              << " GB/s" << std::flush;

    if(!do_validation)
    {
        std::cout << std::endl;
        return true;
    }

    bool pass = true;

    std::vector<Tensor<QDataType>> q_host_refs;
    std::vector<Tensor<KDataType>> k_host_refs;
    std::vector<Tensor<VDataType>> v_host_refs;
    std::vector<Tensor<ODataType>> o_host_refs;
    // std::vector<Tensor<ZDataType>> z_host_refs;
    std::vector<Tensor<AccDataType>> p_hp_host_refs;
    std::vector<Tensor<GemmDataType>> p_lp_host_refs;

    for(ck::index_t wb = 0; wb < batch; ++wb)
    {
        const ck::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck::index_t b            = (mode == mode_enum::batch ? wb : 0);
        const ck::index_t query_offset = (mode == mode_enum::batch ? 0 : seqstart_q_host[wb]);
        const ck::index_t key_offset   = (mode == mode_enum::batch ? 0 : seqstart_k_host[wb]);

        Tensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q}); // q_g_m_k
        Tensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q}); // k_g_n_k
        Tensor<VDataType> v_host_ref({nhead, hdim_v, real_seqlen_k}); // v_g_o_n
        Tensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v}); // o_g_m_o
        Tensor<LSEDataType> lse_host_ref({nhead, real_seqlen_q});     // lse_g_m
        // Tensor<ZDataType> z_host_ref({nhead, real_seqlen_q, real_seqlen_k}); // z_g_m_n
        Tensor<AccDataType> s_host_ref({nhead, real_seqlen_q, real_seqlen_k}); // s_g_m_n
        Tensor<AccDataType> p_hp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // p_hp_g_m_n high precision
        Tensor<GemmDataType> p_lp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // p_lp_g_m_n low precision

        ck::index_t nr = nhead / nhead_k;

        // clang-format off
        // permute
        if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[0], i[1] + query_offset, i[2]); });
        else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[1] + query_offset, i[0], i[2]); });

        if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[0] / nr, i[1] + key_offset, i[2]); });
        else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[1] + key_offset, i[0] / nr, i[2]); });

        // v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d]
        if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[2] + key_offset, i[1]); });
        // v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
        else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[2] + key_offset, i[0] / nr, i[1]); });

        // reference
        // S = scale * Q * K^T
        reference_batched_gemm<QDataType, KDataType, AccDataType, AccDataType>(
            q_host_ref, k_host_ref, s_host_ref,
            ck::identity{}, ck::identity{},
            [&](AccDataType x) { return scale * x; }); // s_g_m_n = scale * q_g_m_k@k_g_n_k

        if(use_bias)
        {
            Tensor<BiasDataType> bias_host_ref({1, real_seqlen_q, real_seqlen_k});
            if(i_perm)
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, 0, i[1] + query_offset, i[2] + key_offset); });
            else
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, i[1] + query_offset, 0, i[2] + key_offset); });

            // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q, real_seqlen_k]
            reference_batched_elementwise<AccDataType, BiasDataType, AccDataType, AccDataType>(
                s_host_ref, bias_host_ref, s_host_ref);
        }

        if(mask.type == mask_enum::no_mask) {
            reference_batched_masking<AccDataType>(s_host_ref, FmhaMasks::NoMask{real_seqlen_q, real_seqlen_k});
        } else if(mask.type == mask_enum::window_generic) {
            reference_batched_masking<AccDataType>(s_host_ref,
                FmhaMasks::GenericMask{mask.y, mask.x, real_seqlen_q, real_seqlen_k});
        } else {
            reference_batched_masking<AccDataType>(s_host_ref,
                FmhaMasks::CausalMask{mask.y, mask.x, real_seqlen_q, real_seqlen_k});
        }
        reference_batched_softmax<AccDataType, LSEDataType, AccDataType>(s_host_ref, p_hp_host_ref, lse_host_ref);

        p_hp_host_ref.ForEach(
            [&](auto& self, auto idx) { p_lp_host_ref(idx) = ck::type_convert<GemmDataType>(self(idx)); });

        // O = P * V
        reference_batched_gemm<GemmDataType, VDataType, AccDataType, ODataType>(p_lp_host_ref, v_host_ref, o_host_ref); // o_g_m_o = p_lp_g_m_n@v_g_o_n

        // permute
        if(o_perm) o_host_ref.ForEach([&](auto& self, auto idx) { o_host(b, idx[0], idx[1] + query_offset, idx[2]) = self(idx); });
        else       o_host_ref.ForEach([&](auto& self, auto idx) { o_host(b, idx[1] + query_offset, idx[0], idx[2]) = self(idx); });

        lse_host_ref.ForEach([&](auto& self, auto idx) { lse_host(b, idx[0], idx[1] + query_offset) = self(idx); });

        q_host_refs.push_back(q_host_ref);
        k_host_refs.push_back(k_host_ref);
        v_host_refs.push_back(v_host_ref);
        o_host_refs.push_back(o_host_ref);
        p_hp_host_refs.push_back(p_hp_host_ref);
        p_lp_host_refs.push_back(p_lp_host_ref);
        // clang-format on
    }

    o_buf.ToDevice(o_host.data());
    lse_buf.ToDevice(lse_host.data());
    dq_buf.SetZero();

    if(check_hdims(hdim_q, hdim_v, 32))
    {
        INVOKE_FMHA_BWD_DOT_DO_O_KERNEL(32);
        INVOKE_FMHA_BWD_KERNEL_V(32);
    }
    else if(check_hdims(hdim_q, hdim_v, 64))
    {
        INVOKE_FMHA_BWD_DOT_DO_O_KERNEL(64);
        INVOKE_FMHA_BWD_KERNEL_V(64);
    }
    else if(check_hdims(hdim_q, hdim_v, 128))
    {
        INVOKE_FMHA_BWD_DOT_DO_O_KERNEL(128);
        INVOKE_FMHA_BWD_KERNEL_V(128);
    }

    dq_buf.FromDevice(dq_host.data());
    dk_buf.FromDevice(dk_host.data());
    dv_buf.FromDevice(dv_host.data());
    dbias_buf.FromDevice(dbias_host.data());

    for(ck::index_t wb = 0; wb < batch; ++wb)
    {
        const ck::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck::index_t b            = (mode == mode_enum::batch ? wb : 0);
        const ck::index_t query_offset = (mode == mode_enum::batch ? 0 : seqstart_q_host[wb]);
        const ck::index_t key_offset   = (mode == mode_enum::batch ? 0 : seqstart_k_host[wb]);

        Tensor<OGradDataType> do_host_ref({nhead, real_seqlen_q, hdim_v}); // do_g_m_o
        Tensor<AccDataType> ds_hp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // ds_g_m_n high precision
        Tensor<GemmDataType> ds_lp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // ds_g_m_n low precision
        Tensor<AccDataType> dp_hp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // dp_g_m_n high precision
        Tensor<BiasGradDataType> dbias_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k});                        // dbias_g_m_n
        Tensor<QGradDataType> dq_host_ref({nhead, real_seqlen_q, hdim_q}); // dq_g_m_k
        Tensor<KGradDataType> dk_host_ref({nhead, real_seqlen_k, hdim_q}); // dk_g_n_k
        Tensor<VGradDataType> dv_host_ref({nhead, real_seqlen_k, hdim_v}); // dv_g_n_o

        // clang-format off
        if(o_perm) do_host_ref.ForEach([&](auto& self, auto i) { self(i) = do_host(b, i[0], i[1] + query_offset, i[2]); });
        else       do_host_ref.ForEach([&](auto& self, auto i) { self(i) = do_host(b, i[1] + query_offset, i[0], i[2]); });

        // dP_dropout = dO@V
        // dP = dO@V w/o dropout
        auto v_t_host_ref = v_host_refs[wb].Transpose({0, 2, 1}); // v_g_o_n -> v_g_n_o
        reference_batched_gemm<OGradDataType, VDataType, AccDataType, AccDataType>(
            do_host_ref, v_t_host_ref, dp_hp_host_ref); // dp_g_m_n = do_g_m_o@v_g_n_o

        // TODO: dP = dP_dropout x Z

        // dS_i_j = P_i_j .* (dP_i_j - dO_i dot O_i)
        ds_hp_host_ref.ForEach([&](auto& self, auto idx_gmn) {
            AccDataType do_dot_o = 0;
            for(int o = 0; o < hdim_v; o++)
            {
                auto idx_gmo = idx_gmn;
                idx_gmo[2]   = o;
                do_dot_o += ck::type_convert<AccDataType>(do_host_ref(idx_gmo)) *
                            ck::type_convert<AccDataType>(o_host_refs[wb](idx_gmo));
            }
            self(idx_gmn) = ck::type_convert<AccDataType>(p_hp_host_refs[wb](idx_gmn) *
                                                           (dp_hp_host_ref(idx_gmn) - do_dot_o));
        });

        if(use_bias)
        {
            ds_hp_host_ref.ForEach(
                [&](auto& self, auto idx) { dbias_host_ref(idx) = ck::type_convert<BiasGradDataType>(self(idx)); });
        }

        ds_hp_host_ref.ForEach(
            [&](auto& self, auto idx) { ds_lp_host_ref(idx) = ck::type_convert<GemmDataType>(self(idx)); });

        // dV = P_drop^T@dO^T
        // dV = P^T@dO^T w/o dropout
        auto p_t_lp_host_ref = p_lp_host_refs[wb].Transpose({0, 2, 1}); // p_lp_g_m_n -> p_lp_g_n_m
        auto do_t_host_ref   = do_host_ref.Transpose({0, 2, 1});        // do_g_m_o -> do_g_o_m
        reference_batched_gemm<GemmDataType, OGradDataType, AccDataType, VGradDataType>(
            p_t_lp_host_ref, do_t_host_ref, dv_host_ref); // dv_g_n_o = p_lp_g_n_m@do_g_o_m

        // dQ = scale * dS@K^T
        auto k_t_host_ref = k_host_refs[wb].Transpose({0, 2, 1}); // k_g_n_k -> k_g_k_n
        reference_batched_gemm<GemmDataType, KDataType, AccDataType, QGradDataType>(
            ds_lp_host_ref,
            k_t_host_ref,
            dq_host_ref,
            [](const GemmDataType& x) { return x; },
            [](const KDataType& x) { return x; },
            [&scale](const AccDataType& x) { return scale * x; }); // dq_g_m_k = ds_g_m_n@k_g_k_n

        // dK = scale * dS^T@Q^T
        auto ds_t_lp_host_ref = ds_lp_host_ref.Transpose({0, 2, 1});     // ds_g_m_n -> ds_g_n_m
        auto q_t_host_ref  = q_host_refs[wb].Transpose({0, 2, 1}); // q_g_m_k -> q_g_k_m
        reference_batched_gemm<GemmDataType, QDataType, AccDataType, KGradDataType>(
            ds_t_lp_host_ref,
            q_t_host_ref,
            dk_host_ref,
            [](const GemmDataType& x) { return x; },
            [](const QDataType& x) { return x; },
            [&scale](const AccDataType& x) { return scale * x; }); // dk_g_n_k = ds_g_n_m@q_g_k_m

        Tensor<QGradDataType> dq_host_result({nhead, real_seqlen_q, hdim_q}); // dq_g_m_k
        Tensor<KGradDataType> dk_host_result({nhead, real_seqlen_k, hdim_q}); // dk_g_n_k
        Tensor<VGradDataType> dv_host_result({nhead, real_seqlen_k, hdim_v}); // dv_g_n_o
        Tensor<BiasGradDataType> dbias_host_result({nhead, real_seqlen_q, real_seqlen_k}); // dbias_g_m_n

        // permute
        if(i_perm) dq_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dq_host(b, idx[0], idx[1] + query_offset, idx[2]); });
        else       dq_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dq_host(b, idx[1] + query_offset, idx[0], idx[2]); });

        if(i_perm) dk_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dk_host(b, idx[0], idx[1] + key_offset, idx[2]); });
        else       dk_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dk_host(b, idx[1] + key_offset, idx[0], idx[2]); });

        if(i_perm) dv_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dv_host(b, idx[0], idx[1] + key_offset, idx[2]); });
        else       dv_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dv_host(b, idx[1] + key_offset, idx[0], idx[2]); });

        if(use_bias)
        {
            if(i_perm) dbias_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dbias_host(b, idx[0], idx[1] + query_offset, idx[2] + key_offset); });
            else       dbias_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dbias_host(b, idx[1] + query_offset, idx[0], idx[2] + key_offset); });
        }
        // clang-format on

        auto [rtol, atol] = get_elimit<DataType>(init_method);
        bool dq_cur_pass  = ck::utils::check_err(dq_host_result,
                                                dq_host_ref,
                                                std::string("Error: QGrad Incorrect results!"),
                                                rtol,
                                                atol);
        bool dk_cur_pass  = ck::utils::check_err(dk_host_result,
                                                dk_host_ref,
                                                std::string("Error: KGrad Incorrect results!"),
                                                rtol,
                                                atol);
        bool dv_cur_pass  = ck::utils::check_err(dv_host_result,
                                                dv_host_ref,
                                                std::string("Error: VGrad Incorrect results!"),
                                                rtol,
                                                atol);

        bool dbias_cur_pass = true;
        if(use_bias)
        {
            dbias_cur_pass = ck::utils::check_err(dbias_host_result,
                                                  dbias_host_ref,
                                                  std::string("Error: BiasGrad Incorrect results!"),
                                                  rtol,
                                                  atol);
        }
        pass &= (dq_cur_pass & dk_cur_pass & dv_cur_pass & dbias_cur_pass);
        if(!(dq_cur_pass & dk_cur_pass & dv_cur_pass & dbias_cur_pass))
        {
            std::cerr << "mismatch found at batch: " << wb << std::endl
                      << "\tseqlen_q: " << real_seqlen_q << std::endl
                      << "\tseqlen_k: " << real_seqlen_k << std::endl
                      << "\tseqstart_q: " << seqstart_q_host << std::endl
                      << "\tseqstart_k: " << seqstart_k_host << std::endl;

            break;
        }
    }

    std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck::bhalf_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
