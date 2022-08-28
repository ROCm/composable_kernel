// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;
using QsElementOp  = ck::Tuple<PassThrough>;
using RsElementOp  = ck::Tuple<PassThrough>;

// ReduceOp
using RsThreadReduceOp = ck::Tuple<ck::reduce::Max>;

using RsGlobalReduceOp =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicMax>;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
template <ck::index_t NDimSpatial>
using DeviceInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleDMultipleR_Xdl_CShuffle
//######| NDimSpatial|              ALayout|              BLayout|              DELayout|              RLayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|     ReduceAccData|     RsData|           A|           B|          CDE|          Qs|          Rs|           Thread|           Global|           Conv|          GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|    CDRThreadTransfer|                  CDE|    RThreadTransfer|
//######|            |                     |                     |                      |                     |      Type|      Type|        Type|         DataType|       Type|      Type|              Type|       Type| Elementwise| Elementwise|  Elementwise| Elementwise| Elementwise|           Reduce|           Reduce|            Fwd|Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|       ClusterLengths| ReduceThreadTransfer| DstScalarPerVector|
//######|            |                     |                     |                      |                     |          |          |            |                 |           |          |                  |           |   Operation|   Operation|    Operation|   Operation|   Operation|        Operation|        Operation| Specialization|              |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _MPerBlock_NPerBlock|      ScalarPerVector|         _MPerBlock|
//######|            |                     |                     |                      |                     |          |          |            |                 |           |          |                  |           |            |            |             |            |            |                 |                 |               |              |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                     |           _NPerBlock|                   |
        < NDimSpatial, ALayout<NDimSpatial>, BLayout<NDimSpatial>, DELayout<NDimSpatial>, RLayout<NDimSpatial>, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType, ReduceAccDataType, RsDataType,  AElementOp,  BElementOp, CDEElementOp, QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp,       ConvSpec,   GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<64, 4>,                    4,                  1>;
// clang-format on

template <ck::index_t NDimSpatial>
using HostInstance = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                  ADataType,
                                                                  BDataType,
                                                                  EDataType,
                                                                  AElementOp,
                                                                  BElementOp,
                                                                  PassThrough>;

template <ck::index_t NDimSpatial>
bool run_conv_fwd_max(const ck::utils::conv::ConvParam& problem_size, const ExecutionConfig& config)
{
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ALayout<NDimSpatial>>(
            problem_size);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<BLayout<NDimSpatial>>(
            problem_size);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<DELayout<NDimSpatial>>(
            problem_size);

    const auto r0_g_wos_desc = make_r0_host_tensor_descriptor(problem_size);

    Tensor<ADataType> in(in_g_n_c_wis_desc);
    Tensor<BDataType> wei(wei_g_k_c_xs_desc);
    Tensor<EDataType> out_host(out_g_n_k_wos_desc);
    Tensor<EDataType> out_device(out_g_n_k_wos_desc);
    Tensor<R0DataType> r0_g(r0_g_wos_desc);

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(ADataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(BDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(EDataType) * out_device.mDesc.GetElementSpaceSize());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_g.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 2> r_g_n_k_wos_lengths;
    std::array<ck::index_t, NDimSpatial + 2> r_g_n_k_wos_strides;
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(r0_g_wos_desc.GetLengths(), r_g_n_k_wos_lengths);
    copy(r0_g_wos_desc.GetStrides(), r_g_n_k_wos_strides);

    copy(problem_size.conv_filter_strides_, conv_filter_strides);
    copy(problem_size.conv_filter_dilations_, conv_filter_dilations);
    copy(problem_size.input_left_pads_, input_left_pads);
    copy(problem_size.input_right_pads_, input_right_pads);

    // run Conv + Reduction on device
    auto conv     = DeviceInstance<NDimSpatial>{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
                                      {r0_device_buf.GetDeviceBuffer()},
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      e_g_n_k_wos_lengths,
                                      e_g_n_k_wos_strides,
                                      r_g_n_k_wos_lengths,
                                      r_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      AElementOp{},
                                      BElementOp{},
                                      CDEElementOp{},
                                      QsElementOp{},
                                      RsElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        std::cerr << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return false;
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = problem_size.GetFlops();
    std::size_t num_btype = problem_size.GetByte<ADataType, BDataType, EDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cerr << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        // run Conv + Reduction on host
        auto ref_conv     = HostInstance<NDimSpatial>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  out_host,
                                                  problem_size.conv_filter_strides_,
                                                  problem_size.conv_filter_dilations_,
                                                  problem_size.input_left_pads_,
                                                  problem_size.input_right_pads_,
                                                  AElementOp{},
                                                  BElementOp{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        Tensor<R0DataType> r0_g_host(r0_g.mDesc);

        auto reduce0_op = RsThreadReduceOp{}[ck::Number<0>{}];

        auto& output_dims = out_g_n_k_wos_desc.GetLengths();

        for(std::size_t g = 0; g < output_dims[0]; ++g)
        {
            for(std::size_t n = 0; n < output_dims[1]; ++n)
            {
                for(std::size_t h = 0; h < output_dims[3]; ++h)
                {
                    for(std::size_t w = 0; w < output_dims[4]; ++w)
                    {
                        auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
                        for(std::size_t k = 0; k < output_dims[2]; ++k)
                        {

                            auto e_val =
                                ck::type_convert<ReduceAccDataType>(out_host(g, n, k, h, w));
                            reduce0_op(reduce0_acc, e_val);
                        }
                        r0_g_host(g, n, h, w) = ck::type_convert<R0DataType>(reduce0_acc);
                    }
                }
            }
        }

        out_device_buf.FromDevice(out_device.mData.data());
        r0_device_buf.FromDevice(r0_g.mData.data());

        return ck::utils::check_err(out_device.mData,
                                    out_host.mData,
                                    "Error: incorrect results! (Matrix C)",
                                    1e-5f,
                                    1e-4f) &&
               ck::utils::check_err(r0_g.mData,
                                    r0_g_host.mData,
                                    "Error: incorrect results! (Matrix R0)",
                                    1e-5f,
                                    1e-4f);
    }

    return true;
}

bool run_conv_fwd_max_example(int argc, char* argv[])
{
    ck::utils::conv::ConvParam problem_size{
        2, 1, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
    ExecutionConfig config;

    if(!parse_cmd_args(argc, argv, problem_size, config))
    {
        return false;
    }

    switch(problem_size.num_dim_spatial_)
    {
    case 1: return run_conv_fwd_max<1>(problem_size, config);
    case 2: return run_conv_fwd_max<2>(problem_size, config);
    case 3: return run_conv_fwd_max<3>(problem_size, config);
    }

    return false;
}
