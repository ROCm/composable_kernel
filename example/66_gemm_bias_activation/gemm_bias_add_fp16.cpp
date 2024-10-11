// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_bias_add.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

using F16 = ck::half_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using AccDataType      = F32;
using CShuffleDataType = F32;

using ALayout  = Row;
using BLayout  = Row;
using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;
using CLayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

namespace ck {
namespace impl {
template <typename Activation>
struct AddActivation
{
    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        Activation{}.template operator()<float>(y, x0 + x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const half_t& x1) const
    {
        float x = x0 + type_convert<float>(x1);
        Activation{}.template operator()<float>(y, x);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const float& x1) const
    {
        float result = 0;
        Activation{}.template operator()<float>(result, x0 + x1);
        y = type_convert<half_t>(result);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        float result = 0;
        Activation{}.template operator()<float>(result, x0 + x1);
        y = type_convert<half_t>(result);
    };
};
} // namespace impl
} // namespace ck
// clang-format off
template <typename ADataType,   typename BDataType, typename DsDataType,  typename CDataType>
using DeviceOpInstance_64_16_16_64 = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
        ALayout,  BLayout, DsLayout, CLayout, ADataType, BDataType,
        DsDataType, CDataType, AccDataType, CShuffleDataType,
        AElementOp,  BElementOp, CDEElementOp,       GemmSpec,
        64,
        16,   16,    64,
        8,    8,
        16,   16,
        1,    1,
        S<8,  8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,
        2,    8,     8,    0,
        S<8, 8, 1>,     S<0, 2, 1>,    S<0, 2, 1>,
        1,    2,     2,    0,
        1,    1,
        S<1, 16, 1, 4>,      S<4, 4>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, F16>;

template <typename ADataType,   typename BDataType, typename DsDataType,  typename CDataType>
using DeviceOpInstance_default = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
        ALayout,  BLayout, DsLayout, CLayout, ADataType, BDataType,
        DsDataType, CDataType, AccDataType, CShuffleDataType,
        AElementOp,  BElementOp, CDEElementOp,       GemmSpec,
        64,
        16,   16,    64,
        8,    8,
        16,   16,
        1,    1,
        S<8,  8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,
        2,    1,     1,    0,
        S<8,  8, 1>,     S<0, 2, 1>,    S<0, 2, 1>,
        1,    1,     1,    0,
        1,    1,
        S<1, 16, 1, 4>,      S<2, 2>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, F16>;

// clang-format on

float gemm_bias_add_fp16(const GemmBiasAddArgs& args, const StreamConfig& config)
{
    using ADataType  = ck::half_t;
    using BDataType  = ck::half_t;
    using CDataType  = ck::half_t;
    using D0DataType = F16;
    using DsDataType = ck::Tuple<D0DataType>;

    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
        std::cout << "gemm_bias_add_fp16: {"
                  << "mat_a: " << args.mat_a << ",  mat_b: " << args.mat_b
                  << ", mat_bias: " << args.mat_bias << ",  mat_c: " << args.mat_c
                  << ", M: " << args.M << ", N: " << args.N << ", K: " << args.K << "}"
                  << std::endl;
    }

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    ck::index_t StrideA = args.K;
    ck::index_t StrideB = args.N;
    ck::index_t StrideD = 0;
    ck::index_t StrideC = args.N;

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    float ave_time = 0;
    auto Run       = [&](auto& gemm) {
        auto argument = gemm.MakeArgument(args.mat_a,
                                          args.mat_b,
                                          std::array<const void*, NumDTensor>{args.mat_bias},
                                          args.mat_c,
                                          args.M,
                                          args.N,
                                          args.K,
                                          StrideA,
                                          StrideB,
                                          std::array<ck::index_t, NumDTensor>{StrideD},
                                          StrideC,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);

        if(!gemm.IsSupportedArgument(argument))
        {
            return false;
        }

        auto invoker = gemm.MakeInvoker();
        ave_time     = invoker.Run(argument, config);

        return true;
    };

    auto gemm = DeviceOpInstance_64_16_16_64<ADataType, BDataType, DsDataType, CDataType>{};
    if(!Run(gemm))
    {
        auto gemm_def = DeviceOpInstance_default<ADataType, BDataType, DsDataType, CDataType>{};
        Run(gemm_def);
    }

    return ave_time;
}
