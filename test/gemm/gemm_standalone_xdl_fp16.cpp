// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_util.hpp"

#include "ck/library/utility/fill.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"

#include "gemm_f16_nn_instance.hpp"
#include "gemm_f16_nt_instance.hpp"
#include "gemm_f16_tn_instance.hpp"
#include "gemm_f16_tt_instance.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using F16              = ck::half_t;
using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

using ck::tensor_operation::device::BaseOperator;
using namespace ck::tensor_operation::device;

using DeviceGemmNN =
    DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using DeviceGemmNT =
    DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using DeviceGemmTN =
    DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using DeviceGemmTT =
    DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;

struct ProblemSize
{
    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA;
    ck::index_t StrideB;
    ck::index_t StrideC;
};

struct ExecutionConfig
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

struct LayoutConfig
{
    bool ARowMajor;
    bool BRowMajor;
    bool CRowMajor;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
bool run_gemm(const ProblemSize& problem_size,
              const ExecutionConfig& config,
              ck::tensor_operation::device::DeviceGemm<ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CElementwiseOperation>* gemm_instance_ptr);


int main(int argc, char* argv[])
{
    // Class DeviceGemm is templated by layout and precision types so it is not an option to contain
    // them in a single vector. Instead we use abstract BaseOperator class and dynamic_cast() it
    // upon invocation.
    // And since DeviceGemm does not expose template arg information, an extra book keeping class
    // LayoutConfig is used for determining which type a BaseOperator instance should be cast to.
    using OpFactoryFn = void (*)(std::vector<std::unique_ptr<BaseOperator>>&);

    const std::vector<std::tuple<ProblemSize, LayoutConfig, OpFactoryFn>> problems = {
        // clang-format off
    // 104 tiles
    {ProblemSize{2048, 3328, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_256x256},
    {ProblemSize{2048, 1664, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_256x128},
    {ProblemSize{1024, 1664, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_128x128},
    {ProblemSize{1024,  832, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_128x64},
    {ProblemSize{2048, 3328, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_256x256},
    {ProblemSize{2048, 1664, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_256x128},
    {ProblemSize{1024, 1664, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_128x128},
    {ProblemSize{1024,  832, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_128x64},
    {ProblemSize{2048, 3328, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_256x128},
    {ProblemSize{2048, 1664, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_256x128},
    {ProblemSize{1024, 1664, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_128x128},
    {ProblemSize{1024,  832, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_128x64},
    {ProblemSize{2048, 3328, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_256x256},
    {ProblemSize{2048, 1664, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_256x128},
    {ProblemSize{1024, 1664, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_128x128},
    {ProblemSize{1024,  832, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_128x64},
    // 110 tiles
    {ProblemSize{2560, 2816, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_256x256},
    {ProblemSize{2560, 1408, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_256x128},
    {ProblemSize{1280, 1408, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_128x128},
    {ProblemSize{1280,  704, 4096, -1, -1, -1}, LayoutConfig{false, false, true}, instance::add_gemm_f16_nn_128x64},
    {ProblemSize{2560, 2816, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_256x256},
    {ProblemSize{2560, 1408, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_256x128},
    {ProblemSize{1280, 1408, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_128x128},
    {ProblemSize{1280,  704, 4096, -1, -1, -1}, LayoutConfig{false, true, true}, instance::add_gemm_f16_nt_128x64},
    {ProblemSize{2560, 2816, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_256x128},
    {ProblemSize{2560, 1408, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_256x128},
    {ProblemSize{1280, 1408, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_128x128},
    {ProblemSize{1280,  704, 4096, -1, -1, -1}, LayoutConfig{true, false, true}, instance::add_gemm_f16_tn_128x64},
    {ProblemSize{2560, 2816, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_256x256},
    {ProblemSize{2560, 1408, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_256x128},
    {ProblemSize{1280, 1408, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_128x128},
    {ProblemSize{1280,  704, 4096, -1, -1, -1}, LayoutConfig{true, true, true}, instance::add_gemm_f16_tt_128x64},
        // clang-format on
    };

    ExecutionConfig config{true, 1, true};

    if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }

    for(auto& p : problems)
    {
        const ProblemSize& problem_size   = std::get<0>(p);
        const LayoutConfig& layout_config = std::get<1>(p);
        const auto& factory               = std::get<2>(p);
        std::vector<std::unique_ptr<BaseOperator>> ops;
        factory(ops);

        if(!layout_config.ARowMajor && !layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmNN*>(ops[0].get());
            run_gemm(problem_size, config, op_ptr);
        }
        else if(!layout_config.ARowMajor && layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmNT*>(ops[0].get());
            run_gemm(problem_size, config, op_ptr);
        }
        else if(layout_config.ARowMajor && !layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmTN*>(ops[0].get());
            run_gemm(problem_size, config, op_ptr);
        }
        else if(layout_config.ARowMajor && layout_config.BRowMajor)
        {
            auto op_ptr = dynamic_cast<DeviceGemmTT*>(ops[0].get());
            run_gemm(problem_size, config, op_ptr);
        }
    }

    return 0;
}

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
bool run_gemm(const ProblemSize& problem_size,
              const ExecutionConfig& config,
              ck::tensor_operation::device::DeviceGemm<ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CElementwiseOperation>* gemm_instance_ptr)
{
    // using namespace ck::literals;

    auto [M, N, K, StrideA, StrideB, StrideC] = problem_size;

    auto f_host_tensor_descriptor =
        [](ck::index_t row, ck::index_t col, ck::index_t& stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                stride = stride == -1 ? col : stride;
                return HostTensorDescriptor({row, col}, {stride, 1});
            }
            else
            {
                stride = stride == -1 ? row : stride;
                return HostTensorDescriptor({row, col}, {1, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k.begin(),
                                                                             a_m_k.end());
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n.begin(),
                                                                             b_k_n.end());
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k.begin(), a_m_k.end());
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n.begin(), b_k_n.end());
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto& gemm   = *gemm_instance_ptr;
    auto invoker = gemm.MakeInvokerPointer();
    auto argument =
        gemm.MakeArgumentPointer(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                 static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                 static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                 M,
                                 N,
                                 K,
                                 StrideA,
                                 StrideB,
                                 StrideC,
                                 a_element_op,
                                 b_element_op,
                                 c_element_op);

    if(!gemm.IsSupportedArgument(argument.get()))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    float ave_time = invoker->Run(argument.get(), StreamConfig{nullptr, config.time_kernel});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        return ck::utils::check_err(c_m_n_device_result.mData, c_m_n_host_result.mData);
    }

    return true;
}
