// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
    GemmSpecialization GemmSpec,
    index_t BlockSize,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t K0PerBlock,
    index_t K1,
    index_t MPerThread,
    index_t NPerThread,
    index_t KPerThread,
    typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
    typename ABlockTransferSrcVectorTensorContiguousDimOrder,
    typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
    typename BThreadTransferSrcDstAccessOrder,
    index_t BThreadTransferSrcVectorDim,
    index_t BThreadTransferSrcScalarPerVector,
    typename CThreadTransferSrcDstAccessOrder,
    index_t CThreadTransferSrcDstVectorDim,
    index_t CThreadTransferDstScalarPerVector,
    enable_if_t<
        is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<CElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
        bool> = false>
struct DeviceGemmDl : public DeviceGemm<ALayout,
                                        BLayout,
                                        CLayout,
                                        ADataType,
                                        BDataType,
                                        CDataType,
                                        AElementwiseOperation,
                                        BElementwiseOperation,
                                        CElementwiseOperation>

{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemmDl_km_kn_mn_v1r3<BlockSize,
                                     ADataType,
                                     AccDataType,
                                     CDataType,
                                     InMemoryDataOperationEnum::Set,
                                     ALayout,
                                     BLayout,
                                     CLayout,
                                     GemmSpec,
                                     MPerBlock,
                                     NPerBlock,
                                     K0PerBlock,
                                     K1,
                                     MPerThread,
                                     NPerThread,
                                     KPerThread,
                                     ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                     ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                     ABlockTransferThreadClusterArrangeOrder,
                                     ABlockTransferSrcAccessOrder,
                                     ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                     ABlockTransferSrcVectorTensorContiguousDimOrder,
                                     ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                     BThreadTransferSrcDstAccessOrder,
                                     BThreadTransferSrcVectorDim,
                                     BThreadTransferSrcScalarPerVector,
                                     CThreadTransferSrcDstAccessOrder,
                                     CThreadTransferSrcDstVectorDim,
                                     CThreadTransferDstScalarPerVector>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              M_{M},
              N_{N},
              K_{K},
              StrideA_{StrideA},
              StrideB_{StrideB},
              StrideC_{StrideC},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;

        index_t M_, N_, K_;
        index_t StrideA_, StrideB_, StrideC_;

        // TODO: unused since gridwise_gemm_dl_v1r3 does NOT support prologue for the time being.
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmDl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(
                   arg.M_, arg.N_, arg.K_, arg.StrideA_, arg.StrideB_, arg.StrideC_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdl_v2r3 has invalid setting");
            }

            const auto a_grid_desc_k0_m_k1 =
                GridwiseGemm::MakeAGridDescriptor_K0_M_K1(arg.M_, arg.K_, arg.StrideA_);
            const auto c_grid_desc_m_n =
                GridwiseGemm::MakeCGridDescriptor_M_N(arg.M_, arg.N_, arg.StrideC_);

            const auto M = c_grid_desc_m_n.GetLength(I0);
            const auto N = c_grid_desc_m_n.GetLength(I1);

            const index_t grid_size = GridwiseGemm::CalculateGridSize(M, N);

            const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
            const bool has_double_tail_k_block_loop =
                GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

            float ave_time = 0;

            if(has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm, ADataType, CDataType, true, true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.M_,
                                                  arg.N_,
                                                  arg.K_,
                                                  arg.StrideA_,
                                                  arg.StrideB_,
                                                  arg.StrideC_);
            }
            else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm, ADataType, CDataType, true, false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.M_,
                                                  arg.N_,
                                                  arg.K_,
                                                  arg.StrideA_,
                                                  arg.StrideB_,
                                                  arg.StrideC_);
            }
            else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm, ADataType, CDataType, false, true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.M_,
                                                  arg.N_,
                                                  arg.K_,
                                                  arg.StrideA_,
                                                  arg.StrideB_,
                                                  arg.StrideC_);
            }
            else
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm, ADataType, CDataType, false, false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.M_,
                                                  arg.N_,
                                                  arg.K_,
                                                  arg.StrideA_,
                                                  arg.StrideB_,
                                                  arg.StrideC_);
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::get_device_name() == "gfx906" || ck::get_device_name() == "gfx1030" ||
           ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
           ck::get_device_name() == "gfx1102")
        {
            return GridwiseGemm::CheckValidity(
                arg.M_, arg.N_, arg.K_, arg.StrideA_, arg.StrideB_, arg.StrideC_);
        }
        else
        {
            return false;
        }
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGemmDl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << MPerThread << ", "
            << NPerThread << ", "
            << KPerThread
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
