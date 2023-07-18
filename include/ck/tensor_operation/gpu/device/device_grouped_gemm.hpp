// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/**
 * @brief      Structure representing single GEMM problem arguments.
 *
 *             The pointer to the vector of those structures is passed
 *             to the GroupedGEMM entry point kernel.
 */
struct GemmKernelArguments
{
    __host__ __device__ GemmKernelArguments(const void* p_a_grid_,
                                            const void* p_b_grid_,
                                            void* p_c_grid_,
                                            index_t M_,
                                            index_t N_,
                                            index_t K_,
                                            index_t StrideA_,
                                            index_t StrideB_,
                                            index_t StrideC_)
        : p_a_grid{p_a_grid_},
          p_b_grid{p_b_grid_},
          p_c_grid{p_c_grid_},
          M{M_},
          N{N_},
          K{K_},
          StrideA{StrideA_},
          StrideB{StrideB_},
          StrideC{StrideC_}
    {
    }

    const void* p_a_grid;
    const void* p_b_grid;
    void* p_c_grid;
    index_t M;
    index_t N;
    index_t K;
    index_t StrideA;
    index_t StrideB;
    index_t StrideC;

    void Print() const
    {
        std::cout << "arg {"
                  << "M:" << M << ", "
                  << "N:" << N << ", "
                  << "K:" << K << ", "
                  << "SA:" << StrideA << ", "
                  << "SB:" << StrideB << ", "
                  << "SC:" << StrideC << "}" << std::endl;
    }
};

struct GemmDesc
{
    ck::index_t M_, N_, K_;
    ck::index_t stride_A_, stride_B_, stride_C_;

    std::vector<ck::index_t> stride_Ds_;
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemm : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsistent NumDTensor");

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<std::array<const void*, NumDTensor>>& p_ds,
                        std::vector<void*>& p_e,
                        std::vector<GemmDesc>& gemm_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
