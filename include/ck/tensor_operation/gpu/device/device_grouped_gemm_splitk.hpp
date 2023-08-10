#pragma once
#include <iostream>
#include <vector>

#include "device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

///
/// @brief      Structure representing single GEMM problem arguments.
///
///             The pointer to the vector of those structures is passed
///             to the GroupedGEMM entry point kernel.
///
struct GroupedGemmKernelArguments
{
    __host__ __device__ GroupedGemmKernelArguments(const void* p_a_grid_,
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
struct DeviceGroupedGemmSplitK : public DeviceGroupedGemm<ALayout,
                                                          BLayout,
                                                          DsLayout,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          DsDataType,
                                                          EDataType,
                                                          AElementwiseOperation,
                                                          BElementwiseOperation,
                                                          CElementwiseOperation>
{
    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the k batch size.
    ///
    /// @param      p_arg   Pointer to the Argument we're going to change.
    /// @param[in]  kbatch  The kbatch value.
    ///
    virtual void SetKBatchSize([[maybe_unused]] BaseArgument* p_arg,
                               [[maybe_unused]] index_t kbatch) const
    {
    }

    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the device kernel arguments pointer.
    ///
    /// @param      p_arg              The pointer to the Argument we're going to update.
    /// @param[in]  p_dev_kernel_args  The pointer to the device memory which contains kernel
    ///                                arguments.
    ///
    virtual void SetDeviceKernelArgs([[maybe_unused]] BaseArgument* p_arg,
                                     [[maybe_unused]] const void* p_dev_kernel_args) const
    {
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
