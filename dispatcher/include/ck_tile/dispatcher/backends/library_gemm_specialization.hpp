// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/dispatcher/backends/library_backend.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_cshuffle.hpp"

namespace ck_tile {
namespace dispatcher {
namespace backends {

/// Specialization for standard GEMM
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOp,
          typename BElementwiseOp,
          typename CElementwiseOp>
class LibraryGemmInstance
    : public LibraryKernelInstance<ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
          ADataType,
          BDataType,
          CDataType,
          AccDataType,
          ALayout,
          BLayout,
          CLayout,
          AElementwiseOp,
          BElementwiseOp,
          CElementwiseOp>>
{
public:
    using DeviceOp = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        ADataType,
        BDataType,
        CDataType,
        AccDataType,
        ALayout,
        BLayout,
        CLayout,
        AElementwiseOp,
        BElementwiseOp,
        CElementwiseOp>;
    
    using Base = LibraryKernelInstance<DeviceOp>;
    using ArgumentType = typename DeviceOp::Argument;
    
    LibraryGemmInstance(std::unique_ptr<DeviceOp> device_op,
                       const KernelKey& key,
                       const std::string& name)
        : Base(std::move(device_op), key, name)
    {
    }
    
    ArgumentType make_argument_impl(const Problem& problem,
                                   const void* a_ptr = nullptr,
                                   const void* b_ptr = nullptr,
                                   void* c_ptr       = nullptr) const
    {
        return ArgumentType{
            static_cast<const ADataType*>(a_ptr),
            static_cast<const BDataType*>(b_ptr),
            static_cast<CDataType*>(c_ptr),
            problem.M,
            problem.N,
            problem.K,
            problem.stride_a,
            problem.stride_b,
            problem.stride_c,
            AElementwiseOp{},
            BElementwiseOp{},
            CElementwiseOp{}};
    }
};

/// Specialization for Split-K GEMM
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOp,
          typename BElementwiseOp,
          typename CElementwiseOp>
class LibrarySplitKGemmInstance
    : public LibraryKernelInstance<ck::tensor_operation::device::DeviceGemm_Xdl_SplitK_CShuffle<
          ADataType,
          BDataType,
          CDataType,
          AccDataType,
          ALayout,
          BLayout,
          CLayout,
          AElementwiseOp,
          BElementwiseOp,
          CElementwiseOp>>
{
public:
    using DeviceOp = ck::tensor_operation::device::DeviceGemm_Xdl_SplitK_CShuffle<
        ADataType,
        BDataType,
        CDataType,
        AccDataType,
        ALayout,
        BLayout,
        CLayout,
        AElementwiseOp,
        BElementwiseOp,
        CElementwiseOp>;
    
    using Base = LibraryKernelInstance<DeviceOp>;
    using ArgumentType = typename DeviceOp::Argument;
    
    LibrarySplitKGemmInstance(std::unique_ptr<DeviceOp> device_op,
                             const KernelKey& key,
                             const std::string& name)
        : Base(std::move(device_op), key, name)
    {
    }
    
    ArgumentType make_argument_impl(const Problem& problem,
                                   const void* a_ptr = nullptr,
                                   const void* b_ptr = nullptr,
                                   void* c_ptr       = nullptr) const
    {
        return ArgumentType{
            static_cast<const ADataType*>(a_ptr),
            static_cast<const BDataType*>(b_ptr),
            static_cast<CDataType*>(c_ptr),
            problem.M,
            problem.N,
            problem.K,
            problem.stride_a,
            problem.stride_b,
            problem.stride_c,
            AElementwiseOp{},
            BElementwiseOp{},
            CElementwiseOp{},
            problem.k_batch};  // Split-K factor
    }
};

/// Specialization for Batched GEMM
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOp,
          typename BElementwiseOp,
          typename CElementwiseOp>
class LibraryBatchedGemmInstance
    : public LibraryKernelInstance<ck::tensor_operation::device::DeviceBatchedGemm_Xdl_CShuffle<
          ADataType,
          BDataType,
          CDataType,
          AccDataType,
          ALayout,
          BLayout,
          CLayout,
          AElementwiseOp,
          BElementwiseOp,
          CElementwiseOp>>
{
public:
    using DeviceOp = ck::tensor_operation::device::DeviceBatchedGemm_Xdl_CShuffle<
        ADataType,
        BDataType,
        CDataType,
        AccDataType,
        ALayout,
        BLayout,
        CLayout,
        AElementwiseOp,
        BElementwiseOp,
        CElementwiseOp>;
    
    using Base = LibraryKernelInstance<DeviceOp>;
    using ArgumentType = typename DeviceOp::Argument;
    
    LibraryBatchedGemmInstance(std::unique_ptr<DeviceOp> device_op,
                              const KernelKey& key,
                              const std::string& name)
        : Base(std::move(device_op), key, name)
    {
    }
    
    ArgumentType make_argument_impl(const Problem& problem,
                                   const void* a_ptr = nullptr,
                                   const void* b_ptr = nullptr,
                                   void* c_ptr       = nullptr) const
    {
        return ArgumentType{
            static_cast<const ADataType*>(a_ptr),
            static_cast<const BDataType*>(b_ptr),
            static_cast<CDataType*>(c_ptr),
            problem.M,
            problem.N,
            problem.K,
            problem.stride_a,
            problem.stride_b,
            problem.stride_c,
            problem.batch_stride_a,
            problem.batch_stride_b,
            problem.batch_stride_c,
            problem.batch_count,
            AElementwiseOp{},
            BElementwiseOp{},
            CElementwiseOp{}};
    }
};

/// Factory function to create appropriate library instance
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOp,
          typename BElementwiseOp,
          typename CElementwiseOp>
std::shared_ptr<KernelInstance> make_library_gemm_instance(
    const KernelKey& key,
    const std::string& name,
    bool is_batched = false,
    bool is_splitk  = false)
{
    if(is_batched)
    {
        using DeviceOp = ck::tensor_operation::device::DeviceBatchedGemm_Xdl_CShuffle<
            ADataType,
            BDataType,
            CDataType,
            AccDataType,
            ALayout,
            BLayout,
            CLayout,
            AElementwiseOp,
            BElementwiseOp,
            CElementwiseOp>;
        
        auto device_op = std::make_unique<DeviceOp>();
        return std::make_shared<LibraryBatchedGemmInstance<
            ADataType,
            BDataType,
            CDataType,
            AccDataType,
            ALayout,
            BLayout,
            CLayout,
            AElementwiseOp,
            BElementwiseOp,
            CElementwiseOp>>(std::move(device_op), key, name);
    }
    else if(is_splitk)
    {
        using DeviceOp = ck::tensor_operation::device::DeviceGemm_Xdl_SplitK_CShuffle<
            ADataType,
            BDataType,
            CDataType,
            AccDataType,
            ALayout,
            BLayout,
            CLayout,
            AElementwiseOp,
            BElementwiseOp,
            CElementwiseOp>;
        
        auto device_op = std::make_unique<DeviceOp>();
        return std::make_shared<LibrarySplitKGemmInstance<
            ADataType,
            BDataType,
            CDataType,
            AccDataType,
            ALayout,
            BLayout,
            CLayout,
            AElementwiseOp,
            BElementwiseOp,
            CElementwiseOp>>(std::move(device_op), key, name);
    }
    else
    {
        using DeviceOp = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
            ADataType,
            BDataType,
            CDataType,
            AccDataType,
            ALayout,
            BLayout,
            CLayout,
            AElementwiseOp,
            BElementwiseOp,
            CElementwiseOp>;
        
        auto device_op = std::make_unique<DeviceOp>();
        return std::make_shared<LibraryGemmInstance<
            ADataType,
            BDataType,
            CDataType,
            AccDataType,
            ALayout,
            BLayout,
            CLayout,
            AElementwiseOp,
            BElementwiseOp,
            CElementwiseOp>>(std::move(device_op), key, name);
    }
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile

