// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/utility/type_traits.hpp"


namespace ck_tile {

// Stream-K specific host arguments
template <index_t NumDTensor = 0>
struct StreamKGemmHostArgs
{
    CK_TILE_HOST StreamKGemmHostArgs() = default;
    CK_TILE_HOST StreamKGemmHostArgs(const void* a_ptr_,
                                     const void* b_ptr_,
                                     const std::array<const void*, NumDTensor>& ds_ptr_,
                                     void* e_ptr_,
                                    index_t k_batch_,
                                    index_t M_,
                                    index_t N_,
                                    index_t K_,
                                    index_t stride_A_,
                                    index_t stride_B_,
                                    const std::array<index_t, NumDTensor>& stride_Ds_,
                                    index_t stride_E_,
                                    index_t streamk_mode_ = 0,        // 0: disabled, 1: stream-k, 2: split-k
                                    index_t num_sms_ = 0,             // Number of SMs for Stream-K
                                    float grid_size_multiplier_ = 1.0f) // Grid size multiplier
     
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          k_batch(k_batch_),
          streamk_mode(streamk_mode_),
          num_sms(num_sms_),
          grid_size_multiplier(grid_size_multiplier_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    const std::array<const void*, NumDTensor> ds_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;

    // Stream-K specific parameters
    index_t streamk_mode;        // 0: data parallel, 1: stream-k, 2: split-k
    index_t num_sms;             // Number of SMs for Stream-K
    float grid_size_multiplier;  // Grid size multiplier for Stream-K
};

/// @brief The GEMM kernel device arguments.
template <index_t NumDTensor = 0>
struct StreamKGemmKernelArgs
{
    /// @brief The A input tensor's pointer to device memory.
    const void* a_ptr;
    /// @brief The B input tensor's pointer to device memory.
    const void* b_ptr;
    /// @brief The Ds input tensor's pointer to device memory.
    const std::array<const void*, NumDTensor> ds_ptr;
    /// @brief The E output tensor's pointer to device memory.
    void* e_ptr;
    /// @brief GEMM's M dimension size.
    index_t M;
    /// @brief GEMM's N dimension size.
    index_t N;
    /// @brief GEMM's K dimension size.
    index_t K;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of A tensor.
    index_t stride_A;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of B tensor.
    index_t stride_B;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of Ds tensor.
    std::array<index_t, NumDTensor> stride_Ds;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of E tensor.
    index_t stride_E;
    index_t k_batch;
    
    // Stream-K specific
    index_t streamk_mode;
    index_t num_sms;
    float grid_size_multiplier;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct StreamKGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    // TODO: GemmPipeline::CLayout -> GemmPipeline::ELayout will be changed for multi-ABD
    using ELayout    = remove_cvref_t<typename GemmPipeline::CLayout>;
    using DsLayout   = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    // Get the persistent kernel if the pipeline has it available
    struct has_persistent_kernel
    {
        template <typename T>
        using has_persistent_type = decltype(T::UsePersistentKernel);

        static constexpr bool value = []() {
            if constexpr(is_detected<has_persistent_type, GemmPipeline>{})
                return GemmPipeline::UsePersistentKernel;
            else
                return false;
        }();
    };
    static constexpr bool PersistentKernel = has_persistent_kernel::value;
    
    // Check if TilePartitioner supports Stream-K
    struct has_streamk_support
    {
        template <typename T>
        using has_streamk_type = decltype(std::declval<T>().GetTileAssignment(0, 0, 0));
        
        static constexpr bool value = is_detected<has_streamk_type, TilePartitioner>{};
    };
    
    static constexpr bool SupportsStreamK = has_streamk_support::value;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>{};

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");
    using KernelArgs = GemmKernelArgs<DsLayout::size()>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_streamk", gemm_prec_str<ADataType, BDataType>, GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(const StreamKGemmHostArgs<NumDTensor>& hostArgs)
    {
        if constexpr(SupportsStreamK)
        {
            if(hostArgs.streamk_mode == 1) // Stream-K mode
            {
                return TilePartitioner::GridSize(hostArgs.M, 
                                               hostArgs.N, 
                                               hostArgs.K, 
                                               hostArgs.num_sms,
                                               hostArgs.grid_size_multiplier);
            }
            else if(hostArgs.streamk_mode == 2) // Split-K mode
            {
                return TilePartitioner::GridSize(hostArgs.M, 
                                               hostArgs.N, 
                                               hostArgs.k_batch);
            }
        }
        
        // Default behavior
        return dim3(TilePartitioner::GridSize(hostArgs.M, hostArgs.N), 1, hostArgs.k_batch);
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     *       `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        using Kernel      = StreamKGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
        const auto kernel = kentry<KernelBlockSize, 1, Kernel, KernelArgs>;
        int occupancy;
        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, KernelBlockSize, 0));
        const int grid_size = get_available_compute_units(s) * occupancy;
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const StreamKGemmHostArgs<NumDTensor>& hostArgs)
    {
        return KernelArgs{hostArgs.a_ptr,
                          hostArgs.b_ptr,
                          hostArgs.ds_ptr,
                          hostArgs.e_ptr,
                          hostArgs.M,
                          hostArgs.N,
                          hostArgs.K,
                          hostArgs.stride_A,
                          hostArgs.stride_B,
                          hostArgs.stride_Ds,
                          hostArgs.stride_E,
                          hostArgs.k_batch,
                          hostArgs.streamk_mode,
                          hostArgs.num_sms,
                          hostArgs.grid_size_multiplier};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = __builtin_amdgcn_readfirstlane(kargs.k_batch * K1);
            const index_t KRead = __builtin_amdgcn_readfirstlane((kargs.K + K_t - 1) / K_t * K1);

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_A);
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_B);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = __builtin_amdgcn_readfirstlane(KRead);
            }
            else
            {
                splitted_k = __builtin_amdgcn_readfirstlane(kargs.K - KRead * (kargs.k_batch - 1));
            }
        }
        
        // Constructor for Stream-K offset calculation
        __device__ SplitKBatchOffset(index_t k_begin, index_t k_end, const KernelArgs& kargs)
        {
            splitted_k = k_end - k_begin;
            
            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = k_begin;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = k_begin * kargs.stride_A;
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = k_begin * kargs.stride_B;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = k_begin;
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        // Stream-K specific checks
        if(kargs.streamk_mode != 0)
        {
            if constexpr(!SupportsStreamK)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Stream-K requested but TilePartitioner doesn't support it!");
                }
                return false;
            }
            
            // Check if atomic operations are supported for the data type
            if(kargs.streamk_mode == 1) // Stream-K requires atomics
            {
                if constexpr(!std::is_same_v<EDataType, float> && 
                            !std::is_same_v<EDataType, double> &&
                            !std::is_same_v<EDataType, int32_t>)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Stream-K requires atomic operations which are not supported for this data type!");
                    }
                    return false;
                }
            }
        }
        
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }

        bool DTesnorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, ELayout> == false)
            {
                DTesnorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                      "NPerBlock without padding!");
                    }
                    DTesnorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    }
                    DTesnorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                      "MPerBlock without padding!");
                    }
                    DTesnorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    }
                    DTesnorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        return DTesnorIsValid;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const ADataType* a_ptr,
                        const BDataType* b_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        EDataType* e_ptr,
                        const KernelArgs& kargs,
                        const SplitKBatchOffset& splitk_batch_offset)
    {
        static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        const auto& b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = splitk_batch_offset.splitted_k / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        b_ptr,
                        make_tuple(splitk_batch_offset.splitted_k, kargs.N),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
            else
            {
                if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = splitk_batch_offset.splitted_k / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<1>{}, sequence<0>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        b_ptr,
                        make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
        }();

        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                using DiLayout   = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                using DDataType_ = remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.N, kargs.M),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
            },
            number<NumDTensor>{});

        // TODO: enable vector write for C in ColMajor
        const auto& e_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_E, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_E),
                    number<1>{},
                    number<1>{});
            }
        }();

        return make_tuple(a_tensor_view, b_tensor_view, ds_tensor_view, e_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadM>{});
            }
        }();

        const auto& b_pad_view = [&]() {
            const auto& b_tensor_view = views.at(I1);
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
        }();

        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                const auto& d_tensor_view = views.at(I2);
                using DiLayout            = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(d_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(d_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadM>{});
                }
            },
            number<NumDTensor>{});

        // TODO vector write in for C in ColMajor
        const auto& e_pad_view = [&]() {
            const auto& e_tensor_view = views.at(I3);
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(a_pad_view, b_pad_view, ds_pad_view, e_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view  = views.at(I0);
        const auto& b_pad_view  = views.at(I1);
        const auto& ds_pad_view = views.at(I2);
        const auto& e_pad_view  = views.at(I3);

        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_m, 0});
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, i_m});
            }
        }();

        const auto& b_block_window = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::NPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_n, 0});
            }
            else
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {0, i_n});
            }
        }();

        const auto ds_block_window = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {i_m, i_n});
                }
                else
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {i_n, i_m});
                }
            },
            number<NumDTensor>{});

        auto e_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return make_tuple(a_block_window, b_block_window, ds_block_window, e_block_window);
    }

     /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param ds_ptr input Ds pointer
     * @param e_ptr output E pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    template <bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void RunGemm(const ADataType* a_ptr,
                                       const BDataType* b_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr_0,
                                       const KernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, ds_ptr, e_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I1);
        const auto& d_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0);

        if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            // Run Epilogue Pipeline
            auto& c_block_window = gemm_tile_windows.at(I3);

            EpiloguePipeline{}.template
            operator()<decltype(c_block_window), decltype(c_block_tile), decltype(d_block_window)>(
                c_block_window, c_block_tile, d_block_window, smem_ptr_0);
        }
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note RunGEMM2LDS in with two shared memory buffers using the ping pong buffer mechanism.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param ds_ptr input Ds pointer
     * @param e_ptr output E pointer
     * @param smem_ptr_0 The starting pointer of 1st shared memory block.
     * @param smem_ptr_1 The starting pointer of 2nd shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm2LDS(const ADataType* a_ptr,
                                           const BDataType* b_ptr,
                                           const std::array<const void*, NumDTensor>& ds_ptr,
                                           EDataType* e_ptr,
                                           void* __restrict__ smem_ptr_0,
                                           void* __restrict__ smem_ptr_1,
                                           const KernelArgs& kargs,
                                           const SplitKBatchOffset& splitk_batch_offset,
                                           const index_t block_idx_m,
                                           const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, ds_ptr, e_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I1);
        const auto& d_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0, smem_ptr_1);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I3);

        EpiloguePipeline{}.template
        operator()<decltype(c_block_window), decltype(c_block_tile), decltype(d_block_window)>(
            c_block_window, c_block_tile, d_block_window, smem_ptr_0);
    }

    // Add Stream-K specific operator()

    // Stream-K kernel entry point
    template <bool U = SupportsStreamK, typename = std::enable_if_t<U>, typename = void, typename = void>
    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        const auto block_id = __builtin_amdgcn_readfirstlane(blockIdx.x);
        
        // Calculate number of tiles
        const index_t num_tile_m = (kargs.M + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock;
        const index_t num_tile_n = (kargs.N + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock;
        const index_t num_tile_k = (kargs.K + TilePartitioner::KPerBlock - 1) / TilePartitioner::KPerBlock;
        
        // Get work range for this block
        const auto [work_start, work_end] = GetStreamKWork(kargs, block_id);

        // Process all work items assigned to this block
        for (index_t work_idx = work_start; work_idx < work_end; ++work_idx) 
        {
            const auto [tile_m, tile_n, k_slice] = WorkIndexToTileCoords(
                work_idx, num_tile_m, num_tile_n, num_tile_k);
            
            const index_t i_m = tile_m * TilePartitioner::MPerBlock;
            const index_t i_n = tile_n * TilePartitioner::NPerBlock;
            
            // Create split-K batch offset for this k-slice
            const typename Base::SplitKBatchOffset splitk_batch_offset(kargs, k_slice);
            
            // Calculate tensor pointers with offsets
            const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) + 
                                   splitk_batch_offset.a_k_split_offset;
            const BDataType* b_ptr = static_cast<const BDataType*>(kargs.b_ptr) + 
                                   splitk_batch_offset.b_k_split_offset;
            CDataType* c_ptr = static_cast<CDataType*>(kargs.e_ptr);
            
            // Allocate LDS
            __shared__ char smem_ptr[GetSmemSize()];
            
            // Determine if this is a partial or full tile computation
            const bool is_partial_k = (num_tile_k > 1);
            
            if (is_partial_k) {
                // For partial tiles, use atomic accumulation
                this->template RunGemm<false>(a_ptr, b_ptr, {}, c_ptr, smem_ptr, 
                                            kargs, splitk_batch_offset, i_m, i_n);
            } else {
                // For full tiles, normal computation
                this->RunGemm(a_ptr, b_ptr, {}, c_ptr, smem_ptr, 
                            kargs, splitk_batch_offset, i_m, i_n);
            }
            
            // Synchronize between work items (if needed)
            if (work_idx + 1 < work_end) {
                __syncthreads();
            }
        }
    }


{
    using Base = GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::EDataType;

    using TilePartitioner  = typename Base::TilePartitioner;
    using GemmPipeline     = typename Base::GemmPipeline;
    using EpiloguePipeline = typename Base::EpiloguePipeline;
    using ALayout          = typename Base::ALayout;
    using BLayout          = typename Base::BLayout;
    using CLayout          = typename Base::ELayout;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        using P_ = GemmPipeline;
        return concat('_', "gemm_streamk", gemm_prec_str<ADataType, BDataType>,
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock), 
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
    }

    struct StreamKKernelArgs : Base::KernelArgs
    {
        index_t sk_blocks;
        index_t sk_big_blocks;
        index_t work_per_block;
        index_t total_tiles;
    };

    using KernelArgs = StreamKKernelArgs;

    __host__ static constexpr auto
    GridSize(index_t M, index_t N, index_t K, index_t sk_blocks = 0)
    {
        if (sk_blocks > 0) {
            return dim3(sk_blocks, 1, 1);
        }
        return dim3(TilePartitioner::GridSize(M, N), 1, 1);
    }

    __host__ static constexpr auto BlockSize() { return dim3(Base::KernelBlockSize); }

    CK_TILE_HOST static constexpr StreamKKernelArgs
    MakeKernelArgs(const StreamKGemmHostArgs& hostArgs)
    {
        const auto [total_tiles, work_per_block, remainder_work, k_splits] = 
            StreamKWorkDecomposition::ComputeStreamKDecomposition(
                hostArgs.M, hostArgs.N, hostArgs.K,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock, TilePartitioner::KPerBlock);

        return StreamKKernelArgs{{hostArgs.a_ptr,
                                  hostArgs.b_ptr,
                                  {},
                                  hostArgs.e_ptr,
                                  hostArgs.M,
                                  hostArgs.N,
                                  hostArgs.K,
                                  hostArgs.stride_A,
                                  hostArgs.stride_B,
                                  {},
                                  hostArgs.stride_E,
                                  k_splits},
                                 hostArgs.sk_blocks,
                                 hostArgs.sk_big_blocks,
                                 work_per_block,
                                 total_tiles};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    // Stream-K work distribution
    CK_TILE_DEVICE static auto GetStreamKWork(const StreamKKernelArgs& kargs, index_t block_id)
    {
        const index_t work_per_block = kargs.work_per_block;
        const index_t remainder_work = kargs.sk_big_blocks;
        
        index_t work_start, work_end;
        
        if (block_id < remainder_work) {
            // Big blocks get extra work
            work_start = block_id * (work_per_block + 1);
            work_end = work_start + work_per_block + 1;
        } else {
            // Regular blocks
            work_start = remainder_work * (work_per_block + 1) + (block_id - remainder_work) * work_per_block;
            work_end = work_start + work_per_block;
        }
        
        return make_tuple(work_start, work_end);
    }

    // Convert linear work index to tile coordinates and k-slice
    CK_TILE_DEVICE static auto WorkIndexToTileCoords(index_t work_idx, 
                                                     index_t num_tile_m, 
                                                     index_t num_tile_n,
                                                     index_t num_tile_k)
    {
        const index_t tiles_per_k_slice = num_tile_m * num_tile_n;
        const index_t k_slice = work_idx / tiles_per_k_slice;
        const index_t tile_idx = work_idx % tiles_per_k_slice;
        
        const index_t tile_m = tile_idx / num_tile_n;
        const index_t tile_n = tile_idx % num_tile_n;
        
        return make_tuple(tile_m, tile_n, k_slice);
    }

    CK_TILE_DEVICE void operator()(StreamKKernelArgs kargs) const
    {
        const auto block_id = __builtin_amdgcn_readfirstlane(blockIdx.x);
        
        // Calculate number of tiles
        const index_t num_tile_m = (kargs.M + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock;
        const index_t num_tile_n = (kargs.N + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock;
        const index_t num_tile_k = (kargs.K + TilePartitioner::KPerBlock - 1) / TilePartitioner::KPerBlock;
        
        // Get work range for this block
        const auto [work_start, work_end] = GetStreamKWork(kargs, block_id);
        
        // Process all work items assigned to this block
        for (index_t work_idx = work_start; work_idx < work_end; ++work_idx) {
            const auto [tile_m, tile_n, k_slice] = WorkIndexToTileCoords(
                work_idx, num_tile_m, num_tile_n, num_tile_k);
            
            const index_t i_m = tile_m * TilePartitioner::MPerBlock;
            const index_t i_n = tile_n * TilePartitioner::NPerBlock;
            
            // Create split-K batch offset for this k-slice
            const typename Base::SplitKBatchOffset splitk_batch_offset(kargs, k_slice);
            
            // Calculate tensor pointers with offsets
            const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) + 
                                   splitk_batch_offset.a_k_split_offset;
            const BDataType* b_ptr = static_cast<const BDataType*>(kargs.b_ptr) + 
                                   splitk_batch_offset.b_k_split_offset;
            CDataType* c_ptr = static_cast<CDataType*>(kargs.e_ptr);
            
            // Allocate LDS
            __shared__ char smem_ptr[GetSmemSize()];
            
            // Determine if this is a partial or full tile computation
            const bool is_partial_k = (num_tile_k > 1);
            
            if (is_partial_k) {
                // For partial tiles, use atomic accumulation
                this->template RunGemm<false>(a_ptr, b_ptr, {}, c_ptr, smem_ptr, 
                                            kargs, splitk_batch_offset, i_m, i_n);
            } else {
                // For full tiles, normal computation
                this->RunGemm(a_ptr, b_ptr, {}, c_ptr, smem_ptr, 
                            kargs, splitk_batch_offset, i_m, i_n);
            }
            
            // Synchronize between work items (if needed)
            if (work_idx + 1 < work_end) {
                __syncthreads();
            }
        }
    }
};

}; 

}// namespace ck_tile