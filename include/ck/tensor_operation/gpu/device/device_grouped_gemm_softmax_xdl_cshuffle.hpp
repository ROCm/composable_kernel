// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_softmax_xdl_cshuffle_v1.hpp"
#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename FloatReduceAcc,
          typename GemmDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename DElementwiseOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_softmax_xdl_cshuffle_v1(
            const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
            const index_t group_count,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const DElementwiseOperation d_element_op,
            const FloatReduceAcc alpha)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t block_id = get_block_1d_id();

    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    index_t left     = 0;
    index_t right    = group_count;
    index_t group_id = index_t((left + right) / 2);
    while((!(block_id >= gemm_desc_ptr[group_id].BlockStart_ &&
             block_id < gemm_desc_ptr[group_id].BlockEnd_)) &&
          left <= right)
    {
        if(block_id < gemm_desc_ptr[group_id].BlockStart_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    GridwiseGemm::template Run<HasMainKBlockLoop>(
        gemm_desc_ptr[group_id].a_ptr,
        gemm_desc_ptr[group_id].b_ptr,
        gemm_desc_ptr[group_id].d_ptr,
        p_shared,
        a_element_op,
        b_element_op,
        d_element_op,
        alpha,
        gemm_desc_ptr[group_id].a_grid_desc_ak0_m_ak1_,
        gemm_desc_ptr[group_id].b_grid_desc_bk0_n_bk1_,
        gemm_desc_ptr[group_id].c_grid_desc_mblock_mperblock_nblock_nperblock_,
        gemm_desc_ptr[group_id].grouped_gemm_block_2_ctile_map_);
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename ReduceAccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename DElementwiseOperation,
          InMemoryDataOperationEnum DGlobalMemoryDataOperation,
          GemmSpecialization GemmSpec,
          index_t NumPrefetch,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          index_t MThreadClusterSize, // CReduceThreadClusterLengths_MPerBlock,
          index_t NThreadClusterSize, // CReduceThreadClusterLengths_NPerBlock,
          index_t MThreadSliceSize,
          index_t NThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceGroupedGemmSoftmax_Xdl_CShuffle
    : public GroupedDeviceGemmSoftmax<AElementwiseOperation,
                                      BElementwiseOperation,
                                      DElementwiseOperation,
                                      ReduceAccDataType>
{
    static_assert(BlockSize == MThreadClusterSize * NThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert(NXdlPerWave == CShuffleNXdlPerWavePerShuffle,
                  "Invalid CShuffleNXdlPerWavePerShuffle&NXdlPerWave assignments!");

    static_assert(NPerBlock == NThreadClusterSize * NThreadSliceSize,
                  "Invalid CShuffleNXdlPerWavePerShuffle&NXdlPerWave assignments!");

    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && NThreadSliceSize % InSrcVectorSize == 0)) &&
                      (NThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto AK1Number       = Number<AK1>{};
    static constexpr auto BK1Number       = Number<BK1>{};
    static constexpr auto NPerBlockNumber = Number<NPerBlock>{};

    // static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    // static constexpr index_t N_BlockTileSize = NThreadClusterSize * NThreadSliceSize;

    static auto MakeAGridDescriptor_AK0_M_AK1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % AK1 == 0);

        const index_t AK0 = K / AK1;

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_BK0_N_BK1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % BK1 == 0);

        const index_t BK0 = K / BK1;

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1));
    using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1));
    using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmSoftmax_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        DDataType,
        ReduceAccDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        DElementwiseOperation,
        DGlobalMemoryDataOperation,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        NumPrefetch,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        MThreadClusterSize,
        NThreadClusterSize,
        MThreadSliceSize,
        NThreadSliceSize,
        InSrcVectorDim,
        InSrcVectorSize,
        OutDstVectorSize,
        LoopSched>;

    struct GroupedGemmBlock2CTileMap
    {
        using UnderlyingBlock2CTileMap = typename GridwiseGemm::DefaultBlock2CTileMap;
        static_assert(
            std::is_same<decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{})),
                         typename GridwiseGemm::DefaultBlock2CTileMap>::value,
            "Wrong! Should be the same type name");
        GroupedGemmBlock2CTileMap()
        {
            block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{});
            BlockStart_        = -1;
        }

        GroupedGemmBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n, ck::index_t BlockStart)
        {
            block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n);
            BlockStart_        = BlockStart;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            return block_2_ctile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[I0] - BlockStart_));
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_2_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_2_ctile_map_.CheckValidity(c_grid_desc_m_n);
        }

        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        ck::index_t BlockStart_;
    };

    struct GemmDescKernelArg
    {
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;

        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        GroupedGemmBlock2CTileMap grouped_gemm_block_2_ctile_map_;

        const ADataType* a_ptr;
        const BDataType* b_ptr;
        DDataType* d_ptr;

        ck::index_t BlockStart_, BlockEnd_;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*>& p_a,
                 std::vector<const void*>& p_b,
                 std::vector<void*>& p_d,
                 std::vector<GemmDesc>& gemm_shapes,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 DElementwiseOperation d_element_op,
                 ReduceAccDataType alpha)
            : a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              d_element_op_{d_element_op},
              alpha_{alpha}
        {
            grid_size_           = 0;
            reduce_total_length_ = gemm_shapes[0].N_;

            gemm_descs_args_workspace_ = nullptr;

            group_count_ = ck::type_convert<ck::index_t>(gemm_shapes.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_a.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_b.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_d.size())))
            {
                throw std::runtime_error("wrong! group_count_ != P_a/b/c/ds.size");
            }

            gemm_desc_kernel_arg_.reserve(group_count_);

            for(std::size_t i = 0; i < gemm_shapes.size(); i++)
            {
                const index_t M = gemm_shapes[i].M_;
                const index_t N = gemm_shapes[i].N_;
                const index_t K = gemm_shapes[i].K_;

                // static_assert(N == reduce_total_length_, "Invalid Gemm Shape!");
                // static_assert(N == NPerBlockNumber, "Invalid NPerBlock Number!");

                const index_t StrideA = gemm_shapes[i].stride_A_;
                const index_t StrideB = gemm_shapes[i].stride_B_;
                const index_t StrideC = gemm_shapes[i].stride_C_;

                const auto a_grid_desc_ak0_m_ak1_ =
                    DeviceGroupedGemmSoftmax_Xdl_CShuffle::MakeAGridDescriptor_AK0_M_AK1(
                        M, K, StrideA);
                const auto b_grid_desc_bk0_n_bk1_ =
                    DeviceGroupedGemmSoftmax_Xdl_CShuffle::MakeBGridDescriptor_BK0_N_BK1(
                        K, N, StrideB);
                const auto c_grid_desc_m_n_ =
                    DeviceGroupedGemmSoftmax_Xdl_CShuffle::MakeCGridDescriptor_M_N(M, N, StrideC);

                const index_t grid_size_grp =
                    GroupedGemmBlock2CTileMap(c_grid_desc_m_n_, 0)
                        .block_2_ctile_map_.CalculateGridSize(c_grid_desc_m_n_);

                const index_t BlockStart = grid_size_;
                const index_t BlockEnd   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                const auto grouped_gemm_block_2_ctile_map_ =
                    GroupedGemmBlock2CTileMap(c_grid_desc_m_n_, BlockStart);

                if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                               b_grid_desc_bk0_n_bk1_,
                                               c_grid_desc_m_n_,
                                               grouped_gemm_block_2_ctile_map_))
                {
                    const auto c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            c_grid_desc_m_n_);

                    gemm_desc_kernel_arg_.push_back(
                        GemmDescKernelArg{a_grid_desc_ak0_m_ak1_,
                                          b_grid_desc_bk0_n_bk1_,
                                          c_grid_desc_m_n_,
                                          c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                          grouped_gemm_block_2_ctile_map_,
                                          static_cast<const ADataType*>(p_a[i]),
                                          static_cast<const BDataType*>(p_b[i]),
                                          static_cast<DDataType*>(p_d[i]),
                                          BlockStart,
                                          BlockEnd});
                }
            }
        }

        //  private:
        index_t group_count_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        DElementwiseOperation d_element_op_;

        std::vector<GemmDescKernelArg> gemm_desc_kernel_arg_;

        void* gemm_descs_args_workspace_;

        index_t grid_size_;

        ReduceAccDataType alpha_;
        index_t reduce_total_length_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGroupedGemmSoftmax_Xdl_CShuffle::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            bool has_main_k_block_loop = true;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                std::cout << "group: " << i << " arg.a_grid_desc_ak0_m_ak1_{"
                          << arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I0)
                          << ", "
                          << arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I1)
                          << ", "
                          << arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I2)
                          << "}";

                std::cout << ", arg.b_grid_desc_bk0_n_bk1_{"
                          << arg.gemm_desc_kernel_arg_[i].b_grid_desc_bk0_n_bk1_.GetLength(I0)
                          << ", "
                          << arg.gemm_desc_kernel_arg_[i].b_grid_desc_bk0_n_bk1_.GetLength(I1)
                          << ", "
                          << arg.gemm_desc_kernel_arg_[i].b_grid_desc_bk0_n_bk1_.GetLength(I2)
                          << "}";

                std::cout << ", arg.c_grid_desc_m_n_{ "
                          << arg.gemm_desc_kernel_arg_[i].c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.gemm_desc_kernel_arg_[i].c_grid_desc_m_n_.GetLength(I1) << "}"
                          << std::endl;

                if(!GridwiseGemm::CheckValidity(
                       arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_,
                       arg.gemm_desc_kernel_arg_[i].b_grid_desc_bk0_n_bk1_,
                       arg.gemm_desc_kernel_arg_[i].c_grid_desc_m_n_,
                       arg.gemm_desc_kernel_arg_[i].grouped_gemm_block_2_ctile_map_))
                {
                    throw std::runtime_error(
                        "wrong! GridwiseGemmSoftmax_k0mk1_k0nk1_mn_xdl_cshuffle_v1 has invalid "
                        "setting");
                }

                const auto K = arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I0) *
                               arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I2);

                if(GridwiseGemm::CalculateHasMainKBlockLoop(K) != has_main_k_block_loop)
                {
                    throw std::runtime_error("wrong! not all gemm has_main_k_block_loop");
                }
            }

            hipGetErrorString(
                hipMemcpy(arg.gemm_descs_args_workspace_,
                          arg.gemm_desc_kernel_arg_.data(),
                          arg.gemm_desc_kernel_arg_.size() * sizeof(GemmDescKernelArg),
                          hipMemcpyHostToDevice));

            float ave_time = 0;

            if(has_main_k_block_loop)
            {
                const auto kernel =
                    kernel_grouped_gemm_softmax_xdl_cshuffle_v1<GridwiseGemm,
                                                                ReduceAccDataType,
                                                                GemmDescKernelArg,
                                                                AElementwiseOperation,
                                                                BElementwiseOperation,
                                                                DElementwiseOperation,
                                                                true>;

                ave_time = launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(arg.grid_size_),
                    dim3(BlockSize),
                    0,
                    cast_pointer_to_constant_address_space(arg.gemm_descs_args_workspace_),
                    arg.gemm_desc_kernel_arg_.size(),
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.d_element_op_,
                    arg.alpha_);
            }
            else
            {
                const auto kernel =
                    kernel_grouped_gemm_softmax_xdl_cshuffle_v1<GridwiseGemm,
                                                                ReduceAccDataType,
                                                                GemmDescKernelArg,
                                                                AElementwiseOperation,
                                                                BElementwiseOperation,
                                                                DElementwiseOperation,
                                                                false>;

                ave_time = launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(arg.grid_size_),
                    dim3(BlockSize),
                    0,
                    cast_pointer_to_constant_address_space(arg.gemm_descs_args_workspace_),
                    arg.gemm_desc_kernel_arg_.size(),
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.d_element_op_,
                    arg.alpha_);
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
        if(ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) != arg.group_count_)
            return false;
        else
            return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_a,
                             std::vector<const void*>& p_b,
                             std::vector<void*>& p_d,
                             std::vector<GemmDesc> gemm_shapes,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             DElementwiseOperation d_element_op,
                             ReduceAccDataType alpha)
    {
        return Argument{
            p_a, p_b, p_d, gemm_shapes, a_element_op, b_element_op, d_element_op, alpha};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(std::vector<const void*>& p_a,
                                                      std::vector<const void*>& p_b,
                                                      std::vector<void*>& p_d,
                                                      std::vector<GemmDesc> gemm_shapes,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      DElementwiseOperation d_element_op,
                                                      ReduceAccDataType alpha) override
    {
        return std::make_unique<Argument>(
            p_a, p_b, p_d, gemm_shapes, a_element_op, b_element_op, d_element_op, alpha);
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
        str << "DeviceGroupedGemmSoftmax_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << MThreadClusterSize << ", "
            << NThreadClusterSize << ", "
            << MThreadSliceSize << ", "
            << NThreadSliceSize
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(GemmDescKernelArg);
    }

    void SetWorkSpacePointer(BaseArgument* p_arg, void* workspace_ptr) const override
    {
        dynamic_cast<Argument*>(p_arg)->gemm_descs_args_workspace_ = workspace_ptr;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
