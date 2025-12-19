// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Can be shared between multiple device implementations....
template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename Block2CTileMap,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_gemm_wmma_fixed_nk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                      const index_t group_count)
{
#if(defined(__gfx11__) || defined(__gfx12__))
    constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
        typename GridwiseGemm::EpilogueCShuffle>();
    __shared__ char p_shared[LDS_size];

    const index_t block_id = get_block_1d_id();
    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    // Binary search lookup to find which group this block is part of
    index_t left     = 0;
    index_t right    = group_count;
    index_t group_id = index_t((left + right) / 2);
    while((!(block_id >= gemm_desc_ptr[group_id].block_start_ &&
             block_id < gemm_desc_ptr[group_id].block_end_)) &&
          left <= right)
    {
        if(block_id < gemm_desc_ptr[group_id].block_start_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    // NOTE: Local copy of the arg struct since SplitKBatchOffset verifies and modifies K index
    // and thus needs a non-const reference. It's also not feasible to store this in global
    // memory as different threads would be writing different K values to the same arg struct
    auto karg = gemm_desc_ptr[group_id].karg_;

#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using c_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(CGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<c_data_type, ck::half_t> ||
                    std::is_same_v<c_data_type, ck::bhalf_t>)))
    {
#endif
        const auto& block_2_ctile_map = gemm_desc_ptr[group_id].block_2_ctile_map_;

        // Tile index first dimension is the K batch
        auto tile_index =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        auto splitk_batch_offset =
            typename GridwiseGemm::SplitKBatchOffset(karg, tile_index[Number<0>{}]);
        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<HasMainKBlockLoop,
                                   CGlobalMemoryDataOperation,
                                   TailNum,
                                   Block2CTileMap,
                                   typename GridwiseGemm::EpilogueCShuffle,
                                   1, // Block2CTileMap MBlock index
                                   2  // Block2CTileMap NBlock index
                                   >(static_cast<void*>(p_shared),
                                     splitk_batch_offset,
                                     karg,
                                     block_2_ctile_map,
                                     epilogue_args);
#if defined(__gfx11__)
    }
#endif
#else
    ignore = gemm_descs_const;
    ignore = group_count;
#endif // end of if(defined(__gfx11__) || defined(__gfx12__))
}

template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,    // ???
          typename ComputeTypeB                       = ComputeTypeA, // ???
          bool PermuteA                               = false,        // ???
          bool PermuteB                               = false>        // ???
struct DeviceGroupedGemm_Wmma_Multi_ABD_Fixed_NK
    : public DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                              BsLayout,
                                              DsLayout,
                                              ELayout,
                                              AsDataType,
                                              BsDataType,
                                              DsDataType,
                                              EDataType,
                                              AElementwiseOperation,
                                              BElementwiseOperation,
                                              CDEElementwiseOperation>
{
    using DeviceOp = DeviceGroupedGemm_Wmma_Multi_ABD_Fixed_NK;

    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    // Note: Pass multiple layout but then using only the first one
    // This is to replicate xdl functionality but it should be extended
    using ALayout = remove_cvref_t<tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<tuple_element_t<0, BsLayout>>;

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        AsDataType,
        BsDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,
        false>;

    using CGridDesc_M_N =
        remove_cvref_t<decltype(GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
            1, 1, 1, 1, 1))>;

    // Move OffsettedBlockToCTileMapMLoops and BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops to helper hpp?
    template <typename UnderlyingBlockToCTileMap>
    struct OffsettedBlockToCTileMapMLoops
    {
        using underlying_type = UnderlyingBlockToCTileMap;

        __host__ __device__ OffsettedBlockToCTileMapMLoops(
            UnderlyingBlockToCTileMap block_to_ctile_map, index_t block_start, index_t id_off = 0)
        {
            block_to_ctile_map_ = block_to_ctile_map;
            block_start_        = block_start;
            id_off_             = id_off;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            auto idx_bot = block_to_ctile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[Number<0>{}] - block_start_ + id_off_));

            return make_tuple(idx_bot[Number<0>{}], idx_bot[Number<1>{}], idx_bot[Number<2>{}]);
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_to_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        template <typename CGridDesc_M_N>
        __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_to_ctile_map_.CheckValidity(c_grid_desc_m_n);
        }

        template <typename CGridDesc_M_N>
        __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_to_ctile_map_.CalculateGridSize(c_grid_desc_m_n);
        }

        UnderlyingBlockToCTileMap block_to_ctile_map_;
        index_t block_start_;
        index_t id_off_;
    };

    template <index_t MPerBlock_, index_t NPerBlock_>
    struct BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops
    {
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops() = default;

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            const BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&
        operator=(const BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&
        operator=(BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&&) = default;

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(index_t M,
                                                                          index_t N,
                                                                          index_t KBatch,
                                                                          index_t M01 = 8)
            : M_(M), N_(N), KBatch_(KBatch), M01_(M01)
        {
        }

        template <typename CGridDesc_M_N>
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            const CGridDesc_M_N& c_grid_desc_m_n, index_t KBatch, index_t M01 = 8)
            : BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
                  c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1), KBatch, M01)
        {
        }

        __host__ __device__ constexpr index_t CalculateGridSize(index_t M, index_t N) const
        {
            const auto M0 = math::integer_divide_ceil(M, MPerBlock);
            const auto N0 = math::integer_divide_ceil(N, NPerBlock);

            return M0 * N0 * KBatch_;
        }

        template <typename CGridDesc_M_N>
        __host__ __device__ constexpr index_t
        CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return CalculateGridSize(c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1));
        }

        template <typename CGridDesc_M_N>
        __host__ bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
        {
            return true;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            auto block_1d_id = idx_top[I0];

            const auto M0 = math::integer_divide_ceil(M_, MPerBlock_);
            const auto N0 = math::integer_divide_ceil(N_, NPerBlock_);

            block_1d_id = block_1d_id % (M0 * N0 * KBatch_); // hide groups

            const index_t idx_ksplit = block_1d_id / (M0 * N0);
            block_1d_id              = block_1d_id % (M0 * N0);

            index_t idx_N0 = block_1d_id % N0;
            index_t idx_M0 = block_1d_id / N0;

            const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

            index_t idx_M00          = idx_M0 / M01_;
            index_t idx_M01          = idx_M0 % M01_;
            index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

            return make_tuple(idx_ksplit,
                              idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                              idx_N0_M01_local / M01_adapt);
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                                 const CTileDim& /* c_tile_dim */) const
        {
            return true; // always valid provided that user gets grid size from CalculateGridSize()
        }

        private:
        index_t M_;
        index_t N_;
        index_t KBatch_;
        index_t M01_;
    };

    using Block2ETileMap = BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops<MPerBlock, NPerBlock>;
    using GroupedGemmBlock2ETileMap = OffsettedBlockToCTileMapMLoops<Block2ETileMap>;

    static constexpr index_t DefaultKBatch = 1;
    using KernelArgument                   = typename GridwiseGemm::Argument;

    template <typename KernelArgument_>
    struct GemmTransKernelArgBase
    {
        KernelArgument_ karg_;
        GroupedGemmBlock2ETileMap block_2_ctile_map_;
        index_t block_start_, block_end_;

        GemmTransKernelArgBase() = default;
        GemmTransKernelArgBase(KernelArgument_&& karg,
                               GroupedGemmBlock2ETileMap&& b2c_map,
                               index_t block_start,
                               index_t block_end)
            : karg_{karg},
              block_2_ctile_map_{b2c_map},
              block_start_{block_start},
              block_end_{block_end}
        {
        }
    };
    using GemmTransKernelArg = GemmTransKernelArgBase<KernelArgument>;

    static constexpr bool CalculateHasMainKBlockLoop(const KernelArgument& karg)
    {
        index_t k_grain = karg.KBatch * KPerBlock;
        index_t K_split = (karg.K + k_grain - 1) / karg.KBatch;
        return GridwiseGemm::CalculateHasMainKBlockLoop(K_split);
    }

    // Argument
    struct Argument : public BaseArgument
    {

        Argument(std::vector<std::array<const void*, NumATensor>>& p_As,
                 std::vector<std::array<const void*, NumBTensor>>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmMultiABDDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op)
            : Argument(p_As,
                       p_Bs,
                       p_Ds,
                       p_Es,
                       gemm_descs,
                       a_element_op,
                       b_element_op,
                       c_element_op,
                       DefaultKBatch)
        {
            // TODO: use occupancy api to calculate appropriate batch size.
        }

        Argument(std::vector<std::array<const void*, NumATensor>>& p_As,
                 std::vector<std::array<const void*, NumBTensor>>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmMultiABDDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op,
                 index_t kbatch)
            : group_count_{ck::type_convert<ck::index_t>(gemm_descs.size())},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              grouped_gemm_kernel_args_dev{nullptr},
              gemm_kernel_host_args_{nullptr},
              grid_size_{0},
              k_batch_{kbatch}
        {

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 ((NumDTensor == 0 && p_Ds.size() == 0) || group_count_ == ck::type_convert<ck::index_t>(p_Ds.size())) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/d/e.size");
            }

            gemm_desc_kernel_arg_.reserve(group_count_);

            const index_t fixed_N = gemm_descs[0].N_;
            const index_t fixed_K = gemm_descs[0].K_;

            for(std::size_t i = 0; i < gemm_descs.size(); i++)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                if(N != fixed_N || K != fixed_K) // M?
                {
                    throw std::runtime_error("wrong! N or K are not fixed across GEMM groups");
                }

                a_mtx_mraw_kraw_.emplace_back(M, K);
                b_mtx_nraw_kraw_.emplace_back(N, K);

                // pointer
                std::array<const void*, NumATensor> p_as_grid;
                std::array<const void*, NumBTensor> p_bs_grid;
                std::array<const void*, NumDTensor> p_ds_grid;

                static_for<0, NumATensor, 1>{}([&](auto j) { p_as_grid[j] = nullptr; });
                static_for<0, NumBTensor, 1>{}([&](auto j) { p_bs_grid[j] = nullptr; });
                static_for<0, NumDTensor, 1>{}([&](auto j) { p_ds_grid[j] = nullptr; });

                std::array<index_t, NumATensor> StrideAs;
                std::array<index_t, NumBTensor> StrideBs;
                std::array<index_t, NumDTensor> StrideDs;

                const index_t StrideE = gemm_descs[i].stride_C_;

                if(gemm_descs[i].stride_As_.size() != NumATensor)
                {
                    throw std::runtime_error(
                        "wrong! gemm_descs[i].stride_As_.size() does not match NumATensor");
                }
                
                static_for<0, NumATensor, 1>{}(
                    [&](auto j) { StrideAs[j] = gemm_descs[i].stride_As_[j]; });

                if(gemm_descs[i].stride_Bs_.size() != NumBTensor)
                {
                    throw std::runtime_error(
                        "wrong! gemm_descs[i].stride_Bs_.size() does not match NumBTensor");
                }

                static_for<0, NumBTensor, 1>{}(
                    [&](auto j) { StrideBs[j] = gemm_descs[i].stride_Bs_[j]; });
                
                if(gemm_descs[i].stride_Ds_.size() != NumDTensor)
                {
                    throw std::runtime_error(
                        "wrong! gemm_descs[i].stride_Ds_.size() does not match NumDTensor");
                }

                static_for<0, NumDTensor, 1>{}(
                    [&](auto j) { StrideDs[j] = gemm_descs[i].stride_Ds_[j]; });

                const index_t m_padded = GridwiseGemm::CalculateMPadded(M);
                const index_t n_padded = GridwiseGemm::CalculateNPadded(N);

                const auto e_grid_desc_m_n =
                    GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                        M, m_padded, N, n_padded, StrideE);

                // block-to-e-tile map
                const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

                grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

                if(!local_b2c_tile_map.CheckValidity(e_grid_desc_m_n))
                {
                    throw std::runtime_error("wrong! block_2_etile_map validation failed");
                }

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp_;

                grid_size_ += grid_size_grp_;

                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                auto karg = KernelArgument(p_as_grid,
                                           p_bs_grid,
                                           p_ds_grid,
                                           type_convert<EDataType*>(p_Es[i]),
                                           M,
                                           N,
                                           K,
                                           StrideAs,
                                           StrideBs,
                                           StrideDs,
                                           StrideE,
                                           k_batch_,
                                           a_element_op,
                                           b_element_op,
                                           c_element_op,
                                           false);

                gemm_desc_kernel_arg_.emplace_back(
                    std::move(karg), std::move(grouped_block_2_ctile_map), block_start, block_end);

                // group_id++;
            }

            const auto e_grid_desc_sum_m_n =
                GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                    group_count_ * gemm_descs[0].M_,
                    group_count_ * gemm_descs[0].M_,
                    gemm_descs[0].N_,
                    gemm_descs[0].N_,
                    gemm_descs[0].stride_C_);
            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_sum_m_n, k_batch_};
            grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_sum_m_n);

            barrier_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_sum_m_n);
        }

        void UpdateKBatch(index_t) {}

        //  private:
        index_t group_count_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation c_element_op_;

        std::vector<GemmTransKernelArg> gemm_desc_kernel_arg_;
        std::vector<Tuple<index_t, index_t>> a_mtx_mraw_kraw_;
        std::vector<Tuple<index_t, index_t>> b_mtx_nraw_kraw_;

        const void* grouped_gemm_kernel_args_dev;
        void* gemm_kernel_host_args_;
        index_t grid_size_;
        index_t grid_size_grp_;
        index_t barrier_size_grp_;
        index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float RunImp(const Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{},
                     hipStream_t cpy_stream            = nullptr,
                     hipEvent_t cpy_event              = nullptr)
        {
            using GemmTransKernelArg_ = GemmTransKernelArgBase<typename GridwiseGemm::Argument>;
            static_assert(sizeof(GemmTransKernelArg_) == sizeof(GemmTransKernelArg));

            bool all_have_kbatch_gt_one = arg.gemm_desc_kernel_arg_[0].karg_.KBatch > 1;
            bool all_have_main_k0_block_loop =
                CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[0].karg_);

            bool not_all_have_main_k0_block_loop_same = false;
            bool not_all_have_kbatch_value_same       = false;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); ++i)
            {
                const auto& karg = reinterpret_cast<const typename GridwiseGemm::Argument&>(
                    arg.gemm_desc_kernel_arg_[i].karg_);

                if(stream_config.log_level_ > 0)
                {
                    karg.Print();
                }

                auto kbatch = karg.KBatch;

                if(!GridwiseGemm::CheckValidity(karg))
                {
                    std::ostringstream err;
                    err << "Group id: " << i << " has invalid GridwiseGemm settings!" << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                not_all_have_main_k0_block_loop_same |=
                    all_have_main_k0_block_loop xor CalculateHasMainKBlockLoop(karg);
                not_all_have_kbatch_value_same |= all_have_kbatch_gt_one xor (kbatch > 1);
            }

            if(not_all_have_main_k0_block_loop_same)
            {
                std::ostringstream err;
                err << "Not all gemms have same value for main_k0_block_loop! in " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                // throw std::runtime_error(err.str());
            }

            if(not_all_have_kbatch_value_same)
            {
                std::ostringstream err;
                err << "Not all gemms have same kbatch value (=1 or >1)! " << " in " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            // If the user provides copy stream and copy event, we assume that they're also
            // responsible for providing allocated host memory (eg. pinned) which
            // would be used to copy kernel arguments to the device.
            if(cpy_stream && cpy_event)
            {
                if(arg.gemm_kernel_host_args_ == nullptr)
                {
                    std::ostringstream err;
                    err << "No memory has been allocated for gemm kernel host args "
                        << "when providing the copy stream and copy event! In " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
                hip_check_error(hipMemcpyAsync(arg.p_workspace_,
                                               arg.gemm_kernel_host_args_,
                                               arg.group_count_ * sizeof(GemmTransKernelArg_),
                                               hipMemcpyHostToDevice,
                                               cpy_stream));

                hip_check_error(hipEventRecord(cpy_event, cpy_stream));

                hip_check_error(hipEventSynchronize(cpy_event));
            }
            else // In this case CK owns memory allocated on host.
            {

                hip_check_error(
                    hipMemcpyAsync(arg.p_workspace_,
                                   arg.gemm_desc_kernel_arg_.data(),
                                   arg.gemm_desc_kernel_arg_.size() * sizeof(GemmTransKernelArg_),
                                   hipMemcpyHostToDevice,
                                   stream_config.stream_id_));
            }

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                if(all_have_kbatch_gt_one)
                {
                    for(const auto& trans_arg : arg.gemm_desc_kernel_arg_)
                    {

                        const auto& karg = trans_arg.karg_;
                        hip_check_error(hipMemsetAsync(karg.p_e_grid,
                                                       0,
                                                       karg.M * karg.N * sizeof(EDataType),
                                                       stream_config.stream_id_));
                    }
                }

                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(arg.grid_size_),
                                           dim3(BlockSize),
                                           0,
                                           cast_pointer_to_constant_address_space(arg.p_workspace_),
                                           arg.gemm_desc_kernel_arg_.size());
            };

            // NOTE: If at least one gemm problem has a main k0 block loop, we include it for all
            if(all_have_main_k0_block_loop || not_all_have_main_k0_block_loop_same)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              true,
                                                              InMemoryDataOperationEnum::AtomicAdd,
                                                              GroupedGemmBlock2ETileMap>;

                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              true,
                                                              InMemoryDataOperationEnum::Set,
                                                              GroupedGemmBlock2ETileMap>;

                        Run(kernel);
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              false,
                                                              InMemoryDataOperationEnum::AtomicAdd,
                                                              GroupedGemmBlock2ETileMap>;

                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              false,
                                                              InMemoryDataOperationEnum::Set,
                                                              GroupedGemmBlock2ETileMap>;

                        Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return RunImp(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) != arg.group_count_)
        {
            return false;
        }

        bool supported = true;

        // If we use padding we do not support vector loads for dimensions not divisible by
        // vector load size.
        if constexpr(GemmSpec != GemmSpecialization::Default)
        {
            // [A|B]BlockTransferSrcVectorDim value define dimension in the block {K0,M,K1}
            // layout, thus we have to adapt it to the {M,K} or {N,K} layout.
            const auto a_raw_vector_dim = ABlockTransferSrcVectorDim != 1 ? 1 : 0;
            const auto b_raw_vector_dim = BBlockTransferSrcVectorDim != 1 ? 1 : 0;

            for(index_t i = 0; i < arg.group_count_; ++i)
            {
                const auto a_vector_dim = arg.a_mtx_mraw_kraw_[i].At(Number<a_raw_vector_dim>{});
                const auto b_vector_dim = arg.b_mtx_nraw_kraw_[i].At(Number<b_raw_vector_dim>{});

                supported = supported & (a_vector_dim % ABlockTransferSrcScalarPerVector == 0);
                supported = supported & (b_vector_dim % BBlockTransferSrcScalarPerVector == 0);
            }
        }

        // For bf16 datatype only kbatch = 1 is supported since there is no AtomicAdd
        // instruction that supports bf16 and we cannot use splitk because of that
        if constexpr(std::is_same<AsDataType, ck::bhalf_t>::value)
        {
            supported = supported & (arg.k_batch_ == 1);
        }

        return supported;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<std::array<const void*, NumATensor>>& p_As,
                             std::vector<std::array<const void*, NumBTensor>>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmMultiABDDesc> gemm_descs,
                             AElementwiseOperation a_element_op   = AElementwiseOperation{},
                             BElementwiseOperation b_element_op   = BElementwiseOperation{},
                             CDEElementwiseOperation c_element_op = CDEElementwiseOperation{})
    {
        return Argument{
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<std::array<const void*, NumATensor>>& p_As,
                        std::vector<std::array<const void*, NumBTensor>>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmMultiABDDesc>& gemm_descs,
                        AElementwiseOperation a_element_op   = AElementwiseOperation{},
                        BElementwiseOperation b_element_op   = BElementwiseOperation{},
                        CDEElementwiseOperation c_element_op = CDEElementwiseOperation{}) override
    {
        return std::make_unique<Argument>(
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, c_element_op);
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
        str << "DeviceGroupedGemm_Wmma_Fixed_Nk"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">";
        // clang-format on

        return str.str();
    }

    static void SetElementwiseOps(Argument& arg,
                                  AElementwiseOperation a_element_op,
                                  BElementwiseOperation b_element_op,
                                  CDEElementwiseOperation c_element_op)
    {
        arg.a_element_op_ = a_element_op;
        arg.b_element_op_ = b_element_op;
        arg.c_element_op_ = c_element_op;
    }

    // polymorphic
    void SetElementwiseOps(BaseArgument* p_arg,
                           AElementwiseOperation a_element_op,
                           BElementwiseOperation b_element_op,
                           CDEElementwiseOperation c_element_op) const override
    {

        SetElementwiseOps(
            *dynamic_cast<Argument*>(p_arg), a_element_op, b_element_op, c_element_op);
    }

    static void SetDeviceKernelArgs(Argument& arg, const void* kernel_args)
    {
        arg.grouped_gemm_kernel_args_dev = kernel_args;
    }

    // polymorphic
    void SetDeviceKernelArgs(BaseArgument* p_arg, const void* kernel_args) const override
    {
        return SetDeviceKernelArgs(*dynamic_cast<Argument*>(p_arg), kernel_args);
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto p_arg_ = dynamic_cast<const Argument*>(p_arg);
        if(p_arg_)
        {
            return p_arg_->gemm_desc_kernel_arg_.size() * sizeof(GemmTransKernelArg);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_CShuffleV3::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        auto arg = *dynamic_cast<const Argument*>(p_arg);

        return arg.group_count_ * sizeof(GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>);
    }

    size_t GetHostKernelArgSize(const BaseArgument* p_arg) const { return GetWorkSpaceSize(p_arg); }

    static void SetKBatch(Argument& arg, index_t k_batch) { arg.UpdateKBatch(k_batch); }

    // polymorphic
    void SetKBatch(BaseArgument* /*p_arg*/, index_t /*kbatch*/) const override
    {
        throw std::runtime_error("??? figure out later");
    }

    void SetHostKernelArgsPointer(BaseArgument* p_arg, void* p_host_kernel_args) const
    {
        Argument* pArg_ = dynamic_cast<Argument*>(p_arg);
        if(!pArg_)
        {
            throw std::runtime_error("Failed to cast argument pointer!");
        }

        pArg_->gemm_kernel_host_args_ = p_host_kernel_args;
        std::copy(pArg_->gemm_desc_kernel_arg_.begin(),
                  pArg_->gemm_desc_kernel_arg_.end(),
                  static_cast<GemmTransKernelArg*>(pArg_->gemm_kernel_host_args_));
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
