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

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename Block2ETileMap,
          typename GroupedGemmBlock2ETileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          index_t MinimumOccupancy,
          TailNumber TailNum,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          GemmSpecialization GemmSpec>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_gemm_wmma_fixed_nk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,

                                      const index_t group_count,
                                      const index_t grid_size_grp,
                                      const index_t k_batch_,
                                      const AElementwiseOperation a_element_op,
                                      const BElementwiseOperation b_element_op,
                                      const CDEElementwiseOperation c_element_op)
{
#if(defined(__gfx11__) || defined(__gfx12__))

    constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
        typename GridwiseGemm::EpilogueCShuffle>();
    __shared__ char p_shared[LDS_size];

    const index_t block_id = get_block_1d_id();
    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));
        

    const index_t group_id = block_id / grid_size_grp;
    if(group_id >= group_count)
        return;
    const index_t group_start = group_id * grid_size_grp;


    const index_t M = gemm_desc_ptr[group_id].M;
    const index_t N = gemm_desc_ptr[group_id].N;
    const index_t K = gemm_desc_ptr[group_id].K;

    if(M == 0 || N == 0 || K == 0)
        return;


    const auto StrideE  = gemm_desc_ptr[group_id].StrideE;
    // const index_t m_padded = GridwiseGemm::CalculateMPadded(M);
    // const index_t n_padded = GridwiseGemm::CalculateNPadded(N);

    const auto e_grid_desc_m_n =
        GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
            M,  N,  StrideE);

    const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

    const auto local_grid_size = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

    constexpr auto NumDTensor = DsDataType::Size();

    using DsGridPointer = decltype(GridwiseGemm::MakeDsGridPointer());

    DsGridPointer p_ds_grid_;

    static_for<0, NumDTensor, 1>{}([&](auto i) {
        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
        // D pointer
        p_ds_grid_(i) = static_cast<const DDataType*>(gemm_desc_ptr[group_id].p_ds_grid[i]);
    });




// #if defined(__gfx11__)
//     // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
//     using c_data_type = remove_cvref_t<remove_pointer_t<decltype(gemm_desc_ptr[group_id].p_e_grid)>>;
//     if constexpr(!(CGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
//                    (std::is_same_v<c_data_type, ck::half_t> ||
//                     std::is_same_v<c_data_type, ck::bhalf_t>)))
//     {
// #endif


        auto epilogue_args =
            typename GridwiseGemm::EpilogueCShuffle{};
        
        const auto& desc = gemm_desc_ptr[group_id];
        const typename GridwiseGemm::Problem problem{
            desc.M,
            desc.N,
            desc.K,
            std::array<index_t, GridwiseGemm::NumATensor>{desc.StrideA},
            std::array<index_t, GridwiseGemm::NumBTensor>{desc.StrideB},
            desc.StrideDs,
            desc.StrideE,
            k_batch_
        };

        using AsGridPointer = typename GridwiseGemm::AsGridPointer;
        using ADataType0    = remove_cvref_t<tuple_element_t<0, AsDataType>>;

        AsGridPointer p_as_grid_ = make_tuple(
            static_cast<const ADataType0*>(gemm_desc_ptr[group_id].p_a_grid)
        );
        using BsGridPointer = typename GridwiseGemm::BsGridPointer;
        using BDataType0    = remove_cvref_t<tuple_element_t<0, BsDataType>>;

        BsGridPointer p_bs_grid_ = make_tuple(
            static_cast<const BDataType0*>(gemm_desc_ptr[group_id].p_b_grid)
        );


        index_t id_off   = 0;
        index_t id_local = get_block_1d_id() - group_start;

        while(id_local < local_grid_size)
        {

            // if(threadIdx.x == 0)
            // {
            //     printf(
            //         "\n[CK GEMM TRACE]\n"
            //         " id_local              = %d\n"
            //         " local_grid_size              = %d\n",
            //         int(id_local),
            //         int(local_grid_size)
            //     );

            // }

            const auto block_2_etile_map =
                GroupedGemmBlock2ETileMap(local_b2c_tile_map, group_start, id_off);

            // auto tile_idx = block_2_etile_map.CalculateBottomIndex(make_multi_index(id_local));

            // const index_t m_tile_idx = tile_idx[Number<0>{}];
            // const index_t n_tile_idx = tile_idx[Number<1>{}];
            // const index_t k_tile_idx = tile_idx[Number<2>{}];

            // calculate ranges for each dimension
            // const index_t m_start = m_tile_idx * MPerBlock;
            // const index_t m_end   = min(m_start + MPerBlock, M);

            // const index_t n_start = n_tile_idx * NPerBlock;
            // const index_t n_end   = min(n_start + NPerBlock, N);

            // const index_t k_start = k_tile_idx * KPerBlock;
            // const index_t k_end   = min(k_start + KPerBlock, K);

            // if(threadIdx.x == 0)
            // {
            //     printf("[CK GEMM TRACE] grid_size=%d, group_id=%d, block_id=%d, "
            //         "m_tile=%d, n_tile=%d, k_tile=%d, "
            //         "M_range=[%d,%d), N_range=[%d,%d), K_range=[%d,%d)\n",
            //         int(local_grid_size),
            //         int(group_id),
            //         int(get_block_1d_id()),
            //         int(m_tile_idx),
            //         int(n_tile_idx),
            //         int(k_tile_idx),
            //         int(m_start),
            //         int(m_end),
            //         int(n_start),
            //         int(n_end),
            //         int(k_start),
            //         int(k_end));
            // }



            GridwiseGemm::template Run<HasMainKBlockLoop,
                                       CGlobalMemoryDataOperation,
                                       TailNum,
                                       remove_cvref_t<decltype(block_2_etile_map)>,
                                       typename GridwiseGemm::EpilogueCShuffle,
                                       1,
                                       2>
            (p_as_grid_,
            p_bs_grid_,
            p_ds_grid_,
            static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
            static_cast<void*>(p_shared),
            problem,
            block_2_etile_map,
            a_element_op,
            b_element_op,
            c_element_op,
            epilogue_args);
             
            // if(threadIdx.x == 0)
            // {
            //     printf(
            //     "\n[CK GEMM TRACE]\n"
            //     " id_local              = %d\n"
            //     " local_grid_size              = %d\n",
            //     int(id_local),
            //     int(local_grid_size));
            // }
            id_off += grid_size_grp;
            id_local += grid_size_grp;
        }

#undef TRACE_THREAD
// #if defined(__gfx11__)
//     }
// #endif
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = grid_size_grp;
    ignore = k_batch_;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
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
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceGroupedGemm_Wmma_Fixed_Nk : public DeviceGroupedGemmFixedNK<ALayout,
                                                                         BLayout,
                                                                         DsLayout,
                                                                         ELayout,
                                                                         ADataType,
                                                                         BDataType,
                                                                         DsDataType,
                                                                         EDataType,
                                                                         AElementwiseOperation,
                                                                         BElementwiseOperation,
                                                                         CDEElementwiseOperation>
{
    using DeviceOp = DeviceGroupedGemm_Wmma_Fixed_Nk;

    static constexpr index_t NumDTensor = DsDataType::Size();

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
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
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
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
        remove_cvref_t<decltype(GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
            1, 1, 1))>;



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
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};

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
        __host__ bool CheckValidity(const CGridDesc_M_N&) const
        {
            return true;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            auto block_1d_id = idx_top[I0];

            const auto M0 = math::integer_divide_ceil(M_, MPerBlock_);
            const auto N0 = math::integer_divide_ceil(N_, NPerBlock_);

            const auto total_tiles_per_group = M0 * N0 * KBatch_;

        // #if defined(__HIP_DEVICE_COMPILE__)
        //     if(threadIdx.x == 0)
        //     {
        //         printf(
        //             "\n[CK TILE MAP TRACE]\n"
        //             " raw block_1d_id     = %d\n"
        //             " M                  = %d\n"
        //             " N                  = %d\n"
        //             " MPerBlock          = %d\n"
        //             " NPerBlock          = %d\n"
        //             " M0 (tiles)         = %d\n"
        //             " N0 (tiles)         = %d\n"
        //             " KBatch             = %d\n"
        //             " tiles/group        = %d\n",
        //             int(block_1d_id),
        //             int(M_),
        //             int(N_),
        //             int(MPerBlock_),
        //             int(NPerBlock_),
        //             int(M0),
        //             int(N0),
        //             int(KBatch_),
        //             int(total_tiles_per_group));
        //     }
        // #endif

            // wrap block id into this group
            block_1d_id = block_1d_id % total_tiles_per_group;

            const index_t idx_ksplit = block_1d_id / (M0 * N0);
            block_1d_id              = block_1d_id % (M0 * N0);

            index_t idx_N0 = block_1d_id % N0;
            index_t idx_M0 = block_1d_id / N0;

            const auto M01_adapt =
                (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

            index_t idx_M00          = idx_M0 / M01_;
            index_t idx_M01          = idx_M0 % M01_;
            index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

        // #if defined(__HIP_DEVICE_COMPILE__)
        //     if(threadIdx.x == 0)
        //     {
        //         printf(
        //             " wrapped block_id   = %d\n"
        //             " idx_ksplit         = %d\n"
        //             " idx_M0             = %d\n"
        //             " idx_N0             = %d\n"
        //             " M01                = %d\n"
        //             " M01_adapt          = %d\n"
        //             " idx_M00            = %d\n"
        //             " idx_M01            = %d\n"
        //             " idx_N0_M01_local   = %d\n"
        //             " --> m_tile         = %d\n"
        //             " --> n_tile         = %d\n"
        //             " --> k_tile         = %d\n"
        //             "\n",
        //             int(block_1d_id),
        //             int(idx_ksplit),
        //             int(idx_M0),
        //             int(idx_N0),
        //             int(M01_),
        //             int(M01_adapt),
        //             int(idx_M00),
        //             int(idx_M01),
        //             int(idx_N0_M01_local),
        //             int(idx_N0_M01_local % M01_adapt + idx_M00 * M01_),
        //             int(idx_N0_M01_local / M01_adapt),
        //             int(idx_ksplit));
        //     }
        // #endif

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


    using GemmTransKernelArg = GroupedGemmKernelArgument<NumDTensor>;

    static constexpr bool CalculateHasMainKBlockLoop(const KernelArgument& karg)
    {
        index_t k_grain = karg.KBatch * KPerBlock;
        index_t K_split = (karg.K + k_grain - 1) / karg.KBatch;
        return GridwiseGemm::CalculateHasMainKBlockLoop(K_split);
    }

    struct Argument : public BaseArgument
    {

        void UpdateKBatch(index_t k_batch)
        {
            k_batch_ = k_batch;

            if(k_batch_ < 1)
            {
                throw std::runtime_error("wrong! k_batch must be > 0");
            }

            const index_t AverM = math::integer_divide_ceil(sum_of_m, group_count_);

            const index_t StrideE = gemm_desc_kernel_arg_[0].StrideE;
            const index_t N       = gemm_desc_kernel_arg_[0].N;

            // const index_t m_padded = GridwiseGemm::CalculateMPadded(AverM);
            // const index_t n_padded = GridwiseGemm::CalculateNPadded(N);
            const auto e_grid_desc_m_n =
                GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                    AverM, N, StrideE);

            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

            grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

            grid_size_ = grid_size_grp_ * group_count_;
        }

        Argument(std::vector<const void*>&,
                 std::vector<const void*>&,
                 std::vector<std::array<const void*, NumDTensor>>&,
                 std::vector<void*>&,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op)
            : a_element_op_{a_element_op}, b_element_op_{b_element_op}, c_element_op_{c_element_op}
        {
            grid_size_ = 0;

            k_batch_ = 1;

            grouped_gemm_kernel_args_dev = nullptr;

            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            gemm_desc_kernel_arg_.reserve(group_count_);

            index_t group_id = 0;

            sum_of_m            = gemm_descs[0].M_;
            const index_t AverM = math::integer_divide_ceil(sum_of_m, group_count_);
            const index_t N     = gemm_descs[0].N_;
            const index_t K     = gemm_descs[0].K_;

            for(std::size_t i = 0; i < gemm_descs.size(); i++)
            {
                if(sum_of_m != gemm_descs[i].M_ || N != gemm_descs[i].N_ || K != gemm_descs[i].K_)
                {
                    throw std::runtime_error("wrong! M/N/K is not identical");
                }

                a_mtx_mraw_kraw_.emplace_back(sum_of_m, K);
                b_mtx_nraw_kraw_.emplace_back(N, K);

                const index_t StrideA = gemm_descs[i].stride_A_;
                const index_t StrideB = gemm_descs[i].stride_B_;
                const index_t StrideE = gemm_descs[i].stride_C_;

                // pointer
                std::array<const void*, NumDTensor> p_ds_grid;

                static_for<0, NumDTensor, 1>{}([&](auto j) { p_ds_grid[j] = nullptr; });

                std::array<index_t, NumDTensor> StrideDs;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    // using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;

                    if(gemm_descs[i].stride_Ds_.size() != NumDTensor)
                    {
                        throw std::runtime_error(
                            "wrong! gemm_descs[i].stride_Ds_.size() does not match NumDTensor");
                    }

                    StrideDs[j] = gemm_descs[i].stride_Ds_[j];
                });
                // const index_t m_padded = GridwiseGemm::CalculateMPadded(AverM);
                // const index_t n_padded = GridwiseGemm::CalculateNPadded(N);
                const auto e_grid_desc_m_n =
                    GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                        AverM, N, StrideE);

                // block-to-e-tile map
                const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

                grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

                if(group_id * grid_size_grp_ != grid_size_)
                {
                    throw std::runtime_error("wrong! grid_size_grp_ is not identical!");
                }

                grid_size_ += grid_size_grp_;

                if(!local_b2c_tile_map.CheckValidity(e_grid_desc_m_n))
                {
                    throw std::runtime_error("wrong! block_2_etile_map validation failed");
                }

                // if(!GridwiseGemm::CheckValidity(arg))
                // {
                //     std::ostringstream err;
                //     err << "Group id: " << i << " has invalid GridwiseGemm settings!" << __FILE__
                //         << ":" << __LINE__ << ", in function: " << __func__;
                //     throw std::runtime_error(err.str());
                // }

                gemm_desc_kernel_arg_.push_back(GemmTransKernelArg{
                    nullptr,
                    nullptr,
                    p_ds_grid,
                    nullptr,
                    AverM,
                    N,
                    K,
                    StrideA,
                    StrideB,
                    StrideDs,
                    StrideE,
                });

                group_id++;
            }
            // const index_t sum_of_m_padded = GridwiseGemm::CalculateMPadded(sum_of_m);
            // const index_t n_padded = GridwiseGemm::CalculateNPadded(gemm_desc_kernel_arg_[0].N);
            const auto e_grid_desc_sum_m_n =
                GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                    sum_of_m, gemm_desc_kernel_arg_[0].N,  
                    gemm_desc_kernel_arg_[0].StrideE);

                
            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_sum_m_n, k_batch_};

            barrier_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_sum_m_n);
        }

        //  private:
        index_t group_count_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation c_element_op_;

        std::vector<GemmTransKernelArg> gemm_desc_kernel_arg_;
        std::vector<Tuple<index_t, index_t>> a_mtx_mraw_kraw_;
        std::vector<Tuple<index_t, index_t>> b_mtx_nraw_kraw_;

        const void* grouped_gemm_kernel_args_dev;

        index_t grid_size_;
        index_t grid_size_grp_;
        index_t barrier_size_grp_;
        index_t sum_of_m;

        index_t k_batch_;
    };


// Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            constexpr bool has_main_k_block_loop = true;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                const auto KPad =
                    GridwiseGemm::CalculateKPadded(arg.gemm_desc_kernel_arg_[i].K, arg.k_batch_);

                if(GridwiseGemm::CalculateHasMainKBlockLoop(KPad) != has_main_k_block_loop)
                {
                    throw std::runtime_error("wrong! not all gemm has_main_k_block_loop");
                }
            }

            if(arg.grouped_gemm_kernel_args_dev == nullptr)
            {
                throw std::runtime_error("wrong! grouped_gemm_kernel_args_dev is nullpr");
            }

            float ave_time = 0;

            auto launch_kernel = [&](auto has_main_k_block_loop_,
                                     auto e_global_memory_operation_,
                                     auto min_occupancy_,
                                     auto tail_num_) {

                if(arg.k_batch_ == 1)
                {
                    const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg,
                                                              has_main_k_block_loop_,
                                                              ELayout,
                                                              Tuple<ADataType>,
                                                              Tuple<BDataType>,
                                                              DsDataType,
                                                              EDataType,
                                                              e_global_memory_operation_,
                                                              Block2ETileMap,
                                                              GroupedGemmBlock2ETileMap,
                                                              AElementwiseOperation,
                                                              BElementwiseOperation,
                                                              CDEElementwiseOperation,
                                                              min_occupancy_,
                                                              tail_num_,
                                                              MPerBlock,
                                                              NPerBlock,
                                                              KPerBlock,
                                                              GemmSpec>;

                    return launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(arg.grid_size_),
                                           dim3(BlockSize),
                                           0,
                                           cast_pointer_to_constant_address_space(arg.grouped_gemm_kernel_args_dev),
                                           arg.gemm_desc_kernel_arg_.size(),
                                           arg.grid_size_grp_,
                                           arg.k_batch_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_);
                }
                else
                {
                    const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg,
                                                              has_main_k_block_loop_,
                                                              ELayout,
                                                              Tuple<ADataType>,
                                                              Tuple<BDataType>,
                                                              DsDataType,
                                                              EDataType,
                                                              e_global_memory_operation_,
                                                              Block2ETileMap,
                                                              GroupedGemmBlock2ETileMap,
                                                              AElementwiseOperation,
                                                              BElementwiseOperation,
                                                              CDEElementwiseOperation,
                                                              min_occupancy_,
                                                              tail_num_,
                                                              MPerBlock,
                                                              NPerBlock,
                                                              KPerBlock,
                                                              GemmSpec>;

                    return launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(arg.grid_size_),
                                           dim3(BlockSize),
                                           0,
                                           cast_pointer_to_constant_address_space(arg.grouped_gemm_kernel_args_dev),
                                           arg.gemm_desc_kernel_arg_.size(),
                                           arg.grid_size_grp_,
                                           arg.k_batch_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_);
                }
            };

            const auto tail_num = GridwiseGemm::CalculateKBlockLoopTailNum(arg.gemm_desc_kernel_arg_[0].K);
            constexpr index_t min_occupancy = 1;


            if constexpr(std::is_same<ADataType, ck::bhalf_t>::value)
            {
                SelectTailNumber(tail_num, [&](auto tail_num_ct) {
                    ave_time = launch_kernel(
                        std::integral_constant<bool, has_main_k_block_loop>{},
                        std::integral_constant<InMemoryDataOperationEnum, InMemoryDataOperationEnum::Set>{},
                        std::integral_constant<index_t, min_occupancy>{},
                        tail_num_ct);
                });
            }
            else
            {
                if(arg.k_batch_ > 1)
                {
                    SelectTailNumber(tail_num, [&](auto tail_num_ct) {
                        ave_time = launch_kernel(
                            std::integral_constant<bool, has_main_k_block_loop>{},
                            std::integral_constant<InMemoryDataOperationEnum, InMemoryDataOperationEnum::AtomicAdd>{},
                            std::integral_constant<index_t, min_occupancy>{},
                            tail_num_ct);
                    });
                }
                else
                {
                    SelectTailNumber(tail_num, [&](auto tail_num_ct) {
                        ave_time = launch_kernel(
                            std::integral_constant<bool, has_main_k_block_loop>{},
                            std::integral_constant<InMemoryDataOperationEnum, InMemoryDataOperationEnum::Set>{},
                            std::integral_constant<index_t, min_occupancy>{},
                            tail_num_ct);
                    });
                }
            }


            

            return ave_time;
        }

        template <typename Lambda>
        void SelectTailNumber(TailNumber tail_num, Lambda&& lambda)
        {
            switch(tail_num)
            {
                case TailNumber::Full:   lambda(std::integral_constant<TailNumber, TailNumber::Full>{}); break;
                case TailNumber::Empty:  lambda(std::integral_constant<TailNumber, TailNumber::Empty>{}); break;
                case TailNumber::One:    lambda(std::integral_constant<TailNumber, TailNumber::One>{}); break;
                case TailNumber::Two:    lambda(std::integral_constant<TailNumber, TailNumber::Two>{}); break;
                case TailNumber::Three:  lambda(std::integral_constant<TailNumber, TailNumber::Three>{}); break;
                case TailNumber::Four:   lambda(std::integral_constant<TailNumber, TailNumber::Four>{}); break;
                case TailNumber::Five:   lambda(std::integral_constant<TailNumber, TailNumber::Five>{}); break;
                case TailNumber::Six:    lambda(std::integral_constant<TailNumber, TailNumber::Six>{}); break;
                case TailNumber::Seven:  lambda(std::integral_constant<TailNumber, TailNumber::Seven>{}); break;
                case TailNumber::Odd:    lambda(std::integral_constant<TailNumber, TailNumber::Odd>{}); break;
                case TailNumber::Even:   lambda(std::integral_constant<TailNumber, TailNumber::Even>{}); break;
                default:                 lambda(std::integral_constant<TailNumber, TailNumber::Full>{}); break;;
            }
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {                                                                                                     
            return RunImp<GridwiseGemm>(arg, stream_config);                                        
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) != arg.group_count_)
        {
            return false;
        }

        bool supported = true;
        if constexpr(GemmSpec != GemmSpecialization::Default)
        {
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
        if constexpr(std::is_same<ADataType, ck::bhalf_t>::value)
        {
            supported = supported & (arg.k_batch_ == 1);
        }

        return supported;
    }

    
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation c_element_op)
    {
        return Argument{
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation c_element_op) override
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


    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& stream_config = StreamConfig{}) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_Fixed_NK::Argument structure!");

        hip_check_error(
            hipMemsetAsync(p_workspace, 0, GetWorkSpaceSize(arg_ptr), stream_config.stream_id_));
    }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* kernel_args) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->grouped_gemm_kernel_args_dev = kernel_args;
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_Fixed_NK::Argument structure!");
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg_ptr = dynamic_cast<const Argument*>(p_arg);
        if(arg_ptr)
        {
            return arg_ptr->group_count_ * arg_ptr->barrier_size_grp_ * sizeof(uint32_t);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_Fixed_NK::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        auto arg_ptr = dynamic_cast<const Argument*>(p_arg);
        if(arg_ptr)
        {
            return arg_ptr->group_count_ * sizeof(GroupedGemmKernelArgument<NumDTensor>);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_Fixed_NK::Argument structure!");
    }


        static void SetKBatch(Argument& arg, index_t k_batch) { arg.UpdateKBatch(k_batch); }

    // polymorphic
    void SetKBatch(BaseArgument* p_arg, index_t k_batch) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->UpdateKBatch(k_batch);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_Fixed_NK::Argument structure!");
    }

    // polymorphic
    void SetKBatchSize(BaseArgument* p_arg, index_t k_batch) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->UpdateKBatch(k_batch);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_Fixed_Nk::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
