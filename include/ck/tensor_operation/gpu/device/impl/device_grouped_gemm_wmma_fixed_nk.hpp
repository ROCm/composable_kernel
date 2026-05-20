// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/ck.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/utility/scheduler_enum.hpp"
#include "ck/utility/tuple.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_fixed_nk_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {
struct SplitKAdd
{
    static constexpr const char* name = "SplitKAdd";

    __host__ __device__ void set_kbatch(const index_t& id, const index_t& total)
    {
        kbatch_id = id;
        KBatch    = total;
    }

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        if(kbatch_id == KBatch - 1)
        {
            add_op(y, x0, x1);
        }
        else
        {
            passthrough_op(y, x0);
        }
    }

    private:
    index_t kbatch_id                    = 0;
    index_t KBatch                       = 1;
    static constexpr auto add_op         = Add{};
    static constexpr auto passthrough_op = PassThrough{};
};
} // namespace element_wise

namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
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

    using EpilogueType = typename std::conditional<GridwiseGemm::IsBWaveTransferApplicable &&
                                                       GridwiseGemm::UseDirectStore,
                                                   typename GridwiseGemm::EpilogueDirectStore,
                                                   typename GridwiseGemm::EpilogueCShuffle>::type;

    constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<EpilogueType>();
    __shared__ char p_shared[LDS_size];

    const index_t block_id = get_block_1d_id();
    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    const index_t group_id = block_id / grid_size_grp;
    if(group_id >= group_count)
        return;
    const index_t group_start = group_id * grid_size_grp;

    auto gemmTransKernelArg = gemm_desc_ptr[group_id];

    const index_t M = gemmTransKernelArg.M;
    const index_t N = gemmTransKernelArg.N;
    const index_t K = gemmTransKernelArg.K;

    if(M == 0 || N == 0 || K == 0)
        return;

    const auto StrideE     = gemmTransKernelArg.StrideE;
    const index_t m_padded = GridwiseGemm::CalculateMPadded(M);
    const index_t n_padded = GridwiseGemm::CalculateNPadded(N);
    const auto e_grid_desc_m_n =
        GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(M, m_padded, N, n_padded, StrideE);

    const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

    const auto local_grid_size = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    if constexpr(!(CGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<EDataType, ck::half_t> ||
                    std::is_same_v<EDataType, ck::bhalf_t>)))
#endif
    {
        using KernelArgument = typename GridwiseGemm::Argument;

        index_t id_off   = 0;
        index_t id_local = get_block_1d_id() - group_start;

        while(id_local < local_grid_size)
        {
            const auto block_2_etile_map =
                GroupedGemmBlock2ETileMap(local_b2c_tile_map, group_start, id_off);

            const auto tile_index =
                block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

            const index_t kbatch_id = __builtin_amdgcn_readfirstlane(tile_index[Number<0>{}]);

            auto c_element_op_copy(c_element_op);
            if constexpr(std::is_same_v<decltype(c_element_op_copy),
                                        ck::tensor_operation::element_wise::SplitKAdd>)
            {
                c_element_op_copy.set_kbatch(kbatch_id, k_batch_);
            }

            KernelArgument kernel_arg{std::array<const void*, 1>{gemmTransKernelArg.p_a_grid},
                                      std::array<const void*, 1>{gemmTransKernelArg.p_b_grid},
                                      gemmTransKernelArg.p_ds_grid,
                                      type_convert<EDataType*>(gemmTransKernelArg.p_e_grid),
                                      gemmTransKernelArg.M,
                                      gemmTransKernelArg.N,
                                      gemmTransKernelArg.K,
                                      std::array<index_t, 1>{gemmTransKernelArg.StrideA},
                                      std::array<index_t, 1>{gemmTransKernelArg.StrideB},
                                      gemmTransKernelArg.StrideDs,
                                      gemmTransKernelArg.StrideE,
                                      k_batch_,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op_copy,
                                      false};

            const auto splitk_batch_offset =
                typename GridwiseGemm::SplitKBatchOffset(kernel_arg, tile_index[Number<0>{}]);

            auto epilogue_args = EpilogueType{};

            GridwiseGemm::template Run<HasMainKBlockLoop,
                                       CGlobalMemoryDataOperation,
                                       TailNum,
                                       GroupedGemmBlock2ETileMap,
                                       EpilogueType,
                                       1, // Block2CTileMap MBlock index
                                       2  // Block2CTileMap NBlock index
                                       >(static_cast<void*>(p_shared),
                                         splitk_batch_offset,
                                         kernel_arg,
                                         block_2_etile_map,
                                         epilogue_args);
            __builtin_amdgcn_s_barrier();
            id_off += grid_size_grp;
            id_local += grid_size_grp;
        }
    }
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
          typename ComputeTypeB                       = ComputeTypeA>
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
        Sequence<CDEBlockTransferScalarPerVector_NPerBlock,
                 CDEBlockTransferScalarPerVector_NPerBlock,
                 CDEBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,
        false>;

    using CGridDesc_M_N =
        remove_cvref_t<decltype(GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
            1, 1, 1, 1, 1))>;

    using Block2ETileMap =
        DeviceGroupedGemm_Fixed_NK_Common::BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops<MPerBlock,
                                                                                         NPerBlock>;
    using GroupedGemmBlock2ETileMap =
        DeviceGroupedGemm_Fixed_NK_Common::OffsettedBlockToCTileMapMLoops<Block2ETileMap>;

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
        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
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

        Argument(std::vector<const void*>&,
                 std::vector<const void*>&,
                 std::vector<std::array<const void*, NumDTensor>>&,
                 std::vector<void*>&,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op,
                 index_t kbatch)
            : a_element_op_{a_element_op}, b_element_op_{b_element_op}, c_element_op_{c_element_op}
        {
            grid_size_ = 0;

            k_batch_ = kbatch;

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
                std::array<index_t, NumDTensor> StrideDs;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    // using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;

                    if(gemm_descs[i].stride_Ds_.size() != NumDTensor)
                    {
                        throw std::runtime_error(
                            "wrong! gemm_descs[i].stride_Ds_.size() does not match NumDTensor");
                    }

                    p_ds_grid[j] = nullptr;
                    StrideDs[j]  = gemm_descs[i].stride_Ds_[j];
                });

                const index_t m_padded = GridwiseGemm::CalculateMPadded(AverM);
                const index_t n_padded = GridwiseGemm::CalculateNPadded(N);
                const auto e_grid_desc_m_n =
                    GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                        AverM, m_padded, N, n_padded, StrideE);

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

                if(!GridwiseGemm::CheckValidity(
                       AverM, N, K, StrideA, StrideB, StrideDs, StrideE, k_batch_))
                {
                    throw std::runtime_error("wrong! GridwiseGemm has invalid "
                                             "setting");
                }
                gemm_desc_kernel_arg_.push_back(KernelArgument(std::array<const void*, 1>{nullptr},
                                                               std::array<const void*, 1>{nullptr},
                                                               p_ds_grid,
                                                               nullptr,
                                                               AverM,
                                                               N,
                                                               K,
                                                               std::array<index_t, 1>{StrideA},
                                                               std::array<index_t, 1>{StrideB},
                                                               StrideDs,
                                                               StrideE,
                                                               k_batch_,
                                                               a_element_op,
                                                               b_element_op,
                                                               c_element_op,
                                                               false));

                group_id++;
            }
            const index_t sum_of_m_padded = GridwiseGemm::CalculateMPadded(sum_of_m);
            const index_t n_padded = GridwiseGemm::CalculateNPadded(gemm_desc_kernel_arg_[0].N);
            const auto e_grid_desc_sum_m_n =
                GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                    sum_of_m,
                    sum_of_m_padded,
                    gemm_desc_kernel_arg_[0].N,
                    n_padded,
                    gemm_desc_kernel_arg_[0].StrideE);

            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_sum_m_n, k_batch_};

            barrier_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_sum_m_n);
        }

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

            const index_t m_padded     = GridwiseGemm::CalculateMPadded(AverM);
            const index_t n_padded     = GridwiseGemm::CalculateNPadded(N);
            const auto e_grid_desc_m_n = GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                AverM, m_padded, N, n_padded, StrideE);

            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

            grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

            grid_size_ = grid_size_grp_ * group_count_;

            for(std::size_t i = 0; i < gemm_desc_kernel_arg_.size(); i++)
            {
                gemm_desc_kernel_arg_[i].KBatch = k_batch_;
            }
        }

        //  private:
        index_t group_count_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation c_element_op_;

        std::vector<KernelArgument> gemm_desc_kernel_arg_;
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
            bool all_have_kbatch_gt_one = arg.gemm_desc_kernel_arg_[0].KBatch > 1;
            bool all_have_main_k0_block_loop =
                CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[0]);

            bool not_all_have_main_k0_block_loop_same = false;
            bool not_all_have_kbatch_value_same       = false;
            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {

                not_all_have_main_k0_block_loop_same |=
                    all_have_main_k0_block_loop xor
                    CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[i]);
                not_all_have_kbatch_value_same |=
                    all_have_kbatch_gt_one xor (arg.gemm_desc_kernel_arg_[i].KBatch > 1);
            }

            if(not_all_have_main_k0_block_loop_same)
            {
                std::ostringstream err;
                err << "Not all gemms have same value for main_k0_block_loop! in " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            if(not_all_have_kbatch_value_same)
            {
                std::ostringstream err;
                err << "Not all gemms have same kbatch value (=1 or >1)! " << " in " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
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
                const auto kernel = kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                                      GemmTransKernelArg,
                                                                      has_main_k_block_loop_,
                                                                      ALayout,
                                                                      BLayout,
                                                                      DsLayout,
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
                                                                      GemmSpec>;

                return launch_and_time_kernel(
                    stream_config,
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
            };

            constexpr index_t min_occupancy = 1;

            if(all_have_main_k0_block_loop || not_all_have_main_k0_block_loop_same)
            {
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        ave_time = launch_kernel(
                            std::integral_constant<bool, true>{},
                            std::integral_constant<InMemoryDataOperationEnum,
                                                   InMemoryDataOperationEnum::AtomicAdd>{},
                            std::integral_constant<index_t, min_occupancy>{},
                            std::integral_constant<TailNumber, TailNumber::Full>{});
                    }
                    else
                    {
                        ave_time =
                            launch_kernel(std::integral_constant<bool, true>{},
                                          std::integral_constant<InMemoryDataOperationEnum,
                                                                 InMemoryDataOperationEnum::Set>{},
                                          std::integral_constant<index_t, min_occupancy>{},
                                          std::integral_constant<TailNumber, TailNumber::Full>{});
                    }
                }
            }
            else
            {
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        ave_time = launch_kernel(
                            std::integral_constant<bool, false>{},
                            std::integral_constant<InMemoryDataOperationEnum,
                                                   InMemoryDataOperationEnum::AtomicAdd>{},
                            std::integral_constant<index_t, min_occupancy>{},
                            std::integral_constant<TailNumber, TailNumber::Full>{});
                    }
                    else
                    {
                        ave_time =
                            launch_kernel(std::integral_constant<bool, false>{},
                                          std::integral_constant<InMemoryDataOperationEnum,
                                                                 InMemoryDataOperationEnum::Set>{},
                                          std::integral_constant<index_t, min_occupancy>{},
                                          std::integral_constant<TailNumber, TailNumber::Full>{});
                    }
                }
            }
            return ave_time;
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
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }
        if constexpr(std::is_same_v<EDataType, ck::half_t> ||
                     std::is_same_v<EDataType, ck::bhalf_t>)
        {
            if(arg.k_batch_ > 1 && ck::is_gfx11_supported())
            {
                // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
                return false;
            }
        }

        if constexpr(!std::is_same_v<CDEElementwiseOperation,
                                     ck::tensor_operation::element_wise::PassThrough> &&
                     !std::is_same_v<CDEElementwiseOperation,
                                     ck::tensor_operation::element_wise::SplitKAdd>)
        {
            if(arg.k_batch_ > 1)
            {
                // Using SplitK and a C element op would require a two stage kernel where the second
                // stage applies the op on the accumulated results
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "C element operators are not supported when using SplitK. Set "
                                 "K_BATCH to 1 or remove the operator."
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                return false;
            }
        }

        if((ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size())) != arg.group_count_)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The group count is not equal to sum of skipped groups "
                             "and kernel args size!"
                          << std::endl;
            }
            return false;
        }

        bool supported = true;
        for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); ++i)
        {

            const auto& a        = arg.gemm_desc_kernel_arg_[i];
            bool group_arg_valid = GridwiseGemm::CheckValidity(a);

            if(not group_arg_valid)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[" << __func__ << "] group id: " << i
                              << " has invalid GridwiseGemm settings!" << std::endl;
                    a.Print();
                }
            }
            supported = supported && group_arg_valid;
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

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceGroupedGemm_Wmma_Fixed_Nk"
            << "<"
            << std::string(ALayout::name)[0] << ","
            << std::string(BLayout::name)[0] << ","
            << std::string(ELayout::name)[0] << ","
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << MRepeat << ", "
            << NRepeat << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer]
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
