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
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_fixed_nk_common.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename Block2ETileMap,
          typename GroupedGemmBlock2ETileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_gemm_wmma_fixed_nk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                      const index_t group_count,
                                      const index_t grid_size_grp,
                                      const AElementwiseOperation a_element_op,
                                      const BElementwiseOperation b_element_op,
                                      const CDEElementwiseOperation cde_element_op)
{
#if defined(__gfx11__) || defined(__gfx12__)
    using EpilogueType = typename std::conditional<GridwiseGemm::IsBWaveTransferApplicable &&
                                                       GridwiseGemm::UseDirectStore,
                                                   typename GridwiseGemm::EpilogueDirectStore,
                                                   typename GridwiseGemm::EpilogueCShuffle>::type;

    constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<EpilogueType>();
    __shared__ char p_shared[LDS_size];

    const index_t KBatch = 1;

    const index_t block_id = get_block_1d_id();

    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    const index_t group_id = block_id / grid_size_grp;

    if(group_id >= group_count)
        return;

    auto karg = gemm_desc_ptr[group_id];

    if(karg.M == 0 || karg.N == 0 || karg.K == 0)
        return;

#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<typename GridwiseGemm::EDataType_, ck::half_t> ||
                    std::is_same_v<typename GridwiseGemm::EDataType_, ck::bhalf_t>)))
#endif
    {

        typename GridwiseGemm::Problem problem(karg.M,
                                               karg.N,
                                               karg.K,
                                               karg.StrideAs,
                                               karg.StrideBs,
                                               karg.StrideDs,
                                               karg.StrideE,
                                               KBatch);

        const auto e_grid_desc_m_n = GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideE);

        const index_t BlockStart = group_id * grid_size_grp;

        const auto local_b2e_tile_map = Block2ETileMap{e_grid_desc_m_n, KBatch};

        const auto local_grid_size = local_b2e_tile_map.CalculateGridSize(e_grid_desc_m_n);

        constexpr auto NumATensor = GridwiseGemm::AsGridPointer::Size();
        constexpr auto NumBTensor = GridwiseGemm::BsGridPointer::Size();
        constexpr auto NumDTensor = GridwiseGemm::DsGridPointer::Size();

        typename GridwiseGemm::AsGridPointer p_as_grid_;
        typename GridwiseGemm::BsGridPointer p_bs_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;

        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType = remove_cvref_t<decltype(p_as_grid_(i))>;
            p_as_grid_(i)   = static_cast<ADataType>(karg.p_as_grid[i]);
        });

        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType = remove_cvref_t<decltype(p_bs_grid_(i))>;
            p_bs_grid_(i)   = static_cast<BDataType>(karg.p_bs_grid[i]);
        });

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType = remove_cvref_t<decltype(p_ds_grid_(i))>;
            p_ds_grid_(i)   = static_cast<DDataType>(karg.p_ds_grid[i]);
        });

        index_t id_off   = 0;
        index_t id_local = get_block_1d_id() - BlockStart;

        while(id_local < local_grid_size)
        {
            const auto block_2_etile_map =
                GroupedGemmBlock2ETileMap(local_b2e_tile_map, BlockStart, id_off);

            auto epilogue_args = EpilogueType{};

            GridwiseGemm::template Run<HasMainKBlockLoop,
                                       EGlobalMemoryDataOperation,
                                       TailNum,
                                       decltype(block_2_etile_map),
                                       EpilogueType,
                                       1,
                                       2>(
                p_as_grid_,
                p_bs_grid_,
                p_ds_grid_,
                static_cast<typename GridwiseGemm::EDataType_*>(karg.p_e_grid),
                p_shared,
                problem,
                block_2_etile_map,
                a_element_op,
                b_element_op,
                cde_element_op,
                epilogue_args);

            id_off += grid_size_grp;
            id_local += grid_size_grp;
            block_sync_lds();
        }
    }
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = grid_size_grp;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
#endif
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
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
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
        typename uniform_sequence_gen<NumDTensor + 1,
                                      CDEBlockTransferScalarPerVector_NPerBlock>::type,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,
        false>;

    using Block2ETileMap =
        DeviceGroupedGemm_Fixed_NK_Common::BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops<MPerBlock,
                                                                                         NPerBlock>;
    using GroupedGemmBlock2ETileMap =
        DeviceGroupedGemm_Fixed_NK_Common::OffsettedBlockToCTileMapMLoops<Block2ETileMap>;

    static constexpr index_t DefaultKBatch = 1; // implementation only supports KBatch == 1
    using KernelArgument                   = typename GridwiseGemm::Argument;

    using GemmTransKernelArg =
        GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>;

    static constexpr bool CalculateHasMainKBlockLoop(const GemmTransKernelArg& karg,
                                                     index_t k_batch)
    {
        index_t k_grain = k_batch * KPerBlock;
        index_t K_split = (karg.K + k_grain - 1) / k_batch;
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

        // Client is expected to manually copy the kernel arguments to the device therefore there is
        // no point in setting tensor device pointers for the argument structure.
        Argument(std::vector<std::array<const void*, NumATensor>>&,
                 std::vector<std::array<const void*, NumBTensor>>&,
                 std::vector<std::array<const void*, NumDTensor>>&,
                 std::vector<void*>&,
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
            gemm_desc_kernel_arg_.reserve(group_count_);

            index_t group_id = 0;

            sum_of_m              = gemm_descs[0].M_;
            const index_t AverM   = math::integer_divide_ceil(sum_of_m, group_count_);
            const index_t fixed_N = gemm_descs[0].N_;
            const index_t fixed_K = gemm_descs[0].K_;

            for(std::size_t g = 0; g < gemm_descs.size(); g++)
            {
                const index_t M = gemm_descs[g].M_;
                const index_t N = gemm_descs[g].N_;
                const index_t K = gemm_descs[g].K_;

                if(M != sum_of_m || N != fixed_N || K != fixed_K)
                {
                    throw std::runtime_error("wrong! M/N/K is not identical");
                }

                a_mtx_mraw_kraw_.emplace_back(sum_of_m, K);
                b_mtx_nraw_kraw_.emplace_back(N, K);

                // pointer
                std::array<const void*, NumATensor> p_as_grid;
                std::array<const void*, NumBTensor> p_bs_grid;
                std::array<const void*, NumDTensor> p_ds_grid;

                static_for<0, NumATensor, 1>{}([&](auto i) { p_as_grid[i] = nullptr; });
                static_for<0, NumBTensor, 1>{}([&](auto i) { p_bs_grid[i] = nullptr; });
                static_for<0, NumDTensor, 1>{}([&](auto i) { p_ds_grid[i] = nullptr; });

                std::array<index_t, NumATensor> StrideAs;
                std::array<index_t, NumBTensor> StrideBs;
                std::array<index_t, NumDTensor> StrideDs;

                const index_t StrideE = gemm_descs[g].stride_C_;

                if(gemm_descs[g].stride_As_.size() != NumATensor)
                {
                    throw std::runtime_error(
                        "wrong! gemm_descs[i].stride_As_.size() does not match NumATensor");
                }

                static_for<0, NumATensor, 1>{}(
                    [&](auto j) { StrideAs[j] = gemm_descs[g].stride_As_[j]; });

                if(gemm_descs[g].stride_Bs_.size() != NumBTensor)
                {
                    throw std::runtime_error(
                        "wrong! gemm_descs[i].stride_Bs_.size() does not match NumBTensor");
                }

                static_for<0, NumBTensor, 1>{}(
                    [&](auto j) { StrideBs[j] = gemm_descs[g].stride_Bs_[j]; });

                if(gemm_descs[g].stride_Ds_.size() != NumDTensor)
                {
                    throw std::runtime_error(
                        "wrong! gemm_descs[i].stride_Ds_.size() does not match NumDTensor");
                }

                static_for<0, NumDTensor, 1>{}(
                    [&](auto j) { StrideDs[j] = gemm_descs[g].stride_Ds_[j]; });

                const auto e_grid_desc_m_n =
                    GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                        AverM, AverM, N, N, StrideE);

                // block-to-e-tile map
                const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

                grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

                if(group_id * grid_size_grp_ != grid_size_)
                {
                    throw std::runtime_error("wrong! grid_size_grp_ is not identical!");
                }

                const index_t block_start = grid_size_;

                grid_size_ += grid_size_grp_;

                if(!local_b2c_tile_map.CheckValidity(e_grid_desc_m_n))
                {
                    throw std::runtime_error("wrong! block_2_etile_map validation failed");
                }

                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                auto karg = GemmTransKernelArg({p_as_grid,
                                                p_bs_grid,
                                                p_ds_grid,
                                                nullptr,
                                                AverM,
                                                N,
                                                K,
                                                StrideAs,
                                                StrideBs,
                                                StrideDs,
                                                StrideE});

                gemm_desc_kernel_arg_.emplace_back(std::move(karg));

                group_id++;
            }
        }

        void UpdateKBatch(index_t) {}

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
        index_t sum_of_m;

        index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(arg.grouped_gemm_kernel_args_dev == nullptr)
            {
                throw std::runtime_error("wrong! grouped_gemm_kernel_args_dev is nullptr");
            }

            if(arg.k_batch_ != 1)
            {
                throw std::runtime_error("Split K functionality is not supported for wmma multi "
                                         "abd fixed nk implementation.");
            }

            float ave_time = 0;

            auto launch_kernel = [&](auto e_global_memory_operation_) {
                const auto kernel = kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                                      GemmTransKernelArg,
                                                                      true, // has_main_k_block_loop
                                                                      e_global_memory_operation_,
                                                                      AsLayout,
                                                                      BsLayout,
                                                                      DsLayout,
                                                                      ELayout,
                                                                      Block2ETileMap,
                                                                      GroupedGemmBlock2ETileMap,
                                                                      AElementwiseOperation,
                                                                      BElementwiseOperation,
                                                                      CDEElementwiseOperation,
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
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.c_element_op_);
            };

            constexpr auto Set = InMemoryDataOperationEnum::Set;
            ave_time           = launch_kernel(integral_constant<InMemoryDataOperationEnum, Set>{});

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
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }

        if(ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) != arg.group_count_)
        {
            return false;
        }

        bool supported = true;

        // If we use padding we do not support vector loads for dimensions not divisible by
        // vector load size.
        if constexpr(GemmSpec != GemmSpecialization::Default)
        {
            // [A|B]BlockTransferSrcVectorDim value define dimension in the block {K0,M,K1} layout,
            // thus we have to adapt it to the {M,K} or {N,K} layout.
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

        for(index_t i = 0; i < arg.group_count_; i++)
        {
            if(CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[i], arg.k_batch_) != true)
            {
                supported = false;
            }
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

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        auto arg = *dynamic_cast<const Argument*>(p_arg);

        return arg.group_count_ *
               sizeof(GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>);
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto p_arg_ = dynamic_cast<const Argument*>(p_arg);
        if(p_arg_)
        {
            return p_arg_->gemm_desc_kernel_arg_.size() * sizeof(GemmTransKernelArg);
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemm_Wmma_Multi_ABD_Fixed_NK::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& stream_config = StreamConfig{}) const override
    {
        auto p_arg_          = dynamic_cast<Argument*>(p_arg);
        p_arg_->p_workspace_ = p_workspace;

        hip_check_error(
            hipMemsetAsync(p_workspace, 0, GetWorkSpaceSize(p_arg), stream_config.stream_id_));
    }

    static void SetKBatch(Argument& arg, index_t k_batch) { arg.UpdateKBatch(k_batch); }

    // polymorphic
    void SetKBatch(BaseArgument* p_arg, index_t k_batch) const override
    {
        return SetKBatch(*dynamic_cast<Argument*>(p_arg), k_batch);
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
