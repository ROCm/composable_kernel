// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_fixed_nk_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          GemmSpecialization GemmSpec,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename Block2ETileMap,
          typename GroupedGemmBlock2ETileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_gemm_xdl_fixed_nk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                     const index_t group_count,
                                     const index_t grid_size_grp,
                                     const AElementwiseOperation a_element_op,
                                     const BElementwiseOperation b_element_op,
                                     const CDEElementwiseOperation cde_element_op)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<EGlobalMemoryDataOperation>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        const index_t KBatch = 1;

        const index_t block_id = get_block_1d_id();

        const auto gemm_desc_ptr = reinterpret_cast<const GemmDesc*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id = block_id / grid_size_grp;

        if(group_id >= group_count)
            return;

        const index_t M = gemm_desc_ptr[group_id].M;
        const index_t N = gemm_desc_ptr[group_id].N;
        const index_t K = gemm_desc_ptr[group_id].K;

        if(M == 0 || N == 0 || K == 0)
            return;

        const auto StrideAs = gemm_desc_ptr[group_id].StrideAs;
        const auto StrideBs = gemm_desc_ptr[group_id].StrideBs;
        const auto StrideDs = gemm_desc_ptr[group_id].StrideDs;
        const auto StrideE  = gemm_desc_ptr[group_id].StrideE;

        const auto e_grid_desc_m_n =
            GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

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
            p_as_grid_(i)   = static_cast<ADataType>(gemm_desc_ptr[group_id].p_as_grid[i]);
        });

        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType = remove_cvref_t<decltype(p_bs_grid_(i))>;
            p_bs_grid_(i)   = static_cast<BDataType>(gemm_desc_ptr[group_id].p_bs_grid[i]);
        });

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType = remove_cvref_t<decltype(p_ds_grid_(i))>;
            p_ds_grid_(i)   = static_cast<DDataType>(gemm_desc_ptr[group_id].p_ds_grid[i]);
        });

        index_t id_off   = 0;
        index_t id_local = get_block_1d_id() - BlockStart;

        while(id_local < local_grid_size)
        {
            const auto block_2_etile_map =
                GroupedGemmBlock2ETileMap(local_b2e_tile_map, BlockStart, id_off);

            GridwiseGemm::
                template Run<HasMainKBlockLoop, GemmSpec, AsLayout, BsLayout, DsLayout, ELayout>(
                    p_as_grid_,
                    p_bs_grid_,
                    p_ds_grid_,
                    gemm_desc_ptr[group_id].p_e_grid,
                    p_shared,
                    a_element_op,
                    b_element_op,
                    cde_element_op,
                    M,
                    N,
                    K,
                    StrideAs,
                    StrideBs,
                    StrideDs,
                    StrideE,
                    block_2_etile_map);

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
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
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
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          typename ComputeType    = EDataType,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK
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
    using DeviceOp                         = DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK;
    static constexpr auto WarpTileConfig64 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               true>();
    static constexpr auto WarpTileConfig32 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               false>();
    static constexpr auto NXdlPerWave64    = WarpTileConfig64.At(3);
    static constexpr auto NXdlPerWave32    = WarpTileConfig32.At(3);
    static constexpr index_t NumATensor    = AsDataType::Size();
    static constexpr index_t NumBTensor    = BsDataType::Size();
    static constexpr index_t NumDTensor    = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t NumGemmKPrefetchStage = 1;

    // GridwiseGemm
    template <typename WarpTileConfig>
    using GridwiseGemmBase = GridwiseGemmMultipleABD_xdl_cshuffle<
        AsDataType,
        BsDataType,
        ComputeType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        WarpTileConfig::At(0),
        WarpTileConfig::At(1),
        WarpTileConfig::At(2),
        WarpTileConfig::At(3),
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
        WarpTileConfig::At(4),
        WarpTileConfig::At(5),
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;
    using GridwiseGemm64 = GridwiseGemmBase<decltype(WarpTileConfig64)>;
    using GridwiseGemm32 = GridwiseGemmBase<decltype(WarpTileConfig32)>;

    using Block2ETileMap =
        DeviceGroupedGemm_Fixed_NK_Common::BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops<MPerBlock,
                                                                                         NPerBlock>;
    using GroupedGemmBlock2ETileMap =
        DeviceGroupedGemm_Fixed_NK_Common::OffsettedBlockToCTileMapMLoops<Block2ETileMap, false>;

    using KernelArgument = GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>;

    // Argument
    struct Argument : public BaseArgument
    {

        void UpdateKBatch(index_t) {}

        Argument(std::vector<std::array<const void*, NumATensor>>&,
                 std::vector<std::array<const void*, NumBTensor>>&,
                 std::vector<std::array<const void*, NumDTensor>>&,
                 std::vector<void*>&,
                 std::vector<GemmMultiABDDesc>& gemm_descs,
                 AElementwiseOperation a_element_op   = AElementwiseOperation{},
                 BElementwiseOperation b_element_op   = BElementwiseOperation{},
                 CDEElementwiseOperation c_element_op = CDEElementwiseOperation{})
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

                const auto e_grid_desc_m_n =
                    GridwiseGemm64::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                        AverM, N, StrideE);

                // block-to-e-tile map
                const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

                grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

                if(group_id * grid_size_grp_ != grid_size_)
                {
                    throw std::runtime_error("wrong! grid_size_grp_ is not identical!");
                }

                grid_size_ += grid_size_grp_;

                // check block-to-E-tile
                if(!local_b2c_tile_map.CheckValidity(e_grid_desc_m_n))
                {
                    throw std::runtime_error("wrong! block_2_etile_map validation failed");
                }

                gemm_desc_kernel_arg_.push_back(KernelArgument{
                    p_as_grid,
                    p_bs_grid,
                    p_ds_grid,
                    nullptr,
                    AverM,
                    N,
                    K,
                    StrideAs,
                    StrideBs,
                    StrideDs,
                    StrideE,
                });

                group_id++;
            }

            const auto e_grid_desc_sum_m_n =
                GridwiseGemm64::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                    sum_of_m, gemm_desc_kernel_arg_[0].N, gemm_desc_kernel_arg_[0].StrideE);

            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_sum_m_n, 1};

            barrier_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_sum_m_n);
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

        index_t k_batch_ = 1;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            bool has_main_k_block_loop = true;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                if(GridwiseGemm::CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[i].K) !=
                   has_main_k_block_loop)
                {
                    throw std::runtime_error("wrong! not all gemm has_main_k_block_loop");
                }
            }

            if(arg.grouped_gemm_kernel_args_dev == nullptr)
            {
                throw std::runtime_error("wrong! grouped_gemm_kernel_args_dev is nullptr");
            }

            float ave_time = 0;

            auto launch_kernel = [&](auto has_main_k_block_loop_, auto e_global_memory_operation_) {
                const auto kernel = kernel_grouped_gemm_xdl_fixed_nk<
                    GridwiseGemm,
                    GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>,
                    GemmSpec,
                    AsLayout,
                    BsLayout,
                    DsLayout,
                    ELayout,
                    Block2ETileMap,
                    GroupedGemmBlock2ETileMap,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    e_global_memory_operation_,
                    has_main_k_block_loop_>;

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

            constexpr auto AtomicAdd = InMemoryDataOperationEnum::AtomicAdd;
            constexpr auto Set       = InMemoryDataOperationEnum::Set;

            if(arg.k_batch_ > 1)
            {
                if(has_main_k_block_loop)
                {
                    ave_time =
                        launch_kernel(integral_constant<bool, true>{},
                                      integral_constant<InMemoryDataOperationEnum, AtomicAdd>{});
                }
                else
                {
                    ave_time =
                        launch_kernel(integral_constant<bool, false>{},
                                      integral_constant<InMemoryDataOperationEnum, AtomicAdd>{});
                }
            }
            else
            {
                if(has_main_k_block_loop)
                {
                    ave_time = launch_kernel(integral_constant<bool, true>{},
                                             integral_constant<InMemoryDataOperationEnum, Set>{});
                }
                else
                {
                    ave_time = launch_kernel(integral_constant<bool, false>{},
                                             integral_constant<InMemoryDataOperationEnum, Set>{});
                }
            }

            return ave_time;
        }

        INVOKER_RUN_IMPL

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {

        if(!ck::is_xdl_wmma_supported<ComputeType,
                                      ComputeType,
                                      MPerXDL,
                                      NPerXDL,
                                      WarpTileConfig32.At(0),
                                      WarpTileConfig32.At(1)>())
        {
            return false;
        }

        // Split-K autodeduction is not supported
        if(arg.k_batch_ < 1)
        {
            return false;
        }

        if(ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) != arg.group_count_)
        {
            return false;
        }

        bool supported = true;

        // If we use padding we do not support vector loads for dimensions not divisible by vector
        // load size.
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
            if(get_warp_size() == 64)
            {
                if(GridwiseGemm64::CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[i].K) !=
                   true)
                {
                    supported = false;
                }
            }
            else
            {
                if(GridwiseGemm32::CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[i].K) !=
                   true)
                {
                    supported = false;
                }
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
        str << "DeviceGroupedGemm_Xdl_Fixed_NK"
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
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
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

    static void SetDeviceKernelArgs(Argument& arg, const void* kernel_args)
    {
        arg.grouped_gemm_kernel_args_dev = kernel_args;
    }

    // polymorphic
    void SetDeviceKernelArgs(BaseArgument* p_arg, const void* kernel_args) const override
    {
        return SetDeviceKernelArgs(*dynamic_cast<Argument*>(p_arg), kernel_args);
    }

    void SetElementwiseOps(BaseArgument* p_arg,
                           AElementwiseOperation a_element_op,
                           BElementwiseOperation b_element_op,
                           CDEElementwiseOperation c_element_op) const override
    {

        SetElementwiseOps(
            *dynamic_cast<Argument*>(p_arg), a_element_op, b_element_op, c_element_op);
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        auto arg = *dynamic_cast<const Argument*>(p_arg);

        return arg.group_count_ *
               sizeof(GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>);
    }

#if 0
    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = *dynamic_cast<const Argument*>(p_arg);

        return arg.group_count_ * arg.barrier_size_grp_ * sizeof(uint32_t);
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
#endif

    static void SetKBatch(Argument& arg, index_t k_batch) { arg.UpdateKBatch(k_batch); }

    // polymorphic
    void SetKBatch(BaseArgument* p_arg, index_t k_batch) const override
    {
        return SetKBatch(*dynamic_cast<Argument*>(p_arg), k_batch);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
