// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <numeric>
#include <array>
#include <queue>
#include <vector>
#include <algorithm>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          typename LDSTypeA                           = ComputeTypeA,
          typename LDSTypeB                           = ComputeTypeB>
struct DeviceGemmMultiD_Xdl_CShuffle_V3 : public DeviceGemmMultipleDSplitK<ALayout,
                                                                           BLayout,
                                                                           DsLayout,
                                                                           CLayout,
                                                                           ADataType,
                                                                           BDataType,
                                                                           DsDataType,
                                                                           CDataType,
                                                                           AElementwiseOperation,
                                                                           BElementwiseOperation,
                                                                           CElementwiseOperation>
{
    static constexpr index_t NumDTensor    = DsDataType::Size();
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
    static constexpr auto NXdlPerWave32    = WarpTileConfig32.At(3); // GridwiseGemm
    template <typename WarpTileConfig>
    using GridwiseGemmBase = GridwiseGemmMultiD_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        CLayout,
        ADataType,
        BDataType,
        GemmAccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        LDSTypeA,
        LDSTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<decltype(WarpTileConfig64)>;
    using GridwiseGemm32 = GridwiseGemmBase<decltype(WarpTileConfig32)>;

    using Argument = typename GridwiseGemm64::Argument;

    struct Partitioner
    {
        using DsGridPointer = typename GridwiseGemm64::DsGridPointer;

        index_t M;
        index_t N;
        index_t StrideA;
        index_t StrideB;
        std::array<index_t, NumDTensor> StrideDs;
        index_t StrideC;

        static constexpr long_index_t TwoGB    = INT32_MAX;
        static constexpr index_t PartitionSize = 256;

        Partitioner() = default;
        Partitioner(index_t M_,
                    index_t N_,
                    index_t StrideA_,
                    index_t StrideB_,
                    std::array<index_t, NumDTensor> StrideDs_,
                    index_t StrideC_)
            : M{M_},
              N{N_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideDs{StrideDs_},
              StrideC{StrideC_}
        {
        }

        __host__ bool isPartitionable() const
        {
            bool row_major = is_same<CLayout, tensor_layout::gemm::RowMajor>::value &&
                             is_same<ALayout, tensor_layout::gemm::RowMajor>::value;
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                row_major &= is_same<DLayout, tensor_layout::gemm::RowMajor>::value;
            });

            bool is_B_descriptor_smaller_than_2GB =
                (static_cast<long_index_t>(N) * static_cast<long_index_t>(StrideB) *
                 sizeof(BDataType)) <= TwoGB;

            return (row_major && is_B_descriptor_smaller_than_2GB) ||
                   areDescriptorsSmallerThan2GB();
        }

        __host__ bool areDescriptorsSmallerThan2GB(index_t m) const
        {

            const bool is_A_descriptor_smaller_than_2GB =
                (static_cast<long_index_t>(m) * static_cast<long_index_t>(StrideA) *
                 sizeof(ADataType)) <= TwoGB;
            const bool is_C_descriptor_smaller_than_2GB =
                (static_cast<long_index_t>(m) * static_cast<long_index_t>(StrideC) *
                 sizeof(CDataType)) <= TwoGB;
            bool are_Ds_descriptors_smaller_than_2GB = true;
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                are_Ds_descriptors_smaller_than_2GB &=
                    (static_cast<long_index_t>(m) * static_cast<long_index_t>(StrideDs[i]) *
                     sizeof(DDataType)) <= TwoGB;
            });

            return is_A_descriptor_smaller_than_2GB && is_C_descriptor_smaller_than_2GB &&
                   are_Ds_descriptors_smaller_than_2GB;
        }

        __host__ bool areDescriptorsSmallerThan2GB() const
        {
            return areDescriptorsSmallerThan2GB(M);
        }

        // Gemm specific size check. Adding it to the grid changes convolution behaviour.
        template <typename Argument>
        __host__ static bool isDescriptorValidForGemm(const Argument& arg)
        {
            return static_cast<long_index_t>(arg.M) * static_cast<long_index_t>(arg.K) *
                           sizeof(ADataType) <=
                       TwoGB &&
                   static_cast<long_index_t>(arg.N) * static_cast<long_index_t>(arg.K) *
                           sizeof(BDataType) <=
                       TwoGB &&
                   static_cast<long_index_t>(arg.M) * static_cast<long_index_t>(arg.N) *
                           sizeof(CDataType) <=
                       TwoGB;
        }

        __host__ auto splitProblem(index_t m,
                                   const ADataType* p_a_grid_left,
                                   DsGridPointer& p_ds_grid_left,
                                   CDataType* p_c_grid_left) const
        {
            const index_t m_left  = ck::math::integer_least_multiple(m / 2, PartitionSize);
            const index_t m_right = m - m_left;

            const long_index_t a_right_offset = static_cast<long_index_t>(m_left) * StrideA;
            const long_index_t c_right_offset = static_cast<long_index_t>(m_left) * StrideC;

            const auto ds_grid_right_ptr = generate_tuple(
                [&](auto i) {
                    const long_index_t ds_right_offset =
                        static_cast<long_index_t>(m_left) * StrideDs[i];
                    return p_ds_grid_left(i) + ds_right_offset;
                },
                Number<NumDTensor>{});

            return ck::make_tuple(m_left,
                                  m_right,
                                  p_a_grid_left + a_right_offset,
                                  ds_grid_right_ptr,
                                  p_c_grid_left + c_right_offset);
        }

        template <typename ArgumentIn, typename ArgumentOut = ArgumentIn>
        std::vector<ArgumentOut> partitionGemmProblem(ArgumentIn const& arg) const
        {
            static constexpr index_t InitialSubArgsSize = 32;

            std::vector<ArgumentOut> sub_arguments;
            sub_arguments.reserve(InitialSubArgsSize);

            std::queue<index_t> split_m({arg.M});
            std::queue<const ADataType*> a_grid_ptrs_queue({arg.p_a_grid});
            std::queue<DsGridPointer> ds_grid_ptrs_queue({arg.p_ds_grid});
            std::queue<CDataType*> c_grid_ptrs_queue({arg.p_c_grid});

            // Algorithm:
            // While queue is not empty:
            //  1. Get batch data from queue.
            //  2. If descs are smaller than 2GB push to result array.
            //  3. If descs are bigger than 2GB split into left and right transformer.
            //  and push the both into the queue.
            while(!split_m.empty())
            {
                index_t m                   = split_m.front();
                const ADataType* a_grid_ptr = a_grid_ptrs_queue.front();
                DsGridPointer ds_grid_ptr   = ds_grid_ptrs_queue.front();
                CDataType* c_grid_ptr       = c_grid_ptrs_queue.front();

                // m <= PartitionSize and descriptors larger then 2GB should not happen.
                // If it does the gemm will be rejected when verifying its argument in the invoker.
                if(areDescriptorsSmallerThan2GB(m) || (m <= PartitionSize))
                {
                    ArgumentOut newArg{a_grid_ptr,
                                       arg.p_b_grid,
                                       {},
                                       c_grid_ptr,
                                       m,
                                       arg.N,
                                       arg.K,
                                       arg.StrideA,
                                       arg.StrideB,
                                       arg.StrideDs,
                                       arg.StrideC,
                                       arg.KBatch,
                                       arg.a_element_op,
                                       arg.b_element_op,
                                       arg.c_element_op};
                    newArg.p_ds_grid = ds_grid_ptr;
                    sub_arguments.emplace_back(std::move(newArg));
                }
                else
                {
                    index_t left_m, right_m;
                    const ADataType* a_grid_right_ptr;
                    DsGridPointer ds_grid_right_ptr;
                    CDataType* c_grid_right_ptr;

                    ck::tie(
                        left_m, right_m, a_grid_right_ptr, ds_grid_right_ptr, c_grid_right_ptr) =
                        splitProblem(m, a_grid_ptr, ds_grid_ptr, c_grid_ptr);

                    split_m.push(left_m);
                    split_m.push(right_m);

                    a_grid_ptrs_queue.push(a_grid_ptr);
                    a_grid_ptrs_queue.push(a_grid_right_ptr);
                    ds_grid_ptrs_queue.push(ds_grid_ptr);
                    ds_grid_ptrs_queue.push(ds_grid_right_ptr);
                    c_grid_ptrs_queue.push(c_grid_ptr);
                    c_grid_ptrs_queue.push(c_grid_right_ptr);
                }

                split_m.pop();
                a_grid_ptrs_queue.pop();
                ds_grid_ptrs_queue.pop();
                c_grid_ptrs_queue.pop();
            }

            return sub_arguments;
        }
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        template <typename GridwiseGemm>
        float RunImpSinglePartition(const typename GridwiseGemm::Argument& arg,
                                    const StreamConfig& stream_config)
        {
            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {

                    std::array<std::size_t, NumDTensor> DsSize;

                    auto arg_ = arg;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideB, arg_.BK0);

                    auto size_a_buffer =
                        a_grid_desc_ak0_m_ak1.GetElementSpaceSize() * sizeof(ADataType);
                    auto size_b_buffer =
                        b_grid_desc_bk0_n_bk1.GetElementSpaceSize() * sizeof(BDataType);

                    const auto ds_grid_desc_m_n = GridwiseGemm::MakeDsGridDescriptor_M_N(
                        arg_.M, arg_.MPadded, arg_.N, arg_.NPadded, arg_.StrideDs);

                    static_for<0, NumDTensor, 1>{}([&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        DsSize[i] = ds_grid_desc_m_n[i].GetElementSpaceSize() * sizeof(DDataType);
                    });
                    ck::utility::RotatingMemWrapperMultiD<typename GridwiseGemm::Argument,
                                                          DsDataType>
                        rotating_mem(arg_,
                                     stream_config.rotating_count,
                                     size_a_buffer,
                                     size_b_buffer,
                                     DsSize);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            hipGetErrorString(hipMemsetAsync(arg_.p_c_grid,
                                                             0,
                                                             arg_.M * arg_.N * sizeof(CDataType),
                                                             stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_);
                }
                else
                {
                    if(arg.KBatch > 1)
                        hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
                                                         0,
                                                         arg.M * arg.N * sizeof(CDataType),
                                                         stream_config.stream_id_));

                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
                }
            };

            constexpr index_t minimum_occupancy = []() {
                if constexpr(BlkGemmPipeSched == BlockGemmPipelineScheduler::Interwave)
                {
                    return 2;
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    return (MPerBlock * NPerBlock / BlockSize <= 128) ? 2 : 1;
                }
                else
                {
                    return 1;
                }
            }();

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                            GridwiseGemm,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3_multi_d<GridwiseGemm,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy>;
                        Run(kernel);
                    }
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_multi_d<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_multi_d<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
                else
                {
                    if(arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_multi_d<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_multi_d<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_d<
                            GridwiseGemm,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3_multi_d<GridwiseGemm,
                                                                false,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        template <typename GridwiseGemm>
        float RunImp(const typename GridwiseGemm::Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            Partitioner partitioner(
                arg.M, arg.N, arg.StrideA, arg.StrideB, arg.StrideDs, arg.StrideC);
            auto sub_arguments = partitioner.partitionGemmProblem(arg);
            return std::accumulate(sub_arguments.begin(),
                                   sub_arguments.end(),
                                   0.0f,
                                   [&](float sum, const auto& sub_arg) {
                                       return sum + RunImpSinglePartition<GridwiseGemm>(
                                                        sub_arg, stream_config);
                                   });
        }

        INVOKER_RUN3_IMPL

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
        if(!ck::is_xdl_wmma_supported<ComputeTypeA,
                                      ComputeTypeB,
                                      MPerXDL,
                                      NPerXDL,
                                      WarpTileConfig32.At(0),
                                      WarpTileConfig32.At(1)>())
        {
            return false;
        }
        if(is_gfx11_supported() && arg.KBatch > 1)
        {
            return false;
        }
        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> && arg.KBatch > 1)
        {
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        Partitioner partitioner(arg.M, arg.N, arg.StrideA, arg.StrideB, arg.StrideDs, arg.StrideC);
        // True if problem is partitionable or valid without partitioning.
        if(!partitioner.isPartitionable())
        {
            return false;
        }

        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                auto sub_arguments = partitioner.partitionGemmProblem(arg);
                return std::all_of(
                    sub_arguments.begin(), sub_arguments.end(), [](const auto& sub_arg) {
                        return GridwiseGemm64::CheckValidity(sub_arg) &&
                               Partitioner::isDescriptorValidForGemm(sub_arg);
                    });
            }
        }
        if(CDEShuffleBlockTransferScalarPerVectors{}[Number<0>{}] <= 1 && (arg.KBatch > 1))
        {
            return false;
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                auto sub_arguments =
                    partitioner.template partitionGemmProblem<typename GridwiseGemm64::Argument,
                                                              typename GridwiseGemm32::Argument>(
                        arg);
                return std::all_of(
                    sub_arguments.begin(), sub_arguments.end(), [](const auto& sub_arg) {
                        return GridwiseGemm32::CheckValidity(sub_arg) &&
                               Partitioner::isDescriptorValidForGemm(sub_arg);
                    });
            }
        }

        return false;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        p_ds,
                        static_cast<CDataType*>(p_c),
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideC,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      std::array<const void*, NumDTensor> p_ds,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      std::array<ck::index_t, NumDTensor> StrideDs,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          p_ds,
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideC,
                                          KBatch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
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
        str << "DeviceGemmMultiD_Xdl_CShuffle_V3"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerXDL<<"x"<<NPerXDL << ", "
            << "WaveMap: "
            << MXdlPerWave<<"x" << NXdlPerWave<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm64::BlockwiseGemmPipe::PrefetchStages << ", "
            << "AK1: "
            << AK1 << ", "
            << "BK1: "
            << BK1;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
