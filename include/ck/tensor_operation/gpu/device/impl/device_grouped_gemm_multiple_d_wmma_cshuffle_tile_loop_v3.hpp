// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <tuple>

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/stream_utility.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

///
/// @brief      Entry point kernel for device-wide Grouped GEMM operation.
///
/// @param[in]  gemm_descs_const  The pointer to the array of GEMM descriptor structures.
/// @param[in]  group_count       The number of together processed GEMMs.
///
/// @tparam     GridwiseGemm                The specific GridwiseGEMM algorithm implementation.
/// @tparam     GemmDesc                    The structure holding all necessary descriptors and
///                                         other data needed for grouped gemm calculation and work
///                                         distribution.
/// @tparam     LocalBlock2ETileMap         The structure providing mapping between workgroup ids,
///                                         the data tiles to process and the output tiles.
///
template <typename GridwiseGemm,
          typename GemmDesc,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          index_t KPerBlock,
          typename OffsettedBlockToCTileMap,
          typename LocalBlock2ETileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_gemm_multiple_d_wmma(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                        const index_t group_count,
                                        const AElementwiseOperation a_element_op,
                                        const BElementwiseOperation b_element_op,
                                        const CDEElementwiseOperation cde_element_op)
{
#if(defined(__gfx11__) || defined(__gfx12__))
    using EpilogueType = typename std::conditional<GridwiseGemm::IsBWaveTransferApplicable &&
                                                       GridwiseGemm::UseDirectStore,
                                                   typename GridwiseGemm::EpilogueDirectStore,
                                                   typename GridwiseGemm::EpilogueCShuffle>::type;

    constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<EpilogueType>();
    __shared__ uint8_t p_shared[LDS_size];

    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    constexpr auto NumDTensor = DsDataType::Size();
    index_t tile_id           = get_block_1d_id();
    index_t tile_offset       = 0;
    index_t group_id          = -1;
    index_t group_offset      = 0;
    index_t grid_size_grp     = 0;

    index_t gemm_tile_id_start = 0;
    index_t gemm_tile_id_end   = 0;

    index_t M = 0, N = 0, K = 0;

    auto b2c_tile_map = OffsettedBlockToCTileMap(LocalBlock2ETileMap(1, 1), 1, 1);

    do
    {
        // Find corresponding GEMM group for our tile
        while(!(tile_id >= gemm_tile_id_start && tile_id < gemm_tile_id_end) &&
              group_id < group_count)
        {
            group_offset += grid_size_grp;
            group_id++;

            if(group_id >= group_count)
                return;

            M = gemm_desc_ptr[group_id].M;
            N = gemm_desc_ptr[group_id].N;
            K = gemm_desc_ptr[group_id].K;

            if(M == 0 || N == 0 || K == 0)
            {
                grid_size_grp = 0;
                continue;
            }

            b2c_tile_map =
                OffsettedBlockToCTileMap(LocalBlock2ETileMap(M, N, 4), group_offset, tile_offset);
            grid_size_grp = b2c_tile_map.CalculateGridSize(M, N);

            gemm_tile_id_start = group_offset;
            gemm_tile_id_end   = group_offset + grid_size_grp;
        }

        // Create A&B grid pointer containing their single tensors
        typename GridwiseGemm::AsGridPointer p_as_grid = Tuple<const ADataType*>(
            static_cast<const ADataType*>(gemm_desc_ptr[group_id].p_a_grid));
        typename GridwiseGemm::BsGridPointer p_bs_grid = Tuple<const BDataType*>(
            static_cast<const BDataType*>(gemm_desc_ptr[group_id].p_b_grid));

        // Make a DsGridPointer instance containing all D tensors
        using DsGridPointer = decltype(GridwiseGemm::MakeDsGridPointer());
        DsGridPointer p_ds_grid;
        std::array<index_t, NumDTensor> stride_Ds;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            p_ds_grid(i)    = static_cast<const DDataType*>(gemm_desc_ptr[group_id].p_ds_grid[i]);
            stride_Ds[i]    = gemm_desc_ptr[group_id].StrideDs[i];
        });

        index_t K_split                  = ck::math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
        const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

        // Update tile offset if we have moved within group
        b2c_tile_map.UpdateTileOffset(tile_offset);

        using Problem = typename GridwiseGemm::Problem;
        auto problem  = Problem(gemm_desc_ptr[group_id].M,
                               gemm_desc_ptr[group_id].N,
                               gemm_desc_ptr[group_id].K,
                               std::array<index_t, 1>{gemm_desc_ptr[group_id].StrideA},
                               std::array<index_t, 1>{gemm_desc_ptr[group_id].StrideB},
                               stride_Ds,
                               gemm_desc_ptr[group_id].StrideE,
                               1);

        auto epilogue_args           = EpilogueType{};
        constexpr TailNumber TailNum = TailNumber::Full;

        if(has_main_k_block_loop)
        {
            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                         BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
            {
                GridwiseGemm::template Run<true, InMemoryDataOperationEnum::Set, TailNum>(
                    p_as_grid,
                    p_bs_grid,
                    p_ds_grid,
                    static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                    static_cast<void*>(p_shared),
                    problem,
                    b2c_tile_map,
                    a_element_op,
                    b_element_op,
                    cde_element_op,
                    epilogue_args);
            }
        }
        else
        {
            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
            {
                GridwiseGemm::template Run<false, InMemoryDataOperationEnum::Set, TailNum>(
                    p_as_grid,
                    p_bs_grid,
                    p_ds_grid,
                    static_cast<EDataType*>(gemm_desc_ptr[group_id].p_e_grid),
                    static_cast<void*>(p_shared),
                    problem,
                    b2c_tile_map,
                    a_element_op,
                    b_element_op,
                    cde_element_op,
                    epilogue_args);
            }
        }

        tile_id += get_grid_size();
        tile_offset += get_grid_size();

    } while(group_id < group_count);
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
#endif // end of if (defined(__gfx11__) || defined(__gfx12__))
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
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA>

struct DeviceGroupedGemmMultipleD_Wmma_CShuffle_TileLoop_V3
    : public DeviceGroupedGemmTileLoop<ALayout,
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
    using DeviceOp = DeviceGroupedGemmMultipleD_Wmma_CShuffle_TileLoop_V3;

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
        CDEBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,  // PermuteA not supported by GridwiseOp.
        false>; // PermuteB not supported by DeviceGroupedGemmTileLoop base class.

    using KernelConfig    = TileLoopKernelConfig<BlockSize>;
    using KernelArguments = GroupedGemmKernelArgument<NumDTensor>;
    using Block2ETileMap  = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;
    using OffsettedLocalBlock2ETileMap = OffsettedBlockToCTileMap2<Block2ETileMap>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*>& /* p_As */,
                 std::vector<const void*>& /* p_Bs */,
                 std::vector<std::array<const void*, NumDTensor>>& /* p_Ds */,
                 std::vector<void*>& /* p_Es */,
                 const std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op,
                 int occupancy_num_blocks,
                 int gpu_cu_count)
            : group_count_{static_cast<index_t>(gemm_descs.size())},
              occupancy_num_blocks_{occupancy_num_blocks},
              gpu_cu_count_{gpu_cu_count},
              gemm_descs_{gemm_descs},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              tile_count_{0}
        {
            for(const auto& desc : gemm_descs)
            {
                const auto M            = desc.M_;
                const auto N            = desc.N_;
                const auto b2c_tile_map = Block2ETileMap(M, N);
                tile_count_ += b2c_tile_map.CalculateGridSize(M, N);
            }
        }

        index_t group_count_;
        const void* p_dev_gemm_args_;
        int occupancy_num_blocks_;
        int gpu_cu_count_;
        const std::vector<GemmDesc>& gemm_descs_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
        index_t tile_count_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using user provided device buffer for kernel
        ///             arguments.
        ///
        /// @param[in]  arg                 The structure containing kernel arguments (in host
        ///                                 memory).
        /// @param[in]  dev_gemm_args       The pointer to device memory with kernel arguments.
        /// @param[in]  stream_config       The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        float Run(const Argument& arg,
                  const void* dev_gemm_args,
                  const StreamConfig& stream_config = StreamConfig{})
        {
            if(dev_gemm_args == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!" << " In " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            const auto kernel = GetKernelFunction();

            int grid_size = KernelConfig::CalculateMaxOccupancyGridSize(kernel, stream_config);

            if(stream_config.log_level_ > 0)
            {
                std::cout << "grid_size: " << grid_size << " tile_count: " << arg.tile_count_
                          << std::endl;
            }

            // run multiple kernels

            return launch_and_time_kernel(stream_config,
                                          kernel,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          cast_pointer_to_constant_address_space(dev_gemm_args),
                                          arg.group_count_,
                                          arg.a_element_op_,
                                          arg.b_element_op_,
                                          arg.cde_element_op_);
        }

        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using device buffers (for kernel arguments and
        ///             for kernel auxiliary workspace) provided with an argument. The user should
        ///             call @see GetDeviceKernelArgSize, and @see SetDeviceKernelArgs, on arg
        ///             parameter to properly allocate those buffers.
        ///
        /// @param[in]  arg            The structure containing kernel arguments (in host memory).
        /// @param[in]  stream_config  The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(arg.p_dev_gemm_args_ == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!" << " In " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            return Run(arg, arg.p_dev_gemm_args_, stream_config);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static auto GetKernelFunction()
    {
        const auto kernel = kernel_grouped_gemm_multiple_d_wmma<GridwiseGemm,
                                                                KernelArguments,
                                                                ADataType,
                                                                BDataType,
                                                                DsDataType,
                                                                EDataType,
                                                                ALayout,
                                                                BLayout,
                                                                DsLayout,
                                                                ELayout,
                                                                KPerBlock,
                                                                OffsettedLocalBlock2ETileMap,
                                                                Block2ETileMap,
                                                                AElementwiseOperation,
                                                                BElementwiseOperation,
                                                                CDEElementwiseOperation,
                                                                BlkGemmPipeSched,
                                                                BlkGemmPipelineVer>;
        return kernel;
    }

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }
        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                return false;
            }
        }

        bool supported = true;
        for(index_t i = 0; i < arg.group_count_; ++i)
        {

            if(ck::is_gfx12_supported() && !GridwiseGemm::CheckValidityAWaveTransfer(
                                               arg.gemm_descs_[i].M_, arg.gemm_descs_[i].K_))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Wave Transfer not applicable for matrix A" << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }

            if(ck::is_gfx12_supported() && !GridwiseGemm::CheckValidityBWaveTransfer(
                                               arg.gemm_descs_[i].N_, arg.gemm_descs_[i].K_))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Wave Transfer not applicable for matrix B" << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }

            std::array<const void*, NumDTensor> placeholder_p_ds_grid{};
            std::array<index_t, NumDTensor> stride_Ds;
            std::copy_n(arg.gemm_descs_[i].stride_Ds_.begin(), NumDTensor, stride_Ds.begin());

            typename GridwiseGemm::Argument gridwise_arg(
                std::array<const void*, 1>{nullptr}, // p_a_grid,
                std::array<const void*, 1>{nullptr}, // p_b_grid,
                placeholder_p_ds_grid,               // p_ds_grid,
                nullptr,                             // p_e_grid  ,
                arg.gemm_descs_[i].M_,
                arg.gemm_descs_[i].N_,
                arg.gemm_descs_[i].K_,
                std::array<index_t, 1>{arg.gemm_descs_[i].stride_A_},
                std::array<index_t, 1>{arg.gemm_descs_[i].stride_B_},
                stride_Ds,
                arg.gemm_descs_[i].stride_C_,
                1, // KBatch
                arg.a_element_op_,
                arg.b_element_op_,
                arg.cde_element_op_,
                false);

            bool group_arg_valid = GridwiseGemm::CheckValidity(gridwise_arg);
            supported            = supported && group_arg_valid;

            if(!group_arg_valid)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[" << __func__ << "] group id: " << i
                              << " has invalid GridwiseGemm settings!" << std::endl;
                    gridwise_arg.Print();
                }
            }
        }

        return supported;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static int GetKernelOccupancy()
    {
        const auto kernel = GetKernelFunction();
        return KernelConfig::GetKernelOccupancy(kernel);
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc>& gemm_descs,
                             AElementwiseOperation a_elementwise_op,
                             BElementwiseOperation b_elementwise_op,
                             CDEElementwiseOperation cde_elementwise_op)
    {
        int occupancy = GetKernelOccupancy();
        int num_cu    = KernelConfig::GetComputeUnitCount();

        return Argument{p_As,
                        p_Bs,
                        p_Ds,
                        p_Es,
                        gemm_descs,
                        a_elementwise_op,
                        b_elementwise_op,
                        cde_elementwise_op,
                        occupancy,
                        num_cu};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_elementwise_op,
                        BElementwiseOperation b_elementwise_op,
                        CDEElementwiseOperation cde_elementwise_op) override
    {
        int occupancy = GetKernelOccupancy();
        int num_cu    = KernelConfig::GetComputeUnitCount();

        return std::make_unique<Argument>(p_As,
                                          p_Bs,
                                          p_Ds,
                                          p_Es,
                                          gemm_descs,
                                          a_elementwise_op,
                                          b_elementwise_op,
                                          cde_elementwise_op,
                                          occupancy,
                                          num_cu);
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::ostringstream();

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
        str << "DeviceGroupedGemmMultipleD_Wmma_CShuffle_TileLoop_V3"
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

    void SetDeviceKernelArgs(Argument& arg,
                             void* p_dev_kernel_args,
                             const void* p_host_kernel_args) const
    {
        arg.p_dev_gemm_args_ = p_dev_kernel_args;
        hip_check_error(hipMemcpyAsync(p_dev_kernel_args,
                                       p_host_kernel_args,
                                       GetDeviceKernelArgSize(&arg),
                                       hipMemcpyHostToDevice));
    }

    virtual void SetDeviceKernelArgs(BaseArgument* p_arg,
                                     void* p_dev_kernel_args,
                                     const void* p_host_kernel_args) const override
    {
        return SetDeviceKernelArgs(
            *dynamic_cast<Argument*>(p_arg), p_dev_kernel_args, p_host_kernel_args);
    }

    void SetDeviceKernelArgs(Argument& arg, void* p_dev_kernel_args) const
    {
        arg.p_dev_gemm_args_ = p_dev_kernel_args;
    }

    virtual void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        return SetDeviceKernelArgs(*dynamic_cast<Argument*>(p_arg), p_dev_kernel_args);
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(KernelArguments);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
