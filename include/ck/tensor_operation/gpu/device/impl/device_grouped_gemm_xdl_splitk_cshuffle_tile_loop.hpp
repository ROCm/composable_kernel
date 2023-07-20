// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <tuple>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/stream_utility.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

//
// @brief      Entry point kernel for device-wide Grouped GEMM operation.
//
// @param[in]  gemm_desc_const             The pointer to the array of GEMM descriptor structures.
// @param[in]  tile_count                  The overall number of output tiles we divided all groups
//                                         into.
// @param[in]  k_batch                     The number of batches we split the K dimension into.
//
// @tparam     GridwiseGemm                The specific GridwiseGEMM algorithm implementation.
// @tparam     GemmDesc                    The structure holding all necessary descriptors and other
//                                         data needed for groupd gemm calculation and work
//                                         distribution.
// @tparam     HasMainKBlockLoop           Flag indicating whether all GEMM problem configurations
//                                         need to loop over tiles in K dimension.
// @tparam     CGlobalMemoryDataOperation  The functor used to store data in output C matrix.
//                                         In example could be: AtomicAdd or Store.
//
template <typename GridwiseGemm,
          typename GemmDesc,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_xdl_splitk(const void* gemm_desc,
                                       const index_t tile_count,
                                       const index_t k_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))

    constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();
    __shared__ uint8_t p_shared[shared_size];

    index_t tile_id          = get_block_1d_id();
    const index_t grid_size  = get_grid_size();
    const auto gemm_desc_ptr = reinterpret_cast<const GemmDesc*>(gemm_desc);

    static constexpr index_t MPerBlock = GridwiseGemm::GetMPerBlock();
    static constexpr index_t NPerBlock = GridwiseGemm::GetNPerBlock();
    static constexpr index_t B2E_M01   = 8;

    using CGridDesc_M_N = typename GridwiseGemm::CGridDesc_M_N;
    using Block2ETileMapKSplit =
        BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>;

    index_t group_id = 0;
    index_t offset   = 0;

    auto M                = gemm_desc_ptr[group_id].M;
    auto N                = gemm_desc_ptr[group_id].N;
    auto StrideC          = gemm_desc_ptr[group_id].StrideC;
    auto c_grid_desc_m_n  = GridwiseGemm::MakeCGridDescriptor_M_N(M, N, StrideC);
    auto b2c_tile_map     = Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, k_batch};
    index_t grid_size_grp = b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

    index_t gemm_tile_id_start = 0;
    index_t gemm_tile_id_end   = grid_size_grp;

    while(tile_id < tile_count)
    {
        // Find corresponding GEMM group for out tile
        while(!(tile_id >= gemm_tile_id_start && tile_id < gemm_tile_id_end))
        {
            offset += grid_size_grp;
            group_id++;

            M               = gemm_desc_ptr[group_id].M;
            N               = gemm_desc_ptr[group_id].N;
            StrideC         = gemm_desc_ptr[group_id].StrideC;
            c_grid_desc_m_n = GridwiseGemm::MakeCGridDescriptor_M_N(M, N, StrideC);
            b2c_tile_map    = Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, k_batch};
            grid_size_grp   = b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

            gemm_tile_id_start = offset;
            gemm_tile_id_end   = offset + grid_size_grp;
        }

        const auto p_a_grid = reinterpret_cast<const FloatA*>(gemm_desc_ptr[group_id].p_a_grid);
        const auto p_b_grid = reinterpret_cast<const FloatB*>(gemm_desc_ptr[group_id].p_b_grid);
        const auto p_c_grid = reinterpret_cast<FloatC*>(gemm_desc_ptr[group_id].p_c_grid);

        const auto K       = gemm_desc_ptr[group_id].K;
        const auto StrideA = gemm_desc_ptr[group_id].StrideA;
        const auto StrideB = gemm_desc_ptr[group_id].StrideB;

        const auto MPadded = GridwiseGemm::CalculateMPadded(M);
        const auto NPadded = GridwiseGemm::CalculateNPadded(N);
        const auto KPadded = GridwiseGemm::CalculateKPadded(K, k_batch);
        const auto K0      = GridwiseGemm::CalculateK0(K, k_batch);

        LocalBlockToCTileMap<Block2ETileMapKSplit> local_b2c{b2c_tile_map, tile_id - offset};

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation>(
            p_a_grid,
            p_b_grid,
            p_c_grid,
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            MPadded,
            NPadded,
            KPadded,
            K0,
            k_batch,
            static_cast<void*>(p_shared),
            local_b2c);

        tile_id += grid_size;
    }

#else
    ignore = gemm_desc;
    ignore = tile_count;
    ignore = k_batch;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
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
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
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
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler(),
          // Current implementation does not support multiple D fusions.
          enable_if_t<AK1 == BK1 && is_same_v<DsLayout, ck::Tuple<>> &&
                          is_same_v<DsDataType, ck::Tuple<>>,
                      bool> = false>
struct DeviceGroupedGemmXdlSplitKCShuffle : public DeviceGroupedGemmSplitK<ALayout,
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
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static_assert(KPerBlock % AK1 == 0);
    static constexpr index_t K0PerBlock = KPerBlock / AK1;

    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        EDataType,
        ALayout,
        BLayout,
        ELayout,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        NumGemmKPrefetchStage,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        AK1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferScalarPerVector_NPerBlock,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        LoopSched,
        PipelineVersion::v2>;

    using CGridDesc_M_N   = typename GridwiseGemm::CGridDesc_M_N;
    using GridwiseGemmArg = typename GridwiseGemm::Argument;
    using KernelArguments = GroupedGemmKernelArguments;
    using Block2ETileMapKSplit =
        BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>;
    // Block2CTileMap configuration parameter.
    static constexpr index_t B2E_M01       = 8;
    static constexpr index_t DefaultKBatch = 1;

    // Argument
    struct Argument : public BaseArgument
    {

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs)
            : Argument(p_As, p_Bs, p_Es, gemm_descs, DefaultKBatch)
        {
        }

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 index_t kbatch)
            : K_BATCH{kbatch}, group_count_{0}, skipped_group_count_{0}, grid_size_{0}
        {
            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/c.size");
            }

            gemm_kernel_args_.reserve(group_count_);

            for(std::size_t i = 0; i < gemm_descs.size(); ++i)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                if(M == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t stride_a = gemm_descs[i].stride_A_;
                const index_t stride_b = gemm_descs[i].stride_B_;
                const index_t stride_c = gemm_descs[i].stride_C_;

                const auto c_grid_desc_m_n = GridwiseGemm::MakeCGridDescriptor_M_N(M, N, stride_c);

                auto local_b2c_tile_map = Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, K_BATCH};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);
                grid_size_ += grid_size_grp;

                gemm_kernel_args_.emplace_back(type_convert<const ADataType*>(p_As[i]),
                                               type_convert<const BDataType*>(p_Bs[i]),
                                               type_convert<EDataType*>(p_Es[i]),
                                               M,
                                               N,
                                               K,
                                               stride_a,
                                               stride_b,
                                               stride_c);
            }
        }

        /**
         * @brief      Set new kbatch value.
         *
         * @param[in]  kbatch  The new splitK parameter value.
         */
        void UpdateKBatch(index_t kbatch)
        {
            K_BATCH    = kbatch;
            grid_size_ = 0;

            for(std::size_t i = 0; i < gemm_kernel_args_.size(); ++i)
            {

                auto& gemm_arg = gemm_kernel_args_[i];

                const auto c_grid_desc_m_n =
                    GridwiseGemm::MakeCGridDescriptor_M_N(gemm_arg.M, gemm_arg.N, gemm_arg.StrideC);

                auto local_b2c_tile_map = Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, K_BATCH};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);
                grid_size_ += grid_size_grp;
            }
        }

        //  private:
        index_t K_BATCH;
        index_t group_count_;
        index_t skipped_group_count_;
        // The overall number of output tiles to be processed.
        index_t grid_size_;
        const void* p_dev_gemm_args_;

        std::vector<KernelArguments> gemm_kernel_args_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        // The oversubscription factor for the number of blocks that can simultaneously reside on
        // GPU.
        static constexpr int BLOCK_SUBSCRIPTION_FACTOR = 1;
        static constexpr int BLOCK_WAVES               = BlockSize / get_warp_size();
        static constexpr int CU_SIMDS                  = 4;
        // Assume we want to have at most 2 waves per SIMD
        static constexpr int CU_BLOCKS = math::integer_divide_floor(2 * CU_SIMDS, BLOCK_WAVES);

        //
        // @brief      Launch Grouped Gemm kernel.
        //
        // @note       This function overload is using user provided device buffer for kernel
        //             arguments.
        //
        // @param[in]  arg            The structure containing kernel arguments (in host memory).
        // @param[in]  dev_gemm_args  The point to device memory with kernel arguments.
        // @param[in]  stream_config  The device stream configuration.
        //
        // @return     The average kernel execution time (if time measurement is enabled.)
        //
        float Run(const Argument& arg,
                  const void* dev_gemm_args,
                  const StreamConfig& stream_config = StreamConfig{})
        {
            auto [all_have_kbatch_gt_one, all_have_main_k0_block_loop] =
                CheckArgument(arg, stream_config);

            if(dev_gemm_args == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments workspace buffer is not allocated!"
                    << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            if(all_have_kbatch_gt_one)
            {
                for(const auto& gemm_arg : arg.gemm_kernel_args_)
                {
                    hip_check_error(hipMemset(
                        gemm_arg.p_c_grid, 0, gemm_arg.M * gemm_arg.N * sizeof(EDataType)));
                }
            }

            float ave_time = 0;

            if(all_have_main_k0_block_loop)
            {
                if(all_have_kbatch_gt_one)
                {
                    ave_time = DispatchKernel<InMemoryDataOperationEnum::AtomicAdd, true>(
                        arg, dev_gemm_args, stream_config);
                }
                else
                {
                    ave_time = DispatchKernel<InMemoryDataOperationEnum::Set, true>(
                        arg, dev_gemm_args, stream_config);
                }
            }
            else
            {
                if(all_have_kbatch_gt_one)
                {
                    ave_time = DispatchKernel<InMemoryDataOperationEnum::AtomicAdd, false>(
                        arg, dev_gemm_args, stream_config);
                }
                else
                {
                    ave_time = DispatchKernel<InMemoryDataOperationEnum::Set, false>(
                        arg, dev_gemm_args, stream_config);
                }
            }

            return ave_time;
        }

        //
        // @brief      Launch Grouped Gemm kernel.
        //
        // @note       This function overload is using device workspace buffer for kernel arguments.
        //             The user should call @see GetWorkSpaceSize and @see SetWorkSpacePointer on
        //             arg parameter to properly allocate this buffer.
        //
        // @param[in]  arg            The structure containing kernel arguments (in host memory).
        // @param[in]  stream_config  The device stream configuration.
        //
        // @return     The average kernel execution time (if time measurement is enabled.)
        //
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(arg.p_workspace_ != nullptr)
            {
                hip_check_error(
                    hipMemcpyWithStream(arg.p_workspace_,
                                        arg.gemm_kernel_args_.data(),
                                        arg.gemm_kernel_args_.size() * sizeof(KernelArguments),
                                        hipMemcpyHostToDevice,
                                        stream_config.stream_id_));
            }
            else
            {
                std::ostringstream err;
                err << "The gemm arguments workspace buffer is not allocated!"
                    << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            return Run(arg, arg.p_workspace_, stream_config);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }

        private:
        auto CheckArgument(const Argument& arg, const StreamConfig& stream_config) const
        {
            index_t K0 = GridwiseGemm::CalculateK0(arg.gemm_kernel_args_[0].K, arg.K_BATCH);
            bool all_have_kbatch_gt_one      = arg.K_BATCH > 1;
            bool all_have_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
            {
                const auto& gemm_arg = arg.gemm_kernel_args_[i];
                if(stream_config.log_level_ > 0)
                {
                    gemm_arg.Print();
                }

                // Currently all groups use same kbatch value.
                auto kbatch = arg.K_BATCH;
                K0          = GridwiseGemm::CalculateK0(arg.gemm_kernel_args_[i].K, arg.K_BATCH);

                if(!GridwiseGemm::CheckValidity(GridwiseGemmArg{nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                gemm_arg.M,
                                                                gemm_arg.N,
                                                                gemm_arg.K,
                                                                gemm_arg.StrideA,
                                                                gemm_arg.StrideB,
                                                                gemm_arg.StrideC,
                                                                0, // MPadded
                                                                0, // NPadded
                                                                0, // KPadded
                                                                K0,
                                                                kbatch}))
                {
                    std::ostringstream err;
                    err << "Group id: " << i << " has invalid GridwiseGemm settings!" << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                bool not_all_have_main_k0_block_loop_same =
                    all_have_main_k0_block_loop xor GridwiseGemm::CalculateHasMainK0BlockLoop(K0);
                bool not_all_have_kbatch_value_same = all_have_kbatch_gt_one xor (kbatch > 1);

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
                    err << "Not all gemms have same kbatch value (=1 or >1)! "
                        << "group [" << i << "], kbatch: " << kbatch
                        << ", group [0], kbatch: " << arg.K_BATCH << " in " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
            }
            return std::make_tuple(all_have_kbatch_gt_one, all_have_main_k0_block_loop);
        }

        template <InMemoryDataOperationEnum CGlobalMemoryDataOperation, bool HasMainKBlockLoop>
        float DispatchKernel(const Argument& arg,
                             const void* dev_gemm_args,
                             const StreamConfig& stream_config) const
        {
            const auto kernel = kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               KernelArguments,
                                                               ADataType,
                                                               BDataType,
                                                               EDataType,
                                                               HasMainKBlockLoop,
                                                               CGlobalMemoryDataOperation>;
            return LaunchKernel(kernel, arg, stream_config);
        }

        template <typename KernelFunction>
        float LaunchKernel(const KernelFunction& kernel,
                           const Argument& arg,
                           const StreamConfig& stream_config) const
        {
            // Calculate max number of workgroups that can simultaneously reside on the CU.
            int num_blocks                = 0;
            size_t dyn_shared_mem_per_blk = 0;
            hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks, kernel, BlockSize, dyn_shared_mem_per_blk));

            int cu_count = getAvailableComputeUnitCount(stream_config);

            if(stream_config.log_level_ > 0)
            {
                std::cout << "MaxActiveBlocksPerCU: " << num_blocks
                          << ", available CUs count: " << cu_count << ", grid size: "
                          << ck::math::min(num_blocks, CU_BLOCKS) * cu_count *
                                 BLOCK_SUBSCRIPTION_FACTOR
                          << std::endl;
            }

            return launch_and_time_kernel(
                stream_config,
                kernel,
                dim3(cu_count * ck::math::min(num_blocks, CU_BLOCKS) * BLOCK_SUBSCRIPTION_FACTOR),
                dim3(BlockSize),
                0,
                arg.p_workspace_,
                arg.grid_size_,
                arg.K_BATCH);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if((ck::type_convert<ck::index_t>(arg.gemm_kernel_args_.size()) +
            arg.skipped_group_count_) != arg.group_count_)
        {
#if DEBUG_LOG
            std::cout << "The group count is not equal to sum of skipped groups "
                         "and kernel args size!"
                      << std::endl;
#endif // DEBUG_LOG
            return false;
        }

        bool supported = true;
        for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
        {
            const auto& gemm_arg = arg.gemm_kernel_args_[i];
            const auto K0        = GridwiseGemm::CalculateK0(gemm_arg.K, arg.K_BATCH);
            bool group_arg_valid = GridwiseGemm::CheckValidity(GridwiseGemmArg{nullptr,
                                                                               nullptr,
                                                                               nullptr,
                                                                               gemm_arg.M,
                                                                               gemm_arg.N,
                                                                               gemm_arg.K,
                                                                               gemm_arg.StrideA,
                                                                               gemm_arg.StrideB,
                                                                               gemm_arg.StrideC,
                                                                               0, // MPadded
                                                                               0, // NPadded
                                                                               0, // KPadded
                                                                               K0,
                                                                               arg.K_BATCH});
            if(not group_arg_valid)
            {
#if DEBUG_LOG
                std::cout << "[" << __func__ << "] group id: " << i
                          << " has invalid GridwiseGemm settings!" << std::endl;
                gemm_arg.Print();
#endif // DEBUG_LOG
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
                             std::vector<std::array<const void*, NumDTensor>>&,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CDEElementwiseOperation)
    {
        return Argument{p_As, p_Bs, p_Es, gemm_descs};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>&,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation) override
    {
        return std::make_unique<Argument>(p_As, p_Bs, p_Es, gemm_descs);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedGemm_XdlSplitKTileLoop"
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

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->gemm_kernel_args_.size() *
               sizeof(KernelArguments);
    }

    static void SetKBatchSize(Argument& arg, index_t kbatch) { arg.UpdateKBatch(kbatch); }
    static void SetDeviceKernelArgs(Argument& arg, const void* p_dev_kernel_args)
    {
        arg.p_dev_gemm_args_ = p_dev_kernel_args;
    }

    void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const override
    {
        return SetKBatchSize(*dynamic_cast<Argument*>(p_arg), kbatch);
    }

    void SetDeviceKernelArgs(BaseArgument* p_arg, const void* p_dev_kernel_args) const override
    {
        return SetDeviceKernelArgs(*dynamic_cast<Argument*>(p_arg), p_dev_kernel_args);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
