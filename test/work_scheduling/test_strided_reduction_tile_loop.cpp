// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include <gtest/gtest.h>

#include <ck/ck.hpp>
#include <ck/host_utility/kernel_launch.hpp>
#include <ck/utility/common_header.hpp>
#include <ck/utility/work_scheduling.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

using namespace ck;

namespace {

template <index_t MPerBlock, index_t NPerBlock, index_t KPerBlock>
__global__ void gemm_naive_strided_tile_loop_reduce(index_t M,
                                                    index_t N,
                                                    index_t K,
                                                    const float* p_A,
                                                    const float* p_B,
                                                    float* p_C,
                                                    float* p_workspace,
                                                    uint32_t* p_flags,
                                                    index_t tile_count,
                                                    index_t k_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))

    StridedReductionTileLoop work_scheduler{tile_count, p_flags};
    const auto c_grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, N));
    BlockToCTileMap_LinearKSplit<MPerBlock, NPerBlock> b2c_tile_map(c_grid_desc_m_n, k_batch);

    float partial_result = 0.f;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    // Assume MK-KN-MN data layout
    const index_t stride_a = K;
    const index_t stride_b = N;
    const index_t stride_c = N;

    // K is the contiguous dim in memory, as well as fastest changing dim in B2C mapping.
    const auto block_work_idx = b2c_tile_map.CalculateBottomIndex(work_scheduler.tile_id_);

    const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
    const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

    do
    {
        const index_t k_batch_id = __builtin_amdgcn_readfirstlane(b2c_tile_map.GetTileKIdx());

        const index_t A_m_tile_offset     = block_m_id * MPerBlock;
        const index_t A_k_tile_offset     = k_batch_id * KPerBlock;
        const index_t A_thread_tile_m_idx = get_thread_local_1d_id() / NPerBlock;

        const index_t B_n_tile_offset     = block_n_id * NPerBlock;
        const index_t B_k_tile_offset     = k_batch_id * KPerBlock;
        const index_t B_thread_tile_n_idx = get_thread_local_1d_id() % NPerBlock;

        for(index_t k = 0; k < KPerBlock; ++k)
        {
            partial_result +=
                p_A[(A_m_tile_offset + A_thread_tile_m_idx) * stride_a + A_k_tile_offset + k] *
                p_B[(B_k_tile_offset + k) * stride_b + B_n_tile_offset + B_thread_tile_n_idx];
        }
    } while(work_scheduler.GetNextTile() && b2c_tile_map.GetNextKTileIdx());

    // if next [M,N] tile
    if(!b2c_tile_map.IsFirstKSplitBlock(work_scheduler.tiles_per_block_))
    {
        // Assume we have MPerBlock x NPerBlock tile per each workgroup in contiguous memory.
        p_workspace[get_block_1d_id() * MPerBlock * NPerBlock + get_thread_local_1d_id()] =
            partial_result;
    }

    work_scheduler.FlagFinished(k_batch, b2c_tile_map.GetOutputTileIdx());

    // The workgroup which processed first K tile accumulates results and stores to GMEM
    if(b2c_tile_map.IsFirstKSplitBlock(work_scheduler.tiles_per_block_))
    {
        // Wait untill all other blocks for this [M,N] tile store their results.
        work_scheduler.WaitForNeighbours(k_batch, b2c_tile_map.GetOutputTileIdx());

        // accumulate partial results
        const index_t workgroups_per_dim =
            (k_batch + work_scheduler.tiles_per_block_ - 1) / work_scheduler.tiles_per_block_;
        for(index_t i = 0; i < workgroups_per_dim; ++i)
        {
            partial_result += p_workspace[(get_block_1d_id()) * MPerBlock * NPerBlock +
                                          i * MPerBlock * NPerBlock + get_thread_local_1d_id()];
        }

        // write result
        const index_t C_m_tile_offset     = block_m_id * MPerBlock;
        const index_t C_thread_tile_m_idx = get_thread_local_1d_id() / NPerBlock;
        const index_t C_n_tile_offset     = block_n_id * NPerBlock;
        const index_t C_thread_tile_n_idx = get_thread_local_1d_id() % NPerBlock;

        p_C[(C_m_tile_offset + C_thread_tile_m_idx) * stride_c + C_n_tile_offset +
            C_thread_tile_n_idx] = partial_result;
    }

#else
    ignore = p_input;
    ignore = p_output;
    ignore = p_workspace;
    ignore = p_flags;
    ignore = tile_count;
    ignore = k_batch;
#endif
}

} // namespace

template <index_t BlockSize, index_t MPerBlock, index_t NPerBlock, index_t KPerBlock>
struct GemmStridedTileLoopReduce
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp  = PassThrough;
    using BElementOp  = PassThrough;
    using CElementOp  = PassThrough;

    using ADataType   = float;
    using BDataType   = float;
    using CDataType   = float;
    using AccDataType = float;

    constexpr static auto DeviceGemmKernel =
        gemm_naive_strided_tile_loop_reduce<MPerBlock, NPerBlock, KPerBlock>;

    using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                            BDataType,
                                                                            CDataType,
                                                                            AccDataType,
                                                                            AElementOp,
                                                                            BElementOp,
                                                                            CElementOp>;

    GemmStridedTileLoopReduce() = default;

    bool Run(index_t M, index_t N, index_t K, index_t k_batch)
    {
        Tensor<float> a_m_k(HostTensorDescriptor({M, K}, {K, 1}));
        Tensor<float> b_k_n(HostTensorDescriptor({K, N}, {N, 1}));

        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);

        Tensor<float> c_m_n_host(HostTensorDescriptor({M, N}, {N, 1}));
        Tensor<float> c_m_n_device(HostTensorDescriptor({M, N}, {N, 1}));

        DeviceMem a_m_k_device_buf(sizeof(float) * a_m_k.mDesc.GetElementSpaceSize());
        DeviceMem b_k_n_device_buf(sizeof(float) * b_k_n.mDesc.GetElementSpaceSize());
        DeviceMem c_m_n_device_buf(sizeof(float) * c_m_n_device.mDesc.GetElementSpaceSize());

        a_m_k_device_buf.ToDevice(a_m_k.mData.data());
        b_k_n_device_buf.ToDevice(b_k_n.mData.data());
        c_m_n_device_buf.SetZero();
        c_m_n_host.SetZero();

        DeviceMem gemm_workspace, gemm_flags;
        BlockToCTileMap_LinearKSplit<MPerBlock, NPerBlock> b2c_tile_map(M, N, k_batch);
        const index_t tile_count      = b2c_tile_map.CalculateGridSize(M, N);
        const index_t grid_size       = tile_count / 4;
        const index_t tiles_per_block = (tile_count + grid_size - 1) / grid_size;
        // This is the number of MN-output tiles which we cover with workgroups.
        // We launch k_batch / tiles_per_block workgroups for each output tile.
        const index_t flag_count = (grid_size * tiles_per_block + k_batch - 1) / k_batch;

        gemm_workspace.Realloc(grid_size * MPerBlock * NPerBlock * sizeof(float));
        gemm_flags.Realloc(flag_count * sizeof(uint32_t));

        gemm_workspace.SetZero();
        gemm_flags.SetZero();

        launch_and_time_kernel(StreamConfig{nullptr, false},
                               DeviceGemmKernel,
                               dim3(grid_size),
                               dim3(BlockSize),
                               0,
                               M,
                               N,
                               K,
                               reinterpret_cast<const float*>(a_m_k_device_buf.GetDeviceBuffer()),
                               reinterpret_cast<const float*>(b_k_n_device_buf.GetDeviceBuffer()),
                               reinterpret_cast<float*>(c_m_n_device_buf.GetDeviceBuffer()),
                               reinterpret_cast<float*>(gemm_workspace.GetDeviceBuffer()),
                               reinterpret_cast<uint32_t*>(gemm_flags.GetDeviceBuffer()),
                               tile_count,
                               k_batch);

        auto a_element_op = AElementOp{};
        auto b_element_op = BElementOp{};
        auto c_element_op = CElementOp{};

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);
        c_m_n_device_buf.FromDevice(c_m_n_device.mData.data());

        return ck::utils::check_err(c_m_n_device, c_m_n_host);
    }
};

TEST(TestStridedReductionTileLoop, SingleDataTile)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 4;

    EXPECT_TRUE((GemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        MPerBlock, NPerBlock, KPerBlock * kbatch, kbatch)));
}

TEST(TestStridedReductionTileLoop, SingleOutputMultipleDataTiles)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 16;

    EXPECT_TRUE((GemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        MPerBlock, NPerBlock, KPerBlock * kbatch, kbatch)));
}

TEST(TestStridedReductionTileLoop, MultipleDataTiles)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 16;

    EXPECT_TRUE((GemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        MPerBlock * 4, NPerBlock * 4, KPerBlock * kbatch, kbatch)));
}
