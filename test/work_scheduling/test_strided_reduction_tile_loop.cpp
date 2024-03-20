// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <memory>
#include <vector>

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

namespace {

using namespace ck;

struct GemmArgDesc
{
    GemmArgDesc(index_t M_,
                index_t N_,
                index_t K_,
                const float* p_A_,
                const float* p_B_,
                float* p_C_,
                index_t tile_count_)
        : M{M_}, N{N_}, K{K_}, p_A{p_A_}, p_B{p_B_}, p_C{p_C_}, tile_count{tile_count_}
    {
    }

    index_t M;
    index_t N;
    index_t K;
    const float* p_A;
    const float* p_B;
    float* p_C;
    index_t tile_count;
};

template <index_t MPerBlock, index_t NPerBlock, index_t KPerBlock>
__global__ void grouped_gemm_naive_strided_tile_loop_reduce(const GemmArgDesc* p_gemm_descs,
                                                            float* p_workspace,
                                                            uint32_t* p_flags,
                                                            index_t tile_count,
                                                            index_t k_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))

    StridedReductionTileLoop work_scheduler{tile_count, p_flags};

    // early exit if no work.
    if(work_scheduler.tile_id_ >= tile_count)
        return;

    index_t group_id      = 0;
    index_t offset        = 0;
    index_t grid_size_grp = p_gemm_descs[group_id].tile_count;

    index_t gemm_tile_id_start = 0;
    index_t gemm_tile_id_end   = grid_size_grp;

    do
    {
        // Find corresponding GEMM group for out tile
        while(!(work_scheduler.tile_id_ >= gemm_tile_id_start &&
                work_scheduler.tile_id_ < gemm_tile_id_end))
        {
            // Step to next GEMM group and update data tile bounds.
            offset += grid_size_grp;
            group_id++;
            grid_size_grp = p_gemm_descs[group_id].tile_count;

            gemm_tile_id_start = offset;
            gemm_tile_id_end   = offset + grid_size_grp;
        }

        const index_t M = p_gemm_descs[group_id].M;
        const index_t N = p_gemm_descs[group_id].N;
        const index_t K = p_gemm_descs[group_id].K;

        auto p_A = const_cast<float*>(p_gemm_descs[group_id].p_A);
        auto p_B = const_cast<float*>(p_gemm_descs[group_id].p_B);
        auto p_C = p_gemm_descs[group_id].p_C;

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
        const auto block_work_idx =
            b2c_tile_map.CalculateBottomIndex(work_scheduler.tile_id_ - offset);

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

            auto a_buffer_resource = make_wave_buffer_resource_with_default_range<float>(
                p_A + A_m_tile_offset * stride_a + A_k_tile_offset);
            auto b_buffer_resource = make_wave_buffer_resource_with_default_range<float>(
                p_B + B_k_tile_offset * stride_b + B_n_tile_offset);

            for(index_t k = 0; k < KPerBlock; ++k)
            {
                float a_val = llvm_amdgcn_raw_buffer_load_fp32(
                    a_buffer_resource,
                    (A_thread_tile_m_idx * stride_a + k) * sizeof(float),
                    0,
                    static_cast<index_t>(AmdBufferCoherenceEnum::DefaultCoherence));
                float b_val = llvm_amdgcn_raw_buffer_load_fp32(
                    b_buffer_resource,
                    (k * stride_b + B_thread_tile_n_idx) * sizeof(float),
                    0,
                    static_cast<index_t>(AmdBufferCoherenceEnum::DefaultCoherence));

                partial_result += a_val * b_val;
                // partial_result +=
                //     p_A[(A_m_tile_offset + A_thread_tile_m_idx) * stride_a + A_k_tile_offset + k]
                //     * p_B[(B_k_tile_offset + k) * stride_b + B_n_tile_offset +
                //     B_thread_tile_n_idx];
            }
        } while(work_scheduler.GetNextTile() && b2c_tile_map.GetNextKTileIdx());

        // if next [M,N] tile
        if(!b2c_tile_map.IsFirstKSplitBlock())
        {
            // Assume we have MPerBlock x NPerBlock tile per each workgroup in contiguous memory.
            auto w_buffer_resource = make_wave_buffer_resource_with_default_range<float>(
                p_workspace + get_block_1d_id() * MPerBlock * NPerBlock);

            llvm_amdgcn_raw_buffer_store_fp32(partial_result,
                                              w_buffer_resource,
                                              get_thread_local_1d_id() * sizeof(float),
                                              0,
                                              static_cast<index_t>(AmdBufferCoherenceEnum::GLC));
            // p_workspace[get_block_1d_id() * MPerBlock * NPerBlock + get_thread_local_1d_id()] =
            //     partial_result;
        }

        work_scheduler.FlagFinished();

        // The workgroup which processed first K tile accumulates results and stores to GMEM
        if(b2c_tile_map.IsFirstKSplitBlock())
        {
            // Wait untill all other blocks for this [M,N] tile store their results.
            index_t neighbour_count =
                work_scheduler.WaitForNeighbours(k_batch, b2c_tile_map.GetTileKIdx());

            // Accumulate partial results. We can have different # of workgroups to reduce, thus we
            // read actual flag value.
            for(index_t i = 1; i <= neighbour_count; ++i)
            {
                // partial_result += p_workspace[(get_block_1d_id()) * MPerBlock * NPerBlock +
                //                               i * MPerBlock * NPerBlock +
                //                               get_thread_local_1d_id()];
                auto w_buffer_resource = make_wave_buffer_resource_with_default_range<float>(
                    p_workspace + get_block_1d_id() * MPerBlock * NPerBlock +
                    i * MPerBlock * NPerBlock);

                float value = llvm_amdgcn_raw_buffer_load_fp32(
                    w_buffer_resource,
                    get_thread_local_1d_id() * sizeof(float),
                    0,
                    static_cast<index_t>(AmdBufferCoherenceEnum::GLC));
                partial_result += value;
            }

            // Signal waiting blocks that they can start use their workspace.
            work_scheduler.Reset(neighbour_count);

            // write result
            const index_t C_m_tile_offset     = block_m_id * MPerBlock;
            const index_t C_thread_tile_m_idx = get_thread_local_1d_id() / NPerBlock;
            const index_t C_n_tile_offset     = block_n_id * NPerBlock;
            const index_t C_thread_tile_n_idx = get_thread_local_1d_id() % NPerBlock;

            auto c_buffer_resource = make_wave_buffer_resource_with_default_range<float>(
                p_C + C_m_tile_offset * stride_c + C_n_tile_offset);
            llvm_amdgcn_raw_buffer_store_fp32(
                partial_result,
                c_buffer_resource,
                (C_thread_tile_m_idx * stride_c + C_thread_tile_n_idx) * sizeof(float),
                0,
                static_cast<index_t>(AmdBufferCoherenceEnum::DefaultCoherence));

            // p_C[(C_m_tile_offset + C_thread_tile_m_idx) * stride_c + C_n_tile_offset +
            //     C_thread_tile_n_idx] = partial_result;
        }
        else if(work_scheduler.HasTile())
        {
            work_scheduler.WaitForReduction();
        }
    } while(work_scheduler.HasTile());

#else
    ignore = p_gemm_descs;
    ignore = p_workspace;
    ignore = p_flags;
    ignore = tile_count;
    ignore = k_batch;
#endif
}

} // namespace

template <index_t BlockSize, index_t MPerBlock, index_t NPerBlock, index_t KPerBlock>
struct GroupedGemmStridedTileLoopReduce
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
        grouped_gemm_naive_strided_tile_loop_reduce<MPerBlock, NPerBlock, KPerBlock>;

    using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                            BDataType,
                                                                            CDataType,
                                                                            AccDataType,
                                                                            AElementOp,
                                                                            BElementOp,
                                                                            CElementOp>;

    GroupedGemmStridedTileLoopReduce() = default;

    bool Run(std::vector<index_t> Ms,
             std::vector<index_t> Ns,
             std::vector<index_t> Ks,
             index_t k_batch,
             index_t grid_size)
    {
        EXPECT_TRUE(Ms.size() == Ns.size() && Ms.size() == Ks.size());
        std::size_t group_count = Ms.size();

        std::vector<Tensor<float>> a_m_k;
        std::vector<Tensor<float>> b_k_n;
        std::vector<Tensor<float>> c_m_n_host;
        std::vector<Tensor<float>> c_m_n_device;

        using DeviceMemPtr = std::unique_ptr<DeviceMem>;

        std::vector<DeviceMemPtr> a_m_k_device_buf;
        std::vector<DeviceMemPtr> b_k_n_device_buf;
        std::vector<DeviceMemPtr> c_m_n_device_buf;

        std::vector<GemmArgDesc> gemm_descs;
        gemm_descs.reserve(group_count);

        index_t tile_count = 0;

        for(std::size_t i = 0; i < group_count; ++i)
        {
            a_m_k.push_back(Tensor<float>(HostTensorDescriptor({Ms[i], Ks[i]}, {Ks[i], 1})));
            b_k_n.push_back(Tensor<float>(HostTensorDescriptor({Ks[i], Ns[i]}, {Ns[i], 1})));
            c_m_n_host.push_back(Tensor<float>(HostTensorDescriptor({Ms[i], Ns[i]}, {Ns[i], 1})));
            c_m_n_device.push_back(Tensor<float>(HostTensorDescriptor({Ms[i], Ns[i]}, {Ns[i], 1})));

            ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k[i]);
            ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n[i]);
            c_m_n_host[i].SetZero();
            c_m_n_device[i].SetZero();

            a_m_k_device_buf.emplace_back(
                std::make_unique<DeviceMem>(sizeof(float) * a_m_k[i].mDesc.GetElementSpaceSize()));
            b_k_n_device_buf.emplace_back(
                std::make_unique<DeviceMem>(sizeof(float) * b_k_n[i].mDesc.GetElementSpaceSize()));
            c_m_n_device_buf.emplace_back(std::make_unique<DeviceMem>(
                sizeof(float) * c_m_n_device[i].mDesc.GetElementSpaceSize()));

            a_m_k_device_buf[i]->ToDevice(a_m_k[i].mData.data());
            b_k_n_device_buf[i]->ToDevice(b_k_n[i].mData.data());
            c_m_n_device_buf[i]->SetZero();

            BlockToCTileMap_LinearKSplit<MPerBlock, NPerBlock> b2c_tile_map(Ms[i], Ns[i], k_batch);
            index_t grp_tile_count = b2c_tile_map.CalculateGridSize(Ms[i], Ns[i]);
            tile_count += grp_tile_count;

            gemm_descs.emplace_back(
                Ms[i],
                Ns[i],
                Ks[i],
                reinterpret_cast<float*>(a_m_k_device_buf[i]->GetDeviceBuffer()),
                reinterpret_cast<float*>(b_k_n_device_buf[i]->GetDeviceBuffer()),
                reinterpret_cast<float*>(c_m_n_device_buf[i]->GetDeviceBuffer()),
                grp_tile_count);
        }

        DeviceMem gemm_descs_device_buf{gemm_descs.size() * sizeof(GemmArgDesc)};
        gemm_descs_device_buf.ToDevice(gemm_descs.data());

        DeviceMem gemm_workspace, gemm_flags;
        const index_t flag_count = grid_size;

        gemm_workspace.Realloc(grid_size * MPerBlock * NPerBlock * sizeof(float));
        gemm_flags.Realloc(flag_count * sizeof(uint32_t));

        gemm_workspace.SetZero();
        gemm_flags.SetZero();

        launch_and_time_kernel(
            StreamConfig{nullptr, false},
            DeviceGemmKernel,
            dim3(grid_size),
            dim3(BlockSize),
            0,
            reinterpret_cast<const GemmArgDesc*>(gemm_descs_device_buf.GetDeviceBuffer()),
            reinterpret_cast<float*>(gemm_workspace.GetDeviceBuffer()),
            reinterpret_cast<uint32_t*>(gemm_flags.GetDeviceBuffer()),
            tile_count,
            k_batch);

        auto a_element_op = AElementOp{};
        auto b_element_op = BElementOp{};
        auto c_element_op = CElementOp{};

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        bool pass = true;

        for(std::size_t i = 0; i < group_count; ++i)
        {
            auto ref_argument = ref_gemm.MakeArgument(
                a_m_k[i], b_k_n[i], c_m_n_host[i], a_element_op, b_element_op, c_element_op);

            ref_invoker.Run(ref_argument);
            c_m_n_device_buf[i]->FromDevice(c_m_n_device[i].mData.data());
            pass = pass && ck::utils::check_err(c_m_n_device[i], c_m_n_host[i]);
        }

        return pass;
    }
};

TEST(TestStridedReductionTileLoop, GroupedGemm_SingleDataTile)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 4;
    const index_t grid_size     = 4;

    std::vector<index_t> Ms(1, MPerBlock);
    std::vector<index_t> Ns(1, NPerBlock);
    std::vector<index_t> Ks(1, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}

TEST(TestStridedReductionTileLoop, GroupedGemm_SingleOutputMultipleDataTiles)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 16;
    const index_t grid_size     = 4;

    std::vector<index_t> Ms(1, MPerBlock);
    std::vector<index_t> Ns(1, NPerBlock);
    std::vector<index_t> Ks(1, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}

TEST(TestStridedReductionTileLoop, GroupedGemm_MultipleDataTiles)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 16;
    const index_t grid_size     = 64;

    std::vector<index_t> Ms(1, MPerBlock * 4);
    std::vector<index_t> Ns(1, NPerBlock * 4);
    std::vector<index_t> Ks(1, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}

TEST(TestStridedReductionTileLoop, GroupedGemm_MultipleOutputDataTilesPerBlock_1Group)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 6;
    const index_t grid_size     = 3;

    std::vector<index_t> Ms(1, MPerBlock * 2);
    std::vector<index_t> Ns(1, NPerBlock);
    std::vector<index_t> Ks(1, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}

TEST(TestStridedReductionTileLoop, GroupedGemm_MultipleOutputDataTilesPerBlock_NGroup)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 6;
    const index_t grid_size     = 6;

    std::vector<index_t> Ms(2, MPerBlock * 2);
    std::vector<index_t> Ns(2, NPerBlock);
    std::vector<index_t> Ks(2, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}

TEST(TestStridedReductionTileLoop, GroupedGemm_CrossGroups_CrossK_TilePerBlockLTKBatch)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 5;
    const index_t grid_size     = 7;
    // tilse_per_block = 3

    std::vector<index_t> Ms(2, MPerBlock * 2);
    std::vector<index_t> Ns(2, NPerBlock);
    std::vector<index_t> Ks(2, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}

TEST(TestStridedReductionTileLoop, GroupedGemm_CrossGroups_CrossK_TilePerBlockGTKBatch)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 5;
    const index_t grid_size     = 5;
    // tiles_per_block = 8

    std::vector<index_t> Ms(2, MPerBlock * 2);
    std::vector<index_t> Ns(2, NPerBlock * 2);
    std::vector<index_t> Ks(2, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}

TEST(TestStridedReductionTileLoop, GroupedGemm_CrossGroups_CrossK_TilePerBlockGTKBatch2)
{
    constexpr index_t MPerBlock = 8;
    constexpr index_t NPerBlock = 32;
    constexpr index_t KPerBlock = 32;
    constexpr index_t BlockSize = 256;
    const index_t kbatch        = 5;
    // The covered number of tiles is more than actual data tiles.
    const index_t grid_size = 6;
    // tilse_per_block = 7

    std::vector<index_t> Ms(2, MPerBlock * 2);
    std::vector<index_t> Ns(2, NPerBlock * 2);
    std::vector<index_t> Ks(2, KPerBlock * kbatch);

    EXPECT_TRUE((GroupedGemmStridedTileLoopReduce<BlockSize, MPerBlock, NPerBlock, KPerBlock>{}.Run(
        Ms, Ns, Ks, kbatch, grid_size)));
}
