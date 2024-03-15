// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile/block_gemm_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bgmem_creg_v1_default_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

// A is block distributed tensor
// B is block window on global memory
// C is block distributed tensor
// This will:
//   1. Load B from global memory into shared memory and then
//   2. Call BlockGemmARegSGmemCRegV1
template <typename Problem_, typename Policy_ = BlockGemmARegBGmemCRegV1DefaultPolicy>
struct BlockGemmARegBGmemCRegV1
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // use BlockGemmARegBSmemCRegV1 as the underlying block-GEMM implementation
    using BlockGemmARegBSmemCRegImpl = BlockGemmARegBSmemCRegV1<
        BlockGemmProblem<ADataType, BDataType, CDataType, kBlockSize, BlockGemmShape>,
        BlockGemmARegBSmemCRegV1DefaultPolicy>;

    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return sizeof(BDataType) *
               Policy::template MakeBSmemBlockDescriptor<Problem>().GetElementSpaceSize();
    }

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensor, typename BBlockGmemWindowTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockTensor& a_block_tensor,
                               const BBlockGmemWindowTmp& b_block_gmem_window_tmp,
                               void* smem_ptr) const
    {
        static_assert(
            is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                is_same_v<BDataType, remove_cv_t<typename BBlockGmemWindowTmp::DataType>> &&
                is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockTensor{}.GetLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockGmemWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockTensor{}.GetLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        const auto b_block_gmem_window =
            make_tile_window(b_block_gmem_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<NPerBlock>{}, Number<KPerBlock>{}),
                             b_block_gmem_window_tmp.GetWindowOrigin(),
                             Policy::template MakeBGmemTileDistribution<Problem>());

        // B LDS and LDS window
        auto b_block_smem = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<BDataType*>(smem_ptr),
            Policy::template MakeBSmemBlockDescriptor<Problem>());

        auto b_block_smem_window = make_tile_window(
            b_block_smem, make_tuple(Number<MPerBlock>{}, Number<KPerBlock>{}), {0, 0});

        // load B tile from global mem
        const auto b_block_tile = load_tile(b_block_gmem_window);

        // store B tile into shared mem
        store_tile(b_block_smem_window, b_block_tile);

        // wait for store_tile to finish
        block_sync_lds();

        // block GEMM
        BlockGemmARegBSmemCRegImpl{}(c_block_tensor, a_block_tensor, b_block_smem_window);
    }

    // C = A * B
    template <typename ABlockTensor, typename BBlockGmemWindowTmp>
    __device__ auto operator()(const ABlockTensor& a_block_tensor,
                               const BBlockGmemWindowTmp& b_block_gmem_window_tmp,
                               void* smem_ptr) const
    {
        static_assert(is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                          is_same_v<BDataType, remove_cv_t<typename BBlockGmemWindowTmp::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockTensor{}.GetLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockGmemWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockTensor{}.GetLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        const auto b_block_gmem_window =
            make_tile_window(b_block_gmem_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<NPerBlock>{}, Number<KPerBlock>{}),
                             b_block_gmem_window_tmp.GetWindowOrigin(),
                             Policy::template MakeBGmemTileDistribution<Problem>());

        // B LDS and LDS window
        auto b_block_smem = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<BDataType*>(smem_ptr),
            Policy::template MakeBSmemBlockDescriptor<Problem>());

        auto b_block_smem_window = make_tile_window(
            b_block_smem, make_tuple(Number<MPerBlock>{}, Number<KPerBlock>{}), {0, 0});

        // load B tile from global mem
        const auto b_block_tile = load_tile(b_block_gmem_window);

        // store B tile into shared mem
        store_tile(b_block_smem_window, b_block_tile);

        // wait for store_tile to finish
        block_sync_lds();

        // block GEMM
        return BlockGemmARegBSmemCRegImpl{}(a_block_tensor, b_block_smem_window);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
