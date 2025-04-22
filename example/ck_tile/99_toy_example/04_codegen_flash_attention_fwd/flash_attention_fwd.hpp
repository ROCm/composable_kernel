// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "../../../example/ck_tile/99_toy_example/02_gemm/block_gemm_pipeline_agmem_bgmem_creg.hpp"
#include "block_gemm_pipeline_problem.hpp"
#include "block_gemm_areg_bsmem_creg_v1.hpp"
#include "flash_attention_fwd_impl.hpp"

namespace ck_tile {


template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType>
struct FlashAttnArgs
{
    // Pointers to device buffers for Q, K, V, O
    QDataType* q_ptr;
    KDataType* k_ptr;
    VDataType* v_ptr;
    ODataType* o_ptr;

    // Problem sizes  
    index_t M0;
    index_t N0;
    index_t K0;
    index_t N1;
    index_t Batch;

    // Strides within a batch  
    index_t strideQ;  
    index_t strideK;  
    index_t strideV;  
    index_t strideO;  

    // Batch strides  
    index_t batchStrideQ;  
    index_t batchStrideK;  
    index_t batchStrideV;  
    index_t batchStrideO;  
};




// S[M0, N0] = Q[M0, K0] * K[N0, K0]
// P[M0, N0] = Softmax(S[M0, N0])
// O[M0, N1] = P[M0, N0] * V[N1, N0]
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType,
          index_t kBlockSize,
          index_t kHeadDim,
          index_t kM0PerBlock,
          index_t kN0PerBlock,
          index_t kK0PerBlock,
          index_t kN1PerBlock,
          index_t kK1PerBlock>
struct FlashAttentionFwd
{
    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               const index_t M0,
                               const index_t N0,
                               const index_t K0,
                               const index_t N1,
                               const index_t /* Batch */,
                               const index_t StrideQ,
                               const index_t StrideK,
                               const index_t StrideV,
                               const index_t StrideO,
                               const index_t BatchStrideQ,
                               const index_t BatchStrideK,
                               const index_t BatchStrideV,
                               const index_t BatchStrideO) const
    {
        const index_t id_block = get_block_id();

        const index_t num_tile_m0 = integer_divide_ceil(M0, kM0PerBlock);
        const index_t num_tile_n1 = integer_divide_ceil(N1, kN1PerBlock);

#if defined(GEMM_OPT)
        const auto block2tile = MakeBlock2TileMap(num_tile_m0, num_tile_n1);

        const index_t id_tile_batch = id_block / num_tile_n1 / num_tile_m0;
        const auto id_tile = block2tile(id_block - id_tile_batch * num_tile_n1 * num_tile_m0);

        const index_t iBatch = __builtin_amdgcn_readfirstlane(id_tile_batch);
        const index_t iM0    = __builtin_amdgcn_readfirstlane(id_tile.template get(number<0>{}) % num_tile_m0 * kM0PerBlock);
        const index_t iN1    = __builtin_amdgcn_readfirstlane(id_tile.template get(number<1>{}) % num_tile_n1 * kN1PerBlock);

#else
        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;

            return make_tuple(quotient, modulus);
        };
        const auto [itmp, id_tile_n]          = f(id_block, num_tile_n1);
        const auto [id_tile_batch, id_tile_m] = f(itmp, num_tile_m0);
        const index_t iBatch = __builtin_amdgcn_readfirstlane(id_tile_batch);
        const index_t iM0    = __builtin_amdgcn_readfirstlane(id_tile_m * kM0PerBlock);
        const index_t iN1    = __builtin_amdgcn_readfirstlane(id_tile_n * kN1PerBlock);

#endif

        const auto kernel_impl = FlashAttentionFwdImpl<QDataType,
                                                       KDataType,
                                                       VDataType,
                                                       SaccDataType,
                                                       SMPLComputeDataType,
                                                       PDataType,
                                                       OaccDataType,
                                                       ODataType,
                                                       kBlockSize,
                                                       kHeadDim,
                                                       kM0PerBlock,
                                                       kN0PerBlock,
                                                       kK0PerBlock,
                                                       kN1PerBlock,
                                                       kK1PerBlock>{};

        kernel_impl(q_ptr + iBatch * BatchStrideQ,
                    k_ptr + iBatch * BatchStrideK,
                    v_ptr + iBatch * BatchStrideV,
                    o_ptr + iBatch * BatchStrideO,
                    M0,
                    N0,
                    K0,
                    N1,
                    StrideQ,
                    StrideK,
                    StrideV,
                    StrideO,
                    iM0,
                    iN1);
    }
};

// // TODO: fwd_api.cpp
// template <typename SaccDataType_,
//           typename SMPLComputeDataType_,
//           typename PDataType_,
//           typename OaccDataType_,
//           index_t kBlockSize_,
//           index_t kHeadDim_,
//           index_t kM0PerBlock_,
//           index_t kN0PerBlock_,
//           index_t kK0PerBlock_,
//           index_t kN1PerBlock_,
//           index_t kK1PerBlock_>
// struct flash_attention_fwd_traits_
// {
//     using SaccDataType = ck_tile::remove_cvref_t<SaccDataType_>;
//     using SMPLComputeDataType = ck_tile::remove_cvref_t<SMPLComputeDataType_>;
//     using PDataType = ck_tile::remove_cvref_t<PDataType_>;
//     using OaccDataType = ck_tile::remove_cvref_t<OaccDataType_>;

//     static constexpr index_t kBlockSize  = kBlockSize_;
//     static constexpr index_t kHeadDim    = kHeadDim_;
//     static constexpr index_t kM0PerBlock = kM0PerBlock_;
//     static constexpr index_t kN0PerBlock = kN0PerBlock_;
//     static constexpr index_t kK0PerBlock = kK0PerBlock_;
//     static constexpr index_t kN1PerBlock = kN1PerBlock_;
//     static constexpr index_t kK1PerBlock = kK1PerBlock_;
    
//     static constexpr ck_tile::index_t kWarpPerCu    = 8; // 2 warps per SIMD
//     static constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / warpSize;
//     static constexpr ck_tile::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;
// };

// // TODO: fwd_api.cpp, fwd_common.cpp
// template <typename SaccDataType,
//           typename SMPLComputeDataType,
//           typename PDataType,
//           typename OaccDataType,
//           index_t kBlockSize,
//           index_t kHeadDim,
//           index_t kM0PerBlock,
//           index_t kN0PerBlock,
//           index_t kK0PerBlock,
//           index_t kN1PerBlock,
//           index_t kK1PerBlock>
// using traits_ = flash_attention_fwd_traits_<SaccDataType,
//                                             SMPLComputeDataType,
//                                             PDataType,
//                                             OaccDataType,
//                                             kBlockSize,
//                                             kHeadDim,
//                                             kM0PerBlock,
//                                             kN0PerBlock,
//                                             kK0PerBlock,
//                                             kN1PerBlock,
//                                             kK1PerBlock>;
// // fw_api.cpp
// // Note: this internal API only declare, not define here, otherwise will block `make -j`
// template <typename QDataType,
//           typename KDataType,
//           typename VDataType,
//           typename ODataType,
//           typename Traits_>
// float flash_attention_fwd_(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a, 
//                           const ck_tile::stream_config& stream_config);

// // TODO: fwd_common.cpp
// template <typename QDataType,
//           typename KDataType,
//           typename VDataType,
//           typename ODataType,
//           typename Traits_>
// float flash_attention_fwd_(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a, 
//                           const ck_tile::stream_config& stream_config) {
//     using SaccDataType        = typename Traits_::SaccDataType;                                                                           
//     using SMPLComputeDataType = typename Traits_::SMPLComputeDataType;                                                                           
//     using PDataType           = typename Traits_::PDataType;                                                                           
//     using OaccDataType        = typename Traits_::OaccDataType;                                                                           
    
//     index_t kGridSize = a.Batch * (a.M0 / Traits_::kM0PerBlock) * (a.N1 / Traits_::kN1PerBlock);

//     std::cout << "grid size " << kGridSize << std::endl;

//     return ck_tile::launch_kernel(stream_config,
//         ck_tile::make_kernel<Traits_::kBlockSize, Traits_::kBlockPerCu>(
//         ck_tile::FlashAttentionFwd<QDataType,
//                                    KDataType,
//                                    VDataType,
//                                    SaccDataType,
//                                    SMPLComputeDataType,
//                                    PDataType,
//                                    OaccDataType,
//                                    ODataType,
//                                    Traits_::kBlockSize,
//                                    Traits_::kHeadDim,
//                                    Traits_::kM0PerBlock,
//                                    Traits_::kN0PerBlock,
//                                    Traits_::kK0PerBlock,
//                                    Traits_::kN1PerBlock,
//                                    Traits_::kK1PerBlock>{},
//         kGridSize,
//         Traits_::kBlockSize,
//         0,
//         a.q_ptr,
//         a.k_ptr,
//         a.v_ptr,
//         a.o_ptr,
//         a.M0,
//         a.N0,
//         a.K0,
//         a.N1,
//         a.Batch,
//         a.strideQ,        // StrideQ
//         a.strideK,        // StrideK
//         a.strideV,        // StrideV
//         a.strideO,        // StrideO
//         a.batchStrideQ,   // BatchStrideQ
//         a.batchStrideK,   // BatchStrideK
//         a.batchStrideV,   // BatchStrideV
//         a.batchStrideO)); // BatchStrideO
// }

// // TODO: change to only declare
// // TODO: fwd_api.cpp
// template <typename QDataType,
//           typename KDataType,
//           typename VDataType,
//           typename SaccDataType,
//           typename SMPLComputeDataType,
//           typename PDataType,
//           typename OaccDataType,
//           typename ODataType>
// float flash_attention_fwd(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a, 
//                           const ck_tile::stream_config& stream_config) {
//     constexpr ck_tile::index_t kM0PerBlock = 128;
//     constexpr ck_tile::index_t kN0PerBlock = 128;
//     constexpr ck_tile::index_t kK0PerBlock = 32;
//     constexpr ck_tile::index_t kN1PerBlock = 128;
//     constexpr ck_tile::index_t kK1PerBlock = 32;

//     constexpr ck_tile::index_t kBlockSize = 256;
//     constexpr ck_tile::index_t kHeadDim   = 128;

//     return flash_attention_fwd_<QDataType, 
//                                 KDataType, 
//                                 VDataType, 
//                                 ODataType, 
//                                 traits_<SaccDataType, 
//                                         SMPLComputeDataType, 
//                                         PDataType, 
//                                         OaccDataType,
//                                         kBlockSize, 
//                                         kHeadDim, 
//                                         kM0PerBlock, 
//                                         kN0PerBlock, 
//                                         kK0PerBlock, 
//                                         kN1PerBlock, 
//                                         kK1PerBlock>>
//             (a, stream_config);

// }


// TODO: change to only declare
// TODO: fwd_api.cpp
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType>
float flash_attention_fwd(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a, 
                          const stream_config& stream_config);


} // namespace ck_tile
