// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1_custom_policy.hpp"

namespace ck_tile {

// A is block window on shared memory
// BQ (scale tensor) is block distributed tensor.
// Consecutive kQuantGroupSize elements of A are quantized with a separate scale.
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_, typename BlockPolicy_>
struct BlockGemmBQuantASmemBRegCRegV1
{
    using Problem        = remove_cvref_t<Problem_>;
    using BlockPolicy    = remove_cvref_t<BlockPolicy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using BQDataType     = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr auto config = BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr uint8_t kA_cvt_scale = std::is_same_v<ADataType, pk_int4_t> ? 16 : 1;
    static constexpr uint8_t kB_cvt_scale = std::is_same_v<BDataType, pk_int4_t> ? 16 : 1;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t kQuantGroupSize = Problem::kQuantGroupSize;
    static constexpr index_t kBlockSize      = Problem::kBlockSize;

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp =
        BlockTile::at(idxN) / (WarpTile::at(idxN) * BlockWarps::at(idxN));
    static constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    static constexpr index_t KPerBlockBQ = KPerBlock / kQuantGroupSize;

    static constexpr index_t QScalesPerBlockRow =
        (KPerBlock + kQuantGroupSize - 1) / kQuantGroupSize;

    static constexpr index_t QScalesPerWarpGemmRow =
        (WG::kK + kQuantGroupSize - 1) / kQuantGroupSize;

    static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;

    template <typename T>
    CK_TILE_DEVICE static float cvt_scale_to_fp32(T& scale)
    {
        float scale_reg_f = 0.f;
        if constexpr(std::is_same_v<BQDataType, ck_tile::fp8_t>)
        {
            scale_reg_f = element_wise::amd_assembly_fp8_to_fp32(static_cast<uint32_t>(scale));
        }
        else if constexpr(std::is_same_v<BQDataType, ck_tile::bf8_t>)
        {
            scale_reg_f = element_wise::amd_assembly_bf8_to_fp32(static_cast<uint32_t>(scale));
        }
        else if constexpr(std::is_same_v<BQDataType, float>)
        {
            scale_reg_f = ck_tile::bit_cast<float>(scale);
        }
        else
        {
            static_assert(false, "BQDataType must be float, fp8_t or bf8_t.");
        }
        return scale_reg_f;
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    // C += A * B
    template <typename CBlockTensor,
              typename ABlockWindow,
              typename BFlatBlockTensor,
              typename BQBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   ABlockWindow& a_warp_windows,
                                   BFlatBlockTensor& b_warp_tensor,
                                   BQBlockTensor& bq_block_tensor) const
    {
        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                CWarpTensor c_warp_tensor;

                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    // read A warp tensor from A block tensor
                    auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                    // warp GEMM
                    if constexpr(kIter % KIterPerQScale == 0)
                    {
                        c_warp_tensor = WG{}(a_warp_tensor, b_warp_tensor(nIter)(kIter));
                    }
                    else
                    {
                        WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor(nIter)(kIter));
                    }

                    if constexpr((kIter + 1) % KIterPerQScale == 0)
                    {
                        constexpr index_t reg_offset =
                            nIter * KPerBlockBQ + ((kIter * WG::kK) / kQuantGroupSize);

                        constexpr auto tbuf_offset =
                            number<typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
                                       merge_sequences(sequence<mIter, nIter>{},
                                                       c_warp_y_index_zeros)) /
                                   CBlockTensor::PackedSize>{};

                        auto& scale_reg   = bq_block_tensor.get_thread_buffer()[reg_offset];
                        float scale_reg_f = cvt_scale_to_fp32(scale_reg);
                        static_for<0, WG::kM / 2, 1>{}([&](auto c_row) {
                            c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
                                (c_warp_tensor.get_thread_buffer()[c_row] * scale_reg_f *
                                 kA_cvt_scale * kB_cvt_scale);
                        });
                    }
                });
            });
        });
    }
};

} // namespace ck_tile
