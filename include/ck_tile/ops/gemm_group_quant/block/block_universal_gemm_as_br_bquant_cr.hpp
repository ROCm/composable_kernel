// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1_custom_policy.hpp"

namespace ck_tile {

// A is block window on shared memory
// BQ (scale tensor) is block distributed tensor.
// Consecutive kQuantGroupSize elements of B are quantized with a separate scale.
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename BlockPolicy_>
struct BlockGemmWeightPreshuffleBQuantASmemBRegCRegV1
{
    using Problem        = remove_cvref_t<Problem_>;
    using BlockPolicy    = remove_cvref_t<BlockPolicy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using BQDataType     = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using ComputeDataType= remove_cvref_t<typename Problem::ComputeDataType>;
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
    // static_assert(std::is_same_v<ADataType, ck_tile::fp8_t>, "ADataType must be pk_int4_t.");
    // static_assert(std::is_same_v<BDataType, ck_tile::fp8_t>, "BDataType must be pk_int4_t.");

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

    static constexpr index_t InterWaveSchedulingMacClusters = 1;

    static constexpr index_t KPack      = WG::kKPerThread;
    static constexpr index_t KPerThread = KIterPerWarp * WG::kKPerThread; //64

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
    template <typename WarpWindow, typename WarpTile>
    CK_TILE_DEVICE static void load_interleaved_pk_type(WarpTile& warp_tile,
                                                        const WarpWindow& warp_window)
    {
        const element_wise::PassThroughPack8 elementwise_op{};
        const index_t UnaryOpSize = 8;
        //static_assert(WarpTile::get_thread_buffer_size() == 0, "Get thread buffer size must be 0.");
        static_assert(WarpTile::get_thread_buffer_size() % UnaryOpSize == 0);
        constexpr index_t thread_buffer_size = WarpTile::get_thread_buffer_size() / UnaryOpSize;
        const auto in_dstr_tensors           = load_tile(warp_window);


        if(get_block_id() == 0 && get_warp_id() == 0 && get_thread_id() == 0){
            auto& a_tb  = in_dstr_tensors.get_thread_buffer();
            printf("---------Warp window thread buffer size=%d, first up to 16:\n", int(decltype(in_dstr_tensors)::get_thread_buffer_size()));
            for(int j = 0; j < (thread_buffer_size < 16 ? thread_buffer_size : 16); ++j)
            {
                float v = pk_int4_t_to_fp32x2_t(a_tb.at(j)).x;
                printf(" --------- Warp Window[%d]=%f\n", j, v);
            }
            //  printf("type convert input[%d] : %f\n",
            //          get_thread_id(), pk_int4_t_to_fp32x2_t((in_dstr_tensors.get_thread_buffer().at(get_thread_id()))).x);
        }
        using ComputeVectorType = ComputeDataType __attribute__((ext_vector_type(UnaryOpSize)));
        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
            elementwise_op(warp_tile.get_thread_buffer().template get_as<ComputeVectorType>()(i),
                           in_dstr_tensors.get_thread_buffer().template get_as<pk_int4x4_t>()[i]);
        });

        if(get_block_id() == 0 && get_warp_id() == 0 && get_thread_id() == 0){
            auto& a_tb  = warp_tile.get_thread_buffer();
            printf("*******Warp Tile thread buffer size=%d, first up to 16:\n", int(WarpTile::get_thread_buffer_size()));
            for(int j = 0; j < (thread_buffer_size < 16 ? thread_buffer_size : 16); ++j)
            {
                float v = type_convert<float>(a_tb.at(j));
                printf(" ******* Warp Tile[%d]=%f\n", j, v);
            }
        }


        // if(get_block_id() == 0 && get_warp_id() == 0){
        //      printf("type_convert output[%d]: %f\n",
        //              get_thread_id(), type_convert<float>(warp_tile.get_thread_buffer().at(get_thread_id())));
        // }
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
