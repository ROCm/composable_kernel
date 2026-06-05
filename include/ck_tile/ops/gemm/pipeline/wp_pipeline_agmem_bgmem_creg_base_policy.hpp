// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_wp_asmem_breg_creg.hpp"
#include "ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

#include <type_traits>

namespace ck_tile {

struct UniversalWeightPreshufflePipelineAgBgCrPolicy
    : public UniversalGemmBasePolicy<UniversalWeightPreshufflePipelineAgBgCrPolicy>
{
    using BasePolicy = UniversalGemmBasePolicy<UniversalWeightPreshufflePipelineAgBgCrPolicy>;

    static constexpr index_t kDramLoadPackBytes = 128;
    static constexpr index_t MaxVecSize         = get_max_mem_vec_inst_width();

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a = sizeof(typename Problem::ADataType) *
                                        MakeALdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();

        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackA()
    {
        return Problem::VectorLoadSize / sizeof(typename Problem::ADataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKBPerLoad()
    {
        using TileShape = typename Problem::BlockGemmShape;
#if defined(__gfx11__)
        constexpr index_t scale = 4;
#else
        constexpr index_t scale = get_warp_size() == 32 ? 2 : 1;
#endif
        if constexpr(TileShape::WarpTile::at(I1) == 32)
        {
            return TileShape::WarpTile::at(I2) * scale / 2;
        }
        else
        {
            static_assert(TileShape::WarpTile::at(I1) == 16);
            return TileShape::WarpTile::at(I2) * scale / 4;
        }
    }

    // The swizzle factor is defined based on the number of contiguous lanes
    // in the instruction.
    template <typename Problem>
    static constexpr auto get_swizzle_factor = [](auto kpack) {
        using WarpTile              = typename Problem::BlockGemmShape::WarpTile;
        using ADataType             = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t WarpTileK = WarpTile::at(I2);
        constexpr bool is_8bit_float =
            std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, bf8_t>;
        constexpr index_t NumAccess = is_8bit_float ? 2 : 1;
        return WarpTileK / kpack / NumAccess;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptorAsync()
    {
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        using ADataType               = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t KPack       = MaxVecSize / sizeof(ADataType) * APackedSize;
        constexpr index_t PacksPerLdsRow =
            (kDramLoadPackBytes / sizeof(ADataType) * APackedSize) / KPack;

        constexpr index_t K2 = KPack;
        constexpr index_t K1 = PacksPerLdsRow;
        constexpr index_t K0 = KPerBlock / (K1 * K2);
        static_assert(K0 * K1 * K2 == KPerBlock, "K0, K1, K2 must cover whole KPerBlock!");

        constexpr index_t WaveSize = get_warp_size();
        using TileShape            = typename Problem::BlockGemmShape;
        constexpr index_t MPerXdl  = TileShape::WarpTile::at(I0);

        constexpr index_t M3 = get_swizzle_factor<Problem>(KPack);
        static_assert(M3 > 0,
                      "Invalid swizzle factor (M3==0). Check WarpTileK, KPack, and NumAccess.");
        static_assert((WaveSize % (K1 * M3)) == 0,
                      "WaveSize must be divisible by K1*M3 for the async LDS layout.");
        constexpr index_t M2 = WaveSize / K1 / M3;
        constexpr index_t M1 = MPerXdl / (M2 * M3);
        constexpr index_t M0 = MPerBlock / (M1 * M2 * M3);
        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "M0, M1, M2, M3 must cover whole MPerBlock!");

        constexpr index_t Pad = 4 * K2;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<M0>{},
                       number<K0>{},
                       number<M1>{},
                       number<M2>{},
                       number<M3>{},
                       number<K1>{},
                       number<K2>{}),
            make_tuple(number<K0*(M1 * (M2 * M3 * K1 * K2) + (M1 - 1) * Pad)>{},
                       number<M1*(M2 * M3 * K1 * K2) + (M1 - 1) * Pad>{},
                       number<M2 * M3 * K1 * K2 + Pad>{},
                       number<M3 * K1 * K2>{},
                       number<K1 * K2>{},
                       number<K2>{},
                       number<1>{}),
            number<K2>{},
            number<1>{});

        constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<K0>{}),
                       make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<M2>{}),
                       make_xor_transform(make_tuple(number<M3>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}));
        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<M0>{}, number<M1>{}, number<M2>{}, number<M3>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<K0>{}, number<K1>{}, number<K2>{}))),
            make_tuple(sequence<0, 2, 3, 4>{}, sequence<1, 5, 6>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        if constexpr(Problem::Async)
        {
            return MakeALdsBlockDescriptorAsync<Problem>();
        }
        else
        {
            return BasePolicy::template MakeALdsBlockDescriptor<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(Problem::Async)
        {
            constexpr index_t VecLoadSize   = Problem::VectorLoadSize / sizeof(ADataType);
            constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

            if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                using TileEncodingPattern =
                    tile_distribution_encoding_pattern_2d<BlockSize,
                                                          MPerBlock,
                                                          KPerBlock,
                                                          VecLoadSize,
                                                          tile_distribution_pattern::warp_raked,
                                                          NumWaveGroups>;
                return TileEncodingPattern::make_2d_static_tile_distribution();
            }
            else
            {
                using TileEncodingPattern =
                    tile_distribution_encoding_pattern_2d<BlockSize,
                                                          KPerBlock,
                                                          MPerBlock,
                                                          VecLoadSize,
                                                          tile_distribution_pattern::warp_raked,
                                                          NumWaveGroups>;
                return TileEncodingPattern::make_2d_static_tile_distribution();
            }
        }
        else
        {
            if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                constexpr index_t M1           = Problem::VectorLoadSize / sizeof(ADataType);
                constexpr index_t M0           = MPerBlock / M1;
                constexpr index_t total_pixels = MPerBlock * KPerBlock / BlockSize;
                static_assert(total_pixels % M1 == 0);
                constexpr index_t K3    = total_pixels / M1;
                constexpr index_t KPack = GetSmemPackA<Problem>();
                static_assert(KPack % K3 == 0);
                constexpr index_t K2 = KPack / K3;
                if constexpr(get_warp_size() >= (K2 * M0))
                {
                    constexpr index_t K1 = get_warp_size() / (K2 * M0);
                    constexpr index_t K0 = BlockSize / get_warp_size();
                    static_assert(KPerBlock == K0 * K1 * K2 * K3);
                    return make_static_tile_distribution(
                        tile_distribution_encoding<
                            sequence<1>,
                            tuple<sequence<M0, M1>, sequence<K0, K1, K2, K3>>,
                            tuple<sequence<2>, sequence<2, 1, 2>>,
                            tuple<sequence<0>, sequence<1, 0, 2>>,
                            sequence<2, 1>,
                            sequence<3, 1>>{});
                }
                else
                {
                    constexpr index_t K1   = (K2 * M0) / get_warp_size();
                    constexpr index_t K2_m = K2 / K1;
                    constexpr index_t K0   = BlockSize / get_warp_size() / K1;
                    static_assert(KPerBlock == K0 * K1 * K2_m * K3);
                    return make_static_tile_distribution(
                        tile_distribution_encoding<
                            sequence<1>,
                            tuple<sequence<M0, M1>, sequence<K0, K1, K2_m, K3>>,
                            tuple<sequence<2, 2>, sequence<1, 2>>,
                            tuple<sequence<0, 1>, sequence<0, 2>>,
                            sequence<2, 1>,
                            sequence<3, 1>>{});
                }
            }
            else
            {
                constexpr index_t K1 = Problem::VectorLoadSize / sizeof(ADataType);
                constexpr index_t K0 = KPerBlock / K1;
                constexpr index_t M2 = get_warp_size() / K0;
                // coalesce reading for each blocks
                if constexpr(get_warp_size() % (M2 * K0) == 0)
                {
                    // block raked
                    constexpr index_t M1 = BlockSize / get_warp_size();
                    static_assert(M2 != 0,
                                  "M2 is zero, which will lead to a division by zero error.");
                    static_assert(M1 != 0,
                                  "M1 is zero, which will lead to a division by zero error.");
                    constexpr index_t M0 = MPerBlock / (M2 * M1);
                    static_assert(M0 * M1 * M2 == MPerBlock,
                                  "Incorrect M0, M2, M1 configuration! "
                                  "M0, M1, M2 must cover whole MPerBlock!");

                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<1>,
                                                   tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                                   tuple<sequence<1>, sequence<1, 2>>,
                                                   tuple<sequence<1>, sequence<2, 0>>,
                                                   sequence<1, 2>,
                                                   sequence<0, 1>>{});
                }
                else
                {
                    // warp raked
                    constexpr index_t M0 = BlockSize / get_warp_size();
                    constexpr index_t M1 = MPerBlock / (M2 * M0);
                    static_assert(M0 * M1 * M2 == MPerBlock,
                                  "Incorrect M0, M1, M2 configuration! "
                                  "M0, M1, M2 must cover whole MPerBlock!");
                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<1>,
                                                   tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                                   tuple<sequence<1>, sequence<1, 2>>,
                                                   tuple<sequence<0>, sequence<2, 0>>,
                                                   sequence<1, 2>,
                                                   sequence<1, 1>>{});
                }
            }
        }
    }

    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeADramWindow(const WindowTmp& window_tmp)
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(Problem::Async)
        {
            using ADataType               = remove_cvref_t<typename Problem::ADataType>;
            constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;

            constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
            static_assert(ndims == 2, "only support 2D tensor");
            auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
            const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

            constexpr index_t KPack = MaxVecSize / sizeof(ADataType) * APackedSize;
            constexpr index_t PacksPerLdsRow =
                (kDramLoadPackBytes / sizeof(ADataType) * APackedSize) / KPack;

            constexpr index_t K2 = KPack;
            constexpr index_t K1 = PacksPerLdsRow;
            const index_t K0     = cols / (K1 * K2);
            const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

            constexpr index_t M1 = get_swizzle_factor<Problem>(KPack);
            const index_t M0     = integer_divide_ceil(rows, M1);
            const auto row_lens  = make_tuple(M0, number<M1>{});

            // fully-packed descriptor with element_space_size = M0*M1*K0*K1*K2 =
            // MPerBlock*KPerBlock (only needed for decltype)
            const auto d0 =
                make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
            // get transforms of d0 tensor descriptor
            const auto desc_0 =
                decltype(d0)(d0.get_transforms(),
                             tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
            const auto desc_1 = transform_tensor_descriptor(
                desc_0,
                make_tuple(make_pass_through_transform(M0),
                           make_xor_transform(make_tuple(number<M1>{}, number<K1>{})),
                           make_pass_through_transform(K0),
                           make_pass_through_transform(number<K2>{})),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));

            const auto desc = transform_tensor_descriptor(
                desc_1,
                make_tuple(make_merge_transform_v3_division_mod(row_lens),
                           make_merge_transform_v3_division_mod(col_lens)),
                make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            auto&& byte_ptr         = &(tensor_view_tmp.get_buffer_view()(0));
            auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

            auto&& origin_tmp = window_tmp.get_window_origin();

            // Create tile distribution vmem load
            constexpr index_t BlockSize = Problem::kBlockSize;
            constexpr index_t WaveSize  = get_warp_size();
            constexpr index_t K0_dstr   = KPerBlock / (K2 * K1);
            constexpr index_t M2_dstr   = WaveSize / K1;
            constexpr index_t M1_dstr   = BlockSize / WaveSize;
            constexpr index_t M0_dstr   = MPerBlock / (M1_dstr * M2_dstr);

            const auto tile_dstr = make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M0_dstr, M1_dstr, M2_dstr>, sequence<K0_dstr, K1, K2>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<1>, sequence<2, 1>>,
                    sequence<1, 2, 2>,
                    sequence<0, 0, 2>>{});

            return make_tile_window(byte_tensor_view,
                                    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                    {origin_tmp[0], origin_tmp[1]},
                                    tile_dstr);
        }
        else
        {
            return make_tile_window(window_tmp.get_bottom_tensor_view(),
                                    make_tuple(MPerBlock, KPerBlock),
                                    window_tmp.get_window_origin(),
                                    MakeADramTileDistribution<Problem>());
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;
        using BDataType = typename Problem::BDataType;

        constexpr index_t kNPerBlock = TileShape::kN;
        constexpr index_t kKPerBlock = TileShape::kK;
        constexpr index_t NIterPerWarp =
            kNPerBlock / TileShape::BlockWarps::at(I1) / TileShape::WarpTile::at(I1);
        constexpr index_t KIterPerWarp = kKPerBlock / TileShape::WarpTile::at(I2);

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

#if defined(__gfx11__)
        constexpr index_t KRepeatInWave = 2;
#else
        constexpr index_t KRepeatInWave = 1;
#endif
        constexpr index_t KBPerLoad = min(
            GetKBPerLoad<Problem>(), KRepeatInWave * 16 / static_cast<index_t>(sizeof(BDataType)));
        constexpr index_t KThdPerWave = WaveSize / KRepeatInWave; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t KRepeat     = KIterPerWarp;
        constexpr index_t KAccess     = GetKBPerLoad<Problem>() / KBPerLoad;
        static_assert(TileShape::flatKPerWarp == KAccess * KThdPerWave * KBPerLoad, "wrong");

        constexpr index_t NBPerLoad   = 1;
        constexpr index_t NThdPerWave = 1;
        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp
        constexpr index_t NRepeat     = NIterPerWarp;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat, KRepeatInWave>,                          // ?
                tuple<sequence<NRepeat, NWavePerBlk, NThdPerWave, NBPerLoad>, // second direction
                      sequence<KRepeat, KAccess, KWavePerBlk, KThdPerWave, KBPerLoad>>,
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<0, 1, 2>, sequence<0, 1, 2>>, // which direction
                tuple<sequence<0, 1, 2>, sequence<1, 2, 3>>, // which index
                // <repeat, vec_load>
                sequence<1, 2, 1, 2, 2>,
                sequence<0, 0, 3, 1, 4>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegBlockDistribution()
    {
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t M1           = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t M0           = kMPerBlock / M1;
        constexpr index_t total_pixels = kMPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % M1 == 0);
        constexpr index_t K3     = total_pixels / M1;
        constexpr index_t kKPack = GetSmemPackA<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t warp_size = get_warp_size();
        if constexpr(warp_size >= (K2 * M0))
        {
            constexpr index_t K1 = warp_size / (K2 * M0);
            constexpr index_t K0 = kBlockSize / warp_size;

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1>, sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<2>, sequence<2, 1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * M0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1>, sequence<K0, K1, K2_m, K3>>,
                                           tuple<sequence<2, 2>, sequence<1, 2>>,
                                           tuple<sequence<0, 1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return GetBlockWeightPreshuffle<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWeightPreshuffle()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        using AComputeDataType = remove_cvref_t<typename Problem::AComputeDataType>;
        using BComputeDataType = remove_cvref_t<typename Problem::BComputeDataType>;
        using ADataType        = remove_cvref_t<typename Problem::ADataType>;
        using BDataType        = remove_cvref_t<typename Problem::BDataType>;
        // Determine compute types to use
        // This logic defaults to A/B DataType, but if one of them is packed falls back to the other
        // If both are packed, it falls back to the explicitly defined ComputeDataType in the
        // problem It might be a good idea to use ComputeDataType anyway, but that would break how
        // this behaviour used to work
        using ATypeToUse =
            mixed_prec_compute_type_from_input_t<ADataType, BDataType, AComputeDataType>;
        using BTypeToUse =
            mixed_prec_compute_type_from_input_t<BDataType, ADataType, BComputeDataType>;
#if defined(__gfx11__) || defined(__gfx12__)
        constexpr auto NumAccess = WGAttrNumAccessEnum::Default;
#else
        constexpr index_t WaveSize = get_warp_size();
        constexpr index_t KLane    = WarpTile::at(I2) * WarpTile::at(I0) / WaveSize;
        // When BDataType is pk_int4_t, it is internally converted to fp8 for computation.
        constexpr index_t KLaneBytes = KLane * sizeof(BTypeToUse);
        constexpr auto NumAccess     = static_cast<WGAttrNumAccessEnum>(max(1, KLaneBytes / 16));
#endif
        using WarpGemm = WarpGemmDispatcher<ATypeToUse,
                                            BTypeToUse,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            NumAccess>;

        using BlockWeightPreshufflePolicy =
            BlockWeightPreshuffleASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                              typename Problem::BDataType,
                                                              typename Problem::CDataType,
                                                              BlockWarps,
                                                              WarpGemm>;
        return BlockWeightPreshuffleASmemBRegCReg<Problem, BlockWeightPreshufflePolicy>{};
    }
    /**
     * @brief Get the vector store size for C tensor.
     *
     * @tparam Problem - Gemm pipeline problem class.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        using BlockGemm = remove_cvref_t<decltype(GetBlockWeightPreshuffle<Problem>())>;
        using WG_       = typename BlockGemm::WarpGemm;

        constexpr bool TransposeC = Problem::TransposeC;
        using CLayout             = typename Problem::CLayout;
        using CWarpDstr           = typename WG_::CWarpDstr;

        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has multiple consecutive elements in
                // N dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG_::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                // In this case each thread has just a single item in Ndim
                return WG_::WarpGemmAttribute::Impl::kCNLane / WG_::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has just a single item in Mdim
                return WG_::WarpGemmAttribute::Impl::kCNLane / WG_::kN;
            }
            else
            {
                // In this case each thread has multiple consecutive elements in
                // M dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG_::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }
};

} // namespace ck_tile
