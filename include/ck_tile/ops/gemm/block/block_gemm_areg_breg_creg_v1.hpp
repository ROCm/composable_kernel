// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_default_policy.hpp"
#include "ck_tile/core/tensor/tile_window_utils.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_,
          typename Policy_ = BlockGemmARegBRegCRegV1DefaultPolicy,
          bool TransposeC_ = false>
struct BlockGemmARegBRegCRegV1
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem        = remove_cvref_t<PipelineProblem_>;
        using Policy         = remove_cvref_t<GemmPolicy_>;
        using ADataType      = remove_cvref_t<typename Problem::ADataType>;
        using BDataType      = remove_cvref_t<typename Problem::BDataType>;
        using CDataType      = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        static constexpr auto KSubTileNum = Policy::KSubTileNum;

        static constexpr index_t MWarp        = config.template at<1>();
        static constexpr index_t NWarp        = config.template at<2>();
        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static constexpr index_t KPackA = WarpGemm::kAKPack;
        static constexpr index_t KPackB = WarpGemm::kBKPack;
    };

    public:
    using Problem                    = remove_cvref_t<Problem_>;
    using Policy                     = remove_cvref_t<Policy_>;
    static constexpr bool TransposeC = TransposeC_;

    using Traits = GemmTraits_<Problem, Policy>;

    using WarpGemm       = typename Traits::WarpGemm;
    using BlockGemmShape = typename Traits::BlockGemmShape;

    using ADataType = remove_cvref_t<typename Traits::ADataType>;
    using BDataType = remove_cvref_t<typename Traits::BDataType>;
    using CDataType = remove_cvref_t<typename Traits::CDataType>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t KSubTileNum = Traits::KSubTileNum;

    static constexpr index_t KPerSubTile = KIterPerWarp / KSubTileNum;

    static constexpr index_t MWarp            = Traits::MWarp;
    static constexpr index_t NWarp            = Traits::NWarp;
    static constexpr bool UseDefaultScheduler = (Problem::NumWaveGroups != 1);

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto a_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<NWarp>,
                                           tuple<sequence<MIterPerWarp>, sequence<KPerSubTile>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};

            constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

            return a_block_dstr_encode;
        }
        else
        {
            constexpr auto a_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<NWarp>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<KPerSubTile>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

            return a_block_dstr_encode;
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto b_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<MWarp>,
                                           tuple<sequence<NIterPerWarp>, sequence<KPerSubTile>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};
            constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

            return b_block_dstr_encode;
        }
        else
        {
            constexpr auto b_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<NIterPerWarp, NWarp>, sequence<KPerSubTile>>,
                tuple<sequence<0, 1>>,
                tuple<sequence<0, 1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

            return b_block_dstr_encode;
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockDistributionEncode()
    {
        using c_distr_ys_major = std::conditional_t<TransposeC, sequence<2, 1>, sequence<1, 2>>;
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<>,
                tuple<>,
                c_distr_ys_major,
                sequence<0, 0>>{};
            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

            return c_block_dstr_encode;
        }
        else
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 1>>,
                c_distr_ys_major,
                sequence<0, 0>>{};
            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

            return c_block_dstr_encode;
        }
    }

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockTensor::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        // check ABC-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeABlockDistributionEncode())>,
                           remove_cvref_t<decltype(ABlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "A distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeBBlockDistributionEncode())>,
                           remove_cvref_t<decltype(BBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "B distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeCBlockDistributionEncode())>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "C distribution is wrong!");

        using AWarpDstr = typename WarpGemm::AWarpDstr;
        using BWarpDstr = typename WarpGemm::BWarpDstr;
        using CWarpDstr = typename WarpGemm::CWarpDstr;

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
            constexpr auto kIter = number<km[number<0>{}]>{};
            constexpr auto mIter = number<km[number<1>{}]>{};
            // read A warp tensor from A Block window
            AWarpTensor a_warp_tensor;
            a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                // read B warp tensor from B block tensor
                BWarpTensor b_warp_tensor;
                b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                // read C warp tensor from C block tensor
                using c_iter_idx =
                    std::conditional_t<TransposeC, sequence<nIter, mIter>, sequence<mIter, nIter>>;
                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                // warp GEMM
                if constexpr(nIter != 0)
                {
                    WarpGemm{}.template operator()<ReuseA<true>, ReuseB<false>>(
                        c_warp_tensor, a_warp_tensor, b_warp_tensor);
                }
                else
                {
                    WarpGemm{}.template operator()<ReuseA<false>, ReuseB<false>>(
                        c_warp_tensor, a_warp_tensor, b_warp_tensor);
                }

                // write C warp tensor into C block tensor
                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());
            });
        });
    }

    // C += A * B with scale value. SubTileIdx: which sub_tile in [0, KSubTileNum), compile-time.
    // used for sub tile based pipelining, where K dimension of block-gemm is divided into
    // KSubTileNum sub-tiles
    template <index_t SubTileIdx,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename AScaleBlockTensor,
              typename BScaleBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor,
                                   const AScaleBlockTensor& a_scale_tensor,
                                   const BScaleBlockTensor& b_scale_tensor) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockTensor::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        // check ABC-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeABlockDistributionEncode())>,
                           remove_cvref_t<decltype(ABlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "A distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeBBlockDistributionEncode())>,
                           remove_cvref_t<decltype(BBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "B distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeCBlockDistributionEncode())>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "C distribution is wrong!");

        using AWarpDstr = typename WarpGemm::AWarpDstr;
        using BWarpDstr = typename WarpGemm::BWarpDstr;
        using CWarpDstr = typename WarpGemm::CWarpDstr;

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr index_t AScaleTypeVal =
            ScaleDataTypeToEnum<typename Problem::AScaleDataType>::value;
        constexpr index_t BScaleTypeVal =
            ScaleDataTypeToEnum<typename Problem::BScaleDataType>::value;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KPerSubTile, 1>{}([&](auto kIter) {
            constexpr index_t scale_k_idx = SubTileIdx * KPerSubTile + decltype(kIter)::value;
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;
                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                index_t scale_a = a_scale_tensor.get_y_sliced_thread_data(
                    sequence<mIter, scale_k_idx, 0>{}, sequence<1, 1, 1>{})[0];

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B block tensor
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    index_t scale_b = b_scale_tensor.get_y_sliced_thread_data(
                        sequence<nIter, scale_k_idx, 0>{}, sequence<1, 1, 1>{})[0];

                    // read C warp tensor from C block tensor
                    using c_iter_idx = std::
                        conditional_t<TransposeC, sequence<nIter, mIter>, sequence<mIter, nIter>>;
                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM with scale
                    if constexpr(nIter != 0)
                    {
                        WarpGemm{}
                            .template operator()<ReuseA<true>,
                                                 ReuseB<false>,
                                                 AScaleDataType<AScaleTypeVal>,
                                                 BScaleDataType<BScaleTypeVal>>(
                                c_warp_tensor, a_warp_tensor, b_warp_tensor, scale_a, scale_b);
                    }
                    else
                    {
                        WarpGemm{}
                            .template operator()<ReuseA<false>,
                                                 ReuseB<false>,
                                                 AScaleDataType<AScaleTypeVal>,
                                                 BScaleDataType<BScaleTypeVal>>(
                                c_warp_tensor, a_warp_tensor, b_warp_tensor, scale_a, scale_b);
                    }

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        using c_distr_ys_major = std::conditional_t<TransposeC, sequence<2, 1>, sequence<1, 2>>;
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<>,
                tuple<>,
                c_distr_ys_major,
                sequence<0, 0>>{};

            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
            constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
            auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
            return c_block_tensor;
        }
        else
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 1>>,
                c_distr_ys_major,
                sequence<0, 0>>{};

            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
            constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
            auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
            return c_block_tensor;
        }
    }

    // C = A * B
    template <typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE auto operator()(const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor, b_block_tensor);
        return c_block_tensor;
    }

    template <WindowSlideMode Mode = WindowSlideMode::Move,
              typename ADstBlockTile,
              typename BDstBlockTile,
              typename ASmemBlockWindow,
              typename BSmemBlockWindow,
              bool ALoadTranspose = false,
              bool BLoadTranspose = false>
    CK_TILE_DEVICE void LocalPrefetch(ADstBlockTile& a_dst_block_tile,
                                      BDstBlockTile& b_dst_block_tile,
                                      ASmemBlockWindow& a_block_window,
                                      BSmemBlockWindow& b_block_window,
                                      bool_constant<ALoadTranspose> = {},
                                      bool_constant<BLoadTranspose> = {})
    {
        constexpr index_t k_sub_tile_offset = KPerSubTile * WarpGemm::kK;
        constexpr auto a_offset             = ALoadTranspose ? multi_index<2>{k_sub_tile_offset, 0}
                                                             : multi_index<2>{0, k_sub_tile_offset};
        constexpr auto b_offset             = BLoadTranspose ? multi_index<2>{k_sub_tile_offset, 0}
                                                             : multi_index<2>{0, k_sub_tile_offset};

        // Load tiles
        if constexpr(ALoadTranspose)
            a_dst_block_tile = load_tile_transpose(a_block_window);
        else
            load_tile(a_dst_block_tile, a_block_window);

        if constexpr(BLoadTranspose)
            b_dst_block_tile = load_tile_transpose(b_block_window);
        else
            load_tile(b_dst_block_tile, b_block_window);

        // Handle window movement
        if constexpr(Mode == WindowSlideMode::Move)
        {
            move_tile_window(a_block_window, a_offset);
            move_tile_window(b_block_window, b_offset);
        }
        else if constexpr(Mode == WindowSlideMode::Reset)
        {
            constexpr index_t reset_offset = KPerSubTile * WarpGemm::kK * (KSubTileNum - 1);
            constexpr auto a_reset         = ALoadTranspose ? multi_index<2>{-reset_offset, 0}
                                                            : multi_index<2>{0, -reset_offset};
            constexpr auto b_reset         = BLoadTranspose ? multi_index<2>{-reset_offset, 0}
                                                            : multi_index<2>{0, -reset_offset};
            move_tile_window(a_block_window, a_reset);
            move_tile_window(b_block_window, b_reset);
        }
        // Mode == WindowSlideMode::Stay: do nothing
    }
};

} // namespace ck_tile
