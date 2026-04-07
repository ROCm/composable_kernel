// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// A is block distributed tensor
// A scale is block distributed tensor
// B is block window on shared memory
// B scale is block distributed tensor
// C is block distributed tensor
// It supports only warp gemms with transposed C.
// TargetCMPerLane_ controls how many consecutive elements of matrix C are calculated by each lane.
template <typename Problem_, typename Policy_, index_t TargetCMPerLane_ = -1>
struct BlockGemmMxARegBSmemCRegV1
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WarpGemm = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
    static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
    static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

    static constexpr index_t CMPerLane = WarpGemm::WarpGemmAttribute::Impl::kCM0PerLane *
                                         WarpGemm::WarpGemmAttribute::Impl::kCM1PerLane;
    static constexpr index_t TargetCMPerLane = max(CMPerLane, TargetCMPerLane_);

    static_assert(TargetCMPerLane % CMPerLane == 0);
    static constexpr index_t NIterPack = TargetCMPerLane / CMPerLane;

    // C += A * B
    template <typename CBlockTensor,
              typename ABlockTensorTmp,
              typename AScaleBlockTensorTmp,
              typename BBlockWindowTmp,
              typename BScaleBlockTensorTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensorTmp& a_block_tensor_tmp,
                                   const AScaleBlockTensorTmp& a_scale_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp,
                                   const BScaleBlockTensorTmp& b_scale_block_tensor_tmp) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                      std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                      std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>);

        static_assert(MPerBlock == ABlockTensorTmp{}.get_lengths()[number<0>{}] &&
                      NPerBlock == BBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                      KPerBlock == ABlockTensorTmp{}.get_lengths()[number<1>{}]);

        const index_t iNWarp = get_warp_id() % NWarp;

        // construct A-block-tensor from A-Block-tensor-tmp
        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());
        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        auto a_scale_block_tensor =
            make_static_distributed_tensor<remove_cv_t<typename AScaleBlockTensorTmp::DataType>>(
                MakeAScaleBlockTileDistribution());
        a_scale_block_tensor.get_thread_buffer() = a_scale_block_tensor_tmp.get_thread_buffer();

        auto b_scale_block_tensor =
            make_static_distributed_tensor<remove_cv_t<typename BScaleBlockTensorTmp::DataType>>(
                MakeBScaleBlockTileDistribution());
        b_scale_block_tensor.get_thread_buffer() = b_scale_block_tensor_tmp.get_thread_buffer();

        // Construct B-warp-window
        // Matrix B is shuffled in such a way that each lane calculates TargetCMPerLane consecutive
        // elements of matrix C. See MakeBScaleBlockTileDistribution and MakeCBlockTile that shuffle
        // B scale and C in the same way.
        auto b_warp_window_tmp = [&] {
            using Impl = typename WarpGemm::WarpGemmAttribute::Impl;

            constexpr index_t N3 = Impl::kCM1PerLane;
            constexpr index_t N2 = TargetCMPerLane / N3;
            constexpr index_t N1 = Impl::kCMLane;
            constexpr index_t N0 = NPerBlock / (N1 * N2 * N3);

            const auto b_lds_unmerged = transform_tensor_view(
                b_block_window_tmp.get_bottom_tensor_view(),
                make_tuple(make_unmerge_transform(
                               make_tuple(number<N0>{}, number<N1>{}, number<N2>{}, number<N3>{})),
                           make_pass_through_transform(number<KPerBlock>{})),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0, 2, 1, 3>{}, sequence<4>{}));

            const auto b_lds_merged = transform_tensor_view(
                b_lds_unmerged,
                make_tuple(make_merge_transform(
                               make_tuple(number<N0>{}, number<N2>{}, number<N1>{}, number<N3>{})),
                           make_pass_through_transform(number<KPerBlock>{})),
                make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tile_window(
                b_lds_merged,
                make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
                b_block_window_tmp.get_window_origin() + multi_index<2>{iNWarp * WarpGemm::kN, 0},
                make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));
        }();

        // check C-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeCBlockTile()
                                                       .get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>);

        using AWarpDstr = typename WarpGemm::AWarpDstr;
        using CWarpDstr = typename WarpGemm::CWarpDstr;

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        using AScaleWarpDstr =
            remove_cvref_t<decltype(make_static_tile_distribution(MakeAScaleWarpDstrEncoding()))>;
        using AScaleWarpTensor =
            static_distributed_tensor<remove_cv_t<typename AScaleBlockTensorTmp::DataType>,
                                      AScaleWarpDstr>;

        using BScaleWarpDstr =
            remove_cvref_t<decltype(make_static_tile_distribution(MakeBScaleWarpDstrEncoding()))>;
        using BScaleWarpTensor =
            static_distributed_tensor<remove_cv_t<typename BScaleBlockTensorTmp::DataType>,
                                      BScaleWarpDstr>;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        constexpr auto a_scale_warp_y_lengths =
            to_sequence(AScaleWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_scale_warp_y_lengths =
            to_sequence(BScaleWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_scale_warp_y_index_zeros =
            uniform_sequence_gen_t<AScaleWarpDstr::NDimY, 0>{};
        constexpr auto b_scale_warp_y_index_zeros =
            uniform_sequence_gen_t<BScaleWarpDstr::NDimY, 0>{};

        // hot loop:
        static_ford<sequence<KIterPerWarp, NIterPerWarp>>{}([&](auto kn) {
            constexpr auto kIter = number<kn[number<0>{}]>{};
            constexpr auto nIter = number<kn[number<1>{}]>{};
            auto b_warp_window   = b_warp_window_tmp;
            move_tile_window(
                b_warp_window,
                {nIter * (NPerBlock / NIterPerWarp), kIter * (KPerBlock / KIterPerWarp)});
            // read B warp tensor from B Block window
            const auto b_warp_tensor = load_tile(b_warp_window);

            BScaleWarpTensor b_scale_warp_tensor;

            b_scale_warp_tensor.get_thread_buffer() = b_scale_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<nIter / NIterPack, nIter % NIterPack, kIter>{},
                                b_scale_warp_y_index_zeros),
                merge_sequences(sequence<1, 1, 1>{}, b_scale_warp_y_lengths));

            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;

                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                AScaleWarpTensor a_scale_warp_tensor;

                a_scale_warp_tensor.get_thread_buffer() =
                    a_scale_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_scale_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_scale_warp_y_lengths));

                // read C warp tensor from C block tensor
                CWarpTensor c_warp_tensor;

                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter / NIterPack, nIter % NIterPack>{},
                                    c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1, 1>{}, c_warp_y_lengths));

                // warp GEMM
                WarpGemm{}.template operator()<0, 0>(
                    c_warp_tensor,
                    a_warp_tensor,
                    b_warp_tensor,
                    int32_t(a_scale_warp_tensor.get_thread_buffer()[0]),
                    int32_t(b_scale_warp_tensor.get_thread_buffer()[0]));

                // write C warp tensor into C block tensor
                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter / NIterPack, nIter % NIterPack>{},
                                    c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());
            });
        });
    }

    template <index_t MPerBlock_ = MPerBlock, index_t KPerBlock_ = KPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeABlockTileDistribution()
    {
        constexpr index_t MIterPerWarp_ = MPerBlock_ / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp_ = KPerBlock_ / WarpGemm::kK;

        constexpr auto a_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<NWarp>,
            tuple<sequence<MIterPerWarp_, MWarp>, sequence<KIterPerWarp_>>,
            tuple<sequence<1, 0>>,
            tuple<sequence<1, 0>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return make_static_tile_distribution(a_block_dstr_encode);
    }

    CK_TILE_DEVICE static constexpr auto MakeAScaleWarpDstrEncoding()
    {
        using Impl = typename WarpGemm::WarpGemmAttribute::Impl;

        constexpr index_t AScaleMLane     = Impl::kAMLane;
        constexpr index_t ABScaleKLane    = Impl::kABKLane;
        constexpr index_t ABScaleKPerLane = Impl::kABKPerLane / Impl::kScaleGranularity;

        return ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<AScaleMLane>,
                           ck_tile::sequence<ABScaleKLane, ABScaleKPerLane>>,
            ck_tile::tuple<ck_tile::sequence<2, 1>>,
            ck_tile::tuple<ck_tile::sequence<0, 0>>,
            ck_tile::sequence<2>,
            ck_tile::sequence<1>>{};
    }

    CK_TILE_DEVICE static constexpr auto MakeBScaleWarpDstrEncoding()
    {
        using Impl = typename WarpGemm::WarpGemmAttribute::Impl;

        constexpr index_t BScaleNLane     = Impl::kBNLane;
        constexpr index_t ABScaleKLane    = Impl::kABKLane;
        constexpr index_t ABScaleKPerLane = Impl::kABKPerLane / Impl::kScaleGranularity;

        return ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<BScaleNLane>,
                           ck_tile::sequence<ABScaleKLane, ABScaleKPerLane>>,
            ck_tile::tuple<ck_tile::sequence<2, 1>>,
            ck_tile::tuple<ck_tile::sequence<0, 0>>,
            ck_tile::sequence<2>,
            ck_tile::sequence<1>>{};
    }

    template <index_t MPerBlock_ = MPerBlock, index_t KPerBlock_ = KPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeAScaleBlockTileDistribution()
    {
        constexpr index_t MIterPerWarp_ = MPerBlock_ / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp_ = KPerBlock_ / WarpGemm::kK;

        constexpr auto a_scale_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<NWarp>,
            tuple<sequence<MIterPerWarp_, MWarp>, sequence<KIterPerWarp_>>,
            tuple<sequence<1, 0>>,
            tuple<sequence<1, 0>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_scale_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_scale_block_outer_dstr_encoding, MakeAScaleWarpDstrEncoding());

        return make_static_tile_distribution(a_scale_block_dstr_encode);
    }

    template <index_t NPerBlock_ = NPerBlock, index_t KPerBlock_ = KPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeBScaleBlockTileDistribution()
    {
        constexpr index_t NIterPerWarp_ = NPerBlock_ / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp_ = KPerBlock_ / WarpGemm::kK;

        using Impl = typename WarpGemm::WarpGemmAttribute::Impl;

        constexpr index_t ABScaleKLane    = Impl::kABKLane;
        constexpr index_t ABScaleKPerLane = Impl::kABKPerLane / Impl::kScaleGranularity;

        constexpr auto b_scale_block_dstr_encode = ck_tile::tile_distribution_encoding<
            ck_tile::sequence<MWarp>,
            ck_tile::tuple<ck_tile::sequence<NIterPerWarp_ / NIterPack,
                                             NWarp,
                                             Impl::kCMLane,
                                             NIterPack,
                                             Impl::kCM0PerLane,
                                             Impl::kCM1PerLane>,
                           ck_tile::sequence<KIterPerWarp_, ABScaleKLane, ABScaleKPerLane>>,
            ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<2, 1, 1, 1>>,
            ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<1, 4, 2, 5>>,
            ck_tile::sequence<1, 1, 2, 2>,
            ck_tile::sequence<0, 3, 0, 2>>{};

        return make_static_tile_distribution(b_scale_block_dstr_encode);
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        using Impl = typename WarpGemm::WarpGemmAttribute::Impl;

        constexpr auto c_block_dstr_encode = ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<MIterPerWarp, MWarp, Impl::kCNLane>,
                           ck_tile::sequence<NIterPerWarp / NIterPack,
                                             NWarp,
                                             Impl::kCMLane,
                                             NIterPack,
                                             Impl::kCM0PerLane,
                                             Impl::kCM1PerLane>>,
            ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<2, 1>>,
            ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
            ck_tile::sequence<1, 2, 2, 2, 2>,
            ck_tile::sequence<0, 0, 3, 4, 5>>{};

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    // C = A * B
    template <typename ABlockTensorTmp,
              typename AScaleBlockTensorTmp,
              typename BBlockWindowTmp,
              typename BScaleBlockTensorTmp>
    CK_TILE_DEVICE auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                                   const AScaleBlockTensorTmp& a_scale_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp,
                                   const BScaleBlockTensorTmp& b_scale_block_tensor_tmp) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor,
                   a_block_tensor_tmp,
                   a_scale_block_tensor_tmp,
                   b_block_window_tmp,
                   b_scale_block_tensor_tmp);
        return c_block_tensor;
    }
};

} // namespace ck_tile
