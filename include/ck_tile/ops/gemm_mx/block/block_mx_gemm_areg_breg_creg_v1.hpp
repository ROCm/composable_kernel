// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_default_policy.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_,
          typename Policy_ = BlockGemmARegBRegCRegV1DefaultPolicy,
          bool TransposeC_ = false>
struct BlockMXGemmARegBRegCRegV1
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

    static constexpr index_t MWarp            = Traits::MWarp;
    static constexpr index_t NWarp            = Traits::NWarp;
    static constexpr bool UseDefaultScheduler = (Problem::NumWaveGroups != 1);

    // Note: distribution encodings have MIterPerWarp and NIterPerWarp contiguous because of scale
    // packing.

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto a_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<NWarp>,
                                           tuple<sequence<MIterPerWarp>, sequence<KIterPerWarp>>,
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
                tuple<sequence<MWarp, MIterPerWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<0, 0>>,
                sequence<1, 2>,
                sequence<1, 0>>{};
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
                                           tuple<sequence<NIterPerWarp>, sequence<KIterPerWarp>>,
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
                tuple<sequence<NWarp, NIterPerWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<0, 1>>,
                tuple<sequence<0, 0>>,
                sequence<1, 2>,
                sequence<1, 0>>{};
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
            using c_distr_ys_minor = std::conditional_t<TransposeC, sequence<1, 0>, sequence<0, 1>>;
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<MIterPerWarp>, sequence<NWarp, NIterPerWarp>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<0, 0>>,
                c_distr_ys_major,
                c_distr_ys_minor>{};
            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

            return c_block_dstr_encode;
        }
        else
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<MWarp, MIterPerWarp>, sequence<NWarp, NIterPerWarp>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<0, 0>>,
                c_distr_ys_major,
                sequence<1, 1>>{};
            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

            return c_block_dstr_encode;
        }
    }

    // C += A * B with MX scaling and packed-in-two (XdlPack) optimization
    // Scale tensors contain pre-packed int32_t: each int32_t holds MXdlPack * KXdlPack e8m0_t
    // values (for A) or NXdlPack * KXdlPack (for B), packed on the host.
    // Uses OpSel (0-3) to select which byte within the packed int32_t for each MFMA call.
    // XdlPack template parameters default to 2; fall back to 1 when iteration count is too small.
    template <typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename ScaleATensor,
              typename ScaleBTensor,
              index_t MXdlPack_ = 2,
              index_t NXdlPack_ = 2,
              index_t KXdlPack_ = 2>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor,
                                   const ScaleATensor& scale_a_tensor,
                                   const ScaleBTensor& scale_b_tensor) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockTensor::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "Datatypes do not match BlockTensor datatypes!");

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

        // Effective XdlPack: fall back to 1 when iteration count is insufficient
        constexpr index_t MXdlPack =
            (MIterPerWarp >= MXdlPack_ && MIterPerWarp % MXdlPack_ == 0) ? MXdlPack_ : 1;
        constexpr index_t NXdlPack =
            (NIterPerWarp >= NXdlPack_ && NIterPerWarp % NXdlPack_ == 0) ? NXdlPack_ : 1;
        constexpr index_t KXdlPack =
            (KIterPerWarp >= KXdlPack_ && KIterPerWarp % KXdlPack_ == 0) ? KXdlPack_ : 1;

        constexpr index_t MPackIterPerWarp = MIterPerWarp / MXdlPack;
        constexpr index_t NPackIterPerWarp = NIterPerWarp / NXdlPack;
        constexpr index_t KPackIterPerWarp = KIterPerWarp / KXdlPack;

        // hot loop with MX scaling and pre-packed int32_t scales:
        // Outer loops iterate over pack groups (scale tile indices)
        static_ford<sequence<KPackIterPerWarp, MPackIterPerWarp>>{}([&](auto ii) {
            constexpr auto ikpack = number<ii[number<0>{}]>{};
            constexpr auto impack = number<ii[number<1>{}]>{};
            // Get pre-packed int32_t A scale (already contains MXdlPack*KXdlPack e8m0_t)
            auto scale_a_slice = scale_a_tensor.get_y_sliced_thread_data(
                sequence<ikpack, impack, 0>{}, sequence<1, 1, 1>{});
            const int32_t a_scale_packed = bit_cast<int32_t>(scale_a_slice[number<0>{}]);

            static_for<0, NPackIterPerWarp, 1>{}([&](auto inpack) {
                // Get pre-packed int32_t B scale
                auto scale_b_slice = scale_b_tensor.get_y_sliced_thread_data(
                    sequence<ikpack, inpack, 0>{}, sequence<1, 1, 1>{});
                const int32_t b_scale_packed = bit_cast<int32_t>(scale_b_slice[number<0>{}]);

                // Inner loops: issue MFMAs within the pack group using OpSel
                static_ford<sequence<KXdlPack, MXdlPack>>{}([&](auto jj) {
                    constexpr auto ikxdl = number<jj[number<0>{}]>{};
                    constexpr auto imxdl = number<jj[number<1>{}]>{};
                    constexpr auto kIter = ikpack * KXdlPack + ikxdl;
                    constexpr auto mIter = impack * MXdlPack + imxdl;

                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    // OpSel for A: selects byte within packed int32_t
                    constexpr index_t kOpSelA = ikxdl * MXdlPack + imxdl;

                    static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                        constexpr auto nIter = inpack * NXdlPack + inxdl;

                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;
                        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        // OpSel for B: selects byte within packed int32_t
                        constexpr index_t kOpSelB = ikxdl * NXdlPack + inxdl;

                        // read C warp tensor from C block tensor
                        using c_iter_idx = std::conditional_t<TransposeC,
                                                              sequence<nIter, mIter>,
                                                              sequence<mIter, nIter>>;
                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM with MX scaling using pre-packed scale and OpSel
                        WarpGemm{}.template operator()<OpSelA<kOpSelA>, OpSelB<kOpSelB>>(
                            c_warp_tensor,
                            a_warp_tensor,
                            b_warp_tensor,
                            a_scale_packed,
                            b_scale_packed);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        });
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return make_static_distributed_tensor<CDataType>(
            make_static_tile_distribution(MakeCBlockDistributionEncode()));
    }
};

} // namespace ck_tile
