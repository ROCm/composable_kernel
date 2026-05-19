// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_default_policy.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmARegBRegCRegV1DefaultPolicy>
struct BlockMXGemmARegBRegCRegEightWavesV1
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem          = remove_cvref_t<PipelineProblem_>;
        using Policy           = remove_cvref_t<GemmPolicy_>;
        using ADataType        = remove_cvref_t<typename Problem::ADataType>;
        using BDataType        = remove_cvref_t<typename Problem::BDataType>;
        using CDataType        = remove_cvref_t<typename Problem::CDataType>;
        using AComputeDataType = remove_cvref_t<typename Problem::AComputeDataType>;
        using BComputeDataType = remove_cvref_t<typename Problem::BComputeDataType>;
        using BlockGemmShape   = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();
        static constexpr index_t KWarp = Problem::BlockGemmShape::BlockWarps::at(number<2>{});

        using I0 = number<0>;
        using I1 = number<1>;

        static_assert(MWarp == BlockGemmShape::BlockWarps::at(I0{}),
                      "Error! WarpGemm's MWarp is not consistent with BlockGemmShape!");
        static_assert(NWarp == BlockGemmShape::BlockWarps::at(I1{}),
                      "Error! WarpGemm's NWarp is not consistent with BlockGemmShape!");
        static_assert(WarpGemm::kM == BlockGemmShape::WarpTile::at(I0{}),
                      "Error! WarpGemm's M is not consistent with BlockGemmShape!");
        static_assert(WarpGemm::kN == BlockGemmShape::WarpTile::at(I1{}),
                      "Error! WarpGemm's N is not consistent with BlockGemmShape!");

        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / (KWarp * WarpGemm::kK);

        // Controls how many MAC clusters (MFMA blocks) we have per wave
        // If InterWaveSchedulingMacClusters = 1;
        // Then we group all WarpGemms into single MAC cluster.
        // But if InterWaveSchedulingMacClusters = 2, then we
        // split the warp gemms into two groups.
        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        static constexpr index_t KPackA     = WarpGemm::kAKPack;
        static constexpr index_t KPackB     = WarpGemm::kBKPack;
        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;
        static constexpr bool TransposeC    = Problem::TransposeC;
    };

    public:
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;
    using Traits  = GemmTraits_<Problem, Policy>;

    using WarpGemm       = typename Traits::WarpGemm;
    using BlockGemmShape = typename Traits::BlockGemmShape;

    using ADataType        = remove_cvref_t<typename Traits::ADataType>;
    using BDataType        = remove_cvref_t<typename Traits::BDataType>;
    using CDataType        = remove_cvref_t<typename Traits::CDataType>;
    using AComputeDataType = remove_cvref_t<typename Traits::AComputeDataType>;
    using BComputeDataType = remove_cvref_t<typename Traits::BComputeDataType>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;
    static constexpr index_t KWarp = Traits::KWarp;

    static constexpr auto Scheduler  = Traits::Scheduler;
    static constexpr bool TransposeC = Traits::TransposeC;

    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    static_assert(std::is_same_v<typename WarpGemm::CDataType, float>);

    static constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    static constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    static constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using I0 = number<0>;
    using I1 = number<1>;

    // Note: distribution encodings have MIterPerWarp and NIterPerWarp contiguous because of scale
    // packing.

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;

        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);

        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KWarp, KIterInterwave>,
                                            sequence<KWarp, KIterPerWarp>>;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<2, NWarp / 2>,
                                       tuple<sequence<MWarp, MIterPerWarp>, KIterSeq>,
                                       tuple<sequence<0, 2, 1, 0>>,
                                       tuple<sequence<0, 0, 0, 1>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return a_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;
        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KWarp, KIterInterwave>,
                                            sequence<KWarp, KIterPerWarp>>;

        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<2, NIterPerWarp, NWarp / 2>, KIterSeq>,
                                       tuple<sequence<2, 1, 0, 1>>,
                                       tuple<sequence<0, 0, 0, 2>>,
                                       sequence<>,
                                       sequence<>>{};

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        return b_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockDistributionEncode()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<KWarp>,
            tuple<sequence<MWarp, MIterPerWarp>, sequence<2, NIterPerWarp, NWarp / 2>>,
            tuple<sequence<2, 0, 1, 2>>,
            tuple<sequence<0, 0, 0, 2>>,
            sequence<1, 2>,
            sequence<1, 1>>{};
        constexpr auto c_block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        return c_block_dstr_encoding;
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return make_static_distributed_tensor<CDataType>(
            make_static_tile_distribution(MakeCBlockDistributionEncode()));
    }

    using ALdsTile  = decltype(make_static_distributed_tensor<AComputeDataType>(
        make_static_tile_distribution(MakeABlockDistributionEncode())));
    using BLdsTiles = statically_indexed_array<
        statically_indexed_array<decltype(make_static_distributed_tensor<BComputeDataType>(
                                     make_static_tile_distribution(
                                         MakeBBlockDistributionEncode()))),
                                 KIterPerWarp>,
        NIterPerWarp>;

    // C += A * B
    template <typename CBlockTensor,
              typename ScaleATensor,
              typename ScaleBTensor,
              index_t MXdlPack_ = 2,
              index_t NXdlPack_ = 2,
              index_t KXdlPack_ = 2>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ALdsTile& a_warp_tile_,
                                   const BLdsTiles& b_warp_tiles_,
                                   const ScaleATensor& scale_a_tensor,
                                   const ScaleBTensor& scale_b_tensor) const
    {
        // checks
        static_assert(std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "CDataType must be same as CBlockTensor::DataType!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeCBlockDistributionEncode())>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "C distribution is wrong!");

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

        // hot loop:
        static_for_product<number<KPackIterPerWarp>,
                           number<NPackIterPerWarp>,
                           number<MPackIterPerWarp>>{}([&](auto ikpack, auto inpack, auto impack) {
            // get A scale for this M-K tile using get_y_sliced_thread_data
            auto scale_a_slice = scale_a_tensor.get_y_sliced_thread_data(
                sequence<ikpack, impack, 0>{}, sequence<1, 1, 1>{});
            const int32_t a_scale_packed = bit_cast<int32_t>(scale_a_slice[number<0>{}]);

            // get B scale for this N-K tile using get_y_sliced_thread_data
            auto scale_b_slice = scale_b_tensor.get_y_sliced_thread_data(
                sequence<ikpack, inpack, 0>{}, sequence<1, 1, 1>{});
            const int32_t b_scale_packed = bit_cast<int32_t>(scale_b_slice[number<0>{}]);

            // Inner loops: issue MFMAs within the pack group using OpSel
            static_for_product<number<KXdlPack>, number<NXdlPack>, number<MXdlPack>>{}(
                [&](auto ikxdl, auto inxdl, auto imxdl) {
                    constexpr auto kIter = ikpack * KXdlPack + ikxdl;
                    constexpr auto mIter = impack * MXdlPack + imxdl;
                    constexpr auto nIter = inpack * NXdlPack + inxdl;

                    // OpSel for A: selects byte within packed int32_t
                    constexpr index_t kOpSelA = ikxdl * MXdlPack + imxdl;

                    // OpSel for B: selects byte within packed int32_t
                    constexpr index_t kOpSelB = ikxdl * NXdlPack + inxdl;

                    // read A warp tensor from A Block window
                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    // read B warp tensor from B block tensor
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() =
                        b_warp_tiles_[number<nIter>{}][number<kIter>{}].get_thread_buffer();

                    // read C warp tensor from C block tensor
                    using c_iter_idx = sequence<mIter, nIter>;
                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM with MX scaling
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
    }
};

} // namespace ck_tile
