// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_eight_waves_v1.hpp"

namespace ck_tile {

// A is block window on shared memory
// AQ (scale tensor) is block distributed tensor.
// BQ (scale tensor) is block distributed tensor.
// Consecutive QuantGroupSize elements of A and B are quantized with a separate scale.
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_,
          typename Policy_     = BlockGemmASmemBSmemCRegV1DefaultPolicy,
          index_t UnaryOpSize_ = 8>
struct ABQuantBlockUniversalGemmAsBsCrAsync : public BlockGemmQuantBase
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem         = remove_cvref_t<PipelineProblem_>;
        using Policy          = remove_cvref_t<GemmPolicy_>;
        using ADataType       = remove_cvref_t<typename Problem::ADataType>;
        using AQDataType      = remove_cvref_t<typename Problem::AQDataType>;
        using BDataType       = remove_cvref_t<typename Problem::BDataType>;
        using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
        using BQLayout        = remove_cvref_t<typename Problem::BQLayout>;
        using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
        using CDataType       = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
        using AQuantGroupSize = remove_cvref_t<typename Problem::AQuantGroupSize>;
        using BQuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        // Threadblock GEMM tile size
        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr index_t NQPerBlock = NPerBlock / BQuantGroupSize::kN;
        static constexpr index_t KQPerBlock = KPerBlock / BQuantGroupSize::kK;
        static constexpr index_t AQPerBlock = KPerBlock / AQuantGroupSize::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        // number of warps along M and N for threadblock's GEMM problem size
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

        static constexpr bool APreshuffleQuant = Problem::Traits::APreshuffleQuant;
        static constexpr bool BPreshuffleQuant = Problem::Traits::BPreshuffleQuant;

        static constexpr index_t QScalesPerBlockRow =
            integer_divide_ceil(KPerBlock / KWarp, BQuantGroupSize::kK);
        static constexpr index_t QScalesPerWarpGemmRow =
            integer_divide_ceil(WarpGemm::kK, BQuantGroupSize::kK);

        static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;

        static_assert(BQuantGroupSize::kK % WarpGemm::kK == 0,
                      "Error! WarpGemm::kK should be a multiple of QuantGroupSize");
        static_assert(QScalesPerWarpGemmRow == 1,
                      "Error! QuantGroupSize shouldn't be smaller than WarpGemm::kK");
        static_assert(KIterPerWarp % QScalesPerBlockRow == 0,
                      "Error! KItersPerWarp should be a multiple of QscalesPerBlockRow");

        static_assert(KPerBlock / KWarp / BQuantGroupSize::kK > 0,
                      "Error! Each row of blockgemm should have a separate scale");

        static_assert(MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock,
                      "Error! Warps should cover all Block tile!");
        static_assert(NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock,
                      "Error! Warps should cover all Block tile!");

        // Currently tested combinations (A, B, BQ)
        // 1. fp8, fp8, fp32 -> f32
        // 2. bf8, bf8, fp32 -> f32
        // 3. i4,  fp8, (fp8/fp32) -> f32
        // 4. i4,  bf8, (fp8/fp32) -> f32
        static_assert(
            (std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, bf8_t> ||
             std::is_same_v<ADataType, ck_tile::pk_int4_t>) &&
            (std::is_same_v<BDataType, fp8_t> || std::is_same_v<BDataType, bf8_t> ||
             std::is_same_v<BDataType, ck_tile::pk_int4_t>) &&
            (std::is_same_v<AQDataType, float> || std::is_same_v<AQDataType, ck_tile::fp8_t> ||
             std::is_same_v<AQDataType, ck_tile::bf8_t>) &&
            (std::is_same_v<BQDataType, float> || std::is_same_v<BQDataType, ck_tile::fp8_t> ||
             std::is_same_v<BQDataType, ck_tile::bf8_t>) &&
            (std::is_same_v<ComputeDataType, fp8_t> || std::is_same_v<ComputeDataType, bf8_t>) &&
            std::is_same_v<CDataType, fp32_t>);

        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        static constexpr index_t KPack      = WarpGemm::kKPerThread;
        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;
        static constexpr bool TransposeC    = Problem::TransposeC;
    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType       = remove_cvref_t<typename Traits::ADataType>;
    using AQDataType      = remove_cvref_t<typename Traits::AQDataType>;
    using BDataType       = remove_cvref_t<typename Traits::BDataType>;
    using BQDataType      = remove_cvref_t<typename Traits::BQDataType>;
    using ComputeDataType = remove_cvref_t<typename Traits::ComputeDataType>;
    using CDataType       = remove_cvref_t<typename Traits::CDataType>;

    // BDataType gets converted from PkInt4 during loading
    using OverrideBDataType =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
    using Base     = BlockGemmQuantBase;
    using WarpGemm = remove_cvref_t<typename Traits::WarpGemm>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;
    static constexpr index_t KWarp = Traits::KWarp;

    static constexpr auto Scheduler = Traits::Scheduler;

    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    static constexpr bool APreshuffleQuant = Traits::APreshuffleQuant;
    static constexpr bool BPreshuffleQuant = Traits::BPreshuffleQuant;

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

    using BlockGemmBase = BlockGemmARegBRegCRegEightWavesV1<Problem_, Policy_>;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        return BlockGemmBase::MakeABlockDistributionEncode();
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        return BlockGemmBase::MakeBBlockDistributionEncode();
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockDistributionEncode()
    {
        return BlockGemmBase::MakeCBlockDistributionEncode();
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return make_static_distributed_tensor<CDataType>(
            make_static_tile_distribution(MakeCBlockDistributionEncode()));
    }

    using ALdsTile  = typename BlockGemmBase::ALdsTile;
    using BLdsTiles = typename BlockGemmBase::BLdsTiles;

    private:
    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits>
    {

        template <typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& /*a_block_window*/,
                                          const BSmemBlockWindow& /*b_block_window*/,
                                          bool_constant<ALoadTranspose> = {},
                                          bool_constant<BLoadTranspose> = {})
        {
            static_assert(false, "Not implemented yet!");
        }

        // C += A * B
        template <typename CBlockTensor, typename AQBlockTensor, typename BQBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ALdsTile& a_warp_tile_,
                                       const BLdsTiles& b_warp_tiles_,
                                       AQBlockTensor& aq_block_tensor,
                                       BQBlockTensor& bq_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as corresponding "
                          "C block tensor data type!");
            constexpr auto warp_size = get_warp_size();

            auto q_block_tensor = aq_block_tensor;
            if constexpr(Traits::NQPerBlock / NWarp == 1)
            {
                constexpr auto aq_spans = AQBlockTensor::get_distributed_spans();
                sweep_tile_span(aq_spans[I0{}], [&](auto im) {
                    sweep_tile_span(aq_spans[I1{}], [&](auto ik) {
                        q_block_tensor(make_tuple(im, ik)) *=
                            bq_block_tensor(make_tuple(tile_distributed_index<0>{}, ik));
                    });
                });
            }

            // hot loop:
            static_for<0, Traits::QScalesPerBlockRow, 1>{}([&](auto kQScale) {
                static_for_product<number<NIterPerWarp>, number<MIterPerWarp>>{}([&](auto nIter,
                                                                                     auto mIter) {
                    CWarpTensor c_warp_tensor;
                    static_for<0, Traits::KIterPerQScale, 1>{}([&](auto kIterInQScale) {
                        static_assert(Traits::KIterPerQScale == 1);
                        constexpr auto kIter =
                            number<kQScale * Traits::KIterPerQScale + kIterInQScale>{};

                        AWarpTensor a_warp_tensor;
                        a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                        BWarpTensor b_warp_tensor;
                        b_warp_tensor.get_thread_buffer() =
                            b_warp_tiles_[nIter][kIter].get_thread_buffer();
                        if constexpr(kIterInQScale == 0)
                        {
                            c_warp_tensor = WarpGemm{}(a_warp_tensor, b_warp_tensor);
                        }
                        else
                        {
                            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                        }
                    });

                    if constexpr(Traits::NQPerBlock / NWarp == 1)
                    {
                        constexpr auto cw_spans = CWarpTensor::get_distributed_spans();
                        static_assert(cw_spans[I0{}].impl_.size() == 0);
                        sweep_tile_span(cw_spans[I1{}], [&](auto in) {
                            constexpr auto block_idx_m = tile_distributed_index<mIter>{};
                            constexpr auto block_idx_n = detail::make_tile_distributed_index(
                                merge_sequences(sequence<nIter>{}, in.impl_));
                            constexpr auto block_idx_kq = tile_distributed_index<kQScale>{};
                            constexpr auto empty_idx    = tile_distributed_index<>{};
                            c_block_tensor(make_tuple(block_idx_m, block_idx_n)) +=
                                c_warp_tensor(make_tuple(empty_idx, in)) *
                                q_block_tensor(make_tuple(block_idx_m, block_idx_kq));
                        });
                    }
                    else
                    {

                        constexpr auto tbuf_offset =
                            number<typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
                                       merge_sequences(sequence<mIter, nIter>{},
                                                       c_warp_y_index_zeros)) /
                                   CBlockTensor::PackedSize>{};
                        // a_scale
                        AQPickerCommon<AQBlockTensor, Traits, mIter, kQScale> aq_picker(
                            aq_block_tensor);

                        if constexpr(BPreshuffleQuant)
                        {
                            constexpr index_t reg_offset = nIter;
                            auto pull_from_lane =
                                (__lane_id() & (WarpGemm::kN - 1)) * Traits::KQPerBlock + kQScale;
                            auto& scale_reg = bq_block_tensor.get_thread_buffer()[reg_offset];
                            // cross lane ops
                            uint32_t scale_reg_dword;

                            if constexpr(std::is_same_v<BQDataType, float>)
                            {
                                scale_reg_dword = ck_tile::bit_cast<uint32_t>(scale_reg);
                            }
                            else
                            {
                                scale_reg_dword = static_cast<uint32_t>(scale_reg);
                            }

                            // cross lane ops to get the value of scale_reg.
                            int gathered_scale_reg = __builtin_amdgcn_ds_bpermute(
                                pull_from_lane << 2, __builtin_bit_cast(int, scale_reg_dword));

                            float b_scale_reg_f =
                                Base::cvt_scale_to_fp32<typename Traits::BQDataType>(
                                    gathered_scale_reg);

                            static_for<0, WarpGemm::kM * WarpGemm::kN / warp_size, 1>{}(
                                [&](auto c_row) {
                                    float a_scale_reg_f = aq_picker.template pick<c_row>();
                                    c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
                                        (c_warp_tensor.get_thread_buffer()[c_row] * a_scale_reg_f *
                                         b_scale_reg_f);
                                });
                        }
                        else
                        {
                            // Multiply bquant with accumulated C
                            constexpr index_t reg_offset = [&]() {
                                if constexpr(GemmTraits::BQuantGroupSize::kN >=
                                             (NWarp * WarpGemm::kN))
                                    return (nIter * NWarp * WarpGemm::kN) /
                                               GemmTraits::BQuantGroupSize::kN *
                                               Traits::KQPerBlock +
                                           kQScale;
                                else
                                {
                                    return nIter * Traits::KQPerBlock + kQScale;
                                }
                            }();

                            auto& scale_reg = bq_block_tensor.get_thread_buffer()[reg_offset];
                            float b_scale_reg_f =
                                Base::cvt_scale_to_fp32<typename Traits::BQDataType>(scale_reg);

                            static_for<0, WarpGemm::kM * WarpGemm::kN / warp_size, 1>{}(
                                [&](auto c_row) {
                                    float a_scale_reg_f = aq_picker.template pick<c_row>();
                                    c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
                                        (c_warp_tensor.get_thread_buffer()[c_row] * a_scale_reg_f *
                                         b_scale_reg_f);
                                });
                        }
                    }
                });
            });
        }
    };

    public:
    template <typename... Args>
    CK_TILE_DEVICE void LocalPrefetch(Args&&... args)
    {
        block_gemm_impl_.LocalPrefetch(std::forward<Args>(args)...);
    }

    // C += A * B
    template <typename CBlockTensor, typename... Rest>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor, Rest&&... rest)
    {
        block_gemm_impl_(c_block_tensor, std::forward<Rest>(rest)...);
    }

    private:
    BlockGemmImpl<Scheduler, Traits> block_gemm_impl_{};
};

} // namespace ck_tile
