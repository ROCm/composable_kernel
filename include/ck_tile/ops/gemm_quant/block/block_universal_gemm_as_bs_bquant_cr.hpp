// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp"

namespace ck_tile {

// A is block window on shared memory
// BQ (scale tensor) is block distributed tensor.
// Consecutive QuantGroupSize elements of B are quantized with a separate scale.
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_,
          typename Policy_     = BlockGemmASmemBSmemCRegV1DefaultPolicy,
          index_t UnaryOpSize_ = 8>
struct BQuantBlockUniversalGemmAsBsCr
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem          = remove_cvref_t<PipelineProblem_>;
        using Policy           = remove_cvref_t<GemmPolicy_>;
        using ADataType        = remove_cvref_t<typename Problem::ADataType>;
        using BDataType        = remove_cvref_t<typename Problem::BDataType>;
        using BQDataType       = remove_cvref_t<typename Problem::BQDataType>;
        using BLayout          = remove_cvref_t<typename Problem::BLayout>;
        using BQLayout         = remove_cvref_t<typename Problem::BQLayout>;
        using AComputeDataType = remove_cvref_t<typename Problem::AComputeDataType>;
        using BComputeDataType = remove_cvref_t<typename Problem::BComputeDataType>;
        using CDataType        = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape   = remove_cvref_t<typename Problem::BlockGemmShape>;
        using BQuantGroupSize  = remove_cvref_t<typename Problem::BQuantGroupSize>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        // Threadblock GEMM tile size
        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr index_t NQPerBlock = NPerBlock / BQuantGroupSize::kN;
        static constexpr index_t KQPerBlock = KPerBlock / BQuantGroupSize::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        // number of warps along M and N for threadblock's GEMM problem size
        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();

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
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static constexpr bool BPreshuffleQuant = Problem::Traits::BPreshuffleQuant;

        static constexpr index_t QScalesPerBlockRow =
            integer_divide_ceil(KPerBlock, BQuantGroupSize::kK);
        static constexpr index_t QScalesPerWarpGemmRow =
            integer_divide_ceil(WarpGemm::kK, BQuantGroupSize::kK);

        static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;

        static_assert(BQuantGroupSize::kK % WarpGemm::kK == 0,
                      "Error! WarpGemm::kK should be a multiple of BQuantGroupSize");
        static_assert(QScalesPerWarpGemmRow == 1,
                      "Error! BQuantGroupSize shouldn't be smaller than WarpGemm::kK");
        static_assert(KIterPerWarp % QScalesPerBlockRow == 0,
                      "Error! KItersPerWarp should be a multiple of QscalesPerBlockRow");

        static_assert(KPerBlock / BQuantGroupSize::kK > 0,
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
        // 5. bf16, (bf16/bf8/fp8/fp4), e8m0 -> f32
        // 6. fp16, (fp16/fp8/bf8/fp4), e8m0 -> f32
        static_assert(
            is_any_of<ADataType, fp8_t, bf8_t, bf16_t, fp16_t>::value &&
            is_any_of<BDataType, fp8_t, bf8_t, pk_int4_t, bf16_t, pk_fp4_t, fp16_t>::value &&
            is_any_of<BQDataType, float, fp8_t, bf8_t, e8m0_t>::value &&
            is_any_of<AComputeDataType, fp8_t, bf8_t, bf16_t, fp16_t>::value &&
            is_any_of<BComputeDataType, fp8_t, bf8_t, bf16_t, fp16_t>::value &&
            std::is_same_v<CDataType, fp32_t>);

        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        static constexpr index_t KPackA     = WarpGemm::kKPerThread;
        static constexpr index_t KPackB     = WarpGemm::kKPerThread;
        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;

        template <typename T>
        using has_bcastpolicy_type = decltype(T::BCastPolicy);

        static constexpr bool IsBCastPolicyBeforeLDSWrite = [] {
            if constexpr(is_detected<has_bcastpolicy_type, Problem>{})
            {
                return Problem::BCastPolicy == CastPolicy::BeforeLDSWrite;
            }
            else
            {
                return false;
            }
        }();
    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType        = remove_cvref_t<typename Traits::ADataType>;
    using BDataType        = remove_cvref_t<typename Traits::BDataType>;
    using BQDataType       = remove_cvref_t<typename Traits::BQDataType>;
    using AComputeDataType = remove_cvref_t<typename Traits::AComputeDataType>;
    using BComputeDataType = remove_cvref_t<typename Traits::BComputeDataType>;
    using CDataType        = remove_cvref_t<typename Traits::CDataType>;

    // BDataType gets converted from PkInt4 during loading
    // OverrideBDataType is only used when BCastPolicy is CastBeforeLDSWrite for microscale.
    // In that case we use ADataType
    using OverrideBDataType = std::conditional_t<
        (std::is_same_v<BDataType, pk_int4_t> &&
         std::is_same_v<typename Traits::BLayout, tensor_layout::gemm::RowMajor>) ||
            Traits::IsBCastPolicyBeforeLDSWrite,
        ADataType,
        BDataType>;

    using Base     = BlockGemmQuantBase;
    using WarpGemm = remove_cvref_t<typename Traits::WarpGemm>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;

    static constexpr auto Scheduler = Traits::Scheduler;

    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

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

    // Use gemm universal block distribution encoding instead of duplicating it
    using BlockGemmBase = BlockUniversalGemmAsBsCr<Problem_, Policy_, UnaryOpSize_>;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        return BlockGemmBase::MakeABlockDistributionEncode();
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        return BlockGemmBase::MakeBBlockDistributionEncode();
    }

    private:
    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    using BlockGemmImplBase = typename BlockUniversalGemmAsBsCr<Problem_, Policy_, UnaryOpSize_>::
        template BlockGemmImpl<GemmPipelineScheduler::Intrawave, Traits>;

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits> : public BlockGemmImplBase
    {
        using BlockGemmImplBase::a_warp_tile_;
        using BlockGemmImplBase::b_warp_tile_;
        using BlockGemmImplBase::BLdsTileDistr;
        // If we apply scale while reading from LDS, then we can use the operator() from
        // BlockUniversalGemmAsBsCr
        using BlockGemmImplBase::operator();

        // static distributed tensor with LDS type
        using BTypeTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));
        BTypeTile b_warp_tile_lds_;

        // Load from LDS (assumption is that the scale will be applied in the block gemm)
        template <typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window,
                                          bool_constant<ALoadTranspose> = {},
                                          bool_constant<BLoadTranspose> = {})
        {
            load_and_convert_tile<UnaryOpSize_, ALoadTranspose>(a_warp_tile_, a_block_window);
            // If B datatype were pkint4 it would be converted prior to storing in LDS
            load_and_convert_tile<UnaryOpSize_, BLoadTranspose>(b_warp_tile_, b_block_window);
        }

        // Load from LDS and scale (then the tile can directly be consumed in the block gemm)
        template <typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  typename BQRegBlockTile,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window,
                                          const BQRegBlockTile& bq_block_tensor,
                                          bool_constant<ALoadTranspose> = {},
                                          bool_constant<BLoadTranspose> = {})
        {
            // Load tile from LDS

            // Do not use load_int4_tile here because it will have support to cast from fp4 to
            // compute type, while here we want to only load from LDS and then apply the scale
            // and cast later
            if constexpr(ALoadTranspose)
            {
                a_warp_tile_ = load_tile_transpose(a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_block_window);
            }

            if constexpr(BLoadTranspose)
            {
                b_warp_tile_lds_ = load_tile_transpose(b_block_window);
            }
            else
            {
                load_tile(b_warp_tile_lds_, b_block_window);
            }

            // Apply scale and cast
            using BDataTypeRaw =
                std::conditional_t<std::is_same_v<BDataType, pk_fp4_t>, pk_fp4_t::type, BDataType>;

            constexpr index_t warp_size          = get_warp_size();
            constexpr index_t nelements          = WarpGemm::kK * WarpGemm::kN / warp_size;
            constexpr index_t thread_buffer_size = nelements / UnaryOpSize_;
            const element_wise::DequantPack8 elementwise_op{};
            using SrcVectorRawType = ext_vector_t<BDataTypeRaw, UnaryOpSize_ / BPackedSize>;
            using DstVectorType    = ext_vector_t<BComputeDataType, UnaryOpSize_>;

            static_ford<sequence<NIterPerWarp, Traits::QScalesPerBlockRow>>{}([&](auto nk) {
                constexpr auto nIter   = number<nk[number<0>{}]>{};
                constexpr auto kQScale = number<nk[number<1>{}]>{};
                // B scale register offset
                constexpr index_t reg_offset = [&]() {
                    if constexpr(GemmTraits::BQuantGroupSize::kN >= (NWarp * WarpGemm::kN))
                        return ((nIter * NWarp * WarpGemm::kN) / GemmTraits::BQuantGroupSize::kN) *
                                   Traits::KQPerBlock +
                               kQScale;
                    else
                    {
                        return nIter * Traits::KQPerBlock + kQScale;
                    }
                }();

                // Get B scale from thread buffer
                auto& scale_reg = bq_block_tensor.get_thread_buffer()[reg_offset];
                float b_scale_f = float(scale_reg);

                static_for<0, Traits::KIterPerQScale, 1>{}([&](auto kIterInQScale) {
                    constexpr auto kIter = kQScale * Traits::KIterPerQScale + kIterInQScale;
                    // Thread buffers
                    using BWarpThreadBuffer = decltype(b_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths)));
                    using BLDSThreadBuffer  = decltype(b_warp_tile_lds_.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths)));

                    BWarpThreadBuffer b_warp_thread_buffer;
                    BLDSThreadBuffer b_lds_thread_buffer;

                    // Load thread buffer from tile (LDS type)
                    b_lds_thread_buffer = b_warp_tile_lds_.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    // Apply scale to B thread buffer and cast
                    static_for<0, thread_buffer_size, 1>{}([&](auto i) {
                        elementwise_op(b_warp_thread_buffer.template get_as<DstVectorType>()(i),
                                       b_lds_thread_buffer.template get_as<SrcVectorRawType>()[i],
                                       b_scale_f);
                    });

                    // Store B thread buffer to tile (MMA type)
                    b_warp_tile_.set_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths),
                        b_warp_thread_buffer);
                });
            });
        }

        // C += A * B
        template <typename CBlockTensor,
                  typename BQBlockTensor,
                  typename ASmemBlockWindow,
                  typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       BQBlockTensor& bq_block_tensor,
                                       [[maybe_unused]] ASmemBlockWindow& a_block_window,
                                       [[maybe_unused]] BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as corresponding "
                          "C block tensor data type!");
            constexpr auto warp_size = get_warp_size();

            // hot loop:
            static_ford<sequence<MIterPerWarp, NIterPerWarp>>{}([&](auto mn) {
                constexpr auto mIter = number<mn[number<0>{}]>{};
                constexpr auto nIter = number<mn[number<1>{}]>{};
                CWarpTensor c_warp_tensor;

                static_for<0, Traits::QScalesPerBlockRow, 1>{}([&](auto kQScale) {
                    static_for<0, Traits::KIterPerQScale, 1>{}([&](auto kIterInQScale) {
                        constexpr auto kIter = kQScale * Traits::KIterPerQScale + kIterInQScale;

                        AWarpTensor a_warp_tensor;
                        a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                        BWarpTensor b_warp_tensor;
                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        if constexpr(kIterInQScale == 0)
                        {
                            c_warp_tensor = WarpGemm{}(a_warp_tensor, b_warp_tensor);
                        }
                        else
                        {
                            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                        }
                    });

                    constexpr auto tbuf_offset =
                        number<typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
                                   merge_sequences(sequence<mIter, nIter>{},
                                                   c_warp_y_index_zeros)) /
                               CBlockTensor::PackedSize>{};

                    if constexpr(BPreshuffleQuant)
                    {
                        constexpr index_t reg_offset = [&]() {
                            if constexpr(GemmTraits::BQuantGroupSize::kN > (NWarp * WarpGemm::kN) &&
                                         Traits::NPerBlock == GemmTraits::BQuantGroupSize::kN)
                            {
                                return kQScale; // prefill: one quant group per block
                            }
                            else
                            {
                                return nIter; // decode or multiple groups per warp
                            }
                        }();

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

                        float scale_reg_f = Base::cvt_scale_to_fp32<typename Traits::BQDataType>(
                            gathered_scale_reg);

                        static_for<0, WarpGemm::kM * WarpGemm::kN / warp_size, 1>{}(
                            [&](auto c_row) {
                                c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
                                    (c_warp_tensor.get_thread_buffer()[c_row] * scale_reg_f);
                            });
                    }
                    else
                    {
                        // Multiply bquant with accumulated C
                        constexpr index_t reg_offset = [&]() {
                            if constexpr(GemmTraits::BQuantGroupSize::kN >= (NWarp * WarpGemm::kN))
                                return (nIter * NWarp * WarpGemm::kN) /
                                           GemmTraits::BQuantGroupSize::kN * Traits::KQPerBlock +
                                       kQScale;
                            else
                            {
                                return nIter * Traits::KQPerBlock + kQScale;
                            }
                        }();

                        auto& scale_reg = bq_block_tensor.get_thread_buffer()[reg_offset];
                        float scale_reg_f =
                            Base::cvt_scale_to_fp32<typename Traits::BQDataType>(scale_reg);
                        static_for<0, WarpGemm::kM * WarpGemm::kN / warp_size, 1>{}(
                            [&](auto c_row) {
                                c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
                                    (c_warp_tensor.get_thread_buffer()[c_row] * scale_reg_f);
                            });
                    }
                });
            });
        }
    };

    public:
    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return BlockGemmQuantCommon<CDataType, WarpGemm, MIterPerWarp, MWarp, NIterPerWarp, NWarp>::
            MakeCBlockTile();
    }

    // Read A and B from LDS
    template <typename ASmemBlockWindow,
              typename BSmemBlockWindow,
              bool ALoadTranspose = false,
              bool BLoadTranspose = false>
    CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                      const BSmemBlockWindow& b_block_window,
                                      bool_constant<ALoadTranspose> a_load_tr = {},
                                      bool_constant<BLoadTranspose> b_load_tr = {})
    {
        block_gemm_impl_.LocalPrefetch(a_block_window, b_block_window, a_load_tr, b_load_tr);
    }

    // Read A and B from LDS and apply scale to B
    template <typename ASmemBlockWindow,
              typename BSmemBlockWindow,
              typename BQRegBlockTile,
              bool ALoadTranspose = false,
              bool BLoadTranspose = false>
    CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                      const BSmemBlockWindow& b_block_window,
                                      BQRegBlockTile bq_block_tile,
                                      bool_constant<ALoadTranspose> a_load_tr = {},
                                      bool_constant<BLoadTranspose> b_load_tr = {})
    {
        block_gemm_impl_.LocalPrefetch(
            a_block_window, b_block_window, bq_block_tile, a_load_tr, b_load_tr);
    }

    // C += A * B
    // Apply scale after MMA
    template <typename CBlockTensor,
              typename BQBlockTensor,
              typename ASmemBlockWindow,
              typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   BQBlockTensor& bq_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_(c_block_tensor, bq_block_tensor, a_block_window, b_block_window);
    }

    // C += A * B
    // Scale has already been applied to B, so this is using the gemm universal block implementation
    template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window);
    }

    private:
    BlockGemmImpl<Scheduler, Traits> block_gemm_impl_{};
};

} // namespace ck_tile
