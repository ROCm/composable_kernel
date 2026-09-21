// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/sageattention/block/block_sageattention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// P[seqlen_q, seqlen_k] = Softmax(S'[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

template <typename SageAttnPipeline_, typename EpiloguePipeline_>
struct SageAttnFwdKernel
{
    using SageAttnPipeline                       = ck_tile::remove_cvref_t<SageAttnPipeline_>;
    using EpiloguePipeline                       = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize = SageAttnPipeline::kBlockSize;

    static constexpr ck_tile::index_t kBlockPerCu = SageAttnPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = SageAttnPipeline::Problem::kBlockPerCu;

    using QDataType    = ck_tile::remove_cvref_t<typename SageAttnPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename SageAttnPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename SageAttnPipeline::VDataType>;
    using PDataType    = ck_tile::remove_cvref_t<typename SageAttnPipeline::PDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename SageAttnPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename SageAttnPipeline::SaccDataType>;

    using VLayout = ck_tile::remove_cvref_t<typename SageAttnPipeline::VLayout>;

    static constexpr bool kIsGroupMode = SageAttnPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = SageAttnPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = SageAttnPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = SageAttnPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = SageAttnPipeline::kPadHeadDimV;
    // logits_soft_cap is always disabled
    static constexpr auto QScaleEnum      = SageAttnPipeline::Problem::QScaleEnum;
    static constexpr bool kSkipMinSeqlenQ = SageAttnPipeline::Problem::kSkipMinSeqlenQ;

    using AttentionVariant = ck_tile::remove_cvref_t<typename SageAttnPipeline::AttentionVariant>;
    using AttnMask         = ck_tile::remove_cvref_t<typename SageAttnPipeline::AttnMask>;
    static constexpr bool kHasMask = AttnMask::IsMasking;

    static constexpr bool kUseAsyncCopy = SageAttnPipeline::Policy::AsyncCopy;

    // Distinct empty bases (I = 0 mask slot, 1 qscale slot, 2 min_seqlen_q slot) avoid duplicate
    // base-class issues under multiple inheritance.
    template <ck_tile::index_t I>
    struct SageAttnFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct SageAttnFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;
        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;
    };

    struct SageAttnFwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct SageAttnFwdCommonQScaleKargs
    {
        const void* q_descale_ptr = nullptr;
        const void* k_descale_ptr = nullptr;
        const void* v_descale_ptr = nullptr;
    };

    struct SageAttnFwdCommonBlockScaleKargs : public SageAttnFwdCommonQScaleKargs
    {
        ck_tile::index_t nhead_stride_q_descale;
        ck_tile::index_t nhead_stride_k_descale;
        ck_tile::index_t nhead_stride_v_descale;

        /// Host must match TileSageAttnTraits / Problem; validated in MakeKargs (device uses
        /// Problem::kBlockScaleSizeQ/K).
        ck_tile::index_t block_scale_size_q = 0;
        ck_tile::index_t block_scale_size_k = 0;
    };

    struct SageAttnFwdBatchBlockScaleKargs : public SageAttnFwdCommonBlockScaleKargs
    {
        ck_tile::index_t batch_stride_q_descale;
        ck_tile::index_t batch_stride_k_descale;
        ck_tile::index_t batch_stride_v_descale;
    };

    struct SageAttnFwdGroupBlockScaleKargs : public SageAttnFwdCommonBlockScaleKargs
    {
        const int32_t* block_scale_seqstart_q_ptr = nullptr;
        const int32_t* block_scale_seqstart_k_ptr = nullptr;
        ck_tile::index_t batch_stride_v_descale;
    };

    struct SageAttnFwdSkipMinSeqlenQKargs
    {
        ck_tile::index_t min_seqlen_q = 0;
    };

    struct SageAttnFwdBatchModeKargs
        : SageAttnFwdCommonKargs,
          std::conditional_t<kHasMask, SageAttnFwdMaskKargs, SageAttnFwdEmptyKargs<0>>,
          std::conditional_t<
              QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR,
              SageAttnFwdCommonQScaleKargs,
              std::conditional_t<QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD,
                                 SageAttnFwdBatchBlockScaleKargs,
                                 SageAttnFwdEmptyKargs<1>>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;

        // Optional cumulative sequence length pointers for batch mode
        // If provided, they override seqlen_q / seqlen_k per-batch to skip tail padding.
        const int32_t* cu_seqlen_q_ptr = nullptr; // cumulative, length without PAD
        const int32_t* cu_seqlen_k_ptr = nullptr; // cumulative, length without PAD
    };

    struct SageAttnFwdGroupModeKargs
        : SageAttnFwdCommonKargs,
          std::conditional_t<kHasMask, SageAttnFwdMaskKargs, SageAttnFwdEmptyKargs<0>>,
          std::conditional_t<
              QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR,
              SageAttnFwdCommonQScaleKargs,
              std::conditional_t<QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD,
                                 SageAttnFwdGroupBlockScaleKargs,
                                 SageAttnFwdEmptyKargs<1>>>,
          std::conditional_t<kSkipMinSeqlenQ,
                             SageAttnFwdSkipMinSeqlenQKargs,
                             SageAttnFwdEmptyKargs<2>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;
        const int32_t* seqlen_k_ptr;

        // Optional per-sequence and cumulative logical (excluding padding) sequence length arrays
        const int32_t* cu_seqlen_q_ptr = nullptr;
        const int32_t* cu_seqlen_k_ptr = nullptr;
    };

    using Kargs =
        std::conditional_t<kIsGroupMode, SageAttnFwdGroupModeKargs, SageAttnFwdBatchModeKargs>;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_q_descale,
              ck_tile::index_t nhead_stride_k_descale,
              ck_tile::index_t nhead_stride_v_descale,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t batch_stride_q_descale,
              ck_tile::index_t batch_stride_k_descale,
              ck_tile::index_t batch_stride_v_descale,
              ck_tile::index_t block_scale_size_q,
              ck_tile::index_t block_scale_size_k,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              const void* cu_seqlen_q_ptr = nullptr,
              const void* cu_seqlen_k_ptr = nullptr)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // mask or SageAttnFwdEmptyKargs<0>
                    {},               // qscale or SageAttnFwdEmptyKargs<1>
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
        {
            using PipelineProblem = typename SageAttnPipeline::Problem;

            if(block_scale_size_q != PipelineProblem::kBlockScaleSizeQ ||
               block_scale_size_k != PipelineProblem::kBlockScaleSizeK)
            {
                throw std::runtime_error(
                    "sageattn_fwd MakeKargs: block_scale_size_q/k must match kernel "
                    "TileSageAttnTraits (Problem::kBlockScaleSizeQ/K)");
            }

            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.batch_stride_q_descale = batch_stride_q_descale;
            kargs.batch_stride_k_descale = batch_stride_k_descale;
            kargs.batch_stride_v_descale = batch_stride_v_descale;

            kargs.block_scale_size_q = block_scale_size_q;
            kargs.block_scale_size_k = block_scale_size_k;
        }
        // logits_soft_cap is always disabled

        kargs.cu_seqlen_q_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr);
        kargs.cu_seqlen_k_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr);
        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_q_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_q_descale,
              ck_tile::index_t nhead_stride_k_descale,
              ck_tile::index_t nhead_stride_v_descale,
              ck_tile::index_t batch_stride_v_descale,
              ck_tile::index_t block_scale_size_q,
              ck_tile::index_t block_scale_size_k,
              const void* block_scale_seqstart_q_ptr,
              const void* block_scale_seqstart_k_ptr,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              ck_tile::index_t min_seqlen_q,
              const void* cu_seqlen_q_ptr = nullptr,
              const void* cu_seqlen_k_ptr = nullptr)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // mask or SageAttnFwdEmptyKargs<0>
                    {},               // qscale or SageAttnFwdEmptyKargs<1>
                    {},               // min_seqlen_q or SageAttnFwdEmptyKargs<2>
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                     QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
        {
            using PipelineProblem = typename SageAttnPipeline::Problem;

            if(block_scale_size_q != PipelineProblem::kBlockScaleSizeQ ||
               block_scale_size_k != PipelineProblem::kBlockScaleSizeK)
            {
                throw std::runtime_error(
                    "sageattn_fwd MakeKargs: block_scale_size_q/k must match kernel "
                    "TileSageAttnTraits (Problem::kBlockScaleSizeQ/K)");
            }

            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.batch_stride_v_descale = batch_stride_v_descale;

            kargs.block_scale_size_q = block_scale_size_q;
            kargs.block_scale_size_k = block_scale_size_k;

            kargs.block_scale_seqstart_q_ptr =
                reinterpret_cast<const int32_t*>(block_scale_seqstart_q_ptr);
            kargs.block_scale_seqstart_k_ptr =
                reinterpret_cast<const int32_t*>(block_scale_seqstart_k_ptr);
        }
        // logits_soft_cap is always disabled
        if constexpr(kSkipMinSeqlenQ)
        {
            kargs.min_seqlen_q = min_seqlen_q;
        }

        kargs.cu_seqlen_q_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr);
        kargs.cu_seqlen_k_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr);
        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_,
                                                bool has_padded_seqlen_k = false)
    {
        // has_padded_seqlen_k is determined by checking (seqlen_k_ptr != nullptr)
        if(has_padded_seqlen_k)
        {
            // TODO: this may need tuning
            return dim3(nhead_,
                        batch_size_,
                        ck_tile::integer_divide_ceil(seqlen_q_, SageAttnPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, SageAttnPipeline::kN1));
        }
        else
        {
            // TODO: this may need tuning
            return dim3(nhead_,
                        ck_tile::integer_divide_ceil(seqlen_q_, SageAttnPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, SageAttnPipeline::kN1),
                        batch_size_);
        }
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        bool has_padded_seqlen_k = false;

        if constexpr(kIsGroupMode)
            has_padded_seqlen_k = (kargs.seqlen_k_ptr != nullptr);

        if(has_padded_seqlen_k)
        {
            // const index_t num_tile_m0 = seqlen_q / kM0;
            const index_t num_tile_n1 =
                ck_tile::integer_divide_ceil(kargs.hdim_v, SageAttnPipeline::kN1);

            const index_t i_block = blockIdx.z;
            const index_t i_nhead = blockIdx.x;
            const index_t i_batch = blockIdx.y;

            const auto f = [](index_t dividend, index_t divisor) {
                index_t quotient = dividend / divisor;
                index_t modulus  = dividend - quotient * divisor;
                return ck_tile::make_tuple(quotient, modulus);
            };

            const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

            if constexpr(kHasMask)
            {
                // assume that num_tile_n1 is always 1
                return ck_tile::make_tuple(gridDim.z - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
            }
        }
        else
        {
            // const index_t num_tile_m0 = seqlen_q / kM0;
            const index_t num_tile_n1 =
                ck_tile::integer_divide_ceil(kargs.hdim_v, SageAttnPipeline::kN1);

            const index_t i_block = blockIdx.y; // blockIdx.x
            const index_t i_nhead = blockIdx.x; // blockIdx.y
            const index_t i_batch = blockIdx.z;

            const auto f = [](index_t dividend, index_t divisor) {
                index_t quotient = dividend / divisor;
                index_t modulus  = dividend - quotient * divisor;
                return ck_tile::make_tuple(quotient, modulus);
            };

            const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

            if constexpr(kHasMask)
            {
                // assume that num_tile_n1 is always 1
                return ck_tile::make_tuple(gridDim.y - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
            }
        }
    }

    CK_TILE_HOST static dim3 BlockSize()
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(SageAttnPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];
        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);
        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * SageAttnPipeline::kM0);
        const index_t i_n1 = amd_wave_read_first_lane(i_tile_n * SageAttnPipeline::kN1);

        long_index_t batch_offset_q         = 0;
        long_index_t batch_offset_k         = 0;
        long_index_t batch_offset_v         = 0;
        long_index_t batch_offset_o         = 0;
        long_index_t batch_offset_q_descale = 0;
        long_index_t batch_offset_k_descale = 0;
        long_index_t batch_offset_v_descale = 0;

        if constexpr(kIsGroupMode)
        {
            // Use seqstart_q_ptr and seqstart_k_ptr for physical starts
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            // DRAM base offsets use physical starts
            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                         QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                         QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
            {
                // BLOCKSCALE, PERWARP, and PERTHREAD all use block_scale_seqstart in group mode
                // They differ only in block size: BLOCKSCALE (Q:128, K:128), PERWARP (Q:32, K:64),
                // PERTHREAD (Q:4, K:16); see TileSageAttnTraits::kBlockScaleSizeQ/K.
                const long_index_t bquery_start = kargs.block_scale_seqstart_q_ptr[i_batch];
                const long_index_t bkey_start   = kargs.block_scale_seqstart_k_ptr[i_batch];
                batch_offset_q_descale          = bquery_start;
                batch_offset_k_descale          = bkey_start;
                // BLOCKSCALE, PERWARP, and PERTHREAD V all use per-channel scale: batch_stride =
                // nhead_k * hdim_v
                batch_offset_v_descale =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_v_descale;
            }
            batch_offset_o = query_start * kargs.stride_o;

            // real logical lengths (exclude PAD)
            // Priority: seqlen_q_ptr > cu_seqlen_q_ptr > calculated from seqstart_q_ptr
            if(kargs.seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q = kargs.seqlen_q_ptr[i_batch];
            }
            else if(kargs.cu_seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
                kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
            }

            if constexpr(kSkipMinSeqlenQ)
            {
                if(kargs.seqlen_q <= kargs.min_seqlen_q)
                {
                    return;
                }
            }

            // terminate unnecessary blocks earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else if(kargs.cu_seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k =
                    kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                         QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                         QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
            {
                batch_offset_q_descale =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_q_descale;
                batch_offset_k_descale =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_k_descale;
                batch_offset_v_descale =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_v_descale;
            }
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

            // If cumulative seqlen pointers are provided, override per-batch effective lengths
            if(kargs.cu_seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            }
            if(kargs.cu_seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k =
                    kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
            }
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr =
            reinterpret_cast<const QDataType*>(kargs.q_ptr) +
            (static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q + batch_offset_q) /
                ck_tile::numeric_traits<QDataType>::PackedSize;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            (static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
             batch_offset_k) /
                ck_tile::numeric_traits<KDataType>::PackedSize;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            (static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
             batch_offset_v) /
                ck_tile::numeric_traits<VDataType>::PackedSize;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<SageAttnPipeline::kAlignmentQ>{},
                number<1>{});
            if constexpr(SageAttnPipeline::kQLoadOnce)
            {
                return pad_tensor_view(q_dram_naive,
                                       make_tuple(number<SageAttnPipeline::kM0>{},
                                                  number<SageAttnPipeline::kSubQKHeaddim>{}),
                                       sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<SageAttnPipeline::kM0>{}, number<SageAttnPipeline::kK0>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<SageAttnPipeline::kAlignmentK>{},
                number<1>{});

            constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<SageAttnPipeline::kN0>{}, number<SageAttnPipeline::kK0>{}),
                sequence<kPadSeqLenK_, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.seqlen_k, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<SageAttnPipeline::kAlignmentV>{},
                    number<1>{});

                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(kargs.seqlen_k)),
                                          make_tuple(sequence<1>{}, sequence<0>{}),
                                          make_tuple(sequence<0>{}, sequence<1>{}));

                constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
                return pad_tensor_view(
                    v_dram_transposed,
                    make_tuple(number<SageAttnPipeline::kN1>{}, number<SageAttnPipeline::kK1>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK_>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen_k),
                    make_tuple(kargs.stride_v, 1),
                    number<SageAttnPipeline::kAlignmentV>{},
                    number<1>{});

                constexpr bool kPadHeadDimV_ = kUseAsyncCopy ? kPadHeadDimV : false;
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<SageAttnPipeline::kN1>{}, number<SageAttnPipeline::kK1>{}),
                    sequence<kPadHeadDimV_, kPadSeqLenK>{});
            }
        }();

        auto q_dram_window =
            make_tile_window(q_dram,
                             [&]() {
                                 if constexpr(SageAttnPipeline::kQLoadOnce)
                                     return make_tuple(number<SageAttnPipeline::kM0>{},
                                                       number<SageAttnPipeline::kSubQKHeaddim>{});
                                 else
                                     return make_tuple(number<SageAttnPipeline::kM0>{},
                                                       number<SageAttnPipeline::kK0>{});
                             }(),
                             {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<SageAttnPipeline::kN0>{}, number<SageAttnPipeline::kK0>{}),
            {0, 0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<SageAttnPipeline::kN1>{}, number<SageAttnPipeline::kK1>{}),
            {i_n1, 0});
        /// FIXME: Before C++20, capturing structured binding variables are not supported.
        /// Remove following copy capture of the 'i_nhead' if in C++20

        AttnMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<AttnMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    0,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return AttnMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        // WA i_batch capture structure binding before c++20
        auto position_encoding = EmptyPositionEncoding<SaccDataType>{};

        AttentionVariant variant;
        const auto variant_params = [&] {
            const float scale_s = [&] {
                if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
                {
                    float q_descale = *(reinterpret_cast<const float*>(kargs.q_descale_ptr));
                    float k_descale = *(reinterpret_cast<const float*>(kargs.k_descale_ptr));

                    return kargs.scale_s * q_descale * k_descale;
                }
                else
                {
                    return kargs.scale_s;
                }
            }();

            // logits_soft_cap is always disabled, use standard attention params
            return ck_tile::StandardAttentionParams<AttnMask>{mask, scale_s};
        }();

        BlockIndices block_indices{i_batch, i_nhead, i_nhead / kargs.nhead_ratio_qk};
        auto o_acc_tile = [&]() {
            using PipelineProblem = typename SageAttnPipeline::Problem;

            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
            {
                // TODO - move global load of descale to pipeline
                float v_descale = *(reinterpret_cast<const float*>(kargs.v_descale_ptr));

                float scale_p = ck_tile::type_convert<float>(ck_tile::numeric<PDataType>::max());
                float scale_o = v_descale / scale_p;

                auto o_acc_element_func = [&]() {
                    if constexpr(std::is_same_v<ODataType, ck_tile::fp8_t>)
                        return make_composes(
                            ck_tile::saturates<ck_tile::fp8_t>{},
                            ck_tile::scales<remove_cvref_t<decltype(scale_o)>>{scale_o});
                    else
                        return ck_tile::scales<remove_cvref_t<decltype(scale_o)>>{scale_o};
                }();
                return SageAttnPipeline{}(
                    q_dram_window,
                    identity{}, // q_element_func
                    k_dram_window,
                    identity{}, // k_element_func
                    v_dram_window,
                    identity{},                                         // v_element_func
                    identity{},                                         // s_acc_element_func
                    scales<remove_cvref_t<decltype(scale_p)>>{scale_p}, // p_compute_element_func
                    o_acc_element_func,                                 // o_acc_element_func
                    mask,
                    position_encoding,
                    variant_params.sm_scale,
                    variant,
                    variant_params,
                    block_indices,
                    smem_ptr);
            }
            else if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE)
            {
                const float* q_descale_ptr =
                    reinterpret_cast<const float*>(kargs.q_descale_ptr) +
                    static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q_descale +
                    batch_offset_q_descale;
                const float* k_descale_ptr =
                    reinterpret_cast<const float*>(kargs.k_descale_ptr) +
                    static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) *
                        kargs.nhead_stride_k_descale +
                    batch_offset_k_descale;
                const float* v_descale_ptr =
                    reinterpret_cast<const float*>(kargs.v_descale_ptr) +
                    static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) *
                        kargs.nhead_stride_v_descale +
                    batch_offset_v_descale;

                // BLOCKSCALE: one q_descale per tile (kBlockScaleSizeQ from traits, typically 128)
                const index_t idx = i_m0 / PipelineProblem::kBlockScaleSizeQ;
                float q_descale   = q_descale_ptr[idx];

                return SageAttnPipeline{}(
                    q_dram_window,
                    identity{}, // q_element_func
                    k_dram_window,
                    identity{}, // k_element_func
                    v_dram_window,
                    identity{},               // v_element_func
                    scales<float>(q_descale), // s_acc_element_func
                    identity{},               // p_compute_element_func - No scaling (done in exp2)
                    identity{}, // o_acc_element_func - No dequant (canceled by rowsum)
                    mask,
                    position_encoding,
                    kargs.scale_s,
                    variant,
                    variant_params,
                    block_indices,
                    smem_ptr,
                    nullptr,
                    k_descale_ptr,
                    v_descale_ptr);
            }
            else if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                              QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
            {
                using SageShape                  = typename SageAttnPipeline::BlockSageAttnShape;
                constexpr index_t kWarpSize      = get_warp_size();
                constexpr index_t kGemm0MPerWarp = SageShape::Gemm0WarpTile::at(number<0>{});
                constexpr index_t kNumWarps      = SageShape::NumWarps;

                static_assert(kWarpSize == 64, "kWarpSize must be 64");
                static_assert(SageAttnPipeline::kM0 == kGemm0MPerWarp * kNumWarps,
                              "PERWARP/PERTHREAD q_descale: kM0 must equal "
                              "Gemm0 MPerWarp * NumWarps");
                static_assert(kWarpSize % kGemm0MPerWarp == 0,
                              "PERWARP/PERTHREAD: warp_size must be divisible by Gemm0 MPerWarp");
                if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP)
                {
                    static_assert(kGemm0MPerWarp == PipelineProblem::kBlockScaleSizeQ,
                                  "PERWARP: Gemm0 MPerWarp must match kBlockScaleSizeQ "
                                  "(one q scale per warp with shared q_descale scalar)");
                }
                static_assert(SageShape::Gemm0WarpTile::at(number<0>{}) == 32 &&
                                  SageShape::Gemm0WarpTile::at(number<1>{}) == 32,
                              "PERWARP/PERTHREAD q_descale assumes Gemm0 warp tile MxN is 32x32");

                const float* q_descale_ptr =
                    reinterpret_cast<const float*>(kargs.q_descale_ptr) +
                    static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q_descale +
                    batch_offset_q_descale;
                const float* k_descale_ptr =
                    reinterpret_cast<const float*>(kargs.k_descale_ptr) +
                    static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) *
                        kargs.nhead_stride_k_descale +
                    batch_offset_k_descale;
                const float* v_descale_ptr =
                    reinterpret_cast<const float*>(kargs.v_descale_ptr) +
                    static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) *
                        kargs.nhead_stride_v_descale +
                    batch_offset_v_descale;

                // Q row from tile origin i_m0 + wave M strip + lane; clamp q_scale_idx to the
                // last scale block for this seqlen_q (e.g. seqlen_q=129, S=32: ceil(129/32)=5
                // blocks, indices 0..4; row 128 -> 128/32=4; padding -> min(raw_idx, max_idx)).
                constexpr index_t kBlockSq = PipelineProblem::kBlockScaleSizeQ;
                const index_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / kWarpSize);
                const index_t q_row_raw =
                    i_m0 + wave_id * kGemm0MPerWarp + threadIdx.x % kGemm0MPerWarp;
                const index_t q_scale_idx_raw = ck_tile::integer_divide_floor(q_row_raw, kBlockSq);
                const index_t max_q_scale_idx =
                    kargs.seqlen_q > 0 ? ck_tile::integer_divide_ceil(kargs.seqlen_q, kBlockSq) - 1
                                       : 0;
                const index_t q_scale_idx =
                    q_scale_idx_raw < max_q_scale_idx ? q_scale_idx_raw : max_q_scale_idx;
                const float q_descale = q_descale_ptr[q_scale_idx];

                return SageAttnPipeline{}(q_dram_window,
                                          identity{}, // q_element_func
                                          k_dram_window,
                                          identity{}, // k_element_func
                                          v_dram_window,
                                          identity{}, // v_element_func
                                          identity{}, // s_acc_element_func (K/V scales in pipeline)
                                          identity{}, // p_compute_element_func
                                          identity{}, // o_acc_element_func
                                          mask,
                                          position_encoding,
                                          kargs.scale_s,
                                          variant,
                                          variant_params,
                                          block_indices,
                                          smem_ptr,
                                          nullptr,
                                          k_descale_ptr,
                                          v_descale_ptr,
                                          q_descale);
            }
            else
            {
                return SageAttnPipeline{}(q_dram_window,
                                          k_dram_window,
                                          v_dram_window,
                                          mask,
                                          position_encoding,
                                          variant_params.sm_scale,
                                          variant,
                                          variant_params,
                                          block_indices,
                                          smem_ptr);
            }
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<SageAttnPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<SageAttnPipeline::kM0>{}, number<SageAttnPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window = make_tile_window(
            o_dram,
            make_tuple(number<SageAttnPipeline::kM0>{}, number<SageAttnPipeline::kN1>{}),
            {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
