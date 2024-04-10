// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include <ck/utility/common_header.hpp>
#include <ck/tensor/tensor_view.hpp>
#include <ck/tile_program/tile/tile_window.hpp>

#include "ck_tiled_fmha_definitions.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// dV[seqlen_k, hdim_v] = P^T[seqlen_k, seqlen_q] @ dO^T[hdim_v, seqlen_q]
// dP[seqlen_q, seqlen_k] = dO[seqlen_q, hdim_v] @ V[seqlen_k, hdim_v]
// D[seqlen_q] = rowsum(dO[seqlen_q, hdim_v] * O[seqlen_q, hdim_v])
// dS''[seqlen_q, seqlen_k] = P[seqlen_q, seqlen_k] * (dP[seqlen_q, seqlen_k] - D[seqlen_q])
// dBias[seqlen_q, seqlen_k] = dS'[seqlen_q, seqlen_k] = dS''[seqlen_q, seqlen_k]
// dK[seqlen_k, hdim_q] = dS'^T[seqlen_k, seqlen_q] @ Q^T[hdim_q, seqlen_q] * Scale[1]
// dQ[seqlen_q, hdim_q] = dS'[seqlen_q, seqlen_k] @ K^T[hdim_q, seqlen_k] * Scale[1]

template <typename TilePartitioner_, typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaBwdKernel
{
    using TilePartitioner                    = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                       = ck::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                   = ck::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    using QDataType             = ck::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType             = ck::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType             = ck::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType          = ck::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using GemmDataType          = ck::remove_cvref_t<typename FmhaPipeline::GemmDataType>;
    using LSEDataType           = ck::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using AccDataType           = ck::remove_cvref_t<typename FmhaPipeline::AccDataType>;
    using DDataType             = ck::remove_cvref_t<typename FmhaPipeline::DDataType>;
    using RandValOutputDataType = ck::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using OGradDataType         = ck::remove_cvref_t<typename FmhaPipeline::OGradDataType>;
    using QGradDataType         = ck::remove_cvref_t<typename FmhaPipeline::QGradDataType>;
    using KGradDataType         = ck::remove_cvref_t<typename FmhaPipeline::KGradDataType>;
    using VGradDataType         = ck::remove_cvref_t<typename FmhaPipeline::VGradDataType>;
    using BiasGradDataType      = ck::remove_cvref_t<typename FmhaPipeline::BiasGradDataType>;

    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasBias     = FmhaPipeline::kHasBias;
    static constexpr bool kHasBiasGrad = FmhaPipeline::kHasBiasGrad;
    static constexpr bool kHasDropout  = FmhaPipeline::kHasDropout;
    using FmhaMask                     = ck::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask     = FmhaMask::IsMasking;

    template <ck::index_t I> // to avoid duplicated base class prblem, introduce an template arg
    struct FmhaBwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* lse_ptr;
        const void* do_ptr;
        const void* d_ptr;
        void* dq_ptr;
        void* dk_ptr;
        void* dv_ptr;

        ck::index_t seqlen_q;
        ck::index_t seqlen_k;
        ck::index_t hdim_q;
        ck::index_t hdim_v;

        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck::index_t num_head_q;
        ck::index_t nhead_ratio_qk;
        float raw_scale;
#if CK_FMHA_FWD_FAST_EXP2
        float scale;
#endif

        ck::index_t stride_q;
        ck::index_t stride_k;
        ck::index_t stride_v;
        ck::index_t stride_do;
        ck::index_t stride_dk;
        ck::index_t stride_dv;

        ck::index_t nhead_stride_q;
        ck::index_t nhead_stride_k;
        ck::index_t nhead_stride_v;
        ck::index_t nhead_stride_do;
        ck::index_t nhead_stride_lsed;

        ck::index_t batch_stride_lsed;

        // only used for handling some strange xformers test
        ck::index_t hdim_stride_do;
    };

    struct FmhaBwdCommonBiasKargs
    {
        const void* bias_ptr          = nullptr;
        ck::index_t stride_bias       = 0;
        ck::index_t nhead_stride_bias = 0;
    };

    struct FmhaBwdBatchModeBiasKargs : FmhaBwdCommonBiasKargs
    {
        ck::index_t batch_stride_bias = 0;
    };

    struct FmhaBwdCommonBiasGradKargs
    {
        void* dbias_ptr                = nullptr;
        ck::index_t stride_dbias       = 0;
        ck::index_t nhead_stride_dbias = 0;
    };

    struct FmhaBwdBatchModeBiasGradKargs : FmhaBwdCommonBiasGradKargs
    {
        ck::index_t batch_stride_dbias = 0;
    };

    struct FmhaBwdMaskKargs
    {
        CausalMaskType mask_type;
        ck::index_t window_size;
    };

    struct FmhaBwdCommonDropoutKargs
    {
        void init_dropout(const float p_drop,
                          const std::tuple<uint64_t, uint64_t>& drop_seed_offset,
                          const float raw_scale)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop       = 1.0 / p_undrop;
            scale_rp_undrop = rp_undrop * raw_scale;

            drop_seed   = std::get<0>(drop_seed_offset);
            drop_offset = std::get<1>(drop_seed_offset);
        }
        float rp_undrop             = 1;
        float scale_rp_undrop       = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        bool is_store_randval       = false;
        uint64_t drop_seed          = 1;
        uint64_t drop_offset        = 0;
        void* rand_val_ptr          = nullptr;

        ck::index_t stride_randval       = 0;
        ck::index_t nhead_stride_randval = 0;
    };
    struct FmhaBwdBatchModeDropoutKargs : FmhaBwdCommonDropoutKargs
    {
        ck::index_t batch_stride_randval = 0;
    };

    struct FmhaBwdBatchModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<kHasBias, FmhaBwdBatchModeBiasKargs, FmhaBwdEmptyKargs<0>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdBatchModeBiasGradKargs, FmhaBwdEmptyKargs<1>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasDropout, FmhaBwdBatchModeDropoutKargs, FmhaBwdEmptyKargs<3>>
    {
        ck::index_t batch_stride_q;
        ck::index_t batch_stride_k;
        ck::index_t batch_stride_v;
        ck::index_t batch_stride_do;
        ck::index_t batch_stride_dk;
        ck::index_t batch_stride_dv;
    };

    struct FmhaBwdGroupModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<kHasBias, FmhaBwdCommonBiasKargs, FmhaBwdEmptyKargs<0>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdCommonBiasGradKargs, FmhaBwdEmptyKargs<1>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasDropout, FmhaBwdCommonDropoutKargs, FmhaBwdEmptyKargs<3>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaBwdGroupModeKargs, FmhaBwdBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* lse_ptr,
              const void* do_ptr,
              const void* d_ptr,
              void* rand_val_ptr,
              void* dq_ptr,
              void* dk_ptr,
              void* dv_ptr,
              void* dbias_ptr,
              ck::index_t seqlen_q,
              ck::index_t seqlen_k,
              ck::index_t hdim_q,
              ck::index_t hdim_v,
              ck::index_t num_head_q,
              ck::index_t nhead_ratio_qk,
              float scale,
              ck::index_t stride_q,
              ck::index_t stride_k,
              ck::index_t stride_v,
              ck::index_t stride_bias,
              ck::index_t stride_randval,
              ck::index_t stride_do,
              ck::index_t stride_dk,
              ck::index_t stride_dv,
              ck::index_t stride_dbias,
              ck::index_t nhead_stride_q,
              ck::index_t nhead_stride_k,
              ck::index_t nhead_stride_v,
              ck::index_t nhead_stride_bias,
              ck::index_t nhead_stride_randval,
              ck::index_t nhead_stride_do,
              ck::index_t nhead_stride_lsed,
              ck::index_t nhead_stride_dbias,
              ck::index_t batch_stride_q,
              ck::index_t batch_stride_k,
              ck::index_t batch_stride_v,
              ck::index_t batch_stride_bias,
              ck::index_t batch_stride_randval,
              ck::index_t batch_stride_do,
              ck::index_t batch_stride_lsed,
              ck::index_t batch_stride_dk,
              ck::index_t batch_stride_dv,
              ck::index_t batch_stride_dbias,
              ck::index_t hdim_stride_do,
              CausalMaskType mask_type,
              ck::index_t window_size,
              float p_drop,
              bool s_randval,
              std::tuple<uint64_t, uint64_t> drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_ptr,
                     do_ptr,
                     d_ptr,
                     dq_ptr,
                     dk_ptr,
                     dv_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     scale,
#if CK_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale * ck::math::log2e_v<>),
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_do,
                     stride_dk,
                     stride_dv,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_do,
                     nhead_stride_lsed,
                     batch_stride_lsed,
                     hdim_stride_do}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for dbias
                    {},               // placeholder for mask
                    {},               // placeholder for dropout
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_do,
                    batch_stride_dk,
                    batch_stride_dv};

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }

        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
            kargs.batch_stride_dbias = batch_stride_dbias;
        }

        if constexpr(kHasMask)
        {
            kargs.mask_type   = mask_type;
            kargs.window_size = window_size;
        }

        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, drop_seed_offset, scale);
            kargs.rand_val_ptr         = rand_val_ptr;
            kargs.stride_randval       = stride_randval;
            kargs.nhead_stride_randval = nhead_stride_randval;
            kargs.batch_stride_randval = batch_stride_randval;
            kargs.is_store_randval     = s_randval;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* lse_ptr,
              const void* do_ptr,
              const void* d_ptr,
              void* rand_val_ptr,
              void* dq_ptr,
              void* dk_ptr,
              void* dv_ptr,
              void* dbias_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck::index_t hdim_q,
              ck::index_t hdim_v,
              ck::index_t num_head_q,
              ck::index_t nhead_ratio_qk,
              float scale,
              ck::index_t stride_q,
              ck::index_t stride_k,
              ck::index_t stride_v,
              ck::index_t stride_bias,
              ck::index_t stride_randval,
              ck::index_t stride_do,
              ck::index_t stride_dk,
              ck::index_t stride_dv,
              ck::index_t stride_dbias,
              ck::index_t nhead_stride_q,
              ck::index_t nhead_stride_k,
              ck::index_t nhead_stride_v,
              ck::index_t nhead_stride_bias,
              ck::index_t nhead_stride_randval,
              ck::index_t nhead_stride_do,
              ck::index_t nhead_stride_lsed,
              ck::index_t nhead_stride_dbias,
              ck::index_t batch_stride_lse,
              ck::index_t hdim_stride_do,
              CausalMaskType mask_type,
              ck::index_t window_size,
              float p_drop,
              bool s_randval,
              std::tuple<uint64_t, uint64_t> drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_ptr,
                     do_ptr,
                     d_ptr,
                     dq_ptr,
                     dk_ptr,
                     dv_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     scale,
#if CK_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale * ck::math::log2e_v<>),
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_do,
                     stride_dk,
                     stride_dv,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_do,
                     nhead_stride_lsed,
                     batch_stride_lse,
                     hdim_stride_do}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for dbias
                    {},               // placeholder for mask
                    {},               // placeholder for dropout
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
        }
        if constexpr(kHasMask)
        {
            kargs.mask_type   = mask_type;
            kargs.window_size = window_size;
        }
        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, drop_seed_offset, scale);
            kargs.rand_val_ptr         = rand_val_ptr;
            kargs.stride_randval       = stride_randval;
            kargs.nhead_stride_randval = nhead_stride_randval;
            kargs.is_store_randval     = s_randval;
        }

        return kargs;
    }

    __host__ static constexpr auto
    GridSize(ck::index_t batch_size_, ck::index_t nhead_, ck::index_t seqlen_k_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_k_);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return ck::math::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_n, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_k);

        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN0);

        long_index_t batch_offset_q       = 0;
        long_index_t batch_offset_k       = 0;
        long_index_t batch_offset_v       = 0;
        long_index_t batch_offset_bias    = 0;
        long_index_t batch_offset_randval = 0;
        long_index_t batch_offset_do      = 0;
        long_index_t batch_offset_lsed    = 0;
        long_index_t batch_offset_dk      = 0;
        long_index_t batch_offset_dv      = 0;
        long_index_t batch_offset_dbias   = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q    = query_start * kargs.stride_q;
            batch_offset_k    = key_start * kargs.stride_k;
            batch_offset_v    = key_start * kargs.stride_v;
            batch_offset_do   = query_start * kargs.stride_do;
            batch_offset_lsed = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;
            batch_offset_dk   = key_start * kargs.stride_dk;
            batch_offset_dv   = key_start * kargs.stride_dv;
            if constexpr(kHasBias)
            {
                batch_offset_bias = query_start * kargs.stride_bias + key_start;
            }
            else
            {
                batch_offset_bias = key_start;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = query_start * kargs.stride_dbias + key_start;
            }
            else
            {
                batch_offset_dbias = key_start;
            }
            if constexpr(kHasDropout)
            {
                batch_offset_randval = query_start * kargs.stride_randval;
            }

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_k <= i_n0)
            {
                return;
            }
        }
        else
        {
            batch_offset_q    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_do   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_lsed = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;
            batch_offset_dk   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
            batch_offset_dv   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
            if constexpr(kHasBias)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dbias;
            }
            if constexpr(kHasDropout)
            {
                batch_offset_randval =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_randval;
            }
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                     batch_offset_lsed;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                 batch_offset_lsed;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        QGradDataType* dq_ptr = reinterpret_cast<QGradDataType*>(kargs.dq_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                batch_offset_q;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_k +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_v +
                                batch_offset_dv;

        // Q/K/V/LSE/D/dO/dQ/dK/dV DRAM and DRAM window
        const auto q_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            Number<FmhaPipeline::kAlignmentQ>{},
            Number<1>{});
        const auto q_dram = [&]() {
            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                    Sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{}),
                    Sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();

        const auto qt_dram_naive =
            transform_tensor_view(q_dram_naive,
                                  make_tuple(make_pass_through_transform(kargs.hdim_q),
                                             make_pass_through_transform(kargs.seqlen_q)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto qt_dram = [&]() {
            if constexpr(FmhaPipeline::kQTLoadOnce)
            {
                return pad_tensor_view(
                    qt_dram_naive,
                    make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kM0>{}),
                    Sequence<kPadHeadDimQ, kPadSeqLenQ>{});
            }
            else
            {
                return pad_tensor_view(
                    qt_dram_naive,
                    make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kK3>{}),
                    Sequence<kPadHeadDimQ, kPadSeqLenQ>{});
            }
        }();

        const auto k_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
            k_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            Number<FmhaPipeline::kAlignmentK>{},
            Number<1>{});
        const auto k_dram = [&]() {
            if constexpr(FmhaPipeline::kKLoadOnce)
            {
                return pad_tensor_view(
                    k_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                    Sequence<kPadSeqLenK, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    k_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{}),
                    Sequence<kPadSeqLenK, kPadHeadDimQ>{});
            }
        }();

        const auto kt_dram_naive =
            transform_tensor_view(k_dram_naive,
                                  make_tuple(make_pass_through_transform(kargs.hdim_q),
                                             make_pass_through_transform(kargs.seqlen_k)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto kt_dram = [&]() {
            if constexpr(FmhaPipeline::kKTLoadOnce)
            {
                return pad_tensor_view(
                    kt_dram_naive,
                    make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kN0>{}),
                    Sequence<kPadHeadDimQ, kPadSeqLenK>{});
            }
            else
            {
                return pad_tensor_view(
                    kt_dram_naive,
                    make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kK4>{}),
                    Sequence<kPadHeadDimQ, kPadSeqLenK>{});
            }
        }();

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                Number<FmhaPipeline::kAlignmentV>{},
                Number<1>{});
            if constexpr(FmhaPipeline::kVLoadOnce)
            {
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kVHeaddim>{}),
                    Sequence<kPadSeqLenK, kPadHeadDimV>{});
            }
            else
            {
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK2>{}),
                    Sequence<kPadSeqLenK, kPadHeadDimV>{});
            }
        }();

        const auto lse_dram = [&]() {
            const auto lse_dram_naive = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                lse_ptr, make_tuple(kargs.seqlen_q), Number<1>{});
            return pad_tensor_view(
                lse_dram_naive, make_tuple(Number<FmhaPipeline::kM0>{}), Sequence<kPadSeqLenQ>{});
        }();

        const auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                d_ptr, make_tuple(kargs.seqlen_q), Number<1>{});
            return pad_tensor_view(
                d_dram_naive, make_tuple(Number<FmhaPipeline::kM0>{}), Sequence<kPadSeqLenQ>{});
        }();

        const auto do_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
            do_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_do, kargs.hdim_stride_do),
            Number<FmhaPipeline::kAlignmentOGrad>{},
            Number<1>{});
        const auto do_dram = [&]() {
            if constexpr(FmhaPipeline::kOGradLoadOnce)
            {
                return pad_tensor_view(
                    do_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kVHeaddim>{}),
                    Sequence<kPadSeqLenQ, kPadHeadDimV>{});
            }
            else
            {
                return pad_tensor_view(
                    do_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK2>{}),
                    Sequence<kPadSeqLenQ, kPadHeadDimV>{});
            }
        }();

        const auto dot_dram_naive =
            transform_tensor_view(do_dram_naive,
                                  make_tuple(make_pass_through_transform(kargs.hdim_v),
                                             make_pass_through_transform(kargs.seqlen_q)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto dot_dram = [&]() {
            if constexpr(FmhaPipeline::kOGradTLoadOnce)
            {
                return pad_tensor_view(
                    dot_dram_naive,
                    make_tuple(Number<FmhaPipeline::kVHeaddim>{}, Number<FmhaPipeline::kM0>{}),
                    Sequence<kPadHeadDimV, kPadSeqLenQ>{});
            }
            else
            {
                return pad_tensor_view(
                    dot_dram_naive,
                    make_tuple(Number<FmhaPipeline::kVHeaddim>{}, Number<FmhaPipeline::kK1>{}),
                    Sequence<kPadHeadDimV, kPadSeqLenQ>{});
            }
        }();

        auto dq_dram = [&]() {
            const auto dq_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global,
                                                              InMemoryDataOperationEnum::AtomicAdd>(
                dq_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                Number<FmhaPipeline::kAlignmentQGrad>{},
                Number<1>{});

            return pad_tensor_view(
                dq_dram_naive,
                make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                Sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kM0>{},
                                      Number<FmhaPipeline::kQKHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{});
            }(),
            {0, 0});

        auto qt_dram_window =
            make_tile_window(qt_dram,
                             [&]() {
                                 if constexpr(FmhaPipeline::kQTLoadOnce)
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kM0>{});
                                 else
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kK3>{});
                             }(),
                             {0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            [&]() {
                if constexpr(FmhaPipeline::kKLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kN0>{},
                                      Number<FmhaPipeline::kQKHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{});
            }(),
            {i_n0, 0});

        auto kt_dram_window =
            make_tile_window(kt_dram,
                             [&]() {
                                 if constexpr(FmhaPipeline::kKTLoadOnce)
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kN0>{});
                                 else
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kK4>{});
                             }(),
                             {0, i_n0});

        auto v_dram_window = make_tile_window(
            v_dram,
            [&]() {
                if constexpr(FmhaPipeline::kVLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kN0>{},
                                      Number<FmhaPipeline::kVHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK2>{});
            }(),
            {i_n0, 0});

        auto do_dram_window = make_tile_window(
            do_dram,
            [&]() {
                if constexpr(FmhaPipeline::kOGradLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kM0>{},
                                      Number<FmhaPipeline::kVHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK2>{});
            }(),
            {0, 0});

        auto dot_dram_window =
            make_tile_window(dot_dram,
                             [&]() {
                                 if constexpr(FmhaPipeline::kOGradTLoadOnce)
                                     return make_tuple(Number<FmhaPipeline::kVHeaddim>{},
                                                       Number<FmhaPipeline::kM0>{});
                                 else
                                     return make_tuple(Number<FmhaPipeline::kVHeaddim>{},
                                                       Number<FmhaPipeline::kK1>{});
                             }(),
                             {0, 0});

        auto dq_dram_window = make_tile_window(
            dq_dram,
            make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
            {0, 0});

        auto lse_dram_window =
            make_tile_window(lse_dram, make_tuple(Number<FmhaPipeline::kM0>{}), {0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(Number<FmhaPipeline::kM0>{}), {0});

        /// FIXME: Before C++20, capturing structured binding variables is not supported. Remove
        /// following copy capture of the 'i_nhead'
        ///        if compiled in C++20
        constexpr auto bias_dram_window_lengths =
            make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN0>{});
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(kHasBias)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                        bias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_bias, 1),
                        Number<FmhaPipeline::kAlignmentBias>{},
                        Number<1>{});

                    return pad_tensor_view(bias_dram_naive,
                                           bias_dram_window_lengths,
                                           Sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        auto dbias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(kHasBiasGrad)
            {
                BiasGradDataType* dbias_ptr =
                    reinterpret_cast<BiasGradDataType*>(kargs.dbias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dbias +
                    batch_offset_dbias;

                auto dbias_dram = [&]() {
                    const auto dbias_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                        dbias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_dbias, 1),
                        Number<FmhaPipeline::kAlignmentBias>{},
                        Number<1>{});

                    return pad_tensor_view(dbias_dram_naive,
                                           bias_dram_window_lengths,
                                           Sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(dbias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // dropout
        float rp_undrop             = 1;
        float scale_rp_undrop       = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        uint64_t drop_seed          = 0;
        uint64_t drop_offset        = 0;
        bool is_store_randval       = false;

        if constexpr(kHasDropout)
        {
            rp_undrop           = kargs.rp_undrop;
            scale_rp_undrop     = kargs.scale_rp_undrop;
            p_undrop_in_uint8_t = kargs.p_undrop_in_uint8_t;
            drop_seed           = kargs.drop_seed;
            drop_offset         = kargs.drop_offset;
            is_store_randval    = kargs.is_store_randval;
        }
        BlockDropout dropout(i_batch,
                             i_nhead,
                             kargs.num_head_q,
                             drop_seed,
                             drop_offset,
                             rp_undrop,
                             p_undrop_in_uint8_t,
                             is_store_randval);

        auto randval_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto randval_dram_window_lengths =
                make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN0>{});
            if constexpr(kHasDropout)
            {
                RandValOutputDataType* rand_val_ptr =
                    reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_randval +
                    batch_offset_randval;

                const auto randval_dram = [&]() {
                    const auto randval_dram_naive =
                        make_naive_tensor_view<AddressSpaceEnum::Global>(
                            rand_val_ptr,
                            make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                            make_tuple(kargs.stride_randval, 1),
                            Number<1>{},
                            Number<1>{});

                    return pad_tensor_view(randval_dram_naive,
                                           randval_dram_window_lengths,
                                           Sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(randval_dram, randval_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(randval_dram_window_lengths);
            }
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
            {
                auto res =
                    ck::make_tuple(ck::index_t{0}, ck::index_t{0}, ck::index_t{0}, ck::index_t{0});

                if(kargs.window_size > 0)
                {
                    if(kargs.mask_type == CausalMaskType::MaskDisabled)
                    {
                        ck::index_t left_size  = kargs.window_size / 2;
                        ck::index_t right_size = kargs.window_size - 1 - left_size;

                        res = ck::make_generic_attention_mask_coordinates_from_lr_window(
                            left_size, right_size, kargs.seqlen_q, kargs.seqlen_k);
                    }
                    else
                    {
                        bool is_topleft =
                            (kargs.mask_type == CausalMaskType::MaskUpperTriangleFromTopLeft);

                        res = ck::make_generic_attention_mask_coordinates_from_lr_window(
                            kargs.window_size - 1, 0, kargs.seqlen_q, kargs.seqlen_k, is_topleft);
                    }
                }
                else
                {
                    if(kargs.mask_type == CausalMaskType::MaskDisabled)
                    {
                        res = ck::make_generic_attention_mask_coordinates_from_lr_window(
                            -1, -1, kargs.seqlen_q, kargs.seqlen_k);
                    }
                    else
                    {
                        bool is_topleft =
                            (kargs.mask_type == CausalMaskType::MaskUpperTriangleFromTopLeft);

                        res = ck::make_generic_attention_mask_coordinates_from_lr_window(
                            -1, 0, kargs.seqlen_q, kargs.seqlen_k, is_topleft);
                    }
                }

                auto y = res.At(ck::Number<0>{});
                auto x = res.At(ck::Number<1>{});

                return FmhaMask{y, x, kargs.seqlen_q, kargs.seqlen_k};
            }
            else
                return FmhaMask{0, 0, kargs.seqlen_q, kargs.seqlen_k};
        }();

        auto [dk_acc_tile, dv_acc_tile] = FmhaPipeline{}(q_dram_window,
                                                         qt_dram_window,
                                                         k_dram_window,
                                                         kt_dram_window,
                                                         v_dram_window,
                                                         bias_dram_window,
                                                         randval_dram_window,
                                                         do_dram_window,
                                                         dot_dram_window,
                                                         lse_dram_window,
                                                         d_dram_window,
                                                         dq_dram_window,
                                                         dbias_dram_window,
                                                         mask,
                                                         kargs.raw_scale,
#if CK_FMHA_FWD_FAST_EXP2
                                                         kargs.scale,
#endif
                                                         rp_undrop,
                                                         scale_rp_undrop,
                                                         smem_ptr,
                                                         dropout);

        auto dk_dram = [&]() {
            const auto dk_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                dk_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_dk, 1),
                Number<FmhaPipeline::kAlignmentKGrad>{},
                Number<1>{});

            return pad_tensor_view(
                dk_dram_naive,
                make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                Sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();

        auto dv_dram = [&]() {
            const auto dv_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                dv_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_dv, 1),
                Number<FmhaPipeline::kAlignmentVGrad>{},
                Number<1>{});

            return pad_tensor_view(
                dv_dram_naive,
                make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kVHeaddim>{}),
                Sequence<kPadSeqLenK, kPadHeadDimV>{});
        }();

        auto dk_dram_window = make_tile_window(
            dk_dram,
            make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto dv_dram_window = make_tile_window(
            dv_dram,
            make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});

        EpiloguePipeline{}(dk_dram_window, dv_dram_window, dk_acc_tile, dv_acc_tile);
    }
};

template <typename TilePartitioner_, typename FmhaBwdOGradDotO_>
struct FmhaBwdOGradDotOKernel
{
    using TilePartitioner                    = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaBwdOGradDotO                   = ck::remove_cvref_t<FmhaBwdOGradDotO_>;
    static constexpr ck::index_t kBlockSize  = FmhaBwdOGradDotO::kBlockSize;
    static constexpr ck::index_t kBlockPerCu = FmhaBwdOGradDotO::kBlockPerCu;
    static constexpr ck::index_t kM0         = kBlockSize;
    static constexpr ck::index_t kVHeaddim   = FmhaBwdOGradDotO::kVHeaddim;

    using DDataType     = ck::remove_cvref_t<typename FmhaBwdOGradDotO::DDataType>;
    using ODataType     = ck::remove_cvref_t<typename FmhaBwdOGradDotO::ODataType>;
    using OGradDataType = ck::remove_cvref_t<typename FmhaBwdOGradDotO::OGradDataType>;

    static constexpr bool kIsGroupMode = FmhaBwdOGradDotO::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaBwdOGradDotO::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV = FmhaBwdOGradDotO::kPadHeadDimV;

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdOGradDotOCommonKargs
    {
        const void* o_ptr;
        const void* do_ptr;
        void* d_ptr;

        float p_undrop;

        ck::index_t seqlen_q;
        ck::index_t hdim_v;

        ck::index_t stride_do;
        ck::index_t stride_o;

        ck::index_t nhead_stride_do;
        ck::index_t nhead_stride_o;
        ck::index_t nhead_stride_d;
        ck::index_t batch_stride_d;

        // only used for handling some strange xformers test
        ck::index_t hdim_stride_do;
    };

    struct FmhaBwdOGradDotOBatchModeKargs : FmhaBwdOGradDotOCommonKargs
    {
        ck::index_t batch_stride_do;
        ck::index_t batch_stride_o;
    };

    struct FmhaBwdOGradDotOGroupModeKargs : FmhaBwdOGradDotOCommonKargs
    {
        const int32_t* seqstart_q_ptr;
    };

    using Kargs = std::
        conditional_t<kIsGroupMode, FmhaBwdOGradDotOGroupModeKargs, FmhaBwdOGradDotOBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(const void* o_ptr,
                                                                      const void* do_ptr,
                                                                      void* d_ptr,
                                                                      float p_undrop,
                                                                      ck::index_t seqlen_q,
                                                                      ck::index_t hdim_v,
                                                                      ck::index_t stride_do,
                                                                      ck::index_t stride_o,
                                                                      ck::index_t nhead_stride_do,
                                                                      ck::index_t nhead_stride_o,
                                                                      ck::index_t nhead_stride_d,
                                                                      ck::index_t batch_stride_do,
                                                                      ck::index_t batch_stride_o,
                                                                      ck::index_t batch_stride_d,
                                                                      ck::index_t hdim_stride_do)
    {
        Kargs kargs{{o_ptr,
                     do_ptr,
                     d_ptr,
                     p_undrop,
                     seqlen_q,
                     hdim_v,
                     stride_do,
                     stride_o,
                     nhead_stride_do,
                     nhead_stride_o,
                     nhead_stride_d,
                     batch_stride_d,
                     hdim_stride_do},
                    batch_stride_do,
                    batch_stride_o};

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(const void* o_ptr,
                                                                      const void* do_ptr,
                                                                      void* d_ptr,
                                                                      float p_undrop,
                                                                      const void* seqstart_q_ptr,
                                                                      ck::index_t hdim_v,
                                                                      ck::index_t stride_do,
                                                                      ck::index_t stride_o,
                                                                      ck::index_t nhead_stride_do,
                                                                      ck::index_t nhead_stride_o,
                                                                      ck::index_t nhead_stride_d,
                                                                      ck::index_t batch_stride_d,
                                                                      ck::index_t hdim_stride_do)
    {
        Kargs kargs{{o_ptr,
                     do_ptr,
                     d_ptr,
                     p_undrop,
                     -1, // seqlen will be updated by another pointer
                     hdim_v,
                     stride_do,
                     stride_o,
                     nhead_stride_do,
                     nhead_stride_o,
                     nhead_stride_d,
                     batch_stride_d,
                     hdim_stride_do},
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr)};

        return kargs;
    }

    __host__ static constexpr auto
    GridSize(ck::index_t batch_size_, ck::index_t nhead_, ck::index_t seqlen_q_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_q);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kM0);

        long_index_t batch_offset_o  = 0;
        long_index_t batch_offset_do = 0;
        long_index_t batch_offset_d  = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_o  = query_start * kargs.stride_o;
            batch_offset_do = query_start * kargs.stride_do;
            batch_offset_d  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_d;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }
        }
        else
        {
            batch_offset_o  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
            batch_offset_do = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_d  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_d;
        }

        // for simplicity, batch stride we just modify the pointer
        const ODataType* o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                                 batch_offset_o;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        DDataType* d_ptr = reinterpret_cast<DDataType*>(kargs.d_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_d +
                           batch_offset_d;

        // O/dO/D DRAM and DRAM window
        const auto o_dram = [&]() {
            auto o_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                Number<FmhaBwdOGradDotO::kAlignmentO>{},
                Number<1>{});
            return pad_tensor_view(o_dram_naive,
                                   make_tuple(Number<kM0>{}, Number<kVHeaddim>{}),
                                   Sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();
        const auto do_dram = [&]() {
            auto do_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                do_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_do, kargs.hdim_stride_do),
                Number<FmhaBwdOGradDotO::kAlignmentOGrad>{},
                Number<1>{});
            return pad_tensor_view(do_dram_naive,
                                   make_tuple(Number<kM0>{}, Number<kVHeaddim>{}),
                                   Sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();
        auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                d_ptr, make_tuple(kargs.seqlen_q), Number<1>{});
            return pad_tensor_view(
                d_dram_naive, make_tuple(Number<kM0>{}), Sequence<kPadSeqLenQ>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram, make_tuple(Number<kM0>{}, Number<kVHeaddim>{}), {i_m0, 0});

        auto do_dram_window =
            make_tile_window(do_dram, make_tuple(Number<kM0>{}, Number<kVHeaddim>{}), {i_m0, 0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(Number<kM0>{}), {i_m0});

        FmhaBwdOGradDotO{}(o_dram_window, do_dram_window, d_dram_window, kargs.p_undrop);
    }
};
