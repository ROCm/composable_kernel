// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include "ck/utility/common_header.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/utility/philox_rand.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] * K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] * V[hdim_v, seqlen_k]

template <typename TilePartitioner_, typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdKernel
{
    using TilePartitioner                    = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                       = ck::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                   = ck::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    using QDataType              = ck::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType              = ck::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType              = ck::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType           = ck::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using RandValOutputDataType  = ck::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using LSEDataType            = ck::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType              = ck::remove_cvref_t<typename FmhaPipeline::ODataType>;
    static constexpr bool kIsFp8 = FmhaPipeline::kIsFp8;

    using VLayout = ck::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasBias     = FmhaPipeline::kHasBias;
    static constexpr bool kStoreLSE    = FmhaPipeline::kStoreLSE;
    static constexpr bool kHasDropout  = FmhaPipeline::kHasDropout;
    using FmhaMask                     = ck::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask     = FmhaMask::IsMasking;

    template <ck::index_t I> // to avoid duplicated base class prblem, introduce an template arg
    struct FmhaFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_ptr;

        ck::index_t seqlen_q;
        ck::index_t seqlen_k;
        ck::index_t hdim_q;
        ck::index_t hdim_v;

        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck::index_t nhead_ratio_qk;
        float scale;

        ck::index_t stride_q;
        ck::index_t stride_k;
        ck::index_t stride_v;
        ck::index_t stride_o;

        ck::index_t nhead_stride_q;
        ck::index_t nhead_stride_k;
        ck::index_t nhead_stride_v;
        ck::index_t nhead_stride_o;
    };

    struct FmhaFwdCommonBiasKargs
    {
        const void* bias_ptr          = nullptr;
        ck::index_t stride_bias       = 0;
        ck::index_t nhead_stride_bias = 0;
    };

    struct FmhaFwdBatchModeBiasKargs : FmhaFwdCommonBiasKargs
    {
        ck::index_t batch_stride_bias = 0;
    };

    struct FmhaFwdMaskKargs
    {
        ck::index_t mask_y, mask_x;
    };

    struct FmhaFwdFP8Kargs
    {
        float descale_qk; // q*k
        float descale_sv; // s*v
        // float * o_amax_ptr;
    };

    struct FmhaFwdCommonLSEKargs
    {
        void* lse_ptr                = nullptr;
        ck::index_t nhead_stride_lse = 0;
    };

    struct FmhaFwdBatchModeLSEKargs : FmhaFwdCommonLSEKargs
    {
        ck::index_t batch_stride_lse = 0;
    };

    struct FmhaFwdDropoutKargs
    {
        void init_dropout(float p_drop, std::tuple<uint64_t, uint64_t>& drop_seeds)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop = 1.0 / p_undrop;

            drop_seed   = std::get<0>(drop_seeds);
            drop_offset = std::get<1>(drop_seeds);
        }
        float rp_undrop               = 1;
        uint8_t p_undrop_in_uint8_t   = std::numeric_limits<uint8_t>::max();
        uint64_t drop_seed            = 1;
        uint64_t drop_offset          = 0;
        void* rand_val_ptr            = nullptr;
        ck::index_t stride_drop       = 0;
        ck::index_t nhead_stride_drop = 0;
    };
    struct FmhaFwdBatchModelDropoutKargs : FmhaFwdDropoutKargs
    {
        ck::index_t batch_stride_drop = 0;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasBias, FmhaFwdBatchModeBiasKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kStoreLSE, FmhaFwdBatchModeLSEKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kIsFp8, FmhaFwdFP8Kargs, FmhaFwdEmptyKargs<3>>,
          std::conditional_t<kHasDropout, FmhaFwdBatchModelDropoutKargs, FmhaFwdEmptyKargs<4>>
    {
        ck::index_t batch_stride_q;
        ck::index_t batch_stride_k;
        ck::index_t batch_stride_v;
        ck::index_t batch_stride_o;
    };

    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasBias, FmhaFwdCommonBiasKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kIsFp8, FmhaFwdFP8Kargs, FmhaFwdEmptyKargs<3>>,
          std::conditional_t<kHasDropout, FmhaFwdDropoutKargs, FmhaFwdEmptyKargs<4>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaFwdGroupModeKargs, FmhaFwdBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* rand_val_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck::index_t seqlen_q,
              ck::index_t seqlen_k,
              ck::index_t hdim_q,
              ck::index_t hdim_v,
              ck::index_t nhead_ratio_qk,
              float scale,
              ck::index_t stride_q,
              ck::index_t stride_k,
              ck::index_t stride_v,
              ck::index_t stride_bias,
              ck::index_t stride_drop,
              ck::index_t stride_o,
              ck::index_t nhead_stride_q,
              ck::index_t nhead_stride_k,
              ck::index_t nhead_stride_v,
              ck::index_t nhead_stride_bias,
              ck::index_t nhead_stride_drop,
              ck::index_t nhead_stride_lse,
              ck::index_t nhead_stride_o,
              ck::index_t batch_stride_q,
              ck::index_t batch_stride_k,
              ck::index_t batch_stride_v,
              ck::index_t batch_stride_bias,
              ck::index_t batch_stride_drop,
              ck::index_t batch_stride_lse,
              ck::index_t batch_stride_o,
              ck::index_t mask_y,
              ck::index_t mask_x,
              float descale_qk,
              float descale_sv,
              float p_drop,
              std::tuple<uint64_t, uint64_t>& drop_seeds)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     nhead_ratio_qk,
#if CK_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale * ck::math::log2e_v<>),
#else
                     scale,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for fp8 args
                    {},               // placeholder for dropout
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.mask_y = mask_y;
            kargs.mask_x = mask_x;
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }
        if constexpr(kIsFp8)
        {
            kargs.descale_qk = descale_qk;
            kargs.descale_sv = descale_sv;
        }

        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, drop_seeds);
            kargs.rand_val_ptr      = rand_val_ptr;
            kargs.stride_drop       = stride_drop;
            kargs.nhead_stride_drop = nhead_stride_drop;
            kargs.batch_stride_drop = batch_stride_drop;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* rand_val_ptr,
              void* lse_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck::index_t hdim_q,
              ck::index_t hdim_v,
              ck::index_t nhead_ratio_qk,
              float scale,
              ck::index_t stride_q,
              ck::index_t stride_k,
              ck::index_t stride_v,
              ck::index_t stride_bias,
              ck::index_t stride_drop,
              ck::index_t stride_o,
              ck::index_t nhead_stride_q,
              ck::index_t nhead_stride_k,
              ck::index_t nhead_stride_v,
              ck::index_t nhead_stride_bias,
              ck::index_t nhead_stride_drop,
              ck::index_t nhead_stride_lse,
              ck::index_t nhead_stride_o,
              ck::index_t mask_y,
              ck::index_t mask_x,
              float descale_qk,
              float descale_sv,
              float p_drop,
              std::tuple<uint64_t, uint64_t>& drop_seeds)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     nhead_ratio_qk,
#if CK_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale * ck::math::log2e_v<>),
#else
                     scale,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for fp8 args
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
        if constexpr(kHasMask)
        {
            kargs.mask_y = mask_y;
            kargs.mask_x = mask_x;
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }
        if constexpr(kIsFp8)
        {
            kargs.descale_qk = descale_qk;
            kargs.descale_sv = descale_sv;
        }

        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, drop_seeds);
            kargs.rand_val_ptr      = rand_val_ptr;
            kargs.stride_drop       = stride_drop;
            kargs.nhead_stride_drop = nhead_stride_drop;
        }

        return kargs;
    }

    __host__ static constexpr auto GridSize(ck::index_t batch_size_,
                                            ck::index_t nhead_,
                                            ck::index_t seqlen_q_,
                                            ck::index_t hdim_v_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_, hdim_v_);
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
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] =
            TilePartitioner{}(kargs.seqlen_q, kargs.hdim_v);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q    = 0;
        long_index_t batch_offset_k    = 0;
        long_index_t batch_offset_v    = 0;
        long_index_t batch_offset_bias = 0;
        long_index_t batch_offset_drop = 0;
        long_index_t batch_offset_lse  = 0;
        long_index_t batch_offset_o    = 0;

        long_index_t nhead_stride_drop = 0;
        index_t max_seqlen_k           = 0;
        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }
            if constexpr(kHasBias)
            {
                batch_offset_bias = query_start * kargs.stride_bias + key_start;
            }
            else
            {
                batch_offset_bias = key_start;
            }
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = query_start;
            }
            if constexpr(kHasDropout)
            {
                batch_offset_drop = query_start * kargs.stride_drop;
                nhead_stride_drop = kargs.nhead_stride_drop;
                max_seqlen_k      = kargs.stride_drop;
            }

            batch_offset_o = query_start * kargs.stride_o;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
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
            if constexpr(kHasBias)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            }
            if constexpr(kHasDropout)
            {
                batch_offset_drop = static_cast<long_index_t>(i_batch) * kargs.batch_stride_drop;
                nhead_stride_drop = kargs.nhead_stride_drop;
                max_seqlen_k      = kargs.stride_drop;
            }
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
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
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                Number<32>{},
                Number<1>{});
            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0BlockLength>{}),
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
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                k_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{}),
                Sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                    v_ptr,
                    make_tuple(kargs.seqlen_k, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    Number<32>{},
                    Number<1>{});

                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(kargs.seqlen_k)),
                                          make_tuple(Sequence<1>{}, Sequence<0>{}),
                                          make_tuple(Sequence<0>{}, Sequence<1>{}));

                return pad_tensor_view(
                    v_dram_transposed,
                    make_tuple(Number<FmhaPipeline::kN1>{}, Number<FmhaPipeline::kK1>{}),
                    Sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, kargs.seqlen_k),
                    make_tuple(kargs.stride_v, 1),
                    Number<32>{},
                    Number<1>{});

                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN1>{}, Number<FmhaPipeline::kK1>{}),
                    Sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kM0>{},
                                      Number<FmhaPipeline::kK0BlockLength>{});
                else
                    return make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(Number<FmhaPipeline::kN1>{}, Number<FmhaPipeline::kK1>{}),
                             {i_n1, 0});
        /// FIXME: Before C++20, capturing structured binding variables is not supported. Remove
        /// following copy capture of the 'i_nhead'
        ///        if compiled in C++20
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto bias_dram_window_lengths =
                make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN0>{});
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
                        Number<32>{},
                        Number<1>{});

                    return pad_tensor_view(bias_dram_naive,
                                           bias_dram_window_lengths,
                                           Sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // lse
        auto lse_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto lse_dram_window_lengths = make_tuple(Number<FmhaPipeline::kM0>{});
            if constexpr(kStoreLSE)
            {
                LSEDataType* lse_ptr =
                    reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse + batch_offset_lse;

                const auto lse_dram = [&]() {
                    const auto lse_dram_naive =
                        make_naive_tensor_view<AddressSpaceEnum::Global>(lse_ptr,
                                                                         make_tuple(kargs.seqlen_q),
                                                                         make_tuple(1),
                                                                         Number<1>{},
                                                                         Number<1>{});

                    return pad_tensor_view(
                        lse_dram_naive, lse_dram_window_lengths, Sequence<kPadSeqLenQ>{});
                }();

                return make_tile_window(lse_dram, lse_dram_window_lengths, {i_m0});
            }
            else
            {
                return make_null_tile_window(lse_dram_window_lengths);
            }
        }();

        // about dropout
        long_index_t i_total_id =
            static_cast<long_index_t>(i_nhead) * nhead_stride_drop + batch_offset_drop;
        auto drop_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto drop_dram_window_lengths =
                make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN0>{});
            if constexpr(kHasDropout)
            {
                RandValOutputDataType* rand_val_ptr =
                    reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) + i_total_id;

                const auto drop_dram = [&]() {
                    const auto drop_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                        rand_val_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_drop, 1),
                        Number<32>{},
                        Number<1>{});

                    return pad_tensor_view(drop_dram_naive,
                                           drop_dram_window_lengths,
                                           Sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(drop_dram, drop_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(drop_dram_window_lengths);
            }
        }();
        float rp_undrop             = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        uint64_t drop_seed          = 0;
        uint64_t drop_offset        = 0;

        if constexpr(kHasDropout)
        {
            rp_undrop           = kargs.rp_undrop;
            p_undrop_in_uint8_t = kargs.p_undrop_in_uint8_t;
            drop_seed           = kargs.drop_seed;
            drop_offset         = kargs.drop_offset;
        }
        ck::philox ph(drop_seed, 0, drop_offset);
        BlockFmhaDropout dropout(i_total_id, rp_undrop, p_undrop_in_uint8_t, max_seqlen_k);

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return FmhaMask{kargs.mask_y, kargs.mask_x, kargs.seqlen_q, kargs.seqlen_k};
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        auto o_acc_tile = [&]() {
            if constexpr(kIsFp8)
            {
                return FmhaPipeline{}(q_dram_window,
                                      k_dram_window,
                                      v_dram_window,
                                      bias_dram_window,
                                      drop_dram_window,
                                      lse_dram_window,
                                      mask,
                                      kargs.scale,
                                      kargs.descale_qk,
                                      kargs.descale_sv,
                                      smem_ptr,
                                      dropout,
                                      ph);
            }
            else
            {
                return FmhaPipeline{}(q_dram_window,
                                      k_dram_window,
                                      v_dram_window,
                                      bias_dram_window,
                                      drop_dram_window,
                                      lse_dram_window,
                                      mask,
                                      kargs.scale,
                                      smem_ptr,
                                      dropout,
                                      ph);
            }
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN1>{}),
                Sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile);
    }
};
