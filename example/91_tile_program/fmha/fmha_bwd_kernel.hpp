// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tile_program/tile/tile_window.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] * K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] * V[hdim_v, seqlen_k]

template <typename TilePartitioner_, typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaBwdKernel
{
    using TilePartitioner                    = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                       = ck::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                   = ck::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    using QDataType    = ck::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using GemmDataType = ck::remove_cvref_t<typename FmhaPipeline::GemmDataType>;
    using LSEDataType  = ck::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using AccDataType  = ck::remove_cvref_t<typename FmhaPipeline::AccDataType>;
    using DDataType    = ck::remove_cvref_t<typename FmhaPipeline::DDataType>;
    // using ZDataType           = ck::remove_cvref_t<typename FmhaPipeline::ZDataType>;
    using OGradDataType    = ck::remove_cvref_t<typename FmhaPipeline::OGradDataType>;
    using QGradDataType    = ck::remove_cvref_t<typename FmhaPipeline::QGradDataType>;
    using KGradDataType    = ck::remove_cvref_t<typename FmhaPipeline::KGradDataType>;
    using VGradDataType    = ck::remove_cvref_t<typename FmhaPipeline::VGradDataType>;
    using BiasGradDataType = ck::remove_cvref_t<typename FmhaPipeline::BiasGradDataType>;

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
        ck::index_t nhead_ratio_qk;
        float scale;

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
    }

    struct FmhaBwdCommonBiasKargs
    {
        const void* bias_ptr           = nullptr;
        void* dbias_ptr                = nullptr;
        ck::index_t stride_bias        = 0;
        ck::index_t stride_dbias       = 0;
        ck::index_t nhead_stride_bias  = 0;
        ck::index_t nhead_stride_dbias = 0;
    };

    struct FmhaBwdBatchModeBiasKargs : FmhaBwdCommonBiasKargs
    {
        ck::index_t batch_stride_bias  = 0;
        ck::index_t batch_stride_dbias = 0;
    };

    struct FmhaBwdMaskKargs
    {
        ck::index_t mask_y, mask_x;
    };

    struct FmhaBwdBatchModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<kHasBias, FmhaBwdBatchModeBiasKargs, FmhaBwdEmptyKargs<0>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<1>>
    {
        ck::index_t batch_stride_q;
        ck::index_t batch_stride_k;
        ck::index_t batch_stride_v;
        ck::index_t batch_stride_do;
        ck::index_t batch_stride_lsed;
        ck::index_t batch_stride_dk;
        ck::index_t batch_stride_dv;
    };

    struct FmhaBwdGroupModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<kHasBias, FmhaBwdCommonBiasKargs, FmhaBwdEmptyKargs<0>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<1>>
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
              void* dq_ptr,
              void* dk_ptr,
              void* dv_ptr,
              void* dbias_ptr,
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
              ck::index_t stride_do,
              ck::index_t stride_dk,
              ck::index_t stride_dv,
              ck::index_t stride_dbias,
              ck::index_t nhead_stride_q,
              ck::index_t nhead_stride_k,
              ck::index_t nhead_stride_v,
              ck::index_t nhead_stride_bias,
              ck::index_t nhead_stride_do,
              ck::index_t nhead_stride_lsed,
              ck::index_t nhead_stride_dbias,
              ck::index_t batch_stride_q,
              ck::index_t batch_stride_k,
              ck::index_t batch_stride_v,
              ck::index_t batch_stride_bias,
              ck::index_t batch_stride_do,
              ck::index_t batch_stride_lsed,
              ck::index_t batch_stride_dk,
              ck::index_t batch_stride_dv,
              ck::index_t batch_stride_dbias,
              ck::index_t mask_y,
              ck::index_t mask_x)
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
                     nhead_ratio_qk,
#if CK_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale * ck::math::log2e_v<>),
#else
                     scale,
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
                     nhead_stride_lsed}, // args for common karg
                    {},                  // placeholder for bias
                    {},                  // placeholder for mask
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_do,
                    batch_stride_lsed,
                    batch_stride_dk,
                    batch_stride_dv};

        if constexpr(kHasBias)
        {
            kargs.bias_ptr           = bias_ptr;
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_bias        = stride_bias;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_bias  = nhead_stride_bias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
            kargs.batch_stride_bias  = batch_stride_bias;
            kargs.batch_stride_dbias = batch_stride_dbias;
        }

        if constexpr(kHasMask)
        {
            kargs.mask_y = mask_y;
            kargs.mask_x = mask_x;
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
              void* dq_ptr,
              void* dk_ptr,
              void* dv_ptr,
              void* dbias_ptr,
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
              ck::index_t stride_do,
              ck::index_t stride_dk,
              ck::index_t stride_dv,
              ck::index_t stride_dbias,
              ck::index_t nhead_stride_q,
              ck::index_t nhead_stride_k,
              ck::index_t nhead_stride_v,
              ck::index_t nhead_stride_bias,
              ck::index_t nhead_stride_do,
              ck::index_t nhead_stride_lsed,
              ck::index_t nhead_stride_dbias,
              ck::index_t mask_y,
              ck::index_t mask_x)
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
                     nhead_ratio_qk,
#if CK_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale * ck::math::log2e_v<>),
#else
                     scale,
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
                     nhead_stride_lsed}, // args for common karg
                    {},                  // placeholder for bias
                    {},                  // placeholder for mask
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};

        if constexpr(kHasBias)
        {
            kargs.bias_ptr           = bias_ptr;
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_bias        = stride_bias;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_bias  = nhead_stride_bias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
        }
        if constexpr(kHasMask)
        {
            kargs.mask_y = mask_y;
            kargs.mask_x = mask_x;
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

        long_index_t batch_offset_q     = 0;
        long_index_t batch_offset_k     = 0;
        long_index_t batch_offset_v     = 0;
        long_index_t batch_offset_bias  = 0;
        long_index_t batch_offset_do    = 0;
        long_index_t batch_offset_lsed  = 0;
        long_index_t batch_offset_dk    = 0;
        long_index_t batch_offset_dv    = 0;
        long_index_t batch_offset_dbias = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q    = query_start * kargs.stride_q;
            batch_offset_k    = key_start * kargs.stride_k;
            batch_offset_v    = key_start * kargs.stride_v;
            batch_offset_do   = query_start * kargs.stride_do;
            batch_offset_lsed = query_start;
            batch_offset_dk   = key_start * kargs.stride_dk;
            batch_offset_dv   = key_start * kargs.stride_dv;
            if constexpr(kHasBias)
            {
                batch_offset_bias  = query_start * kargs.stride_bias + key_start;
                batch_offset_dbias = query_start * kargs.stride_dbias + key_start;
            }
            else
            {
                batch_offset_bias  = key_start;
                batch_offset_dbias = key_start;
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
                batch_offset_bias  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
                batch_offset_dbias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dbias;
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
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv +
                                batch_offset_dv;

        // Q/K/V/LSE/D/dO/dQ/dK/dV DRAM and DRAM window
        const auto q_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            Number<32>{},
            Number<1>{});
        const auto q_dram = [&]() {
            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                    Sequence<kM0NeedPadding, kK0N1NeedPadding>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{}),
                    Sequence<kM0NeedPadding, kK0N1NeedPadding>{});
            }
        }();

        const auto qt_dram_naive =
            transform_tensor_view(q_dram_naive,
                                  make_tuple(make_pass_through_transform(kargs.hdim_q),
                                             make_pass_through_transform(kargs.seqlen_q)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto qt_dram =
            [&]() {
                /// FIXME: The return value of xx_dram_naive.GetTensorDescriptor().GetLength() is
                /// same as
                ///   xx_dram_transposed.GetTensorDescriptor().GetLength(). Replace following
                ///   if-clause by pad_tensor_view() call after fixing this issue.
                if constexpr(kK0N1NeedPadding || kM0NeedPadding)
                {
                    const auto transform_m = [&] {
                        if constexpr(kM0NeedPadding)
                        {
                            const index_t m_pad_length =
                                [&]() {
                                    if constexpr(FmhaPipeline::kQTLoadOnce)
                                    {
                                        return FmhaPipeline::kM0 *
                                                   ck::math::integer_divide_ceil(
                                                       kargs.seqlen_q, FmhaPipeline::kM0) -
                                               kargs.seqlen_q;
                                    }
                                    else
                                    {
                                        return FmhaPipeline::kK3 *
                                                   ck::math::integer_divide_ceil(
                                                       kargs.seqlen_q, FmhaPipeline::kK3) -
                                               kargs.seqlen_q;
                                    }
                                }

                            return make_right_pad_transform(kargs.seqlen_q, m_pad_length);
                        }
                        else
                        {
                            return make_pass_through_transform(kargs.seqlen_q);
                        }
                    }();

                    const auto transform_k = [&] {
                        if constexpr(kK0N1NeedPadding)
                        {
                            const index_t k_pad_length = FmhaPipeline::kQKHeaddim - kargs.hdim_q;

                            return make_right_pad_transform(kargs.hdim_q, k_pad_length);
                        }
                        else
                        {
                            return make_pass_through_transform(kargs.hdim_q);
                        }
                    }();

                    return transform_tensor_view(qt_dram_naive,
                                                 make_tuple(transform_k, transform_m),
                                                 make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                 make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
                else
                {
                    return qt_dram_naive;
                }
            }
        // const auto qt_dram = [&]() {
        //     if constexpr(FmhaPipeline::kQTLoadOnce)
        //     {
        //         return pad_tensor_view(
        //             qt_dram_naive,
        //             make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kM0>{}),
        //             Sequence<kK0N1NeedPadding, kM0NeedPadding>{});
        //     }
        //     else
        //     {
        //         return pad_tensor_view(
        //             qt_dram_naive,
        //             make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kK3>{}),
        //             Sequence<kK0N1NeedPadding, kM0NeedPadding>{});
        //     }
        // }();

        const auto k_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
            k_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            Number<32>{},
            Number<1>{});
        const auto k_dram = [&]() {
            if constexpr(FmhaPipeline::kKLoadOnce)
            {
                return pad_tensor_view(
                    k_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                    Sequence<kN0K1NeedPadding, kK0N1NeedPadding>{});
            }
            else
            {
                return pad_tensor_view(
                    k_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{}),
                    Sequence<kN0K1NeedPadding, kK0N1NeedPadding>{});
            }
        }();

        const auto kt_dram_naive =
            transform_tensor_view(k_dram_naive,
                                  make_tuple(make_pass_through_transform(kargs.hdim_q),
                                             make_pass_through_transform(kargs.seqlen_k)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto kt_dram =
            [&]() {
                /// FIXME: The return value of xx_dram_naive.GetTensorDescriptor().GetLength() is
                /// same as
                ///   xx_dram_transposed.GetTensorDescriptor().GetLength(). Replace following
                ///   if-clause by pad_tensor_view() call after fixing this issue.
                if constexpr(kK0N1NeedPadding || kN0K1NeedPadding)
                {
                    const auto transform_n = [&] {
                        if constexpr(kN0K1NeedPadding)
                        {
                            const index_t n_pad_length =
                                [&]() {
                                    if constexpr(FmhaPipeline::kKTLoadOnce)
                                    {
                                        return FmhaPipeline::kN0 *
                                                   ck::math::integer_divide_ceil(
                                                       kargs.seqlen_k, FmhaPipeline::kN0) -
                                               kargs.seqlen_k;
                                    }
                                    else
                                    {
                                        return FmhaPipeline::kK4 *
                                                   ck::math::integer_divide_ceil(
                                                       kargs.seqlen_q, FmhaPipeline::kK4) -
                                               kargs.seqlen_k;
                                    }
                                }

                            return make_right_pad_transform(kargs.seqlen_k, n_pad_length);
                        }
                        else
                        {
                            return make_pass_through_transform(kargs.seqlen_k);
                        }
                    }();

                    const auto transform_k = [&] {
                        if constexpr(kK0N1NeedPadding)
                        {
                            const index_t k_pad_length = FmhaPipeline::kQKHeaddim - kargs.hdim_q;

                            return make_right_pad_transform(kargs.hdim_q, k_pad_length);
                        }
                        else
                        {
                            return make_pass_through_transform(kargs.hdim_q);
                        }
                    }();

                    return transform_tensor_view(qt_dram_naive,
                                                 make_tuple(transform_k, transform_n),
                                                 make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                 make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
                else
                {
                    return qt_dram_naive;
                }
            }
        // const auto kt_dram = [&]() {
        //     if constexpr(FmhaPipeline::kKTLoadOnce)
        //     {
        //         return pad_tensor_view(
        //             kt_dram_naive,
        //             make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kN0>{}),
        //             Sequence<kK0N1NeedPadding, kN0K1NeedPadding>{});
        //     }
        //     else
        //     {
        //         return pad_tensor_view(
        //             kt_dram_naive,
        //             make_tuple(Number<FmhaPipeline::kQKHeaddim>{}, Number<FmhaPipeline::kK4>{}),
        //             Sequence<kK0N1NeedPadding, kN0K1NeedPadding>{});
        //     }
        // }();

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                Number<32>{},
                Number<1>{});
            if constexpr(FmhaPipeline::kVLoadOnce)
            {
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kVHeaddim>{}),
                    Sequence<kN0K1NeedPadding, kK0N1NeedPadding>{});
            }
            else
            {
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK2>{}),
                    Sequence<kN0K1NeedPadding, kK0N1NeedPadding>{});
            }
        }();

        const auto lse_dram = [&]() {
            const auto lse_dram_naive = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                lse_ptr, make_tuple(kargs.seqlen_q), Number<32>{});
            return pad_tensor_view(lse_dram_naive,
                                   make_tuple(Number<FmhaPipeline::kM0>{}),
                                   Sequence<kM0NeedPadding>{});
        }();

        const auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                d_ptr, make_tuple(kargs.seqlen_q), Number<32>{});
            return pad_tensor_view(
                d_dram_naive, make_tuple(Number<FmhaPipeline::kM0>{}), Sequence<kM0NeedPadding>{});
        }();

        const auto do_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
            do_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_o, 1),
            Number<32>{},
            Number<1>{});
        const auto do_dram = [&]() {
            if constexpr(FmhaPipeline::kOGradLoadOnce)
            {
                return pad_tensor_view(
                    do_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kVHeaddim>{}),
                    Sequence<kM0NeedPadding, kK0N1NeedPadding>{});
            }
            else
            {
                return pad_tensor_view(
                    do_dram_naive,
                    make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK2>{}),
                    Sequence<kM0NeedPadding, kK0N1NeedPadding>{});
            }
        }();

        const auto dot_dram_naive =
            transform_tensor_view(do_dram_naive,
                                  make_tuple(make_pass_through_transform(kargs.hdim_v),
                                             make_pass_through_transform(kargs.seqlen_q)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto dot_dram =
            [&]() {
                /// FIXME: The return value of xx_dram_naive.GetTensorDescriptor().GetLength() is
                /// same as
                ///   xx_dram_transposed.GetTensorDescriptor().GetLength(). Replace following
                ///   if-clause by pad_tensor_view() call after fixing this issue.
                if constexpr(kK0N1NeedPadding || kM0NeedPadding)
                {
                    const auto transform_m = [&] {
                        if constexpr(kM0NeedPadding)
                        {
                            const index_t m_pad_length =
                                [&]() {
                                    if constexpr(FmhaPipeline::kQTLoadOnce)
                                    {
                                        return FmhaPipeline::kM0 *
                                                   ck::math::integer_divide_ceil(
                                                       kargs.seqlen_q, FmhaPipeline::kM0) -
                                               kargs.seqlen_q;
                                    }
                                    else
                                    {
                                        return FmhaPipeline::kK1 *
                                                   ck::math::integer_divide_ceil(
                                                       kargs.seqlen_q, FmhaPipeline::kK1) -
                                               kargs.seqlen_q;
                                    }
                                }

                            return make_right_pad_transform(kargs.seqlen_q, m_pad_length);
                        }
                        else
                        {
                            return make_pass_through_transform(kargs.seqlen_q);
                        }
                    }();

                    const auto transform_k = [&] {
                        if constexpr(kK0N1NeedPadding)
                        {
                            const index_t k_pad_length = FmhaPipeline::kVHeaddim - kargs.hdim_v;

                            return make_right_pad_transform(kargs.hdim_v, k_pad_length);
                        }
                        else
                        {
                            return make_pass_through_transform(kargs.hdim_v);
                        }
                    }();

                    return transform_tensor_view(dot_dram_naive,
                                                 make_tuple(transform_k, transform_m),
                                                 make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                 make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
                else
                {
                    return dot_dram_naive;
                }
            }
        // const auto dot_dram = [&]() {
        //     if constexpr(FmhaPipeline::kOGradTLoadOnce)
        //     {
        //         return pad_tensor_view(
        //             dot_dram_naive,
        //             make_tuple(Number<FmhaPipeline::kVHeaddim>{}, Number<FmhaPipeline::kM0>{}),
        //             Sequence<kK0N1NeedPadding, kM0NeedPadding>{});
        //     }
        //     else
        //     {
        //         return pad_tensor_view(
        //             dot_dram_naive,
        //             make_tuple(Number<FmhaPipeline::kVHeaddim>{}, Number<FmhaPipeline::kK1>{}),
        //             Sequence<kK0N1NeedPadding, kM0NeedPadding>{});
        //     }
        // }();

        auto dq_dram = [&]() {
            const auto dq_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global,
                                                              InMemoryDataOperationEnum::AtomicAdd>(
                dq_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(
                dq_dram_naive,
                make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                Sequence<kM0NeedPadding, kK0N1NeedPadding>{});
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
                        Number<32>{},
                        Number<1>{});

                    return pad_tensor_view(bias_dram_naive,
                                           bias_dram_window_lengths,
                                           Sequence<kM0NeedPadding, kN0K1NeedPadding>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        const auto dbias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(kHasBias)
            {
                const BiasGradDataType* dbias_ptr =
                    reinterpret_cast<const BiasGradDataType*>(kargs.dbias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dbias +
                    batch_offset_dbias;

                const auto dbias_dram = [&]() {
                    const auto dbias_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                        dbias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_dbias, 1),
                        Number<32>{},
                        Number<1>{});

                    return pad_tensor_view(dbias_dram_naive,
                                           bias_dram_window_lengths,
                                           Sequence<kM0NeedPadding, kN0K1NeedPadding>{});
                }();

                return make_tile_window(dbias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return FmhaMask{kargs.mask_y, kargs.mask_x, kargs.seqlen_q, kargs.seqlen_k};
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        auto [dk_acc_tile, dv_acc_tile] = FmhaPipeline{}(q_dram_window,
                                                         qt_dram_window,
                                                         k_dram_window,
                                                         kt_dram_window,
                                                         v_dram_window,
                                                         bias_dram_window,
                                                         do_dram_window,
                                                         dot_dram_window,
                                                         lse_dram_window,
                                                         d_dram_window,
                                                         dq_dram_window,
                                                         dbias_dram_window,
                                                         mask,
                                                         kargs.scale,
                                                         smem_ptr);

        auto dk_dram = [&]() {
            const auto dk_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                dk_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(
                dk_dram_naive,
                make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
                Sequence<kN0K1NeedPadding, kK0N1NeedPadding>{});
        }();

        auto dv_dram = [&]() {
            const auto dv_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                dv_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                Number<32>{},
                Number<1>{});

            return pad_tensor_view(
                dv_dram_naive,
                make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kVHeaddim>{}),
                Sequence<kN0K1NeedPadding, kK0N1NeedPadding>{});
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

template <typename TilePartitioner_, typename FmhaOGradDotO_>
struct FmhaBwdOGradDotOKernel
{
    using TilePartitioner = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaOGradDotO   = ck::remove_cvref_t<FmhaOGradDotO_>;

    using DDataType     = ck::remove_cvref_t<typename FmhaOGradDotO::DDataType>;
    using ODataType     = ck::remove_cvref_t<typename FmhaOGradDotO::ODataType>;
    using OGradDataType = ck::remove_cvref_t<typename FmhaOGradDotO::OGradDataType>;

    static constexpr ck::index_t kBlockSize  = FmhaOGradDotO::kBlockSize;
    static constexpr ck::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    static constexpr ck::index_t kM0       = kBlockSize;
    static constexpr ck::index_t kVHeaddim = FmhaOGradDotO::kVHeaddim;

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdOGradDotOCommonKargs
    {
        const void* o_ptr;
        const void* do_ptr;
        void* d_ptr;

        ck::index_t seqlen_q;
        ck::index_t hdim_v;

        ck::index_t stride_o;

        ck::index_t nhead_stride_o;
        ck::index_t nhead_stride_d;
    }

    struct FmhaBwdOGradDotOBatchModeKargs : FmhaBwdOGradDotOCommonKargs
    {
        ck::index_t batch_stride_o;
        ck::index_t batch_stride_d;
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
                                                                      ck::index_t seqlen_q,
                                                                      ck::index_t hdim_v,
                                                                      ck::index_t stride_o,
                                                                      ck::index_t nhead_stride_o,
                                                                      ck::index_t nhead_stride_d,
                                                                      ck::index_t batch_stride_o,
                                                                      ck::index_t batch_stride_d)
    {
        Kargs kargs{{o_ptr,
                     do_ptr,
                     d_ptr,
                     seqlen_q,
                     hdim_v,
                     stride_o,
                     nhead_stride_o,
                     nhead_stride_d,
                     batch_stride_o,
                     batch_stride_d}};

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(const void* o_ptr,
                                                                      const void* do_ptr,
                                                                      void* d_ptr,
                                                                      ck::index_t seqlen_q,
                                                                      ck::index_t hdim_v,
                                                                      ck::index_t stride_o,
                                                                      ck::index_t nhead_stride_o,
                                                                      ck::index_t nhead_stride_d,
                                                                      const void* seqstart_q_ptr)
    {
        Kargs kargs{o_ptr,
                    do_ptr,
                    d_ptr,
                    -1, // seqlen will be updated by another pointer
                    hdim_v,
                    stride_o,
                    nhead_stride_o,
                    nhead_stride_d,
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

        // allocate LDS
        // __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_q);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kM0);

        long_index_t batch_offset_o = 0;
        long_index_t batch_offset_d = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_o = query_start * kargs.stride_o;
            batch_offset_d = query_start;

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
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
            batch_offset_d = static_cast<long_index_t>(i_batch) * kargs.batch_stride_d;
        }

        // for simplicity, batch stride we just modify the pointer
        const ODataType* o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                                 batch_offset_o;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                                      batch_offset_o;
        DDataType* d_ptr = reinterpret_cast<DDataType*>(kargs.d_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_d +
                           batch_offset_d;

        // O/dO/D DRAM and DRAM window
        const auto o_dram = [&]() {
            auto o_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                Number<32>{},
                Number<1>{});
            return pad_tensor_view(o_dram_naive,
                                   make_tuple(Number<kM0>{}, Number<kVHeaddim>{}),
                                   Sequence<kM0NeedPadding, kK0N1NeedPadding>{});
        }();
        const auto do_dram = [&]() {
            auto do_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                do_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                Number<32>{},
                Number<1>{});
            return pad_tensor_view(do_dram_naive,
                                   make_tuple(Number<kM0>{}, Number<kVHeaddim>{}),
                                   Sequence<kM0NeedPadding, kK0N1NeedPadding>{});
        }();
        auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                d_ptr, make_tuple(kargs.seqlen_q), Number<32>{});
            return pad_tensor_view(
                d_dram_naive, make_tuple(Number<kM0>{}), Sequence<kM0NeedPadding>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram, make_tuple(Number<kM0>{}, Number<kVHeaddim>{}), {i_m0, 0});

        auto do_dram_window =
            make_tile_window(do_dram, make_tuple(Number<kM0>{}, Number<kVHeaddim>{}), {i_m0, 0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(Number<kM0>{}), {i_m0});

        FmhaOGradDotO{}(o_dram_window, do_dram_window, d_dram_window);
    }
};
