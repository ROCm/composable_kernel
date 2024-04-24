// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

#include "ck_tile/host/reference/reference_batched_elementwise.hpp"
#include "ck_tile/host/reference/reference_batched_gemm.hpp"
#include "ck_tile/host/reference/reference_batched_masking.hpp"
#include "ck_tile/host/reference/reference_batched_softmax.hpp"

namespace ck_tile {

template <typename BiasDataType,
          typename LSEDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename QTensor,
          typename KTensor,
          typename VTensor,
          typename OTensor,
          typename MaskingType,
          typename PComputeElementFunction = ck_tile::identity,
          typename OAccElementFunction     = ck_tile::identity>
CK_TILE_HOST void
reference_batched_fmha(const QTensor& q,
                       const KTensor& k,
                       const VTensor& v,
                       OTensor& o,
                       index_t nhead_k,
                       float scale_s,
                       const MaskingType& mask,
                       PComputeElementFunction p_compute_element_func,
                       OAccElementFunction oacc_element_func,
                       std::optional<HostTensorView<const BiasDataType>> bias = std::nullopt,
                       std::optional<HostTensorView<LSEDataType>> lse         = std::nullopt)
{
    index_t batch = q.get_length(0);
    index_t nhead = q.get_length(1);

    using QDataType = tensor_value_t<QTensor>;
    using KDataType = tensor_value_t<KTensor>;
    using VDataType = tensor_value_t<VTensor>;
    using ODataType = tensor_value_t<OTensor>;

    // verify result individually for each batch/group
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        const ck_tile::index_t real_seqlen_q = q.get_length(2);
        const ck_tile::index_t real_seqlen_k = k.get_length(2);

        // adjust matrix index according to the mode
        const ck_tile::index_t query_start = 0;
        const ck_tile::index_t query_end   = query_start + real_seqlen_q;
        const ck_tile::index_t key_start   = 0;
        const ck_tile::index_t key_end     = key_start + real_seqlen_k;
        const ck_tile::index_t nr          = nhead / nhead_k;

        // clang-format off
        using Slice = ck_tile::HostTensorSlice;
        // tensor layout will be in [h, s, d] layout in verification
        auto q_host_view_hsd = q
                .index({Slice(0, b, b + 1), Slice(2, query_start, query_end)})
                .squeeze(0);
        auto k_host_view_hsd = k
                .index({Slice(0, b, b + 1), Slice(2, key_start, key_end)})
                .squeeze(0)
                .repeat({nr, 1, 1});
        auto v_host_view_hsd = v
                .index({Slice(0, b, b + 1), Slice(3, key_start, key_end)})
                .squeeze(0)
                .repeat({nr, 1, 1});
        auto o_host_view_hsd = o
                .index({Slice(0, b, b + 1), Slice(2, query_start, query_end)})
                .squeeze(0);
        // clang-format on

        // create local tensors to speed-up computation
        ck_tile::HostTensor<QDataType> q_host_ref(q_host_view_hsd.get_lengths());
        ck_tile::HostTensor<KDataType> k_host_ref(k_host_view_hsd.get_lengths());
        ck_tile::HostTensor<VDataType> v_host_ref(v_host_view_hsd.get_lengths());
        ck_tile::HostTensor<ODataType> o_host_ref(o_host_view_hsd.get_lengths());
        // create local tensors for holding intermediate result
        ck_tile::HostTensor<SMPLComputeDataType> s_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<SMPLComputeDataType> lse_host_ref({nhead, real_seqlen_q});

        q_host_ref.for_each([&](auto& self, auto i) { self(i) = q_host_view_hsd(i); });
        k_host_ref.for_each([&](auto& self, auto i) { self(i) = k_host_view_hsd(i); });
        v_host_ref.for_each([&](auto& self, auto i) { self(i) = v_host_view_hsd(i); });

        // reference
        ck_tile::reference_batched_gemm<SaccDataType>(q_host_ref,
                                                      k_host_ref,
                                                      s_host_ref,
                                                      ck_tile::identity{},
                                                      ck_tile::identity{},
                                                      ck_tile::scales(scale_s));

        if(bias.has_value())
        {
            // clang-format off
            auto bias_host_view_hsd = (*bias)
                    .index({Slice(2, query_start, query_end), Slice(3, key_start, key_end)})
                    .squeeze(0);
            // clang-format on

            // create local tensor to speed-up computation
            ck_tile::HostTensor<BiasDataType> bias_host_ref(bias_host_view_hsd.get_lengths());
            bias_host_ref.for_each([&](auto& self, auto i) { self(i) = bias_host_view_hsd(i); });

            // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q,
            // real_seqlen_k]
            ck_tile::reference_batched_elementwise<SMPLComputeDataType>(
                s_host_ref, bias_host_ref, s_host_ref);
        }

        if(mask.type == mask_enum::no_mask)
        {
            ck_tile::reference_batched_masking(s_host_ref,
                                               FmhaMasks::NoMask{real_seqlen_q, real_seqlen_k});
        }
        else if(mask.type == mask_enum::window_generic)
        {
            ck_tile::reference_batched_masking(
                s_host_ref,
                ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                    mask.left, mask.right, real_seqlen_q, real_seqlen_k));
        }
        else
        {
            // if left window size is negative, means causal
            // else means generic (for current batch)
            if(mask.left < 0)
                ck_tile::reference_batched_masking(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::CausalMask>(
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        mask.type == mask_enum::mask_top_left));
            else
                ck_tile::reference_batched_masking(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        mask.type == mask_enum::mask_top_left));
        }

        if(lse.has_value())
        {
            // clang-format off
            auto les_host_view_hsd = (*lse)
                    .index({Slice(0, b, b + 1), Slice(2, query_start, query_end)})
                    .squeeze(0);
            // clang-format on

            ck_tile::reference_batched_softmax<SMPLComputeDataType>(
                s_host_ref, p_host_ref, p_compute_element_func, lse_host_ref);
        }
        else
        {
            ck_tile::reference_batched_softmax<SMPLComputeDataType>(
                s_host_ref, p_host_ref, p_compute_element_func);
        }

        ck_tile::reference_batched_gemm<OaccDataType>(p_host_ref,
                                                      v_host_ref,
                                                      o_host_ref,
                                                      ck_tile::identity{},
                                                      ck_tile::identity{},
                                                      oacc_element_func);

        o_host_view_hsd.for_each([&](auto& self, auto i) { self(i) = o_host_ref(i); });
    }
}

} // namespace ck_tile
