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

template <typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename QueryTensor,
          typename KeyTensor,
          typename ValueTensor,
          typename BiasTensor,
          typename OutputTensor,
          typename LSETensor,
          typename MaskingType,
          typename PComputeElementFunction = ck_tile::identity,
          typename OAccElementFunction     = ck_tile::identity>
CK_TILE_HOST void
reference_batched_fmha(const QueryTensor& query_bhsd,
                       const KeyTensor& key_bhsd,
                       const ValueTensor& value_bhsd,
                       std::optional<BiasTensor> bias_bhss,
                       OutputTensor& output_bhsd,
                       std::optional<LSETensor> lse_bhs,
                       index_t nhead_k,
                       float scale_s,
                       const MaskingType& mask,
                       std::optional<span<const int32_t>> seqstart_q, // only used in group mode
                       std::optional<span<const int32_t>> seqstart_k, // only used in group mode
                       PComputeElementFunction p_compute_element_func = {},
                       OAccElementFunction oacc_element_func          = {})
{
    assert(!(seqstart_q.has_value() ^ seqstart_k.has_value()));

    const bool is_batch_mode     = !seqstart_q.has_value();
    const ck_tile::index_t batch = (is_batch_mode ? query_bhsd.get_length(0) : 1);
    const ck_tile::index_t nhead = query_bhsd.get_length(1);

    using QueryDataType = tensor_value_t<QueryTensor>;
    using KeyDataType   = tensor_value_t<KeyTensor>;
    using ValueDataType = tensor_value_t<ValueTensor>;
    using BiasDataType  = tensor_value_t<BiasTensor>;

    // verify result individually for each batch/group
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        const ck_tile::index_t real_seqlen_q =
            (is_batch_mode ? query_bhsd.get_length(2) : (*seqstart_q)[b + 1] - (*seqstart_q)[b]);
        const ck_tile::index_t real_seqlen_k =
            (is_batch_mode ? key_bhsd.get_length(2) : (*seqstart_k)[b + 1] - (*seqstart_k)[b]);

        // adjust matrix index according to the mode
        const ck_tile::index_t batch_start = (is_batch_mode ? b : 0);
        const ck_tile::index_t batch_end   = batch_start + 1;
        const ck_tile::index_t query_start = (is_batch_mode ? 0 : (*seqstart_q)[b]);
        const ck_tile::index_t query_end   = query_start + real_seqlen_q;
        const ck_tile::index_t key_start   = (is_batch_mode ? 0 : (*seqstart_k)[b]);
        const ck_tile::index_t key_end     = key_start + real_seqlen_k;
        const ck_tile::index_t nr          = nhead / nhead_k;

        // clang-format off
        using Slice = ck_tile::HostTensorSlice;
        // tensor layout will be in [h, s, d] layout in verification
        auto query_view_hsd = query_bhsd
                .index({Slice(0, batch_start, batch_end), Slice(2, query_start, query_end)})
                .squeeze(0);
        auto key_view_hsd = key_bhsd
                .index({Slice(0, batch_start, batch_end), Slice(2, key_start, key_end)})
                .squeeze(0)
                .repeat({nr, 1, 1});
        auto value_view_hsd = value_bhsd
                .index({Slice(0, batch_start, batch_end), Slice(3, key_start, key_end)})
                .squeeze(0)
                .repeat({nr, 1, 1});
        auto output_view_hsd = output_bhsd
                .index({Slice(0, batch_start, batch_end), Slice(2, query_start, query_end)})
                .squeeze(0);
        // clang-format on

        // create local tensors to speed-up computation
        ck_tile::HostTensor<QueryDataType> q_host_ref(query_view_hsd.get_lengths());
        ck_tile::HostTensor<KeyDataType> k_host_ref(key_view_hsd.get_lengths());
        ck_tile::HostTensor<ValueDataType> v_host_ref(value_view_hsd.get_lengths());
        // create local tensors for holding intermediate result
        ck_tile::HostTensor<SMPLComputeDataType> s_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<SMPLComputeDataType> lse_host_ref({nhead, real_seqlen_q});

        q_host_ref.for_each([&](auto& self, auto i) { self(i) = query_view_hsd(i); });
        k_host_ref.for_each([&](auto& self, auto i) { self(i) = key_view_hsd(i); });
        v_host_ref.for_each([&](auto& self, auto i) { self(i) = value_view_hsd(i); });

        // reference
        ck_tile::reference_batched_gemm<SaccDataType>(q_host_ref,
                                                      k_host_ref,
                                                      s_host_ref,
                                                      ck_tile::identity{},
                                                      ck_tile::identity{},
                                                      ck_tile::scales(scale_s));

        if(bias_bhss.has_value())
        {
            // clang-format off
            auto bias_view_hss = (*bias_bhss)
                    .index({Slice(2, query_start, query_end), Slice(3, key_start, key_end)})
                    .squeeze(0);
            // clang-format on

            // create local tensor to speed-up computation
            ck_tile::HostTensor<BiasDataType> bias_host_ref(bias_view_hss.get_lengths());
            bias_host_ref.for_each([&](auto& self, auto i) { self(i) = bias_view_hss(i); });

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

        if(lse_bhs.has_value())
        {
            // clang-format off
            auto lse_view_hs = (*lse_bhs)
                    .index({Slice(0, batch_start, batch_end), Slice(2, query_start, query_end)})
                    .squeeze(0);
            // clang-format on

            ck_tile::reference_batched_softmax<SMPLComputeDataType>(
                s_host_ref, p_host_ref, p_compute_element_func, lse_view_hs);
        }
        else
        {
            ck_tile::reference_batched_softmax<SMPLComputeDataType>(
                s_host_ref, p_host_ref, p_compute_element_func);
        }

        ck_tile::reference_batched_gemm<OaccDataType>(p_host_ref,
                                                      v_host_ref,
                                                      output_view_hsd,
                                                      ck_tile::identity{},
                                                      ck_tile::identity{},
                                                      oacc_element_func);
    }
}

} // namespace ck_tile
