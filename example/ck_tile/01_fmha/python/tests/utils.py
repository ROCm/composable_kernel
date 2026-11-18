# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared helpers for CK attention validation and dropout buffers."""

import math

import torch
import torch.nn.functional as F


def is_bwd_hdim_supported(d):
    return d <= 256


def is_bwd_supported(d, deterministic):
    if not is_bwd_hdim_supported(d):
        return False

    return True


def get_bwd_unsupported_reason(d, deterministic):
    if is_bwd_hdim_supported(d) is False:
        return f"CK backward is not supported for head dim {d}."

    return "CK backward is not supported on this arch/configuration."


def ck_randval_to_dropout_mask(randval, p):
    # If p = 0.3, randval in 255 * (0.7, 1.0] will be dropout
    # randval in 255 * [0, 0.7] will be kept
    # If return dropout_mask >=0, value will be kept
    return math.floor(255.0 * (1 - p)) - randval.to(torch.float32)


def pad_rearrange_dropout_mask_hts_to_bhss(
    S_dmask, cu_seqlens_q, seqlen_q_rounded, seqlen_k_rounded
):
    """pad + rearrange [nheads, total_q, max_seqlen_k] into [b, nheads, seqlen_q_rounded, seqlen_k_rounded]
    Arguments:
        S_dmask: (nheads, total_q, max_seqlen_k)
        cu_seqlens_q: (b + 1)
    Output:
        S_dmask: (b, nheads, seqlen_q_rounded, seqlen_k_rounded)
    """
    batch_size = cu_seqlens_q.numel() - 1
    seqlens_q = torch.roll(cu_seqlens_q, shifts=-1) - cu_seqlens_q
    seqlens_q = seqlens_q[0:batch_size].tolist()
    S_dmask = torch.split(S_dmask, seqlens_q, dim=1)
    # [(nheads, seqlen_q0, max_seqlen_k), (nheads, seqlen_q1, max_seqlen_k), ..., (nheads, seqlen_qb, max_seqlen_k)]
    masks = ()
    for mask in S_dmask:
        # (nheads, seqlen_qi, max_seqlen_k) -> (nheads, seqlen_q_rounded, seqlen_k_rounded)
        mask = F.pad(
            mask, (0, seqlen_k_rounded - mask.shape[2], 0, seqlen_q_rounded - mask.shape[1], 0, 0)
        ).unsqueeze(1)
        masks = masks + (mask,)
    S_dmask = torch.cat(masks, dim=1)

    S_dmask = S_dmask.transpose(0, 1)
    return S_dmask
