# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch
from einops import repeat
from reference import (
    _generate_block_kvcache,
    attention_kvpacked_ref,
    attention_qkvpacked_ref,
    attention_ref,
    attn_bias_from_alibi_slopes,
    convert_fmha_S_to_softmax,
    generate_qkv,
    generate_random_padding_mask,
)
from utils import (
    ck_randval_to_dropout_mask,
    get_bwd_unsupported_reason,
    is_bwd_supported,
    pad_rearrange_dropout_mask_hts_to_bhss,
)

from ck_tile_fmha import (
    fmha_varlen_func,
    fmha_varlen_kvpacked_func,
    fmha_varlen_qkvpacked_func,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 257, 384, 512, 768, 1025, 2048])
@pytest.mark.parametrize("dropout_p", [0, 0.17])
def test_fmha_varlen_qkvpacked(seqlen, d, dropout_p, causal, local, alibi, deterministic, dtype):
    if d > 256:
        pytest.skip()

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 6
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen, seqlen, key_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        *qkv.unbind(dim=2), key_padding_mask, key_padding_mask, qkvpacked=True
    )

    out_unpad, sm_lse, S_dmask = fmha_varlen_qkvpacked_func(
        qkv_unpad,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        # TODO - move to c++ mha_varlen_fwd()
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask = pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens, seqlen, seqlen)

        S_dmask_converted = convert_fmha_S_to_softmax(
            S_dmask,
            seqlen,
            seqlen,
            key_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )

        dropout_mask = S_dmask_converted >= 0
        # CK does not return P. Hence, we don't test the attn here.
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(
        qkv,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_qkvpacked_ref(
        qkv,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FMHA's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    g = torch.randn_like(out)
    if is_bwd_supported(d, deterministic):
        (dqkv_unpad,) = torch.autograd.grad(out, qkv_unpad, g)
        dqkv = dqkv_pad_fn(dqkv_unpad)
        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
        (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
        print(f"dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

        # TODO - use 10 times to check, wait for ck to fix bwd precision issue
        assert (dqkv - dqkv_ref).abs().max().item() <= 10 * (dqkv_pt - dqkv_ref).abs().max().item()


@pytest.mark.parametrize("kvpacked", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_fmha_varlen_output(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked
):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        ) = generate_qkv(q, *kv.unbind(dim=2), query_padding_mask, key_padding_mask, kvpacked=True)
        out_unpad, sm_lse, S_dmask = fmha_varlen_kvpacked_func(
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        out_unpad, sm_lse, S_dmask = fmha_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        # TODO - move to c++ mha_varlen_fwd()
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask = pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens_q, seqlen_q, seqlen_k)
        S_dmask_converted = convert_fmha_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        # CK does not return P. Hence, we don't test the attn here.
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FMHA's numerical error is at most 4 times the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 4 * (out_pt - out_ref).abs().max().item()

    g = torch.randn_like(out)
    if is_bwd_supported(d, deterministic):
        if kvpacked:
            (
                dq_unpad,
                dkv_unpad,
            ) = torch.autograd.grad(out, (q_unpad, kv_unpad), g)
            dk, dv = dkv_pad_fn(dkv_unpad).unbind(2)
            (
                dq_ref,
                dkv_ref,
            ) = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            (
                dq_pt,
                dkv_pt,
            ) = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        else:
            (
                dq_unpad,
                dk_unpad,
                dv_unpad,
            ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            (
                dq_ref,
                dk_ref,
                dv_ref,
            ) = torch.autograd.grad(out_ref, (q, k, v), g)
            (
                dq_pt,
                dk_pt,
                dv_pt,
            ) = torch.autograd.grad(out_pt, (q, k, v), g)
        dq = dq_pad_fn(dq_unpad)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

        # TODO - use 10 times to check, wait for ck to fix bwd precision issue
        assert (dq - dq_ref).abs().max().item() <= 10 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 10 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 10 * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
@pytest.mark.parametrize("paged_kv_block_size", [None, 256, 512])
def test_fmha_varlen_causal(seqlen_q, seqlen_k, swap_sq_sk, d, local, paged_kv_block_size, dtype):
    if max(seqlen_q, seqlen_k) >= 2048:
        pytest.skip()
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    causal = True
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)

    if paged_kv_block_size is None:
        k = torch.randn(
            batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        block_table = None
    else:
        k, v, block_table, k_cache_paged, v_cache_paged, num_blocks = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads, d, device, dtype
        )
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    out_unpad = fmha_varlen_func(
        q_unpad,
        k_unpad if paged_kv_block_size is None else k_cache_paged,
        v_unpad if paged_kv_block_size is None else v_cache_paged,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    out = output_pad_fn(out_unpad)
    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FMHA's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    g = torch.randn_like(out)
    test_backward = block_table is None and is_bwd_supported(d, deterministic=False)
    if test_backward:
        do_o = (g.float() * out.float()).sum(-1)
        (
            dq_unpad,
            dk_unpad,
            dv_unpad,
        ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

        # TODO - use 10 times to check, wait for ck to fix bwd precision issue
        assert (dq - dq_ref).abs().max().item() <= 10 * (dq_pt - dq_ref).abs().max().item() + 1e-5
        assert (dk - dk_ref).abs().max().item() <= 10 * (dk_pt - dk_ref).abs().max().item() + 1e-5
        assert (dv - dv_ref).abs().max().item() <= 10 * (dv_pt - dv_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [16, 32, 64])
def test_fmha_bwd_varlen_overflow(d, causal, dtype):
    """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ,
    in the case where seqlen % 128 != 0 or varlen.
    """
    if not is_bwd_supported(d, deterministic=False):
        pytest.skip(get_bwd_unsupported_reason(d, deterministic=False))

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    nheads = 5
    q_cuseqlen = torch.tensor([0, 76, 110, 256], device=device, dtype=torch.int32)
    k_cuseqlen = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    Mq = 256
    Mk = 3

    q = torch.randn([Mq, nheads, d], dtype=dtype, device=device) * 3
    k, v = [torch.randn([Mk, nheads, d], dtype=dtype, device=device) * 3 for _ in range(2)]
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = fmha_varlen_func(q, k, v, q_cuseqlen, k_cuseqlen, Mq, Mk, causal=causal)
    g = torch.randn_like(out)
    out.backward(g)

    assert not q.grad.isnan().any()
    assert not k.grad.isnan().any()
    assert not v.grad.isnan().any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("nheads_kv", [1, 8])  # GQA (1) and MHA (8)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
def test_fmha_bwd_varlen_seqq_zero(d, causal, nheads_kv, deterministic, dtype):
    """Regression test: NaN in dK/dV for the zero-length Q subsequence (run for both GQA and MHA)."""
    if not is_bwd_supported(d, deterministic=deterministic):
        pytest.skip(get_bwd_unsupported_reason(d, deterministic=deterministic))

    device = "cuda"
    torch.random.manual_seed(0)
    nheads = 8  # n_q_heads; GQA when nheads_kv < nheads
    q_cuseqlen = torch.tensor([0, 0, 256, 512], device=device, dtype=torch.int32)
    k_cuseqlen = torch.tensor([0, 503, 768, 1536], device=device, dtype=torch.int32)
    total_q = int(q_cuseqlen[-1])  # 512
    total_k = int(k_cuseqlen[-1])  # 1536
    Mq = 256
    Mk = 768

    q = torch.randn([total_q, nheads, d], dtype=dtype, device=device)
    k = torch.randn([total_k, nheads_kv, d], dtype=dtype, device=device)
    v = torch.randn([total_k, nheads_kv, d], dtype=dtype, device=device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = fmha_varlen_func(
        q, k, v, q_cuseqlen, k_cuseqlen, Mq, Mk, causal=causal, deterministic=deterministic
    )
    g = torch.randn_like(out)
    out.backward(g)

    # The bug produced NaN specifically in dK/dV for the zero-length Q subsequence
    assert not q.grad.isnan().any()
    assert not k.grad.isnan().any()
    assert not v.grad.isnan().any()

    # Additionally, for the batch with seqlen_q == 0, the corresponding K/V segment
    # should contribute nothing to the loss, so dK/dV for that segment must be zero.
    end_kv_zero = int(k_cuseqlen[1])
    assert k.grad[:end_kv_zero].abs().max() == 0
    assert v.grad[:end_kv_zero].abs().max() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
def test_fmha_varlen_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    out = fmha_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=causal,
        window_size=window_size,
        deterministic=True,
    )

    if not is_bwd_supported(d, deterministic=True):
        pytest.skip(get_bwd_unsupported_reason(d, deterministic=True))

    g = torch.randn_like(out)
    dq0, dk0, dv0 = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
    for _ in range(50):
        dq, dk, dv = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
        assert torch.equal(dv, dv0)
        assert torch.equal(dk, dk0)
        assert torch.equal(dq, dq0)
