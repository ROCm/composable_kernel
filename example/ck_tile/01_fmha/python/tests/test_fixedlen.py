# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch
from einops import rearrange, repeat
from reference import (
    attention_kvpacked_ref,
    attention_qkvpacked_ref,
    attention_ref,
    attn_bias_from_alibi_slopes,
    convert_fmha_S_to_softmax,
)
from utils import (
    ck_randval_to_dropout_mask,
    get_bwd_unsupported_reason,
    is_bwd_supported,
)

from ck_tile_fmha import (
    fmha_func,
    fmha_kvpacked_func,
    fmha_qkvpacked_func,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 384, 768, 1024, 1025, 2048])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_fmha_qkvpacked(seqlen, d, dropout_p, causal, local, alibi, deterministic, dtype):
    if d > 256:
        pytest.skip()

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen, seqlen, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None
    out, lse, S_dmask = fmha_qkvpacked_func(
        qkv,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    if dropout_p > 0.0:
        # TODO - move to c++ mha_varlen_fwd()
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask_converted = convert_fmha_S_to_softmax(
            S_dmask,
            seqlen,
            seqlen,
            None,
            None,
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
        qkv, None, attn_bias, dropout_p, dropout_mask, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_qkvpacked_ref(
        qkv,
        None,
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
        (dqkv,) = torch.autograd.grad(out, qkv, g)
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
@pytest.mark.parametrize("d", [32, 40, 59, 64, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
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
def test_fmha_output(
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
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        out, lse, S_dmask = fmha_kvpacked_func(
            q,
            kv,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        out, lse, S_dmask = fmha_func(
            q,
            k,
            v,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    if dropout_p > 0.0:
        # TODO - move to c++ mha_varlen_fwd()
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask_converted = convert_fmha_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            None,
            None,
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
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
        if kvpacked:
            (
                dq,
                dkv,
            ) = torch.autograd.grad(out, (q, kv), g)
            dk, dv = dkv.unbind(2)
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
                dq,
                dk,
                dv,
            ) = torch.autograd.grad(out, (q, k, v), g)
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
def test_fmha_causal(seqlen_q, seqlen_k, swap_sq_sk, d, local, dtype):
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
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out = fmha_func(q, k, v, 0.0, causal=causal, window_size=window_size)
    out_ref, attn_ref = attention_ref(
        q, k, v, None, None, None, 0.0, None, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
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

    # Check that FMHA's numerical error is at most 4 times the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 4 * (out_pt - out_ref).abs().max().item() + 1e-5

    g = torch.randn_like(out)
    if is_bwd_supported(d, deterministic=False):
        do_o = (g.float() * out.float()).sum(-1)
        (
            dq,
            dk,
            dv,
        ) = torch.autograd.grad(out, (q, k, v), g)
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
        assert (dq - dq_ref).abs().max().item() <= 10 * (dq_pt - dq_ref).abs().max().item() + 1e-4
        assert (dk - dk_ref).abs().max().item() <= 10 * (dk_pt - dk_ref).abs().max().item() + 1e-4
        assert (dv - dv_ref).abs().max().item() <= 10 * (dv_pt - dv_ref).abs().max().item() + 1e-4


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (1024, 128),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
        (384, 384),
        (512, 512),
        (768, 768),
        # (1024, 1024),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_fmha_race_condition(seqlen_q, seqlen_k, d, dropout_p, causal, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    torch.random.manual_seed(42)
    out0, lse0, _ = fmha_func(q, k, v, dropout_p, causal=causal, return_attn_probs=True)
    g = torch.randn_like(out0)
    test_backward = dropout_p == 0 and is_bwd_supported(d, deterministic=False)
    if test_backward:
        (
            dq0,
            dk0,
            dv0,
        ) = torch.autograd.grad(out0, (q, k, v), g)
        # Numerical error if we just do any arithmetic on dq
        dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

    for i in range(250):
        torch.random.manual_seed(42)
        out, lse, _ = fmha_func(q, k, v, dropout_p, causal=causal, return_attn_probs=True)
        assert torch.equal(out, out0)
        assert torch.equal(lse, lse0)

        if test_backward:
            (
                dq,
                dk,
                dv,
            ) = torch.autograd.grad(out, (q, k, v), g)
            dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
            if not dq_equal:
                print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")

            assert torch.equal(dv, dv0)
            assert torch.equal(dk, dk0)
            assert dq_equal


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [16, 32, 64])
@pytest.mark.parametrize("seqlen", [1, 2, 5, 17, 128])
def test_fmha_bwd_overflow(seqlen, d, causal, dtype):
    """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ,
    in the case where seqlen % 128 != 0.
    """

    # TODO - 1 or 2 might fail, need to check
    if seqlen == 1 or seqlen == 2:
        pytest.skip()
    if not is_bwd_supported(d, deterministic=False):
        pytest.skip(get_bwd_unsupported_reason(d, deterministic=False))

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 5
    q = torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 5
    k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 3
        for _ in range(2)
    ]
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = fmha_func(q, k, v, causal=causal)
    g = torch.randn_like(out)
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt, _ = attention_ref(q_pt, k_pt, v_pt, causal=causal, upcast=False, reorder_ops=True)
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref, attn_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 7 * (
        q_pt.grad - q_ref.grad
    ).abs().max().item() + 1e-3
    assert (k.grad - k_ref.grad).abs().max().item() <= 5 * (
        k_pt.grad - k_ref.grad
    ).abs().max().item() + 1e-3
    assert (v.grad - v_ref.grad).abs().max().item() <= 5 * (
        v_pt.grad - v_ref.grad
    ).abs().max().item() + 1e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 256])
def test_fmha_bwd_transpose(seqlen, d, causal, dtype):
    """We previously had a bug where we were using the wrong strides of dout, which shows up
    when dout is not contiguous.
    """
    if not is_bwd_supported(d, deterministic=False):
        pytest.skip(get_bwd_unsupported_reason(d, deterministic=False))

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 2
    q, k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    out = rearrange(fmha_func(q, k, v, causal=causal), "b s ... -> s b ...")
    # So g is not contiguous
    g = torch.randn(seqlen, 2 * batch_size, nheads, d, dtype=dtype, device="cuda")[:, ::2]
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt, attn_pt = attention_ref(q_pt, k_pt, v_pt, causal=causal, upcast=False, reorder_ops=True)
    out_pt = rearrange(out_pt, "b s ... -> s b ...")
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref, attn_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_ref = rearrange(out_ref, "b s ... -> s b ...")
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 2 * (
        q_pt.grad - q_ref.grad
    ).abs().max().item()
    assert (k.grad - k_ref.grad).abs().max().item() <= 2 * (
        k_pt.grad - k_ref.grad
    ).abs().max().item()
    assert (v.grad - v_ref.grad).abs().max().item() <= 2 * (
        v_pt.grad - v_ref.grad
    ).abs().max().item()


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
def test_fmha_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
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
    batch_size = 4
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out = fmha_func(q, k, v, 0.0, causal=causal, window_size=window_size, deterministic=True)

    if not is_bwd_supported(d, deterministic=True):
        pytest.skip(get_bwd_unsupported_reason(d, deterministic=True))

    g = torch.randn_like(out)
    dq0, dk0, dv0 = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
    for _ in range(50):
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
        assert torch.equal(dv, dv0)
        assert torch.equal(dk, dk0)
        assert torch.equal(dq, dq0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("api", ["separate", "kvpacked", "qkvpacked"])
@pytest.mark.parametrize("mask", ["none", "causal", "local"])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("bias_grad", [False, True])
@pytest.mark.parametrize("sq,sk", [(97, 113), (16, 33), (32, 65)])
def test_bias_forward_backward(dtype, api, mask, dropout_p, bias_grad, sq, sk):
    torch.manual_seed(0)
    h, hk, d = 4, 2, 64
    if api == "qkvpacked":
        sk, hk = sq, h
    q = torch.randn(2, sq, h, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, sk, hk, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    # A row-strided view checks that stride_bias is forwarded, including in backward.
    bias = torch.randn(sq * 2, sk, device="cuda", dtype=dtype)[::2]
    bias.requires_grad_(bias_grad)
    causal, window = mask == "causal", (17, 9) if mask == "local" else (-1, -1)
    scale = 0.23  # Deliberately differs from 1/sqrt(d), including for dbias.
    kwargs = dict(
        bias=bias,
        causal=causal,
        window_size=window,
        softmax_scale=scale,
        dropout_p=dropout_p,
        return_attn_probs=True,
    )
    if api == "separate":
        out, lse, randval = fmha_func(q, k, v, **kwargs)
    elif api == "kvpacked":
        out, lse, randval = fmha_kvpacked_func(q, torch.stack((k, v), dim=2), **kwargs)
    else:
        out, lse, randval = fmha_qkvpacked_func(torch.stack((q, k, v), dim=2), **kwargs)
    dropout_mask = ck_randval_to_dropout_mask(randval, dropout_p) >= 0 if dropout_p else None
    out_ref, _, lse_ref = attention_ref(
        q.float(),
        k.float(),
        v.float(),
        attn_bias=bias.float(),
        softmax_scale=scale,
        causal=causal,
        window_size=window,
        dropout_p=dropout_p,
        dropout_mask=dropout_mask,
        return_lse=True,
    )
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(out.float(), out_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(lse, lse_ref, atol=2e-3, rtol=2e-3)
    dout = torch.randn_like(out)
    inputs = (q, k, v, bias) if bias_grad else (q, k, v)
    grads = torch.autograd.grad(out, inputs, dout)
    refs = torch.autograd.grad(out_ref, inputs, dout.float())
    for actual, expected in zip(grads[:3], refs[:3]):
        torch.testing.assert_close(actual, expected, atol=tol * 4, rtol=tol * 4)
    if bias_grad:
        # CK computes the softmax derivative using D = sum(dout * out), where
        # out has already been rounded to FP16/BF16. For each bias element the
        # resulting error is bounded by sum_bh(P * sum_d(abs(dout * delta_out))).
        # P <= 1 gives a row-wise bound, including the reduction across heads and
        # batches. A one-key causal row exposes this even though its true dbias
        # is zero. Keep the same tolerance for error beyond output quantization.
        rounding_bound = (
            (dout.float().abs() * (out.detach().float() - out_ref.detach()).abs())
            .sum(dim=-1)
            .sum(dim=(0, 2))
        )
        error = (grads[3].float() - refs[3].float()).abs()
        allowed = tol * 4 * (1 + refs[3].float().abs()) + rounding_bound[:, None]
        assert torch.all(error <= allowed), f"dbias error/bound: {(error / allowed).max().item()}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "sq,sk,d,hk",
    [
        (1, 113, 64, 1),
        (113, 97, 59, 2),
        (64, 64, 128, 4),
        (32, 16, 64, 2),
        (32, 33, 64, 2),
        (32, 129, 64, 2),
        (16, 33, 128, 2),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_bias_shapes_and_determinism(dtype, sq, sk, d, hk, deterministic, causal):
    torch.manual_seed(1)
    q = torch.randn(2, sq, 4, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, sk, hk, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    bias = torch.randn(sq, sk, device="cuda", dtype=dtype, requires_grad=True)
    out, lse, randval = fmha_func(
        q, k, v, bias=bias, causal=causal, deterministic=deterministic, return_attn_probs=True
    )
    ref, _, lse_ref = attention_ref(
        q.float(), k.float(), v.float(), attn_bias=bias.float(), causal=causal, return_lse=True
    )
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)
    torch.testing.assert_close(lse, lse_ref, atol=2e-3, rtol=2e-3)
    dout = torch.randn_like(out)
    inputs = (q, k, v, bias)
    grads = torch.autograd.grad(out, inputs, dout, retain_graph=deterministic)
    refs = torch.autograd.grad(ref, inputs, dout.float())
    for actual, expected in zip(grads, refs):
        torch.testing.assert_close(actual, expected, atol=tol * 4, rtol=tol * 4)
    if deterministic:
        repeated = torch.autograd.grad(out, inputs, dout)
        for actual, other in zip(grads, repeated):
            assert torch.equal(actual, other)


@pytest.mark.parametrize("api", ["separate", "kvpacked", "qkvpacked"])
def test_only_bias_requires_grad(api):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 32, 2, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    bias = torch.randn(32, 32, device="cuda", dtype=q.dtype, requires_grad=True)
    if api == "separate":
        out = fmha_func(q, k, v, bias=bias)
    elif api == "kvpacked":
        out = fmha_kvpacked_func(q, torch.stack((k, v), dim=2), bias=bias)
    else:
        out = fmha_qkvpacked_func(torch.stack((q, k, v), dim=2), bias=bias)
    ref, _ = attention_ref(q.float(), k.float(), v.float(), attn_bias=bias.float())
    (actual,) = torch.autograd.grad(out.sum(), (bias,))
    (expected,) = torch.autograd.grad(ref.sum(), (bias,))
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    with torch.no_grad():
        assert not fmha_func(q, k, v, bias=bias).requires_grad
