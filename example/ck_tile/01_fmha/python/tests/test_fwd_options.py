# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Forward-only options of fmha_func and fmha_varlen_func: broadcast_q and lse_log2."""

import pytest
import torch

from ck_tile_fmha import fmha_func, fmha_varlen_func

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

LOG2E = 1.4426950408889634


def offsets(lengths, device="cuda"):
    return torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32, device=device
    )


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("lq, k_lens", [(40, [300, 0, 33, 129]), (512, [5252, 5253, 1000])])
def test_broadcast_q(lq, k_lens, d, dtype):
    """One Q block for every batch: the same as Q copied per batch, without the copy."""
    torch.random.manual_seed(0)
    batch, nheads = len(k_lens), 3
    q = torch.randn(lq, nheads, d, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(sum(k_lens), nheads, d, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(sum(k_lens), nheads, d, device="cuda", dtype=dtype)
    k_off = offsets(k_lens)
    # cu_seqlens_q only gives the batch size: the kernel reads no seqstart for Q
    q_off = torch.full((batch + 1,), -7, dtype=torch.int32, device="cuda")
    out, lse, _ = fmha_varlen_func(
        q, k, v, q_off, k_off, lq, max(k_lens), return_attn_probs=True, broadcast_q=True
    )
    out_copy, lse_copy, _ = fmha_varlen_func(
        q.repeat(batch, 1, 1), k, v, offsets([lq] * batch), k_off, lq, max(k_lens),
        return_attn_probs=True,
    )  # fmt: skip
    assert out.shape == (batch * lq, nheads, d)
    assert lse.shape == (nheads, batch * lq)
    torch.testing.assert_close(out, out_copy, rtol=0, atol=0)
    torch.testing.assert_close(lse, lse_copy, rtol=0, atol=0)
    # and against an fp32 softmax reference on a batch with keys
    i = next(i for i, lk in enumerate(k_lens) if lk)
    ks = sum(k_lens[:i])
    s = torch.einsum("qhd,khd->hqk", q.float(), k[ks : ks + k_lens[i]].float()) * d**-0.5
    ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), v[ks : ks + k_lens[i]].float())
    assert rel_l2(out[i * lq : (i + 1) * lq], ref) < 1e-2


def test_broadcast_q_is_forward_only():
    q = torch.randn(16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(32, 2, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        fmha_varlen_func(q, k, k, offsets([16, 16]), offsets([16, 16]), 16, 16, broadcast_q=True)


def test_broadcast_q_rejects_paged_kv():
    q = torch.randn(16, 2, 64, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(2, 128, 2, 64, device="cuda", dtype=torch.bfloat16)
    block_table = torch.arange(2, dtype=torch.int32, device="cuda").view(2, 1)
    with pytest.raises(NotImplementedError, match="paged KV"):
        fmha_varlen_func(
            q, k_cache, k_cache, offsets([16, 16]), offsets([100, 128]), 16, 128,
            block_table=block_table, broadcast_q=True,
        )  # fmt: skip


@pytest.mark.parametrize("d", [64, 128, 160, 256])
@pytest.mark.parametrize("kv_shared", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_lse_log2(dropout_p, kv_shared, d):
    """The LSE in base 2: the natural-log LSE times log2(e), with the same output. The gfx1250
    qr_tdm kernels store it; the others (here: dropout, head dims without them) store the
    natural log and the binding rescales it."""
    torch.random.manual_seed(1)
    q = torch.randn(64, 16, 1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(64, 200, 1, d, device="cuda", dtype=torch.bfloat16)
    v = k if kv_shared else torch.randn_like(k)

    def run(**kw):
        torch.manual_seed(5)  # the same dropout mask for both runs
        return fmha_func(q, k, v, dropout_p=dropout_p, return_attn_probs=True, **kw)

    out_e, lse_e, _ = run()
    out_2, lse_2, _ = run(lse_log2=True)
    assert torch.equal(out_e, out_2)
    torch.testing.assert_close(lse_2, lse_e * LOG2E, rtol=1e-6, atol=1e-6)

    lens_q, lens_k = [5, 70, 33], [256, 0, 300]
    qv = torch.randn(sum(lens_q), 2, d, device="cuda", dtype=torch.bfloat16)
    kvv = torch.randn(sum(lens_k), 2, d, device="cuda", dtype=torch.bfloat16)
    qo, ko = offsets(lens_q), offsets(lens_k)
    out_e, lse_e, _ = fmha_varlen_func(qv, kvv, kvv, qo, ko, 70, 300, return_attn_probs=True)
    out_2, lse_2, _ = fmha_varlen_func(
        qv, kvv, kvv, qo, ko, 70, 300, return_attn_probs=True, lse_log2=True
    )
    assert torch.equal(out_e, out_2)
    torch.testing.assert_close(lse_2, lse_e * LOG2E, rtol=1e-6, atol=1e-6)


def test_lse_log2_paged_kv():
    """The split-KV kernels behind a block table store the natural log; the binding rescales."""
    torch.random.manual_seed(2)
    lens_q, lens_k, page = [3, 17], [200, 128], 128
    q = torch.randn(sum(lens_q), 2, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(4, page, 2, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    args = (q, k_cache, v_cache, offsets(lens_q), offsets(lens_k), 17, 256)
    out_e, lse_e, _ = fmha_varlen_func(*args, block_table=block_table, return_attn_probs=True)
    out_2, lse_2, _ = fmha_varlen_func(
        *args, block_table=block_table, return_attn_probs=True, lse_log2=True
    )
    assert torch.equal(out_e, out_2)
    torch.testing.assert_close(lse_2, lse_e * LOG2E, rtol=1e-6, atol=1e-6)


def test_lse_log2_is_forward_only():
    q = torch.randn(2, 16, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="forward only"):
        fmha_func(q, k, k, lse_log2=True)
