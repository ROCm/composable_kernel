# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

from ck_tile_fmha import fmha_varlen_gdpa_func, gdpa_func, gdpa_infer, gdpa_varlen_func


def is_gfx125():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).gcnArchName.startswith(
        "gfx125"
    )


pytestmark = pytest.mark.skipif(not is_gfx125(), reason="GDPA is gfx1250 forward")


def offsets(lengths, device):
    return torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32, device=device
    )


def gdpa_ref(q, k, v, q_lens, k_lens, scale):
    """fp32 act(scale q k^T) v per batch, P rounded to q's dtype as the kernel rounds it."""
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    group = q.shape[1] // k.shape[1]
    qs, ks = 0, 0
    for lq, lk in zip(q_lens, k_lens):
        if lq and lk:
            qb = q[qs : qs + lq].float().transpose(0, 1)
            kb = k[ks : ks + lk].float().transpose(0, 1).repeat_interleave(group, 0)
            vb = v[ks : ks + lk].float().transpose(0, 1).repeat_interleave(group, 0)
            p = F.gelu(scale * qb @ kb.transpose(-1, -2), approximate="tanh").to(q.dtype).float()
            out[qs : qs + lq] = (p @ vb).transpose(0, 1)
        qs, ks = qs + lq, ks + lk
    return out


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def tolerance(dtype):
    return 2e-3 if dtype == torch.float16 else 1e-2


def jagged(q_lens, k_lens, nheads, nheads_k, dtype, d=128):
    q = torch.randn(sum(q_lens), nheads, d, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(sum(k_lens), nheads_k, d, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(sum(k_lens), nheads_k, d, device="cuda", dtype=dtype)
    return q, k, v


# GDPA is specialized to meta-tcx's calls: d128 (other head dims have no kernel)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize(
    "q_lens, k_lens",
    [
        ([5, 0, 70, 33], [256, 256, 256, 256]),  # short queries
        ([600, 517, 640], [256, 200, 300]),  # several query tiles per workgroup, partial key tiles
        ([1030, 2047, 129], [64, 33, 512]),  # one key tile, a partial one
        ([300, 129, 260], [1024, 2000, 0]),  # many keys, an empty key set
    ],
)
def test_gdpa_varlen(q_lens, k_lens, mha_type, dtype):
    torch.random.manual_seed(0)
    nheads = 6
    nheads_k = nheads if mha_type == "mha" else 2
    q, k, v = jagged(q_lens, k_lens, nheads, nheads_k, dtype)
    scale = 128**-0.5
    qo, ko = offsets(q_lens, "cuda"), offsets(k_lens, "cuda")
    out = gdpa_varlen_func(q, k, v, qo, ko, max(q_lens), max(k_lens), scale=scale)
    ref = gdpa_ref(q, k, v, q_lens, k_lens, scale)
    assert out.shape == q.shape and out.dtype == dtype
    assert rel_l2(out, ref) < tolerance(dtype)
    # rows of a batch with no keys are 0
    for i, lk in enumerate(k_lens):
        if lk == 0:
            start = sum(q_lens[:i])
            assert out[start : start + q_lens[i]].abs().max().item() == 0
    # gdpa_infer's group mode and the earlier name run the same kernel
    assert torch.equal(gdpa_infer(q, k, v, qo, ko, max(q_lens), max(k_lens), scale=scale), out)
    assert torch.equal(
        fmha_varlen_gdpa_func(q, k, v, qo, ko, max(q_lens), max(k_lens), scale=scale), out
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("b, s_q, s_k", [(3, 700, 256), (2, 33, 100), (1, 1024, 512)])
def test_gdpa_batch(b, s_q, s_k, dtype):
    torch.random.manual_seed(0)
    q = torch.randn(b, s_q, 4, 128, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(b, s_k, 2, 128, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(b, s_k, 2, 128, device="cuda", dtype=dtype)
    out = gdpa_func(q, k, v, scale=0.125)
    ref = gdpa_ref(
        q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), [s_q] * b, [s_k] * b, 0.125
    ).view(q.shape)
    assert out.shape == q.shape and out.dtype == dtype
    assert rel_l2(out, ref) < tolerance(dtype)
    assert torch.equal(gdpa_infer(q, k, v, scale=0.125), out)
    buf = torch.empty_like(q)
    assert gdpa_infer(q, k, v, scale=0.125, out=buf) is buf and torch.equal(buf, out)


@pytest.mark.parametrize("d", [64, 96, 112, 256])
def test_gdpa_other_head_dims_raise(d):
    q = torch.randn(10, 2, d, device="cuda", dtype=torch.bfloat16)
    qo = offsets([10], "cuda")
    with pytest.raises(NotImplementedError):
        gdpa_varlen_func(q, q, q, qo, qo, 10, 10)


def test_gdpa_rejects_unknown_activation():
    q = torch.randn(64, 2, 128, device="cuda", dtype=torch.bfloat16)
    lens = offsets([64], "cuda")
    with pytest.raises(RuntimeError, match="activation must be"):
        gdpa_varlen_func(q, q, q, lens, lens, 64, 64, activation="relu")
    with pytest.raises(ValueError, match="softmax is the fmha_"):
        gdpa_varlen_func(q, q, q, lens, lens, 64, 64, activation="softmax")
    with pytest.raises(ValueError, match="softmax is the fmha_"):
        gdpa_infer(q, q, q, lens, lens, 64, 64, activation="softmax")


def test_gdpa_is_forward_only():
    q = torch.randn(10, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(20, 2, 128, device="cuda", dtype=torch.bfloat16)
    qo, ko = offsets([10], "cuda"), offsets([20], "cuda")
    with pytest.raises(NotImplementedError, match="forward only"):
        gdpa_varlen_func(q, k, k, qo, ko, 10, 20)
    with pytest.raises(NotImplementedError, match="forward only"):
        gdpa_infer(q, k, k, qo, ko, 10, 20)
    with torch.no_grad():
        assert gdpa_varlen_func(q, k, k, qo, ko, 10, 20).shape == q.shape
