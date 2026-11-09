# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

from ck_tile_fmha import fmha_varlen_gdpa_func


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 256])  # the GDPA head dims
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize(
    "q_lens, k_lens",
    [
        ([5, 0, 70, 33], [256, 256, 256, 256]),  # short queries, few keys
        ([600, 517, 640], [256, 200, 300]),  # few keys: qr_tdm_pingpong
        ([300, 129, 260], [1024, 2000, 0]),  # many keys, an empty key set: qr_tdm_sched
        ([100, 64, 127], [1024, 777, 1500]),  # many keys, under 128 rows: qr_tdm
    ],
)
def test_gdpa_output(q_lens, k_lens, mha_type, d, dtype):
    torch.random.manual_seed(0)
    device = "cuda"
    nheads = 6
    nheads_k = nheads if mha_type == "mha" else 2
    q = torch.randn(sum(q_lens), nheads, d, device=device, dtype=dtype) * 0.5
    k = torch.randn(sum(k_lens), nheads_k, d, device=device, dtype=dtype) * 0.5
    v = torch.randn(sum(k_lens), nheads_k, d, device=device, dtype=dtype)
    scale = d**-0.5
    out = fmha_varlen_gdpa_func(
        q,
        k,
        v,
        offsets(q_lens, device),
        offsets(k_lens, device),
        max(q_lens),
        max(k_lens),
        scale=scale,
    )
    ref = gdpa_ref(q, k, v, q_lens, k_lens, scale)
    assert out.shape == q.shape and out.dtype == dtype
    assert rel_l2(out, ref) < (2e-3 if dtype == torch.float16 else 1e-2)
    # rows of a batch with no keys are 0
    empty = [i for i, lk in enumerate(k_lens) if lk == 0]
    for i in empty:
        start = sum(q_lens[:i])
        assert out[start : start + q_lens[i]].abs().max().item() == 0


def test_gdpa_rejects_unknown_activation():
    device = "cuda"
    q = torch.randn(64, 2, 128, device=device, dtype=torch.bfloat16)
    lens = offsets([64], device)
    with pytest.raises(RuntimeError, match="activation must be"):
        fmha_varlen_gdpa_func(q, q, q, lens, lens, 64, 64, activation="relu")
    with pytest.raises(ValueError, match="softmax is fmha_varlen_func"):
        fmha_varlen_gdpa_func(q, q, q, lens, lens, 64, 64, activation="softmax")


@pytest.mark.parametrize("d", [40, 200])
def test_gdpa_pads_other_head_dims(d):
    # every head dim up to 256 lands on a padded GDPA instance of the next head dim up
    torch.random.manual_seed(0)
    q = torch.randn(10, 2, d, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(20, 2, d, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(20, 2, d, device="cuda", dtype=torch.bfloat16)
    qo, ko = offsets([10], "cuda"), offsets([20], "cuda")
    out = fmha_varlen_gdpa_func(q, k, v, qo, ko, 10, 20, scale=d**-0.5)
    assert rel_l2(out, gdpa_ref(q, k, v, [10], [20], d**-0.5)) < 1e-2


def test_gdpa_is_forward_only():
    q = torch.randn(10, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(20, 2, 64, device="cuda", dtype=torch.bfloat16)
    qo, ko = offsets([10], "cuda"), offsets([20], "cuda")
    with pytest.raises(NotImplementedError, match="forward only"):
        fmha_varlen_gdpa_func(q, k, k, qo, ko, 10, 20)
    with torch.no_grad():
        assert fmha_varlen_gdpa_func(q, k, k, qo, ko, 10, 20).shape == q.shape
