# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

from ck_tile_fmha import fmha_func, fmha_infer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

LOG2E = 1.4426950408889634


def heads_last(t):
    """[b, s, d] as one head, [b, s, 1, d], the layout fmha_func takes."""
    return t.unsqueeze(2) if t.dim() == 3 else t


def make_qkv(layout, d, dtype, sq, sk, b=2):
    h, hk = {"mha": (4, 4), "gqa": (6, 2), "one_head": (1, 1)}[layout]
    q = torch.randn(b, sq, h, d, device="cuda", dtype=dtype)
    k = torch.randn(b, sk, hk, d, device="cuda", dtype=dtype)
    v = torch.randn(b, sk, hk, d, device="cuda", dtype=dtype)
    if layout == "one_head":
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
    return q, k, v


def fmha_func_ref(q, k, v, **kw):
    """fmha_func, the training API on the same kernels, with fmha_infer's layout."""
    out, lse, _ = fmha_func(
        heads_last(q), heads_last(k), heads_last(v), return_attn_probs=True, **kw
    )
    if q.dim() == 3:
        return out.squeeze(2), lse.squeeze(1)
    return out, lse


def fp32_ref(q, k, v, scale, causal=False, bias=None):
    """fp32 softmax(scale q k^T + bias) v, causal bottom-right aligned. [b, s, h, d] inputs."""
    group = q.shape[2] // k.shape[2]
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(group, 1)
    vf = v.float().transpose(1, 2).repeat_interleave(group, 1)
    s = scale * qf @ kf.transpose(-1, -2)
    if bias is not None:
        s = s + bias.float()
    if causal:
        sq, sk = s.shape[-2:]
        row = torch.arange(sq, device=q.device)[:, None] + (sk - sq)
        s = s.masked_fill(torch.arange(sk, device=q.device)[None, :] > row, float("-inf"))
    return (torch.softmax(s, -1) @ vf).transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("layout", ["mha", "gqa", "one_head"])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
@pytest.mark.parametrize("sq, sk", [(1, 77), (16, 200), (300, 333), (128, 128)])
def test_infer_matches_fmha_func(sq, sk, mask, layout, d, dtype):
    torch.random.manual_seed(0)
    q, k, v = make_qkv(layout, d, dtype, sq, sk)
    kw = {"causal": mask == "causal", "window_size": (64, 16) if mask == "window" else (-1, -1)}
    out, lse = fmha_infer(q, k, v, return_lse=True, **kw)
    ref_out, ref_lse = fmha_func_ref(q, k, v, **kw)
    assert out.shape == q.shape and out.dtype == dtype
    assert lse.shape == ref_lse.shape and lse.dtype == torch.float32
    torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-5)
    # without the LSE: the kernels that skip it give the same output
    torch.testing.assert_close(fmha_infer(q, k, v, **kw), ref_out, atol=2e-3, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["mha", "gqa"])
@pytest.mark.parametrize("bias_kind", ["alibi_h", "alibi_bh", "elementwise"])
def test_infer_bias(bias_kind, layout, dtype):
    torch.random.manual_seed(0)
    b, sq, sk = 2, 300, 333
    q, k, v = make_qkv(layout, 128, dtype, sq, sk, b)
    h = q.shape[2]
    if bias_kind == "elementwise":
        kw = {"bias": torch.randn(sq, sk, device="cuda", dtype=dtype)}
    else:
        shape = (h,) if bias_kind == "alibi_h" else (b, h)
        kw = {"alibi_slopes": torch.rand(shape, device="cuda", dtype=torch.float32)}
    out, lse = fmha_infer(q, k, v, return_lse=True, **kw)
    ref_out, ref_lse = fmha_func_ref(q, k, v, **kw)
    torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(fmha_infer(q, k, v, **kw), ref_out, atol=2e-3, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_infer_fp32_reference(with_bias, causal):
    torch.random.manual_seed(0)
    q, k, v = make_qkv("gqa", 128, torch.bfloat16, 200, 250)
    bias = torch.randn(200, 250, device="cuda", dtype=torch.bfloat16) if with_bias else None
    scale = 0.1
    out = fmha_infer(q, k, v, softmax_scale=scale, causal=causal, bias=bias)
    torch.testing.assert_close(
        out.float(), fp32_ref(q, k, v, scale, causal, bias), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("sk", [1, 5])
def test_infer_right_window_longer_than_keys(sk):
    """More queries than keys: a right window of seqlen_k or more keys still masks the first
    rows (bottom-right aligned), which then attend to nothing."""
    torch.random.manual_seed(0)
    q, k, v = make_qkv("mha", 64, torch.float16, 16, sk)
    wl, wr = 40, 5
    out = fmha_infer(q, k, v, window_size=(wl, wr))
    s = 0.125 * q.float().transpose(1, 2) @ k.float().permute(0, 2, 3, 1)
    row = torch.arange(16, device="cuda")[:, None] + (sk - 16)
    col = torch.arange(sk, device="cuda")[None, :]
    s = s.masked_fill((col > row + wr) | (col < row - wl), float("-inf"))
    ref = (torch.softmax(s, -1).nan_to_num(0.0) @ v.float().transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=0)


@pytest.mark.parametrize("d", [128, 256])
def test_infer_shared_kv_lse_log2(d):
    """One tensor as K and V in the head-less layout: qr_tdm_kvshared's call on gfx1250."""
    torch.random.manual_seed(0)
    q = torch.randn(64, 16, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(64, 200, d, device="cuda", dtype=torch.bfloat16)
    out, lse2 = fmha_infer(q, kv, kv, return_lse=True, lse_log2=True)
    ref_out, ref_lse = fmha_func_ref(q, kv, kv.clone())
    assert lse2.shape == (64, 16)
    # K is V picks qr_tdm_kvshared on gfx1250, the clone another kernel: bf16 rounding apart
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(lse2, ref_lse * LOG2E, atol=1e-4, rtol=1e-5)


def test_infer_out_argument():
    torch.random.manual_seed(0)
    q, k, v = make_qkv("mha", 128, torch.float16, 40, 90)
    out = torch.empty_like(q)
    res = fmha_infer(q, k, v, out=out)
    assert res.data_ptr() == out.data_ptr()
    torch.testing.assert_close(res, fmha_func_ref(q, k, v)[0], atol=2e-3, rtol=0)


def test_infer_strided_inputs():
    """Any batch, sequence and head strides; a non-contiguous head dim is copied."""
    torch.random.manual_seed(0)
    qkv = torch.randn(2, 100, 3, 4, 64, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv.unbind(2)  # sequence stride 3 * 4 * 64
    ref = fmha_func_ref(q.contiguous(), k.contiguous(), v.contiguous())[0]
    torch.testing.assert_close(fmha_infer(q, k, v), ref, atol=2e-3, rtol=0)
    qt = q.transpose(-1, -2).contiguous().transpose(-1, -2)  # head dim stride != 1
    torch.testing.assert_close(fmha_infer(qt, k, v), ref, atol=2e-3, rtol=0)


def test_infer_empty_keys():
    q, k, v = make_qkv("mha", 64, torch.float16, 8, 0)
    out, lse = fmha_infer(q, k, v, return_lse=True)
    assert torch.all(out == 0) and torch.all(lse == math.inf)


@pytest.mark.parametrize("grad_input", ["q", "bias", "alibi_slopes"])
def test_infer_is_forward_only(grad_input):
    q, k, v = make_qkv("mha", 64, torch.float16, 8, 8)
    kw = {}
    if grad_input == "q":
        q.requires_grad_()
    elif grad_input == "bias":
        kw["bias"] = torch.randn(8, 8, device="cuda", dtype=torch.float16, requires_grad=True)
    else:
        kw["alibi_slopes"] = torch.rand(4, device="cuda", requires_grad=True)
    with pytest.raises(NotImplementedError, match="forward only"):
        fmha_infer(q, k, v, **kw)
    with torch.no_grad():
        fmha_infer(q, k, v, **kw)  # no gradient wanted


def test_infer_decode_gqa_out_argument():
    """One query over grouped KV heads runs as one block per KV head, writing out through a
    view: a preallocated out gets the result, and the LSE comes back [b, h, 1]."""
    torch.random.manual_seed(0)
    q, k, v = make_qkv("gqa", 128, torch.bfloat16, 1, 300)
    out = torch.empty_like(q)
    res, lse = fmha_infer(q, k, v, return_lse=True, out=out)
    assert res.data_ptr() == out.data_ptr() and lse.shape == (2, 6, 1)
    ref_out, ref_lse = fmha_func_ref(q, k, v)
    torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize(
    "case, error, match",
    [
        # valid, but no kernel serves it: callers may fall back
        ("fp32", NotImplementedError, "fp16 and bf16"),
        ("hdim", NotImplementedError, "multiple of 8"),
        # malformed
        ("rank_mix", RuntimeError, "all"),
        ("kv_shape", RuntimeError, "do not match"),
        ("heads", RuntimeError, "divide"),
        ("alibi_shape", RuntimeError, "ALiBi"),
        ("bias_shape", RuntimeError, "bias shape"),
        ("out_shape", RuntimeError, "out must match"),
    ],
)
def test_infer_rejects(case, error, match):
    q, k, v = make_qkv("gqa", 64, torch.float16, 8, 8)
    kw = {}
    if case == "fp32":
        q, k, v = q.float(), k.float(), v.float()
    elif case == "rank_mix":
        q = q[:, :, 0]
    elif case == "kv_shape":
        v = v[:, :4]
    elif case == "heads":
        q = q[:, :, :5]
    elif case == "hdim":
        q, k, v = q[..., :60], k[..., :60], v[..., :60]
    elif case == "alibi_shape":
        kw["alibi_slopes"] = torch.ones(5, device="cuda")
    elif case == "bias_shape":
        kw["bias"] = torch.ones(8, 9, device="cuda", dtype=torch.float16)
    elif case == "out_shape":
        kw["out"] = torch.empty(2, 8, 6, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(error, match=match) as raised:
        fmha_infer(q, k, v, **kw)
    if error is RuntimeError:
        assert not isinstance(raised.value, NotImplementedError)


@pytest.mark.skipif(
    tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 4),
    reason="torch.compile support needs torch 2.4",
)
def test_infer_compiles():
    torch.random.manual_seed(0)
    q, k, v = make_qkv("gqa", 128, torch.bfloat16, 64, 96)

    def f(q, k, v):
        out, lse = fmha_infer(q, k, v, causal=True, return_lse=True)
        return out * 2, lse + 1

    out, lse = torch.compile(f, fullgraph=True)(q, k, v)
    ref_out, ref_lse = fmha_infer(q, k, v, causal=True, return_lse=True)
    torch.testing.assert_close(out, ref_out * 2, atol=0, rtol=0)
    torch.testing.assert_close(lse, ref_lse + 1, atol=0, rtol=0)
