# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch

from ck_tile_fmha import fmha_fp8_func, fmha_func, fmha_varlen_fp8_func, fmha_varlen_func


def fp8_unsupported_reason(qscale, d, return_lse=False, mask="none"):
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if arch.startswith("gfx1250"):
        return "gfx1250 FP8 requires head dimension <= 128" if d > 128 else None
    if qscale == "perhead":
        return "FP8 per-head kernels require gfx1250"
    if return_lse and not (
        arch.startswith("gfx950")
        and d == 128
        and qscale in ("no_scale", "pertensor")
        and mask in ("none", "causal")
    ):
        return "FP8 LSE requires gfx950, head dimension 128, no/per-tensor scale and no/causal mask"
    return None


def make_fp8_input(shape, qscale, block_size=128, is_value=False):
    """Quantized input, FP32 descale, and dequantized reference with varying scales."""
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    dtype = torch.float8_e4m3fnuz if arch.startswith("gfx94") else torch.float8_e4m3fn
    b, s, h, _ = shape
    x = torch.randn(shape, device="cuda") * 0.5
    if qscale == "no_scale":
        quant = x.to(dtype)
        return quant, None, quant.float()
    if qscale == "pertensor":
        descale = torch.rand(1, device="cuda") * 0.03 + 0.01
        broadcast = descale
    elif qscale == "perhead":
        descale = (torch.rand(h * 2, b, device="cuda") * 0.03 + 0.01).t()[:, ::2]
        broadcast = descale[:, None, :, None]
    else:
        blocks = (s + block_size - 1) // block_size
        descale = (torch.rand(h, b, blocks, device="cuda") * 0.03 + 0.01).transpose(0, 1)
        if is_value and arch.startswith("gfx1250"):
            descale = torch.exp2(
                torch.randint(-6, -3, (h, b, blocks), device="cuda").float()
            ).transpose(0, 1)
        broadcast = descale.repeat_interleave(block_size, -1)[..., :s].transpose(1, 2)[..., None]
    # Different physical values and quantization grids across heads, batches and blocks.
    x = x * (broadcast / 0.02)
    quant = (x / broadcast).to(dtype)
    return quant, descale, quant.float() * broadcast


def check_fp8_output(out, q, k, v, scale, causal=False, window=(-1, -1)):
    # Compare the same dequantized FP8 data, rounded to BF16 for the baseline kernel.
    q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    kwargs = dict(softmax_scale=scale, causal=causal, window_size=window)
    ref, lse, _ = fmha_func(q, k, v, **kwargs, return_attn_probs=True)
    magnitude = fmha_func(q, k, v.abs(), **kwargs).float()
    ref = ref.float()
    valid = torch.isfinite(lse)
    check_fp8_numerics(out, ref, magnitude, valid)
    return lse, valid


def check_fp8_numerics(out, ref, magnitude, valid):
    # Use the C++ example's magnitude and gain checks with a BF16 baseline.
    # Attention(Q, K, abs(V)) estimates P @ abs(V) without materializing P.
    u_p, u_o = 2**-3, 2**-7
    error = out.float() - ref
    bound = u_p * magnitude + (u_o * ref.abs()).clamp_min(u_o * torch.finfo(torch.bfloat16).tiny)
    assert torch.all(error.abs() <= bound), (
        f"FP8 rounding bound exceeded: {(error.abs() - bound).max().item()}"
    )
    # Also detect a systematic gain per (batch, head), which the bound alone can miss.
    denominator = ref.square().sum(dim=(1, 3))
    alpha = (error * ref).sum(dim=(1, 3)) / denominator.clamp_min(1e-30)
    rows = valid.sum(dim=-1).clamp_min(1)
    alpha_tol = 2 * (u_p**2 / (12 * rows) + u_o**2 / (12 * rows * out.shape[-1])) ** 0.5
    assert torch.all(alpha.abs() <= alpha_tol), (
        f"FP8 gain error: {alpha.abs().max().item()} > {alpha_tol}"
    )


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "perhead", "blockscale"])
@pytest.mark.parametrize("d", [32, 59, 64, 111, 128])
@pytest.mark.parametrize("mask", ["none", "causal", "local", "local_causal"])
@pytest.mark.parametrize(
    "sq,sk,h,hk", [(97, 203, 4, 4), (1, 257, 6, 2), (193, 113, 6, 1), (256, 256, 4, 4)]
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_fp8_fixedlen(qscale, d, mask, sq, sk, h, hk, return_lse):
    reason = fp8_unsupported_reason(qscale, d, return_lse, mask)
    if reason:
        pytest.skip(reason)
    torch.manual_seed(0)
    q, qs, qref = make_fp8_input((2, sq, h, d), qscale)
    k, ks, kref = make_fp8_input((2, sk, hk, d), qscale)
    v, vs, vref = make_fp8_input((2, sk, hk, d), qscale, is_value=True)
    causal = mask in ("causal", "local_causal")
    window = (37, 11) if mask.startswith("local") else (-1, -1)
    scale = 0.7 / d**0.5
    actual = fmha_fp8_func(
        q,
        k,
        v,
        qs,
        ks,
        vs,
        qscale=qscale,
        causal=causal,
        window_size=window,
        softmax_scale=scale,
        return_lse=return_lse,
    )
    out = actual[0] if return_lse else actual
    lse_ref, valid = check_fp8_output(out, qref, kref, vref, scale, causal, window)
    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    if return_lse:
        lse = actual[1]
        assert lse.dtype == torch.float32
        assert lse.shape == (2, h, sq)
        torch.testing.assert_close(lse[valid], lse_ref[valid], atol=0.01, rtol=0.001)
        assert torch.isneginf(lse[~valid]).all()


@pytest.mark.parametrize(
    "block_size_q,block_size_kv", [(128, 32), (128, 64), (128, 128), (256, 128), (256, 256)]
)
def test_fp8_block_sizes(block_size_q, block_size_kv):
    qscale = "blockscale"

    if (
        not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
        and block_size_kv % 128
    ):
        pytest.skip("The gfx9 FP8 head-dimension 128 kernel requires KV blocks divisible by 128")
    torch.manual_seed(1)
    q, qs, qr = make_fp8_input((2, 273, 6, 128), qscale, block_size_q)
    k, ks, kr = make_fp8_input((2, 397, 2, 128), qscale, block_size_kv)
    v, vs, vr = make_fp8_input((2, 397, 2, 128), qscale, block_size_kv, is_value=True)
    out = fmha_fp8_func(
        q, k, v, qs, ks, vs, qscale=qscale, block_size_q=block_size_q, block_size_kv=block_size_kv
    )
    check_fp8_output(out, qr, kr, vr, 128**-0.5)


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "blockscale"])
@pytest.mark.parametrize("d", [160, 192, 256])
def test_fp8_large_head(qscale, d):
    torch.manual_seed(2)
    q, qs, qr = make_fp8_input((2, 97, 4, d), qscale)
    k, ks, kr = make_fp8_input((2, 203, 2, d), qscale)
    v, vs, vr = make_fp8_input((2, 203, 2, d), qscale)
    if torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250"):
        with pytest.raises(RuntimeError, match="No FMHA forward kernel"):
            fmha_fp8_func(q, k, v, qs, ks, vs, qscale=qscale)
    else:
        out = fmha_fp8_func(q, k, v, qs, ks, vs, qscale=qscale)
        check_fp8_output(out, qr, kr, vr, d**-0.5)


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "perhead", "blockscale"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("d", [64, 128])
def test_fp8_operator_and_compile(qscale, return_lse, d):
    reason = fp8_unsupported_reason(qscale, d, return_lse)
    if reason:
        pytest.skip(reason)
    q, qs, _ = make_fp8_input((2, 32, 4, d), qscale)
    k, ks, _ = make_fp8_input((2, 67, 2, d), qscale)
    v, vs, _ = make_fp8_input((2, 67, 2, d), qscale, is_value=True)
    args = (
        q,
        k,
        v,
        0.0,
        0.125,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
        qs,
        ks,
        vs,
        qscale,
        return_lse,
    )
    torch.library.opcheck(
        torch.ops.ck_tile_fmha._fmha_forward,
        args,
        test_utils=("test_faketensor",),
    )
    # PyTorch's schema checker calls allclose on FP8, which ROCm does not implement.
    # Check input mutation explicitly using exact FP32 representations instead.
    inputs = [x for x in (q, k, v, qs, ks, vs) if x is not None]
    before = [x.float().clone() for x in inputs]
    # Each parameter set compiles independently of Dynamo's recompilation limit.
    torch.compiler.reset()
    compiled = torch.compile(fmha_fp8_func, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(q, k, v, qs, ks, vs, qscale=qscale, return_lse=return_lse),
        fmha_fp8_func(q, k, v, qs, ks, vs, qscale=qscale, return_lse=return_lse),
    )
    for x, saved in zip(inputs, before):
        assert torch.equal(x.float(), saved)


@pytest.mark.parametrize(
    "qscale,d,return_lse,window",
    [
        ("perhead", 128, False, (-1, -1)),
        ("pertensor", 64, True, (-1, -1)),
        ("blockscale", 128, True, (-1, -1)),
        ("pertensor", 128, True, (3, 2)),
    ],
)
def test_fp8_gfx950_unsupported(qscale, d, return_lse, window):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 kernel restrictions")
    q, qs, _ = make_fp8_input((1, 32, 4, d), qscale)
    k, ks, _ = make_fp8_input((1, 67, 2, d), qscale)
    v, vs, _ = make_fp8_input((1, 67, 2, d), qscale)
    message = (
        "per-head attention requires gfx1250" if qscale == "perhead" else "No FMHA forward kernel"
    )
    with pytest.raises(RuntimeError, match=message):
        fmha_fp8_func(q, k, v, qs, ks, vs, qscale=qscale, return_lse=return_lse, window_size=window)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "dtype",
        "device",
        "shape",
        "grad",
        "mode",
        "no_scale",
        "block_size",
        "block_shape",
        "block_stride",
        "fp8_dtype",
        "qkv_dtype",
    ],
)
def test_fp8_validation(invalid):
    q, qs, _ = make_fp8_input((2, 32, 4, 64), "pertensor")
    k, ks, _ = make_fp8_input((2, 97, 2, 64), "pertensor")
    v, vs, _ = make_fp8_input((2, 97, 2, 64), "pertensor")
    kwargs = {}
    if invalid == "missing":
        qs, message = None, "required"
    elif invalid == "dtype":
        qs, message = qs.half(), "float32"
    elif invalid == "device":
        qs, message = qs.cpu(), "same device"
    elif invalid == "shape":
        qs, message = qs.expand(2), "one scale"
    elif invalid == "grad":
        qs, message = qs.requires_grad_(), "forward only"
    elif invalid == "mode":
        kwargs["qscale"], message = "unknown", "Unsupported FP8 qscale"
    elif invalid == "no_scale":
        kwargs["qscale"], message = "no_scale", "does not accept descales"
    elif invalid == "block_size":
        kwargs.update(qscale="blockscale", block_size_kv=0)
        message = "block sizes must be positive"
    elif invalid == "block_shape":
        kwargs["qscale"], message = "blockscale", "must have shape"
    elif invalid == "block_stride":
        kwargs.update(qscale="blockscale", block_size_q=16)
        qs = torch.ones(2, 4, 4, device="cuda")[..., ::2]
        message = "contiguous block dimension"
    elif invalid == "fp8_dtype":
        other = torch.float8_e4m3fnuz if q.dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
        q, k, v = (x.float().to(other) for x in (q, k, v))
        message = "dtype does not match the GPU"
    else:
        k, message = k.half(), "same dtype"
    with pytest.raises((RuntimeError, ValueError), match=message):
        fmha_fp8_func(q, k, v, qs, ks, vs, **kwargs)


def make_fp8_varlen_input(lengths, heads, d, qscale):
    quant, descale, ref = make_fp8_input((len(lengths), max(lengths), heads, d), qscale)
    quant = torch.cat([quant[b, :length] for b, length in enumerate(lengths)])
    ref = torch.cat([ref[b, :length] for b, length in enumerate(lengths)])
    cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return quant, descale, ref, cu


def check_fp8_varlen_output(out, qr, kr, vr, cuq, cuk, lengths_q, lengths_k, scale, causal, window):
    qr, kr, vr = (x.to(torch.bfloat16) for x in (qr, kr, vr))
    args = (cuq, cuk, max(lengths_q), max(lengths_k))
    kwargs = dict(softmax_scale=scale, causal=causal, window_size=window)
    ref, lse, _ = fmha_varlen_func(qr, kr, vr, *args, **kwargs, return_attn_probs=True)
    magnitude = fmha_varlen_func(qr, kr, vr.abs(), *args, **kwargs).float()
    offset = 0
    for length in lengths_q:
        sl = slice(offset, offset + length)
        check_fp8_numerics(
            out[sl].unsqueeze(0),
            ref[sl].float().unsqueeze(0),
            magnitude[sl].unsqueeze(0),
            torch.isfinite(lse[:, sl]).unsqueeze(0),
        )
        offset += length
    return lse


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "perhead"])
@pytest.mark.parametrize("d", [32, 59, 64, 111, 128])
@pytest.mark.parametrize("mask", ["none", "causal", "local", "local_causal"])
@pytest.mark.parametrize(
    "lengths_q,lengths_k,h,hk",
    [
        ([97, 1, 193], [203, 257, 113], 4, 4),
        ([3, 131, 65], [129, 27, 260], 6, 2),
        ([1, 7, 129], [259, 1, 97], 6, 1),
        ([0, 7, 127, 1], [13, 0, 257, 31], 6, 2),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_fp8_varlen(qscale, d, mask, lengths_q, lengths_k, h, hk, return_lse):
    reason = fp8_unsupported_reason(qscale, d, return_lse, mask)
    if reason:
        pytest.skip(reason)
    torch.manual_seed(31)
    q, qs, qr, cuq = make_fp8_varlen_input(lengths_q, h, d, qscale)
    k, ks, kr, cuk = make_fp8_varlen_input(lengths_k, hk, d, qscale)
    v, vs, vr, _ = make_fp8_varlen_input(lengths_k, hk, d, qscale)
    causal = mask in ("causal", "local_causal")
    window = (37, 11) if mask.startswith("local") else (-1, -1)
    scale = 0.7 / d**0.5
    actual = fmha_varlen_fp8_func(
        q,
        k,
        v,
        cuq,
        cuk,
        max(lengths_q),
        max(lengths_k),
        qs,
        ks,
        vs,
        qscale=qscale,
        causal=causal,
        window_size=window,
        softmax_scale=scale,
        return_lse=return_lse,
    )
    out = actual[0] if return_lse else actual
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    lse_ref = check_fp8_varlen_output(
        out,
        qr,
        kr,
        vr,
        cuq,
        cuk,
        lengths_q,
        lengths_k,
        scale,
        causal,
        window,
    )
    if return_lse:
        lse = actual[1]
        assert lse.dtype == torch.float32 and lse.shape == (h, sum(lengths_q))
        valid = torch.isfinite(lse_ref)
        torch.testing.assert_close(lse[valid], lse_ref[valid], atol=0.01, rtol=0.001)
        assert torch.equal(lse[~valid], lse_ref[~valid])
        assert not lse.isnan().any()


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "perhead"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fp8_varlen_operator_and_compile(qscale, return_lse):
    reason = fp8_unsupported_reason(qscale, 128, return_lse)
    if reason:
        pytest.skip(reason)
    q, qs, _, cuq = make_fp8_varlen_input([3, 17], 4, 128, qscale)
    k, ks, _, cuk = make_fp8_varlen_input([23, 9], 2, 128, qscale)
    v, vs, _, _ = make_fp8_varlen_input([23, 9], 2, 128, qscale)
    args = (q, k, v, cuq, cuk, 17, 23, 0.0, 128**-0.5, False)
    kwargs = dict(q_descale=qs, k_descale=ks, v_descale=vs, qscale=qscale, return_lse=return_lse)
    torch.library.opcheck(
        torch.ops.ck_tile_fmha._fmha_varlen_forward, args, kwargs, test_utils=("test_faketensor",)
    )
    inputs = [x for x in (q, k, v, cuq, cuk, qs, ks, vs) if x is not None]
    before = [x.float().clone() for x in inputs]
    torch.compiler.reset()
    compiled = torch.compile(fmha_varlen_fp8_func, backend="aot_eager", fullgraph=True)
    args = (q, k, v, cuq, cuk, 17, 23)
    torch.testing.assert_close(compiled(*args, **kwargs), fmha_varlen_fp8_func(*args, **kwargs))
    for x, saved in zip(inputs, before):
        assert torch.equal(x.float(), saved)


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "perhead"])
@pytest.mark.parametrize("empty", ["q", "k", "both"])
def test_fp8_varlen_empty(qscale, empty):
    reason = fp8_unsupported_reason(qscale, 128)
    if reason:
        pytest.skip(reason)
    lq = [0, 0] if empty in ("q", "both") else [3, 7]
    lk = [0, 0] if empty in ("k", "both") else [9, 13]
    q, qs, _, cuq = make_fp8_varlen_input(lq, 4, 128, qscale)
    k, ks, _, cuk = make_fp8_varlen_input(lk, 2, 128, qscale)
    v, vs, _, _ = make_fp8_varlen_input(lk, 2, 128, qscale)
    out = fmha_varlen_fp8_func(q, k, v, cuq, cuk, max(lq), max(lk), qs, ks, vs, qscale=qscale)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("qscale", ["pertensor", "perhead"])
def test_fp8_varlen_strides_and_pytorch(qscale):
    from reference import attention_ref

    reason = fp8_unsupported_reason(qscale, 64)
    if reason:
        pytest.skip(reason)
    torch.manual_seed(19)
    lq, lk = [5, 17, 3], [11, 7, 23]
    q, qs, qr, cuq = make_fp8_varlen_input(lq, 6, 64, qscale)
    k, ks, kr, cuk = make_fp8_varlen_input(lk, 2, 64, qscale)
    v, vs, vr, _ = make_fp8_varlen_input(lk, 2, 64, qscale)
    strided = []
    for x in (q, k, v):
        storage = torch.empty((x.shape[0] * 2, x.shape[1] * 2, 64), device="cuda", dtype=x.dtype)
        view = storage[::2, ::2]
        view.copy_(x)
        strided.append(view)
    out = fmha_varlen_fp8_func(
        *strided, cuq, cuk, max(lq), max(lk), qs, ks, vs, qscale=qscale, causal=True
    )
    oq = ok = 0
    for sq, sk in zip(lq, lk):
        a, b, c = qr[oq : oq + sq][None], kr[ok : ok + sk][None], vr[ok : ok + sk][None]
        ref, _, lse = attention_ref(a, b, c, causal=True, return_lse=True)
        magnitude, _ = attention_ref(a, b, c.abs(), causal=True)
        check_fp8_numerics(out[oq : oq + sq][None], ref, magnitude, torch.isfinite(lse))
        oq += sq
        ok += sk


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "dtype",
        "device",
        "tensor_shape",
        "head_shape",
        "grad",
        "mode",
        "blockscale",
        "no_scale",
        "qkv_dtype",
        "fp8_dtype",
        "rank",
        "cu_dtype",
        "max_length",
    ],
)
def test_fp8_varlen_validation(invalid):
    q, qs, _, cuq = make_fp8_varlen_input([3, 7], 4, 64, "pertensor")
    k, ks, _, cuk = make_fp8_varlen_input([9, 13], 2, 64, "pertensor")
    v, vs, _, _ = make_fp8_varlen_input([9, 13], 2, 64, "pertensor")
    kwargs = {}
    maxq = 7
    if invalid == "missing":
        qs, message = None, "required"
    elif invalid == "dtype":
        qs, message = qs.half(), "float32"
    elif invalid == "device":
        qs, message = qs.cpu(), "same device"
    elif invalid == "tensor_shape":
        qs, message = qs.expand(2), "one scale"
    elif invalid == "head_shape":
        if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250"):
            pytest.skip("per-head kernels require gfx1250")
        kwargs["qscale"], message = "perhead", "must have shape"
    elif invalid == "grad":
        qs, message = qs.requires_grad_(), "forward only"
    elif invalid in ("mode", "blockscale"):
        kwargs["qscale"], message = invalid, "Unsupported FP8 varlen qscale"
    elif invalid == "no_scale":
        kwargs["qscale"], message = "no_scale", "does not accept descales"
    elif invalid == "qkv_dtype":
        k, message = k.half(), "same dtype"
    elif invalid == "fp8_dtype":
        other = torch.float8_e4m3fnuz if q.dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
        q, k, v = (x.float().to(other) for x in (q, k, v))
        message = "dtype does not match the GPU"
    elif invalid == "rank":
        q, message = q[None], "three-dimensional"
    elif invalid == "cu_dtype":
        cuq, message = cuq.long(), "int32"
    else:
        maxq, message = -1, "nonnegative"
    with pytest.raises((RuntimeError, ValueError), match=message):
        fmha_varlen_fp8_func(q, k, v, cuq, cuk, maxq, 13, qs, ks, vs, **kwargs)


@pytest.mark.parametrize("option", ["out", "dropout", "alibi", "splits", "seqused_k"])
def test_fp8_varlen_native_options(option):
    from ck_tile_fmha import _C

    q, qs, _, cuq = make_fp8_varlen_input([3, 17], 4, 128, "pertensor")
    k, ks, _, cuk = make_fp8_varlen_input([23, 9], 2, 128, "pertensor")
    v, vs, _, _ = make_fp8_varlen_input([23, 9], 2, 128, "pertensor")
    out = torch.empty_like(q, dtype=torch.bfloat16)
    args = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        seqused_k=None,
        leftpad_k=None,
        block_table=None,
        alibi_slopes=None,
        max_seqlen_q=17,
        max_seqlen_k=23,
        dropout_p=0.0,
        softmax_scale=128**-0.5,
        zero_tensors=False,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
        gen=None,
        num_splits=0,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        qscale="pertensor",
        return_lse=False,
    )
    if option == "out":
        result = _C.varlen_fwd(**args)[0]
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(
            result, fmha_varlen_fp8_func(q, k, v, cuq, cuk, 17, 23, qs, ks, vs)
        )
        args["out"] = torch.empty_like(q)
        message = "incorrect dtype"
    elif option == "dropout":
        args["dropout_p"], message = 0.1, "does not support dropout"
    elif option == "alibi":
        args["alibi_slopes"], message = torch.ones(4, device="cuda"), "does not support ALiBi"
    elif option == "splits":
        args["num_splits"], message = 2, "without paged KV or splits"
    else:
        args["seqused_k"], message = cuk[1:] - cuk[:-1], "without paged KV or splits"
    with pytest.raises(RuntimeError, match=message):
        _C.varlen_fwd(**args)
