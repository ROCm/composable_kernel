# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch

from ck_tile_fmha import fmha_func, fmha_varlen_func, fmha_with_kvcache
from reference import apply_rotary_emb


@pytest.mark.parametrize("api", ["fixed", "varlen", "kvcache"])
def test_softcap_is_rejected(api):
    q = torch.randn(1, 4, 2, 32, dtype=torch.float16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    with pytest.raises(RuntimeError, match="do not support softcap"):
        if api == "fixed":
            fmha_func(q, k, v, softcap=1.0)
        elif api == "varlen":
            cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
            fmha_varlen_func(q[0], k[0], v[0], cu_seqlens, cu_seqlens, 4, 4, softcap=1.0)
        else:
            fmha_with_kvcache(q, k, v, softcap=1.0)


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_registered_operator_fake_metadata(varlen, backward):
    # Compare fake metadata with actual CK outputs, including the uint8 dropout
    # buffer and CK-specific variable-length layouts used by torch.compile.
    q = torch.randn(2, 97, 4, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 113, 2, 64, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    sequence_args = ()
    if varlen:
        q, k, v = (x.flatten(0, 1) for x in (q, k, v))
        cu_q = torch.tensor([0, 97, 194], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, 113, 226], device="cuda", dtype=torch.int32)
        sequence_args = (cu_q, cu_k, 97, 113)
    name = "_fmha_varlen_" if varlen else "_fmha_"
    op = getattr(torch.ops.ck_tile_fmha, name + "forward")
    args = (q, k, v, *sequence_args, 0.17, 0.125, False, -1, -1, 0.0, None, True)
    if backward:
        out, lse, _, rng = op(*args)
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        op = getattr(torch.ops.ck_tile_fmha, name + "backward")
        args = (
            torch.randn_like(out),
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            *sequence_args,
            0.17,
            0.125,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng,
        )
    torch.library.opcheck(op, args, test_utils=("test_faketensor",))


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_reference_batch_offsets(interleaved):
    # Batch zero rotates by pi/2; batch one uses the identity. The last two
    # features are outside rotary_dim and must survive unchanged.
    x = torch.tensor([1, 2, 3, 4, 5, 6], device="cuda", dtype=torch.float16)
    x = x.expand(2, 1, 1, 6).contiguous()
    cos = torch.tensor([[0, 0], [1, 1]], device="cuda", dtype=x.dtype)
    sin = torch.tensor([[1, 1], [0, 0]], device="cuda", dtype=x.dtype)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    actual = apply_rotary_emb(x, cos, sin, interleaved=interleaved, seqlen_offsets=offsets)
    first = [-2, 1, -4, 3, 5, 6] if interleaved else [-3, -4, 1, 2, 5, 6]
    expected = torch.tensor([first, [1, 2, 3, 4, 5, 6]], device="cuda", dtype=x.dtype)
    assert torch.equal(actual[:, 0, 0], expected)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "stride", "alibi"])
def test_bias_validation(invalid):
    q = torch.randn(1, 4, 2, 64, device="cuda", dtype=torch.float16)
    bias = torch.zeros(4, 4, device=q.device, dtype=q.dtype)
    kwargs = {}
    if invalid == "shape":
        bias = bias.unsqueeze(0)
        message = "bias shape"
    elif invalid == "dtype":
        bias = bias.float()
        message = "same dtype"
    elif invalid == "device":
        bias = bias.cpu()
        message = "same device"
    elif invalid == "stride":
        bias = bias.t()
        message = "contiguous last dimension"
    else:
        kwargs["alibi_slopes"] = torch.ones(2, device=q.device)
        message = "cannot apply bias and alibi"
    with pytest.raises(RuntimeError, match=message):
        fmha_func(q, q, q, bias=bias, **kwargs)


def test_bias_operator_metadata():
    q = torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    bias = torch.randn(32, 32, device=q.device, dtype=q.dtype)
    fwd = torch.ops.ck_tile_fmha._fmha_forward
    args = (q, k, v, 0.17, 0.125, False, -1, -1, 0.0, None, True, bias)
    torch.library.opcheck(fwd, args, test_utils=("test_schema", "test_faketensor"))
    out, lse, _, rng = fwd(*args)
    dq, dk, dv, dbias = (torch.empty_like(x) for x in (q, k, v, bias))
    args = (
        torch.randn_like(out),
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        0.17,
        0.125,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        rng,
        bias,
        dbias,
    )
    torch.library.opcheck(
        torch.ops.ck_tile_fmha._fmha_backward,
        args,
        test_utils=("test_schema", "test_faketensor"),
    )


def test_bias_torch_compile():
    q, k, v = (
        torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    )
    bias = torch.randn(32, 32, device=q.device, dtype=q.dtype, requires_grad=True)
    compiled = torch.compile(fmha_func, backend="aot_eager", fullgraph=True)
    actual = compiled(q, k, v, bias=bias)
    expected = fmha_func(q, k, v, bias=bias)
    torch.testing.assert_close(actual, expected)
    dout = torch.randn_like(actual)
    args = (q, k, v, bias)
    for grad, ref in zip(
        torch.autograd.grad(actual, args, dout), torch.autograd.grad(expected, args, dout)
    ):
        torch.testing.assert_close(grad, ref)


def test_native_bias_defaults_preserve_positional_api():
    from ck_tile_fmha import _C

    q, k, v = (torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    args = (q, k, v, None, None, 0.0, 0.125, False, -1, -1, 0.0, False, None)
    out, lse, _, rng = _C.fwd(*args)
    explicit = _C.fwd(*args, None)
    torch.testing.assert_close(out, explicit[0])
    torch.testing.assert_close(lse, explicit[1])
    args = (
        torch.randn_like(out),
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        None,
        None,
        0.0,
        0.125,
        False,
        -1,
        -1,
        0.0,
        False,
        None,
        rng,
    )
    for actual, expected in zip(_C.bwd(*args), _C.bwd(*args, None, None)):
        torch.testing.assert_close(actual, expected)
