# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the tile_engine GEMM sweep driver's --verify and exit code.

* ``GpuGemmRunner.reference`` builds the fp32 reference from the values the
  kernel reads (the host encoding run() uses), so an exact kernel verifies for
  every dtype. Against unquantized inputs fp8/bf8 input rounding alone exceeds
  the default tolerance and every kernel was reported as MISMATCH.
* ``GpuBatchedGemmRunner`` shares it, and the GEMM, stream-K and batched
  workers use it.
* The workers generate, encode and verify against their inputs once per
  shape (each kernel still gets the verdict of its own output), keep one
  shape's operands alive at a time, and no other caller's inputs are ever
  served stale encoded bytes.
* A sweep with failed measurements exits 1.
* A search's host data (``_search_host_data``) is made in the kernel's
  memory order and a block at a time: the same outcome with or without the
  search's host cache, whose entries hold it once per problem, dtype and
  layout, byte for byte the same whether a preparation or a call stored them,
  in about 5-6 bytes per input element.

The ctypes lib is replaced by a host model of an exact kernel.
"""

import ast
import gc
import itertools
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(GEMM_TE_DIR.parent / "common"))

import gemm_full_benchmark as drv  # noqa: E402
import gemm_search as GS  # noqa: E402
import gemm_utils  # noqa: E402
import host_cache as HC  # noqa: E402
import run_one_batched_gemm_kernel  # noqa: E402
import run_one_gemm_kernel  # noqa: E402
import run_one_streamk_gemm_kernel  # noqa: E402
from batched_gemm_utils import BatchedGemmProblem, GpuBatchedGemmRunner  # noqa: E402
from gemm_utils import (  # noqa: E402
    GemmProblem,
    GpuGemmRunner,
    _bf8_u8_to_fp32,
    _bf16_u16_to_fp32,
    _fp8_u8_to_fp32,
    _fp32_to_bf16_u16,
)

DEFAULT_TOL = 2e-2
PROBLEM = {"M": 64, "N": 96, "K": 256}
BATCHED_PROBLEM = {"batch_count": 3, **PROBLEM}


def _native(h, ocp):
    return h.astype(np.float32)


# How the device reads each host input buffer, per kernel dtype.
DEVICE_DECODE = {
    "fp16": _native,
    "fp32": _native,
    "bf16": lambda h, ocp: _bf16_u16_to_fp32(h),
    "fp8": _fp8_u8_to_fp32,
    "bf8": _bf8_u8_to_fp32,
}
# Unit roundoff of the C the kernel stores (fp8/bf8 store fp16): with exact
# compute, only this rounding is left between C and the reference (fp32: plus
# the summation order of the host matmul).
C_EPS = {"bf16": 2.0**-8, "fp32": 2.0**-20}


def _c_eps(dtype):
    return C_EPS.get(dtype, 2.0**-11)


# fp8/bf8 in both the FNUZ (gfx942) and OCP (gfx950, gfx12) formats.
CASES = [("fp16", None), ("bf16", None), ("fp32", None)] + [
    (d, o) for d in ("fp8", "bf8") for o in (False, True)
]


class _ExactKernelLib:
    """Stands in for the plain or batched ctypes lib: computes C from the host
    buffers it is given in the kernel's layout, as an exact kernel would."""

    def __init__(self, dtype, layout, use_ocp, bias=0.0):
        self.dtype, self.layout, self.use_ocp, self.bias = dtype, layout, use_ocp, bias

    def run(self, A_h, B_h, C_h, *dims, **kw):
        def mat(X, lay):
            return X if lay == "r" else np.swapaxes(X, -1, -2)

        dec = DEVICE_DECODE[self.dtype]
        A = mat(dec(A_h, self.use_ocp), self.layout[0])
        B = mat(dec(B_h, self.use_ocp), self.layout[1])
        C = mat(A @ B + self.bias if self.bias else A @ B, self.layout[2])
        # bf16 C is a uint16 bit pattern; fp8/bf8 store fp16.
        C_h[...] = _fp32_to_bf16_u16(C) if self.dtype == "bf16" else C
        return 0, 1.0


def _runner(dtype, use_ocp, layout="rcr", cls=GpuGemmRunner, bias=0.0):
    """A runner of an exact kernel, or of one off by ``bias`` everywhere."""
    r = object.__new__(cls)
    r._kernel_name = f"gemm_{dtype}_{layout}_compv3_cshuffle_intrawave_128x128x64"
    r._use_ocp = use_ocp
    r.lib = _ExactKernelLib(dtype, layout, use_ocp, bias)
    return r


def _shapes(problem=PROBLEM):
    lead = (problem["batch_count"],) if "batch_count" in problem else ()
    M, N, K = problem["M"], problem["N"], problem["K"]
    return (*lead, M, K), (*lead, K, N)


def _inputs(problem=PROBLEM):
    rng = np.random.RandomState(42)
    return tuple((rng.randn(*s) * 0.1).astype(np.float32) for s in _shapes(problem))


def _max_rel(got, ref):
    # The workers' per-kernel verify metric, bit for bit.
    denom = float(np.max(np.abs(ref))) or 1.0
    return float(np.max(np.abs(got.astype(np.float32) - ref)) / denom)


def _outputs(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]


@pytest.fixture(autouse=True)
def _cold_sweep_cache():
    # Each test starts and ends without cached sweep inputs, buffers or reference.
    gemm_utils._SWEEP_CACHE.clear()
    yield
    gemm_utils._SWEEP_CACHE.clear()


RUNNERS = {
    "gemm": (GpuGemmRunner, GemmProblem, PROBLEM),
    "batched": (GpuBatchedGemmRunner, BatchedGemmProblem, BATCHED_PROBLEM),
}


@pytest.mark.parametrize("kind", RUNNERS)
@pytest.mark.parametrize("layout", ["rcr", "rrr", "crr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_reference_matches_an_exact_kernel(dtype, use_ocp, layout, kind):
    cls, problem_cls, problem = RUNNERS[kind]
    runner = _runner(dtype, use_ocp, layout, cls)
    A, B = _inputs(problem)
    out = runner.run(A, B, problem_cls.from_dict(problem)).output
    assert _max_rel(out, runner.reference(A, B)) < _c_eps(dtype)


@pytest.mark.parametrize("dtype,use_ocp", [c for c in CASES if c[0] in ("fp8", "bf8")])
def test_unquantized_reference_mismatches_fp8_bf8(dtype, use_ocp):
    # The bug the reference fixes: input rounding alone exceeds the tolerance.
    runner = _runner(dtype, use_ocp)
    A, B = _inputs()
    out = runner.run(A, B, GemmProblem.from_dict(PROBLEM)).output
    assert _max_rel(out, A @ B) > DEFAULT_TOL


def test_reference_follows_the_fp8_format():
    A, B = _inputs()
    fnuz = _runner("fp8", False).reference(A, B)
    ocp = _runner("fp8", True).reference(A, B)
    # OCP E4M3 has one more exponent step than FNUZ, so the values differ.
    assert not np.array_equal(fnuz, ocp)


# ------------------------------------------------------------------ workers


WORKERS = {
    "gemm": (run_one_gemm_kernel, GpuGemmRunner, PROBLEM),
    "streamk": (run_one_streamk_gemm_kernel, GpuGemmRunner, PROBLEM),
    "batched": (run_one_batched_gemm_kernel, GpuBatchedGemmRunner, BATCHED_PROBLEM),
}


@pytest.mark.parametrize("kind", WORKERS)
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_worker_verifies_an_exact_kernel(monkeypatch, capsys, kind, dtype, use_ocp):
    worker, cls, problem = WORKERS[kind]
    runner = _runner(dtype, use_ocp, cls=cls)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: runner)
    worker._run_one(0, "fake.so", problem, runner.kernel_name, verify=True)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] and out["verified"], out
    assert out["max_rel"] < _c_eps(dtype)


def test_gemm_worker_reuses_the_reference_per_shape_and_encoding(monkeypatch, capsys):
    # The host matmul depends only on the shape's inputs and their encoding
    # (dtype, fp8 format), never on the kernel, so a sweep computes it once per
    # shape. Kernels with another layout reuse it; the batch below also mixes
    # encodings and returns to an earlier shape, which a real sweep never does.
    worker = run_one_gemm_kernel
    calls = []
    reference = GpuGemmRunner.reference
    monkeypatch.setattr(
        GpuGemmRunner,
        "reference",
        lambda self, A, B: calls.append(self.kernel_name) or reference(self, A, B),
    )
    other = {"M": 32, "N": 64, "K": 128}
    items = [
        (_runner("bf16", None), PROBLEM),
        (_runner("bf16", None, layout="crr"), PROBLEM),
        (_runner("fp8", False), PROBLEM),
        (_runner("fp8", True), PROBLEM),
        (_runner("bf16", None), other),
        (_runner("bf16", None, layout="ccr"), PROBLEM),
    ]
    for idx, (runner, problem) in enumerate(items):
        monkeypatch.setattr(worker, "GpuGemmRunner", lambda **kw: runner)
        worker._run_one(idx, "fake.so", problem, runner.kernel_name, verify=True)
    outs = _outputs(capsys)
    assert len(outs) == len(items)
    for out, (runner, _) in zip(outs, items):
        assert out["ok"] and out["verified"], out
        assert out["max_rel"] < _c_eps(_dtype_of(runner)), out
    # bf16 on PROBLEM, fp8 FNUZ, fp8 OCP, bf16 on the other shape, and bf16 on
    # PROBLEM again: only the current shape's operands are kept.
    assert len(calls) == 5, calls


def _spy_run(*runners):
    """Record (A, B, output) of every run() of the runners, in call order."""
    seen = []
    for runner in runners:

        def spy(A, B, p, _run=runner.run, **kw):
            result = _run(A, B, p, **kw)
            seen.append((A, B, result.output))
            return result

        runner.run = spy
    return seen


def _spy_encodes(monkeypatch):
    """Record the (shape, dtype) of every operand encoded for a kernel."""
    calls = []
    encode = gemm_utils._encode_operand

    def spy(x, dtype, use_ocp=None):
        calls.append((x.shape, dtype))
        return encode(x, dtype, use_ocp)

    monkeypatch.setattr(gemm_utils, "_encode_operand", spy)
    return calls


def test_gemm_worker_reuses_encoded_operands_and_isolates_shapes_dtypes_layouts(
    monkeypatch, capsys
):
    # Kernels sharing (shape, dtype, layout, fp8 format) reuse the encoded
    # (A_h, B_h); a change in any one of them alone re-encodes, and every kernel
    # verifies against its own encoding.
    worker = run_one_gemm_kernel
    encodes = _spy_encodes(monkeypatch)
    other = {"M": 32, "N": 64, "K": 128}
    items = [
        # 3 fp8 FNUZ rcr kernels on PROBLEM -> one encode of (A, B)
        (_runner("fp8", False, layout="rcr"), PROBLEM),
        (_runner("fp8", False, layout="rcr"), PROBLEM),
        (_runner("fp8", False, layout="rcr"), PROBLEM),
        # layout only (rrr) -> one more
        (_runner("fp8", False, layout="rrr"), PROBLEM),
        (_runner("fp8", False, layout="rrr"), PROBLEM),
        # fp8 format only (OCP) -> one more
        (_runner("fp8", True, layout="rrr"), PROBLEM),
        (_runner("fp8", True, layout="rrr"), PROBLEM),
        # dtype only (bf8, still OCP) -> one more
        (_runner("bf8", True, layout="rrr"), PROBLEM),
        (_runner("bf8", True, layout="rrr"), PROBLEM),
        # shape only -> one more
        (_runner("bf8", True, layout="rrr"), other),
        (_runner("bf8", True, layout="rrr"), other),
    ]
    for idx, (runner, problem) in enumerate(items):
        monkeypatch.setattr(worker, "GpuGemmRunner", lambda **kw: runner)
        worker._run_one(idx, "fake.so", problem, runner.kernel_name, verify=True)
    outs = _outputs(capsys)
    assert len(outs) == len(items)
    for out, (runner, _) in zip(outs, items):
        assert out["ok"] and out["verified"], out
        assert out["max_rel"] < _c_eps(_dtype_of(runner)), out
    # 5 distinct keys x (A, B); reference() encodes on its own, not through here.
    assert len(encodes) == 10, encodes


@pytest.mark.parametrize("kind", WORKERS)
def test_every_worker_encodes_and_verifies_once_per_shape(monkeypatch, capsys, kind):
    # Problem B in all three sweep workers: kernels of one shape reuse the
    # encoded (A, B) and the verify reference; a new shape builds each once.
    worker, cls, problem = WORKERS[kind]
    encodes = _spy_encodes(monkeypatch)
    refs = []
    reference = cls.reference
    monkeypatch.setattr(
        cls, "reference", lambda self, A, B: refs.append(1) or reference(self, A, B)
    )
    runner = _runner("fp8", False, cls=cls)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: runner)
    other = {**problem, "M": 32, "N": 64, "K": 128}
    for idx, prob in enumerate([problem] * 3 + [other] * 2):
        worker._run_one(idx, "fake.so", prob, runner.kernel_name, verify=True)
    outs = _outputs(capsys)
    assert len(outs) == 5 and all(o["ok"] and o["verified"] for o in outs), outs
    assert (len(encodes), len(refs)) == (4, 2), encodes


@pytest.mark.parametrize("kind", WORKERS)
def test_worker_inputs_are_the_seed_42_draws(monkeypatch, capsys, kind):
    # Caching must not change what is measured: every kernel is fed the
    # RandomState(42) draws the workers used before (stream-K seeded the global
    # RNG with 42, the same stream), generated once per shape and read-only.
    worker, cls, problem = WORKERS[kind]
    runner = _runner("bf16", None, cls=cls)
    seen = _spy_run(runner)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: runner)
    for idx in range(2):
        worker._run_one(idx, "fake.so", problem, runner.kernel_name)
    assert all(o["ok"] for o in _outputs(capsys))
    assert len(seen) == 2 and all(x is y for x, y in zip(seen[0][:2], seen[1][:2]))
    for X, want in zip(seen[0], _inputs(problem)):
        assert X.dtype == np.float32 and not X.flags.writeable
        assert X.shape == want.shape and X.tobytes() == want.tobytes()


@pytest.mark.parametrize("kind", WORKERS)
def test_worker_inputs_follow_every_problem_dimension(monkeypatch, capsys, kind):
    # Each step changes ONE problem dimension, so a sweep-input key that missed
    # any of them (batch, M, N or K) would hand the kernel the previous shape's
    # A or B.
    worker, cls, problem = WORKERS[kind]
    runner = _runner("fp8", False, cls=cls)
    seen = _spy_run(runner)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: runner)
    probs = [problem]
    for dim in [d for d in ("batch_count", "M", "N", "K") if d in problem]:
        probs.append({**probs[-1], dim: probs[-1][dim] // 2})
    for idx, prob in enumerate(probs):
        worker._run_one(idx, "fake.so", prob, runner.kernel_name, verify=True)
    outs = _outputs(capsys)
    assert all(o["ok"] and o["verified"] for o in outs), outs
    assert len(outs) == len(seen) == len(probs)
    for (A, B, _), prob in zip(seen, probs):
        for X, want in zip((A, B), _inputs(prob)):
            assert X.shape == want.shape and X.tobytes() == want.tobytes(), prob


@pytest.mark.parametrize("change", ["shape", "dtype"])
@pytest.mark.parametrize("kind", WORKERS)
def test_worker_frees_old_operands_before_building_new_ones(
    monkeypatch, capsys, kind, change
):
    # A new shape frees the old fp32 (A, B), encoded buffers and reference
    # before its inputs are drawn and encoded; a new encoding of the same shape
    # frees the old encoded buffers before encoding. So the peak is one shape's
    # operands in one encoding, not two.
    worker, cls, problem = WORKERS[kind]
    runner = _runner("fp8", False, cls=cls)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: runner)
    worker._run_one(0, "fake.so", problem, runner.kernel_name, verify=True)
    cache = gemm_utils._SWEEP_CACHE
    olds = cache["enc"][1]
    if change == "shape":
        olds += cache["ab"] + cache["ref"][1]
    old = [weakref.ref(x) for x in olds]
    del olds
    alive = []

    def probe(real):
        def wrapped(*args, **kw):
            gc.collect()
            alive.append(sum(r() is not None for r in old))
            return real(*args, **kw)

        return wrapped

    monkeypatch.setattr(np.random, "RandomState", probe(np.random.RandomState))
    monkeypatch.setattr(
        gemm_utils, "_encode_operand", probe(gemm_utils._encode_operand)
    )
    if change == "shape":
        problem = {**problem, "M": 32, "N": 64, "K": 128}
    else:
        runner = _runner("bf16", None, cls=cls)
    worker._run_one(1, "fake.so", problem, runner.kernel_name, verify=True)
    assert all(o["ok"] and o["verified"] for o in _outputs(capsys))
    # RandomState (new shape only), then one encode each of A and B
    assert alive == [0] * (3 if change == "shape" else 2), alive


@pytest.mark.parametrize("kind", WORKERS)
def test_worker_verdicts_stay_per_kernel_with_a_shared_reference(
    monkeypatch, capsys, kind
):
    # Kernels of a shape share the reference, not the metric: a wrong (C x 2) and
    # a dead (C = 0) kernel between exact ones each get the metric of their own
    # output, so a broken kernel never inherits an earlier kernel's verdict.
    worker, cls, problem = WORKERS[kind]
    scales = [1.0, 2.0, 1.0, 0.0]
    runners = [_runner("fp8", False, cls=cls) for _ in scales]
    for runner, scale in zip(runners, scales):

        def scaled(A_h, B_h, C_h, *dims, _run=runner.lib.run, _s=scale, **kw):
            status = _run(A_h, B_h, C_h, *dims, **kw)
            C_h *= _s  # fp8 kernels store an fp16 C
            return status

        runner.lib.run = scaled
    seen = _spy_run(*runners)
    pending = iter(runners)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: next(pending))
    for idx, runner in enumerate(runners):
        worker._run_one(idx, "fake.so", problem, runner.kernel_name, verify=True)
    outs = _outputs(capsys)
    assert [o["verified"] for o in outs] == [True, False, True, False], outs
    for out, runner, (A, B, got) in zip(outs, runners, seen):
        assert out["max_rel"] == _max_rel(got, runner.reference(A, B)), out


_NEXT_DTYPE = {
    "fp8": "bf8",
    "bf8": "bf16",
    "bf16": "fp16",
    "fp16": "fp32",
    "fp32": "int8",
    "int8": "fp8",
}
_FLIP = {"r": "c", "c": "r"}
_NEXT_OCP = {False: True, True: None, None: False}


@pytest.mark.parametrize("kind", RUNNERS)
def test_encoded_operand_cache_key_separates_each_field(monkeypatch, kind):
    # One encoding is kept, so a key field that is not compared only shows when
    # two consecutive calls differ in that field alone: follow every
    # (dtype, la, lb, fp8 format) with each of its one-field neighbours and
    # check every call against a fresh encode of the layout-arranged operand.
    # use_ocp=None is the local arch's format, made OCP here: it must share the
    # OCP entry and never hit the FNUZ one.
    monkeypatch.setattr(gemm_utils, "_default_use_ocp", lambda: True)
    problem = RUNNERS[kind][2]
    A, B = gemm_utils._sweep_inputs(*_shapes(problem))
    for key in itertools.product(_NEXT_DTYPE, "rc", "rc", _NEXT_OCP):
        d, la, lb, ocp = key
        for nxt in (
            (_NEXT_DTYPE[d], la, lb, ocp),
            (d, _FLIP[la], lb, ocp),
            (d, la, _FLIP[lb], ocp),
            (d, la, lb, _NEXT_OCP[ocp]),
        ):
            for d2, la2, lb2, ocp2 in (key, nxt):
                got = gemm_utils._encode_ab_cached(A, B, d2, la2, lb2, ocp2)
                for X, lay, X_h in zip((A, B), (la2, lb2), got):
                    want = gemm_utils._encode_operand(
                        X if lay == "r" else np.swapaxes(X, -1, -2), d2, ocp2
                    )
                    assert X_h.dtype == want.dtype and X_h.flags.c_contiguous
                    np.testing.assert_array_equal(X_h, want, err_msg=str((key, nxt)))


def test_cached_operand_buffers_are_read_only():
    # Every kernel of a shape is handed the same buffers and verified against the
    # same reference, so an in-place write by one consumer would silently change
    # the inputs or the verdicts of the kernels after it.
    A, B = gemm_utils._sweep_inputs(*_shapes())
    for dtype in ("fp8", "bf8", "bf16", "fp16", "fp32", "int8"):
        for la in "rc":
            for X_h in gemm_utils._encode_ab_cached(A, B, dtype, la, "c", False):
                assert not X_h.flags.writeable, (dtype, la)
    gemm_utils._sweep_max_rel(_runner("fp8", False), A, B, A @ B)
    assert not any(x.flags.writeable for x in gemm_utils._SWEEP_CACHE["ref"][1])


@pytest.mark.parametrize("how", ["writeable", "read_only_view", "refrozen"])
@pytest.mark.parametrize("kind", RUNNERS)
def test_runner_sees_input_changes_between_runs(kind, how):
    # Only the sweep inputs are cached. Any other caller's A may change between
    # two runs -- in place, through the writeable base of a read-only view, or
    # by its owner unfreezing it -- and the second run must use the new values.
    cls, problem_cls, problem = RUNNERS[kind]
    runner = _runner("fp8", False, cls=cls)
    prob = problem_cls.from_dict(problem)
    data, B = _inputs(problem)
    A = data.view() if how == "read_only_view" else data
    A.flags.writeable = how == "writeable"
    B.flags.writeable = False
    out1 = runner.run(A, B, prob).output.copy()
    data.flags.writeable = True
    data *= 2.0
    A.flags.writeable = how == "writeable"
    out2 = runner.run(A, B, prob).output
    assert not np.allclose(out1, out2)
    assert _max_rel(out2, runner.reference(A, B)) < _c_eps("fp8")


@pytest.mark.parametrize("swept", ["A", "B"])
@pytest.mark.parametrize("kind", RUNNERS)
def test_runner_encodes_a_sweep_operand_paired_with_a_callers_afresh(kind, swept):
    # Only the (A, B) pair _sweep_inputs returned is cached: either of its arrays
    # paired with a caller's own array, which may change between runs, is
    # encoded fresh on every run.
    cls, problem_cls, problem = RUNNERS[kind]
    runner = _runner("fp8", False, cls=cls)
    prob = problem_cls.from_dict(problem)
    ops = dict(zip("AB", gemm_utils._sweep_inputs(*_shapes(problem))))
    mine = "B" if swept == "A" else "A"
    ops[mine] = _inputs(problem)["AB".index(mine)]
    out1 = runner.run(ops["A"], ops["B"], prob).output.copy()
    ops[mine] *= 2.0
    out2 = runner.run(ops["A"], ops["B"], prob).output
    assert not np.allclose(out1, out2)
    assert _max_rel(out2, runner.reference(ops["A"], ops["B"])) < _c_eps("fp8")


@pytest.mark.parametrize("kind", RUNNERS)
def test_runner_leaves_callers_arrays_writeable(kind):
    # Only the buffers the cache keeps are frozen. A caller's own arrays stay
    # writeable after a run, including when a row-major fp32 operand is handed
    # to the kernel as-is.
    cls, problem_cls, problem = RUNNERS[kind]
    prob = problem_cls.from_dict(problem)
    for dtype in ("fp32", "fp16", "bf16", "fp8"):
        for layout in ("rrr", "ccr"):
            A, B = _inputs(problem)
            _runner(dtype, False, layout, cls).run(A, B, prob)
            assert A.flags.writeable and B.flags.writeable, (dtype, layout)


@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_sweep_metric_equals_the_per_kernel_metric(dtype, use_ocp):
    # Reusing the reference must not move any verify verdict: for every kernel
    # the metric is bit-identical to max|out - ref| / max|ref| computed for that
    # kernel alone, as the workers did before. The second layout reuses the
    # cached reference; -A puts the largest |ref| on a negative entry, and a zero
    # A gives an all-zero reference, which the metric divides by 1.0.
    A, B = gemm_utils._sweep_inputs(*_shapes())
    ref = _runner(dtype, use_ocp).reference(A, B)
    assert np.max(ref) > -np.min(ref)  # the largest |ref| is positive
    for X, layout in ((A, "rcr"), (A, "crr"), (-A, "rcr"), (0 * A, "rcr")):
        runner = _runner(dtype, use_ocp, layout)
        out = runner.run(X, B, GemmProblem.from_dict(PROBLEM)).output
        want = _max_rel(out, runner.reference(X, B))
        assert gemm_utils._sweep_max_rel(runner, X, B, out) == want, layout


def _dtype_of(runner):
    return runner.kernel_name.split("_")[1]


# ------------------------------------------------------------- host cache


@pytest.fixture
def gemm_worker(monkeypatch, capsys):
    """run(runner, problem, **kw): the GEMM worker's JSON line for ``runner``
    (fresh in-process caches); ``counts`` of the legacy inputs and references
    and of the search's host data made."""
    worker = run_one_gemm_kernel
    counts = {"inputs": 0, "reference": 0, "search": 0}
    inputs, reference = worker._sweep_inputs, GpuGemmRunner.reference
    search = worker._search_host_data

    def count(what, fn):
        def wrapped(*a):
            counts[what] += 1
            return fn(*a)
        return wrapped

    monkeypatch.setattr(worker, "_sweep_inputs", count("inputs", inputs))
    monkeypatch.setattr(GpuGemmRunner, "reference", count("reference", reference))
    monkeypatch.setattr(worker, "_search_host_data", count("search", search))

    def run(runner, problem=PROBLEM, **kw):
        monkeypatch.setattr(worker, "GpuGemmRunner", lambda **k: runner)
        gemm_utils._SWEEP_CACHE.clear()
        worker._run_one(0, "fake.so", problem, runner.kernel_name, verify=True, **kw)
        return capsys.readouterr().out.strip()

    run.counts = counts
    return run


@pytest.mark.parametrize("layout", ["rcr", "crr", "rrr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_host_cache_gives_the_uncached_outcome(gemm_worker, tmp_path, dtype, use_ocp,
                                               layout):
    # A search's call without the cache, its first call with it (a miss) and
    # a call that reads the entry give the same line; the legacy inputs are
    # other values, verified as before.
    runner = _runner(dtype, use_ocp, layout)
    legacy = json.loads(gemm_worker(runner))
    plain = gemm_worker(runner, search_host_data=True)
    with HC.HostCache(tmp_path) as cache:
        miss = gemm_worker(runner, host_cache=cache.spec)
        hit = gemm_worker(runner, host_cache=cache.spec)
        assert HC.usage(cache.dir)[0] == 1
    assert plain == miss == hit and json.loads(hit)["verified"], (plain, hit)
    for out in (legacy, json.loads(hit)):
        assert out["verified"] and out["max_rel"] < _c_eps(dtype), out
    # The hit read everything: the search's host data made by the first two.
    assert gemm_worker.counts == {"inputs": 1, "reference": 1, "search": 2}


@pytest.mark.parametrize("layout", ["rcr", "crr", "rrr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_prepared_entry_gives_the_uncached_outcome(gemm_worker, tmp_path, monkeypatch,
                                                   capsys, dtype, use_ocp, layout):
    # The search's preparation loads no library: it encodes with
    # GpuGemmRunner's own codec and resolves the fp8/bf8 format as the measure
    # worker's runner does (no arch given: the local GPU's), so it stores the
    # entry a measure call would, and that call then computes nothing.
    import gemm_utils

    monkeypatch.setattr(gemm_utils, "_default_use_ocp", lambda: bool(use_ocp))
    runner = _runner(dtype, None, layout)
    plain = gemm_worker(runner, search_host_data=True)

    def no_library(*a, **kw):
        raise AssertionError("a preparation loaded a library")

    monkeypatch.setattr(gemm_utils, "GemmDispatcherLib", no_library)
    with HC.HostCache(tmp_path) as cache:
        run_one_gemm_kernel._prepare(PROBLEM, runner.kernel_name, cache.spec)
        assert json.loads(capsys.readouterr().out) == {"ok": True, "stored": True}
        hit = gemm_worker(runner, host_cache=cache.spec)
        assert HC.usage(cache.dir)[0] == 1
    assert hit == plain and json.loads(hit)["verified"]
    assert gemm_worker.counts == {"inputs": 0, "reference": 0, "search": 2}


def test_preparation_reports_what_it_could_not_store(capsys, tmp_path):
    name = _runner("bf16", None).kernel_name
    for args, want in (((PROBLEM, name, None), {"ok": True, "stored": False}),
                       ((None, name, {"dir": str(tmp_path)}), {"ok": False})):
        run_one_gemm_kernel._prepare(*args)
        out = json.loads(capsys.readouterr().out)
        assert {k: out[k] for k in want} == want
    assert out["error"].startswith("TypeError: ")  # the type, not only its text


def test_host_cache_keeps_dtypes_layouts_and_problems_apart(gemm_worker, tmp_path):
    # One entry per (problem, dtype, layout, fp8 format): a crr kernel handed
    # an rcr entry's buffers (or bf16 an fp16 one's) would fail verification.
    # So does a change of K alone, which leaves C's shape: another K's A and B
    # have the wrong length for the kernel.
    kernels = [(_runner("bf16", None, lay), PROBLEM) for lay in ("rcr", "crr", "ccr")]
    kernels += [(_runner("fp16", None), PROBLEM), (_runner("fp8", False), PROBLEM),
                (_runner("fp8", True), PROBLEM)]
    kernels += [(_runner("bf16", None), {**PROBLEM, **o})
                for o in ({"M": 96, "N": 64}, {"K": 128})]
    with HC.HostCache(tmp_path) as cache:
        first = [gemm_worker(r, p, host_cache=cache.spec) for r, p in kernels]
        again = [gemm_worker(r, p, host_cache=cache.spec) for r, p in kernels]
        assert HC.usage(cache.dir)[0] == len(kernels)
    assert first == again
    for line, (runner, _) in zip(first, kernels):
        out = json.loads(line)
        assert out["verified"] and out["max_rel"] < _c_eps(_dtype_of(runner)), out
    assert gemm_worker.counts["search"] == len(kernels)


@pytest.mark.parametrize("layout", ["rcr", "crr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_preparation_and_call_store_the_same_entry_bytes(gemm_worker, tmp_path,
                                                         monkeypatch, dtype, use_ocp,
                                                         layout):
    # A preparation and a call that misses compute the entry with the same
    # function: every file of the two entries is the same, byte for byte.
    import gemm_utils

    monkeypatch.setattr(gemm_utils, "_default_use_ocp", lambda: bool(use_ocp))
    runner = _runner(dtype, None, layout)
    entries = []
    for name in ("prepared", "called"):
        (tmp_path / name).mkdir()
        with HC.HostCache(tmp_path / name) as cache:
            if name == "prepared":
                run_one_gemm_kernel._prepare(PROBLEM, runner.kernel_name, cache.spec)
            else:
                gemm_worker(runner, host_cache=cache.spec)
            (entry,) = [p for p in cache.dir.iterdir() if p.is_dir()]
            entries.append({f.name: f.read_bytes() for f in entry.iterdir()})
    assert set(entries[0]) == {"a.npy", "b.npy", "ref.npy", "denom.npy", "meta.json"}
    assert entries[0] == entries[1]


def test_search_inputs_do_not_depend_on_the_block_size(monkeypatch):
    runner = _runner("fp8", False, "crr")
    problem = GemmProblem.from_dict(PROBLEM)
    whole = run_one_gemm_kernel._search_host_data(runner, problem)
    monkeypatch.setattr(run_one_gemm_kernel, "BLOCK", 7)
    blocks = run_one_gemm_kernel._search_host_data(runner, problem)
    assert {k: v.tobytes() for k, v in whole.items()} == {
        k: v.tobytes() for k, v in blocks.items()}
    assert whole["a"].shape == (PROBLEM["K"], PROBLEM["M"])  # a "c" A, as stored


@pytest.mark.parametrize("most", [1, 2, 3, 4, 7, 8, 64])
def test_ccr_reference_blocks_cover_the_rows_in_odd_counts(most):
    # A block's transposed copy is slow when its row count is a power of two.
    for m in range(1, 70):
        blocks = list(run_one_gemm_kernel._odd_blocks(m, most))
        assert [i for i, _ in blocks] == [sum(r for _, r in blocks[:j])
                                          for j in range(len(blocks))]
        assert sum(r for _, r in blocks) == m
        assert all(r % 2 == 1 and r <= most for _, r in blocks), (m, blocks)


@pytest.mark.parametrize("rows", [1, 4, 64])
def test_ccr_reference_in_blocks_is_the_product(monkeypatch, rows):
    # Blocks of 1, 3 (of 4) and 63 + 1 rows (all 64): the reference is A @ B.
    monkeypatch.setattr(run_one_gemm_kernel, "REF_BLOCK", rows * PROBLEM["N"])
    host = run_one_gemm_kernel._search_host_data(_runner("bf16", None, "ccr"),
                                                 GemmProblem.from_dict(PROBLEM))
    a, b = (_bf16_u16_to_fp32(host[x]).T for x in "ab")  # stored K x M, N x K
    np.testing.assert_allclose(host["ref"], a @ b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rows", [1, 5, 64, 1000])
def test_comparison_in_blocks_of_rows_gives_the_same_values(rows):
    rng = np.random.RandomState(1)
    ref = rng.randn(64, 96).astype(np.float32)
    out = (ref + 1e-3 * rng.randn(64, 96)).astype(np.float16)
    nan = out.copy()
    nan[40, 3] = np.nan
    for output in (out, np.asfortranarray(out), nan):
        whole, blocks = {}, {}
        run_one_gemm_kernel._check(whole, output, ref, 2.5, 2e-2)
        run_one_gemm_kernel._check(blocks, output, ref, 2.5, 2e-2, rows)
        assert json.dumps(whole) == json.dumps(blocks)


def test_search_calls_compare_a_block_of_rows_at_a_time(gemm_worker, monkeypatch):
    # C and the reference are compared BLOCK elements at a time in a search;
    # the legacy path is unchanged: the sweep's whole-array metric.
    seen, check = [], run_one_gemm_kernel._check
    legacy, max_rel = [], run_one_gemm_kernel._sweep_max_rel
    monkeypatch.setattr(run_one_gemm_kernel, "_check",
                        lambda *a: seen.append(a[5:]) or check(*a))
    monkeypatch.setattr(run_one_gemm_kernel, "_sweep_max_rel",
                        lambda *a: legacy.append(a) or max_rel(*a))
    gemm_worker(_runner("bf16", None), search_host_data=True)
    assert seen == [(run_one_gemm_kernel.BLOCK // PROBLEM["N"],)] and not legacy
    gemm_worker(_runner("bf16", None))
    assert len(seen) == 1 and len(legacy) == 1


@pytest.mark.parametrize("bias", [-0.5, 0.5])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_search_calls_refuse_a_kernel_off_in_either_direction(gemm_worker, dtype,
                                                              use_ocp, bias):
    # The metric is max|C - ref| / max|ref|: a kernel whose every output is
    # too small fails as one whose outputs are too large.
    out = json.loads(gemm_worker(_runner(dtype, use_ocp, bias=bias), search_host_data=True))
    assert out["verified"] is False and out["max_rel"] > 0.1, out


@pytest.mark.parametrize("sign", [1, -1])
def test_search_denominator_is_the_largest_magnitude_of_the_reference(monkeypatch,
                                                                      sign):
    # B negated negates the reference: in one of the two runs its largest
    # magnitude is a negative entry, and the denominator is still that magnitude.
    operand, made = run_one_gemm_kernel._search_operand, []

    def signed(*a):
        enc, vals = operand(*a)
        made.append(vals)
        if len(made) == 2:  # B: the reference reads its fp32 values
            vals *= sign
        return enc, vals

    monkeypatch.setattr(run_one_gemm_kernel, "_search_operand", signed)
    host = run_one_gemm_kernel._search_host_data(_runner("fp16", None),
                                                 GemmProblem.from_dict(PROBLEM))
    ref = host["ref"]
    assert float(np.max(ref)) != -float(np.min(ref))
    assert float(host["denom"]) == float(np.max(np.abs(ref)))


@pytest.mark.parametrize("make", [MemoryError, lambda: np.empty(1 << 50, np.uint8)])
def test_search_errors_name_their_type(gemm_worker, monkeypatch, make):
    # A search call's error starts with its type, so the search tells a
    # MemoryError (whose text can be empty) from a kernel's failure. A plain
    # run's error is its text alone, as before.
    runner = _runner("bf16", None)

    def fail(*a):
        raise make()  # numpy's raises while it allocates

    monkeypatch.setattr(runner, "run_encoded", fail)
    monkeypatch.setattr(runner, "run", fail)
    search = json.loads(gemm_worker(runner, search_host_data=True))
    plain = json.loads(gemm_worker(runner))
    assert not search["ok"] and not plain["ok"]
    assert search["error"] == "MemoryError: " + plain["error"], (search, plain)
    assert bool(plain["error"]) == (make is not MemoryError)


def test_legacy_inputs_are_unchanged():
    # A run without the search keys measures with these inputs, as before.
    got = run_one_gemm_kernel._sweep_inputs(*_shapes())
    for a, b in zip(got, _inputs()):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_search_workers_ask_to_be_ended_first_when_memory_runs_out(monkeypatch,
                                                                   tmp_path):
    # A search worker (a preparation, or a call with search_host_data) sets its
    # oom_score_adj to 1000; a plain run's worker leaves it alone.
    import io

    adj = tmp_path / "oom_score_adj"
    monkeypatch.setattr(run_one_gemm_kernel, "OOM_SCORE_ADJ", str(adj))
    monkeypatch.setattr(run_one_gemm_kernel, "_run_one", lambda *a, **kw: None)
    monkeypatch.setattr(run_one_gemm_kernel, "_prepare", lambda *a: None)
    plain = {"so_path": "x.so", "problem": PROBLEM, "kernel_name": "k", "verify": True}
    for payload, first in ((plain, False), ({**plain, "check_identity": True}, False),
                           ({**plain, "search_host_data": True}, True),
                           ({"prepare": True, "problem": PROBLEM}, True)):
        if adj.exists():
            adj.unlink()
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(
            json.dumps(payload).encode())))
        run_one_gemm_kernel.main()
        assert (adj.read_text() if adj.exists() else None) == ("1000" if first else None)
    monkeypatch.setattr(run_one_gemm_kernel, "OOM_SCORE_ADJ", str(tmp_path / "no" / "f"))
    run_one_gemm_kernel._first_to_go()  # cannot be written: nothing happens


_RSS_JOB = r"""
import json, sys
sys.path[:0] = json.loads(sys.argv[1])
import gemm_utils
import run_one_gemm_kernel as w

name, mode = sys.argv[2], sys.argv[4]


class Lib:  # a kernel that writes every element of C
    def run(self, A, B, C, *dims):
        C[...] = 1
        return 0, 1.0


runner = object.__new__(gemm_utils.GpuGemmRunner)
runner._kernel_name, runner._use_ocp, runner.lib = name, False, Lib()
w.GpuGemmRunner = lambda lib_path: runner


def run(problem):  # _search_host_data alone, or a whole search call without an entry
    if mode == "data":
        return w._search_host_data(runner, gemm_utils.GemmProblem.from_dict(problem))
    w._run_one(0, "x.so", problem, name, verify=True, search_host_data=True)


def status(field):
    with open("/proc/self/status") as f:
        return next(int(ln.split()[1]) * 1024 for ln in f if ln.startswith(field + ":"))


run({"M": 64, "N": 64, "K": 4096})
try:
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")  # the peak (VmHWM) restarts at the current RSS
except OSError:
    print("skip")
    sys.exit()
base = status("VmRSS")
host = run(json.loads(sys.argv[3]))
print(status("VmHWM") - base)
"""


def _peak_rss(dtype, problem, mode):
    """Peak RSS (bytes, imports excluded) of ``mode``: "data" for
    ``_search_host_data``, "call" for a whole search call without an entry
    (a kernel that writes C, then the comparison). Its bound is the guard's
    estimate without the fixed part (the GPU runtime, the interpreter) plus
    128 MiB (numpy's and the fp8 encoder's temporaries of one block)."""
    paths = [str(DISPATCHER_DIR / "python"), str(GEMM_TE_DIR), str(GEMM_TE_DIR.parent /
                                                                   "common")]
    got = subprocess.run([sys.executable, "-c", _RSS_JOB, json.dumps(paths),
                          f"gemm_{dtype}_rcr_x", json.dumps(problem), mode],
                         capture_output=True, text=True, check=True).stdout.split()
    if got[-1:] == ["skip"]:  # "call" mode prints its warm-up result first
        pytest.skip("/proc/self/clear_refs is not writable")
    bound = GS.host_compute_bytes(problem, dtype) - GS.HOST_FIXED_BYTES + (128 << 20)
    assert int(got[-1]) <= bound, (problem, int(got[-1]), bound)
    return int(got[-1])


@pytest.mark.skipif(not os.path.exists("/proc/self/clear_refs"), reason="Linux only")
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
def test_search_host_data_needs_its_encoded_inputs_plus_4_bytes_each(dtype):
    # Peak RSS of _search_host_data at two K (256x768): each more input
    # element costs its encoded bytes plus its fp32 value, 5 B in fp8 and
    # 6 B in bf16, against about 58 and 16.5 B for the inputs, encoding and
    # reference a search made before (float64 normals, the fp8 encoder on
    # whole arrays, transposed copies, the inputs encoded twice). The guard's
    # estimate, less its fixed part, covers the peak.
    per = (_peak_rss(dtype, {"M": 256, "N": 768, "K": 36864}, "data")
           - _peak_rss(dtype, {"M": 256, "N": 768, "K": 4096}, "data")) / (
        (256 + 768) * (36864 - 4096))
    assert 0 < per <= GS.HOST_INPUT_BYTES[dtype] + 4 + 0.25, per


@pytest.mark.skipif(not os.path.exists("/proc/self/clear_refs"), reason="Linux only")
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
def test_search_call_needs_at_most_16_bytes_per_output_element(dtype):
    # A whole search call without an entry on output-heavy shapes: each more
    # output element costs the fp32 reference, C (bf16: also decoded to fp32)
    # and its comparison, within HOST_OUTPUT_BYTES.
    small, large = ({"M": n, "N": n, "K": 64} for n in (1024, 3072))
    inputs = (2 * 3072 - 2 * 1024) * 64 * (GS.HOST_INPUT_BYTES[dtype] + 4)
    per = (_peak_rss(dtype, large, "call") - _peak_rss(dtype, small, "call")
           - inputs) / (3072 ** 2 - 1024 ** 2)
    assert 4 < per <= GS.HOST_OUTPUT_BYTES, per


def test_host_cache_over_its_cap_computes_as_without_it(gemm_worker, tmp_path):
    runner = _runner("bf16", None, "crr")
    plain = gemm_worker(runner, search_host_data=True)
    with HC.HostCache(tmp_path, max_entry_bytes=1) as cache:
        lines = [gemm_worker(runner, host_cache=cache.spec) for _ in range(2)]
        assert HC.usage(cache.dir) == (0, 0)
    assert lines == [plain, plain]
    assert gemm_worker.counts == {"inputs": 0, "reference": 0, "search": 3}


# ------------------------------------------------------------- exit code


def _main(monkeypatch, capsys, tmp_path, n_fail):
    lib = tmp_path / "libfake.so"
    lib.write_bytes(b"")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemm_full_benchmark.py",
            "--arch",
            "gfx950",
            "--max-kernels",
            "1",
            "--csv",
            str(tmp_path / "out.csv"),
        ],
    )
    monkeypatch.setattr(drv, "resolve_devices", lambda spec: ["0"])
    monkeypatch.setattr(
        drv, "setup_multiple_gemm_dispatchers", lambda cfgs, **kw: [lib] * len(cfgs)
    )
    monkeypatch.setattr(
        drv, "_run_batch_on_device", lambda *a, **kw: ([], [], n_fail, 0)
    )
    rc = drv.main()
    return rc, capsys.readouterr().out


@pytest.mark.parametrize("n_fail,rc", [(0, 0), (1, 1)])
def test_sweep_exit_code_reflects_failures(monkeypatch, capsys, tmp_path, n_fail, rc):
    got, out = _main(monkeypatch, capsys, tmp_path, n_fail)
    assert got == rc
    assert "BENCHMARK COMPLETE" in out


DRIVERS = [
    "gemm_full_benchmark.py",
    "batched_gemm_full_benchmark.py",
    "gemm_multi_d_full_benchmark.py",
    "streamk_gemm_full_benchmark.py",
    "grouped_gemm_full_benchmark.py",
    "block_scale_gemm/gemm_bquant/gemm_bquant_full_benchmark.py",
    "mx_gemm_full_benchmark.py",
]


@pytest.mark.parametrize("path", DRIVERS)
def test_every_gemm_sweep_driver_returns_its_failures(path):
    # The other drivers share the summary tail; their last return must depend
    # on the failure count instead of being a constant 0.
    tree = ast.parse((GEMM_TE_DIR / path).read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    last = main.body[-1]
    assert isinstance(last, ast.Return)
    assert not isinstance(last.value, ast.Constant), ast.unparse(last)
    assert "fail" in ast.unparse(last.value)
