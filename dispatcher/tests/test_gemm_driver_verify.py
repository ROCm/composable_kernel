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

The ctypes lib is replaced by a host model of an exact kernel.
"""

import ast
import gc
import itertools
import json
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))

import gemm_full_benchmark as drv  # noqa: E402
import gemm_utils  # noqa: E402
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

    def __init__(self, dtype, layout, use_ocp):
        self.dtype, self.layout, self.use_ocp = dtype, layout, use_ocp

    def run(self, A_h, B_h, C_h, *dims, **kw):
        def mat(X, lay):
            return X if lay == "r" else np.swapaxes(X, -1, -2)

        dec = DEVICE_DECODE[self.dtype]
        A = mat(dec(A_h, self.use_ocp), self.layout[0])
        B = mat(dec(B_h, self.use_ocp), self.layout[1])
        C = mat(A @ B, self.layout[2])
        # bf16 C is a uint16 bit pattern; fp8/bf8 store fp16.
        C_h[...] = _fp32_to_bf16_u16(C) if self.dtype == "bf16" else C
        return 0, 1.0


def _runner(dtype, use_ocp, layout="rcr", cls=GpuGemmRunner):
    r = object.__new__(cls)
    r._kernel_name = f"gemm_{dtype}_{layout}_compv3_cshuffle_intrawave_128x128x64"
    r._use_ocp = use_ocp
    r.lib = _ExactKernelLib(dtype, layout, use_ocp)
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
