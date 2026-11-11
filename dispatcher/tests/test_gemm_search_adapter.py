#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
The GEMM adapter of the adaptive kernel search (tile_engine/ops/gemm/gemm_search.py).

A fake toolchain stands in for the compiler and the GPU: a built library is a
text file holding the kernel name it reports, and the real worker
(run_one_gemm_kernel.main) runs in-process on a runner that reads that name
back. Measurements go through the driver's real benchmark_on_devices and
_run_batch_on_device, so rows, failure records, compact indices, the identity
check and the per-call output files are all exercised as in a real run.

No GPU or compiler needed. Run: python3 -m pytest -q tests/test_gemm_search_adapter.py
"""

import argparse
import contextlib
import csv
import fcntl
import functools
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(GEMM_TE_DIR.parent / "common"))

import adaptive_search as AS  # noqa: E402
import gemm_full_benchmark as drv  # noqa: E402
import gemm_search as GS  # noqa: E402
import host_cache as HC  # noqa: E402
import gemm_utils  # noqa: E402
import run_one_gemm_kernel as worker  # noqa: E402
from gemm_utils import GemmKernelConfig  # noqa: E402

BASE = GemmKernelConfig(gfx_arch="gfx950")
CFGS = [
    replace(BASE, tile_m=m, tile_n=n)
    for m, n in ((64, 64), (128, 64), (128, 128), (256, 128))
]
A, B, C, D = (c.name for c in CFGS)
P1, P2 = {"M": 64, "N": 32, "K": 16}, {"M": 32, "N": 64, "K": 16}
K1, K2 = "64x32x16", "32x64x16"


class _Runner:
    """GpuGemmRunner stand-in: an exact kernel, one whose output is wrong for
    the problems ``wrong(problem)`` names, or one returning ``status``."""

    def __init__(self, name, ms, wrong=lambda problem: False, status=0,
                 oom=lambda problem: False):
        self.kernel_name, self._ms, self._wrong, self._use_ocp = name, ms, wrong, None
        self._status, self._oom = status, oom

    def run(self, A, B, problem):
        if self._status:
            return SimpleNamespace(success=False, status=self._status)
        out = A @ B + (1.0 if self._wrong(problem) else 0.0)
        return SimpleNamespace(success=True, status=0, time_ms=self._ms, tflops=1.0,
                               output=out)

    def reference(self, A, B):
        return A @ B

    def encode_inputs(self, A, B):
        return A, B

    def run_encoded(self, A_h, B_h, problem):
        if self._oom(problem):
            raise MemoryError()
        # The search's host data is in the kernel's memory order (a "c"
        # operand transposed) and dtype; the legacy path's is logical fp32.
        la, lb, _ = self.kernel_name.split("_")[2]
        return self.run((A_h if la == "r" else A_h.T).astype("float32"),
                        (B_h if lb == "r" else B_h.T).astype("float32"), problem)


_WORKER_LOCK = threading.Lock()  # sys.stdin and stdout are the process's


def _worker(payload: bytes) -> str:
    """The real worker main()'s output for ``payload``, run in-process (one
    at a time: device threads may call it at once)."""
    out = io.StringIO()
    with _WORKER_LOCK:
        stdin, sys.stdin = sys.stdin, io.TextIOWrapper(io.BytesIO(payload))
        try:
            with contextlib.redirect_stdout(out):
                worker.main()
        finally:
            sys.stdin = stdin
    return out.getvalue()


class _InProcess:
    """subprocess.Popen stand-in running the real worker main() in-process."""

    returncode = 0

    def __init__(self, tc):
        self.tc = tc

    def communicate(self, input=None, timeout=None):
        payload = json.loads(input)
        self.tc.payloads.append(payload)
        if any(self.tc.hit(self.tc.killed, it["kernel_name"], it["problem"]["M"])
               for it in payload["items"]):
            self.returncode = -9  # the out-of-memory killer ended it: no output
            return b"", b""
        out = _worker(input)
        self.tc.outputs.extend(json.loads(ln) for ln in out.splitlines())
        return out.encode(), b""

    def kill(self):
        pass

    def poll(self):
        return 0


class Toolchain:
    """Fake build + GPU. ``fail``: names whose build fails; ``mislabel``: name
    -> the name its library reports; ``lat``: name -> ms; ``wrong``: names, or
    (name, M) pairs, whose output is wrong (for every problem, or for M);
    ``killed`` and ``oom`` likewise, whose worker the out-of-memory killer
    ends (exit code -9) or raises MemoryError. Preparations run the real
    worker in-process too, on the stand-in's codec; ``prepares`` holds each
    call's payloads, limits and width."""

    def __init__(self, tmp_path, monkeypatch, fail=(), mislabel=None, lat=None,
                 wrong=(), killed=(), oom=()):
        self.lib_dir = tmp_path / "examples"
        self.lib_dir.mkdir()
        self.fail, self.mislabel = set(fail), dict(mislabel or {})
        self.lat, self.wrong = dict(lat or {}), set(wrong)
        self.killed, self.oom = set(killed), set(oom)
        self.builds, self.payloads, self.outputs, self.loaded = [], [], [], []
        self.prepares, self.arches = [], []
        monkeypatch.setattr(worker, "GpuGemmRunner", self.runner)
        monkeypatch.setattr(worker, "_HostCodec", lambda name, arch=None: _Runner(name, 1.0))
        monkeypatch.setattr(AS, "prepare_host_data", self.prepare)
        monkeypatch.setattr(drv, "subprocess", SimpleNamespace(
            Popen=lambda *a, **kw: _InProcess(self), PIPE=subprocess.PIPE,
            DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired))
        monkeypatch.setattr(gemm_utils, "_SWEEP_CACHE", {})

    def build(self, cfgs, verbose=True, max_workers=None, *, errors=None,
              build_timeout=None):
        self.builds.append({"names": [c.name for c in cfgs], "verbose": verbose,
                            "max_workers": max_workers, "build_timeout": build_timeout})
        out = []
        for c in cfgs:
            if c.name in self.fail:
                errors[c.name] = "Compile failed: k.cpp:1:2: error: boom\nmore"
                out.append(None)
                continue
            lib = self.lib_dir / f"lib{c.name}.so"
            lib.write_text(self.mislabel.get(c.name, c.name))
            out.append(lib)
        return out

    def prepare(self, argv, env, payloads, limits, width, needs=None, memory=None,
                keeps=None):
        """``prepare_host_data`` with each worker run in-process, one at a
        time, each within ``memory()`` when ``needs`` are given."""
        self.prepares.append({"payloads": {k: json.loads(p) for k, p in payloads.items()},
                              "limits": dict(limits), "width": width, "needs": needs,
                              "keeps": keeps})
        out = {}
        for k, p in payloads.items():
            budget = None if needs is None else memory()
            if budget is not None and needs[k] > budget:
                out[k] = AS._prepared(None, "", 0.0, AS.host_memory_error(needs[k], budget))
            else:
                out[k] = AS._prepared(0, _worker(p.encode()), 0.0, "")
        return out

    @staticmethod
    def hit(which, name, m) -> bool:
        """Whether ``which`` (names, or (name, M) pairs) names this pair."""
        return bool({name, (name, m)} & which)

    def runner(self, lib_path, arch=None):
        self.loaded.append(Path(lib_path))
        self.arches.append(arch)
        name = Path(lib_path).read_text()
        return _Runner(name, self.lat.get(name, 1.0),
                       lambda prob: self.hit(self.wrong, name, prob.M),
                       oom=lambda prob: self.hit(self.oom, name, prob.M))


@pytest.fixture(autouse=True)
def _own_host_lock(tmp_path, monkeypatch):
    """Each test's search runs take the host lock in its own temp dir, and
    its in-process workers leave the test process's oom_score_adj alone."""
    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "OOM_SCORE_ADJ", str(tmp_path / "oom_score_adj"))


HOST_MEMORY = AS.host_memory  # the reader itself, which _ample_host_memory replaces


@pytest.fixture(autouse=True)
def _ample_host_memory(monkeypatch):
    """The host memory guard sees 1 TiB available, whatever this host has
    free; a test of the guard sets its own reading."""
    monkeypatch.setattr(AS, "host_memory", lambda: 1 << 40)


def _memory(tmp_path, total_gib, avail_gib):
    """``adaptive_search.host_memory`` on a /proc/meminfo with these sizes."""
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: {int(total_gib * 2**20)} kB\n"
                    f"MemAvailable: {int(avail_gib * 2**20)} kB\n")
    return functools.partial(HOST_MEMORY, str(path), str(tmp_path / "no-cgroup"))


def _args(**kw):
    a = dict(verify=False, verify_tol=0.02, batch_size=20, kernel_timeout=30,
             workers=3, build_timeout=None, dtype="fp16", problems=None, pairs=None,
             max_kernels=0, no_vector_fallback=False, tune_c_vector_width=False,
             search_output_dir=None)
    a.update(kw)
    return argparse.Namespace(**a)


def _hooks(build, measure=drv.benchmark_on_devices):
    return GS.DriverHooks(build=build, measure=measure, record=drv._kernel_record,
                          problem_key=drv.problem_key, load_pairs=drv.load_pair_list,
                          build_reason=drv.build_failure_summary)


def _adapter(tmp_path, build, measure=drv.benchmark_on_devices, problems=(P1, P2),
             eligible=None, devices=("0",), timeout=None, configs=CFGS, cache=None,
             **args):
    eligible = eligible or [[A, B, C, D]] * len(problems)
    return GS.GemmSearchAdapter(
        _args(**args), _hooks(build, measure), list(problems), configs, eligible,
        list(devices), (Path("worker.py"), {}), tmp_path / "out", {"arch": "gfx950"},
        **({} if timeout is None else {"build_timeout": timeout}), cache=cache,
    )


def _req(problem, name, n):
    return AS.PairRequest(problem, name, name, f"{problem}:{n}", f"d{n}")


class _Recorder:
    """benchmark_on_devices wrapper recording each call's compact inputs."""

    def __init__(self):
        self.calls = []

    def __call__(self, problems, pairs, built, devices, args, *rest, **kw):
        self.calls.append({"problems": problems, "pairs": pairs,
                           "built": [(cfg.name, Path(p)) for cfg, p in built],
                           "verify": args.verify, "batch_size": args.batch_size,
                           "extra": kw.get("extra")})
        return drv.benchmark_on_devices(problems, pairs, built, devices, args, *rest, **kw)


# ------------------------------------------------------------ reconcile


@pytest.mark.parametrize("value, want", [
    (True, True), ("True", True), (False, False), ("False", False),
    ("", None), (None, None), ("true", None), (1, None), ("1", None),
])
def test_parse_verified_accepts_only_true_and_its_csv_form(value, want):
    assert GS.parse_verified(value) is want


def _row(**kw):
    r = {"kernel": A, "problem_idx": 0, **P1, "device": "0", "latency_ms": 1.5,
         "tflops": 1.0, "non_zero": 1, "max_rel": 0.0, "verified": True}
    r.update(kw)
    return r


def _fail(kind, error="x", kernel=A):
    return {"kernel": kernel, "problem_idx": 0, **P1, "device": "0", "kind": kind,
            "error": error}


@pytest.mark.parametrize("rows, fails, status", [
    ([_row()], [], "ok"),
    ([_row(verified="True")], [], "ok"),
    ([_row(verified="False")], [], "numeric_fail"),
    ([_row(verified=False)], [_fail("mismatch")], "numeric_fail"),
    ([_row(verified="")], [], "invalid_measurement"),
    ([_row(latency_ms=math.nan)], [], "invalid_measurement"),
    ([_row(latency_ms=0.0)], [], "invalid_measurement"),
    ([_row(latency_ms=-1.0)], [], "invalid_measurement"),
    ([_row(latency_ms=True)], [], "invalid_measurement"),
    ([_row(), _row()], [], "invalid_measurement"),
    ([_row(kernel=B)], [], "run_error"),
    ([_row(M=128)], [], "run_error"),
    ([], [_fail("timeout")], "timeout"),
    ([], [_fail("missing")], "run_error"),
    ([], [_fail("unsupported")], "run_error"),
    ([], [_fail("batch-error")], "run_error"),
    ([_row()], [_fail("worker-exit", kernel="")], "run_error"),
    ([_row()], [_fail("bad-output", kernel="")], "invalid_measurement"),
    ([_row(verified=False)], [_fail("mismatch"), _fail("timeout")], "timeout"),
    ([], [_fail("something-new")], "run_error"),
    ([], [], "unattempted"),
])
def test_reconcile_normalizes_rows_and_failures(rows, fails, status):
    got = GS.reconcile(_req(K1, A, 1), P1, rows, fails)
    assert got[0] == status
    assert (got[1] == 1.5) if status == "ok" else got[1] is None


KILLED_CALL = ("host_memory: its call's worker was killed (exit code -9), likely by "
               "the out-of-memory killer")


@pytest.mark.parametrize("rows, fails, want", [
    ([], [_fail("missing", "worker exited with code -9")], KILLED_CALL),
    ([_row()], [_fail("worker-exit", "worker exited with code -9", kernel="")], KILLED_CALL),
    ([], [_fail("error", "MemoryError: "), _fail("missing", "worker exited with code 1")],
     "host_memory: its call's worker ran out of memory (MemoryError: )"),
    ([], [_fail("missing", "worker exited with code -11")], "run_error"),
    ([], [_fail("missing", "worker exited with code 9")], "run_error"),
    ([], [_fail("error", "RuntimeError: no MemoryError")], "run_error"),
    ([], [_fail("timeout", "batch of 1 exceeded 30 s")], "timeout"),
])
def test_reconcile_tells_a_worker_out_of_host_memory_from_a_failed_kernel(rows, fails,
                                                                          want):
    # The driver kills a worker only at its timeout ("timeout"): one that
    # exited by SIGKILL (-9) was killed from outside, on a host short of
    # memory by the out-of-memory killer. That, or a MemoryError, is the
    # host's: host_memory. A crash (-11) or a plain exit stays a run_error.
    status, lat, diag = GS.reconcile(_req(K1, A, 1), P1, rows, fails)
    if want.startswith("host_memory"):
        assert (status, lat, diag) == ("host_memory", None, want)
    else:
        assert (status, lat) == (want, None)


@pytest.mark.parametrize("code, why", [("-2", "no kernel selected"),
                                       ("-3", "does not support the problem"),
                                       ("-1", None)])
def test_reconcile_names_the_dispatcher_status(code, why):
    fail = _fail("error", f"kernel returned status {code}")
    status, _, diag = GS.reconcile(_req(K1, A, 1), P1, [], [fail])
    assert status == "run_error" and diag.startswith(f"error: kernel returned status {code}")
    assert (why in diag) if why else not diag.endswith(")")


def _ranked(tmp_path, text, mode="w"):
    path = tmp_path / "ranked.csv"
    (path.write_text if mode == "w" else path.write_bytes)(text)
    return path


def test_read_ranked_csv_is_a_latency_prior(tmp_path):
    # select_kernels.py --out_ranked: predicted TFLOPS -> predicted ms.
    path = _ranked(tmp_path, "problem,rank,kernel,predicted_tflops,candidates\n"
                   "64x32x16,1,kB,4.0,3\n64x32x16,2,kA,2.0,3\n64x32x16,3,kC,0.0,3\n"
                   "8x8x8,1,kA,nan,1\n")
    flops = 2 * 64 * 32 * 16
    assert GS.read_ranked_csv(path) == {
        "64x32x16": {"kB": pytest.approx(flops / 4e9), "kA": pytest.approx(flops / 2e9)}}


@pytest.mark.parametrize("text, match", [
    ("problem,kernel\n64x32x16,kA\n", "expected columns"),
    ("problem,rank,kernel,predicted_tflops\n64x32x16,1,kA\n", "line 2"),  # short row
    ("problem,rank,kernel,predicted_tflops\n64x32,1,kA,2.0\n", "line 2"),
    ("problem,rank,kernel,predicted_tflops\n64x32x16,1,kA,fast\n", "line 2"),
    (b"problem,rank,kernel,predicted_tflops\n64x32x16,1,k\xff,2.0\n", "utf-8"),
])
def test_unusable_ranked_csv_is_a_settings_error(tmp_path, text, match):
    path = _ranked(tmp_path, text, "w" if isinstance(text, str) else "b")
    with pytest.raises(ValueError, match=match):
        GS.read_ranked_csv(path)
    # Through the coordinator it is invalid settings (the driver exits 2).
    settings = AS.SearchSettings.from_json({
        "schema_version": 1, "strategy": "tpe", "trials_per_problem": 2,
        "new_builds_per_run": 2, "warm_start": {"ranked_csv": str(path),
                                                "use_prior": True}})
    with pytest.raises(AS.SettingsError, match="warm_start.ranked_csv"):
        AS.SearchCoordinator(None, settings, tmp_path / "out",
                             prior_loader=GS.read_ranked_csv)


# -------------------------------------------------------------- adapter


def test_candidates_are_the_eligible_kernels_with_scalar_records(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, problems=(P1, P2, P1),
                  eligible=[[A, B, A], [B, C], [D]])
    got = ad.enumerate_eligible()
    assert {k: [c.kernel_name for c in v] for k, v in got.items()} == {
        K1: [A, B], K2: [B, C]}
    b1, b2 = got[K1][1], got[K2][0]
    assert b1 is b2 and b1.canonical_key == b1.kernel_name == B
    assert b1.original_config is CFGS[1]
    assert set(b1.record) == set(drv.KERNEL_RECORD_FIELDS)
    assert not any(isinstance(v, str) and v.startswith("gemm_") for v in b1.record.values())
    ctx = ad.context()
    assert ctx["arch"] == "gfx950" and ctx["measurement"]["verify"] is True
    assert ctx["measurement"]["batch_size"] == 1
    assert any("searched once" in n for n in ctx["domain_notes"])
    json.dumps(ctx, allow_nan=False)


def test_runs_of_one_checkout_build_and_store_one_at_a_time(tmp_path, monkeypatch):
    # Two runs of one checkout, for different targets, build kernel A at
    # once. The build writes A's library to one path from its name alone
    # (no arch); under the build lock each run stores its own library.
    shared = tmp_path / "examples"
    shared.mkdir()
    building = threading.Event()

    def build_for(arch):
        def build(cfgs, verbose=True, max_workers=None, *, errors=None,
                  build_timeout=None):
            libs = []
            for c in cfgs:
                lib = shared / f"lib{c.name}.so"
                lib.write_text(arch)
                building.set()
                time.sleep(0.5)  # where the other run's build would land
                libs.append(lib)
            return libs
        return build

    ads = {}
    for arch in ("gfx950", "gfx1250"):
        (tmp_path / arch).mkdir()
        ads[arch] = _adapter(tmp_path / arch, build_for(arch))
        ads[arch].hooks = replace(ads[arch].hooks, build_dir=tmp_path / "build")
    stored = {}

    def run(arch):
        stored[arch] = Path(ads[arch].build_missing([ads[arch].cands[A]])[A].path)

    first = threading.Thread(target=run, args=("gfx950",))
    first.start()
    assert building.wait(30)
    run("gfx1250")
    first.join()
    assert {arch: p.read_text() for arch, p in stored.items()} == {
        "gfx950": "gfx950", "gfx1250": "gfx1250"}
    assert (tmp_path / "build" / GS.BUILD_LOCK).exists()


@pytest.mark.parametrize("timeout, used", [(None, 180), (900, 900)])
def test_build_gets_the_driver_options_and_reports_one_line(
    tmp_path, monkeypatch, timeout, used
):
    # The search's build timeout, not the driver's --build-timeout (rejected
    # with --search-config): its own default, or build_timeout_s.
    tc = Toolchain(tmp_path, monkeypatch, fail={B})
    ad = _adapter(tmp_path, tc.build, timeout=timeout, workers=5, build_timeout=7)
    res = ad.build_missing([ad.cands[n] for n in (A, B)])
    assert tc.builds == [{"names": [A, B], "verbose": True, "max_workers": 5,
                          "build_timeout": used}]
    build = ad.context()["build"]
    assert (build["workers"], build["build_timeout"]) == (5, used)
    assert build["build_timeout_applies_to"].startswith(
        "every kernel" if timeout else "rounds of native kernels only")
    assert res[B] == AS.BuildResult(None, "Compile failed: k.cpp:1:2: error: boom")
    # The one-line reason goes to the events; the full error is kept on disk.
    log = json.loads((tmp_path / "out" / "rounds" / "0001-build.json").read_text())
    assert log == {"kernels": [A, B], "build_timeout": used,
                   "errors": {B: "Compile failed: k.cpp:1:2: error: boom\nmore"}}
    art = tmp_path / "out" / "artifacts" / f"{A}.so"
    assert res[A] == AS.BuildResult(str(art), digest=hashlib.sha256(
        A.encode()).hexdigest())


@pytest.mark.parametrize("timeout, used", [(None, [180, None, None]), (900, [900] * 3)])
def test_fixed_width_kernels_keep_the_build_functions_limits(
    tmp_path, monkeypatch, timeout, used
):
    # The search's default was derived from native kernels only: a build call
    # holding a fixed-vector-width kernel passes no limit, so the build
    # function keeps its own (1200 s to compile that kernel, as in a full run).
    # A given build_timeout_s applies to every kernel.
    fixed = replace(CFGS[0], vector_size_a=4, vector_size_b=4, vector_size_c=4)
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, problems=(P1,), eligible=[[A, B, fixed.name]],
                  configs=CFGS + [fixed], timeout=timeout)
    for names in ([A], [fixed.name], [B, fixed.name]):
        ad.build_missing([ad.cands[n] for n in names])
    assert [b["build_timeout"] for b in tc.builds] == used
    logs = sorted((tmp_path / "out" / "rounds").glob("*-build.json"))
    assert [json.loads(p.read_text())["build_timeout"] for p in logs] == used


def test_build_returning_the_wrong_number_of_paths_is_refused(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, lambda cfgs, **kw: tc.build(cfgs, **kw)[:-1])
    with pytest.raises(RuntimeError, match="build returned 1 paths for 2 kernels"):
        ad.build_missing([ad.cands[n] for n in (A, B)])


def test_failed_build_compacts_indices_and_stale_files_are_never_loaded(
    tmp_path, monkeypatch
):
    # Fixture 3: A, B, C requested; B fails. The compact list is A, C, so C is
    # index 1. A stale B library (an earlier build) and stale copies in the
    # artifacts directory are never loaded.
    tc = Toolchain(tmp_path, monkeypatch, fail={B}, lat={A: 2.0, C: 1.0})
    art = tmp_path / "out" / "artifacts"
    art.mkdir(parents=True)
    (tc.lib_dir / f"lib{B}.so").write_text(B)
    (art / f"{B}.so").write_text(B)
    (art / f"{A}.so").write_text(D)  # stale, overwritten by the fresh copy
    rec = _Recorder()
    ad = _adapter(tmp_path, tc.build, rec)
    ad.build_missing([ad.cands[n] for n in (A, B, C)])
    out = ad.measure_pairs([_req(K1, A, 1), _req(K1, B, 2), _req(K1, C, 3)])
    assert [(o.request.kernel_name, o.status, o.latency_ms) for o in out] == [
        (A, "ok", 2.0), (B, "compile_error", None), (C, "ok", 1.0)]
    (call,) = rec.calls
    assert call["built"] == [(A, art / f"{A}.so"), (C, art / f"{C}.so")]
    assert call["pairs"] == [[0], [1]] and call["problems"] == [P1, P1]
    assert (call["verify"], call["batch_size"]) == (True, 1)
    assert call["extra"] == {"check_identity": True, "search_host_data": True,
                             "arch": "gfx950"}
    assert tc.loaded == [art / f"{A}.so", art / f"{C}.so"]
    assert tc.arches == ["gfx950", "gfx950"]  # the run's --arch pins the fp8 format
    assert all(B not in it["so_path"] for p in tc.payloads for it in p["items"])
    # The round's CSV holds both rows; its JSON maps problem_idx to the request.
    rounds = tmp_path / "out" / "rounds"
    with open(rounds / "0002-measure.csv", newline="") as f:
        assert [(r["kernel"], r["problem_idx"]) for r in csv.DictReader(f)] == [
            (A, "0"), (C, "1")]
    side = json.loads((rounds / "0002-measure.json").read_text())
    assert [(r["problem_idx"], r["call_id"]) for r in side["requests"]] == [
        (0, "d1"), (1, "d3")]


def test_identity_mismatch_is_a_run_error(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch, mislabel={A: C})
    ad = _adapter(tmp_path, tc.build)
    ad.build_missing([ad.cands[n] for n in (A, C)])
    a, c = ad.measure_pairs([_req(K1, A, 1), _req(K2, C, 2)])
    assert a.status == "run_error" and "identity mismatch" in a.diag, a
    assert c.status == "ok"
    assert all(p["check_identity"] is True for p in tc.payloads)


def test_artifact_changed_after_its_build_is_refused(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build)
    ad.build_missing([ad.cands[A]])
    (tmp_path / "out" / "artifacts" / f"{A}.so").write_text(A + " ")
    (o,) = ad.measure_pairs([_req(K1, A, 1)])
    assert o.status == "run_error" and "changed" in o.diag
    assert not tc.loaded and not tc.payloads


def test_rows_and_failures_belong_to_their_request(tmp_path, monkeypatch):
    # Two kernels on two problems, B wrong on P2 only: every row and failure
    # record goes to its own request, and the confirmation writes its own files.
    tc = Toolchain(tmp_path, monkeypatch, wrong={(B, P2["M"])})
    ad = _adapter(tmp_path, tc.build)
    ad.build_missing([ad.cands[n] for n in (A, B)])
    out = ad.measure_pairs([_req(K1, A, 1), _req(K2, A, 2), _req(K1, B, 3),
                            _req(K2, B, 4)])
    assert [o.status for o in out] == ["ok", "ok", "ok", "numeric_fail"]
    (o,) = ad.confirm([AS.PairRequest(K2, A, A, f"{K2}:2", "c5")])
    assert o.status == "ok"
    names = sorted(p.name for p in (tmp_path / "out" / "rounds").iterdir())
    assert names == ["0001-build.json", "0002-measure.csv", "0002-measure.json",
                     "0003-confirm.csv", "0003-confirm.json"]


def test_attribution_holds_across_devices(tmp_path, monkeypatch):
    # Two device threads take the single-kernel units in any order; each
    # outcome still carries its own latency or its own failure.
    def ms(name, prob):
        return prob["M"] + 1.0 + [A, B, C].index(name)

    def run_batch(device_id, unit, args, worker_path, base_env, extra=None,
                  failures=None, workers=None):
        prob_idx, prob, batch = unit
        rows, n_fail = [], 0
        for _, cfg, _ in batch:
            base = {"kernel": cfg.name, "problem_idx": prob_idx, **prob,
                    "device": device_id}
            if (cfg.name, prob["M"]) == (B, P2["M"]):
                n_fail += 1
                failures.append({**base, "kind": "timeout", "error": "over 30 s"})
                continue
            rows.append({**base, "latency_ms": ms(cfg.name, prob), "tflops": 1.0,
                         "non_zero": 1, "max_rel": 0.0, "verified": True})
        return rows, [], n_fail, 0

    monkeypatch.setattr(drv, "_run_batch_on_device", run_batch)
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, devices=("0", "1"))
    ad.build_missing([ad.cands[n] for n in (A, B, C)])
    reqs = [_req(k, n, i) for i, (n, k) in enumerate(
        ((n, k) for n in (A, B, C) for k in (K1, K2)), 1)]
    got = {(o.request.problem_key, o.request.kernel_name): (o.status, o.latency_ms)
           for o in ad.measure_pairs(reqs)}
    prob = {K1: P1, K2: P2}
    assert got == {(k, n): ("timeout", None) if (n, k) == (B, K2)
                   else ("ok", ms(n, prob[k])) for n in (A, B, C) for k in (K1, K2)}


def test_workers_share_the_host_cache_with_unchanged_outcomes(tmp_path, monkeypatch):
    # The real worker gets the run's host cache in its payload and fills one
    # entry per problem; outcomes and rows (latency, max_rel, verified) are
    # those of workers without it. B is wrong on P2.
    reqs = [_req(k, n, i) for i, (k, n) in enumerate(
        ((K1, A), (K2, A), (K1, B), (K2, B)), 1)]
    got = {}
    for name in ("plain", "cached"):
        root = tmp_path / name
        root.mkdir()
        tc = Toolchain(root, monkeypatch, wrong={(B, P2["M"])})
        with HC.HostCache(root) as cache:
            ad = _adapter(root, tc.build, cache=cache if name == "cached" else None)
            ad.build_missing([ad.cands[n] for n in (A, B)])
            outs = ad.measure_pairs(reqs) + ad.measure_pairs(reqs)
            outs += ad.confirm([AS.PairRequest(K2, A, A, f"{K2}:2", "c9")])
            entries = HC.usage(cache.dir)[0]
            specs = [p.get("host_cache") for p in tc.payloads]
        with open(root / "out" / "rounds" / "0002-measure.csv", newline="") as f:
            rows = [{k: v for k, v in r.items() if k != "device"}
                    for r in csv.DictReader(f)]
        got[name] = [(o.status, o.latency_ms, o.diag) for o in outs], rows
        if name == "cached":
            assert entries == 2 and specs == [cache.spec] * len(specs) == [
                ad.extra["host_cache"]] * 9
        else:
            assert entries == 0 and specs == [None] * 9
    assert got["plain"] == got["cached"]
    assert [o[0] for o in got["plain"][0]] == ["ok", "ok", "ok", "numeric_fail"] * 2 + ["ok"]


def test_first_calls_find_their_problem_prepared(tmp_path, monkeypatch):
    # With the host cache, each problem is prepared once, before its first
    # call and outside its timeout (one worker per new problem, at most one
    # per device at a time): no measure worker generates inputs, and the
    # call's timeout keeps the per-dtype allowance.
    made, compute = [], worker._search_host_data
    monkeypatch.setattr(worker, "_search_host_data",
                        lambda r, p: made.append(p.M) or compute(r, p))
    tc = Toolchain(tmp_path, monkeypatch)
    seen = []
    with HC.HostCache(tmp_path) as cache:
        def measure(*a, **kw):
            seen.append((a[4].kernel_timeout, HC.usage(cache.dir)[0], len(made)))
            return drv.benchmark_on_devices(*a, **kw)

        ad = _adapter(tmp_path, tc.build, measure, devices=("0", "1"), cache=cache)
        ad.build_missing([ad.cands[n] for n in (A, B)])
        outs = ad.measure_pairs([_req(K1, A, 1), _req(K1, B, 2)])
        outs += ad.measure_pairs([_req(K1, A, 3), _req(K2, B, 4)])
        outs += ad.confirm([_req(K2, A, 5)])
    assert [o.status for o in outs] == ["ok"] * 5
    assert seen == [(31, 1, 1), (31, 2, 2), (31, 2, 2)] and made == [64, 32]
    assert [(list(p["payloads"]), p["limits"], p["width"]) for p in tc.prepares] == [
        ([K1], {K1: 70}, 2), ([K2], {K2: 70}, 2)]
    assert tc.prepares[0]["payloads"][K1] == {
        "prepare": True, "host_cache": cache.spec, "problem": P1, "kernel_name": A,
        "arch": "gfx950"}
    rounds = tmp_path / "out" / "rounds"
    sides = [json.loads((rounds / n).read_text())["prepared"] for n in (
        "0002-measure.json", "0003-measure.json", "0004-confirm.json")]
    rec = {"limit_s": 70, "ok": True, "stored": True, "seconds": 0.0, "error": ""}
    assert sides == [[{"problem": K1, "kernel": A, **rec}],
                     [{"problem": K2, "kernel": B, **rec}], []]


@pytest.mark.parametrize("case, record, want", [
    ("cache off", None, 365),
    ("stored", {"ok": True, "stored": True, "seconds": 339.0, "error": ""}, 365),
    ("over the caps", {"ok": True, "stored": False, "seconds": 339.0, "error": ""}, 678),
    ("over the caps, 2 devices", {"ok": True, "stored": False, "seconds": 339.0,
                                  "error": ""}, 1356),
    ("failed", {"ok": False, "stored": False, "seconds": 3710.0,
                "error": "no result in 3710 s"}, 365),
])
def test_unprepared_problems_keep_at_least_the_per_dtype_allowance(
        tmp_path, monkeypatch, capsys, case, record, want):
    # fp8 256x768x874159 has a 365 s host allowance, at rates with about 2x
    # headroom over the host they were measured on; on a host about 2x slower
    # than that one, its host data took 339 s. Prepared, those 339 s are spent
    # before the call, which keeps 30 + 365 s for loading the entry and running
    # the kernel. Without an entry (the cache off, or
    # over its caps) every call computes it all: the allowance stays, or is
    # twice what the preparation took on this host, times the devices whose
    # calls then share the CPUs, when that is more. A problem is prepared
    # once, whatever the outcome.
    seen, preps = [], []
    tc = Toolchain(tmp_path, monkeypatch)
    monkeypatch.setattr(AS, "prepare_host_data", lambda argv, env, payloads, lim, w, **kw: (
        preps.append(lim) or {k: dict(record) for k in payloads}))
    with HC.HostCache(tmp_path) as cache:
        ad = _adapter(tmp_path, tc.build, lambda *a, **kw: seen.append(a[4].kernel_timeout),
                      problems=(P1, SKINNY), dtype="fp8",
                      cache=None if case == "cache off" else cache,
                      devices=("0", "1") if "2 devices" in case else ("0",))
        ad.memory = lambda: None  # no guard: only the timeouts are checked here
        ad.build_missing([ad.cands[A]])
        for n in (1, 2):
            ad.measure_pairs([_req("256x768x874159", A, n)])
    assert seen == [30 + want] * 2
    assert preps == ([] if case == "cache off" else [{"256x768x874159": 3710}])
    out = capsys.readouterr().out
    assert ("not stored" in out) == case.startswith(("over the caps", "failed"))


BIG = {"M": 1024, "N": 1024, "K": 65536}  # fp8: 0.6 GiB of host data and 1 GiB fixed
KBIG = "1024x1024x65536"


def _between(n):
    """MemAvailable (GiB, with 8 GiB in all) for which ``n`` workers computing
    P1's fp8 host data at once fit, and ``n`` computing BIG's do not."""
    p1, big = (GS.host_compute_bytes(p, "fp8") for p in (P1, BIG))
    return n * (p1 + big) / 2 / AS.GIB + AS.HOST_MEMORY_MARGIN * 8


KILLED = AS._prepared(-9, "", 197.0, "")  # by another process: the OOM killer


@pytest.mark.parametrize("case, devices, avail, cached", [
    ("preparation", 1, _between(1), True),
    ("calls without an entry", 2, _between(2), "refused"),
    ("cache off", 2, _between(2), False),
    ("cache off, one device", 1, _between(2), False),
    ("preparation killed", 1, _between(2), "killed"),
])
def test_host_memory_guard_skips_a_problem_that_does_not_fit(
        tmp_path, monkeypatch, capsys, case, devices, avail, cached):
    # The guard reads MemAvailable less 10% of MemTotal (8 GiB here) when a
    # problem's first call comes. Its preparation, and then up to one call
    # per device at once (reading the entry, or computing it all without
    # one), must fit; a problem that does not is never prepared nor
    # measured: its requests get host_memory and one console line names it.
    # BIG fits on one device in the last two cases; in the last, its
    # preparation was killed from outside (exit code -9): its calls, which
    # would compute the same data, are not started either.
    monkeypatch.setattr(GS.host_cache, "fs_type", lambda path: "ext4")
    tc = Toolchain(tmp_path, monkeypatch)
    if cached in ("refused", "killed"):
        refused = AS._prepared(0, json.dumps({"ok": True, "stored": False}), 1.0, "")
        monkeypatch.setattr(AS, "prepare_host_data", lambda argv, env, jobs, *a, **kw: {
            k: KILLED if cached == "killed" and k == KBIG else refused for k in jobs})
    seen = []
    with HC.HostCache(tmp_path) as cache:
        ad = _adapter(tmp_path, tc.build, lambda *a, **kw: seen.extend(a[0]),
                      problems=(P1, BIG), dtype="fp8", cache=cache if cached else None,
                      devices=[str(d) for d in range(devices)])
        ad.memory = _memory(tmp_path, 8, avail)
        ad.build_missing([ad.cands[A]])
        out = ad.measure_pairs([_req(K1, A, 1), _req(KBIG, A, 2), _req(KBIG, A, 3)])
        out += ad.measure_pairs([_req(KBIG, A, 4)])
    budget = ad.memory()
    each = GS.host_compute_bytes(BIG, "fp8")
    two = AS.host_memory_error(2 * each, budget, (
        f"2 calls at once without a cache entry, {each / AS.GIB:.1f} GiB each"))
    want = {"preparation": AS.host_memory_error(each, budget),
            "calls without an entry": two, "cache off": two,
            "preparation killed": KILLED["error"]}.get(case)
    assert KILLED["error"] == ("host_memory: its preparation was killed (exit code -9), "
                               "likely by the out-of-memory killer")
    if want is None:  # it fits
        assert [o.status for o in out] == ["unattempted"] * 4 and seen == [P1, BIG, BIG, BIG]
        return
    assert [(o.status, o.diag) for o in out[1:]] == [("host_memory", want)] * 3
    assert seen == [P1] and ad.skipped == {KBIG: want}
    console = capsys.readouterr().out
    assert console.count(f"    {KBIG}: skipped, {want}\n") == 1
    calls = [json.loads(p.read_text()) for p in sorted(
        (tmp_path / "out" / "rounds").glob("*-measure.json"))]
    assert [c["host_memory"] for c in calls] == [{"d2": want, "d3": want}, {"d4": want}]
    assert [r["problem"] for c in calls for r in c["requests"]] == [K1]
    if cached is True:  # the preparation of P1 ran, that of BIG did not start
        assert tc.prepares[0]["needs"] == {K1: GS.host_compute_bytes(P1, "fp8"),
                                           KBIG: each}
        assert [(r["problem"], r["stored"], r["error"]) for r in calls[0]["prepared"]] == [
            (K1, True, ""), (KBIG, False, want)]


@pytest.mark.parametrize("fs", ["tmpfs", "ext4"])
def test_preparation_on_a_tmpfs_also_needs_its_entry(tmp_path, monkeypatch, fs):
    # A tmpfs holds the entries in RAM: written while the preparation still
    # holds its arrays, the entry adds its size to that preparation's need.
    monkeypatch.setattr(GS.host_cache, "fs_type", lambda path: fs)
    tc = Toolchain(tmp_path, monkeypatch)
    with HC.HostCache(tmp_path) as cache:
        ad = _adapter(tmp_path, tc.build, lambda *a, **kw: None, problems=(P1, BIG),
                      dtype="bf16", cache=cache)
        ad.memory = lambda: None  # no guard: only the needs passed are checked
        ad.build_missing([ad.cands[A]])
        ad.measure_pairs([_req(K1, A, 1)])
    entry = GS.host_entry_bytes(P1, "bf16") if fs == "tmpfs" else 0
    assert tc.prepares[0]["needs"] == {K1: GS.host_compute_bytes(P1, "bf16") + entry}
    assert tc.prepares[0]["keeps"] == {K1: entry}  # it stays once the worker ends
    # bf16 BIG: 2 B per input element and 4 B per output element stored, plus
    # 4 B per input element while the reference is computed and 12 B per
    # output element for C and its comparison.
    assert GS.host_entry_bytes(BIG, "bf16") == 2 * 2048 * 65536 + 4 * 2**20
    assert GS.host_compute_bytes(BIG, "bf16") == (
        6 * 2048 * 65536 + 16 * 2**20 + GS.HOST_FIXED_BYTES)


@pytest.mark.parametrize("fs", ["tmpfs", "ext4"])
def test_calls_keep_memory_for_the_tmpfs_entries_to_come(tmp_path, monkeypatch, fs):
    # P1's calls compute its host data (the cache refused its entry). On a
    # tmpfs, the entries still to come (P1's and BIG's, within the cache's
    # room) will hold RAM while P1's calls run in later rounds, so the calls
    # check keeps them: P1 fits without them, not with them (the next test
    # checks what is kept).
    monkeypatch.setattr(GS.host_cache, "fs_type", lambda path: fs)
    monkeypatch.setattr(HC.shutil, "disk_usage", lambda p: SimpleNamespace(
        total=1 << 50, used=0, free=1 << 50))
    tc = Toolchain(tmp_path, monkeypatch)
    monkeypatch.setattr(AS, "prepare_host_data", lambda argv, env, jobs, *a, **kw: {
        k: AS._prepared(0, json.dumps({"ok": True, "stored": False}), 1.0, "")
        for k in jobs})
    seen = []
    with HC.HostCache(tmp_path, max_total_bytes=1 << 30) as cache:
        kept = sum(GS.host_entry_bytes(p, "fp8") for p in (P1, BIG))
        need = GS.host_compute_bytes(P1, "fp8")
        ad = _adapter(tmp_path, tc.build, lambda *a, **kw: seen.extend(a[0]),
                      problems=(P1, BIG), dtype="fp8", cache=cache)
        ad.memory = lambda: need + kept // 2
        ad.build_missing([ad.cands[A]])
        out = ad.measure_pairs([_req(K1, A, 1)])
    if fs == "ext4":
        assert seen == [P1] and not ad.skipped
        return
    want = AS.host_memory_error(need, need + kept // 2 - kept, (
        f"a call without a cache entry; {kept / AS.GIB:.1f} GiB kept for the tmpfs "
        "cache's entries to come"))
    assert seen == [] and [(o.status, o.diag) for o in out] == [("host_memory", want)]


P3, P4 = {"M": 48, "N": 32, "K": 16}, {"M": 16, "N": 16, "K": 16}
K3, K4 = "48x32x16", "16x16x16"


@pytest.mark.parametrize("max_total, kept, keep", [
    (1 << 30, 9728 + 7424, 9728),  # P1's and P3's entries
    (12000, 12000, 9728),  # no more than the room
    (9000, 7424, 0),  # P1's entry does not fit in the room: P3's alone
])
def test_calls_keep_the_entries_to_come_by_value(tmp_path, monkeypatch, max_total,
                                                 kept, keep):
    # On a tmpfs, a preparation needs its entry when the cache may store it
    # (within max_entry_bytes and the room), and the calls check keeps the
    # entries still to come: those of the problems neither stored nor
    # skipped, each within the cap and the room, and no more than the room.
    # fp8 entries: P1 and P2 9728 B, P3 7424, P4 1536, BIG 132 MiB (over the
    # 1 MiB cap). P2 is stored (MemAvailable lacks it already), P4 skipped.
    monkeypatch.setattr(GS.host_cache, "fs_type", lambda path: "tmpfs")
    monkeypatch.setattr(HC.shutil, "disk_usage", lambda p: SimpleNamespace(
        total=1 << 50, used=0, free=1 << 50))
    tc = Toolchain(tmp_path, monkeypatch)
    keeps, seen = [], []
    monkeypatch.setattr(AS, "prepare_host_data", lambda argv, env, jobs, *a, **kw: (
        keeps.append(kw["keeps"]) or {
            k: AS._prepared(0, json.dumps({"ok": True, "stored": False}), 1.0, "")
            for k in jobs}))
    check = AS.calls_memory_error
    monkeypatch.setattr(AS, "calls_memory_error", lambda *a: seen.append(a[4]) or check(*a))
    with HC.HostCache(tmp_path, max_entry_bytes=1 << 20, max_total_bytes=max_total) as cache:
        ad = _adapter(tmp_path, tc.build, lambda *a, **kw: None,
                      problems=(P1, P2, P3, P4, BIG), dtype="fp8", cache=cache)
        ad.memory = lambda: 1 << 40
        ad.prepared[K2] = {"ok": True, "stored": True, "seconds": 0.0, "error": ""}
        ad.skipped[K4] = "host_memory: (a test)"
        ad.checked.update((K2, K4))
        ad.build_missing([ad.cands[A]])
        ad.measure_pairs([_req(K1, A, 1)])
    assert [GS.host_entry_bytes(p, "fp8") for p in (P1, P2, P3, P4)] == [9728, 9728,
                                                                         7424, 1536]
    assert keeps == [{K1: keep}] and seen == [kept]


def test_a_stored_entry_whose_calls_do_not_fit_is_dropped(tmp_path, monkeypatch,
                                                          capsys):
    # P2's entry is stored and its calls fit. Later the host has less memory:
    # P1's preparation stores its entry, then its 4 calls at once do not fit
    # (1 GiB each), so P1 is skipped and its entry, which no call will read,
    # is removed (on a tmpfs it would hold RAM for the rest of the run).
    monkeypatch.setattr(GS.host_cache, "fs_type", lambda path: "tmpfs")
    tc = Toolchain(tmp_path, monkeypatch)
    with HC.HostCache(tmp_path) as cache:
        ad = _adapter(tmp_path, tc.build, lambda *a, **kw: None, problems=(P1, P2),
                      devices=[str(d) for d in range(4)], cache=cache)
        ad.build_missing([ad.cands[A]])
        ad.memory = lambda: 8 * AS.GIB
        assert ad.measure_pairs([_req(K2, A, 1)])[0].status == "unattempted"
        ad.memory = lambda: 2 * AS.GIB
        (out,) = ad.measure_pairs([_req(K1, A, 2)])
        assert [ad.prepared[k]["stored"] for k in (K2, K1)] == [True, True]
        assert out.status == "host_memory" and "4 calls at once reading its entry" in out.diag
        assert list(ad.skipped) == [K1] and HC.usage(cache.dir)[0] == 1
        (entry,) = [p for p in cache.dir.iterdir() if p.is_dir()]
        assert json.loads((entry / "meta.json").read_text())["fields"]["problem"] == P2
        assert list(HC.claims(cache.dir)) == [entry.name]
    assert capsys.readouterr().out.count(f"    {K1}: skipped, host_memory: ") == 1


def test_preparations_never_overlap_builds(tmp_path, monkeypatch):
    # A run's rounds build, then prepare and measure. Across the search runs
    # of a host, a round's builds hold the host lock shared and its
    # preparations exclusive: a preparation of another run waits for them.
    tc = Toolchain(tmp_path, monkeypatch)
    spans, build, prepare = [], tc.build, AS.prepare_host_data

    def held_against(mode):
        """Whether another run is refused the host lock in ``mode`` now."""
        fd = os.open(str(tmp_path / AS.HOST_LOCK), os.O_RDONLY)
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
            return False
        except OSError:
            return True
        finally:
            os.close(fd)

    def timed(kind, fn, mode):
        def wrapped(*a, **kw):
            t0, held = time.monotonic(), held_against(mode)
            got = fn(*a, **kw)
            spans.append((kind, t0, time.monotonic(), held))
            return got
        return wrapped

    monkeypatch.setattr(tc, "build", timed("build", build, fcntl.LOCK_EX))
    monkeypatch.setattr(AS, "prepare_host_data", timed("prepare", prepare, fcntl.LOCK_SH))
    out = tmp_path / "out"
    settings = AS.SearchSettings.from_json({
        "schema_version": 1, "strategy": "exhaustive", "trials_per_problem": 3,
        "new_builds_per_run": 4, "batch_per_problem": 1,
        "host_cache": {"dir": str(tmp_path)}})
    rc = GS.run_gemm_search(
        _args(search_output_dir=str(out)), settings, _hooks(tc.build),
        problems=[P1, P2], configs=CFGS, eligible=[[A, B, C]] * 2, devices=["0"],
        worker=(Path("w.py"), {}), context={"arch": "gfx950"})
    assert rc == 0
    kinds = [s[0] for s in spans]
    assert kinds.count("build") == 3 and kinds.count("prepare") == 1
    assert all(held for *_, held in spans)
    ordered = sorted(spans, key=lambda s: s[1])
    assert all(a[2] <= b[1] for a, b in zip(ordered, ordered[1:]))


def test_search_skips_a_problem_whose_host_data_does_not_fit(tmp_path, monkeypatch,
                                                           capsys):
    # End to end: BIG's host data does not fit, so it is never prepared nor
    # measured, its strategy is told nothing and it ends unresolved with the
    # reason; P1 gets its winner, and the run is partial.
    monkeypatch.setattr(GS.host_cache, "fs_type", lambda path: "ext4")
    tc = Toolchain(tmp_path, monkeypatch)
    monkeypatch.setattr(AS, "host_memory", _memory(tmp_path, 8, _between(1)))
    out = tmp_path / "out"
    settings = AS.SearchSettings.from_json({
        "schema_version": 1, "strategy": "exhaustive", "trials_per_problem": 3,
        "new_builds_per_run": 3, "host_cache": {"dir": str(tmp_path)}})
    rc = GS.run_gemm_search(
        _args(search_output_dir=str(out), dtype="fp8"), settings, _hooks(tc.build),
        problems=[P1, BIG], configs=CFGS, eligible=[[A, B, C]] * 2, devices=["0"],
        worker=(Path("w.py"), {}), context={"arch": "gfx950"})
    why = AS.host_memory_error(GS.host_compute_bytes(BIG, "fp8"), AS.host_memory())
    summary = json.loads((out / "summary.json").read_text())
    assert (rc, summary["status"], list(summary["winners"])) == (1, "partial", [K1])
    assert summary["problems"][KBIG]["finished"] == why and why in summary["unresolved"][KBIG]
    assert summary["counts"]["host_memory"] == 1
    assert summary["counts"]["failed_by_status"] == {}
    rows = [json.loads(ln) for ln in (out / "trials.jsonl").read_text().splitlines()]
    assert not [r for r in rows if r["event"] == "result" and r.get("problem") == KBIG]
    assert len([r for r in rows if r["event"] == "host_memory"]) == 3
    assert [p["items"][0]["problem"]["M"] for p in tc.payloads] == [64] * 9  # BIG: never
    assert capsys.readouterr().out.count(f"{KBIG}: skipped, {why}") == 1


@pytest.mark.parametrize("how", ["killed", "oom"])
def test_a_call_out_of_host_memory_stops_its_problem_and_blames_no_kernel(
        tmp_path, monkeypatch, capsys, how):
    # The guard let P1's calls run, then its call of B ran out of host memory:
    # its worker was killed from outside (exit code -9: the driver kills one
    # only at its timeout) or raised MemoryError. That is the host's failure,
    # not B's: P1's strategy is told nothing of B and P1 stops (C is never
    # measured), but A, measured before, is still confirmed and wins. P1 is
    # not skipped, and P2 runs on to its winner.
    monkeypatch.setattr(GS.host_cache, "fs_type", lambda path: "ext4")
    tc = Toolchain(tmp_path, monkeypatch, **{how: {(B, P1["M"])}})
    out = tmp_path / "out"
    settings = AS.SearchSettings.from_json({
        "schema_version": 1, "strategy": "exhaustive", "trials_per_problem": 3,
        "new_builds_per_run": 3, "batch_per_problem": 1,
        "host_cache": {"dir": str(tmp_path)}})
    rc = GS.run_gemm_search(
        _args(search_output_dir=str(out)), settings, _hooks(tc.build),
        problems=[P1, P2], configs=CFGS, eligible=[[A, B, C]] * 2, devices=["0"],
        worker=(Path("w.py"), {}), context={"arch": "gfx950"})
    why = KILLED_CALL if how == "killed" else (
        "host_memory: its call's worker ran out of memory (MemoryError: )")
    summary = json.loads((out / "summary.json").read_text())
    assert (rc, summary["status"], sorted(summary["winners"])) == (0, "complete", [K2, K1])
    assert summary["winners"][K1]["kernel"] == A
    assert summary["problems"][K1]["finished"] == why
    assert summary["counts"]["host_memory"] == 1
    assert summary["counts"]["failed_by_status"] == {}
    rows = [json.loads(ln) for ln in (out / "trials.jsonl").read_text().splitlines()]
    assert [r["kernel"] for r in rows if r["event"] == "result" and r["problem"] == K1] == [A]
    assert [(r["kernel"], r["diag"]) for r in rows if r["event"] == "host_memory"] == [
        (B, why)]
    ran = [(it["kernel_name"], it["problem"]["M"]) for p in tc.payloads for it in p["items"]]
    assert (C, P1["M"]) not in ran and ran.count((A, P1["M"])) > 1  # A confirmed
    assert "skipped" not in capsys.readouterr().out


def test_verified_calls_get_time_for_the_host_reference(tmp_path, monkeypatch):
    # --kernel-timeout covers the kernel; a verified one-kernel call also builds
    # the inputs and the fp32 reference on the host: 8192^3 gets 33 s more
    # (22 s of reference, 10 s of inputs).
    big = {"M": 8192, "N": 8192, "K": 8192}
    seen = []

    def measure(problems, pairs, built, devices, args, *rest, **kw):
        seen.append(args.kernel_timeout)

    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, measure, problems=(P1, big), kernel_timeout=30)
    ad.build_missing([ad.cands[A]])
    ad.measure_pairs([_req(K1, A, 1)])
    ad.measure_pairs([_req(K1, A, 2), _req("8192x8192x8192", A, 3)])
    assert seen == [31, 63] and ad.args.kernel_timeout == 30


SKINNY = {"M": 256, "N": 768, "K": 874159}


@pytest.mark.parametrize("dtype,want", [
    ("fp16", [1, 33, 75]), ("bf16", [1, 33, 75]), ("fp32", [1, 33, 75]),
    ("fp8", [1, 76, 365]), ("bf8", [1, 76, 365]), ("fp4", [1, 76, 365]),
])
def test_host_allowance_is_per_input_dtype(tmp_path, monkeypatch, dtype, want):
    # A skinny large-K shape is mostly inputs: 75 s, of which 7 s reference.
    # fp8 and bf8 inputs cost about 4x a 16-bit one on the host (their
    # encoding): GEMM's rates are 400 ns per element against 75 ns, so the same
    # fp8 shape gets 365 s. A dtype without a rate of its own gets the slowest.
    big = {"M": 8192, "N": 8192, "K": 8192}
    assert [GS.verify_allowance_s(p, dtype) for p in (P1, big, SKINNY)] == want
    seen = []
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, lambda *a, **kw: seen.append(a[4].kernel_timeout),
                  problems=(P1, SKINNY), kernel_timeout=30, dtype=dtype)
    ad.build_missing([ad.cands[A]])
    ad.measure_pairs([_req(K1, A, 1), _req("256x768x874159", A, 2)])
    assert seen == [30 + want[2]]
    rate = "4e-07" if want[2] == 365 else "7.5e-08"
    assert ad.context()["measurement"]["timeout_s"] == (
        f"kernel_timeout + ceil(2*M*N*K / 5e+10 + (M+N)*K * {rate}) per call")


def test_confirmation_runs_each_problem_on_one_device(tmp_path, monkeypatch):
    # Two devices, two problems with two finalists each: every confirmation
    # unit of a problem runs on that problem's device (K1 on 0, K2 on 1), and
    # each outcome says where it ran.
    ran = []

    def run_batch(device_id, unit, args, worker_path, base_env, extra=None,
                  failures=None, workers=None):
        prob_idx, prob, batch = unit
        ran.append((device_id, prob["M"]))
        return [{"kernel": cfg.name, "problem_idx": prob_idx, **prob,
                 "device": device_id, "latency_ms": 1.0, "tflops": 1.0, "non_zero": 1,
                 "max_rel": 0.0, "verified": True} for _, cfg, _ in batch], [], 0, 0

    monkeypatch.setattr(drv, "_run_batch_on_device", run_batch)
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, devices=("0", "1"))
    ad.build_missing([ad.cands[n] for n in (A, B)])
    reqs = [AS.PairRequest(k, n, n, f"{k}:{n}", f"c{i}")
            for i, (k, n) in enumerate((k, n) for k in (K1, K2) for n in (A, B))]
    for _ in range(4):
        assert [o.device for o in ad.confirm(reqs)] == ["0", "0", "1", "1"]
    assert sorted(set(ran)) == [("0", P1["M"]), ("1", P2["M"])] and len(ran) == 16


@pytest.mark.parametrize("confirm_ok", [False, True])
def test_misleading_rows_never_become_winners(tmp_path, monkeypatch, confirm_ok):
    # Fixture 2: a fast row with verified "False", one with a blank verified and
    # a slower valid one. Only the valid one is a finalist; when its fresh calls
    # fail there is no winner, never a fallback to the invalid rows.
    fields = {A: {"latency_ms": 0.1, "verified": "False"},
              B: {"latency_ms": 0.2, "verified": ""},
              C: {"latency_ms": 0.9, "verified": True}}
    calls = []

    def measure(problems, pairs, built, devices, args, wp, env, csv_path,
                extra=None, *, record_failures=None, measured=None, **kw):
        confirm = "confirm" in Path(csv_path).name
        for i, (prob, idx) in enumerate(zip(problems, pairs)):
            name = built[idx[0]][0].name
            calls.append((confirm, name))
            base = {"kernel": name, "problem_idx": i, **prob, "device": "0"}
            if confirm and not confirm_ok:
                record_failures([{**base, "kind": "error",
                                  "error": "kernel returned status -1"}])
            else:
                measured.append({**base, "tflops": 1.0, "non_zero": 1, "max_rel": "",
                                 **fields[name]})

    tc = Toolchain(tmp_path, monkeypatch)
    out = tmp_path / "out"
    settings = AS.SearchSettings.from_json({
        "schema_version": 1, "strategy": "exhaustive", "trials_per_problem": 3,
        "new_builds_per_run": 3, "confirm": {"finalists": 2, "fresh_calls": 2}})
    rc = GS.run_gemm_search(
        _args(search_output_dir=str(out)), settings, _hooks(tc.build, measure),
        problems=[P1], configs=CFGS, eligible=[[A, B, C]], devices=["0"],
        worker=(Path("w.py"), {}), context={"arch": "gfx950"},
    )
    assert sorted(n for confirm, n in calls if not confirm) == sorted([A, B, C])
    assert {n for confirm, n in calls if confirm} == {C}
    results = {e["kernel"]: e["status"] for e in map(
        json.loads, (out / "trials.jsonl").read_text().splitlines())
        if e["event"] == "result"}
    assert results == {A: "numeric_fail", B: "invalid_measurement", C: "ok"}
    winners = drv.load_pair_list(out / "winners.json")
    summary = json.loads((out / "summary.json").read_text())
    if confirm_ok:
        assert (rc, winners, summary["status"]) == (0, {K1: [C]}, "complete")
    else:
        assert (rc, winners, summary["status"]) == (1, {}, "no_winner")
        assert "no finalist passed confirmation" in summary["unresolved"][K1]


def test_worker_checks_the_library_identity_only_when_asked(monkeypatch, capsys):
    monkeypatch.setattr(gemm_utils, "_SWEEP_CACHE", {})
    monkeypatch.setattr(worker, "GpuGemmRunner",
                        lambda lib_path, arch=None: _Runner(C, 1.0))
    worker._run_one(0, "a.so", P1, A, verify=True)
    worker._run_one(1, "a.so", P1, A, verify=True, check_identity=True)
    monkeypatch.setattr(worker, "GpuGemmRunner",
                        lambda lib_path, arch=None: _Runner(A, 1.0, status=-3))
    worker._run_one(2, "a.so", P1, A, check_identity=True)
    legacy, checked, status = (
        json.loads(ln) for ln in capsys.readouterr().out.splitlines())
    assert legacy["ok"] and legacy["verified"] and legacy["kernel"] == A
    assert set(legacy) == {"idx", "ok", "ms", "tflops", "non_zero", "kernel",
                           "max_rel", "verified"}
    assert not checked["ok"] and "identity mismatch" in checked["error"]
    assert repr(C) in checked["error"] and repr(A) in checked["error"]
    assert status == {"idx": 2, "ok": False, "error": "kernel returned status -3",
                      "kernel": A}


def test_legacy_worker_payload_and_rows_are_unchanged(tmp_path, monkeypatch):
    # A run without the search sends the worker what it always did (no
    # check_identity) and gets back the keys it always got.
    tc = Toolchain(tmp_path, monkeypatch)
    (lib,) = tc.build([CFGS[0]])
    rows = []
    drv.benchmark_on_devices([P1], [[0]], [(CFGS[0], lib)], ["0"], _args(verify=True),
                             *drv.worker_setup(), tmp_path / "legacy.csv", measured=rows)
    (payload,) = tc.payloads
    assert set(payload) == {"items", "verify", "verify_tol"}
    assert set(tc.outputs[0]) == {"idx", "ok", "ms", "tflops", "non_zero", "kernel",
                                  "max_rel", "verified"}
    (row,) = rows
    assert row["verified"] is True
    with open(tmp_path / "legacy.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames == [
            "kernel", "problem_idx", "M", "N", "K", "device", "latency_ms", "tflops",
            "non_zero", "max_rel", "verified"]


def test_worker_checks_the_identity_of_a_multi_abd_library(monkeypatch, capsys):
    ran = []

    class MultiABD:
        kernel_name = "gemm_fp16_rcr_other_multiabd_x"

        def __init__(self, lib_path, **kw):
            pass

        def run(self, problem, verify=False, verify_tol=None):
            ran.append(problem)

    monkeypatch.setattr(worker, "GpuMultiABDRunner", MultiABD)
    worker._run_one(0, "m.so", P1, "gemm_fp16_rcr_multiabd_x", check_identity=True)
    (out,) = (json.loads(ln) for ln in capsys.readouterr().out.splitlines())
    assert not out["ok"] and "identity mismatch" in out["error"] and not ran
