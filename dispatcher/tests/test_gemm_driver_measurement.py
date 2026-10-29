#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Measurement options of the tile_engine GEMM sweep driver (gemm_full_benchmark.py):
--build-timeout reaches every codegen/compile/link step, --failures-csv records each
failed measurement and build with its reason, --repeat-top re-times the fastest
kernels per problem in separate launches, and none of them changes a run that does
not ask for it.

No GPU or compiler needed: the build and the GPU batch runner are stubbed.
Run: python3 -m pytest -q tests/test_gemm_driver_measurement.py
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import ctypes_utils as cu  # noqa: E402
import gemm_full_benchmark as drv  # noqa: E402
import gemm_utils as gu  # noqa: E402

PROBLEMS = [{"M": 512, "N": 512, "K": 512}, {"M": 1024, "N": 512, "K": 256}]


# ------------------------------------------------------------- build timeouts


class _Run:
    """subprocess.run stand-in recording each timeout; optionally times out."""

    def __init__(self, expire_on=None):
        self.timeouts, self.expire_on, self.calls = [], expire_on, 0

    def __call__(self, cmd, capture_output, text, timeout):
        self.calls += 1
        self.timeouts.append(timeout)
        if self.calls == self.expire_on:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 0, "", "")


def _hipcc_job(tmp_path, **extra):
    return {
        "compile_cmd": ["cc"],
        "link_cmd": ["ld"],
        "lib_path": str(tmp_path / "lib.so"),
        **extra,
    }


def test_hipcc_step_limits_default_to_300s(monkeypatch, tmp_path):
    run = _Run()
    monkeypatch.setattr(subprocess, "run", run)
    assert cu._run_hipcc_subprocess(_hipcc_job(tmp_path))[0]
    assert run.timeouts == [300, 300]


def test_hipcc_step_limits_are_configurable(monkeypatch, tmp_path):
    run = _Run()
    monkeypatch.setattr(subprocess, "run", run)
    job = _hipcc_job(tmp_path, compile_timeout=7, link_timeout=9)
    assert cu._run_hipcc_subprocess(job)[0]
    assert run.timeouts == [7, 9]


@pytest.mark.parametrize("expire_on, step, limit", [(1, "compile", 7), (2, "link", 9)])
def test_hipcc_timeout_names_the_step_and_limit(
    monkeypatch, tmp_path, expire_on, step, limit
):
    monkeypatch.setattr(subprocess, "run", _Run(expire_on=expire_on))
    job = _hipcc_job(tmp_path, compile_timeout=7, link_timeout=9)
    ok, lib, err = cu._run_hipcc_subprocess(job)
    assert not ok and lib is None
    assert err == f"Timeout: {step} exceeded {limit} s"


@pytest.mark.parametrize("extra, expected", [({}, 300), ({"timeout": 11}, 11)])
def test_codegen_step_limit(monkeypatch, tmp_path, extra, expected):
    run = _Run(expire_on=1)
    monkeypatch.setattr(subprocess, "run", run)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(cfg_dir))
    args = {
        "output_dir": str(tmp_path),
        "tile_config_json": {},
        "python": "py",
        "codegen_script": "gen.py",
        "dtype": "fp16",
        "layout": "rcr",
        "gpu_target": "gfx950",
        "hpp_glob_pattern": "x.hpp",
        **extra,
    }
    ok, hdr, err = cu._generate_single_kernel_subprocess(args)
    assert not ok and run.timeouts == [expected]
    # Same form as a compile or link timeout, and the config file is removed.
    assert err == f"Timeout: codegen exceeded {expected} s"
    assert list(cfg_dir.iterdir()) == []


class _SyncPool:
    """ProcessPoolExecutor stand-in that runs each job in-process."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, arg):
        from concurrent.futures import Future

        fut = Future()
        fut.set_result(fn(arg))
        return fut


@pytest.mark.parametrize("build_timeout", [None, 900])
def test_setup_applies_build_timeout_to_every_step(
    monkeypatch, tmp_path, build_timeout
):
    codegen, compile_jobs = [], []
    (tmp_path / "libck_tile_dispatcher.a").touch()
    monkeypatch.setattr(gu, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(gu, "as_completed", lambda futs: list(futs))
    monkeypatch.setattr(cu, "get_build_dir", lambda: tmp_path)
    monkeypatch.setattr(cu, "get_generated_kernels_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cu,
        "_generate_single_kernel_subprocess",
        lambda a: codegen.append(a) or (True, str(tmp_path / "k.hpp"), ""),
    )
    monkeypatch.setattr(
        gu,
        "_build_compile_jobs",
        lambda c, h: (
            {"lib_path": str(tmp_path / f"{c.name}.so"), "compile_timeout": 300},
            None,
        ),
    )
    monkeypatch.setattr(
        cu,
        "_run_hipcc_subprocess",
        lambda job: compile_jobs.append(job) or (True, job["lib_path"], ""),
    )
    cfg = gu.GemmKernelConfig(gfx_arch="gfx950")
    kw = {} if build_timeout is None else {"build_timeout": build_timeout}
    paths = gu.setup_multiple_gemm_dispatchers([cfg], verbose=False, **kw)
    assert paths[0] is not None
    (cg,), (job,) = codegen, compile_jobs
    if build_timeout is None:
        assert "timeout" not in cg and "link_timeout" not in job
        assert job["compile_timeout"] == 300
    else:
        assert cg["timeout"] == 900
        assert job["compile_timeout"] == job["link_timeout"] == 900


def _stub_setup(monkeypatch, tmp_path, codegen_fails=(), compile_fails=()):
    (tmp_path / "libck_tile_dispatcher.a").touch()
    monkeypatch.setattr(gu, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(gu, "as_completed", lambda futs: list(futs))
    monkeypatch.setattr(cu, "get_build_dir", lambda: tmp_path)
    monkeypatch.setattr(cu, "get_generated_kernels_dir", lambda: tmp_path)

    def codegen(a):
        name = a["hpp_glob_pattern"][: -len(".hpp")]
        if name in codegen_fails:
            return False, None, "Timeout: codegen exceeded 300 s"
        return True, str(tmp_path / f"{name}.hpp"), ""

    def hipcc(job):
        if Path(job["lib_path"]).stem in compile_fails:
            return (
                False,
                None,
                "Compile failed: In file included from k.hpp:1:\nk.hpp:9:3: error: boom",
            )
        return True, job["lib_path"], ""

    monkeypatch.setattr(cu, "_generate_single_kernel_subprocess", codegen)
    monkeypatch.setattr(
        gu,
        "_build_compile_jobs",
        lambda c, h: ({"lib_path": str(tmp_path / f"{c.name}.so")}, None),
    )
    monkeypatch.setattr(cu, "_run_hipcc_subprocess", hipcc)


def test_setup_reports_each_failed_build(monkeypatch, tmp_path):
    cfgs = [gu.GemmKernelConfig(gfx_arch="gfx950", tile_m=t) for t in (64, 128, 256)]
    _stub_setup(
        monkeypatch,
        tmp_path,
        codegen_fails={cfgs[0].name},
        compile_fails={cfgs[1].name},
    )
    errors = {}
    paths = gu.setup_multiple_gemm_dispatchers(
        cfgs + cfgs[:1], verbose=False, errors=errors
    )
    assert [p is None for p in paths] == [True, True, False, True]
    assert errors == {
        cfgs[0].name: "Timeout: codegen exceeded 300 s",
        cfgs[
            1
        ].name: "Compile failed: In file included from k.hpp:1:\nk.hpp:9:3: error: boom",
    }


@pytest.mark.parametrize(
    "err, line",
    [
        (None, ""),
        ("Timeout: compile exceeded 1200 s", "Timeout: compile exceeded 1200 s"),
        (
            "Compile failed: In file included from k.hpp:1:\nk.hpp:9:3: error: boom\n1 error generated.",
            "Compile failed: k.hpp:9:3: error: boom",
        ),
        (
            "Link failed: ld.lld: error: undefined symbol: f",
            "Link failed: ld.lld: error: undefined symbol: f",
        ),
        ("Codegen failed: ", "Codegen failed"),
        (
            "Expected one .hpp matching x.hpp after codegen, got 0",
            "Expected one .hpp matching x.hpp after codegen, got 0",
        ),
    ],
)
def test_build_failure_summary_is_one_line(err, line):
    assert drv.build_failure_summary(err) == line


# ------------------------------------------------- failure records per batch


class _Proc:
    """subprocess.Popen stand-in returning canned worker output."""

    def __init__(self, out, returncode=0, expire=False):
        self.out, self.returncode, self.expire = out, returncode, expire

    def __call__(self, *a, **kw):
        return self

    def communicate(self, input=None, timeout=None):
        if self.expire:
            self.expire = False
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.out.encode(), b""

    def kill(self):
        pass

    def poll(self):
        return self.returncode


class _Cfg:
    def __init__(self, name):
        self.name, self.variant = name, "standard"


def _batch_args(verify=True):
    return drv.argparse.Namespace(verify=verify, verify_tol=0.02, kernel_timeout=30)


def _unit(n):
    return (
        3,
        {"M": 64, "N": 32, "K": 16},
        [(i, _Cfg(f"k{i}"), "/x.so") for i in range(n)],
    )


def test_batch_failures_are_recorded_with_their_kind(monkeypatch):
    out = "\n".join(
        json.dumps(r)
        for r in [
            {
                "idx": 0,
                "ok": True,
                "ms": 1.0,
                "tflops": 2.0,
                "non_zero": 5,
                "verified": True,
                "max_rel": 0.0,
            },
            {
                "idx": 1,
                "ok": True,
                "ms": 0.5,
                "tflops": 4.0,
                "non_zero": 5,
                "verified": False,
                "max_rel": 0.9,
            },
            {"idx": 2, "ok": False, "error": "kernel returned status -3"},
        ]
    )
    monkeypatch.setattr(subprocess, "Popen", _Proc(out, returncode=1))
    fails = []
    rows, lines, n_fail, n_skip = drv._run_batch_on_device(
        0, _unit(4), _batch_args(), "w.py", {}, failures=fails
    )
    assert [r["kernel"] for r in rows] == ["k0", "k1"] and n_skip == 0
    assert n_fail == 3 == len(fails)
    assert [(f["kernel"], f["kind"]) for f in fails] == [
        ("k1", "mismatch"),
        ("k2", "error"),
        ("k3", "missing"),
    ]
    assert fails[1]["error"] == "kernel returned status -3"
    assert {(f["problem_idx"], f["M"], f["N"], f["K"]) for f in fails} == {
        (3, 64, 32, 16)
    }

    # Without a failure list the call returns exactly the same.
    monkeypatch.setattr(subprocess, "Popen", _Proc(out, returncode=1))
    assert drv._run_batch_on_device(0, _unit(4), _batch_args(), "w.py", {}) == (
        rows,
        lines,
        n_fail,
        n_skip,
    )


def test_batch_unsupported_and_worker_exit_are_recorded(monkeypatch):
    out = "\n".join(
        json.dumps(r)
        for r in [
            {"idx": 0, "ok": False, "skipped": True, "error": "unsupported shape"},
            {"idx": 1, "ok": True, "ms": 1.0, "tflops": 2.0, "non_zero": 5},
        ]
    )
    monkeypatch.setattr(subprocess, "Popen", _Proc(out, returncode=1))
    fails = []
    rows, _, n_fail, n_skip = drv._run_batch_on_device(
        0, _unit(2), _batch_args(verify=False), "w.py", {}, failures=fails
    )
    assert [r["kernel"] for r in rows] == ["k1"]
    # The skip is not a failure; the worker's non-zero exit after reporting
    # every kernel is one failure, recorded without a kernel name.
    assert (n_fail, n_skip) == (1, 1)
    assert [(f["kernel"], f["kind"], f["error"]) for f in fails] == [
        ("k0", "unsupported", "unsupported shape"),
        ("", "worker-exit", "worker exited with code 1"),
    ]


def test_batch_timeout_records_every_kernel(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _Proc("", expire=True))
    fails = []
    _, _, n_fail, _ = drv._run_batch_on_device(
        0, _unit(2), _batch_args(), "w.py", {}, failures=fails
    )
    assert n_fail == 2
    assert [(f["kernel"], f["kind"]) for f in fails] == [
        ("k0", "timeout"),
        ("k1", "timeout"),
    ]
    assert fails[0]["error"] == "batch of 2 exceeded 60 s"


# ----------------------------------------------------------------- main()


def _main(
    monkeypatch, capsys, tmp_path, *argv, fail_build=(), fail_run=(), mismatch=()
):
    """Run the driver with a stub build (fake .so paths) and a stub GPU runner
    whose latency is fixed per kernel name. Returns (rc, out, build kwargs, calls)."""
    probs = tmp_path / "problems.json"
    probs.write_text(json.dumps({"problems": PROBLEMS}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemm_full_benchmark.py",
            "--arch",
            "gfx950",
            "--dtype",
            "fp16",
            "--layout",
            "rcr",
            "--problems",
            str(probs),
            "--csv",
            str(tmp_path / "out.csv"),
            *argv,
        ],
    )
    monkeypatch.setattr(drv, "resolve_devices", lambda spec: ["0"])
    monkeypatch.setattr(drv, "_resolve_arch", lambda a: "gfx950")
    built = []

    def build(cfgs, **kw):
        built.append(dict(kw))
        if kw.get("errors") is not None:
            for c in cfgs:
                if c.name in fail_build:
                    kw["errors"][c.name] = (
                        f"Compile failed: {c.name}.cpp:1:2: error: boom"
                    )
        return [
            None
            if c.name in fail_build or "*" in fail_build
            else Path(f"/fake/{c.name}.so")
            for c in cfgs
        ]

    calls = []

    def run_batch(device_id, unit, args, worker_path, base_env, extra=None, **kw):
        prob_idx, prob, batch = unit
        calls.append((prob_idx, [cfg.name for _, cfg, _ in batch], args.verify, kw))
        failures = kw.get("failures")
        rows, n_fail = [], 0
        for _, cfg, _ in batch:
            if cfg.name in fail_run:
                n_fail += 1
                if failures is not None:
                    failures.append(
                        {
                            "kernel": cfg.name,
                            "problem_idx": prob_idx,
                            **prob,
                            "device": device_id,
                            "kind": "error",
                            "error": "refused",
                        }
                    )
                continue
            ms = 1.0 + (sum(map(ord, cfg.name)) % 97) / 100.0
            rows.append(
                {
                    "kernel": cfg.name,
                    "problem_idx": prob_idx,
                    **prob,
                    "device": device_id,
                    "latency_ms": ms,
                    "tflops": 1.0 / ms,
                    "non_zero": 1,
                    "max_rel": "",
                    "verified": (cfg.name not in mismatch) if args.verify else "",
                }
            )
        return rows, [], n_fail, 0

    monkeypatch.setattr(drv, "setup_multiple_gemm_dispatchers", build)
    monkeypatch.setattr(drv, "_run_batch_on_device", run_batch)
    rc = drv.main()
    return rc, capsys.readouterr().out, built, calls


def _csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_defaults_add_no_files_and_keep_every_call(monkeypatch, capsys, tmp_path):
    rc, out, built, calls = _main(monkeypatch, capsys, tmp_path)
    assert rc == 0 and built == [{"verbose": True, "max_workers": 8}]
    assert calls and all(kw == {} for *_, kw in calls)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.csv", "problems.json"]
    assert "Re-timing" not in out and "Failures:" not in out


def test_build_timeout_reaches_the_build(monkeypatch, capsys, tmp_path):
    _, out, built, _ = _main(monkeypatch, capsys, tmp_path, "--build-timeout", "900")
    assert built[0]["build_timeout"] == 900
    assert "Build timeout: 900s" in out


@pytest.mark.parametrize(
    "argv",
    [
        ("--build-timeout", "0"),
        ("--repeat-top", "-1"),
        ("--repeat-top", "2", "--repeat-count", "0"),
    ],
)
def test_invalid_values_are_rejected(monkeypatch, capsys, tmp_path, argv):
    with pytest.raises(SystemExit):
        _main(monkeypatch, capsys, tmp_path, *argv)


def test_failures_csv_records_builds_and_runs(monkeypatch, capsys, tmp_path):
    _, _, _, calls = _main(monkeypatch, capsys, tmp_path)
    names = sorted({n for _, batch, *_ in calls for n in batch})
    bad_build, bad_run = names[0], names[1]
    fpath = tmp_path / "fail.csv"
    rc, out, _, _ = _main(
        monkeypatch,
        capsys,
        tmp_path,
        "--failures-csv",
        str(fpath),
        fail_build={bad_build},
        fail_run={bad_run},
    )
    assert rc == 1
    rows = _csv(fpath)
    assert {
        "kernel": bad_build,
        "kind": "build-failed",
        "error": f"Compile failed: {bad_build}.cpp:1:2: error: boom",
    }.items() <= rows[0].items()
    runs = [r for r in rows if r["kind"] == "error"]
    assert {r["kernel"] for r in runs} == {bad_run}
    assert {(r["M"], r["N"], r["K"]) for r in runs} <= {
        (str(p["M"]), str(p["N"]), str(p["K"])) for p in PROBLEMS
    }
    assert bad_run not in {r["kernel"] for r in _csv(tmp_path / "out.csv")}


def test_repeat_top_retimes_the_fastest_in_single_launches(
    monkeypatch, capsys, tmp_path
):
    rc, out, _, calls = _main(
        monkeypatch,
        capsys,
        tmp_path,
        "--verify",
        "--repeat-top",
        "2",
        "--repeat-count",
        "3",
    )
    assert rc == 0
    main_rows = _csv(tmp_path / "out.csv")
    reps = _csv(tmp_path / "out.repeats.csv")
    assert len(reps) == len(PROBLEMS) * 2 * 3
    for p in range(len(PROBLEMS)):
        mine = sorted(
            (float(r["latency_ms"]), r["kernel"])
            for r in main_rows
            if r["problem_idx"] == str(p)
        )
        expected = {k for _, k in mine[:2]}
        got = [r for r in reps if r["problem_idx"] == str(p)]
        assert {r["kernel"] for r in got} == expected
        assert sorted(r["rep"] for r in got) == ["0", "0", "1", "1", "2", "2"]
    repeat_calls = calls[-len(reps) :]
    assert all(
        len(batch) == 1 and verify is False for _, batch, verify, _ in repeat_calls
    )
    # rep-major: every kernel's first launch comes before any second launch
    assert [r["rep"] for r in reps][: len(PROBLEMS) * 2] == ["0"] * (len(PROBLEMS) * 2)
    assert "Re-timing: 12 launches ok, 0 failed" in out


def test_repeat_top_skips_verification_failures(monkeypatch, capsys, tmp_path):
    _, _, _, calls = _main(monkeypatch, capsys, tmp_path)
    fastest = min(
        (1.0 + (sum(map(ord, n)) % 97) / 100.0, n)
        for prob_idx, batch, *_ in calls
        if prob_idx == 0
        for n in batch
    )[1]
    _main(
        monkeypatch,
        capsys,
        tmp_path,
        "--verify",
        "--repeat-top",
        "1",
        "--repeat-count",
        "1",
        "--repeat-csv",
        str(tmp_path / "r.csv"),
        mismatch={fastest},
    )
    reps = _csv(tmp_path / "r.csv")
    assert fastest not in {r["kernel"] for r in reps}
    assert len(reps) == len(PROBLEMS)


def test_failures_csv_is_written_when_no_kernel_builds(monkeypatch, capsys, tmp_path):
    fpath = tmp_path / "fail.csv"
    rc, out, _, calls = _main(
        monkeypatch, capsys, tmp_path, "--failures-csv", str(fpath), fail_build={"*"}
    )
    assert rc == 1 and not calls
    assert "No kernels built successfully" in out and f"Failures: {fpath}" in out
    rows = _csv(fpath)
    assert rows and {r["kind"] for r in rows} == {"build-failed"}
    assert len({r["kernel"] for r in rows}) == len(rows)


@pytest.mark.parametrize(
    "argv",
    [
        ("--failures-csv", "{csv}"),
        ("--repeat-top", "1", "--repeat-csv", "{csv}"),
        (
            "--failures-csv",
            "{tmp}/f.csv",
            "--repeat-top",
            "1",
            "--repeat-csv",
            "{tmp}/f.csv",
        ),
        ("--failures-csv", "{tmp}/out.repeats.csv", "--repeat-top", "1"),
        ("--pairs", "{csv}"),
    ],
)
def test_output_files_must_be_distinct(monkeypatch, capsys, tmp_path, argv):
    argv = [a.format(csv=tmp_path / "out.csv", tmp=tmp_path) for a in argv]
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, capsys, tmp_path, *argv)
    assert e.value.code == 2
    assert "name the same file" in capsys.readouterr().err
    assert not (tmp_path / "out.csv").exists()


def test_repeat_csv_alone_does_not_collide(monkeypatch, capsys, tmp_path):
    # --repeat-csv is not written without --repeat-top, so it may name any file.
    rc, *_ = _main(
        monkeypatch, capsys, tmp_path, "--repeat-csv", str(tmp_path / "out.csv")
    )
    assert rc == 0
