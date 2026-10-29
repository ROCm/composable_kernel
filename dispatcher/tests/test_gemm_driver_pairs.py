#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
--list-pairs and --pairs of the tile_engine GEMM sweep driver (gemm_full_benchmark.py).

--list-pairs writes exactly the (problem, kernel) pairs a full run measures, without
building. --pairs builds only the listed kernels and measures only the listed pairs,
judged by the pairing a full run applies; fed a --list-pairs file it reproduces the
full run. The problems include one whose K is not a multiple of the native vector
width, so fixed-vector-width kernels take part.

No GPU or compiler needed: the build and the GPU batch runner are stubbed.
Run: python3 -m pytest -q tests/test_gemm_driver_pairs.py
"""

import csv
import json
import sys
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import gemm_full_benchmark as drv  # noqa: E402

PROBLEMS = [
    {"M": 512, "N": 512, "K": 512},
    {"M": 1024, "N": 512, "K": 250},  # K % 8 != 0: needs fixed-width A/B access
    {"M": 2048, "N": 1024, "K": 256},
]
KEYS = [f"{p['M']}x{p['N']}x{p['K']}" for p in PROBLEMS]


def _main(monkeypatch, capsys, tmp_path, *argv, fail_build=()):
    """Run the driver with a stub build and a stub GPU runner.

    Returns (rc, out, built kernel names or None, {problem key: measured names})."""
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
        built.extend(c.name for c in cfgs)
        return [
            None if c.name in fail_build else Path(f"/fake/{c.name}.so") for c in cfgs
        ]

    measured = {k: set() for k in KEYS}

    def run_batch(device_id, unit, args, worker_path, base_env, extra=None, **kw):
        prob_idx, prob, batch = unit
        measured[KEYS[prob_idx]].update(cfg.name for _, cfg, _ in batch)
        rows = [
            {
                "kernel": cfg.name,
                "problem_idx": prob_idx,
                **prob,
                "device": device_id,
                "latency_ms": 1.0,
                "tflops": 1.0,
                "non_zero": 1,
                "max_rel": "",
                "verified": "",
            }
            for _, cfg, _ in batch
        ]
        return rows, [], 0, 0

    monkeypatch.setattr(drv, "setup_multiple_gemm_dispatchers", build)
    monkeypatch.setattr(drv, "_run_batch_on_device", run_batch)
    rc = drv.main()
    return rc, capsys.readouterr().out, (set(built) if built else None), measured


@pytest.fixture
def full(monkeypatch, capsys, tmp_path):
    """The full run, and its --list-pairs file."""
    rc, _, built, measured = _main(monkeypatch, capsys, tmp_path)
    assert rc == 0 and all(measured.values())
    listing = tmp_path / "pairs.json"
    rc, out, b2, m2 = _main(monkeypatch, capsys, tmp_path, "--list-pairs", str(listing))
    assert rc == 0 and b2 is None and not any(m2.values())
    assert "dry run, nothing built" in out
    return built, measured, json.loads(listing.read_text()), listing


def test_list_pairs_lists_exactly_the_measured_pairs(full):
    built, measured, listing, _ = full
    assert {k: set(v) for k, v in listing["pairs"].items()} == measured
    assert set(listing["kernels"]) == {n for v in measured.values() for n in v}
    assert set(listing["kernels"]) <= built
    assert (listing["arch"], listing["dtype"], listing["layout"]) == (
        "gfx950",
        "fp16",
        "rcr",
    )
    rec = next(iter(listing["kernels"].values()))
    assert set(drv.KERNEL_RECORD_FIELDS) <= set(rec)
    # The misaligned problem is measured with fixed-width kernels only.
    assert all("_vec" in n for n in listing["pairs"][KEYS[1]])


def test_list_pairs_file_as_pairs_reproduces_the_full_run(
    monkeypatch, capsys, tmp_path, full
):
    built, measured, listing, path = full
    rc, _, b, m = _main(monkeypatch, capsys, tmp_path, "--pairs", str(path))
    assert rc == 0 and m == measured
    assert b == set(listing["kernels"])


def test_pairs_builds_the_union_and_measures_only_the_listed(
    monkeypatch, capsys, tmp_path, full
):
    _, _, listing, _ = full
    picks = {k: names[:2] for k, names in listing["pairs"].items()}
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(picks))
    rc, out, b, m = _main(monkeypatch, capsys, tmp_path, "--pairs", str(path))
    assert rc == 0
    assert b == {n for v in picks.values() for n in v}
    assert m == {k: set(v) for k, v in picks.items()}
    assert "Listed pair not measured" not in out


def test_star_measures_what_the_full_run_measures_for_that_problem(
    monkeypatch, capsys, tmp_path, full
):
    _, measured, _, _ = full
    path = tmp_path / "star.json"
    path.write_text(json.dumps({KEYS[1]: "*"}))
    rc, _, b, m = _main(monkeypatch, capsys, tmp_path, "--pairs", str(path))
    assert rc == 0
    assert m[KEYS[1]] == measured[KEYS[1]] and not m[KEYS[0]] and not m[KEYS[2]]
    assert b == measured[KEYS[1]]


def test_star_falls_back_like_the_full_run_when_a_native_fails(
    monkeypatch, capsys, tmp_path, full
):
    _, measured, listing, _ = full
    # Native kernels on the aligned problem whose fixed-width variants are gated
    # there; when they fail to build, a full run measures those variants instead.
    natives = sorted(n for n in measured[KEYS[0]] if "_vec" not in n)
    failing = set(natives[: max(1, len(natives) // 3)])
    _, _, _, full_f = _main(monkeypatch, capsys, tmp_path, fail_build=failing)
    assert full_f[KEYS[0]] - measured[KEYS[0]], "expected fixed-width fallbacks"
    path = tmp_path / "star.json"
    path.write_text(json.dumps({k: "*" for k in KEYS}))
    rc, out, _, star = _main(
        monkeypatch, capsys, tmp_path, "--pairs", str(path), fail_build=failing
    )
    assert star == full_f
    assert "Listed pair not measured" not in out


def test_unpaired_and_unbuilt_picks_are_reported(monkeypatch, capsys, tmp_path, full):
    _, measured, listing, _ = full
    # A fixed-width kernel the full run does not pair with the aligned problem
    # (its native kernel serves it), and a pick whose build fails.
    unpaired = next(
        n
        for n in sorted(listing["kernels"])
        if "_vec" in n and n not in measured[KEYS[0]]
    )
    unbuilt = sorted(measured[KEYS[2]])[0]
    path = tmp_path / "picks.json"
    path.write_text(json.dumps({KEYS[0]: [unpaired], KEYS[2]: [unbuilt]}))
    fails = tmp_path / "fail.csv"
    rc, out, _, m = _main(
        monkeypatch,
        capsys,
        tmp_path,
        "--pairs",
        str(path),
        "--failures-csv",
        str(fails),
        fail_build={unbuilt},
    )
    assert not any(m.values())
    assert f"Listed pair not measured: {KEYS[0]} {unpaired}: not paired" in out
    assert f"Listed pair not measured: {KEYS[2]} {unbuilt}: not built" in out
    with open(fails, newline="") as f:
        kinds = {(r["kernel"], r["kind"]) for r in csv.DictReader(f)}
    assert {
        (unpaired, "not-measured"),
        (unbuilt, "not-measured"),
        (unbuilt, "build-failed"),
    } <= kinds


@pytest.mark.parametrize(
    "entry, what",
    [({"7x7x7": "*"}, "problems"), ({KEYS[0]: ["no_such_kernel"]}, "kernels")],
)
def test_pairs_the_run_does_not_have_are_an_error(
    monkeypatch, capsys, tmp_path, entry, what
):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(entry))
    rc, out, b, m = _main(monkeypatch, capsys, tmp_path, "--pairs", str(path))
    assert rc == 1 and b is None and not any(m.values())
    assert f"--pairs lists 1 {what} this run does not have" in out


def test_malformed_pairs_file_is_an_error(monkeypatch, capsys, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({KEYS[0]: "first"}))
    rc, out, b, _ = _main(monkeypatch, capsys, tmp_path, "--pairs", str(path))
    assert rc == 1 and b is None and "must be a list of kernel names or '*'" in out


def test_pairs_and_list_pairs_are_exclusive(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit):
        _main(
            monkeypatch, capsys, tmp_path, "--pairs", "a.json", "--list-pairs", "b.json"
        )
