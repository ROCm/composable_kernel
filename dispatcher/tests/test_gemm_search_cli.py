#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
--search-config / --search-output-dir of the tile_engine GEMM sweep driver
(gemm_full_benchmark.py).

A run without them is unchanged and never loads the search modules; invalid
combinations exit 2; a search run measures only verified single-kernel batches
of the pairs a full run would measure, writes a winners.json the driver's
--pairs accepts, and exits 0 only when every problem has a confirmed winner.
Also: get_arch_filter_data no longer grows sys.path on every call.

No GPU or compiler needed: the build and the GPU batch runner are stubbed, the
real ck_search strategies run. The host-data preparations report "not run"
(their problems' calls compute their own, as without them), except in the
test that runs the real workers for them.
Run: python3 -m pytest -q tests/test_gemm_search_cli.py
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(GEMM_TE_DIR.parent / "common"))

import ctypes_utils as cu  # noqa: E402
import gemm_full_benchmark as drv  # noqa: E402
from _gemm_driver_stubs import run_stub_driver  # noqa: E402

PROBLEMS = [
    {"M": 512, "N": 512, "K": 512},
    {"M": 1024, "N": 512, "K": 256},
    {"M": 256, "N": 1024, "K": 512},
]
# Pairs with only the fixed-vector-width kernels: a candidate list that ignores
# the pairing rule shows up on it (the aligned problems pair with every native).
MISALIGNED = {"M": 1000, "N": 520, "K": 250}
KEYS = [drv.problem_key(p) for p in PROBLEMS]
SEARCH_MODULES = ("gemm_search", "adaptive_search", "ck_search", "host_cache")
LEGACY_CSV = ["kernel", "problem_idx", "M", "N", "K", "device", "latency_ms",
              "tflops", "non_zero", "max_rel", "verified"]


@pytest.fixture(autouse=True)
def _own_host_lock(tmp_path, monkeypatch):
    """Each test's search runs take the host lock in its own temp dir."""
    import adaptive_search as AS

    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _ample_host_memory(monkeypatch):
    """The host memory guard sees 1 TiB available, whatever this host has free."""
    import adaptive_search as AS

    monkeypatch.setattr(AS, "host_memory", lambda: 1 << 40)


def _ms(name):
    return 1.0 + (sum(map(ord, name)) % 97) / 100.0


def _main(monkeypatch, capsys, tmp_path, *argv, bad=(), problems=PROBLEMS,
          devices=("0",), build_kw=None, prepare=False, fixed=None):
    """Run the driver with a stub build (real files, so the search can copy
    them) and a stub GPU runner; ``bad`` names fail verification ("*": all),
    ``build_kw`` gets each build call's keyword arguments, plus "fixed": which
    of its kernels have fixed vector widths; ``prepare``: a search's
    host-data preparations run the real workers; ``fixed``: the arch, dtype
    or layout instead of the stub driver's (gfx950, fp16, rcr).
    Returns (rc, out, built names, calls)."""
    if "--search-config" in argv and not prepare:
        import adaptive_search as AS  # only for a search: see the no-search test

        monkeypatch.setattr(AS, "prepare_host_data", lambda argv, env, jobs, *a, **kw: {
            k: AS._prepared(None, "", 0.0, "not run") for k in jobs})
    libs = tmp_path / "libs"
    libs.mkdir(exist_ok=True)
    built, calls = [], []

    def build(cfgs, **kw):
        if build_kw is not None:
            build_kw.append({**kw, "fixed": [any(c.vector_sizes) for c in cfgs]})
        built.extend(c.name for c in cfgs)
        for c in cfgs:
            (libs / f"lib{c.name}.so").write_text(c.name)
        return [libs / f"lib{c.name}.so" for c in cfgs]

    def run_batch(device_id, unit, args, worker_path, base_env, extra=None, **kw):
        prob_idx, prob, batch = unit
        names = [cfg.name for _, cfg, _ in batch]
        calls.append({"names": names, "prob": prob, "verify": args.verify,
                      "batch_size": args.batch_size, "extra": extra,
                      "worker": (worker_path, base_env.get("GEMM_PYPATH"))})
        rows, n_fail = [], 0
        for name in names:
            ok = not (bad == "*" or name in bad)
            rows.append({"kernel": name, "problem_idx": prob_idx, **prob,
                         "device": device_id, "latency_ms": _ms(name), "tflops": 1.0,
                         "non_zero": 1, "max_rel": 0.0,
                         "verified": ok if args.verify else ""})
            if not ok:
                n_fail += 1
                if kw.get("failures") is not None:
                    kw["failures"].append({
                        "kernel": name, "problem_idx": prob_idx, **prob,
                        "device": device_id, "kind": "mismatch", "error": "max_rel 0.9"})
        return rows, [], n_fail, 0

    rc, cap = run_stub_driver(drv, monkeypatch, capsys, tmp_path, problems, *argv,
                              build=build, run_batch=run_batch, devices=devices,
                              **(fixed or {}))
    return rc, cap, built, calls


def _assert_worker(calls):
    """Every batch ran the driver's worker with its GEMM_PYPATH."""
    want = (GEMM_TE_DIR / "run_one_gemm_kernel.py", os.pathsep.join(
        [str(DISPATCHER_DIR / "python"), str(GEMM_TE_DIR)]))
    assert calls and all(c["worker"] == want for c in calls)


def _settings(tmp_path, **kw):
    s = {"schema_version": 1, "strategy": "tpe", "seed": 3, "trials_per_problem": 5,
         "new_builds_per_run": 12, "batch_per_problem": 3,
         "confirm": {"finalists": 2, "fresh_calls": 2}}
    s.update(kw)
    path = tmp_path / "search.json"
    path.write_text(json.dumps(s))
    return str(path)


def _search(monkeypatch, capsys, tmp_path, *argv, bad=(), problems=PROBLEMS,
            devices=("0",), build_kw=None, prepare=False, fixed=None, **settings):
    out = tmp_path / "run"
    got = _main(monkeypatch, capsys, tmp_path, "--search-config",
                _settings(tmp_path, **settings), "--search-output-dir", str(out),
                *argv, bad=bad, problems=problems, devices=devices, build_kw=build_kw,
                prepare=prepare, fixed=fixed)
    return (*got, out)


def _listing(monkeypatch, capsys, tmp_path, *argv, problems=PROBLEMS, fixed=None):
    path = tmp_path / "pairs.json"
    rc, _, built, _ = _main(monkeypatch, capsys, tmp_path, *argv, "--list-pairs",
                            str(path), problems=problems, fixed=fixed)
    assert rc == 0 and not built
    return json.loads(path.read_text())["pairs"]


def _read(out, name):
    return json.loads((out / name).read_text())


def test_run_without_search_is_unchanged_and_loads_no_search_module(
    monkeypatch, capsys, tmp_path
):
    for name in list(sys.modules):
        if name.split(".")[0] in SEARCH_MODULES:
            monkeypatch.delitem(sys.modules, name)
    rc, _, built, calls = _main(monkeypatch, capsys, tmp_path, "--verify")
    assert rc == 0 and built
    # The workers get the run's arch for the fp8/bf8 host format, as a search's do.
    assert all(c["extra"] == {"arch": "gfx950"} and c["batch_size"] == 20 for c in calls)
    _assert_worker(calls)
    assert not [n for n in sys.modules if n.split(".")[0] in SEARCH_MODULES]
    with open(tmp_path / "out.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames == LEGACY_CSV


@pytest.mark.parametrize("argv, message", [
    (["--search-config", "{S}"], "go together"),
    (["--search-output-dir", "{O}"], "go together"),
    (["--search-config", "", "--search-output-dir", ""], "go together"),
    (["--search-config", "{S}", "--search-output-dir", "{O}", "--list-pairs", "x.json"],
     "does not take --list-pairs"),
    (["--search-config", "{S}", "--search-output-dir", "{O}", "--repeat-top", "2"],
     "does not take --repeat-top"),
    (["--search-config", "{S}", "--search-output-dir", "{O}", "--failures-csv", "f.csv"],
     "does not take --failures-csv"),
    (["--search-config", "{S}", "--search-output-dir", "{O}", "--build-timeout", "600"],
     "does not take --build-timeout (set build_timeout_s in the search settings)"),
    (["--search-config", "{S}", "--search-output-dir", "{O}", "--variant",
      "gemm_multi_d"], "gemm_universal only"),
    (["--search-config", "{BAD}", "--search-output-dir", "{O}"], "unknown key"),
    (["--search-config", "{S}", "--search-output-dir", "{OLD}"], "already holds a run"),
    (["--search-config", "{S}", "--search-output-dir", "{FILE}"], "not a directory"),
    (["--search-config", "{S}", "--search-output-dir", "{RO}"], "not writable"),
])
def test_invalid_search_options_exit_2(monkeypatch, capsys, tmp_path, argv, message):
    if "{RO}" in argv and os.geteuid() == 0:
        pytest.skip("root writes anywhere")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "strategy": "tpe",
                               "trials_per_problem": 2, "new_builds_per_run": 4,
                               "bogus": 1}))
    old = tmp_path / "old"
    old.mkdir()
    (old / "run.json").write_text("{}")
    (tmp_path / "file").write_text("")
    (tmp_path / "ro").mkdir(mode=0o555)
    subs = {"{S}": _settings(tmp_path), "{O}": str(tmp_path / "new"),
            "{BAD}": str(bad), "{OLD}": str(old), "{FILE}": str(tmp_path / "file"),
            "{RO}": str(tmp_path / "ro" / "run")}
    argv = [subs.get(a, a) for a in argv]
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, capsys, tmp_path, *argv)
    assert e.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "new").exists() and not any((tmp_path / "libs").iterdir())


@pytest.mark.parametrize("case, message", [
    ("no_devices", "no GPU to measure on"),
    ("pairs_file", "warm_start.pairs_file: entry"),
    ("ranked_csv", "ranked.csv line 2: expected an MxNxK problem and a number"),
])
def test_search_setup_errors_exit_2_before_any_build(
    monkeypatch, capsys, tmp_path, case, message
):
    (tmp_path / "warm.json").write_text(json.dumps({KEYS[0]: 7}))
    (tmp_path / "ranked.csv").write_text(
        f"problem,rank,kernel,predicted_tflops\n{KEYS[0]},1,kA\n")
    warm = {"pairs_file": {"pairs_file": "warm.json"},
            "ranked_csv": {"ranked_csv": "ranked.csv", "use_prior": True}}.get(case)
    rc, cap, built, calls, out = _search(
        monkeypatch, capsys, tmp_path, devices=() if case == "no_devices" else ("0",),
        **({"warm_start": warm} if warm else {}))
    assert rc == 2 and message in cap.out
    assert not built and not calls and not (out / "summary.json").exists()


def test_search_writes_winners_the_driver_accepts(monkeypatch, capsys, tmp_path):
    eligible = _listing(monkeypatch, capsys, tmp_path)
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path)
    assert rc == 0, cap.out
    # Every measurement is one verified kernel with the identity check, and
    # its worker gets the run's host cache: a temp dir, gone after the run.
    cache = calls[0]["extra"]["host_cache"]
    assert calls and all(
        len(c["names"]) == 1 and c["verify"] and c["batch_size"] == 1
        and c["extra"] == {"check_identity": True, "search_host_data": True,
                           "host_cache": cache, "arch": "gfx950"} for c in calls)
    assert Path(cache["dir"]).name.startswith("ck_search_host_cache-")
    assert not Path(cache["dir"]).exists() and "Host cache: 0 entries" in cap.out
    _assert_worker(calls)
    assert len(set(built)) == len(built) <= 12
    assert {p.name for p in (out / "artifacts").iterdir()} == {f"{n}.so" for n in built}
    winners = drv.load_pair_list(out / "winners.json")
    assert set(winners) == set(KEYS)
    for key, prob in zip(KEYS, PROBLEMS):
        (name,) = winners[key]
        assert name in eligible[key]
        tried = {c["names"][0] for c in calls if c["prob"] == prob}
        # Confirmation repeats the discovery latency: the fastest tried wins.
        assert len(tried) <= 5 and _ms(name) == min(map(_ms, tried))
    summary, run = _read(out, "summary.json"), _read(out, "run.json")
    assert (summary["status"], summary["exit_code"]) == ("complete", 0)
    assert summary["counts"]["confirm_calls"] == 2 * 2 * len(KEYS)
    assert (run["sampling_policy"], run["exhaustive"]) == ("adaptive", False)
    assert (run["context"]["arch"], run["context"]["variant"]) == ("gfx950", "standard")
    assert "pipelines" not in run["context"]
    assert all("--pipelines" not in n for n in run["context"]["domain_notes"])
    assert "SEARCH COMPLETE: 3/3 problems" in cap.out
    # Progress: a line per round and build, a title per measure call, no banner.
    assert "[search] round 1: building" in cap.out and "Phase 3" not in cap.out
    assert "-measure: " in cap.out and "-confirm: " in cap.out
    # Today's driver re-checks the winners from the file as a --pairs list.
    rc, _, rebuilt, recalls = _main(monkeypatch, capsys, tmp_path, "--pairs",
                                    str(out / "winners.json"), "--verify")
    assert rc == 0 and set(rebuilt) == {n for v in winners.values() for n in v}
    assert {(c["prob"]["M"], c["prob"]["N"], c["prob"]["K"], n)
            for c in recalls for n in c["names"]} == {
        (p["M"], p["N"], p["K"], winners[k][0]) for k, p in zip(KEYS, PROBLEMS)}


@pytest.mark.parametrize("off", [False, True])
def test_host_cache_settings_reach_the_workers(monkeypatch, capsys, tmp_path, off):
    # host_cache.dir holds the run's cache dir; max_total_gib 0 turns it off.
    where = tmp_path / "scratch"
    where.mkdir()
    host = {"dir": str(where), "max_entry_gib": 0.5, "max_total_gib": 0 if off else 2}
    rc, cap, _, calls, out = _search(monkeypatch, capsys, tmp_path, host_cache=host)
    assert rc == 0, cap.out
    specs = [c["extra"].get("host_cache") for c in calls]
    if off:
        assert specs == [None] * len(calls) and "Host cache" not in cap.out
    else:
        assert all(s == specs[0] for s in specs)
        assert Path(specs[0]["dir"]).parent == where
        assert (specs[0]["max_entry_bytes"], specs[0]["max_total_bytes"]) == (
            2**29, 2**31)
    assert _read(out, "run.json")["context"]["measurement"]["host_cache"] == specs[0]
    assert list(where.iterdir()) == []


def test_real_workers_prepare_each_problem_before_its_first_call(
        monkeypatch, capsys, tmp_path):
    # The driver's worker script, started by the search as a subprocess with
    # the driver's environment, stores each problem's host data once (no
    # library, no GPU), before the problem's first measure call: 3 entries.
    rc, cap, _, calls, out = _search(monkeypatch, capsys, tmp_path, prepare=True,
                                     devices=("0", "1"))
    assert rc == 0, cap.out
    assert "Host cache: 3 entries" in cap.out
    sides = [json.loads(p.read_text())
             for p in sorted((out / "rounds").glob("*-measure.json"))]
    sides += [json.loads(p.read_text())
              for p in sorted((out / "rounds").glob("*-confirm.json"))]
    prepared = [r for side in sides for r in side["prepared"]]
    assert sorted(r["problem"] for r in prepared) == sorted(KEYS)
    assert all(r["ok"] and r["stored"] and not r["error"] for r in prepared)
    first = sides[0]
    assert {r["problem"] for r in first["prepared"]} == {
        r["problem"] for r in first["requests"]}
    eligible = _listing(monkeypatch, capsys, tmp_path)
    assert all(r["kernel"] in eligible[r["problem"]] for r in prepared)
    assert "preparing the host data of 3 problem(s)" in cap.out
    assert "3 of 3 stored" in cap.out and "not stored" not in cap.out
    measurement = _read(out, "run.json")["context"]["measurement"]
    assert measurement["prepare"].startswith("before a problem's first call")
    # Calls keep the per-dtype allowance: 30 s + 1 s for these small shapes.
    assert "(timeout 31s per kernel)" in cap.out


def test_interrupted_search_removes_its_host_cache(monkeypatch, capsys, tmp_path):
    # Ctrl-C or SIGTERM reaches the run as KeyboardInterrupt (here from the
    # build): it exits 130 and its host cache directory is gone.
    where = tmp_path / "scratch"
    where.mkdir()

    def build(cfgs, **kw):
        (cache,) = where.iterdir()
        (cache / "entry").mkdir()
        (cache / "entry" / "a.npy").write_bytes(b"x" * 100)
        raise KeyboardInterrupt

    rc, cap = run_stub_driver(
        drv, monkeypatch, capsys, tmp_path, PROBLEMS, "--search-config",
        _settings(tmp_path, host_cache={"dir": str(where)}), "--search-output-dir",
        str(tmp_path / "run"), build=build, run_batch=None)
    assert rc == 130 and list(where.iterdir()) == []
    assert "Host cache: 0 entries, 0.00 GiB, removed" in cap.out


@pytest.mark.parametrize("problems", [PROBLEMS, [MISALIGNED], PROBLEMS[:1] + [MISALIGNED]],
                         ids=["aligned", "misaligned", "mixed"])
@pytest.mark.parametrize("extra", [{}, {"build_timeout_s": 600}], ids=["default", "set"])
def test_search_has_its_own_build_timeout(monkeypatch, capsys, tmp_path, extra, problems):
    # A round measures once all its builds end: the search limits each build
    # step itself. Its 180 s default comes from native kernels, so a build call
    # with a fixed-vector-width kernel (the only kind MISALIGNED pairs with)
    # keeps the build function's limits (None: 1200 s for that compile, as in a
    # full run); build_timeout_s applies to every kernel. A full run's build
    # call is unchanged.
    kws = []
    rc, cap, built, _, out = _search(monkeypatch, capsys, tmp_path, build_kw=kws,
                                     problems=problems, **extra)
    assert rc == 0 and built and kws, cap.out
    assert any(any(k["fixed"]) for k in kws) == (MISALIGNED in problems)
    given = extra.get("build_timeout_s")
    assert [k["build_timeout"] for k in kws] == [
        given or (None if any(k["fixed"]) else 180) for k in kws]
    assert _read(out, "run.json")["context"]["build"]["build_timeout"] == (given or 180)
    kws.clear()
    rc, _, built, _ = _main(monkeypatch, capsys, tmp_path, build_kw=kws)
    assert rc == 0 and built and kws and all("build_timeout" not in k for k in kws)


def test_confirmation_keeps_each_problem_on_one_device(monkeypatch, capsys, tmp_path):
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path,
                                         devices=("0", "1"))
    assert rc == 0, cap.out
    confirm = [e for e in map(json.loads, (out / "trials.jsonl").read_text()
                              .splitlines()) if e["event"] == "confirm"]
    devices = {}
    for e in confirm:
        devices.setdefault(e["problem"], set()).add(e["device"])
    assert devices == {k: {"01"[i % 2]} for i, k in enumerate(KEYS)}


def test_ranked_csv_is_the_search_prior(monkeypatch, capsys, tmp_path):
    eligible = _listing(monkeypatch, capsys, tmp_path)
    rows = [f"{KEYS[0]},{i + 1},{name},{10.0 - i}"
            for i, name in enumerate(eligible[KEYS[0]][:3])]
    (tmp_path / "ranked.csv").write_text(
        "problem,rank,kernel,predicted_tflops\n" + "\n".join(rows) + "\n")
    rc, cap, built, calls, out = _search(
        monkeypatch, capsys, tmp_path,
        warm_start={"ranked_csv": "ranked.csv", "use_prior": True})
    assert rc == 0, cap.out
    entries = {k: p["prior_entries"] for k, p in _read(out, "run.json")["problems"].items()}
    assert entries == {KEYS[0]: 3, KEYS[1]: 0, KEYS[2]: 0}


def _domain_sha256(pairs):
    return hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()


def test_search_candidates_are_the_pairs_a_full_run_measures(
    monkeypatch, capsys, tmp_path
):
    problems = PROBLEMS[:1] + [MISALIGNED]
    eligible = _listing(monkeypatch, capsys, tmp_path, problems=problems)
    sizes = {k: len(v) for k, v in eligible.items()}
    assert sizes[drv.problem_key(MISALIGNED)] > sizes[KEYS[0]]
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path,
                                         problems=problems, strategy="random",
                                         new_builds_per_run=24)
    assert rc == 0, cap.out
    run = _read(out, "run.json")
    assert {k: p["eligible"] for k, p in run["problems"].items()} == sizes
    assert run["context"]["domain_sha256"] == _domain_sha256(eligible)
    for c in calls:
        assert c["names"][0] in eligible[drv.problem_key(c["prob"])]


def test_pairs_restricts_the_search_domain(monkeypatch, capsys, tmp_path):
    problems = PROBLEMS + [MISALIGNED]
    eligible = _listing(monkeypatch, capsys, tmp_path, problems=problems)
    odd = drv.problem_key(MISALIGNED)
    unpaired = next(n for n in eligible[odd] if n not in eligible[KEYS[0]])
    # Two paired kernels per problem, plus one the first problem does not pair.
    picks = {k: names[:2] for k, names in eligible.items()}
    picks[KEYS[0]] = picks[KEYS[0]] + [unpaired]
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(picks))
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path,
                                         "--pairs", str(path), problems=problems)
    assert rc == 0, cap.out
    assert "the search builds only the kernels it proposes" in cap.out
    assert "; building" not in cap.out  # the plain --pairs plan line
    domain = {k: [n for n in v if n in eligible[k]] for k, v in picks.items()}
    assert unpaired not in domain[KEYS[0]]
    assert set(built) <= {n for v in domain.values() for n in v}
    run = _read(out, "run.json")
    assert run["context"]["pairs_file"] == str(path)
    assert run["context"]["domain_sha256"] == _domain_sha256(domain)
    # Two candidates and five trials per problem: searched exhaustively.
    assert all(p["strategy"] == "exhaustive" and p["eligible"] == 2
               for p in run["problems"].values())


def test_pairs_subset_searches_only_its_problems(monkeypatch, capsys, tmp_path):
    eligible = _listing(monkeypatch, capsys, tmp_path)
    path = tmp_path / "picks.json"
    path.write_text(json.dumps({KEYS[1]: eligible[KEYS[1]][:3]}))
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path,
                                         "--pairs", str(path))
    assert rc == 0, cap.out
    assert list(_read(out, "run.json")["problems"]) == [KEYS[1]]
    assert _read(out, "winners.json").keys() == {KEYS[1]}
    assert {drv.problem_key(c["prob"]) for c in calls} == {KEYS[1]}
    assert "SEARCH COMPLETE: 1/1 problems" in cap.out


def test_pairs_problem_without_a_paired_kernel_is_unresolved(
    monkeypatch, capsys, tmp_path
):
    problems = PROBLEMS[:1] + [MISALIGNED]
    eligible = _listing(monkeypatch, capsys, tmp_path, problems=problems)
    unpaired = next(n for n in eligible[drv.problem_key(MISALIGNED)]
                    if n not in eligible[KEYS[0]])
    path = tmp_path / "picks.json"
    path.write_text(json.dumps({KEYS[0]: [unpaired]}))
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path,
                                         "--pairs", str(path), problems=problems)
    assert rc == 1 and not built and not calls
    summary = _read(out, "summary.json")
    assert summary["unresolved"] == {KEYS[0]: "no eligible candidates"}


def test_search_without_a_winner_exits_1(monkeypatch, capsys, tmp_path):
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path, bad="*",
                                         strategy="random")
    assert rc == 1 and built and calls
    assert _read(out, "winners.json") == {}
    summary = _read(out, "summary.json")
    assert (summary["status"], summary["exit_code"]) == ("no_winner", 1)
    assert summary["counts"]["failed_by_status"]["numeric_fail"] == len(calls)
    assert summary["counts"]["confirm_calls"] == 0


# ------------------------------------------- gfx1250 TDM search (CPU only)

GFX1250 = {"arch": "gfx1250"}
TDM = ("comp_async", "comp_tdm", "comp_tdm_v2", "comp_tdm_pc")  # bridged TDM/async
GFX1250_CONFIG = str(
    GEMM_TE_DIR / "gemm_universal" / "configs" / "default_config_gfx1250.json"
)
EXAMPLE_SETTINGS = GEMM_TE_DIR / "configs" / "search_settings_gfx1250_tdm.json"


def _kernels(monkeypatch, capsys, tmp_path, *argv, fixed=None):
    """--list-pairs: ({kernel: pipeline}, {problem: [kernels]})."""
    path = tmp_path / "kernels.json"
    rc, _, built, _ = _main(
        monkeypatch, capsys, tmp_path, *argv, "--list-pairs", str(path), fixed=fixed
    )
    assert rc == 0 and not built
    data = json.loads(path.read_text())
    if "--pipelines" in argv:
        want = drv.parse_pipelines(argv[argv.index("--pipelines") + 1])
        assert data["pipelines"] == list(want)
    else:
        assert "pipelines" not in data
    return {k: v["pipeline"] for k, v in data["kernels"].items()}, data["pairs"]


@pytest.mark.parametrize(
    "value, message",
    [
        ("", "non-empty"),
        (",compv4", "non-empty"),
        ("compv4,", "non-empty"),
        ("compv4,,mem", "non-empty"),
        ("compv4,compv4", "duplicate"),
        ("compv4,typo", "unknown pipeline"),
    ],
)
def test_pipeline_filter_rejects_bad_values(
    monkeypatch, capsys, tmp_path, value, message
):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, capsys, tmp_path, "--pipelines", value)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not any((tmp_path / "libs").iterdir())
    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize("config_count", [0, 1, 2])
def test_pipeline_filter_is_shared_before_limits_and_pairs(
    monkeypatch, capsys, tmp_path, config_count
):
    # Implicit CI selection and explicit (including multiple) configs use
    # the same filter before --max-kernels and --pairs in every run mode.
    configs = [str(GEMM_TE_DIR / "configs" / "default_ci_config.json")] * config_count
    pipe, full = _kernels(monkeypatch, capsys, tmp_path, *configs)
    expected = next(n for n in full[KEYS[0]] if pipe[n] == "compv4")
    argv = [*configs, "--pipelines", " compv4 ", "--max-kernels", "1"]
    selected, listed = _kernels(monkeypatch, capsys, tmp_path, *argv)
    assert selected == {expected: "compv4"}
    assert listed == {k: [expected] for k in KEYS}
    pairs = tmp_path / "selected.json"
    for wanted in ([expected], "*"):
        pairs.write_text(json.dumps({KEYS[0]: wanted}))
        rc, cap, built, calls = _main(
            monkeypatch, capsys, tmp_path, *argv, "--pairs", str(pairs)
        )
        assert rc == 0, cap.out
        assert set(built) == {expected}
        assert calls and all(
            c["prob"] == PROBLEMS[0] and c["names"] == [expected] for c in calls
        )
    rc, cap, built, calls, out = _search(
        monkeypatch,
        capsys,
        tmp_path,
        *argv,
        "--pairs",
        str(pairs),
        host_cache={"max_total_gib": 0},
    )
    assert rc == 0, cap.out
    assert set(built) == {expected}
    assert calls and all(
        c["prob"] == PROBLEMS[0] and c["names"] == [expected] for c in calls
    )
    assert _read(out, "winners.json") == {KEYS[0]: [expected]}
    ctx = _read(out, "run.json")["context"]
    assert ctx["pipelines"] == ["compv4"]
    assert "--pipelines compv4 restricts the expanded configs" in ctx["domain_notes"]


@pytest.mark.parametrize("search", [False, True])
def test_pipeline_filter_rejects_excluded_explicit_pairs(
    monkeypatch, capsys, tmp_path, search
):
    pipe, full = _kernels(monkeypatch, capsys, tmp_path)
    excluded = next(n for n in full[KEYS[0]] if pipe[n] != "compv4")
    pairs = tmp_path / "excluded.json"
    pairs.write_text(json.dumps({KEYS[0]: [excluded]}))
    runner = _search if search else _main
    rc, cap, built, calls, *_ = runner(
        monkeypatch, capsys, tmp_path, "--pipelines", "compv4", "--pairs", str(pairs)
    )
    assert rc == 1 and not built and not calls
    assert "this run does not have" in cap.out and "--pipelines" in cap.out


def test_pipeline_filter_no_matching_config_is_an_error(monkeypatch, capsys, tmp_path):
    # The ordinary CI config does not contain TDM; the filter must not silently
    # switch to a full config or succeed with an empty list.
    path = tmp_path / "empty.json"
    rc, cap, built, calls = _main(
        monkeypatch,
        capsys,
        tmp_path,
        "--pipelines",
        "comp_tdm",
        "--list-pairs",
        str(path),
        fixed=GFX1250,
    )
    assert rc == 1 and not built and not calls and not path.exists()
    assert "0 configs left after --pipelines comp_tdm" in cap.out
    assert "expanded only:" in cap.out


@pytest.mark.parametrize(
    "dtype, layout",
    [("fp16", "rcr"), ("bf16", "rcr"), ("fp8", "rcr"), ("bf16", "ccr"), ("fp8", "crr")],
)
def test_gfx1250_pipeline_filter_preserves_the_default_space(
    monkeypatch, capsys, tmp_path, dtype, layout
):
    # With --arch gfx1250 every problem gets every bridged TDM/async pipeline
    # (comp_async in rcr only); the no_pc variant drops comp_tdm_pc only.
    fixed = {**GFX1250, "dtype": dtype, "layout": layout}
    reference = json.loads(Path(GFX1250_CONFIG).read_text())
    path = tmp_path / "tdm-reference.json"
    for want in (TDM, TDM[:-1]):
        # Compare actual filtering with expanding the same default config
        # directly restricted to these pipelines, without persisted copies.
        reference["trait_config"]["pipeline"]["values"] = list(want)
        path.write_text(json.dumps(reference))
        expected_pipe, expected_pairs = _kernels(
            monkeypatch, capsys, tmp_path, str(path), fixed=fixed
        )
        pipe, pairs = _kernels(
            monkeypatch,
            capsys,
            tmp_path,
            GFX1250_CONFIG,
            "--pipelines",
            ",".join(want),
            fixed=fixed,
        )
        assert pairs == expected_pairs
        assert pipe == expected_pipe
        want = set(want) - (set() if layout == "rcr" else {"comp_async"})
        assert set(pairs) == set(KEYS)
        assert all({pipe[n] for n in names} == want for names in pairs.values())


@pytest.mark.parametrize("dtype, layout", [("fp8", "rcr"), ("bf16", "ccr")])
def test_gfx1250_tdm_search_tries_every_pipeline(
    monkeypatch, capsys, tmp_path, dtype, layout
):
    # The example settings (fewer trials) on the TDM config, end to end with
    # the build and the GPU stubbed: warm_start.cover tries each pipeline on
    # each problem first, build_timeout_s bounds every build, and the workers
    # get the run's arch for the fp8 format.
    fixed = {**GFX1250, "dtype": dtype, "layout": layout}
    argv = [GFX1250_CONFIG, "--pipelines", ",".join(TDM)]
    pipe, pairs = _kernels(monkeypatch, capsys, tmp_path, *argv, fixed=fixed)
    want = set(TDM) - (set() if layout == "rcr" else {"comp_async"})
    settings = json.loads(EXAMPLE_SETTINGS.read_text())
    settings.update(
        trials_per_problem=6,
        new_builds_per_run=40,
        batch_per_problem=2,
        confirm={"finalists": 2, "fresh_calls": 1},
    )
    kws = []
    rc, cap, built, calls, out = _search(
        monkeypatch, capsys, tmp_path, *argv, build_kw=kws, fixed=fixed, **settings
    )
    assert rc == 0, cap.out
    run = _read(out, "run.json")
    assert (
        run["context"]["arch"] == "gfx1250"
        and run["settings"]["build_timeout_s"] == 300
    )
    assert kws and all(k["build_timeout"] == 300 for k in kws)
    assert calls and all(c["extra"]["arch"] == "gfx1250" for c in calls)
    for key, prob in zip(KEYS, PROBLEMS):
        assert set(run["problems"][key]["warm_cover"]) == want
        tried = {pipe[c["names"][0]] for c in calls if c["prob"] == prob}
        assert tried == want  # every candidate is one of them
    winners = drv.load_pair_list(out / "winners.json")
    assert set(winners) == set(KEYS) and all(winners[k][0] in pairs[k] for k in KEYS)


def test_get_arch_filter_data_does_not_grow_sys_path(monkeypatch):
    # It runs once per expanded config: a full sweep used to add one entry per call.
    codegen = str(cu.get_dispatcher_root() / "codegen")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != codegen])
    n = len(sys.path)
    for _ in range(3):
        assert cu.get_arch_filter_data()["supported_archs"]
    assert len(sys.path) == n + 1 and sys.path.count(codegen) == 1
