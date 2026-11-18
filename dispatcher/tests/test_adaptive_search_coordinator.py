#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
The adaptive search coordinator (tile_engine/ops/common/adaptive_search.py).

A fake OperationAdapter records every build, measurement and confirmation call;
scripted strategies propose fixed kernel names round by round, so each test can
state exactly which pairs are built and measured. The real ck_search strategies
run once end to end at the bottom.

No GPU or compiler needed. Run: python3 -m pytest -q tests/test_adaptive_search_coordinator.py
"""

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
TE_OPS = DISPATCHER_DIR.parent / "tile_engine" / "ops"
COMMON_DIR = TE_OPS / "common"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(COMMON_DIR))

import adaptive_search as AS  # noqa: E402
import ck_search  # noqa: E402
import host_cache as HC  # noqa: E402
from _py38_check import py38_problems  # noqa: E402

KERNELS = "ABCDEFGHIJKL"
# The modules this change adds (also ASCII only) or edits, besides ck_search.
NEW_MODULES = (COMMON_DIR / "adaptive_search.py", COMMON_DIR / "host_cache.py",
               TE_OPS / "gemm" / "gemm_search.py")
EDITED_MODULES = (
    TE_OPS / "gemm" / "gemm_full_benchmark.py",
    TE_OPS / "gemm" / "run_one_gemm_kernel.py",
    DISPATCHER_DIR / "python" / "ctypes_utils.py",
    DISPATCHER_DIR / "python" / "gemm_utils.py",
)


def rec(name):
    """One distinct scalar record per kernel, the same in every problem."""
    i = KERNELS.index(name)
    return {"tile_m": 16 * (i + 1), "tile_n": 64 if i % 2 else 128}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeAdapter:
    """``problems`` {problem: [kernel names]}; ``lat`` {(problem, name): ms};
    ``script`` {(problem, name): [status per measurement]} (default ok);
    ``confirm_script`` {(problem, name): [(status, ms) per fresh call]}
    (default ok at the discovery latency)."""

    def __init__(self, problems, lat=None, fail_build=(), script=None,
                 confirm_script=None, clock=None, advance=0.0):
        self.problems = problems
        self.lat = lat or {}
        self.fail_build = set(fail_build)
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.confirm_script = {k: list(v) for k, v in (confirm_script or {}).items()}
        self.clock, self.advance = clock, advance
        self.builds, self.measured, self.confirmed = [], [], []

    def context(self):
        return {"arch": "gfx950", "dtype": "fp16", "layout": "rcr", "variant": "fake"}

    def enumerate_eligible(self):
        return {
            p: [AS.Candidate(n, n, rec(n), {"cfg": n}) for n in names]
            for p, names in self.problems.items()
        }

    def build_missing(self, cands):
        self.builds.append([c.canonical_key for c in cands])
        return {
            c.canonical_key: AS.BuildResult(None, "fake compile error")
            if c.canonical_key in self.fail_build
            else AS.BuildResult(f"artifacts/{c.canonical_key}.so", digest="0" * 64)
            for c in cands
        }

    def _lat(self, r):
        return self.lat.get((r.problem_key, r.kernel_name), 1.0)

    def measure_pairs(self, requests):
        self.measured.extend(requests)
        if self.clock is not None:
            self.clock.t += self.advance
        out = []
        for r in requests:
            todo = self.script.get((r.problem_key, r.kernel_name))
            status = todo.pop(0) if todo else "ok"
            lat = self._lat(r) if status == "ok" else None
            out.append(AS.PairOutcome(r, status, lat, "" if lat else f"fake {status}"))
        return out

    def confirm(self, requests):
        self.confirmed.append(list(requests))
        out = []
        for r in requests:
            todo = self.confirm_script.get((r.problem_key, r.kernel_name))
            status, lat = todo.pop(0) if todo else ("ok", self._lat(r))
            out.append(AS.PairOutcome(r, status, lat))
        return out


class Scripted:
    """Proposes ``steps`` (lists of kernel names, or raw genome tuples), one
    step per ask."""

    def __init__(self, space, steps):
        self.space, self.steps = space, [list(s) for s in steps]
        self.asks, self.told, self.pending = [], [], {}
        self._uid = 0

    def ask(self, n):
        self.asks.append(n)
        if not self.steps:
            return []
        step = self.steps.pop(0)
        if len(step) > n:
            self.steps.insert(0, step[n:])
            step = step[:n]
        out = []
        for name in step:
            self._uid += 1
            g = name if isinstance(name, tuple) else self.space.encode(rec(name))
            t = ck_search.Trial(genome=g, uid=f"{self.space.name}:{self._uid}")
            self.pending[t.uid] = t
            out.append(t)
        return out

    def tell(self, observations):
        for ob in observations:
            self.pending.pop(ob.trial.uid, None)
            self.told.append(ob)

    @property
    def done(self):
        return not self.steps


def scripted(scripts, calls=None):
    made = {}

    def factory(name, space, seed, warm_records, prior):
        if calls is not None:
            calls.append({"name": name, "problem": space.name, "seed": seed,
                          "warm": list(warm_records), "prior": prior})
        made[space.name] = Scripted(space, scripts.get(space.name, []))
        return made[space.name]

    return factory, made


def settings(**kw):
    d = {"schema_version": 1, "strategy": "random", "trials_per_problem": 4,
         "new_builds_per_run": 20, "batch_per_problem": 2}
    d.update(kw)
    return AS.SearchSettings.from_json(d)


def run(tmp_path, adapter, scripts, clock=None, calls=None, pair_loader=None,
        prior_loader=None, factory=None, **kw):
    """One run with scripted strategies (or ``factory``); (summary, strategies
    by problem, coordinator)."""
    made = {}
    if factory is None:
        factory, made = scripted(scripts, calls)
    co = AS.SearchCoordinator(
        adapter, settings(**kw), tmp_path / "out", clock=clock or FakeClock(),
        strategy_factory=factory, pair_loader=pair_loader, prior_loader=prior_loader,
    )
    return co.run(), made, co


def json_pairs(path):
    """A pair_loader for tests that need no driver: the --pairs JSON as is."""
    return json.loads(Path(path).read_text())


def _no_constant(c):
    raise AssertionError(f"non-strict JSON constant {c}")


def strict(path):
    return json.loads(Path(path).read_text(), parse_constant=_no_constant)


def events(out_dir, kind=None):
    rows = [json.loads(line, parse_constant=_no_constant)
            for line in (Path(out_dir) / "trials.jsonl").read_text().splitlines()]
    return [r for r in rows if kind is None or r["event"] == kind]


def pairs(requests):
    return [(r.problem_key, r.kernel_name) for r in requests]


ABCD = list("ABCD")


# ----------------------------------------------------------------- import


@pytest.mark.parametrize("path", NEW_MODULES + EDITED_MODULES, ids=lambda p: p.name)
def test_module_is_python38_compatible(path):
    src = path.read_text(encoding="utf-8")
    assert py38_problems(src, path.name) == []
    assert src.isascii() or path not in NEW_MODULES


def test_import_is_lean():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import adaptive_search; "
        "bad = [m for m in ('ck_search', 'numpy', 'rocke', 'lightgbm', 'pandas') "
        "if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-B", "-c", code, str(COMMON_DIR)], check=True)


# ------------------------------------------------------- shared artifacts


def test_shared_artifacts_built_once_and_winners_confirmed(tmp_path):
    """P1 asks A,B then C; P2 asks B,C then A,D: 4 builds, 7 measurements."""
    lat = {("P1", "A"): 1.0, ("P1", "B"): 2.0, ("P1", "C"): 1.5,
           ("P2", "B"): 1.2, ("P2", "C"): 0.9, ("P2", "A"): 3.0, ("P2", "D"): 1.1}
    # P1's discovery best (A) was a lucky sample: confirmation picks C.
    confirm = {("P1", "A"): [("ok", 3.0), ("ok", 3.1), ("ok", 2.9)]}
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD}, lat=lat, confirm_script=confirm)
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"], ["C"]],
                                     "P2": [["B", "C"], ["A", "D"]]})
    assert ad.builds == [["A", "B", "C"], ["D"]]
    assert sorted(pairs(ad.measured)) == sorted(
        [("P1", "A"), ("P1", "B"), ("P2", "B"), ("P2", "C"),
         ("P1", "C"), ("P2", "A"), ("P2", "D")])
    # Confirmation: 3 fresh calls x 2 finalists x 2 problems, interleaved by call.
    assert [len(c) for c in ad.confirmed] == [4, 4, 4]
    ids = [r.call_id for c in ad.confirmed for r in c]
    assert len(set(ids)) == 12 and not set(ids) & {r.call_id for r in ad.measured}
    assert s.counts["measured"] == 7 and s.counts["confirm_calls"] == 12
    assert s.counts["built"] == 4 and co.ledger.builds == 4
    assert s.winners == {"P1": "C", "P2": "C"}
    assert (s.status, s.exit_code, s.stop_reason) == ("complete", 0, "searched")
    assert strict(tmp_path / "out" / "winners.json") == {"P1": ["C"], "P2": ["C"]}
    assert [len(made[p].told) for p in ("P1", "P2")] == [3, 4]
    assert len(events(tmp_path / "out", "build")) == 4
    assert len(events(tmp_path / "out", "result")) == 7
    assert len(events(tmp_path / "out", "confirm")) == 12


def test_failed_confirmation_is_no_winner(tmp_path):
    """Only a verified finite result can be a finalist; when it fails its fresh
    calls the problem has no winner (no fallback to the invalid rows)."""
    lat = {("P1", "B"): math.nan, ("P1", "C"): 2.0}
    ad = FakeAdapter({"P1": ABCD}, lat=lat,
                     script={("P1", "A"): ["numeric_fail"], ("P1", "D"): ["run_error"]},
                     confirm_script={("P1", "C"): [("ok", 2.0), ("numeric_fail", None)]})
    s, made, _ = run(tmp_path, ad, {"P1": [["A", "B"], ["C", "D"]]})
    assert [pairs(c) for c in ad.confirmed] == [[("P1", "C")], [("P1", "C")]]
    assert (s.status, s.exit_code, s.winners) == ("no_winner", 1, {})
    assert "no finalist passed confirmation" in s.unresolved["P1"]
    assert strict(tmp_path / "out" / "winners.json") == {}
    assert s.counts["failed_by_status"] == {
        "numeric_fail": 1, "invalid_measurement": 1, "run_error": 1}
    told = {ob.trial.uid: ob for ob in made["P1"].told}
    assert [ob.status for ob in told.values()] == [
        "numeric_fail", "run_error", "ok", "run_error"]
    assert told["P1:2"].detail.startswith("invalid_measurement")
    assert all(ob.value == math.inf for ob in told.values() if ob.status != "ok")


def test_partial_and_no_eligible_problem(tmp_path):
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": []}, fail_build={"C", "D"})
    s, _, _ = run(tmp_path, ad, {"P1": [["A"]], "P2": [["C", "D"]]})
    assert (s.status, s.exit_code) == ("partial", 1)
    assert s.winners == {"P1": "A"}
    assert s.unresolved["P2"].startswith("no verified discovery measurement")
    assert s.unresolved["P3"] == "no eligible candidates"


# ------------------------------------------------------------- failures


def test_compile_failure_shared_by_problems_is_one_build(tmp_path):
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": ABCD}, fail_build={"B"})
    s, made, co = run(tmp_path, ad, {"P1": [["B", "A"]], "P2": [["B", "C"]],
                                     "P3": [["D"], ["B"]]})
    assert ad.builds == [["B", "A", "C", "D"]]  # round 2 re-asks B: no rebuild
    assert (s.counts["built"], s.counts["build_failed"], co.ledger.builds) == (3, 1, 4)
    assert "B" not in [r.kernel_name for r in ad.measured]
    errs = {p: [ob.cached for ob in made[p].told if ob.status == "compile_error"]
            for p in made}
    # P1 and P2 asked for B in the round that built it, each with a trial; P3
    # asks once its build is known to have failed: told that result, no trial.
    assert errs == {"P1": [False], "P2": [False], "P3": [True]}
    assert s.counts["failed_by_status"] == {"compile_error": 2}
    assert s.counts["known_failed"] == 1 and co.ledger.trials["P3"] == 1
    assert made["P3"].told[-1].detail == "fake compile error"
    build = events(tmp_path / "out", "build")
    assert [b["waiting"] for b in build if b["kernel"] == "B"] == [["P1", "P2"]]
    (known,) = [e for e in events(tmp_path / "out", "trial")
                if e["admission"] == "known_failed"]
    assert (known["problem"], known["kernel"]) == ("P3", "B")


def test_known_build_failures_cost_no_trial(tmp_path):
    # P1's builds of A..D all fail. P2 then proposes A..D: each is told its
    # compile_error from the registry, so P2's trials go to G, E and F only.
    ad = FakeAdapter({"P1": ABCD, "P2": list("ABCDEFG")}, fail_build=set(ABCD))
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [["G"], ABCD, ["E", "F"]]},
                      trials_per_problem=6, batch_per_problem=4, max_stall_rounds=2)
    assert ad.builds == [list("ABCDG"), ["E", "F"]]
    assert sorted(pairs(ad.measured)) == [("P2", n) for n in "EFG"]
    assert co.ledger.trials == {"P1": 4, "P2": 3} and s.counts["known_failed"] == 4
    assert all(ob.cached for ob in made["P2"].told if ob.status == "compile_error")
    assert s.winners == {"P2": "E"} and s.stop_reason == "searched"


def test_failed_builds_are_no_work_after_the_cap(tmp_path):
    # Round 1 spends the cap on A..D, which all fail. P2's untried candidates
    # are then failed builds and E, which needs a build: nothing is left to
    # run, so the search stops at the cap instead of asking again.
    ad = FakeAdapter({"P1": ABCD, "P2": list("ABCDE")}, fail_build=set(ABCD))
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [["A"], ["E"]]},
                      new_builds_per_run=4, batch_per_problem=4)
    assert s.stop_reason == "build_cap" and s.counts["rounds"] == 1
    assert len(made["P2"].asks) == 1 and co.ledger.builds == 4


def test_unattempted_is_requeued_once(tmp_path):
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted", "ok"]})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"], ["C"]]})
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B"), ("P1", "B"), ("P1", "C")]
    assert ad.measured[1].trial_uid == ad.measured[2].trial_uid
    assert ad.measured[1].call_id != ad.measured[2].call_id
    assert made["P1"].asks == [2, 1]  # the re-queued trial takes a batch slot
    assert s.counts["requeued"] == 1 and co.ledger.trials["P1"] == 3
    assert [ob.status for ob in made["P1"].told] == ["ok", "ok", "ok"]
    assert [e["admission"] for e in events(tmp_path / "out", "trial")] == [
        "admitted", "admitted", "readmitted", "admitted"]


def test_host_memory_stops_the_problem_and_is_never_told(tmp_path):
    # The adapter could not fit P2's host data in the host's memory: no
    # kernel failed, so its strategy is told nothing, its trials are refunded
    # and P2 stops with the adapter's diag, while P1 runs on to its winner.
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD},
                     script={("P2", n): ["host_memory"] for n in ABCD})
    # P2's strategy is done once asked: its stop reason stays the diag.
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"], ["C"]], "P2": [["A", "B"]]})
    assert made["P2"].told == [] and made["P2"].asks == [2]
    assert co.ledger.trials == {"P1": 3, "P2": 0}
    assert s.counts["host_memory"] == 1 and s.counts["failed_by_status"] == {}
    assert len(events(tmp_path / "out", "host_memory")) == 2
    summary = strict(tmp_path / "out" / "summary.json")
    assert summary["problems"]["P2"]["finished"] == "fake host_memory"
    assert "fake host_memory" in s.unresolved["P2"]
    assert (s.status, s.exit_code, s.winners) == ("partial", 1, {"P1": "A"})


def test_host_memory_after_measurements_keeps_the_problems_best_kernels(tmp_path):
    # P1's call of B ran out of host memory (its worker was killed from
    # outside) after A was measured: B is told nothing, P1 stops, and its best
    # kernel so far (A) is still confirmed and wins. P2 runs on as before.
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD}, script={("P1", "B"): ["host_memory"]})
    s, made, co = run(tmp_path, ad, {"P1": [["A"], ["B"], ["C"]], "P2": [["A"], ["B"]]},
                      batch_per_problem=1)
    assert [ob.trial.uid for ob in made["P1"].told] == ["P1:1"]  # A only
    assert [r.kernel_name for r in ad.measured if r.problem_key == "P1"] == ["A", "B"]
    assert co.ledger.trials == {"P1": 1, "P2": 2}  # B refunded
    assert s.counts["host_memory"] == 1 and s.counts["failed_by_status"] == {}
    assert [r.kernel_name for c in ad.confirmed for r in c if r.problem_key == "P1"] == [
        "A"] * co.settings.confirm.fresh_calls
    summary = strict(tmp_path / "out" / "summary.json")
    assert summary["problems"]["P1"]["finished"] == "fake host_memory"
    assert (s.status, s.exit_code, s.winners) == ("complete", 0, {"P1": "A", "P2": "A"})


def test_unattempted_twice_is_run_error_and_cancel_is_dropped(tmp_path):
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted", "unattempted"],
                                           ("P1", "C"): ["cancelled"]})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B", "C"]]}, batch_per_problem=3)
    told = {ob.trial.uid: ob for ob in made["P1"].told}
    assert told["P1:2"].status == "run_error"
    assert "unattempted twice" in told["P1:2"].detail
    assert "P1:3" not in told and s.counts["cancelled"] == 1
    assert co.ledger.trials["P1"] == 2  # A and B charged, cancelled C refunded


def _warm_real(tmp_path, name, script, trials=10):
    """A real strategy on P1 (A..L, A fastest) warm-started with A then B."""
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"P1": ["A", "B"]}))
    lat = {("P1", n): 1.0 + i for i, n in enumerate(KERNELS)}
    ad = FakeAdapter({"P1": list(KERNELS)}, lat=lat, script=script)
    co = AS.SearchCoordinator(
        ad, settings(strategy=name, trials_per_problem=trials, new_builds_per_run=40,
                     batch_per_problem=4, warm_start={"pairs_file": str(picks)}),
        tmp_path / "out", clock=FakeClock(), pair_loader=json_pairs,
    )
    return co.run(), ad, co


@pytest.mark.parametrize("name", ["hill", "portfolio", "de"])
def test_requeued_trial_is_pending_not_done(tmp_path, name):
    # Hill and the portfolio's warm-start barrier wait for the re-admitted A:
    # their empty ask in that round means waiting, not done.
    s, ad, co = _warm_real(tmp_path, name, {("P1", "A"): ["unattempted"]})
    assert s.counts["requeued"] == 1 and co.ledger.trials["P1"] == 10
    summary = strict(tmp_path / "out" / "summary.json")
    assert summary["problems"]["P1"]["finished"] == "trial budget"
    assert s.winners == {"P1": "A"}


def test_cancelled_trial_ends_a_strategy_that_waits_for_it(tmp_path):
    # The portfolio starts its members once every warm start is known; a
    # cancelled warm start is never told, so it would wait forever.
    s, ad, co = _warm_real(tmp_path, "portfolio", {("P1", "A"): ["cancelled"]})
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B")]
    assert s.counts["cancelled"] == 1 and len(events(tmp_path / "out", "cancel")) == 1
    finished = strict(tmp_path / "out" / "summary.json")["problems"]["P1"]["finished"]
    assert finished == "strategy waits for 1 cancelled trial(s)"
    assert s.stop_reason == "searched" and s.winners == {"P1": "B"}


def test_requeued_trial_cancelled_when_the_search_stops(tmp_path):
    clock = FakeClock()
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted"]},
                     clock=clock, advance=10.0)
    s, _, co = run(tmp_path, ad, {"P1": [["A", "B"]]}, clock=clock,
                   trials_per_problem=2, deadline_s=10)
    assert s.stop_reason == "deadline" and pairs(ad.measured) == [
        ("P1", "A"), ("P1", "B")]
    assert s.counts["cancelled"] == 1 and co.ledger.trials["P1"] == 1
    (cancel,) = events(tmp_path / "out", "cancel")
    assert cancel["trial_uid"] == "P1:2" and "before re-admission" in cancel["diag"]


def test_same_round_duplicate_waits_for_the_first(tmp_path):
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "A"): ["unattempted", "ok"]})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "A"]]})
    # One sample of A: the re-queued trial runs again, its twin is told the result.
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "A")]
    assert ad.measured[0].trial_uid == ad.measured[1].trial_uid == "P1:1"
    told = {ob.trial.uid: ob for ob in made["P1"].told}
    assert not told["P1:1"].cached and told["P1:2"].cached
    assert told["P1:2"].value == told["P1:1"].value == 1.0
    assert co.ledger.trials["P1"] == 1 and s.counts["cached"] == 1


def test_config_refused_by_the_space_is_not_charged(tmp_path):
    ad = FakeAdapter({"P1": ABCD})
    s, made, co = run(tmp_path, ad, {"P1": [[(0, 0), "B"]]})  # (16, 64): no kernel
    assert pairs(ad.measured) == [("P1", "B")]
    assert s.counts["invalid"] == 1 and co.ledger.trials["P1"] == 1
    assert made["P1"].told[0].status == "invalid" and made["P1"].told[0].key is None


def test_run_is_single_use(tmp_path):
    co = AS.SearchCoordinator(FakeAdapter({"P1": ABCD}), settings(), tmp_path / "o",
                              strategy_factory=scripted({"P1": [["A"]]})[0])
    assert co.run().status == "complete"
    with pytest.raises(RuntimeError, match="runs once"):
        co.run()


# -------------------------------------------------------------- budgets


def test_trial_budget_boundary(tmp_path):
    ad = FakeAdapter({"P1": list("ABCDEFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [[n] for n in "ABCDEFGH"]},
                      trials_per_problem=3, batch_per_problem=2)
    assert made["P1"].asks == [2, 2, 1]  # the last ask is cut to the trials left
    assert co.ledger.trials["P1"] == 3
    s2, made2, co2 = run(tmp_path / "b", FakeAdapter({"P1": list("ABCDEFGH")}),
                         {"P1": [list("ABCDEFGH")]}, trials_per_problem=3,
                         batch_per_problem=2)
    assert made2["P1"].asks == [2, 1]
    assert s2.counts["measured"] == 3 and co2.ledger.trials["P1"] == 3
    summary = strict(tmp_path / "b" / "out" / "summary.json")
    assert summary["problems"]["P1"]["finished"] == "trial budget"


def test_build_cap_is_never_exceeded(tmp_path):
    ad = FakeAdapter({"P1": ABCD, "P2": list("EFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [list("EFGH")]},
                      new_builds_per_run=4, batch_per_problem=4)
    assert ad.builds == [ABCD] and co.ledger.builds == 4
    assert made["P2"].asks == []  # no capacity left for P2's first ask
    assert s.stop_reason == "build_cap" and s.status == "partial"
    assert "build_cap" in s.unresolved["P2"]


def test_build_cap_counts_new_builds_not_cached_trials(tmp_path):
    """Capacity is debited by the new builds a round needs: P2 re-using P1's
    kernels leaves room for P3 in the same round."""
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": list("EFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"]], "P2": [["A", "B"]],
                                     "P3": [["E", "F"]]},
                      new_builds_per_run=4, batch_per_problem=2)
    assert ad.builds == [["A", "B", "E", "F"]] and co.ledger.builds == 4
    assert [made[p].asks for p in ("P1", "P2", "P3")] == [[2], [2], [2]]
    assert s.counts["measured"] == 6 and s.status == "complete"


@pytest.mark.parametrize("factory", ["scripted", "random"])
def test_build_cap_admits_work_that_needs_no_build(tmp_path, factory):
    """Two identical four-kernel domains, trials = batch = cap = 4: P2 re-uses
    the four kernels P1 builds, in the same round, without another build."""
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD})
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [ABCD]},
                      factory=None if factory == "scripted" else AS._default_factory,
                      new_builds_per_run=4, batch_per_problem=4)
    assert [sorted(b) for b in ad.builds] == [ABCD] and co.ledger.builds == 4
    assert sorted(pairs(ad.measured)) == [(p, n) for p in ("P1", "P2") for n in ABCD]
    assert s.status == "complete" and s.counts["cancelled"] == 0


def test_requeued_trial_runs_again_after_the_build_cap(tmp_path):
    # The cap is spent in round 1, but B is built: re-admitting it needs none.
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted"]})
    s, _, co = run(tmp_path, ad, {"P1": [["A", "B"]]}, trials_per_problem=2,
                   new_builds_per_run=2)
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B"), ("P1", "B")]
    assert s.stop_reason == "searched" and s.counts["cancelled"] == 0
    assert co.ledger.trials["P1"] == 2 and co.ledger.builds == 2


def test_proposal_needing_a_build_past_the_cap_is_told_invalid(tmp_path):
    # Round 1: P1 plans the whole cap (A, B); P2 may still ask for kernels that
    # need no build: A runs, E would need a third build and is told invalid at
    # once, without a trial, and its slot goes to B, which needs no new build.
    ad = FakeAdapter({"P1": ABCD, "P2": list("ABCE")})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"]], "P2": [["A", "E"], ["C"]]},
                      trials_per_problem=2, new_builds_per_run=2)
    assert ad.builds == [["A", "B"]] and made["P2"].asks == [2]
    assert sorted(pairs(ad.measured)) == [("P1", "A"), ("P1", "B"), ("P2", "A"),
                                          ("P2", "B")]
    assert not events(tmp_path / "out", "cancel") and s.counts["cancelled"] == 0
    trials = {e["trial_uid"]: e for e in events(tmp_path / "out", "trial")
              if e["problem"] == "P2"}
    assert {u: (e["kernel"], e["admission"]) for u, e in trials.items()} == {
        "P2:1": ("A", "admitted"), "P2:2": ("E", "past_build_cap"),
        "refill:1": ("B", "refilled")}
    assert trials["refill:1"]["origin"] == "refill"
    told = {ob.trial.uid: ob for ob in made["P2"].told}
    assert told["P2:2"].status == "invalid" and told["P2:2"].key is not None
    assert "new_builds_per_run" in told["P2:2"].detail
    assert told["refill:1"].ok and told["refill:1"].value == 1.0
    assert (s.counts["past_build_cap"], s.counts["refilled"]) == (1, 1)
    assert co.ledger.trials["P2"] == 2 and not made["P2"].pending
    assert s.status == "complete"


def test_past_the_cap_a_refill_keeps_the_run_from_stalling(tmp_path):
    # P1 spends the cap on A, B, C in round 1. In round 2 P2's strategy proposes
    # only kernels that need a build (D, E): each is told invalid, and its slot
    # goes to a built kernel P2 has not tried, the fastest on P1 first (C, B).
    # Without the refill round 2 has no new result: the run stops stalled with
    # P2 at 1 of its 3 trials and A, its slowest kernel, as its winner.
    lat = {("P1", "A"): 3.0, ("P1", "B"): 2.0, ("P1", "C"): 1.0,
           ("P2", "A"): 3.0, ("P2", "B"): 2.0, ("P2", "C"): 1.0}
    ad = FakeAdapter({"P1": list("ABCDE"), "P2": list("ABCDE")}, lat=lat)
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B", "C"]], "P2": [["A"], ["D", "E"]]},
                      trials_per_problem=3, new_builds_per_run=3, batch_per_problem=3,
                      max_stall_rounds=1)
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B"), ("P1", "C"), ("P2", "A"),
                                  ("P2", "C"), ("P2", "B")]
    assert (s.stop_reason, s.winners) == ("searched", {"P1": "C", "P2": "C"})
    assert (s.counts["past_build_cap"], s.counts["refilled"]) == (2, 2)
    assert co.ledger.trials["P2"] == 3 and not made["P2"].pending
    assert [ob.trial.origin for ob in made["P2"].told] == ["", "", "", "refill", "refill"]



def test_refill_ranks_by_relative_latency_on_other_problems(tmp_path):
    # P1 (best 1 ms) and P2 (best 100 ms) spend the cap in round 1; E fails
    # verification on P1, so no problem measured it. In round 2 P3 proposes
    # four kernels that need a build: the refill takes P3's built, untried
    # kernels by their lowest latency relative to each other problem's best:
    # C (1.1 on P2), B (1.5 on P1), D (5.0 on P2), then E (unmeasured). By the
    # worst ratio or by absolute latency B would come before C; unmeasured
    # first would start with E.
    lat = {("P1", "A"): 1.0, ("P1", "B"): 1.5, ("P1", "C"): 3.0,
           ("P2", "A"): 100.0, ("P2", "C"): 110.0, ("P2", "D"): 500.0}
    ad = FakeAdapter({"P1": list("ABCE"), "P2": list("ACD"), "P3": list("ABCDEFGHI")},
                     lat=lat, script={("P1", "E"): ["numeric_fail"]})
    s, made, co = run(tmp_path, ad, {"P1": [list("ABCE")], "P2": [list("ACD")],
                                     "P3": [["A"], list("FGHI")]},
                      trials_per_problem=5, new_builds_per_run=5, batch_per_problem=4)
    assert ad.builds == [list("ABCED")] and s.counts["refilled"] == 4
    assert [n for p, n in pairs(ad.measured) if p == "P3"] == list("ACBDE")


def _cap_grid():
    """144 kernels, the same list for every problem."""
    shapes = itertools.product((32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128),
                               ("mem", "compv3", "compv4"))
    return {f"k{i:03d}": {"tile_m": m, "tile_n": n, "tile_k": k, "pipeline": pl}
            for i, (m, n, k, pl) in enumerate(shapes)}


class CapAdapter(FakeAdapter):
    """``_cap_grid`` for every problem; a fixed pseudo-random latency per pair."""

    def __init__(self, problems):
        super().__init__({p: [] for p in problems})
        self.grid = _cap_grid()

    def enumerate_eligible(self):
        return {p: [AS.Candidate(n, n, r, None) for n, r in self.grid.items()]
                for p in self.problems}

    def _lat(self, r):
        h = hashlib.sha256((r.problem_key + r.kernel_name).encode()).hexdigest()
        return 1.0 + int(h[:6], 16) % 997 / 100


@pytest.mark.parametrize("name", ["de", "hill", "gp", "portfolio"])
def test_binding_build_cap_strands_no_strategy(tmp_path, name):
    # Three problems on one 144-kernel list, 32 trials each, 40 builds. Past the
    # cap, proposals that need a build are told invalid at once: no strategy
    # waits for them (DE for its generation, hill for its stage, the portfolio
    # for its members), GP keeps no lie for them, and each problem goes on
    # with the kernels already built (before: DE 72 and hill 57 of the 96
    # trials, GP and the portfolio about 100 and 60 pending trials a problem).
    ad = CapAdapter(["P1", "P2", "P3"])
    co = AS.SearchCoordinator(
        ad, settings(strategy=name, trials_per_problem=32, new_builds_per_run=40,
                     batch_per_problem=4),
        tmp_path / "out", clock=FakeClock())
    s = co.run()
    assert co.ledger.builds == 40 and s.counts["past_build_cap"] > 0
    assert s.counts["cancelled"] == 0
    for p in co.problems:
        assert not p.strategy.pending and "waits" not in p.finished, p.finished
    assert sum(co.ledger.trials.values()) >= 90, co.ledger.trials


def test_progress_lines_per_round_and_build(tmp_path):
    lines = []
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD}, fail_build={"D"})
    co = AS.SearchCoordinator(
        ad, settings(deadline_s=600), tmp_path / "out", clock=FakeClock(),
        strategy_factory=scripted({"P1": [["A", "B"]], "P2": [["B", "D"], ["C"]]})[0],
        log=lines.append)
    co.run()
    assert lines == [
        "  [search] round 1: building 3 kernel(s) (0/20 builds); 0/600 s",
        "  [search] round 1: 4 proposed, 4 admitted, 0 cached, 0 known failed, 0 past"
        " the build cap, 0 refilled; 2 built, 1 failed (3/20 builds); 3 measured,"
        " 3 ok; 1/2 problems active; 0/600 s",
        "  [search] round 2: building 1 kernel(s) (3/20 builds); 0/600 s",
        "  [search] round 2: 1 proposed, 1 admitted, 0 cached, 0 known failed, 0 past"
        " the build cap, 0 refilled; 1 built, 0 failed (4/20 builds); 1 measured,"
        " 1 ok; 0/2 problems active; 0/600 s",
        "  [search] confirming 4 finalist(s) of 2 problem(s), 3 fresh call(s) each",
    ]


def test_build_cap_rotates_the_first_problem(tmp_path):
    # Round 2 starts with P2, so the last two builds of the cap go to P2.
    ad = FakeAdapter({"P1": ABCD, "P2": list("EFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"], ["C", "D"]],
                                     "P2": [["E", "F"], ["G", "H"]]},
                      new_builds_per_run=6, batch_per_problem=2)
    assert ad.builds == [["A", "B", "E", "F"], ["G", "H"]]
    assert s.stop_reason == "build_cap" and co.ledger.builds == 6


@pytest.mark.parametrize("deadline, measured, wall", [(25, "ABC", 30.0),
                                                      (20, "AB", 20.0)])
def test_deadline_stops_admission_then_confirms(tmp_path, deadline, measured, wall):
    # Each measurement takes 10 s; a clock exactly at the deadline admits nothing.
    clock = FakeClock()
    ad = FakeAdapter({"P1": list("ABCDEF")}, clock=clock, advance=10.0)
    s, made, _ = run(tmp_path, ad, {"P1": [[n] for n in "ABCDEF"]}, clock=clock,
                     trials_per_problem=6, batch_per_problem=1, deadline_s=deadline)
    assert pairs(ad.measured) == [("P1", n) for n in measured]
    assert s.stop_reason == "deadline"
    assert s.counts["confirm_calls"] == 6 and s.status == "complete"
    assert s.wall_s == wall


def test_no_deadline_unless_one_is_set(tmp_path):
    # Each measurement takes 1000 s: without deadline_s the run still spends
    # its whole trial budget.
    clock = FakeClock()
    ad = FakeAdapter({"P1": list("ABCDEF")}, clock=clock, advance=1000.0)
    s, _, co = run(tmp_path, ad, {"P1": [[n] for n in "ABCDEF"]}, clock=clock,
                   trials_per_problem=6, batch_per_problem=1)
    assert len(ad.measured) == 6 and s.stop_reason == "searched"
    assert co.ledger.deadline_s is None and s.wall_s == 6000.0
    assert strict(tmp_path / "out" / "summary.json")["budget"]["deadline_s"] is None


def test_deadline_is_checked_before_every_ask(tmp_path):
    clock = FakeClock()

    class Slow(Scripted):
        def ask(self, n):  # a strategy that takes 10 s to propose
            clock.t += 10
            return super().ask(n)

    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": ABCD})
    s, _, _ = run(tmp_path, ad, {}, clock=clock, deadline_s=15,
                  factory=lambda name, space, *a: Slow(space, [["A"]]))
    assert pairs(ad.measured) == [("P1", "A"), ("P2", "A")]
    assert s.stop_reason == "deadline" and "deadline" in s.unresolved["P3"]


class Undone(Scripted):
    """Never reports done: only its empty ask can finish its problem."""

    @property
    def done(self):
        return False


def test_empty_ask_with_nothing_pending_finishes_the_problem(tmp_path):
    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {}, max_stall_rounds=50,
                  factory=lambda name, space, *a: Undone(space, [["A"], ["B"]]))
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B")]
    summary = strict(tmp_path / "out" / "summary.json")
    assert summary["problems"]["P1"]["finished"] == "strategy done"
    assert s.stop_reason == "searched" and s.counts["rounds"] == 3


def test_stall_counter_resets_on_progress(tmp_path):
    # Two stalled rounds in a row stop the run, but never two in a row here.
    steps = [["A"], ["A"], ["B"], ["A"], ["C"], ["B"], ["D"]]
    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {}, max_stall_rounds=2,
                  factory=lambda name, space, *a: Undone(space, steps))
    assert [n for _, n in pairs(ad.measured)] == ["A", "B", "C", "D"]
    assert s.stop_reason == "searched" and s.counts["cached"] == 3


def test_stall_stops_duplicate_proposals(tmp_path):
    ad = FakeAdapter({"P1": ABCD})
    s, made, _ = run(tmp_path, ad, {"P1": [["A"]] * 20}, max_stall_rounds=3)
    assert pairs(ad.measured) == [("P1", "A")]
    assert s.stop_reason == "stalled" and s.counts["cached"] == 3
    cached = [ob for ob in made["P1"].told if ob.cached]
    assert len(cached) == 3 and all(ob.value == 1.0 for ob in cached)
    assert s.winners == {"P1": "A"}


# -------------------------------------------------------------- outputs


def test_outputs_are_strict_json_and_winners_load_as_pairs(tmp_path):
    sys.path.insert(0, str(TE_OPS / "gemm"))
    sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
    import gemm_full_benchmark as drv

    lat = {("512x512x512", "A"): math.inf, ("512x512x512", "B"): 0.5}
    ad = FakeAdapter({"512x512x512": ABCD}, lat=lat)
    s, _, _ = run(tmp_path, ad, {"512x512x512": [["A", "B"]]})
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted(AS.OUTPUT_FILES)
    for name in ("run.json", "summary.json", "winners.json"):
        strict(out / name)
    text = (out / "trials.jsonl").read_text()
    assert "Infinity" not in text and "NaN" not in text
    assert all(e["latency_ms"] is None for e in events(out, "result")
               if e["kernel"] == "A")
    assert drv.load_pair_list(out / "winners.json") == {"512x512x512": ["B"]}
    meta = strict(out / "run.json")
    assert (meta["sampling_policy"], meta["exhaustive"]) == ("adaptive", False)
    assert meta["ck_search"]["origin_sha"] == ck_search.ORIGIN_SHA
    assert meta["context"]["arch"] == "gfx950"
    assert meta["problems"]["512x512x512"]["eligible"] == 4
    summary = strict(out / "summary.json")
    assert summary["counts"]["failed_by_status"] == {"invalid_measurement": 1}
    assert summary["winners"]["512x512x512"]["calls_ms"] == [0.5, 0.5, 0.5]


def test_output_dir_holding_a_run_is_refused(tmp_path):
    run(tmp_path, FakeAdapter({"P1": ABCD}), {"P1": [["A"]]})
    with pytest.raises(AS.SettingsError, match="already holds a run"):
        run(tmp_path, FakeAdapter({"P1": ABCD}), {"P1": [["A"]]})


@pytest.mark.parametrize("out", ["out", "out/sub"])
def test_output_dir_that_is_a_file_is_refused(tmp_path, out):
    (tmp_path / "out").write_text("")
    with pytest.raises(AS.SettingsError, match="not a directory"):
        AS.RunWriter(tmp_path / out).check_fresh()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
@pytest.mark.parametrize("out", ["ro", "ro/run", "ro/run/sub"])
def test_output_dir_without_write_access_is_refused(tmp_path, out):
    (tmp_path / "ro").mkdir(mode=0o555)
    with pytest.raises(AS.SettingsError, match="ro is not writable"):
        AS.RunWriter(tmp_path / out).check_fresh()


class _FullDisk:
    """A trials.jsonl handle on a full disk."""

    def write(self, text):
        raise OSError(28, "No space left on device")

    def flush(self):
        pass

    def close(self):
        raise OSError(28, "No space left on device")


def test_output_dir_failing_mid_run_ends_with_a_summary_not_a_traceback(
        tmp_path, capsys):
    # The first failed write ends the search as an error; recording that error
    # and the confirmation are skipped instead of failing again, the final
    # files are still tried (here only trials.jsonl fails).
    ad = FakeAdapter({"P1": ABCD})
    co = AS.SearchCoordinator(ad, settings(), tmp_path / "out", clock=FakeClock(),
                              strategy_factory=scripted({"P1": [["A"], ["B"]]})[0])
    real = ad.measure_pairs

    def measure(requests):
        co.writer._fh = _FullDisk()
        return real(requests)

    ad.measure_pairs = measure
    s = co.run()
    assert (s.stop_reason, s.status, s.exit_code) == ("error", "no_winner", 1)
    assert "No space left on device" in s.error and "cannot write" in s.error
    assert ad.confirmed == [] and len(ad.measured) == 1
    summary = strict(tmp_path / "out" / "summary.json")
    assert (summary["status"], summary["exit_code"], summary["error"]) == (
        "no_winner", 1, s.error)
    assert "cannot write" in capsys.readouterr().err


@pytest.mark.parametrize("fail_on", ["winners.json", "stop", "summary.json"])
def test_failed_final_write_exits_1_with_what_can_be_written(
        tmp_path, capsys, monkeypatch, fail_on):
    # The first failed write is one of the last three: the run is not
    # complete, and each final file after it is still written and says why.
    # The stop event (trials.jsonl) is skipped after a failed write, like every
    # event, and cannot know of a later one.
    def failing(real):
        def write(self, name, data):
            if name == fail_on:
                raise OSError(28, "No space left on device")
            return real(self, name, data)
        return write

    for attr in ("_event", "_write_json"):
        monkeypatch.setattr(AS.RunWriter, attr, failing(getattr(AS.RunWriter, attr)))
    out = tmp_path / "out"
    rc = AS.run_search(FakeAdapter({"P1": ABCD}), settings(), out, ["0"])
    cap = capsys.readouterr()
    assert rc == 1 and "SEARCH PARTIAL: 1/1" in cap.out and "COMPLETE" not in cap.out
    assert "No space left on device" in cap.err
    written = {p.name for p in out.iterdir()}
    assert ("winners.json" in written) == (fail_on != "winners.json")
    assert ("summary.json" in written) == (fail_on != "summary.json")
    if fail_on != "summary.json":
        summary = strict(out / "summary.json")
        assert (summary["status"], summary["exit_code"]) == ("partial", 1)
        assert "No space left on device" in summary["error"]
    stops = [e["status"] for e in events(out, "stop")]
    assert stops == {"winners.json": [], "stop": [],
                     "summary.json": ["complete"]}[fail_on]


class Interrupting(FakeAdapter):
    """Ctrl-C during round 2's build, or during the first confirmation call."""

    def __init__(self, problems, when):
        super().__init__(problems)
        self.when = when

    def build_missing(self, cands):
        if self.when == "search" and self.builds:
            raise KeyboardInterrupt
        return super().build_missing(cands)

    def confirm(self, requests):
        if self.when == "confirmation":
            raise KeyboardInterrupt
        return super().confirm(requests)


@pytest.mark.parametrize("when", ["search", "confirmation"])
def test_interrupt_writes_the_summary_and_exits_130(tmp_path, when):
    ad = Interrupting({"P1": ABCD, "P2": ABCD}, when)
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]], "P2": [["C"], ["D"]]})
    assert (s.stop_reason, s.exit_code, s.winners) == ("interrupted", 130, {})
    assert f"interrupted during the {when}" in s.error
    out = tmp_path / "out"
    summary = strict(out / "summary.json")
    assert (summary["status"], summary["exit_code"]) == ("no_winner", 130)
    assert strict(out / "winners.json") == {}
    # The discovery results are kept, unconfirmed, for a rerun or a --pairs list.
    assert {k: p["discovery_best"]["kernel"] for k, p in summary["problems"].items()
            } == {"P1": "A", "P2": "C"}
    assert events(out, "stop")[-1]["stop_reason"] == "interrupted"
    assert ad.confirmed == [] or when == "confirmation"


@pytest.mark.parametrize("sigint", [signal.default_int_handler, signal.SIG_IGN],
                         ids=["sigint_default", "sigint_ignored"])
def test_run_search_turns_sigterm_into_an_interrupt(tmp_path, capsys, sigint):
    seen = []

    class Terminated(FakeAdapter):
        def measure_pairs(self, requests):
            seen.append([signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)])
            os.kill(os.getpid(), signal.SIGTERM)  # a batch system's time limit
            return super().measure_pairs(requests)

    old = signal.signal(signal.SIGINT, sigint)
    try:
        before = [signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)]
        rc = AS.run_search(Terminated({"P1": ABCD}), settings(), tmp_path / "out", ["0"])
        after = [signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)]
    finally:
        signal.signal(signal.SIGINT, old)
    assert rc == 130 and after == before
    assert strict(tmp_path / "out" / "summary.json")["stop_reason"] == "interrupted"
    assert "SEARCH NO_WINNER" in capsys.readouterr().out
    # Ctrl-C takes the same handler, unless SIGINT is ignored (a background job).
    (term, intr), = seen
    assert intr is (term if sigint is signal.default_int_handler else signal.SIG_IGN)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_interrupt_handler_ends_a_forked_child(sig):
    # A forked build worker inherits the handler: there the signal ends the
    # process (as by default) instead of raising in the job it runs.
    handler = AS._interrupt_handler(os.getpid())
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        try:
            handler(sig, None)
        finally:
            os._exit(3)  # reached only if the handler returned or raised
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == sig
    with pytest.raises(KeyboardInterrupt):
        handler(sig, None)


# The pool job lives in a module of its own: a worker started by forkserver or
# spawn imports it (it cannot unpickle a function of a "python -c" __main__).
_POOL_JOB = """
import time

def build(marker):  # one build: say it started, then compile for a while
    open(marker, "w").close()
    time.sleep(60)
"""
_POOL_RUN = """
import multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pool_job import build

def main(tests, tmp, method):
    sys.path.insert(0, tests)
    from test_adaptive_search_coordinator import ABCD, AS, FakeAdapter, settings

    class Pool(FakeAdapter):  # builds like setup_multiple_gemm_dispatchers
        def build_missing(self, cands):
            ctx = multiprocessing.get_context(method)
            with ProcessPoolExecutor(1, mp_context=ctx) as ex:  # the 2nd build waits
                futs = [ex.submit(build, tmp + "/started-" + c.canonical_key)
                        for c in cands]
                for fut in as_completed(futs):
                    fut.result()
            return super().build_missing(cands)

    return AS.run_search(Pool({"P1": ABCD}), settings(), tmp + "/out", ["0"])

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
"""


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs process groups")
@pytest.mark.parametrize("method", [m for m in ("fork", "forkserver", "spawn")
                                    if m in multiprocessing.get_all_start_methods()])
def test_sigterm_to_the_process_group_does_not_wait_for_queued_builds(tmp_path, method):
    # A batch system's time limit sends SIGTERM to every process of the job.
    # The build worker dies whatever its start method (forked, it inherits the
    # handler; started by forkserver or spawn, it has SIGTERM's default), the
    # pool breaks and the summary is written at once; a worker that turned it
    # into one job's error would start the queued build (a second marker) and
    # the summary would wait for it.
    (tmp_path / "pool_job.py").write_text(_POOL_JOB)
    (tmp_path / "pool_run.py").write_text(_POOL_RUN)
    proc = subprocess.Popen([sys.executable, str(tmp_path / "pool_run.py"),
                             str(Path(__file__).parent), str(tmp_path), method],
                            start_new_session=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    try:
        for _ in range(600):
            if list(tmp_path.glob("started-*")) or proc.poll() is not None:
                break
            time.sleep(0.05)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        rc = proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    assert rc == 130, proc.stderr.read().decode()
    assert len(list(tmp_path.glob("started-*"))) == 1
    assert strict(tmp_path / "out" / "summary.json")["stop_reason"] == "interrupted"


def _ended_by_error(tmp_path, s, message):
    """The run stopped on an error and still wrote every output file."""
    out = tmp_path / "out"
    assert (s.stop_reason, s.exit_code) == ("error", 1) and message in s.error
    assert sorted(p.name for p in out.iterdir()) == sorted(AS.OUTPUT_FILES)
    summary = strict(out / "summary.json")
    assert (summary["error"], summary["exit_code"]) == (s.error, 1)
    (err,) = events(out, "error")
    assert message in err["message"] and "Traceback" in err["traceback"]
    assert events(out, "stop")[-1]["stop_reason"] == "error"


class Broken(FakeAdapter):
    """Correct in round 1, then breaks its contract as ``how`` says."""

    def __init__(self, problems, how):
        super().__init__(problems)
        self.how, self.calls = how, 0

    def build_missing(self, cands):
        out = super().build_missing(cands)
        if self.how == "build_keys" and len(self.builds) > 1:
            out[cands[0].canonical_key + "x"] = out.pop(cands[0].canonical_key)
        return out

    def measure_pairs(self, requests):
        out = super().measure_pairs(requests)
        self.calls += 1
        if self.calls == 1:
            return out
        first = out[0]
        if self.how == "lost":
            return out[1:]
        if self.how == "repeated":
            return out + [first]
        if self.how == "unrequested":
            return [replace(first, request=replace(first.request, call_id="z9"))]
        if self.how == "status":
            return [replace(first, status="skipped")]
        return out


@pytest.mark.parametrize("how, message", [
    ("lost", "no outcome"),
    ("repeated", "unrequested or repeated"),
    ("unrequested", "unrequested or repeated"),
    ("status", "expected a PairOutcome"),
    ("build_keys", "one BuildResult per requested key"),
])
def test_adapter_contract_violation_ends_the_run_unconfirmed(tmp_path, how, message):
    # A in round 1 was fine, but an adapter that broke its contract is not
    # trusted with a confirmation either.
    ad = Broken({"P1": ABCD}, how)
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]]})
    _ended_by_error(tmp_path, s, message)
    assert pairs(ad.measured)[0] == ("P1", "A") and ad.confirmed == []
    assert (s.status, s.unresolved) == ("no_winner", {"P1": "search ended: error"})


class DiesInRound2(FakeAdapter):
    def build_missing(self, cands):
        if self.builds:
            raise RuntimeError("BrokenProcessPool: a build worker died")
        return super().build_missing(cands)


def test_adapter_failure_still_confirms_what_was_measured(tmp_path):
    ad = DiesInRound2({"P1": ABCD, "P2": ABCD, "P3": ABCD}, fail_build={"D"})
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]], "P2": [["C"], ["B"]],
                                 "P3": [["D"], ["B"]]})
    _ended_by_error(tmp_path, s, "build_missing failed: RuntimeError: BrokenProcessPool")
    assert s.counts["confirm_calls"] == 6 and s.winners == {"P1": "A", "P2": "C"}
    assert (s.status, s.exit_code) == ("partial", 1)  # never complete after an error
    assert strict(tmp_path / "out" / "winners.json") == {"P1": ["A"], "P2": ["C"]}
    assert s.unresolved["P3"].startswith("no verified discovery measurement")


def test_a_run_with_an_error_is_never_complete(tmp_path):
    ad = DiesInRound2({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]]})
    assert s.winners == {"P1": "A"} and (s.status, s.exit_code) == ("partial", 1)


def test_confirmation_failure_is_recorded(tmp_path):
    ad = FakeAdapter({"P1": ABCD})

    def boom(requests):
        raise OSError("device lost")

    ad.confirm = boom
    s, _, _ = run(tmp_path, ad, {"P1": [["A"]]})
    assert (s.stop_reason, s.status, s.exit_code) == ("searched", "no_winner", 1)
    assert s.error == "confirmation: confirm failed: OSError: device lost"
    assert s.unresolved == {"P1": s.error}
    assert strict(tmp_path / "out" / "summary.json")["error"] == s.error


def test_strategy_error_still_writes_the_summary(tmp_path):
    class Singular(Scripted):
        def ask(self, n):
            if self.told:
                raise FloatingPointError("singular kernel matrix")
            return super().ask(n)

    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {},
                  factory=lambda name, space, *a: Singular(space, [["A"], ["B"]]))
    _ended_by_error(tmp_path, s, "FloatingPointError: singular kernel matrix")
    assert s.winners == {"P1": "A"} and s.status == "partial"


def test_strategy_asking_more_than_n_ends_the_run(tmp_path):
    class Greedy(Scripted):
        def ask(self, n):
            return super().ask(n + 1)

    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {},
                  factory=lambda name, space, *a: Greedy(space, [["A", "B", "C"]]))
    _ended_by_error(tmp_path, s, "RuntimeError: P1: strategy asked 3 > 2 trials")
    assert ad.measured == [] and s.status == "no_winner"


def test_adapter_failure_at_setup_still_writes_the_summary(tmp_path):
    ad = FakeAdapter({"P1": ABCD})

    def boom():
        raise OSError("disk full")

    ad.enumerate_eligible = boom
    s, _, _ = run(tmp_path, ad, {"P1": [["A"]]})
    _ended_by_error(tmp_path, s, "enumerate_eligible failed: OSError: disk full")
    assert s.status == "no_winner" and not s.unresolved
    assert strict(tmp_path / "out" / "run.json")["error"] == s.error


# ------------------------------------------------------------- settings


BASE = {"schema_version": 1, "strategy": "tpe", "trials_per_problem": 4,
        "new_builds_per_run": 8}


@pytest.mark.parametrize("change, match", [
    ({"schema_version": None}, "schema_version: required"),
    ({"schema_version": 2}, "expected 1"),
    ({"strategy": "annealing"}, "expected one of"),
    ({"trials_per_problem": 0}, ">= 1"),
    ({"trials_per_problem": True}, "expected type int"),
    ({"trials_per_problem": 4.0}, "expected type int"),
    ({"new_builds_per_run": 3}, ">= trials_per_problem"),
    ({"deadline_s": 0}, "> 0"),
    ({"deadline_s": math.inf}, "expected type float"),
    ({"deadline_s": math.nan}, "expected type float"),
    ({"deadline_s": "60"}, "expected type float"),
    ({"seed": 1.5}, "expected type int"),
    ({"seed": "7"}, "expected type int"),
    ({"seed_policy": "random"}, "one of"),
    ({"measurement_policy": "quick"}, "one of"),
    ({"batch_per_problem": 0}, ">= 1"),
    ({"max_stall_rounds": 0}, ">= 1"),
    ({"build_timeout_s": 0}, ">= 1"),
    ({"build_timeout_s": 90.5}, "expected type int"),
    ({"verbose": True}, "unknown key"),
    ({"confirm": {"finalists": 2, "calls": 3}}, "unknown key"),
    ({"confirm": {"fresh_calls": 0}}, ">= 1"),
    ({"confirm": {"finalists": 0}}, ">= 1"),
    ({"warm_start": {"top_k": 3}}, "pairs_file, ranked_csv and/or cover"),
    ({"warm_start": {"cover": ""}}, "a candidate field name"),
    ({"warm_start": {"cover": "   "}}, "a candidate field name"),
    ({"warm_start": {"cover": " pipeline "}}, "a candidate field name"),
    ({"warm_start": {"cover": ["pipeline"]}}, "expected type str"),
    ({"warm_start": {"pairs_file": "missing.json"}}, "no such file"),
    ({"warm_start": {"pairs_file": 7}}, "expected type str"),
    ({"warm_start": None}, "expected a JSON object"),
    ({"host_cache": {"dir": "missing-dir"}}, "no such directory"),
    ({"host_cache": {"max_entry_gib": 0}}, "> 0"),
    ({"host_cache": {"max_total_gib": -1}}, ">= 0"),
    ({"host_cache": {"max_total_gib": "8"}}, "expected type float"),
    ({"host_cache": {"enabled": False}}, "unknown key"),
])
def test_settings_errors(change, match):
    d = {**BASE, **change}
    d = {k: v for k, v in d.items() if v is not None or k == "warm_start"}
    with pytest.raises(AS.SettingsError, match=match):
        AS.SearchSettings.from_json(d)


def test_settings_warm_rules_and_files(tmp_path):
    (tmp_path / "picks.json").write_text("{}")
    (tmp_path / "picks.csv").write_text("problem,rank,kernel,predicted_tflops\n")
    bad = [
        ({"strategy": "random", "warm_start": {"pairs_file": "picks.json"}}, "ignores"),
        ({"strategy": "random", "warm_start": {"cover": "pipeline"}}, "cover: .* ignores"),
        ({"strategy": "exhaustive", "warm_start": {"cover": "pipeline"}}, "ignores"),
        ({"strategy": "de", "warm_start": {"ranked_csv": "picks.csv",
                                          "use_prior": True}}, "read a prior"),
        ({"warm_start": {"ranked_csv": "picks.csv"}}, "use_prior"),
        ({"warm_start": {"pairs_file": "picks.json", "use_prior": True}}, "use_prior"),
        ({"warm_start": {"pairs_file": "picks.json", "top_k": 0}}, ">= 1"),
        ({"warm_start": {"ranked_csv": "picks.csv", "use_prior": 1}}, "type bool"),
        ({"warm_start": {"ranked_csv": "picks.csv", "use_prior": "yes"}}, "type bool"),
        ({"host_cache": {"dir": "picks.json"}}, "no such directory"),
    ]
    for change, match in bad:
        (tmp_path / "s.json").write_text(json.dumps({**BASE, **change}))
        with pytest.raises(AS.SettingsError, match=match):
            AS.SearchSettings.from_json(tmp_path / "s.json")
    for text, match in [("{", "invalid JSON"), ('{"seed": NaN}', "NaN"),
                        ('{"seed": 1, "seed": 2}', "duplicate")]:
        (tmp_path / "s.json").write_text(text)
        with pytest.raises(AS.SettingsError, match=match):
            AS.SearchSettings.from_json(str(tmp_path / "s.json"))


def test_settings_defaults_paths_and_round_trip(tmp_path, monkeypatch):
    (tmp_path / "picks.json").write_text("{}")
    (tmp_path / "cache").mkdir()
    cfg = {**BASE, "warm_start": {"pairs_file": "picks.json"},
           "host_cache": {"dir": "cache"}}
    (tmp_path / "search.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path / "cache")  # relative paths: the file's dir
    s = AS.SearchSettings.from_json(tmp_path / "search.json")
    assert s.warm_start.pairs_file == str(tmp_path / "picks.json")
    assert s.host_cache.dir == str(tmp_path / "cache")
    assert (s.seed, s.seed_policy, s.deadline_s, s.batch_per_problem) == (
        0, "shared", None, 4)
    assert "deadline_s" not in s.to_json() and "build_timeout_s" not in s.to_json()
    assert (s.max_stall_rounds, s.confirm.finalists, s.confirm.fresh_calls) == (8, 2, 3)
    assert s.warm_start.top_k == 5 and not s.warm_start.use_prior
    assert s.build_timeout_s is None  # the op's search default
    assert AS.SearchSettings.from_json(s.to_json()) == s
    timed = AS.SearchSettings.from_json({**BASE, "build_timeout_s": 240})
    assert AS.SearchSettings.from_json(timed.to_json()).build_timeout_s == 240
    assert AS.SearchSettings.from_json(BASE).to_json()["confirm"] == {
        "finalists": 2, "fresh_calls": 3}
    plain = AS.SearchSettings.from_json(BASE)
    assert plain.host_cache == AS.HostCacheSettings(None, 4.0, None)
    assert plain.to_json()["host_cache"] == {"max_entry_gib": 4.0}
    cached = AS.SearchSettings.from_json({**BASE, "host_cache": {
        "dir": str(tmp_path / "cache"), "max_total_gib": 0}})
    assert cached.host_cache == AS.HostCacheSettings(str(tmp_path / "cache"), 4.0, 0.0)
    assert AS.SearchSettings.from_json(cached.to_json()) == cached


# ------------------------------------------------- seeds, warm, prior


def test_option_check_reads_only_the_options_the_op_names(tmp_path):
    # A driver without --variant (batched, multi_d, stream-k) binds the shared
    # check with its own options: no other attribute of its args is read.
    cfg = tmp_path / "search.json"
    cfg.write_text(json.dumps({"schema_version": 1, "strategy": "tpe",
                               "trials_per_problem": 2, "new_builds_per_run": 4}))
    args = argparse.Namespace(search_config=str(cfg), batch_count=8,
                              search_output_dir=str(tmp_path / "out"), csv=None)
    assert AS.check_search_options(args).strategy == "tpe"
    assert AS.check_search_options(args, [("--csv", "csv")]).trials_per_problem == 2
    with pytest.raises(AS.SettingsError, match=r"does not take --batch-count \(why\)$"):
        AS.check_search_options(args, [("--batch-count", "batch_count", "why")])
    with pytest.raises(AS.SettingsError, match="supports --batch-count 4 only, got 8"):
        AS.check_search_options(args, required=[("--batch-count", "batch_count", 4)])


@pytest.mark.parametrize("policy, seeds", [("shared", [7, 7, 7]),
                                           ("per_problem", [7, 8, 9])])
def test_seed_policies(tmp_path, policy, seeds):
    calls = []
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": ABCD})
    run(tmp_path, ad, {}, calls=calls, seed=7, seed_policy=policy)
    assert [c["seed"] for c in calls] == seeds
    meta = strict(tmp_path / "out" / "run.json")
    assert [meta["problems"][p]["seed"] for p in ("P1", "P2", "P3")] == seeds


def test_warm_start_order_and_pair_loader(tmp_path):
    sys.path.insert(0, str(TE_OPS / "gemm"))
    sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
    import gemm_full_benchmark as drv

    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"P1": ["D", "Z", "B", "A"], "P2": "*", "P9": ["A"]}))
    warm = {"pairs_file": str(picks), "top_k": 2}
    ad = FakeAdapter({"P1": ABCD, "P2": ["C", "A", "B", "D"], "P3": ABCD})
    with pytest.raises(AS.SettingsError, match="pair_loader"):
        run(tmp_path, ad, {}, strategy="hill", warm_start=warm)
    calls = []
    run(tmp_path, ad, {}, calls=calls, strategy="hill", warm_start=warm,
        pair_loader=drv.load_pair_list)
    assert [c["warm"] for c in calls] == [
        [rec("D"), rec("B")], [rec("C"), rec("A")], []]
    meta = strict(tmp_path / "out" / "run.json")
    assert [meta["problems"][p]["warm_start"] for p in ("P1", "P2", "P3")] == [
        ["D", "B"], ["C", "A"], []]


PIPES = ("p0", "p1", "p2")


class PipeAdapter(FakeAdapter):
    """``FakeAdapter`` whose records also hold a pipeline, kernel i's PIPES[i % 3]."""

    def enumerate_eligible(self):
        return {p: [AS.Candidate(n, n, {**rec(n), "pipeline": PIPES[KERNELS.index(n) % 3]},
                                 None) for n in names]
                for p, names in self.problems.items()}


def test_warm_start_cover_alone_one_kernel_per_value(tmp_path):
    # No pairs_file or ranked_csv: one candidate per pipeline, values in
    # candidate order, drawn with the problem's seed (the same on a re-run,
    # and other seeds draw other kernels).
    s = AS.SearchSettings.from_json({**BASE, "warm_start": {"cover": "pipeline"}})
    assert s.warm_start.cover == "pipeline" and s.warm_start.pairs_file is None
    assert AS.SearchSettings.from_json(s.to_json()) == s
    names = list(KERNELS[:9])
    runs = []
    for i, seed in enumerate((0, 0, 1, 2, 3)):
        calls = []
        run(tmp_path / str(i), PipeAdapter({"P1": names}), {}, calls=calls,
            strategy="hill", seed=seed, warm_start={"cover": "pipeline"})
        meta = strict(tmp_path / str(i) / "out" / "run.json")["problems"]["P1"]
        assert [w["pipeline"] for w in calls[0]["warm"]] == list(PIPES)
        assert meta["warm_start"] == [meta["warm_cover"][v] for v in PIPES]
        runs.append(meta["warm_start"])
    assert runs[0] == runs[1] and len({tuple(r) for r in runs}) > 2


@pytest.mark.parametrize("leading", ["", "ADG"])
def test_warm_start_cover_shared_groups_ignore_other_groups(tmp_path, leading):
    # Odd-K problems can lose comp_async while retaining identical TDM groups.
    # Removing or resizing that leading group must not cost extra shared builds.
    shared = "BCEFHIKL"
    calls = []
    run(
        tmp_path,
        PipeAdapter({"P1": list("ADGJ" + shared), "P2": list(leading + shared)}),
        {},
        calls=calls,
        strategy="tpe",
        seed=2,
        warm_start={"cover": "pipeline"},
    )
    assert len(calls) == 2 and calls[0]["seed"] == calls[1]["seed"]
    picks = [{r["pipeline"]: r for r in c["warm"]} for c in calls]
    assert list(picks[0]) == list(PIPES)
    assert list(picks[1]) == list(PIPES if leading else PIPES[1:])
    for pipeline in PIPES[1:]:
        assert picks[0][pipeline] == picks[1][pipeline]


def test_warm_start_cover_is_not_truncated_by_top_k(tmp_path):
    # top_k limits the pairs_file picks only; cover adds the values they lack.
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"P1": ["D", "A", "B"]}))  # D: p0, A: p0, B: p1
    calls = []
    run(tmp_path, PipeAdapter({"P1": list(KERNELS[:9])}), {}, calls=calls,
        strategy="tpe", pair_loader=json_pairs,
        warm_start={"pairs_file": str(picks), "top_k": 1, "cover": "pipeline"})
    warm = calls[0]["warm"]
    assert warm[0] == {**rec("D"), "pipeline": "p0"}
    assert [w["pipeline"] for w in warm] == ["p0", "p1", "p2"]
    meta = strict(tmp_path / "out" / "run.json")["problems"]["P1"]
    assert meta["warm_start"][0] == "D" and set(meta["warm_cover"]) == {"p1", "p2"}


def test_warm_start_cover_handles_partial_fields_and_duplicate_pairs(tmp_path):
    # A pick missing the covered field does not mask a None group, and
    # duplicate listed names in pairs_file do not consume extra top_k slots.
    vals = (None, "x", "y")

    class PartialAdapter(FakeAdapter):
        def enumerate_eligible(self):
            cands = [AS.Candidate("A", "A", rec("A"), None)]
            for i, name in enumerate(KERNELS[1:4]):
                cands.append(
                    AS.Candidate(name, name, {**rec(name), "tag": vals[i]}, None)
                )
            for name in KERNELS[4:6]:
                cands.append(AS.Candidate(name, name, rec(name), None))
            return {"P1": cands}

    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"P1": ["A", "A", "C"]}))  # A lacks tag; C has "x"
    ad = PartialAdapter({"P1": list(KERNELS[:6])})
    s = AS.SearchCoordinator(
        ad,
        settings(
            strategy="tpe",
            trials_per_problem=5,
            new_builds_per_run=10,
            batch_per_problem=1,
            warm_start={"pairs_file": str(picks), "top_k": 2, "cover": "tag"},
        ),
        tmp_path / "out",
        pair_loader=json_pairs,
        clock=FakeClock(),
    ).run()
    assert s.exit_code == 0, s.error
    meta = strict(tmp_path / "out" / "run.json")["problems"]["P1"]
    assert meta["warm_start"] == ["A", "C", "B", "D"]
    assert meta["warm_cover"] == {"None": "B", "y": "D"}
    assert [r.kernel_name for r in ad.measured[:4]] == ["A", "C", "B", "D"]


def test_warm_start_cover_unknown_field_ends_the_run(tmp_path):
    s, _, _ = run(tmp_path, PipeAdapter({"P1": list(KERNELS[:9])}), {},
                  strategy="tpe", warm_start={"cover": "pipline"})
    assert s.exit_code == 1 and "no candidate of P1 has the field 'pipline'" in s.error


def test_warm_start_cover_checks_every_problem_before_any_work(tmp_path):
    # A typo is caught when every problem is small (exhaustive, no warm start)
    # too, and a field whose value is not a single one is a settings error.
    class ListAdapter(FakeAdapter):
        def enumerate_eligible(self):
            return {p: [AS.Candidate(n, n, {**rec(n), "dims": [1, 2]}, None)
                        for n in names] for p, names in self.problems.items()}

    for i, (ad, cover, want) in enumerate((
            (PipeAdapter({"P1": list(KERNELS[:2])}), "pipline",
             "no candidate of P1 has the field 'pipline'"),
            (ListAdapter({"P1": list(KERNELS[:9])}), "dims",
             "field 'dims' of P1 has the value [1, 2]"))):
        s, _, _ = run(tmp_path / str(i), ad, {}, strategy="tpe",
                      warm_start={"cover": cover})
        assert s.exit_code == 1 and want in s.error and not ad.measured


def test_warm_start_cover_must_leave_trials_to_the_strategy(tmp_path):
    # As many warm picks as trials_per_problem would only measure the first
    # values in candidate order and never run the strategy: an error.
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"P1": ["D", "B"]}))  # D: p0, B: p1
    for i, (warm, covered) in enumerate((({}, 3), ({"pairs_file": str(picks)}, 1))):
        ad = PipeAdapter({"P1": list(KERNELS[:9])})
        s, _, _ = run(tmp_path / str(i), ad, {}, strategy="tpe", pair_loader=json_pairs,
                      trials_per_problem=3, warm_start={"cover": "pipeline", **warm})
        assert s.exit_code == 1 and not ad.measured
        assert f"P1 has 3 warm picks ({covered} from cover 'pipeline')" in s.error


@pytest.mark.parametrize("name", ["hill", "tpe", "gp", "de", "portfolio"])
def test_warm_start_cover_reaches_the_real_strategies(tmp_path, name):
    # Every pipeline is measured before the strategy's own picks, whatever the
    # batch size, so a rare pipeline is never left out of a short search.
    names = list(KERNELS)
    ad = PipeAdapter({"P1": names})
    co = AS.SearchCoordinator(
        ad, settings(strategy=name, trials_per_problem=6, new_builds_per_run=12,
                     batch_per_problem=1, warm_start={"cover": "pipeline"}),
        tmp_path / "out", clock=FakeClock())
    co.run()
    first = [PIPES[KERNELS.index(r.kernel_name) % 3] for r in ad.measured[:3]]
    assert first == list(PIPES)


def _ranked_settings(tmp_path, strategy="tpe"):
    """use_prior settings; the op's prior_loader reads the (placeholder) file."""
    path = tmp_path / "ranked.csv"
    path.write_text("read by the op's prior_loader\n")
    return {"strategy": strategy,
            "warm_start": {"ranked_csv": str(path), "use_prior": True}}


def test_prior_from_the_ops_prior_loader(tmp_path):
    # The op reads the file (GEMM: gemm_search.read_ranked_csv); the coordinator
    # keys the predicted latencies by config and makes a log-scale TablePrior.
    calls, loaded = [], []

    def prior_loader(path):
        loaded.append(path)
        return {"P1": {"B": 0.5, "A": 1.0, "Z": 0.1}}

    ad = FakeAdapter({"P1": ABCD, "P2": ["A", "B"]})
    with pytest.raises(AS.SettingsError, match="needs a prior_loader"):
        run(tmp_path, ad, {}, **_ranked_settings(tmp_path))
    run(tmp_path, ad, {}, calls=calls, prior_loader=prior_loader,
        **_ranked_settings(tmp_path))
    assert loaded == [str(tmp_path / "ranked.csv")]
    prior = calls[0]["prior"]
    got = prior.predict([rec("A"), rec("B"), rec("C"), rec("D")])
    assert got[:2] == pytest.approx([1.0, 0.5])
    assert math.isnan(got[2]) and math.isnan(got[3])
    assert prior.scale == "log" and prior.sign == 1.0  # lower is better
    assert calls[1]["name"] == "exhaustive" and calls[1]["prior"] is None
    meta = strict(tmp_path / "out" / "run.json")
    assert meta["problems"]["P1"]["prior_entries"] == 2


@pytest.mark.parametrize("error", [ValueError("line 3: bad number"),
                                   UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
                                   OSError("permission denied")])
def test_unreadable_warm_start_file_is_a_settings_error(tmp_path, error):
    def prior_loader(path):
        raise error

    with pytest.raises(AS.SettingsError, match="warm_start.ranked_csv: "):
        run(tmp_path, FakeAdapter({"P1": ABCD}), {}, prior_loader=prior_loader,
            **_ranked_settings(tmp_path))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["hill", "tpe", "gp", "de", "portfolio"])
def test_warm_starts_reach_the_real_strategies(tmp_path, name):
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"64x512x512": ["k17", "k03", "k05"]}))
    ad = GridAdapter(["64x512x512"])
    co = AS.SearchCoordinator(
        ad, settings(strategy=name, trials_per_problem=10, new_builds_per_run=24,
                     batch_per_problem=1, warm_start={"pairs_file": str(picks),
                                                      "top_k": 2}),
        tmp_path / "out", clock=FakeClock(), pair_loader=json_pairs,
    )
    co.run()
    assert [r.kernel_name for r in ad.measured[:2]] == ["k17", "k03"]
    assert co.ledger.trials["64x512x512"] == len(ad.measured) <= 10


def test_every_strategy_shares_the_runs_noise_model(tmp_path):
    ad = GridAdapter(["64x512x512", "256x512x512"])
    co = AS.SearchCoordinator(
        ad, settings(strategy="portfolio", trials_per_problem=6, new_builds_per_run=24),
        tmp_path / "out", clock=FakeClock())
    co.run()
    assert isinstance(co.noise, ck_search.NoiseModel)
    for p in co.problems:
        assert p.strategy.noise is co.noise
        assert all(m.noise is co.noise for m in p.strategy.members)
    meta = strict(tmp_path / "out" / "run.json")
    assert meta["ck_search"]["noise_model"]["floor"] == co.noise.floor


def test_prior_reaches_the_real_strategy(tmp_path, monkeypatch):
    made, real = [], ck_search.make_strategy

    def spy(name, space, seed=0, **kw):
        made.append((name, kw))
        return real(name, space, seed, **kw)

    monkeypatch.setattr(ck_search, "make_strategy", spy)
    ad = GridAdapter(["64x512x512"])
    co = AS.SearchCoordinator(
        ad, settings(trials_per_problem=10, new_builds_per_run=24,
                     **_ranked_settings(tmp_path)),
        tmp_path / "out", clock=FakeClock(),
        prior_loader=lambda path: {"64x512x512": {"k03": 1.0, "k17": 2.0}},
    )
    assert co.run().status == "complete"
    ((name, kw),) = made
    assert name == "tpe" and isinstance(kw["prior"], ck_search.TablePrior)
    got = kw["prior"].predict([ad.grid["k03"], ad.grid["k17"], ad.grid["k00"]])
    assert got[:2] == pytest.approx([1.0, 2.0]) and math.isnan(got[2])


def test_exhaustive_fallback_for_tiny_spaces(tmp_path):
    lat = {("P1", "A"): 2.0, ("P1", "B"): 1.0}
    ad = FakeAdapter({"P1": ["A", "B"], "P2": list("ABCDEF")}, lat=lat)
    co = AS.SearchCoordinator(ad, settings(trials_per_problem=3, batch_per_problem=3),
                              tmp_path / "out", clock=FakeClock())
    s = co.run()
    meta = strict(tmp_path / "out" / "run.json")
    assert meta["problems"]["P1"]["strategy"] == "exhaustive"
    assert "exhaustive" in meta["problems"]["P1"]["note"]
    assert meta["problems"]["P2"]["strategy"] == "random"
    assert sorted(r.kernel_name for r in ad.measured if r.problem_key == "P1") == [
        "A", "B"]
    assert s.winners["P1"] == "B" and s.status == "complete"


# -------------------------------------------------- real strategies


def _grid():
    out = {}
    for i, (tm, tn, k) in enumerate((tm, tn, k) for tm in (64, 128, 192, 256)
                                    for tn in (64, 128, 256) for k in (32, 64)):
        out[f"k{i:02d}"] = {"tile_m": tm, "tile_n": tn, "tile_k": k}
    return out


class GridAdapter(FakeAdapter):
    """24 kernels on a smooth landscape; each problem prefers another corner."""

    def __init__(self, problems):
        super().__init__(problems)
        self.grid = _grid()

    def enumerate_eligible(self):
        return {p: [AS.Candidate(n, n, r, None) for n, r in self.grid.items()]
                for p in self.problems}

    def _lat(self, r):
        g, m = self.grid[r.kernel_name], int(r.problem_key.split("x")[0])
        return 1.0 + abs(g["tile_m"] - m) / 64 + g["tile_n"] / 256 + g["tile_k"] / 64


@pytest.mark.parametrize("name", ck_search.STRATEGY_NAMES)
def test_real_strategies_end_to_end(tmp_path, name):
    ad = GridAdapter(["64x512x512", "256x512x512"])
    cfg = {"strategy": name, "trials_per_problem": 10, "new_builds_per_run": 24,
           "batch_per_problem": 3}
    co = AS.SearchCoordinator(ad, settings(**cfg), tmp_path / "out", clock=FakeClock())
    s = co.run()
    assert s.status == "complete" and s.exit_code == 0
    built = [k for b in ad.builds for k in b]
    assert len(built) == len(set(built)) == co.ledger.builds <= 24
    seen = set()
    for r in ad.measured:
        assert (r.problem_key, r.kernel_name) not in seen  # no duplicate samples
        seen.add((r.problem_key, r.kernel_name))
    for p in ad.problems:
        assert co.ledger.trials[p] == sum(r.problem_key == p for r in ad.measured)
        assert co.ledger.trials[p] <= 10
    assert len(events(tmp_path / "out", "result")) == len(ad.measured)


# ------------------------------------------------------------- host cache


def _cache_calls(spec, fields, value=1.0, shape=(3, 4), fn=None):
    """``fn`` (HC.cached by default) on (spec, fields, compute), and how many
    times compute ran."""
    import numpy as np

    ran = []

    def compute():
        ran.append(1)
        return {"x": np.full(shape, value, np.float32), "s": np.float64(value)}

    got = (fn or HC.cached)(spec, fields, compute)
    return got, len(ran)


def _entries(cache):
    return HC.usage(cache.dir)[0]


def test_host_cache_computes_once_per_key(tmp_path):
    import numpy as np

    cache = HC.HostCache(tmp_path)
    a = {"problem": {"M": 8, "N": 4, "K": 2}, "dtype": "bf16", "layout": "rcr"}
    first, ran1 = _cache_calls(cache.spec, a)
    again, ran2 = _cache_calls(cache.spec, dict(reversed(list(a.items()))))
    assert (ran1, ran2, _entries(cache)) == (1, 0, 1)
    assert set(again) == {"x", "s"} and (again["x"] == first["x"]).all()
    # A reader maps the entry read-only: workers reading it at once share it.
    assert isinstance(again["x"], np.memmap) and not again["x"].flags.writeable
    assert float(again["s"]) == 1.0
    # Every field is part of the key: another layout, dtype or problem misses.
    for change in ({"layout": "crr"}, {"dtype": "fp16"},
                   {"problem": {"M": 8, "N": 4, "K": 4}}):
        assert _cache_calls(cache.spec, {**a, **change}, 2.0)[1] == 1
    assert _entries(cache) == 4
    # Without a spec there is no cache.
    assert _cache_calls(None, a)[1] == _cache_calls(None, a)[1] == 1
    cache.close()
    assert not cache.dir.exists()


def test_host_cache_caps_fall_back_to_computing(tmp_path, monkeypatch):
    # An entry holds 48 + 8 bytes of arrays (x and s).
    small = HC.HostCache(tmp_path, max_entry_bytes=55)
    assert [_cache_calls(small.spec, {"k": 1})[1] for _ in range(2)] == [1, 1]
    assert _entries(small) == 0
    # The total counts every writer's reservation, the new entry's included,
    # and an entry takes no more than it reserved.
    full = HC.HostCache(tmp_path)
    assert _cache_calls(full.spec, {"k": 1})[1] == 1
    (one,) = HC.claims(full.dir).values()
    assert 56 < HC.usage(full.dir)[1] <= one
    spec = {**full.spec, "max_total_bytes": 2 * one - 1}
    got, ran = _cache_calls(spec, {"k": 2}, 2.0)
    assert ran == 1 and (got["x"] == 2.0).all() and _entries(full) == 1
    spec["max_total_bytes"] = 2 * one
    assert _cache_calls(spec, {"k": 2})[1] == 1 and _entries(full) == 2
    assert _cache_calls(spec, {"k": 2})[1] == 0
    # 10% of the filesystem stays free once every reserved entry is written:
    # the new one, and one still being written (a claim without its entry).
    disk = HC.shutil.disk_usage(tmp_path)
    floor = math.ceil(disk.total * HC.MIN_FREE_FRACTION)

    def free(n):
        monkeypatch.setattr(HC.shutil, "disk_usage", lambda p: disk._replace(free=floor + n))

    free(one // 2)
    assert _cache_calls(full.spec, {"k": 4})[1] == 1 and _entries(full) == 2
    (full.dir / f"{HC.CLAIM}{'0' * 32}-{one}").touch()
    free(one + one // 2)
    assert _cache_calls(full.spec, {"k": 4})[1] == 1 and _entries(full) == 2
    free(2 * one)
    assert _cache_calls(full.spec, {"k": 4})[1] == 1 and _entries(full) == 3
    # The default total: a quarter of the free space at the start.
    default = HC.HostCache(tmp_path)
    assert default.spec["max_total_bytes"] == (floor + 2 * one) // 4
    for cache in (small, full, default):
        cache.close()


def test_host_cache_room_is_its_cap_or_its_filesystem_less_reservations(tmp_path,
                                                                        monkeypatch):
    # room(): max_total_bytes less every reservation (written or not), and no
    # more than the filesystem's free space less the reservations not written
    # yet and 10% of its size; 0 when either is used up, or without a cache.
    gib = AS.GIB
    cache = HC.HostCache(tmp_path, max_total_bytes=10 * gib)
    (cache.dir / f"{HC.CLAIM}{'a' * 32}-{gib}").touch()  # written: its entry is there
    (cache.dir / ("a" * 32)).mkdir()
    (cache.dir / f"{HC.CLAIM}{'b' * 32}-{2 * gib}").touch()  # still being written

    def disk(total, free):
        monkeypatch.setattr(HC.shutil, "disk_usage", lambda p: argparse.Namespace(
            total=total * gib, used=(total - free) * gib, free=free * gib))

    disk(1000, 500)  # the cap is less: 10 - 1 - 2
    assert HC.room(cache.spec) == 7 * gib
    disk(100, 16)  # the filesystem is less: 16 - 2 - 10
    assert HC.room(cache.spec) == 4 * gib
    disk(100, 11)
    assert HC.room(cache.spec) == 0
    assert HC.room({**cache.spec, "max_total_bytes": 2 * gib}) == 0
    assert HC.room(None) == 0
    cache.close()
    assert HC.room(cache.spec) == 0  # unreadable


def test_host_cache_drops_the_entries_it_is_told_to(tmp_path):
    # An entry no call will read (its problem was skipped) is removed with its
    # writer's claim, so the room it held is free again; the others stay.
    cache = HC.HostCache(tmp_path)
    for k in (1, 2):
        assert _cache_calls(cache.spec, {"k": k})[1] == 1
    room = HC.room(cache.spec)
    (one,) = {n for key, n in HC.claims(cache.dir).items()}
    assert HC.drop(cache.spec, lambda fields: fields == {"k": 1}) == 1
    assert not HC.stored(cache.spec, {"k": 1}) and HC.stored(cache.spec, {"k": 2})
    assert list(HC.claims(cache.dir)) == [HC.entry_key({"k": 2})]
    assert HC.room(cache.spec) == room + one
    assert HC.drop(cache.spec, lambda fields: True) == 1 and _entries(cache) == 0
    assert HC.drop(None, lambda fields: True) == 0
    cache.close()
    assert HC.drop(cache.spec, lambda fields: True) == 0


def test_host_cache_bad_entries_are_misses(tmp_path):
    cache = HC.HostCache(tmp_path)
    fields = {"k": 1}
    path = cache.dir / HC.entry_key(fields)
    # A complete entry of other fields under this key (a hash collision).
    _cache_calls(cache.spec, {"k": 2}, 7.0)
    (cache.dir / HC.entry_key({"k": 2})).rename(path)
    got, ran = _cache_calls(cache.spec, fields, 5.0)
    assert ran == 1 and (got["x"] == 5.0).all()
    for meta in ("{", "[]", '{"fields": {"k": 1}, "arrays": ["missing"]}'):
        (path / "meta.json").write_text(meta)
        got, ran = _cache_calls(cache.spec, fields, 5.0)
        assert ran == 1 and (got["x"] == 5.0).all()
    # A write that fails (the run removed the dir) leaves the result alone.
    cache.close()
    got, ran = _cache_calls(cache.spec, {"k": 9}, 6.0)
    assert ran == 1 and (got["x"] == 6.0).all() and not cache.dir.exists()


def test_host_cache_concurrent_writers_publish_one_entry(tmp_path):
    import threading

    import numpy as np

    cache, start, out = HC.HostCache(tmp_path), threading.Barrier(8), []

    def compute():
        start.wait()  # all 8 miss together, then all publish
        return {"x": np.arange(1000, dtype=np.float32)}

    def worker():
        out.append(HC.cached(cache.spec, {"k": "same"}, compute)["x"])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 8 and all((x == out[0]).all() for x in out)
    path = cache.dir / HC.entry_key({"k": "same"})
    assert [p for p in cache.dir.iterdir() if p.is_dir()] == [path]
    assert list(HC.claims(cache.dir)) == [path.name]
    assert _cache_calls(cache.spec, {"k": "same"})[1] == 0
    # A published entry is never replaced: its key keeps its writer's claim,
    # and without the claim the rename onto the entry fails.
    inode = (path / "meta.json").stat().st_ino
    HC._publish(cache.spec, path, {"k": "same"}, {"x": out[0]})
    next(cache.dir.glob(HC.CLAIM + "*")).unlink()
    with pytest.raises(OSError):
        HC._publish(cache.spec, path, {"k": "same"}, {"x": out[0]})
    assert (path / "meta.json").stat().st_ino == inode
    assert [p for p in cache.dir.iterdir() if p.is_dir()] == [path]
    assert HC.claims(cache.dir) == {}  # a failed writer gives its claim back
    cache.close()


# A worker process of the concurrency test below: misses key argv[3] of the
# cache spec argv[2] when the file "go" appears, then publishes. The host
# checks (disk_usage) and the writes (np.save) are slowed down, so the checks
# of all workers overlap, and all of them run before any write lands. argv[5]
# > 0 replaces the filesystem's free bytes.
_RACER = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import host_cache as HC
spec, key, log, free = json.loads(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
disk, save = HC.shutil.disk_usage(spec["dir"]), np.save

def disk_usage(path):
    time.sleep(0.02)
    return disk._replace(free=free) if free else disk

def slow_save(file, a, **kw):
    (log / Path(file).parent.name).touch()  # one name per writer
    time.sleep(0.6)  # longer than all 8 checks
    save(file, a, **kw)

HC.shutil.disk_usage, np.save = disk_usage, slow_save
(log.parent / ("ready-" + key + "-" + sys.argv[6])).touch()
while not (log.parent / "go").exists():
    time.sleep(0.002)
HC.cached(spec, {"k": key}, lambda: {"x": np.ones(4096, np.float32)})
"""


def test_host_cache_concurrent_misses_hold_the_caps(tmp_path):
    # A problem's first round on several devices: its worker processes miss
    # together, then each publishes while the others are still writing. One
    # of them writes the key, and the total cap and the free-space guard count
    # every reservation, so they hold however many workers write at once.
    log = tmp_path / "writers"
    log.mkdir()

    def race(cache, keys, free=0):
        runs = [subprocess.Popen([sys.executable, "-c", _RACER, str(COMMON_DIR),
                                  json.dumps(cache.spec), k, str(log), str(free), str(i)])
                for i, k in enumerate(keys)]
        deadline = time.monotonic() + 60
        while (len(list(tmp_path.glob("ready-*"))) < len(keys)
               and time.monotonic() < deadline):
            time.sleep(0.01)
        (tmp_path / "go").touch()
        assert [r.wait(60) for r in runs] == [0] * len(keys)
        writers = len(list(log.iterdir()))
        for f in [*log.iterdir(), *tmp_path.glob("ready-*"), tmp_path / "go"]:
            f.unlink()
        return writers, _entries(cache), HC.usage(cache.dir)[1]

    keys = [f"k{i}" for i in range(8)]
    with HC.HostCache(tmp_path) as cache:
        assert race(cache, ["k0"] * 8)[:2] == (1, 1)
        (one,) = HC.claims(cache.dir).values()
    cap = 2 * one + one // 2
    with HC.HostCache(tmp_path, max_total_bytes=cap) as cache:
        writers, entries, size = race(cache, keys)
        assert (writers, entries) == (2, 2) and size <= cap
    disk = HC.shutil.disk_usage(tmp_path)  # constant: as if no write landed
    free = math.ceil(disk.total * HC.MIN_FREE_FRACTION) + cap
    with HC.HostCache(tmp_path, max_total_bytes=10 * cap) as cache:
        assert race(cache, keys, free)[:2] == (2, 2)


@pytest.mark.parametrize("kind", ["ext4", "nfs", "nfs4"])
def test_host_cache_warns_when_its_filesystem_is_too_full_or_nfs(tmp_path, monkeypatch,
                                                                 kind):
    # Every write leaves 10% of the filesystem free, so on one already under
    # that (a large, nearly full root disk) nothing is stored: the run says so.
    # On NFS a worker that waits for the cache's lock backs off for seconds:
    # the run says to use a local filesystem. Both lines come before any call.
    disk = HC.shutil.disk_usage(tmp_path)
    floor = math.ceil(disk.total * HC.MIN_FREE_FRACTION)
    settings = AS.HostCacheSettings(str(tmp_path), 1.0, 1.0)
    monkeypatch.setattr(HC, "fs_type", lambda p: kind)
    nfs = kind != "ext4"
    for free, full in ((floor - 1, True), (floor + 2**20, False)):
        monkeypatch.setattr(HC.shutil, "disk_usage", lambda p, f=free: disk._replace(free=f))
        lines = []
        with HC.run_cache(settings, lines.append) as cache:
            assert len(lines) == full + nfs  # before any call
            assert [_cache_calls(cache.spec, {"k": 1})[1] for _ in range(2)] == [1, int(full)]
        assert len(lines) == 1 + full + nfs and lines[-1].startswith("  Host cache: ")
        assert lines[0].startswith("  Host cache: WARNING: stores nothing: ") == full
        assert not full or f"under {tmp_path}, and a write must leave 10% free;" in lines[0]
        assert not full or lines[0].endswith("a tmpfs or another local filesystem")
        assert [n.startswith(f"  Host cache: NOTE: {tmp_path} is on {kind}, ")
                for n in lines[:-1]] == [False] * full + [True] * nfs


def test_fs_type_reads_the_mount_table(tmp_path):
    # The last mount on the longest mount point holding the path; a mount
    # table escapes a space as \040; a symlink counts where it points.
    root = os.path.realpath(tmp_path)
    table = tmp_path / "mounts"
    table.write_text(f"/dev/md0 / ext4 rw 0 0\nauto {root}/h autofs rw 0 0\n"
                     f"srv:/h {root}/h nfs4 rw 0 0\ntmpfs {root}/h/a\\040b tmpfs rw 0 0\n")
    (tmp_path / "h").mkdir()
    os.symlink(root + "/h", root + "/l")
    paths = ("", "/h", "/h/u/x", "/h/a b/c", "/h/a bc", "/hx", "/l/u")
    assert [HC.fs_type(root + p, str(table)) for p in paths] == [
        "ext4", "nfs4", "nfs4", "tmpfs", "nfs4", "ext4", "nfs4"]
    assert HC.fs_type(root, str(tmp_path / "none")) == ""


def test_host_allowance_takes_the_ops_counts_and_rate():
    # The shared formula has no rates of its own: the op passes its reference
    # flops, its input element count and its seconds per element.
    cases = ((0, 0, 0), (1, 1, 1e-9), (1e11, 0, 1), (5e10, 10**6, 1.5e-6))
    assert [AS.host_allowance_s(*c) for c in cases] == [0, 1, 2, 3]


def test_host_cache_prepare_stores_an_entry_the_calls_read(tmp_path):
    cache = HC.HostCache(tmp_path)
    f = {"k": 1}
    assert not HC.stored(cache.spec, f)
    assert _cache_calls(cache.spec, f, fn=HC.prepare) == (True, 1)
    assert HC.stored(cache.spec, f) and _entries(cache) == 1
    # Held already: neither preparing again nor a call computes it.
    assert _cache_calls(cache.spec, f, fn=HC.prepare) == (True, 0)
    assert _cache_calls(cache.spec, f, 5.0)[1] == 0
    # Nothing held: no cache, an entry over the caps, a broken entry.
    assert _cache_calls(None, f, fn=HC.prepare) == (False, 1)
    small = {**cache.spec, "max_entry_bytes": 1}
    assert _cache_calls(small, {"k": 2}, fn=HC.prepare) == (False, 1)
    assert not HC.stored(cache.spec, {"k": 2}) and _entries(cache) == 1
    (cache.dir / HC.entry_key(f) / "x.npy").unlink()
    assert not HC.stored(cache.spec, f)
    assert _cache_calls(cache.spec, f, fn=HC.prepare) == (False, 1)  # the key has its writer
    cache.close()
    assert not HC.stored(cache.spec, {"k": 3})


# A preparation worker: reads {"out", "sleep", "exit", "log", "pidfile",
# "sigkill"} and prints ``out`` after ``sleep`` s; ``log`` gets its start and
# end times; ``sigkill``: it is killed by SIGKILL (as by the OOM killer).
_PREP_JOB = r"""
import json, os, signal, sys, time
d = json.loads(sys.stdin.read())
if "pidfile" in d:
    with open(d["pidfile"] + ".tmp", "w") as f:
        f.write(str(os.getpid()))
    os.replace(d["pidfile"] + ".tmp", d["pidfile"])
t0 = time.time()
time.sleep(d.get("sleep", 0))
if "log" in d:
    with open(d["log"], "w") as f:
        f.write(f"{t0} {time.time()}")
sys.stderr.write(d.get("err", ""))
sys.stderr.flush()
if d.get("sigkill"):
    os.kill(os.getpid(), signal.SIGKILL)
sys.stdout.write(d.get("out", ""))
sys.stdout.flush()
os._exit(d.get("exit", 0))
"""


def _prep(jobs, limits=None, width=4, argv=None):
    payloads = {k: json.dumps(v) for k, v in jobs.items()}
    return AS.prepare_host_data(argv or [sys.executable, "-c", _PREP_JOB], None,
                                payloads, limits or {k: 60 for k in jobs}, width,
                                poll_s=0.02)


def test_prepare_host_data_reports_each_workers_outcome(monkeypatch):
    ok = json.dumps({"ok": True, "stored": True})
    got = _prep({
        "stored": {"out": ok},
        "after_noise": {"out": "a warning\n" + ok + "\n", "err": "a warning\n"},
        "refused": {"out": json.dumps({"ok": True, "stored": False})},
        "error": {"out": json.dumps({"ok": False, "error": "boom"}), "exit": 0},
        "no_ok": {"out": json.dumps({"ok": 1, "stored": True})},
        "crash": {"exit": 3},
        "traceback": {"exit": 1, "err": "x" * 1000 + "\n  File \"w.py\", line 9\n"
                      "ValueError: \n"},
        "garbled": {"out": "{"},
        "not_a_dict": {"out": "[]"},
        "hang": {"out": ok, "sleep": 30},
        "oom_killed": {"sigkill": True, "err": "a line\n"},
        "memory_error": {"out": json.dumps({"ok": False, "error": "MemoryError: "})},
        "import_oom": {"exit": 1, "err": "OpenBLAS error: Memory allocation still "
                       "failed after 10 retries, giving up.\n"},
        "noted": {"out": json.dumps({"ok": False, "error": "boom"}),
                  "err": "MemoryError: a warning\n"},
    }, {"hang": 0.5, **{k: 60 for k in ("stored", "after_noise", "refused", "error",
                                       "no_ok", "crash", "traceback", "garbled",
                                       "not_a_dict", "oom_killed", "memory_error",
                                       "import_oom", "noted")}})
    assert list(got) == ["stored", "after_noise", "refused", "error", "no_ok", "crash",
                         "traceback", "garbled", "not_a_dict", "hang", "oom_killed",
                         "memory_error", "import_oom", "noted"]
    brief = {k: (g["ok"], g["stored"], g["error"]) for k, g in got.items()}
    assert brief == {
        "stored": (True, True, ""), "after_noise": (True, True, ""),
        "refused": (True, False, ""), "error": (False, False, "boom"),
        "no_ok": (False, False, "failed (exit code 0)"),
        "crash": (False, False, "no result line (exit code 3)"),
        "traceback": (False, False, "no result line (exit code 1); stderr: "  # last 400
                      + "x" * 366 + " |   File \"w.py\", line 9 | ValueError:"),
        "garbled": (False, False, "no result line (exit code 0)"),
        "not_a_dict": (False, False, "no result line (exit code 0)"),
        # Killed at its limit (by SIGKILL too): not a memory error.
        "hang": (False, False, "no result in 0.5 s"),
        # Killed by another process, or out of memory: the problem is skipped.
        "oom_killed": (False, False, "host_memory: its preparation was killed (exit code "
                       "-9), likely by the out-of-memory killer; stderr: a line"),
        "memory_error": (False, False, "host_memory: its preparation ran out of memory "
                         "(MemoryError: )"),
        # numpy or its BLAS could not allocate at import, as its stderr says.
        "import_oom": (False, False, "host_memory: its preparation ran out of memory "
                       "(exit code 1); stderr: OpenBLAS error: Memory allocation still "
                       "failed after 10 retries, giving up."),
        # It reported its own failure (exit code 0): stderr is only a note.
        "noted": (False, False, "boom; stderr: MemoryError: a warning"),
    }
    assert 0.5 <= got["hang"]["seconds"] < 10
    bad = _prep({"k": {}}, argv=[os.devnull + "-missing"])
    assert not bad["k"]["ok"] and bad["k"]["error"].startswith("could not start")

    def no_tmp():
        raise OSError("no temp dir")

    monkeypatch.setattr(AS.tempfile, "TemporaryFile", no_tmp)
    assert _prep({"k": {}})["k"]["error"] == "could not start: no temp dir"


def test_prepare_host_data_runs_at_most_width_workers_at_once(tmp_path):
    jobs = {f"p{i}": {"out": json.dumps({"ok": True, "stored": True}), "sleep": 0.4,
                      "log": str(tmp_path / f"p{i}")} for i in range(5)}
    assert all(g["stored"] for g in _prep(jobs, width=2).values())
    spans = [tuple(map(float, (tmp_path / k).read_text().split())) for k in jobs]
    most = max(sum(a <= t < b for a, b in spans) for t, _ in spans)
    assert most == 2


def test_prepare_host_data_runs_within_the_memory_it_reads(tmp_path):
    # With needs, the preparations running at once fit in the memory read
    # whenever none runs (4 reads here): b waits for a, c does not fit even
    # alone and is never started (host_memory), d runs.
    gib, reads = AS.GIB, []
    jobs = {k: {"out": json.dumps({"ok": True, "stored": True}), "sleep": 0.3,
                "log": str(tmp_path / k)} for k in "abcd"}
    got = AS.prepare_host_data(
        [sys.executable, "-c", _PREP_JOB], None, {k: json.dumps(v) for k, v in jobs.items()},
        {k: 60 for k in jobs}, 4, poll_s=0.02,
        needs={"a": 6 * gib, "b": 6 * gib, "c": 20 * gib, "d": 3 * gib},
        memory=lambda: reads.append(1) or 10 * gib)
    assert {k: (g["ok"], g["error"]) for k, g in got.items()} == {
        "a": (True, ""), "b": (True, ""), "d": (True, ""),
        "c": (False, "host_memory: needs 20.0 GiB, 10.0 GiB available")}
    assert not (tmp_path / "c").exists() and len(reads) == 4
    spans = {k: tuple(map(float, (tmp_path / k).read_text().split())) for k in "abd"}
    assert spans["a"][1] <= spans["b"][0] and spans["b"][1] <= spans["d"][0]
    # Unknown memory (no /proc/meminfo): no guard.
    got = AS.prepare_host_data(
        [sys.executable, "-c", _PREP_JOB], None, {"c": json.dumps(jobs["a"])}, {"c": 60}, 1,
        poll_s=0.02, needs={"c": 20 * gib}, memory=lambda: None)
    assert got["c"]["stored"]


@pytest.mark.parametrize("b_stored", [True, False])
def test_prepare_host_data_keeps_the_entries_of_ended_preparations(tmp_path, b_stored):
    # b's entry (3 of its 9) stays in RAM on a tmpfs once b stored it, and
    # memory is read again only when none runs: c (10) must then wait for a
    # (30), since 30 + 10 + 3 > 40 even though 30 + 10 alone would fit. That
    # read counts b's entry (here: memory stays 40), so d (38) then fits
    # alone. An entry the cache did not store keeps nothing: c runs beside a.
    gib = AS.GIB
    jobs = {k: {"out": json.dumps({"ok": True, "stored": b_stored or k != "b"}),
                "sleep": s, "log": str(tmp_path / k)}
            for k, s in (("a", 0.8), ("b", 0.1), ("c", 0.1), ("d", 0.1))}
    got = AS.prepare_host_data(
        [sys.executable, "-c", _PREP_JOB], None, {k: json.dumps(v) for k, v in jobs.items()},
        {k: 60 for k in jobs}, 4, poll_s=0.02, memory=lambda: 40 * gib,
        needs={"a": 30 * gib, "b": 9 * gib, "c": 10 * gib, "d": 38 * gib},
        keeps={"a": 0, "b": 3 * gib, "c": 0, "d": 0})
    assert [g["stored"] for g in got.values()] == [True, b_stored, True, True], got
    spans = {k: tuple(map(float, (tmp_path / k).read_text().split())) for k in jobs}
    assert spans["b"][0] < spans["a"][1] and spans["c"][1] <= spans["d"][0]
    assert (spans["a"][1] <= spans["c"][0]) == b_stored


def _meminfo(tmp_path, total_gib, avail_gib):
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal:       {total_gib << 20} kB\nMemFree:  1 kB\n"
                    f"MemAvailable:   {avail_gib << 20} kB\n")
    return str(path)


def test_host_memory_reads_meminfo_and_cgroup_limits(tmp_path):
    gib, none = AS.GIB, str(tmp_path / "none")
    # MemAvailable less 10% of MemTotal.
    margin = AS.HOST_MEMORY_MARGIN
    got = AS.host_memory(_meminfo(tmp_path, 62, 40), none)
    assert got == int(40 * gib - margin * (62 * gib))
    assert AS.host_memory(_meminfo(tmp_path, 62, 3), none) == 0
    assert AS.host_memory(none, none) is None
    # cgroup v2: the room under a memory.max (its inactive file pages count
    # as free) and that limit as the total, at any level of the hierarchy.
    root = tmp_path / "cg"
    for d, lim in (("job", 8 * gib), ("job/step", "max")):
        (root / d).mkdir(parents=True)
        (root / d / "memory.max").write_text(f"{lim}\n")
        (root / d / "memory.current").write_text(f"{6 * gib}\n")
        (root / d / "memory.stat").write_text(f"anon 1\ninactive_file {gib}\n")
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/job/step\n")
    got = AS.host_memory(_meminfo(tmp_path, 503, 490), str(cgroup), str(root))
    assert got == int(3 * gib - margin * (8 * gib))
    # cgroup v2 memory.high (where reclaim and throttling start) below memory.max.
    (root / "job" / "step" / "memory.high").write_text(f"{7 * gib}\n")
    got = AS.host_memory(_meminfo(tmp_path, 503, 490), str(cgroup), str(root))
    assert got == int(2 * gib - margin * (7 * gib))
    # Both finite at one level: the lesser, either way round.
    (root / "job" / "step" / "memory.high").unlink()
    for high, least in ((7.5, 7.5), (9, 8)):
        (root / "job" / "memory.high").write_text(f"{int(high * gib)}\n")
        got = AS.host_memory(_meminfo(tmp_path, 503, 490), str(cgroup), str(root))
        assert got == int((least - 5) * gib - margin * (least * gib)), high
    # cgroup v1: memory.limit_in_bytes under the memory controller's own tree
    # (an unlimited level reads as a huge number).
    v1 = tmp_path / "v1"
    for d, lim in (("job", 8 * gib), ("", 2**63 - 4096)):
        (v1 / "memory" / d).mkdir(parents=True, exist_ok=True)
        (v1 / "memory" / d / "memory.limit_in_bytes").write_text(f"{lim}\n")
        (v1 / "memory" / d / "memory.usage_in_bytes").write_text(f"{6 * gib}\n")
        (v1 / "memory" / d / "memory.stat").write_text(
            f"inactive_file 0\ntotal_inactive_file {gib}\n")
    cgroup.write_text("1:name=systemd:/other\n4:cpu,memory:/job\n")
    got = AS.host_memory(_meminfo(tmp_path, 503, 490), str(cgroup), str(v1))
    assert got == int(3 * gib - margin * (8 * gib))


def test_host_lock_keeps_preparations_apart_from_builds_and_each_other(tmp_path,
                                                                       monkeypatch):
    # Builds share the lock; a preparation waits for every build and for the
    # preparation before it (another search run's, here a thread's), and says so.
    import threading

    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path))
    spans, logs = {}, []

    def hold(name, exclusive, start, s):
        time.sleep(start)
        with AS.host_lock(exclusive, logs.append, name):
            t0 = time.monotonic()
            time.sleep(s)
            spans[name] = (t0, time.monotonic())

    threads = [threading.Thread(target=hold, args=a) for a in (
        ("build1", False, 0.0, 0.6), ("build2", False, 0.1, 0.6),
        ("prep1", True, 0.2, 0.3), ("prep2", True, 0.3, 0.3))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    b1, b2, p1, p2 = (spans[k] for k in ("build1", "build2", "prep1", "prep2"))
    assert b2[0] < b1[1]  # builds overlap
    first, second = sorted((p1, p2))
    assert max(b1[1], b2[1]) <= first[0] and first[1] <= second[0]
    assert sorted(logs) == ["  [search] waiting for prep1", "  [search] waiting for prep2"]


def test_host_lock_opens_another_users_lock_file(tmp_path, monkeypatch):
    # Where fs.protected_regular is set, O_CREAT on another user's file in a
    # sticky dir such as /tmp fails (EACCES) although the file exists: an
    # existing lock file is opened without it, and still locks.
    import fcntl

    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path))
    lock = tmp_path / AS.HOST_LOCK
    lock.touch()
    real, logs = os.open, []

    def protected(path, flags, *a):
        if flags & os.O_CREAT and os.path.exists(path):
            raise PermissionError(13, "Permission denied", path)
        return real(path, flags, *a)

    def held():
        fd = real(str(lock), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return False
        except OSError:
            return True
        finally:
            os.close(fd)

    monkeypatch.setattr(os, "open", protected)
    with AS.host_lock(True, logs.append):
        assert held()
    assert logs == []


def test_host_lock_opens_the_file_another_run_created_meanwhile(tmp_path, monkeypatch):
    # Missing at the first open, created by another run before ours (O_EXCL
    # refuses): the run opens that file and locks it, and does not say it
    # could not.
    import fcntl

    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path))
    lock, real, logs, opens = tmp_path / AS.HOST_LOCK, os.open, [], []

    def racing(path, flags, *a):
        opens.append(flags)
        if len(opens) == 1:
            os.close(real(path, os.O_WRONLY | os.O_CREAT, 0o644))  # the other run's
            raise FileNotFoundError(2, "No such file or directory", path)
        return real(path, flags, *a)

    monkeypatch.setattr(os, "open", racing)
    with AS.host_lock(True, logs.append):
        fd = real(str(lock), os.O_RDONLY)
        with pytest.raises(OSError):
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        os.close(fd)
    assert opens == [os.O_RDONLY, os.O_RDONLY | os.O_CREAT | os.O_EXCL, os.O_RDONLY]
    assert logs == []


def test_host_lock_creates_a_lock_file_every_user_can_open(tmp_path, monkeypatch):
    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path))
    old = os.umask(0o077)
    try:
        with AS.host_lock(False):
            pass
    finally:
        os.umask(old)
    assert (tmp_path / AS.HOST_LOCK).stat().st_mode & 0o777 == 0o666


def test_host_lock_says_once_when_it_cannot_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path / "missing-dir"))
    monkeypatch.setattr(AS, "_NO_HOST_LOCK", set())
    logs, ran = [], []
    for exclusive in (False, True):
        with AS.host_lock(exclusive, logs.append):
            ran.append(exclusive)
    path = tmp_path / "missing-dir" / AS.HOST_LOCK
    assert ran == [False, True] and len(logs) == 1
    assert logs[0].startswith(f"  [search] NOTE: cannot open the host lock {path} (")
    assert logs[0].endswith("not kept apart from other search runs'")


def test_prepare_host_data_kills_its_workers_when_interrupted(tmp_path, monkeypatch):
    # Ctrl-C or SIGTERM reaches the run as KeyboardInterrupt while it waits.
    pidfile = tmp_path / "pid"

    def sleep(s):
        if pidfile.exists():
            raise KeyboardInterrupt
        time.sleep(s)

    monkeypatch.setattr(AS, "time", argparse.Namespace(monotonic=time.monotonic,
                                                       sleep=sleep))
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _prep({"k": {"pidfile": str(pidfile), "sleep": 60}}, {"k": 120})
    assert time.monotonic() - t0 < 30  # killed, not waited for
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_prepare_host_data_kills_every_worker_on_a_second_interrupt(monkeypatch):
    # A second Ctrl-C while the first one's cleanup waits for a worker to end
    # leaves none running: each was killed before any was waited for.
    procs = []

    class Popen(subprocess.Popen):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            procs.append(self)

        def wait(self, timeout=None):
            if timeout is None:  # the cleanup's wait (this test's have a timeout)
                raise KeyboardInterrupt
            return super().wait(timeout)

    def sleep(s):
        raise KeyboardInterrupt

    monkeypatch.setattr(AS, "subprocess", argparse.Namespace(Popen=Popen))
    monkeypatch.setattr(AS, "time", argparse.Namespace(monotonic=time.monotonic,
                                                       sleep=sleep))
    try:
        with pytest.raises(KeyboardInterrupt):
            _prep({k: {"sleep": 60} for k in "ab"}, width=2)
        assert [p.wait(timeout=30) for p in procs] == [-signal.SIGKILL] * 2
    finally:
        for p in procs:
            p.kill()
            subprocess.Popen.wait(p)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_prepare_host_data_kills_a_worker_interrupted_inside_popen(monkeypatch, sig):
    # A search run's SIGTERM or Ctrl-C arriving inside Popen, after the worker
    # was spawned but before Popen returned: it is held back until the
    # worker is recorded, so the cleanup kills and reaps it.
    pids, real = [], subprocess.Popen._execute_child

    def spawn(self, *a, **kw):
        real(self, *a, **kw)
        pids.append(self.pid)
        os.kill(os.getpid(), sig)
        for _ in range(100):  # the handler runs here, inside Popen
            time.sleep(0.001)

    monkeypatch.setattr(subprocess.Popen, "_execute_child", spawn)
    old = signal.signal(sig, AS._interrupt_handler(os.getpid()))
    try:
        with pytest.raises(KeyboardInterrupt):
            _prep({"a": {"sleep": 60}}, width=1)
        assert len(pids) == 1
        with pytest.raises(ChildProcessError):  # killed and reaped
            os.waitpid(pids[0], os.WNOHANG)
    finally:
        signal.signal(sig, old)
        for pid in pids:  # without the fix the worker is still running
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)


@pytest.mark.parametrize("record, calls, want", [
    (None, 4, 365),
    ({"ok": True, "stored": True, "seconds": 339.0}, 4, 365),
    ({"ok": True, "stored": False, "seconds": 339.0}, 1, 678),
    ({"ok": True, "stored": False, "seconds": 339.0}, 4, 2712),
    ({"ok": True, "stored": False, "seconds": 100.4}, 1, 365),
    ({"ok": True, "stored": False, "seconds": 100.4}, 2, 402),
    ({"ok": False, "stored": False, "seconds": 3710.0}, 4, 365),
])
def test_call_allowance_is_never_below_the_ops_allowance(record, calls, want):
    # A preparation that stored its entry took its time outside the calls; one
    # that finished without storing it measured what each call will compute,
    # perhaps alone on the CPUs that up to one call per device then shares:
    # the calls get twice that times the calls at once when it is more. A
    # failed one measured nothing.
    assert AS.call_allowance_s(365, record, calls) == want
    assert AS.prepare_limit_s(365) == 10 * 365 + 60


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="needs a directory this user cannot write")
def test_a_host_cache_that_cannot_be_created_is_left_out(tmp_path):
    # An existing host_cache.dir the run cannot write: one warning line and
    # no cache (it is optional), instead of a traceback before the search.
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    lines = []
    try:
        with HC.run_cache(AS.HostCacheSettings(str(ro), 1.0, 1.0), lines.append) as cache:
            assert cache is None
    finally:
        ro.chmod(0o755)
    assert len(lines) == 1, lines
    assert lines[0].startswith(f"  Host cache: WARNING: cannot create it under {ro} (")
    assert lines[0].endswith("; the run goes on without it") and list(ro.iterdir()) == []


def test_host_cache_dir_is_removed_on_exit_and_interrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(HC, "fs_type", lambda p: "")  # no NFS note, wherever tmp is
    lines = []
    settings = AS.HostCacheSettings(str(tmp_path), 1.0, 1.0)
    with HC.run_cache(settings, lines.append) as cache:
        assert cache.spec["max_entry_bytes"] == cache.spec["max_total_bytes"] == 2**30
        _cache_calls(cache.spec, {"k": 1})
    assert not cache.dir.exists() and lines == [
        f"  Host cache: 1 entries, 0.00 GiB, removed ({cache.dir})"]
    with pytest.raises(KeyboardInterrupt):
        with HC.run_cache(settings, lines.append) as cache:
            _cache_calls(cache.spec, {"k": 1})
            raise KeyboardInterrupt
    assert not cache.dir.exists()
    with HC.run_cache(AS.HostCacheSettings(str(tmp_path), 1.0, 0.0)) as off:
        assert off is None
    # A cache never closed is removed when the interpreter exits.
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import host_cache; "
            "c = host_cache.HostCache(sys.argv[2]); (c.dir / 'f').write_text('x'); "
            "print(c.dir); sys.exit(3)")
    run = subprocess.run([sys.executable, "-c", code, str(COMMON_DIR), str(tmp_path)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 3 and not Path(run.stdout.strip()).exists()
    assert list(tmp_path.iterdir()) == []
