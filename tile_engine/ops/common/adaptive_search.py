# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adaptive kernel search: an op-agnostic coordinator for the tile_engine drivers.

Instead of building and measuring every eligible (problem, kernel) pair, one
search strategy per problem (from the vendored ``ck_search`` package) proposes
which eligible kernel to measure next and learns from each verified result.
The coordinator owns the bookkeeping around that ask/tell loop:

- ``BudgetLedger``: per-problem trial attempts, a run-level cap on new builds
  and a deadline that stops admission (running work is drained);
- ``ArtifactRegistry``: one build per kernel per run, shared by every problem;
  a failed build is never retried and charges one physical build;
- ``SearchCoordinator``: synchronous rounds (ask, admit, build the union of
  misses, measure, reconcile, tell), then confirmation of the finalists with
  fresh verified calls; only confirmed winners are exported;
- ``RunWriter``: winners.json (a --pairs file), run.json, trials.jsonl and
  summary.json, all strict JSON.

The operation itself (which kernels fit a problem, how to build and measure
them) lives behind ``OperationAdapter``; the GEMM one is
``tile_engine/ops/gemm/gemm_search.py``. The plumbing every op binding shares
is here too: ``ArtifactStore`` (the run's own copies of the built kernels),
``CallFiles`` (per-call files under rounds/), ``measure_built``,
``host_allowance_s`` (a verified call's host time, at the op's rates),
``prepare_host_data`` (a problem's host cache entry, stored before its first
call and outside its timeout), the host memory guard (``host_memory``,
``calls_memory_error``, ``entries_to_come``, ``memory_failure``,
``host_lock``), ``source_info``,
``check_search_options`` and ``run_search`` (the driver's search mode, with
its exit code). Importing this module imports neither ``ck_search`` nor
numpy: the search core is loaded when a search runs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

import host_cache  # this directory's: imports no numpy

SCHEMA_VERSION = 1
STATUSES = (
    "ok",
    "compile_error",
    "numeric_fail",
    "timeout",
    "run_error",
    "invalid_measurement",
    "unattempted",
    "cancelled",
    "host_memory",
)
# Outcome status -> ck_search ``Status`` attribute told to the strategy. The
# three scheduler events are never told: ``unattempted`` is re-queued once
# (then it is a run_error), ``cancelled`` is dropped (it stays pending in its
# strategy, so a strategy that waits for it ends that problem's search) and
# ``host_memory`` (the problem's host data does not fit in this host's memory,
# see ``host_memory``, or a worker ran out of it, see ``memory_failure``)
# stops the problem: no kernel failed.
TELLABLE = {
    "ok": "OK",
    "compile_error": "COMPILE_ERROR",
    "numeric_fail": "NUMERIC_FAIL",
    "timeout": "TIMEOUT",
    "run_error": "RUN_ERROR",
    "invalid_measurement": "RUN_ERROR",
}
SEED_POLICIES = ("shared", "per_problem")
MEASUREMENT_POLICIES = ("full",)
PRIOR_STRATEGIES = ("tpe", "gp", "portfolio")  # the strategies that read a prior
NO_WARM_STRATEGIES = ("random", "exhaustive")  # the strategies that ignore warm starts
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
OUTPUT_FILES = ("run.json", "trials.jsonl", "winners.json", "summary.json")
_DISPATCHER_PY = Path(__file__).resolve().parents[3] / "dispatcher" / "python"


def _ck_search():
    """The vendored search core, imported on first use."""
    path = str(_DISPATCHER_PY)
    if path not in sys.path:
        sys.path.insert(0, path)
    import ck_search

    return ck_search


class SettingsError(ValueError):
    """Invalid search settings or options (the driver exits with code 2)."""


class AdapterError(RuntimeError):
    """An adapter call failed: the search ends, what it measured is still
    confirmed, and the run exits 1."""


class ContractError(AdapterError):
    """An adapter broke its contract (a lost, repeated or unknown outcome, a
    wrong build map): none of its results are trusted, nothing is confirmed."""


# ---------------------------------------------------------------- contract


@dataclass(frozen=True)
class Candidate:
    """One eligible kernel. ``record`` holds only the scalar config fields the
    search space is built from (no names, dtype or arch strings);
    ``canonical_key`` equals ``kernel_name`` within one run (one build context)."""

    canonical_key: str
    kernel_name: str
    record: dict
    original_config: object


@dataclass(frozen=True)
class BuildResult:
    """``path`` of the run-owned artifact, or None when the build failed."""

    path: Optional[str]
    reason: str = ""
    digest: str = ""


@dataclass(frozen=True)
class PairRequest:
    problem_key: str
    candidate_key: str
    kernel_name: str
    trial_uid: str
    call_id: str


@dataclass(frozen=True)
class PairOutcome:
    request: PairRequest
    status: str
    latency_ms: Optional[float]
    diag: str = ""
    device: str = ""  # where it ran, when the adapter knows (recorded only)


class OperationAdapter(Protocol):
    def enumerate_eligible(self) -> Dict[str, List[Candidate]]:
        """problem_key -> eligible candidates, in a stable order."""

    def build_missing(self, cands: Sequence[Candidate]) -> Dict[str, BuildResult]:
        """canonical_key -> BuildResult, one per candidate."""

    def measure_pairs(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        """Exactly one outcome per request; ``host_memory`` (its diag says
        why) for a request whose problem the host cannot hold, measured or
        not, or whose worker ran out of host memory: the problem stops."""

    def confirm(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        """Fresh verified calls, exactly one outcome per request. The
        requests of one problem should run on one device: they are compared."""

    def context(self) -> dict:
        """arch/dtype/layout/variant/sources, recorded in run.json."""


# ---------------------------------------------------------------- settings

_REQUIRED = object()


def _check_keys(data, allowed, where):
    if not isinstance(data, Mapping):
        raise SettingsError(f"{where}: expected a JSON object, got {data!r}")
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise SettingsError(
            f"{where}: unknown key(s) {unknown}, expected some of {sorted(allowed)}"
        )


def _get(data, where, name, kind, default=_REQUIRED, ok=None, want=""):
    """``data[name]`` checked against ``kind`` (bools are not ints) and ``ok``."""
    if name not in data:
        if default is _REQUIRED:
            raise SettingsError(f"{where}.{name}: required")
        return default
    v = data[name]
    if kind is float:
        good = isinstance(v, (int, float)) and math.isfinite(v)
    else:
        good = isinstance(v, kind)
    if not good or (kind is not bool and isinstance(v, bool)):
        raise SettingsError(f"{where}.{name}={v!r}: expected type {kind.__name__}")
    if ok is not None and not ok(v):
        raise SettingsError(f"{where}.{name}={v!r}: expected {want}")
    return float(v) if kind is float else v


def _get_file(data, where, name, base, kind="file"):
    """``data[name]``, a path relative to ``base``, of an existing ``kind``
    ("file" or "directory")."""
    v = _get(data, where, name, str, None, bool, "a path")
    if v is None:
        return None
    p = Path(v) if base is None else base / v  # an absolute v ignores base
    if not (p.is_file() if kind == "file" else p.is_dir()):
        raise SettingsError(f"{where}.{name}={v!r}: no such {kind} ({p})")
    return str(p)


def _positive(v):
    return v >= 1


def _strict_json(text, where):
    def no_constant(c):
        raise SettingsError(f"{where}: {c} is not valid JSON")

    def no_duplicates(pairs):
        keys = [k for k, _ in pairs]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise SettingsError(f"{where}: duplicate key(s) {dup}")
        return dict(pairs)

    try:
        return json.loads(
            text, parse_constant=no_constant, object_pairs_hook=no_duplicates
        )
    except ValueError as e:
        if isinstance(e, SettingsError):
            raise
        raise SettingsError(f"{where}: invalid JSON ({e})") from None


@dataclass(frozen=True)
class WarmStartSettings:
    pairs_file: Optional[str] = None
    ranked_csv: Optional[str] = None
    top_k: int = 5
    use_prior: bool = False


@dataclass(frozen=True)
class ConfirmSettings:
    finalists: int = 2
    fresh_calls: int = 3


@dataclass(frozen=True)
class HostCacheSettings:
    """The run's cache of each problem's verification inputs and reference
    (``host_cache.py``; the op binding uses it). ``dir`` None: the system temp
    directory; ``max_total_gib`` None: a quarter of its free space; 0: off."""

    dir: Optional[str] = None
    max_entry_gib: float = 4.0
    max_total_gib: Optional[float] = None


@dataclass(frozen=True)
class SearchSettings:
    """A validated ``--search-config`` (schema_version 1).

    Relative warm-start and host-cache paths are resolved against the settings
    file's directory (against the working directory when the settings come as
    a dict). There is no deadline unless ``deadline_s`` is given: the trial
    and build budgets bound a run. ``build_timeout_s`` limits each step of a
    kernel's build in the search; None leaves it to the op's search default.
    The coordinator reads neither it nor ``host_cache``: the op binding
    passes them to its build and its workers.
    """

    strategy: str
    trials_per_problem: int
    new_builds_per_run: int
    schema_version: int = SCHEMA_VERSION
    seed: int = 0
    seed_policy: str = "shared"
    deadline_s: Optional[float] = None
    batch_per_problem: int = 4
    max_stall_rounds: int = 8
    build_timeout_s: Optional[int] = None
    warm_start: Optional[WarmStartSettings] = None
    confirm: ConfirmSettings = field(default_factory=ConfirmSettings)
    measurement_policy: str = "full"
    host_cache: HostCacheSettings = field(default_factory=HostCacheSettings)

    @classmethod
    def from_json(cls, source) -> "SearchSettings":
        base = None
        if isinstance(source, Mapping):
            data = source
        elif isinstance(source, (str, os.PathLike)):
            path = Path(source)
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError) as e:
                raise SettingsError(f"search settings {path}: {e}") from None
            data = _strict_json(text, f"search settings {path}")
            base = path.resolve().parent
        else:
            raise SettingsError(f"search settings={source!r}: expected a path or dict")
        w = "settings"
        _check_keys(data, [f.name for f in cls.__dataclass_fields__.values()], w)
        _get(data, w, "schema_version", int, ok=lambda v: v == SCHEMA_VERSION,
             want=str(SCHEMA_VERSION))
        names = tuple(_ck_search().STRATEGY_NAMES)
        strategy = _get(data, w, "strategy", str, ok=lambda v: v in names,
                        want=f"one of {names}")
        trials = _get(data, w, "trials_per_problem", int, ok=_positive, want=">= 1")
        warm = None
        if "warm_start" in data:
            wd, ww = data["warm_start"], w + ".warm_start"
            _check_keys(wd, [f for f in WarmStartSettings.__dataclass_fields__], ww)
            warm = WarmStartSettings(
                pairs_file=_get_file(wd, ww, "pairs_file", base),
                ranked_csv=_get_file(wd, ww, "ranked_csv", base),
                top_k=_get(wd, ww, "top_k", int, 5, _positive, ">= 1"),
                use_prior=_get(wd, ww, "use_prior", bool, False),
            )
            if warm.pairs_file is None and warm.ranked_csv is None:
                raise SettingsError(f"{ww}: expected pairs_file and/or ranked_csv")
            if warm.use_prior != (warm.ranked_csv is not None):
                raise SettingsError(
                    f"{ww}: ranked_csv is read only as a prior: give both "
                    "ranked_csv and use_prior=true, or neither"
                )
            if warm.pairs_file is not None and strategy in NO_WARM_STRATEGIES:
                raise SettingsError(f"{ww}.pairs_file: strategy {strategy!r} ignores it")
            if warm.use_prior and strategy not in PRIOR_STRATEGIES:
                raise SettingsError(
                    f"{ww}.use_prior: only {PRIOR_STRATEGIES} read a prior, "
                    f"strategy is {strategy!r}"
                )
        cd, cw = data.get("confirm", {}), w + ".confirm"
        _check_keys(cd, [f for f in ConfirmSettings.__dataclass_fields__], cw)
        hd, hw = data.get("host_cache", {}), w + ".host_cache"
        _check_keys(hd, [f for f in HostCacheSettings.__dataclass_fields__], hw)
        return cls(
            strategy=strategy,
            trials_per_problem=trials,
            new_builds_per_run=_get(
                data, w, "new_builds_per_run", int, ok=lambda v: v >= trials,
                want=f">= trials_per_problem ({trials})",
            ),
            seed=_get(data, w, "seed", int, 0),
            seed_policy=_get(data, w, "seed_policy", str, "shared",
                             lambda v: v in SEED_POLICIES, f"one of {SEED_POLICIES}"),
            deadline_s=_get(data, w, "deadline_s", float, None, lambda v: v > 0, "> 0"),
            batch_per_problem=_get(data, w, "batch_per_problem", int, 4, _positive, ">= 1"),
            max_stall_rounds=_get(data, w, "max_stall_rounds", int, 8, _positive, ">= 1"),
            build_timeout_s=_get(data, w, "build_timeout_s", int, None, _positive, ">= 1"),
            warm_start=warm,
            confirm=ConfirmSettings(
                finalists=_get(cd, cw, "finalists", int, 2, _positive, ">= 1"),
                fresh_calls=_get(cd, cw, "fresh_calls", int, 3, _positive, ">= 1"),
            ),
            measurement_policy=_get(
                data, w, "measurement_policy", str, "full",
                lambda v: v in MEASUREMENT_POLICIES, f"one of {MEASUREMENT_POLICIES}",
            ),
            host_cache=HostCacheSettings(
                dir=_get_file(hd, hw, "dir", base, "directory"),
                max_entry_gib=_get(hd, hw, "max_entry_gib", float, 4.0,
                                   lambda v: v > 0, "> 0"),
                max_total_gib=_get(hd, hw, "max_total_gib", float, None,
                                   lambda v: v >= 0, ">= 0"),
            ),
        )

    def to_json(self) -> dict:
        """The settings as accepted by ``from_json`` (paths resolved)."""
        d = {f: getattr(self, f) for f in self.__dataclass_fields__}
        d["confirm"] = vars(self.confirm).copy()
        d["host_cache"] = {k: v for k, v in vars(self.host_cache).items()
                           if v is not None}
        for name in ("deadline_s", "build_timeout_s"):
            if d[name] is None:
                del d[name]
        if self.warm_start is None:
            del d["warm_start"]
        else:
            d["warm_start"] = {
                k: v for k, v in vars(self.warm_start).items() if v is not None
            }
        return d

    def problem_seed(self, index: int) -> int:
        return self.seed + (index if self.seed_policy == "per_problem" else 0)


# ---------------------------------------------------------- budgets, artifacts


class BudgetLedger:
    """Trial attempts per problem, new builds per run, confirmation calls and
    the deadline. A trial is charged when admitted and refunded only when it
    never ran (unattempted, cancelled); a build is charged once per distinct
    kernel attempted, failures included."""

    def __init__(self, trials_per_problem, new_builds_per_run, deadline_s, clock):
        self.trials_per_problem = int(trials_per_problem)
        self.new_builds_per_run = int(new_builds_per_run)
        self.deadline_s = None if deadline_s is None else float(deadline_s)
        self.clock = clock
        self.start = clock()
        self.trials: Dict[str, int] = {}
        self.builds = 0
        self.confirm_calls = 0

    def trials_left(self, problem: str) -> int:
        return self.trials_per_problem - self.trials.get(problem, 0)

    def admit(self, problem: str) -> None:
        if self.trials_left(problem) <= 0:
            raise RuntimeError(f"trial budget of {problem!r} overspent")
        self.trials[problem] = self.trials.get(problem, 0) + 1

    def refund(self, problem: str) -> None:
        self.trials[problem] -= 1

    def builds_left(self) -> int:
        return self.new_builds_per_run - self.builds

    def charge_builds(self, n: int) -> None:
        if n > self.builds_left():
            raise RuntimeError(f"{n} builds overspend the cap ({self.builds_left()} left)")
        self.builds += n

    def elapsed(self) -> float:
        return self.clock() - self.start

    def deadline_passed(self) -> bool:
        return self.deadline_s is not None and self.elapsed() >= self.deadline_s

    def to_json(self) -> dict:
        return {
            "trials_per_problem": self.trials_per_problem,
            "trials_charged": dict(self.trials),
            "new_builds_per_run": self.new_builds_per_run,
            "builds_charged": self.builds,
            "confirm_calls": self.confirm_calls,
            "deadline_s": self.deadline_s,
        }


class ArtifactRegistry:
    """canonical_key -> BuildResult of this run. Entries are never replaced: a
    built kernel serves every problem, a failed build is not retried."""

    def __init__(self):
        self._results: Dict[str, BuildResult] = {}

    def get(self, key: str) -> Optional[BuildResult]:
        return self._results.get(key)

    def missing(self, cands: Sequence[Candidate]) -> List[Candidate]:
        """The candidates without a result, deduplicated, in first-seen order."""
        out: Dict[str, Candidate] = {}
        for c in cands:
            if c.canonical_key not in self._results:
                out.setdefault(c.canonical_key, c)
        return list(out.values())

    def publish(self, results: Mapping[str, BuildResult]) -> None:
        for key, res in results.items():
            if key in self._results:
                raise RuntimeError(f"artifact {key!r} is already published")
            self._results[key] = res

    def to_json(self) -> dict:
        return {k: vars(r).copy() for k, r in self._results.items()}


# ------------------------------------------------------------------ outputs


def _jsonable(x):
    """``x`` as strict-JSON data: non-finite floats become null."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    return str(x)


def _dumps(obj, indent=None) -> str:
    return json.dumps(_jsonable(obj), allow_nan=False, indent=indent)


class RunWriter:
    """The run's output directory: trials.jsonl is appended and flushed event by
    event; the JSON files are written atomically (temp file + rename). The
    first write that fails raises its OSError and is kept in ``failed``; every
    later write is skipped, so the error path cannot fail the same way again,
    except a ``final`` JSON file (the run's last word, e.g. summary.json): it
    is still tried, and its failure is kept, never raised."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self._fh = None
        self._seq = 0
        self.failed = ""

    def check_fresh(self) -> None:
        # The directory, or else its nearest existing parent, must be a
        # directory the run can write in.
        near = next((p for p in (self.out_dir, *self.out_dir.parents) if p.exists()),
                    None)
        if near is not None and not near.is_dir():
            raise SettingsError(
                f"search output dir {self.out_dir}: {near} is not a directory"
            )
        if near is not None and not os.access(near, os.W_OK | os.X_OK):
            raise SettingsError(
                f"search output dir {self.out_dir}: {near} is not writable"
            )
        old = [n for n in OUTPUT_FILES if (self.out_dir / n).exists()]
        if old:
            raise SettingsError(
                f"search output dir {self.out_dir} already holds a run ({old}); "
                "use a new directory"
            )

    def _guard(self, fn, *args, final=False):
        if self.failed and not final:
            return None
        try:
            return fn(*args)
        except OSError as e:
            self.failed = self.failed or f"cannot write {self.out_dir}: {e}"
            if not final:
                raise
            return None

    def event(self, kind: str, **fields) -> None:
        self._guard(self._event, kind, fields)

    def _event(self, kind, fields) -> None:
        if self._fh is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.out_dir / "trials.jsonl", "a")
        self._seq += 1
        self._fh.write(_dumps({"seq": self._seq, "event": kind, **fields}) + "\n")
        self._fh.flush()

    def write_json(self, name: str, data, final: bool = False) -> Optional[Path]:
        return self._guard(self._write_json, name, data, final=final)

    def _write_json(self, name, data) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        tmp = path.with_name(f".{name}.tmp")
        with open(tmp, "w") as f:
            f.write(_dumps(data, indent=1) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    def close(self) -> None:
        if self._fh is not None:
            fh, self._fh = self._fh, None
            try:
                fh.close()
            except OSError:
                pass


# -------------------------------------------------------------- coordinator


@dataclass(frozen=True)
class RunSummary:
    status: str  # complete | partial | no_winner
    # 0 only when complete (every problem has a winner, no error); 130 when
    # interrupted
    exit_code: int
    stop_reason: str
    winners: Dict[str, str]
    unresolved: Dict[str, str]
    counts: Dict[str, Any]
    wall_s: float
    error: str = ""


def valid_latency(v) -> bool:
    """A measured latency: a finite positive number (a bool is not one)."""
    return (
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(v) and v > 0
    )


def _checked(outcome: PairOutcome):
    """(status, latency_ms, diag): the single acceptance rule. An ``ok`` without
    a finite positive latency is an invalid_measurement, never evidence."""
    status, lat, diag = outcome.status, outcome.latency_ms, outcome.diag
    if status != "ok":
        return status, None, diag
    if not valid_latency(lat):
        return "invalid_measurement", None, f"ok with latency {lat!r} {diag}".strip()
    return status, float(lat), diag


def _join(error: str, more: str) -> str:
    return f"{error}; {more}" if error else more


def _load_warm(path, loader, what: str, loader_name: str):
    """``loader(path)`` for the warm-start file ``what``; no loader, or a file
    the loader cannot read, is a SettingsError."""
    if loader is None:
        raise SettingsError(f"warm_start.{what} needs a {loader_name}")
    try:
        return loader(path)
    except (OSError, ValueError) as e:  # SettingsError and UnicodeDecodeError too
        raise SettingsError(f"warm_start.{what}: {e}") from None


class _Problem:
    """Per-problem search state."""

    def __init__(self, key: str, index: int, cands: Sequence[Candidate]):
        self.key = key
        self.index = index
        self.cands = list(cands)
        self.by_key = {c.canonical_key: c for c in self.cands}
        self.space = None
        self.members: Dict[str, int] = {}
        self.strategy = None
        self.strategy_name = ""
        self.seed = 0
        self.warm: List[str] = []
        self.prior_entries = 0
        self.requeue: list = []  # trials to re-admit (unattempted once)
        self.retried: set = set()  # uids already re-queued once
        self.deferred: list = []  # same-round duplicates of an in-flight key
        self.seen: dict = {}  # space key -> Observation told for it
        self.best: Dict[str, tuple] = {}  # canonical_key -> (latency_ms, trial uid)
        self.asked_empty = False
        self.finished = ""  # why the problem stopped asking ("" = active)

    def cand(self, key: str) -> Candidate:
        """The candidate a space key names (``members`` maps it to its row)."""
        return self.cands[self.members[key]]


StrategyFactory = Callable[[str, Any, int, Sequence[Mapping], Any], Any]


def _default_factory(name, space, seed, warm_records, prior):
    return _ck_search().make_strategy(
        name, space, seed, warm_records=warm_records, prior=prior
    )


class SearchCoordinator:
    """Runs the search over every problem of ``adapter`` and writes ``out_dir``.

    ``strategy_factory(name, space, seed, warm_records, prior)`` defaults to
    ``ck_search.make_strategy``. The op reads its warm-start files:
    ``pair_loader(path) -> {problem: [names] | "*"}`` reads
    ``warm_start.pairs_file`` (the GEMM driver passes its ``load_pair_list``),
    ``prior_loader(path) -> {problem: {kernel name: predicted latency ms}}``
    reads ``warm_start.ranked_csv``. ``log``, if given, gets one progress line
    per round and per build (the driver passes print). Invalid options raise
    ``SettingsError`` here, before any work.

    Rounds: each active problem in turn (the first one rotates) re-admits its
    re-queued trials, then asks its strategy for at most ``batch_per_problem``
    trials in total, capped by its trials left and by the new-build capacity
    left once the distinct new builds this round already needs are subtracted,
    plus its untried candidates that need no build (built or planned already).
    Two kinds of proposal are told at once, without a trial: a kernel whose
    build failed already (for any problem) gets that ``compile_error`` as a
    cached result, and one that needs a build once this round's builds spend
    the cap is ``invalid`` (it cannot be built in this run), so a round never
    plans more builds than the cap allows and no strategy waits for such a
    proposal. Its slot goes to the problem's best untried candidate that needs
    no build (``_refill``), told as a trial the strategy did not ask: past the
    cap a strategy reaches a built kernel only by proposing it, and most of
    its proposals then need a build, so without the refill the stall guard
    would end problems with trials and built kernels left. Every trial is
    told before the strategy is asked again; a problem stops at its trial
    budget, or when its strategy is done or asks nothing while waiting for no
    result (or only for trials the adapter cancelled, which are never told).
    The run stops when no problem is
    active, at the deadline (checked before every ask), at the build cap once
    no active problem has an untried candidate that is built or planned, or
    after ``max_stall_rounds`` rounds in a row without a new result. Any error
    ends the search; the finalists measured so far are still confirmed, unless
    the adapter broke its contract.
    """

    def __init__(
        self,
        adapter: OperationAdapter,
        settings,
        out_dir,
        *,
        clock: Callable[[], float] = time.monotonic,
        strategy_factory: Optional[StrategyFactory] = None,
        pair_loader: Optional[Callable[[str], Mapping]] = None,
        prior_loader: Optional[Callable[[str], Mapping]] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(settings, SearchSettings):
            settings = SearchSettings.from_json(settings)
        self.adapter = adapter
        self.settings = settings
        self.clock = clock
        self.log = log
        self.strategy_factory = strategy_factory or _default_factory
        self.writer = RunWriter(out_dir)
        self.writer.check_fresh()
        warm = settings.warm_start
        self.warm_pairs: Mapping = {}
        self.ranked: Mapping = {}
        if warm is not None and warm.pairs_file is not None:
            self.warm_pairs = _load_warm(warm.pairs_file, pair_loader, "pairs_file",
                                         "pair_loader")
        if warm is not None and warm.use_prior:
            self.ranked = _load_warm(warm.ranked_csv, prior_loader, "ranked_csv",
                                     "prior_loader")
        self.ledger = None
        self.registry = ArtifactRegistry()
        self.problems: List[_Problem] = []
        self.counts: Dict[str, Any] = dict.fromkeys(
            ("rounds", "proposed", "admitted", "cached", "invalid", "known_failed",
             "past_build_cap", "refilled", "built", "build_failed", "measured", "ok",
             "requeued", "cancelled", "confirm_calls", "host_memory"), 0
        )
        self.counts["failed_by_status"] = {}
        self.unresolved: Dict[str, str] = {}
        self._calls = 0
        self._round = 0
        self._ck = None
        self._run_json = None
        self.noise = None  # the timing noise model shared by every strategy

    # ------------------------------------------------------------ public

    def run(self) -> RunSummary:
        if self.ledger is not None:
            raise RuntimeError("SearchCoordinator.run() runs once; make a new one")
        self.ledger = BudgetLedger(
            self.settings.trials_per_problem, self.settings.new_builds_per_run,
            self.settings.deadline_s, self.clock,
        )
        self._ck = _ck_search()
        self.noise = self._ck.NoiseModel()
        stop, error, winners, broken = "error", "", {}, False
        try:
            try:
                self._setup()
                stop = self._search() if self.problems else "no_problems"
            except KeyboardInterrupt:  # SIGINT, or SIGTERM through run_search
                stop, error = "interrupted", self._interrupted("search")
            except Exception as e:  # noqa: BLE001 - recorded, the run exits 1
                error, broken = self._error(e), isinstance(e, ContractError)
            for p in self.problems:  # re-queued trials that never ran again
                for t in p.requeue:
                    self.counts["cancelled"] += 1
                    self._note("cancel", problem=p.key, trial_uid=t.uid,
                               diag="search stopped before re-admission")
            # What was measured before an error is still confirmed, unless the
            # adapter broke its contract, the run was interrupted or its output
            # dir failed (nothing could be exported).
            if not (broken or stop == "interrupted" or self.writer.failed):
                try:
                    winners = self._confirm(stop)
                except KeyboardInterrupt:
                    stop, winners = "interrupted", {}
                    error = _join(error, self._interrupted("confirmation"))
                except Exception as e:  # noqa: BLE001 - recorded, the run exits 1
                    why = "confirmation: " + self._error(e)
                    error = _join(error, why)
                    for p in self.problems:
                        self.unresolved.setdefault(p.key, why)
            return self._finish(winners, stop, error)
        finally:
            self.writer.close()

    def _note(self, kind: str, **fields) -> None:
        """``_event`` on the stop and error paths: a failed write is already
        kept in ``writer.failed``, so it does not raise again here."""
        try:
            self._event(kind, **fields)
        except OSError:
            pass

    def _error(self, e: Exception) -> str:
        """Record ``e`` (an error event with its traceback); its message."""
        msg = str(e) if isinstance(e, AdapterError) else f"{type(e).__name__}: {e}"
        self._note("error", message=msg, traceback=traceback.format_exc())
        return msg

    def _interrupted(self, during: str) -> str:
        msg = f"interrupted during the {during}: nothing confirmed"
        self._note("error", message=msg, traceback=traceback.format_exc())
        return msg

    # ------------------------------------------------------------- setup

    def _call(self, what: str, fn, *args):
        try:
            return fn(*args)
        except AdapterError:
            raise
        except Exception as e:  # noqa: BLE001 - an adapter failure ends the run
            raise AdapterError(f"{what} failed: {type(e).__name__}: {e}") from e

    def _setup(self) -> None:
        s = self.settings
        context = self._call("context", self.adapter.context)
        eligible = self._call("enumerate_eligible", self.adapter.enumerate_eligible)
        if not isinstance(eligible, Mapping):
            raise ContractError(f"enumerate_eligible returned {type(eligible).__name__}")
        names: Dict[str, str] = {}
        for i, (pkey, cands) in enumerate(eligible.items()):
            cands = list(cands)
            for c in cands:
                if not isinstance(c, Candidate):
                    raise ContractError(f"{pkey}: candidate {c!r} is not a Candidate")
                if names.setdefault(c.canonical_key, c.kernel_name) != c.kernel_name:
                    raise ContractError(
                        f"candidate key {c.canonical_key!r} names two kernels")
            p = _Problem(str(pkey), i, cands)
            if len(p.by_key) != len(p.cands):
                raise ContractError(f"{pkey}: a candidate key repeats")
            self.problems.append(p)
            if not p.cands:
                p.finished = "no eligible candidates"
                continue
            try:
                p.space = self._ck.SearchSpace.from_configs(
                    [dict(c.record) for c in p.cands], name=p.key
                )
            except ValueError as e:
                raise ContractError(f"{p.key}: candidate records: {e}") from None
            p.members = p.space.members
            p.seed = s.problem_seed(i)
            p.strategy_name = s.strategy
            warm, prior = (), None
            if len(p.cands) < s.trials_per_problem:
                p.strategy_name = "exhaustive"
            else:
                warm = self._warm_records(p)
                prior = self._prior(p)
            p.strategy = self.strategy_factory(p.strategy_name, p.space, p.seed, warm, prior)
            # As upstream's run loop does: a gain counts only beyond timing noise.
            attach = getattr(p.strategy, "attach_noise", None)
            if attach is not None:
                attach(self.noise)
        self._run_json = {
            "schema_version": SCHEMA_VERSION,
            "sampling_policy": "adaptive",
            "exhaustive": False,
            "settings": s.to_json(),
            "context": context,
            "ck_search": {
                "origin_sha": getattr(self._ck, "ORIGIN_SHA", ""),
                "strategies": list(getattr(self._ck, "STRATEGY_NAMES", ())),
                "noise_model": self.noise.to_json(),
            },
            "problems": {
                p.key: {
                    "index": p.index,
                    "eligible": len(p.cands),
                    "params": 0 if p.space is None else len(p.space.params),
                    "strategy": p.strategy_name or None,
                    "seed": p.seed,
                    "warm_start": p.warm,
                    "prior_entries": p.prior_entries,
                    "note": p.finished or (
                        "exhaustive: fewer eligible candidates than trials_per_problem"
                        if p.strategy_name != s.strategy else ""
                    ),
                }
                for p in self.problems
            },
            "domain": [
                "candidates of a problem are exactly the adapter's eligible list; "
                "the search never proposes a config outside it",
                "problems with fewer eligible candidates than trials_per_problem "
                "are searched exhaustively (no warm start, no prior)",
                f"measurement_policy {s.measurement_policy}: every discovery and "
                "confirmation call is a full verified measurement",
                "new builds are capped per run; a failed build is charged once",
            ],
        }
        self.writer.write_json("run.json", self._run_json)

    def _warm_records(self, p: _Problem):
        """The first ``top_k`` listed names that are eligible, in listed order."""
        names = self.warm_pairs.get(p.key, ())
        if names == "*":
            names = [c.kernel_name for c in p.cands]
        by_name = {c.kernel_name: c for c in p.cands}
        picks = [by_name[n] for n in names if n in by_name]
        if picks:
            picks = picks[: self.settings.warm_start.top_k]
        p.warm = [c.kernel_name for c in picks]
        return tuple(dict(c.record) for c in picks)

    def _prior(self, p: _Problem):
        pred = self.ranked.get(p.key, {})
        table = {
            k: pred[p.cands[i].kernel_name]
            for k, i in p.members.items() if p.cands[i].kernel_name in pred
        }
        p.prior_entries = len(table)
        if not table:
            return None
        space = p.space  # strategies predict decoded configs: key them like members
        return self._ck.TablePrior(
            table, lambda cfg: space.key(space.encode(cfg)), higher_is_better=False,
            scale="log",
        )

    # ------------------------------------------------------------ rounds

    def _event(self, kind: str, **fields) -> None:
        elapsed = self.ledger.elapsed() if self.ledger is not None else 0.0
        self.writer.event(kind, round=self._round, t=round(elapsed, 6), **fields)

    def _call_id(self, prefix: str) -> str:
        self._calls += 1
        return f"{prefix}{self._calls}"

    def _needs_build(self, c: Candidate, planned) -> bool:
        """Admitting ``c`` costs a new build: no result yet, none planned."""
        return c.canonical_key not in planned and self.registry.get(c.canonical_key) is None

    def _build_free_rows(self, p: _Problem, planned=(), inflight=()) -> List[int]:
        """``p``'s candidates (rows) without a result, nor in flight this round,
        that are built, or planned this round, already: work that needs no new
        build. A failed build is not work: proposing it is answered from its
        result."""
        def ready(c):
            res = self.registry.get(c.canonical_key)
            return c.canonical_key in planned or (res is not None and res.path is not None)

        return [i for k, i in p.members.items()
                if k not in p.seen and (p.key, k) not in inflight and ready(p.cands[i])]

    def _build_free(self, p: _Problem, planned=()) -> int:
        return len(self._build_free_rows(p, planned))

    def _refill(self, p: _Problem, n: int, planned, inflight) -> List[Candidate]:
        """At most ``n`` of ``p``'s build-free candidates (``_build_free_rows``),
        best first: by the lowest latency relative to the best of another
        problem that measured it; unmeasured ones last, in candidate order.
        Past the build cap a strategy reaches a built kernel only by proposing
        it, so each of its proposals told past the cap leaves its slot to one
        of these, and a cap-bound run spends its trial budgets on the kernels
        it built instead of stalling."""
        if n <= 0:
            return []
        floor = {q.key: min(v[0] for v in q.best.values())
                 for q in self.problems if q is not p and q.best}

        def rank(i):
            ck = p.cands[i].canonical_key
            rel = [q.best[ck][0] / floor[q.key] for q in self.problems
                   if q.key in floor and ck in q.best]
            return (min(rel, default=math.inf), i)

        rows = sorted(self._build_free_rows(p, planned, inflight), key=rank)
        return [p.cands[i] for i in rows[:n]]

    def _search(self) -> str:
        stall = 0
        while True:
            active = [p for p in self.problems if not p.finished]
            if not active:
                return "searched"
            if stall >= self.settings.max_stall_rounds:
                return "stalled"
            if self.ledger.deadline_passed():
                return "deadline"
            if self.ledger.builds_left() <= 0 and not any(
                    self._build_free(p) for p in active):
                return "build_cap"
            self._round += 1
            self.counts["rounds"] += 1
            before = {k: v for k, v in self.counts.items() if isinstance(v, int)}
            stall = 0 if self._one_round(active) else stall + 1
            self._log_round(before)

    def _progress(self, line: str) -> None:
        if self.log is not None:
            took = f"{self.ledger.elapsed():.0f}"
            if self.ledger.deadline_s is not None:
                took += f"/{self.ledger.deadline_s:.0f}"
            self.log(f"  [search] round {self._round}: {line}; {took} s")

    def _log_round(self, before: Mapping[str, int]) -> None:
        """One line: what the round did (counts since ``before``)."""
        d = {k: self.counts[k] - v for k, v in before.items()}
        self._progress(
            f"{d['proposed']} proposed, {d['admitted']} admitted, {d['cached']} cached, "
            f"{d['known_failed']} known failed, {d['past_build_cap']} past the build "
            f"cap, {d['refilled']} refilled; {d['built']} built, {d['build_failed']} "
            f"failed ({self.ledger.builds}/"
            f"{self.ledger.new_builds_per_run} builds); {d['measured']} measured, "
            f"{d['ok']} ok; {sum(not p.finished for p in self.problems)}/"
            f"{len(self.problems)} problems active")

    def _one_round(self, active: List[_Problem]) -> bool:
        """Ask, admit, build, measure, tell. True when a trial got a result."""
        admitted = []  # (problem, trial, space key, candidate)
        planned = set()  # canonical keys this round needs built
        inflight = set()  # (problem, space key) admitted this round
        told: Dict[str, list] = {p.key: [] for p in active}
        batch = self.settings.batch_per_problem

        def answer(p, t, key, admission, status, detail, **kw):
            """Tell ``t`` at once, without a trial; ``key`` keeps the answer."""
            ob = self._ck.Observation(trial=t, key=key, status=status, detail=detail, **kw)
            told[p.key].append(ob)
            if key is not None:
                p.seen[key] = ob
            self.counts[admission] += 1
            kernel = {} if key is None else {"kernel": p.cand(key).kernel_name}
            self._event("trial", admission=admission, problem=p.key, trial_uid=t.uid,
                        origin=t.origin, **kernel)

        def place(p, t, kind):
            """Admit ``t`` (charged), or answer it without spending: a config the
            space refuses, one this problem already has a result for (cached), a
            kernel whose build failed already (its compile_error, cached), or one
            that needs a build once this round's builds spend the cap (invalid:
            it cannot be built in this run). A second trial of a key in flight
            waits for that result (deferred)."""
            key = p.space.key(t.genome)
            ev = dict(problem=p.key, trial_uid=t.uid, origin=t.origin)
            if key is None:
                answer(p, t, None, "invalid", self._ck.Status.INVALID,
                       "refused by the space")
                return
            c = p.cand(key)
            res = self.registry.get(c.canonical_key)
            if key in p.seen:
                self.counts["cached"] += 1
                told[p.key].append(p.seen[key].replace_trial(t, cached=True))
                self._event("trial", kernel=c.kernel_name, admission="cached", **ev)
            elif (p.key, key) in inflight:
                p.deferred.append(t)
            elif res is not None and res.path is None:
                # Known like upstream's cache of structural failures: no trial.
                answer(p, t, key, "known_failed", self._ck.Status.COMPILE_ERROR,
                       res.reason or "build failed", cached=True)
            elif self._needs_build(c, planned) and len(planned) >= self.ledger.builds_left():
                # This round's builds spend the cap: it can never be built in
                # this run. Told now, so no strategy waits for it.
                answer(p, t, key, "past_build_cap", self._ck.Status.INVALID,
                       "needs a build past new_builds_per_run")
            else:
                self.ledger.admit(p.key)
                self.counts["admitted"] += 1
                inflight.add((p.key, key))
                admitted.append((p, t, key, c))
                if self.registry.get(c.canonical_key) is None:
                    planned.add(c.canonical_key)
                self._event("trial", kernel=c.kernel_name, admission=kind, **ev)

        start = (self._round - 1) % len(active)
        for p in active[start:] + active[:start]:
            if self.ledger.deadline_passed():
                break  # the deadline stops admission; admitted work is drained
            p.asked_empty = False
            room = batch
            while p.requeue and room > 0 and self.ledger.trials_left(p.key) > 0:
                place(p, p.requeue.pop(0), "readmitted")
                room -= 1
            # A trial is capped by the build capacity left only when it needs a
            # build: candidates built (or planned) already cost none.
            n = min(room, self.ledger.trials_left(p.key))
            capacity = self.ledger.builds_left() - len(planned)
            if capacity < n:
                n = min(n, capacity + self._build_free(p, planned))
            if n <= 0:
                continue
            trials = list(p.strategy.ask(n))
            if len(trials) > n:
                raise RuntimeError(f"{p.key}: strategy asked {len(trials)} > {n} trials")
            self.counts["proposed"] += len(trials)
            # An empty ask means done only when every trial was told before it;
            # a trial re-admitted just above is still pending in the strategy.
            p.asked_empty = not trials and room == batch
            past = self.counts["past_build_cap"]
            for t in trials:
                place(p, t, "admitted")
            # Its result is told to the strategy as a trial it did not ask.
            for c in self._refill(p, self.counts["past_build_cap"] - past, planned,
                                  inflight):
                self.counts["refilled"] += 1
                uid = f"refill:{self.counts['refilled']}"
                place(p, self._ck.Trial(genome=p.space.encode(dict(c.record)), uid=uid,
                                        origin="refill"), "refilled")

        progress = False
        missing = self.registry.missing([c for _, _, _, c in admitted])
        if missing:
            self._build(missing, admitted)
        requests, waiting = [], {}
        for p, t, key, c in admitted:
            res = self.registry.get(c.canonical_key)
            if res.path is None:
                progress |= self._settle(p, t, key, c, told, "compile_error", None,
                                         res.reason or "build failed", None)
                continue
            req = PairRequest(p.key, c.canonical_key, c.kernel_name, t.uid,
                              self._call_id("d"))
            requests.append(req)
            waiting[req.call_id] = (p, t, key, c)
        if requests:
            self.counts["measured"] += len(requests)
            outcomes = self._exchange("measure_pairs", self.adapter.measure_pairs, requests)
            for req in requests:
                status, lat, diag = _checked(outcomes[req.call_id])
                p, t, key, c = waiting[req.call_id]
                progress |= self._settle(p, t, key, c, told, status, lat, diag, req.call_id,
                                         outcomes[req.call_id].device)
        for p in active:
            for t in p.deferred:  # resolved by the in-flight trial of the same key
                key = p.space.key(t.genome)
                if key in p.seen:
                    self.counts["cached"] += 1
                    told[p.key].append(p.seen[key].replace_trial(t, cached=True))
                else:
                    p.requeue.append(t)
            p.deferred = []
            if told[p.key]:
                p.strategy.tell(told[p.key])
            if not p.requeue and not p.finished:
                if self.ledger.trials_left(p.key) <= 0:
                    p.finished = "trial budget"
                elif p.strategy.done:
                    p.finished = "strategy done"
                elif p.asked_empty:
                    # Every trial is told by now except those the adapter
                    # cancelled, which never are: a strategy waiting for one
                    # asks nothing more.
                    waiting = len(getattr(p.strategy, "pending", None) or ())
                    p.finished = (f"strategy waits for {waiting} cancelled trial(s)"
                                  if waiting else "strategy done")
        return progress

    def _build(self, missing: List[Candidate], admitted) -> None:
        self._progress(f"building {len(missing)} kernel(s) "
                       f"({self.ledger.builds}/{self.ledger.new_builds_per_run} builds)")
        results = self._call("build_missing", self.adapter.build_missing, missing)
        want = {c.canonical_key for c in missing}
        if not isinstance(results, Mapping) or set(results) != want or not all(
            isinstance(r, BuildResult) for r in results.values()
        ):
            raise ContractError(
                f"build_missing must return one BuildResult per requested key "
                f"{sorted(want)}, got {results!r}"
            )
        self.ledger.charge_builds(len(missing))
        self.registry.publish({c.canonical_key: results[c.canonical_key] for c in missing})
        for c in missing:
            r = results[c.canonical_key]
            self.counts["built" if r.path is not None else "build_failed"] += 1
            self._event(
                "build", kernel=c.kernel_name, candidate_key=c.canonical_key,
                ok=r.path is not None, path=r.path, reason=r.reason, digest=r.digest,
                waiting=sorted({p.key for p, _, _, a in admitted
                                if a.canonical_key == c.canonical_key}),
            )

    def _exchange(self, what: str, fn, requests: List[PairRequest]) -> dict:
        """``fn(requests)`` checked to return exactly one outcome per request."""
        outcomes = self._call(what, fn, list(requests))
        expected = {r.call_id: r for r in requests}
        got: Dict[str, PairOutcome] = {}
        for o in outcomes if isinstance(outcomes, (list, tuple)) else [outcomes]:
            if not isinstance(o, PairOutcome) or o.status not in STATUSES:
                raise ContractError(f"{what} returned {o!r}, expected a PairOutcome")
            cid = o.request.call_id
            if expected.get(cid) != o.request or cid in got:
                raise ContractError(f"{what} returned an unrequested or repeated {o!r}")
            got[cid] = o
        lost = [r for cid, r in expected.items() if cid not in got]
        if lost:
            raise ContractError(f"{what} returned no outcome for {len(lost)} request(s), "
                               f"first {lost[0]!r}")
        return got

    def _settle(self, p, t, key, c, told, status, lat, diag, call_id, device="") -> bool:
        """Turn one outcome into a told observation or a scheduler event."""
        base = dict(problem=p.key, trial_uid=t.uid, kernel=c.kernel_name,
                    call_id=call_id, diag=diag, device=device)
        if status == "unattempted" and t.uid not in p.retried:
            p.retried.add(t.uid)
            p.requeue.append(t)
            self.ledger.refund(p.key)
            self.counts["requeued"] += 1
            self._event("requeue", **base)
            return False
        if status == "cancelled":
            self.ledger.refund(p.key)
            self.counts["cancelled"] += 1
            self._event("cancel", **base)
            return False
        if status == HOST_MEMORY:  # not the kernel's: told nothing, the problem stops
            self.ledger.refund(p.key)
            if not p.finished:
                p.finished = diag or HOST_MEMORY
                self.counts["host_memory"] += 1
            self._event("host_memory", **base)
            return False
        if status == "unattempted":
            status, diag = "run_error", f"unattempted twice: {diag}".strip()
        rocke = getattr(self._ck.Status, TELLABLE[status])
        if status == "ok":
            ob = self._ck.Observation(trial=t, key=key, status=rocke, value=lat,
                                      samples=(lat,), detail=diag)
            self.counts["ok"] += 1
            if c.canonical_key not in p.best or lat < p.best[c.canonical_key][0]:
                p.best[c.canonical_key] = (lat, t.uid)
        else:
            detail = diag if status == rocke else f"{status}: {diag}"
            ob = self._ck.Observation(trial=t, key=key, status=rocke, detail=detail)
            fails = self.counts["failed_by_status"]
            fails[status] = fails.get(status, 0) + 1
        told[p.key].append(ob)
        p.seen[key] = ob
        self._event("result", **{**base, "status": status, "told": rocke,
                                 "latency_ms": lat})
        return True

    # ------------------------------------------------------- confirmation

    def _status(self, winners, stop: str, error: str):
        """(status, exit code, error); the error includes an output failure."""
        if self.writer.failed and self.writer.failed not in error:
            error = _join(error, self.writer.failed)
        # A run that hit an error is never complete, even when its finalists
        # were confirmed: the search did not run to its stop.
        if self.problems and len(winners) == len(self.problems) and not error:
            status = "complete"
        else:
            status = "partial" if winners else "no_winner"
        exit_code = (EXIT_INTERRUPTED if stop == "interrupted"
                     else EXIT_OK if status == "complete" else EXIT_FAIL)
        return status, exit_code, error

    def _write_outputs(self, winners, stop, error, wall) -> None:
        """The final files, each tried even after a failed write (see
        ``RunWriter``). The status each one records counts the failed writes
        before it, so outputs that are incomplete never say ``complete``."""
        out = self.writer
        if self._run_json is None:  # setup failed: still say what was asked
            out.write_json("run.json", {
                "schema_version": SCHEMA_VERSION, "sampling_policy": "adaptive",
                "exhaustive": False, "settings": self.settings.to_json(),
                "error": self._status(winners, stop, error)[2],
            }, final=True)
        out.write_json("winners.json", {k: [v["kernel"]] for k, v in winners.items()},
                       final=True)
        self._note("stop", stop_reason=stop, status=self._status(winners, stop, error)[0])
        status, exit_code, error = self._status(winners, stop, error)
        problems = {}
        for p in self.problems:
            best = min(p.best.items(), key=lambda kv: (kv[1][0], kv[0]), default=None)
            problems[p.key] = {
                "strategy": p.strategy_name or None,
                "trials": self.ledger.trials.get(p.key, 0),
                "ok": len(p.best), "finished": p.finished or stop,
                # Unconfirmed: what discovery measured, e.g. before an interrupt.
                "discovery_best": None if best is None else {
                    "kernel": p.by_key[best[0]].kernel_name, "latency_ms": best[1][0]},
            }
        out.write_json("summary.json", {
            "status": status,
            "exit_code": exit_code,
            "stop_reason": stop,
            "error": error,
            "winners": winners,
            "unresolved": self.unresolved,
            "counts": self.counts,
            "budget": self.ledger.to_json(),
            "artifacts": self.registry.to_json(),
            "problems": problems,
            "wall_s": wall,
        }, final=True)

    def _confirm(self, stop: str) -> Dict[str, dict]:
        """Re-measure each problem's finalists with ``fresh_calls`` fresh verified
        calls. Call j of every surviving finalist goes into one adapter call, so
        competing kernels interleave (on one device per problem, see
        ``OperationAdapter.confirm``); a finalist is dropped at its first failed
        call. The winner is the lowest median among the finalists whose calls all
        passed: there is no fallback to unconfirmed discovery results."""
        cf = self.settings.confirm
        fins = {
            p: sorted(p.best.items(), key=lambda kv: (kv[1][0], kv[0]))[: cf.finalists]
            for p in self.problems
        }
        calls = {(p.key, ck): [] for p, best in fins.items() for ck, _ in best}
        if self.log is not None and calls:
            self.log(f"  [search] confirming {len(calls)} finalist(s) of "
                     f"{sum(1 for b in fins.values() if b)} problem(s), "
                     f"{cf.fresh_calls} fresh call(s) each")
        for j in range(cf.fresh_calls):
            reqs = [
                PairRequest(p.key, ck, p.by_key[ck].kernel_name, uid, self._call_id("c"))
                for p, best in fins.items() for ck, (_lat, uid) in best
                if all(st == "ok" for st, _ in calls[(p.key, ck)])
            ]
            if not reqs:
                break
            self.counts["confirm_calls"] += len(reqs)
            self.ledger.confirm_calls += len(reqs)
            got = self._exchange("confirm", self.adapter.confirm, reqs)
            for r in reqs:
                status, lat, diag = _checked(got[r.call_id])
                calls[(r.problem_key, r.candidate_key)].append((status, lat))
                self._event("confirm", problem=r.problem_key, kernel=r.kernel_name,
                            trial_uid=r.trial_uid, call_id=r.call_id, call=j,
                            status=status, latency_ms=lat, diag=diag,
                            device=got[r.call_id].device)
        winners = {}
        for p, best in fins.items():
            passed = []
            for ck, (lat, _uid) in best:
                res = calls[(p.key, ck)]
                if len(res) == cf.fresh_calls and all(st == "ok" for st, _ in res):
                    vals = [v for _, v in res]
                    name = p.by_key[ck].kernel_name
                    passed.append((statistics.median(vals), name, vals, lat))
            if passed:
                med, name, vals, lat = min(passed)
                winners[p.key] = {"kernel": name, "median_ms": med, "calls_ms": vals,
                                  "discovery_ms": lat}
                self._event("final", problem=p.key, kernel=name, median_ms=med)
                continue
            if best:
                why = "no finalist passed confirmation (" + "; ".join(
                    p.by_key[ck].kernel_name + ": "
                    + ",".join(st for st, _ in calls[(p.key, ck)]) for ck, _ in best
                ) + ")"
            elif not p.cands:
                why = p.finished
            else:
                why = f"no verified discovery measurement (search: {p.finished or stop})"
            self.unresolved[p.key] = why
            self._event("final", problem=p.key, unresolved=why)
        return winners

    # ------------------------------------------------------------ finish

    def _finish(self, winners: Dict[str, dict], stop: str, error: str) -> RunSummary:
        for p in self.problems:
            if p.key not in winners and p.key not in self.unresolved:
                self.unresolved[p.key] = f"search ended: {stop}"
        wall = self.ledger.elapsed()
        self._write_outputs(winners, stop, error, wall)
        # A failed write, the last one included, is an error: never complete.
        status, exit_code, error = self._status(winners, stop, error)
        if self.writer.failed:
            print(f"adaptive search: {self.writer.failed}", file=sys.stderr)
        return RunSummary(
            status=status, exit_code=exit_code, stop_reason=stop,
            winners={k: w["kernel"] for k, w in winners.items()},
            unresolved=dict(self.unresolved), counts=self.counts, wall_s=wall,
            error=error,
        )


# ------------------------------------------------------- op binding plumbing
# Shared by the op bindings (GEMM: tile_engine/ops/gemm/gemm_search.py), so a
# new op adds only its candidates, its build and measure hooks and the
# reconciliation of its driver's rows into outcomes.


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def source_info(files, domain: Mapping) -> dict:
    """run.json provenance: the CK commit, the sha256 of each given input file
    (``None`` entries skipped) and of ``domain`` ({problem: [kernel names]})."""
    return {
        "sources": {"ck_commit": _git_commit(),
                    "sha256": {str(p): sha256_file(p) for p in files if p}},
        "domain_sha256": hashlib.sha256(
            json.dumps(domain, sort_keys=True).encode()).hexdigest(),
    }


class ArtifactStore:
    """The run's own copies of its built kernels, under ``<out>/artifacts``.
    Only these copies are measured: a file the run did not copy is never
    loaded, and a copy that changed on disk since its build is refused."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir) / "artifacts"
        self._copies: Dict[str, tuple] = {}  # key -> (path, sha256)

    def add(self, key: str, src, name: str) -> BuildResult:
        """Copy the built ``src`` in as ``name`` (temp file + rename)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        dst = self.dir / name
        tmp = dst.with_name(f".{name}.tmp")
        shutil.copyfile(src, tmp)
        digest = sha256_file(tmp)
        tmp.replace(dst)
        self._copies[key] = (dst, digest)
        return BuildResult(str(dst), digest=digest)

    def path(self, key: str) -> Path:
        return self._copies[key][0]

    def unusable(self, key: str):
        """(status, diag) when ``key`` has no usable copy of this run, else None."""
        art = self._copies.get(key)
        if art is None:
            return "compile_error", "not built in this run"
        try:
            same = sha256_file(art[0]) == art[1]
        except OSError:
            same = False
        return None if same else ("run_error", f"artifact {art[0]} changed since its build")


def measure_built(store: ArtifactStore, requests: Sequence[PairRequest],
                  run: Callable[[List[PairRequest]], List[PairOutcome]]
                  ) -> List[PairOutcome]:
    """One outcome per request, in order: a request without a usable artifact
    gets its outcome here, ``run`` measures the others (one outcome each, in
    order)."""
    out, rest, bad = {}, [], {}
    for r in requests:
        key = r.candidate_key
        if key not in bad:
            bad[key] = store.unusable(key)
        if bad[key]:
            out[r.call_id] = PairOutcome(r, bad[key][0], None, bad[key][1])
        else:
            rest.append(r)
    if rest:
        for r, o in zip(rest, run(rest)):
            out[r.call_id] = o
    return [out[r.call_id] for r in requests]


# The host side of a verified call outside its kernel, for an op whose worker
# generates random inputs, encodes them for the kernel and computes an fp32
# numpy reference (a miss of the run's host cache pays all of it): the
# reference at HOST_REFERENCE_FLOPS, a rate the host keeps with several device
# workers sharing its cores (when they get the node's CPUs), and each input
# element at the op's own rate, which covers generating, encoding and decoding
# it (GEMM's, by dtype: ``gemm_search.HOST_S_PER_INPUT``).
HOST_REFERENCE_FLOPS = 5e10


def host_allowance_s(flops, inputs, s_per_input) -> int:
    """Seconds of host work of a verified call whose reference takes ``flops``
    and whose ``inputs`` elements take ``s_per_input`` seconds each. An op
    binds it with its own counts and rates (the GEMM one:
    ``gemm_search.verify_allowance_s``)."""
    return math.ceil(flops / HOST_REFERENCE_FLOPS + inputs * s_per_input)


# With the host cache on, a problem's host data is prepared before its first
# measure call, in a worker of its own and outside that call's timeout, so a
# host slower than the op's rates makes the round longer instead of timing the
# kernel out. The preparation's limit only stops one that hangs:
# PREPARE_FACTOR times the problem's host allowance, plus PREPARE_START_S for
# the worker's start. A preparation that finishes without storing the entry
# (the cache's caps) leaves calls that compute it all: their allowance is at
# least PREPARE_MARGIN times what it took on this host, times the calls that
# may compute at once (one per device), which share the CPUs it may have had
# alone. A failed preparation's error ends with the last PREPARE_STDERR_CHARS
# characters of its stderr.
PREPARE_FACTOR, PREPARE_START_S, PREPARE_MARGIN = 10, 60, 2
PREPARE_STDERR_CHARS = 400


# The host memory guard. A problem whose host data (a preparation, or the
# calls that compute or read it) needs more than ``host_memory()`` gets the
# outcome HOST_MEMORY instead of a worker that the operating system's
# out-of-memory killer would end. ``host_memory()`` is the memory available
# now (MemAvailable in /proc/meminfo, or the room left under a memory limit
# of this process's cgroup and its parents when that is less: memory.max or
# memory.high in cgroup v2, memory.limit_in_bytes in v1), less
# HOST_MEMORY_MARGIN of the total (the least of MemTotal and those limits),
# left to the system and to other processes.
HOST_MEMORY = "host_memory"
HOST_MEMORY_MARGIN = 0.1
GIB = 1 << 30


def host_memory(meminfo="/proc/meminfo", cgroup="/proc/self/cgroup",
                cgroup_root="/sys/fs/cgroup") -> Optional[int]:
    """Bytes of host data the search may plan for now (see above); None
    where ``meminfo`` is unreadable (not Linux): no guard."""
    try:
        with open(meminfo) as f:
            info = dict(line.split()[:2] for line in f if len(line.split()) >= 2)
        avail, total = (int(info[k + ":"]) * 1024 for k in ("MemAvailable", "MemTotal"))
    except (OSError, KeyError, ValueError):
        return None
    for room, limit in _cgroup_limits(cgroup, cgroup_root):
        avail, total = min(avail, room), min(total, limit)
    return max(0, int(avail - HOST_MEMORY_MARGIN * total))


# Per cgroup version: (its memory controller's line in /proc/self/cgroup, its
# dir under the cgroup root, its limit files, usage file, inactive_file field).
CGROUP_MEMORY = (
    (lambda ctrl: ctrl == "", "", ("memory.max", "memory.high"), "memory.current",
     "inactive_file"),
    (lambda ctrl: "memory" in ctrl.split(","), "memory", ("memory.limit_in_bytes",),
     "memory.usage_in_bytes", "total_inactive_file"),
)


def _cgroup_limits(cgroup, root) -> List[tuple]:
    """(room, limit) of each level of this process's memory cgroups (v2, or
    v1) with a limit (the least of its limit files): the limit less its usage
    plus its inactive file pages (page cache it can drop)."""
    try:
        with open(cgroup) as f:
            lines = [line.rstrip("\n").split(":", 2) for line in f]
    except OSError:
        return []
    out = []
    for is_memory, sub, limits, usage, inactive in CGROUP_MEMORY:
        rel = next((x[2] for x in lines if len(x) == 3 and is_memory(x[1])), None)
        path = None if rel is None else Path("/" + rel.lstrip("/"))
        while path is not None:
            d = Path(root) / sub / str(path).lstrip("/")
            found = []
            for name in limits:  # "max" (v2): no limit
                with contextlib.suppress(OSError, ValueError):
                    found.append(int((d / name).read_text()))
            try:
                if found:
                    stat = dict(ln.split()[:2] for ln in (d / "memory.stat")
                                .read_text().splitlines() if len(ln.split()) >= 2)
                    used = int((d / usage).read_text())
                    out.append((min(found) - used + int(stat.get(inactive, 0)),
                                min(found)))
            except (OSError, ValueError):
                pass
            path = None if path == path.parent else path.parent
    return out


def host_memory_error(need, avail, detail="") -> str:
    """The diag of a problem whose host data needs ``need`` bytes while
    ``host_memory()`` gives ``avail``."""
    more = f" ({detail})" if detail else ""
    return (f"{HOST_MEMORY}: needs {need / GIB:.1f} GiB{more}, "
            f"{avail / GIB:.1f} GiB available")


# A search's worker ended by SIGKILL was killed by another process: the search
# kills one only at its limit (a preparation's, or a call's timeout), and
# reports that as such. On a host short of memory that is the out-of-memory
# killer, which the search's workers ask to end first. Such a worker, one that
# reports a MemoryError, or one that fails at start with one of OUT_OF_MEMORY
# as the last line of its stderr (numpy or its BLAS refused memory at import)
# ran out of host memory: its problem gets HOST_MEMORY, and no kernel failed.
SIGKILLED = -getattr(signal, "SIGKILL", 9)  # the exit code of a SIGKILLed worker
OUT_OF_MEMORY = ("MemoryError", "Memory allocation", "Cannot allocate memory")


def memory_failure(what: str, code: Optional[int], error: str, stderr: str = "") -> str:
    """The ``host_memory`` diag of a failed worker (``what``: "its
    preparation", ...) that exited with ``code`` (None: unknown) and reported
    ``error``, when it ran out of host memory (see above); else ""."""
    last = (stderr.strip().splitlines() or [""])[-1]
    if code == SIGKILLED:
        return (f"{HOST_MEMORY}: {what} was killed (exit code {code}), "
                "likely by the out-of-memory killer")
    if error.startswith("MemoryError"):
        return f"{HOST_MEMORY}: {what} ran out of memory ({error})"
    if code and any(t in last for t in OUT_OF_MEMORY):
        return f"{HOST_MEMORY}: {what} ran out of memory (exit code {code})"
    return ""


def calls_memory_error(n: int, each: int, stored, avail: Optional[int],
                       kept: int = 0) -> str:
    """The diag of a problem whose calls, up to ``n`` at once (one per
    device) of ``each`` bytes (reading its cache entry when ``stored``, else
    computing its host data), do not fit in ``avail`` bytes of
    ``host_memory()`` less ``kept`` (left to cache entries still to come);
    "" when they fit or ``avail`` is None."""
    if avail is None or n * each <= avail - kept:
        return ""
    how = "reading its entry" if stored else "without a cache entry"
    detail = (f"{n} calls at once {how}, {each / GIB:.1f} GiB each" if n > 1
              else f"a call {how}")
    if kept:
        detail += f"; {kept / GIB:.1f} GiB kept for the tmpfs cache's entries to come"
    return host_memory_error(n * each, max(0, avail - kept), detail)


def entries_to_come(spec: Optional[Mapping], problems: Mapping[str, Any],
                    entry_bytes: Callable[[Any], int]) -> int:
    """Bytes the host cache ``spec`` may still store for ``problems`` (key ->
    problem, whose entry has ``entry_bytes(problem)`` bytes): their entries
    within its ``max_entry_bytes`` and its ``host_cache.room``, and no more
    than that room; 0 without a cache. On a tmpfs these bytes will hold RAM."""
    if not spec:
        return 0
    room = host_cache.room(spec)
    most = min(room, spec["max_entry_bytes"])
    return min(room, sum(n for n in map(entry_bytes, problems.values()) if n <= most))


# The search runs that share a temp dir (those of one host, or of one
# container) coordinate through HOST_LOCK in HOST_LOCK_DIR (None: the system
# temp dir): a round's builds hold it shared, its preparations exclusive, so
# no preparation runs while another run's builds or preparations do (a run's
# own rounds build, then measure). A run that cannot open the lock file says
# so once and goes on without it.
HOST_LOCK, HOST_LOCK_DIR = "ck_search_host.lock", None
_NO_HOST_LOCK = set()  # lock files this process reported it cannot open


def _open_host_lock(path, access: int = os.O_RDONLY) -> int:
    """A descriptor (``access``: read-only, or O_RDWR for an exclusive lock on
    NFS, which needs a writable file) of the lock file ``path``, created
    (mode 0666, so other users' runs can open it) when missing. An existing
    file is opened without O_CREAT: where fs.protected_regular is set (the
    default on most Linux hosts), O_CREAT on another user's file in a sticky
    directory such as /tmp is refused even though the file exists."""
    for flags in (access, access | os.O_CREAT | os.O_EXCL, access):
        try:
            fd = os.open(path, flags, 0o666)
        except (FileNotFoundError, FileExistsError) as e:  # missing, or created meanwhile
            error = e
            continue
        if flags & os.O_CREAT:
            with contextlib.suppress(OSError):
                os.fchmod(fd, 0o666)  # whatever the umask
        return fd
    raise error


@contextlib.contextmanager
def file_lock(path, exclusive: bool, log=None, what: str = "", unlocked: str = "",
              name: str = "lock", access: int = os.O_RDONLY):
    """Hold an flock on the lock file ``path`` (the ``name`` in log lines,
    opened with ``access``, see ``_open_host_lock``): ``exclusive``, or shared.
    One ``log`` line when it has to wait for another run (``what`` it waits
    for). Without fcntl it does not lock; with a lock file it cannot open it
    does not either, and logs that once per process (``unlocked``: what is
    then not kept apart)."""
    try:
        import fcntl

        fd = _open_host_lock(path, access)
    except ImportError:
        yield
        return
    except OSError as e:
        if log is not None and path not in _NO_HOST_LOCK:
            _NO_HOST_LOCK.add(path)
            log(f"  [search] NOTE: cannot open the {name} {path} ({e.strerror or e}); "
                f"{unlocked}")
        yield
        return
    try:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        except OSError:
            if log is not None:
                log(f"  [search] waiting for {what}")
            fcntl.flock(fd, mode)
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def host_lock(exclusive: bool, log=None, what: str = ""):
    """Hold HOST_LOCK: ``exclusive`` for preparations, shared for builds."""
    with file_lock(os.path.join(HOST_LOCK_DIR or tempfile.gettempdir(), HOST_LOCK),
                   exclusive, log, what or "another search run on this host",
                   "this run's builds and preparations are not kept apart from "
                   "other search runs'", "host lock"):
        yield


def prepare_limit_s(allowance_s) -> int:
    """The limit of a preparation whose problem's host allowance is
    ``allowance_s`` seconds."""
    return PREPARE_FACTOR * allowance_s + PREPARE_START_S


def call_allowance_s(allowance_s, prepared: Optional[Mapping] = None,
                     calls: int = 1) -> int:
    """The host seconds a call's timeout adds for a problem whose host
    allowance is ``allowance_s`` and whose preparation record (one value of
    ``prepare_host_data``) is ``prepared``, with up to ``calls`` calls at
    once: never less than ``allowance_s``."""
    took = 0.0
    if prepared and prepared["ok"] and not prepared["stored"]:
        took = prepared["seconds"]
    return max(allowance_s, math.ceil(PREPARE_MARGIN * took * max(1, calls)))


def prepare_host_data(argv, env, payloads: Mapping[str, str],
                      limits: Mapping[str, float], width: int,
                      poll_s: float = 0.1, needs: Optional[Mapping[str, int]] = None,
                      memory: Callable[[], Optional[int]] = host_memory,
                      keeps: Optional[Mapping[str, int]] = None) -> Dict[str, dict]:
    """Run the op's worker ``argv`` once per problem key of ``payloads`` (its
    JSON payload on stdin) to store that problem's host cache entry, at most
    ``width`` at a time, each killed after its ``limits`` seconds. With
    ``needs`` (bytes per key), the ones running at once must also fit in
    ``memory()``, read whenever none is running: one that does not fit
    alone is not started, and its error is ``host_memory_error``'s. Of a
    need, ``keeps`` bytes stay in memory once its worker stored its entry (on
    a tmpfs): they count until ``memory()`` is read again. The worker prints
    ``{"ok": true, "stored": <bool>}`` or ``{"ok": false, "error": ...}`` as
    its last line. Returns ``{key: {"ok", "stored", "seconds", "error"}}`` in
    the order of ``payloads``; ``ok`` is False for an error, a crash or a kill
    at the limit, and ``error`` then ends with the tail of the worker's
    stderr. A worker that ran out of memory (``memory_failure``) gets a
    ``host_memory`` error. Workers still running when this raises (an
    interrupt) are killed."""
    out, todo, running, budget, kept = {}, list(payloads), {}, None, 0
    try:
        while todo or running:
            while todo and len(running) < max(1, width):
                key = todo[0]
                if needs is not None:
                    if not running:
                        budget, kept = memory(), 0
                    used = kept + sum(needs[k] for k in running)
                    if budget is not None and used + needs[key] > budget:
                        if running:
                            break  # it waits for the running ones to end
                        todo.pop(0)
                        out[key] = _prepared(None, "", 0.0,
                                             host_memory_error(needs[key], budget))
                        continue
                todo.pop(0)
                files = []  # the worker's stdin, stdout and stderr
                # A search run's interrupt arriving while Popen starts the
                # worker is raised once the worker is in running, where the
                # cleanup below kills it.
                with interrupts_held():
                    try:
                        while len(files) < 3:
                            files.append(tempfile.TemporaryFile())
                        files[0].write(payloads[key].encode())
                        files[0].seek(0)
                        t0 = time.monotonic()
                        running[key] = (subprocess.Popen(argv, stdin=files[0],
                                                         stdout=files[1],
                                                         stderr=files[2], env=env),
                                        files, t0)
                    except OSError as e:
                        for f in files:
                            f.close()
                        out[key] = _prepared(None, "", 0.0, f"could not start: {e}")
            time.sleep(poll_s)
            for key, (proc, files, t0) in list(running.items()):
                took = time.monotonic() - t0
                if proc.poll() is None and took < limits[key]:
                    continue
                killed = proc.poll() is None
                if killed:
                    proc.kill()
                proc.wait()
                del running[key]
                files[0].close()
                text, stderr = [_drain(f) for f in files[1:]]
                out[key] = _prepared(proc.returncode, text, took,
                                     f"no result in {limits[key]:g} s" if killed else "",
                                     stderr)
                if out[key]["stored"]:
                    kept += (keeps or {}).get(key, 0)
    finally:
        # Each is killed before any is waited for: a second interrupt during a
        # wait leaves none running, only some not yet reaped.
        for proc, _, _ in running.values():
            proc.kill()
        for proc, files, _ in running.values():
            proc.wait()
            for f in files:
                f.close()
    return {key: out[key] for key in payloads}


def _drain(f) -> str:
    """The text written to the temp file ``f``, which is then closed."""
    with f:
        f.seek(0)
        return f.read().decode(errors="replace")


def _prepared(code, text, took, error, stderr="") -> dict:
    """A preparation's record from its exit ``code``, output ``text`` and
    ``took`` seconds; ``error`` set when it never got to report. A failure's
    error ends with the tail of its ``stderr``."""
    got = None
    if not error:
        try:
            got = json.loads((text.strip().splitlines() or [""])[-1])
        except ValueError:
            got = None
        if not isinstance(got, dict):
            error = f"no result line (exit code {code})"
        elif got.get("ok") is not True:
            error = str(got.get("error") or f"failed (exit code {code})")
        if error:  # (one killed at its limit has its error already)
            error = memory_failure("its preparation", code, error, stderr) or error
    tail = " | ".join(stderr.strip()[-PREPARE_STDERR_CHARS:].splitlines())
    if error and tail:
        error += f"; stderr: {tail}"
    return {"ok": not error, "stored": not error and got.get("stored") is True,
            "seconds": round(took, 1), "error": error}


class CallFiles:
    """One new file per adapter call under ``<out>/rounds``:
    ``NNNN-<kind><suffix>``, never an existing file."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir) / "rounds"
        self._n = 0

    def new(self, kind: str, suffix: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        while True:
            self._n += 1
            path = self.dir / f"{self._n:04d}-{kind}{suffix}"
            if not path.exists():
                return path

    def write_json(self, path: Path, data) -> Path:
        """``data`` next to the call's ``path`` (same stem, .json)."""
        return RunWriter(self.dir).write_json(path.with_suffix(".json").name, data)


def check_search_options(args, conflicts=(), required=()) -> SearchSettings:
    """The validated ``--search-config`` of a driver run. The op passes its
    driver's options: ``conflicts`` the (option, args attribute[, hint])
    entries a search run does not take, ``required`` the (option, args
    attribute, value) triples it needs; no other driver option is read. Raises
    SettingsError for a missing or conflicting option, another required
    value, invalid settings or an output dir holding an earlier run."""
    if not (args.search_config and args.search_output_dir):
        raise SettingsError(
            "--search-config and --search-output-dir go together, each with a path")
    for opt, attr, *hint in conflicts:
        if getattr(args, attr):
            raise SettingsError(f"--search-config does not take {opt}"
                                + "".join(f" ({h})" for h in hint))
    for opt, attr, value in required:
        if getattr(args, attr) != value:
            raise SettingsError(f"--search-config supports {opt} {value} only, "
                                f"got {getattr(args, attr)}")
    settings = SearchSettings.from_json(args.search_config)
    RunWriter(args.search_output_dir).check_fresh()
    return settings


# The signal _interrupt_handler held back in the main thread inside
# interrupts_held (the first one), and how deep that nests.
_HELD = {"depth": 0, "signum": None}


@contextlib.contextmanager
def interrupts_held():
    """Hold back the KeyboardInterrupt that ``_interrupt_handler`` raises
    (a search run's SIGTERM, and its SIGINT while that has Python's default
    handler) inside the block, and raise it when the block ends. Main thread
    only, where the handler runs. A process started inside, between the
    system call and the return of Popen, is then in the caller's bookkeeping
    before the interrupt unwinds it."""
    _HELD["depth"] += 1
    try:
        yield
    finally:
        _HELD["depth"] -= 1
        if not _HELD["depth"] and _HELD["signum"] is not None:
            signum, _HELD["signum"] = _HELD["signum"], None
            raise KeyboardInterrupt(f"signal {signum}")


def _interrupt_handler(pid: int):
    """A signal handler raising KeyboardInterrupt in process ``pid`` only. A
    process forked from it (a build pool's worker) inherits the handler; there
    the signal ends the process as by default, so a pool whose workers get it
    breaks at once instead of turning it into one job's error and taking the
    next queued build. A worker started by forkserver (Linux's default from
    Python 3.14) or spawn does not inherit it: SIGTERM still ends it (its
    default), but SIGINT raises in its job, so after Ctrl-C to the process
    group such a pool still runs its queued builds."""
    def handler(signum, frame):
        if os.getpid() != pid:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        if _HELD["depth"]:
            if _HELD["signum"] is None:
                _HELD["signum"] = signum
            return
        raise KeyboardInterrupt(f"signal {signum}")

    return handler


def _run_terminable(coordinator: SearchCoordinator) -> RunSummary:
    """``coordinator.run()`` with SIGTERM (a batch system's time limit) raised
    as KeyboardInterrupt, like Ctrl-C: the run still writes its summary. Both
    go through ``_interrupt_handler`` (SIGINT only while it has Python's
    default handler), so a forked build worker that gets them stops."""
    if threading.current_thread() is not threading.main_thread():
        return coordinator.run()
    sigs = [signal.SIGTERM]
    if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        sigs.append(signal.SIGINT)
    handler = _interrupt_handler(os.getpid())
    old = {sig: signal.signal(sig, handler) for sig in sigs}
    try:
        return coordinator.run()
    finally:
        for sig, h in old.items():
            signal.signal(sig, signal.SIG_DFL if h is None else h)


def run_search(adapter: OperationAdapter, settings: SearchSettings, out_dir,
               devices: Sequence[str], **loaders) -> int:
    """A driver's search mode: runs ``adapter``'s search into ``out_dir`` and
    prints its outcome. ``loaders`` are the coordinator's ``pair_loader`` and
    ``prior_loader``. Returns the exit code: 0 when every problem has a
    confirmed winner, 1 otherwise, 2 for invalid options, 130 when interrupted
    (SIGINT or SIGTERM)."""
    bar = "=" * 80
    print(f"\n{bar}\nAdaptive search: {settings.strategy}, "
          f"{settings.trials_per_problem} trials per problem, at most "
          f"{settings.new_builds_per_run} builds\n{bar}")
    if not devices:
        print("  ERROR: no GPU to measure on (--devices)")
        return EXIT_USAGE
    try:
        coordinator = SearchCoordinator(adapter, settings, out_dir,
                                        log=lambda line: print(line, flush=True),
                                        **loaders)
    except SettingsError as e:
        print(f"  ERROR: {e}")
        return EXIT_USAGE
    s = _run_terminable(coordinator)
    n = len(s.winners) + len(s.unresolved)
    print(f"\n{bar}\nSEARCH {s.status.upper()}: {len(s.winners)}/{n} problems with a "
          f"confirmed winner (stopped: {s.stop_reason})\n{bar}")
    c = s.counts
    print(f"  Builds: {c['built']} ok, {c['build_failed']} failed; measurements: "
          f"{c['measured']}; confirmation calls: {c['confirm_calls']}; "
          f"{s.wall_s:.0f}s")
    for key, why in sorted(s.unresolved.items()):
        print(f"  Unresolved {key}: {why}")
    if s.error:
        print(f"  ERROR: {s.error}")
    print(f"  Output: {out_dir} (winners.json is a --pairs file)")
    return s.exit_code
