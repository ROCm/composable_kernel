# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adaptive kernel search for the GEMM sweep driver (``--search-config``).

``GemmSearchAdapter`` is the GEMM ``OperationAdapter`` of the op-agnostic
coordinator in ``tile_engine/ops/common/adaptive_search.py``. It reuses the
driver's own build and measure functions, which the driver passes in as
``DriverHooks`` (this module never imports the driver):

- candidates: per problem, the kernels a full run would measure with it, one
  per kernel name (with ``--pairs``: only its problems, and the listed kernels
  among those);
- builds: ``setup_multiple_gemm_dispatchers`` on the round's missing kernels,
  verbose like a full run's, with the search's limit per build step
  (``build_timeout_s``; else ``SEARCH_BUILD_TIMEOUT_S`` when they are all
  native, and the build function's own limits when one has fixed vector
  widths); each built .so goes into the run's ``ArtifactStore`` as
  ``<out>/artifacts/<name>.so``, and each call's limit and full build errors
  go to ``<out>/rounds/NNNN-build.json``;
- measurements: ``benchmark_on_devices`` with --verify on, one kernel per worker
  and one problem entry per request, so every row and failure record belongs to
  exactly one request; the worker also checks that the library reports the
  requested kernel (``check_identity``) and reads the problem's inputs and
  reference from the run's host cache (``host_cache.py``) instead of computing
  them on every call. A problem's first call is preceded by its preparation:
  a worker that only stores them in the cache, outside the call's timeout,
  and by the host memory guard: a problem whose host data does not fit in
  the host's memory (``host_compute_bytes``, ``host_read_bytes``) gets
  ``host_memory`` instead of a worker that the out-of-memory killer ends.
  Each call writes its own CSV under ``<out>/rounds/``, next to a JSON file
  mapping its ``problem_idx`` to the request (call id, trial) and holding its
  failure records and the preparations run before it;
- confirmation: the same path with fresh calls, each problem's on one device
  (problem i on device i mod n), so its finalists run on the same GPU.

``reconcile`` turns a request's rows and failure records into its one outcome;
``read_ranked_csv`` turns ``select_kernels.py --out_ranked`` predictions into
the latency prior of ``warm_start.ranked_csv``. The op-agnostic plumbing
(artifact copies, per-call files, provenance, option checks, the entry point)
is in ``adaptive_search``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_COMMON_DIR = str(Path(__file__).resolve().parents[1] / "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import adaptive_search  # noqa: E402
import host_cache  # noqa: E402
from adaptive_search import (  # noqa: E402
    ArtifactStore,
    BuildResult,
    CallFiles,
    Candidate,
    PairOutcome,
    PairRequest,
    SearchSettings,
    SettingsError,  # noqa: F401 - the driver imports it from here
    measure_built,
    source_info,
    valid_latency,
)

# Driver options a search run does not take: it writes its own outputs,
# confirms its finalists itself and has its own build timeout.
CONFLICTS = (
    ("--list-pairs", "list_pairs"),
    ("--repeat-top", "repeat_top"),
    ("--failures-csv", "failures_csv"),
    ("--build-timeout", "build_timeout", "set build_timeout_s in the search settings"),
)
REQUIRED = (("--variant", "variant", "gemm_universal"),)  # the one searched
# Driver failure kind (--failures-csv) -> outcome status. The first kind in
# this order that a request has decides its outcome, unless its worker ran out
# of host memory (``adaptive_search.memory_failure``): ended by SIGKILL, which
# the driver sends only at a call's timeout (a "timeout" record), or reporting
# a MemoryError. That request gets ``host_memory``: its problem stops, no
# kernel is blamed, and its best kernels so far are still confirmed.
FAILURE_STATUS = {
    "timeout": "timeout",
    "batch-error": "run_error",
    "error": "run_error",
    "missing": "run_error",
    "worker-exit": "run_error",
    "unsupported": "run_error",
    "bad-output": "invalid_measurement",
    "mismatch": "numeric_fail",
}
# dispatcher_run_gemm codes in "kernel returned status N" (gemm_ctypes_lib.cpp).
# The pairing should have excluded both: they point at an eligibility gap.
RUN_STATUS = {
    "-2": "no kernel selected for the problem",
    "-3": "the kernel does not support the problem",
}
# A verified one-kernel worker also pays the host side of verification (random
# inputs, their encoding, the fp32 numpy reference GEMM, the comparison), which
# a batched --verify run pays once per batch. With the host cache, each
# problem's entry is prepared before its first call and outside its timeout
# (adaptive_search.prepare_host_data), and its calls only load and compare;
# the calls of a problem without an entry (the cache off, or over its caps)
# pay it all. Each call's timeout is --kernel-timeout plus that time
# (verify_allowance_s, at the per-dtype rates of HOST_S_PER_INPUT), or twice
# what the problem's preparation took when it stored nothing, if that is more.
# The inputs dominate skinny large-K shapes: 34 s of bf16 inputs at
# 256x768x874159, and about 4x that in fp8.
BUILD_ERROR_CHARS = 8000  # per failed kernel in rounds/NNNN-build.json
# A search round measures once all its builds end, so a kernel that never
# builds holds the round for its whole limit (a full run keeps the build
# function's: 300 s per step, 1200 s to compile a fixed-vector-width kernel).
# 180 s per step: 1.6x the slowest gfx950 fp16 native build outside the one
# slow-compiling class (warp tile 4x64x16), see "Build timeout" in README.md.
# It applies to a build call of native kernels only: fixed-width compile times
# were not measured, and one call has one limit, so a round that builds a
# fixed-width kernel (up to 1200 s to compile) keeps 300 s for its native
# kernels too. A shorter native limit would shorten such a round only when a
# native kernel never builds and every fixed-width one ends sooner.
# ``build_timeout_s`` replaces it, for every kernel.
SEARCH_BUILD_TIMEOUT_S = 180
# The lock file, in DriverHooks.build_dir, of a build until its libraries are stored.
BUILD_LOCK = "ck_search_build.lock"


# Host seconds per A or B element of a verified call (generating, encoding and
# decoding it), by dtype. Measured single-threaded with the gemm_utils codecs:
# 33-60 ns per element for fp16, bf16 and fp32, 177-213 ns for fp8 and bf8
# (their encoding dominates). A dtype not listed gets the slowest listed rate.
HOST_S_PER_INPUT = {"fp16": 75e-9, "bf16": 75e-9, "fp32": 75e-9,
                    "fp8": 400e-9, "bf8": 400e-9}


# Host memory of a search worker's host data, measured with
# run_one_gemm_kernel._search_host_data (see "Host verification memory" in
# README.md): per input element its encoded bytes (HOST_INPUT_BYTES, by dtype;
# 4 for a dtype not listed) and its fp32 value while the reference is
# computed; per output element the fp32 reference, C and its comparison
# (HOST_OUTPUT_BYTES); plus HOST_FIXED_BYTES for the interpreter, numpy, one
# block's temporaries and the GPU runtime (a measure worker on an MI350X
# (gfx950) held about 0.9 GB besides its host data).
HOST_INPUT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "bf8": 1}
HOST_OUTPUT_BYTES = 16
HOST_FIXED_BYTES = 1 << 30


def host_entry_bytes(problem, dtype) -> int:
    """Bytes of ``problem``'s host cache entry: A and B encoded, the fp32
    reference."""
    m, n, k = problem["M"], problem["N"], problem["K"]
    return (m + n) * k * HOST_INPUT_BYTES.get(dtype, 4) + 4 * m * n


def host_read_bytes(problem) -> int:
    """Peak bytes of a call that reads ``problem``'s entry: it maps the entry
    (no copy), so C and its comparison, and the fixed part."""
    return (HOST_OUTPUT_BYTES - 4) * problem["M"] * problem["N"] + HOST_FIXED_BYTES


def host_compute_bytes(problem, dtype) -> int:
    """Peak bytes of a worker that computes ``problem``'s host data (a
    preparation, or a call without an entry): the entry, the fp32 values of A
    and B while the reference is computed, and what a reading call needs."""
    m, n, k = problem["M"], problem["N"], problem["K"]
    return host_entry_bytes(problem, dtype) + 4 * (m + n) * k + host_read_bytes(problem)


def host_input_rate(dtype) -> float:
    """Host seconds per input element of ``dtype`` (``HOST_S_PER_INPUT``)."""
    return HOST_S_PER_INPUT.get(dtype, max(HOST_S_PER_INPUT.values()))


def verify_allowance_s(problem, dtype) -> int:
    """Seconds added to ``--kernel-timeout`` for a verified call on ``problem``
    with ``dtype`` inputs: its fp32 reference and its A and B."""
    m, n, k = problem["M"], problem["N"], problem["K"]
    return adaptive_search.host_allowance_s(2 * m * n * k, (m + n) * k,
                                            host_input_rate(dtype))


@dataclass(frozen=True)
class DriverHooks:
    """The driver functions the search reuses."""

    build: Callable  # setup_multiple_gemm_dispatchers
    measure: Callable  # benchmark_on_devices
    record: Callable  # _kernel_record: config -> its scalar fields
    problem_key: Callable  # problem dict -> "MxNxK"
    load_pairs: Callable  # load_pair_list, reads warm_start.pairs_file
    build_reason: Callable  # build_failure_summary: build error -> one line
    # The directory of build's outputs, shared by every run of a checkout
    # (ctypes_utils.get_build_dir()); None: builds are not locked.
    build_dir: Optional[Path] = None


def check_search_options(args) -> SearchSettings:
    """The validated ``--search-config`` of a GEMM driver run (SettingsError
    otherwise, see ``adaptive_search.check_search_options``)."""
    return adaptive_search.check_search_options(args, CONFLICTS, REQUIRED)


def parse_verified(value) -> Optional[bool]:
    """A row's ``verified``: True only for True or "True" (rows in memory carry
    a bool, the CSV the string), False for False or "False", else None."""
    if value is True or (isinstance(value, str) and value == "True"):
        return True
    if value is False or (isinstance(value, str) and value == "False"):
        return False
    return None


def _failure_diag(fail) -> str:
    kind, err = fail.get("kind", ""), str(fail.get("error", ""))
    m = re.search(r"status (-\d+)", err)
    why = RUN_STATUS.get(m.group(1)) if m else None
    return f"{kind}: {err}" + (f" ({why})" if why else "")


def _memory_failure(fails: Sequence[dict]) -> str:
    """The ``host_memory`` diag of a request whose worker ran out of host
    memory, from its failure records; else ""."""
    for f in fails:
        err = str(f.get("error", ""))
        code = re.fullmatch(r"worker exited with code (-?\d+)", err)
        why = adaptive_search.memory_failure(
            "its call's worker", code and int(code.group(1)), err)
        if why:
            return why
    return ""


def reconcile(req: PairRequest, problem: dict, rows: Sequence[dict],
              fails: Sequence[dict]):
    """(status, latency_ms, diag) of one request from its measured rows and
    failure records. Only a verified row of the requested kernel and problem
    with a finite positive latency, and no failure record, is ``ok``."""
    why = _memory_failure(fails)
    if why:
        return adaptive_search.HOST_MEMORY, None, why
    if fails:
        kinds = [f.get("kind") for f in fails]
        kind = next((k for k in FAILURE_STATUS if k in kinds), kinds[0])
        status = FAILURE_STATUS.get(kind, "run_error")
        return status, None, _failure_diag(fails[kinds.index(kind)])
    if not rows:
        return "unattempted", None, "no row and no failure: never run"
    if len(rows) > 1:
        return "invalid_measurement", None, f"{len(rows)} rows for one request"
    row = rows[0]
    got = tuple(str(row.get(f)) for f in ("kernel", "M", "N", "K"))
    want = (req.kernel_name, *(str(problem[d]) for d in "MNK"))
    if got != want:
        return "run_error", None, f"row {got} is not the requested pair {want}"
    verified = parse_verified(row.get("verified"))
    if verified is not True:
        status = "numeric_fail" if verified is False else "invalid_measurement"
        diag = f"verified={row.get('verified')!r} max_rel={row.get('max_rel')!r}"
        return status, None, diag
    lat = row.get("latency_ms")
    if not valid_latency(lat):
        return "invalid_measurement", None, f"latency_ms={lat!r}"
    return "ok", float(lat), ""


def read_ranked_csv(path) -> Dict[str, Dict[str, float]]:
    """The search prior from ``select_kernels.py --out_ranked``: {problem:
    {kernel: predicted latency ms}} with latency = 2*M*N*K / (predicted_tflops
    * 1e9); rows without a finite positive prediction are skipped. Raises
    ValueError (OSError when unreadable) for a file it cannot use."""
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"problem", "kernel", "predicted_tflops"}
        if not need <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {sorted(need)}")
        for i, row in enumerate(reader, 2):
            try:  # a short row holds None
                m, n, k = (int(d) for d in row["problem"].split("x"))
                tflops = float(row["predicted_tflops"])
            except (AttributeError, TypeError, ValueError):
                raise ValueError(
                    f"{path} line {i}: expected an MxNxK problem and a number, "
                    f"got {row['problem']!r}, {row['predicted_tflops']!r}"
                ) from None
            if math.isfinite(tflops) and tflops > 0:
                out.setdefault(row["problem"], {})[row["kernel"]] = (
                    2 * m * n * k / (tflops * 1e9))
    return out


class GemmSearchAdapter:
    """``OperationAdapter`` for gemm_universal (see the module docstring).

    ``eligible[i]`` lists the kernel names ``problems[i]`` may run, in order;
    ``worker`` is the driver's ``(worker_path, base_env)``; ``context`` adds
    the driver's build context (arch, layout, variant, configs) to run.json;
    ``build_timeout`` (seconds) limits each step of every kernel's build, and
    ``None`` means the search default (``_build_timeout_for``); the workers
    share ``cache`` (a ``host_cache.HostCache``; None: each computes its own
    inputs and reference). ``memory`` (``adaptive_search.host_memory``) is
    the host memory guard's reading.
    """

    def __init__(self, args, hooks: DriverHooks, problems, configs, eligible,
                 devices, worker, out_dir, context=None, build_timeout=None,
                 cache=None):
        if len(problems) != len(eligible):
            raise ValueError(f"{len(problems)} problems, {len(eligible)} pair lists")
        self.args = args
        self.hooks = hooks
        self.devices = list(devices)
        self.worker_path, self.base_env = worker
        self._context = dict(context or {})
        self.build_timeout = build_timeout
        self.host_cache = None if cache is None else cache.spec
        # A tmpfs holds the entries in RAM: a preparation's need counts its entry.
        self.tmpfs = cache is not None and host_cache.fs_type(cache.dir) == "tmpfs"
        self.memory = adaptive_search.host_memory
        self.prepared: Dict[str, dict] = {}  # problem key -> its preparation's record
        self.checked = set()  # problem keys whose host memory was checked
        self.skipped: Dict[str, str] = {}  # problem key -> its host_memory diag
        # Verified, one kernel per worker: a batch failure names one request.
        self.margs = argparse.Namespace(**{**vars(args), "verify": True, "batch_size": 1})
        self.extra = {"check_identity": True, "search_host_data": True}
        if self.host_cache:
            self.extra["host_cache"] = self.host_cache
        by_name = {}
        for cfg in configs:
            by_name.setdefault(cfg.name, cfg)
        self.cands: Dict[str, Candidate] = {}
        self.problems: Dict[str, dict] = {}
        self.eligible: Dict[str, List[Candidate]] = {}
        self.repeated = []  # problem keys listed more than once, searched once
        for prob, names in zip(problems, eligible):
            key = hooks.problem_key(prob)
            if key in self.problems:
                self.repeated.append(key)
                continue
            self.problems[key] = {d: int(prob[d]) for d in "MNK"}
            self.eligible[key] = [self._cand(by_name[n]) for n in dict.fromkeys(names)]
        self.store = ArtifactStore(out_dir)
        self.calls = CallFiles(out_dir)

    def _cand(self, cfg) -> Candidate:
        c = self.cands.get(cfg.name)
        if c is None:
            c = Candidate(cfg.name, cfg.name, self.hooks.record(cfg), cfg)
            self.cands[cfg.name] = c
        return c

    # -------------------------------------------------- OperationAdapter

    def context(self) -> dict:
        a = self.args
        files = [*self._context.get("configs", ()), a.problems, a.pairs]
        domain = {k: [c.kernel_name for c in v] for k, v in self.eligible.items()}
        notes = ["candidates are the pairs a full run measures when every kernel builds"]
        if not a.no_vector_fallback:
            notes.append("fixed-vector-width kernels are candidates where a full run "
                         "pairs them; one a full run measures only because its native "
                         "kernel failed to build is not")
        if a.pairs:
            notes.append(f"--pairs {a.pairs} restricts the problems to its keys and "
                         "the candidates to its pairs")
        if a.max_kernels:
            notes.append(f"--max-kernels {a.max_kernels} limits the expansion")
        if self.repeated:
            notes.append(f"repeated problems searched once: {sorted(set(self.repeated))}")
        flops = adaptive_search.HOST_REFERENCE_FLOPS
        rate = host_input_rate(a.dtype)
        each = HOST_INPUT_BYTES.get(a.dtype, 4)
        return {
            **self._context,
            "dtype": a.dtype,
            "problems_file": a.problems,
            "pairs_file": a.pairs,
            "max_kernels": a.max_kernels,
            "vector_fallback": not a.no_vector_fallback,
            "tune_c_vector_width": a.tune_c_vector_width,
            "devices": self.devices,
            "measurement": {"verify": True, "verify_tol": a.verify_tol,
                            "batch_size": 1, "kernel_timeout": a.kernel_timeout,
                            "timeout_s": f"kernel_timeout + ceil(2*M*N*K / {flops:g} "
                                         f"+ (M+N)*K * {rate:g}) per call",
                            "check_identity": True, "host_cache": self.host_cache,
                            "search_host_data": True,
                            "host_memory": {
                                "computing_bytes": f"(M+N)*K*{each + 4} + "
                                f"{HOST_OUTPUT_BYTES}*M*N + {HOST_FIXED_BYTES}",
                                "reading_bytes": f"{HOST_OUTPUT_BYTES - 4}*M*N + "
                                f"{HOST_FIXED_BYTES}",
                                "available": "MemAvailable (or a cgroup's room) less "
                                f"{adaptive_search.HOST_MEMORY_MARGIN:.0%} of the total",
                                "tmpfs_entries": None if not self.tmpfs else (
                                    "a preparation's own entry, and for the calls "
                                    "the entries still to come")},
                            "prepare": None if not self.host_cache else (
                                "before a problem's first call, outside its timeout, "
                                f"limit {adaptive_search.PREPARE_FACTOR} x the host "
                                f"allowance + {adaptive_search.PREPARE_START_S} s; a "
                                "problem it prepared but could not store gets at least "
                                f"{adaptive_search.PREPARE_MARGIN} x its time x "
                                f"{len(self.devices)} (the devices) per call")},
            "build": {"workers": a.workers,
                      "build_timeout": self.build_timeout or SEARCH_BUILD_TIMEOUT_S,
                      "build_timeout_applies_to": "every kernel" if self.build_timeout
                      else "rounds of native kernels only; a round with a fixed-vector-"
                           "width kernel keeps the build function's limits"},
            **source_info(files, domain),
            "domain_notes": notes,
        }

    def enumerate_eligible(self) -> Dict[str, List[Candidate]]:
        return {k: list(v) for k, v in self.eligible.items()}

    def _build_timeout_for(self, cands: Sequence[Candidate]) -> Optional[int]:
        """The limit per build step of one build call over ``cands``: the
        given ``build_timeout``; else SEARCH_BUILD_TIMEOUT_S when every kernel
        is native, and None (the build function's own limits, as in a full
        run) when one has fixed vector widths."""
        fixed = any(any(c.original_config.vector_sizes) for c in cands)
        return self.build_timeout or (None if fixed else SEARCH_BUILD_TIMEOUT_S)

    def _build_outputs_lock(self):
        """An exclusive lock on the build's outputs, held from a build until
        its libraries are in the store. The build writes each kernel's header,
        .o and .so to paths from its name alone (no arch), shared by every run
        of this checkout: two runs building a kernel at once would otherwise
        overwrite, and could store, each other's files."""
        d = self.hooks.build_dir
        if d is None:
            return contextlib.nullcontext()
        with contextlib.suppress(OSError):
            Path(d).mkdir(parents=True, exist_ok=True)
        return adaptive_search.file_lock(
            Path(d) / BUILD_LOCK, True, _log,
            "another search run's build in this checkout",
            "this run's builds are not kept apart from other search runs' in "
            "this checkout", "build lock", os.O_RDWR)

    def build_missing(self, cands: Sequence[Candidate]) -> Dict[str, BuildResult]:
        with self._build_outputs_lock():
            return self._build_and_store(cands)

    def _build_and_store(self, cands: Sequence[Candidate]) -> Dict[str, BuildResult]:
        errors: Dict[str, str] = {}
        limit = self._build_timeout_for(cands)
        with adaptive_search.host_lock(False, _log,
                                       "another search run's host data preparation"):
            paths = self.hooks.build(
                [c.original_config for c in cands], verbose=True,
                max_workers=self.args.workers, errors=errors, build_timeout=limit,
            )
        if len(paths) != len(cands):
            raise RuntimeError(f"build returned {len(paths)} paths for {len(cands)} kernels")
        log = self.calls.new("build", ".json")
        self.calls.write_json(log, {
            "kernels": [c.kernel_name for c in cands],
            "build_timeout": limit,
            "errors": {c.kernel_name: str(errors.get(c.kernel_name, ""))[:BUILD_ERROR_CHARS]
                       for c, lib in zip(cands, paths) if lib is None},
        })
        out = {}
        for c, lib in zip(cands, paths):
            if lib is None:
                why = self.hooks.build_reason(errors.get(c.kernel_name))
                out[c.canonical_key] = BuildResult(None, why or "build failed")
            else:
                out[c.canonical_key] = self.store.add(c.canonical_key, lib,
                                                      f"{c.kernel_name}.so")
        return out

    def measure_pairs(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        return measure_built(self.store, requests, lambda rs: self._run(rs, "measure"))

    def confirm(self, requests: Sequence[PairRequest]) -> List[PairOutcome]:
        return measure_built(self.store, requests, lambda rs: self._run(rs, "confirm"))

    # ----------------------------------------------------------- measure

    def _allowance_s(self, key) -> int:
        """Host seconds a call's timeout adds for problem ``key``."""
        return adaptive_search.call_allowance_s(
            verify_allowance_s(self.problems[key], self.args.dtype), self.prepared.get(key),
            len(self.devices))

    def _prepare(self, reqs, name) -> List[dict]:
        """Before the call ``name``, guard the host memory of each problem of
        ``reqs`` new to the run and, with the host cache on, prepare it: store
        its entry outside the call's timeout, under the host lock and within
        the memory available (``host_compute_bytes``, plus the entry on a
        tmpfs). Then up to one call per device at once reads the entry, or
        computes it all when none was stored, within the memory available
        less, on a tmpfs, the entries still to come: a problem that does not
        fit, or whose preparation ran out of memory, is skipped
        (``host_memory``, one console line, its entry dropped). Returns the
        preparations' records, for that call's JSON file."""
        first: Dict[str, str] = {}
        for r in reqs:  # one dtype and layout per run: any kernel of it will do
            if r.problem_key not in self.checked:
                first.setdefault(r.problem_key, r.kernel_name)
        if not first:
            return []
        self.checked.update(first)
        dtype, records = self.args.dtype, []
        to_come = functools.partial(adaptive_search.entries_to_come, self.host_cache,
                                    entry_bytes=lambda p: host_entry_bytes(p, dtype))
        if self.host_cache:
            limits = {k: adaptive_search.prepare_limit_s(
                verify_allowance_s(self.problems[k], dtype)) for k in first}
            # On a tmpfs an entry the cache may store stays in RAM: its
            # preparation needs it too.
            keeps = {k: to_come({k: self.problems[k]}) if self.tmpfs else 0 for k in first}
            needs = {k: host_compute_bytes(self.problems[k], dtype) + keeps[k]
                     for k in first}
            print(f"  [search] {name}: preparing the host data of {len(first)} "
                  f"problem(s) (limit {max(limits.values())} s)", flush=True)
            with adaptive_search.host_lock(True, _log,
                                           "another search run's builds or preparation"):
                got = adaptive_search.prepare_host_data(
                    [sys.executable, str(self.worker_path)], self.base_env,
                    {k: json.dumps({"prepare": True, "host_cache": self.host_cache,
                                    "problem": self.problems[k], "kernel_name": n})
                     for k, n in first.items()}, limits, len(self.devices),
                    needs=needs, memory=self.memory, keeps=keeps)
            self.prepared.update(got)
            print(f"  [search] {name}: {sum(g['stored'] for g in got.values())} of "
                  f"{len(got)} stored, {max(g['seconds'] for g in got.values()):.0f} s",
                  flush=True)
            for k, g in got.items():
                if g["error"].startswith(adaptive_search.HOST_MEMORY + ":"):
                    self._skip(k, g["error"])
                elif not g["stored"]:
                    print(f"    {k}: not stored ({g['error'] or 'the cache refused it'}); "
                          "its calls compute their own", flush=True)
            records = [{"problem": k, "kernel": first[k], "limit_s": limits[k], **got[k]}
                       for k in first]
        budget, kept = self.memory(), 0
        if self.tmpfs:  # the entries still to come will hold RAM during the calls
            kept = to_come({k: p for k, p in self.problems.items() if k not in self.skipped
                            and not self.prepared.get(k, {}).get("stored")})
        for k in first:
            stored = self.prepared.get(k, {}).get("stored")
            why = k not in self.skipped and adaptive_search.calls_memory_error(
                len(self.devices), host_read_bytes(self.problems[k]) if stored
                else host_compute_bytes(self.problems[k], dtype), stored, budget, kept)
            if why:
                self._skip(k, why)
                if stored:  # no call will read it
                    host_cache.drop(self.host_cache, lambda fields, p=self.problems[k]:
                                    fields.get("problem") == p)
        return records

    def _skip(self, key, diag) -> None:
        self.skipped[key] = diag
        print(f"    {key}: skipped, {diag}", flush=True)

    def _run(self, reqs, kind) -> List[PairOutcome]:
        """One measure call over ``reqs`` (all with a usable artifact)."""
        built, index = [], {}
        for r in reqs:
            if r.candidate_key not in index:
                index[r.candidate_key] = len(built)
                cfg = self.cands[r.candidate_key].original_config
                built.append((cfg, self.store.path(r.candidate_key)))
        csv_path = self.calls.new(kind, ".csv")
        rows, fails, pin = [], [], {}
        prepared = self._prepare(reqs, csv_path.stem)
        skipped = {r.call_id: self.skipped[r.problem_key] for r in reqs
                   if r.problem_key in self.skipped}
        run = [r for r in reqs if r.call_id not in skipped]
        if run:
            keys = list(dict.fromkeys(r.problem_key for r in run))
            title = (f"[search] {csv_path.stem}: {len(run)} request(s) on "
                     + ", ".join(keys[:6]) + (", ..." if len(keys) > 6 else ""))
            if kind == "confirm":  # a problem's finalists are compared on one GPU
                order = {key: i for i, key in enumerate(self.problems)}
                pin["device_of"] = [self.devices[order[r.problem_key] % len(self.devices)]
                                    for r in run]
            problems = [self.problems[r.problem_key] for r in run]
            margs = argparse.Namespace(**{**vars(self.margs), "kernel_timeout": (
                self.margs.kernel_timeout + max(map(self._allowance_s, keys)))})
            self.hooks.measure(
                problems, [[index[r.candidate_key]] for r in run],
                built, self.devices, margs, self.worker_path, self.base_env,
                csv_path, extra=self.extra,
                record_failures=fails.extend, measured=rows, title=title, **pin,
            )
        self.calls.write_json(csv_path, {
            "requests": [
                {"problem_idx": i, "call_id": r.call_id, "trial_uid": r.trial_uid,
                 "problem": r.problem_key, "kernel": r.kernel_name}
                for i, r in enumerate(run)
            ],
            "failures": fails,
            "prepared": prepared,
            "host_memory": skipped,
        })
        by_rows, by_fails = defaultdict(list), defaultdict(list)
        for row in rows:
            by_rows[row.get("problem_idx")].append(row)
        for f in fails:
            by_fails[f.get("problem_idx")].append(f)
        out = {cid: PairOutcome(r, adaptive_search.HOST_MEMORY, None, skipped[cid])
               for r in reqs for cid in [r.call_id] if cid in skipped}
        for i, r in enumerate(run):
            ran = by_rows[i] + by_fails[i]
            out[r.call_id] = PairOutcome(
                r, *reconcile(r, self.problems[r.problem_key], by_rows[i], by_fails[i]),
                device=str(ran[0].get("device", "")) if ran else "")
        return [out[r.call_id] for r in reqs]


def _log(line) -> None:
    print(line, flush=True)


def run_gemm_search(args, settings: SearchSettings, hooks: DriverHooks, *,
                    problems, configs, eligible, devices, worker,
                    context=None) -> int:
    """The driver's search mode (``adaptive_search.run_search``): its exit code."""
    with host_cache.run_cache(settings.host_cache) as cache:
        adapter = GemmSearchAdapter(args, hooks, problems, configs, eligible, devices,
                                    worker, args.search_output_dir, context,
                                    settings.build_timeout_s, cache)
        return adaptive_search.run_search(adapter, settings, args.search_output_dir,
                                          devices, pair_loader=hooks.load_pairs,
                                          prior_loader=read_ranked_csv)
