#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Vendored search core (dispatcher/python/ck_search): it imports without ROCKE,
the GPU stack or the ML stack, parses as Python 3.8 without runtime PEP 585 /
604 aliases, every strategy improves on a synthetic listed space, warm starts
are honoured, exhaustive visits every config, an empty ask after telling
every trial means done, and a seed asks the same trials whatever the
interpreter's PYTHONHASHSEED.

With ROCKE_PY set to the ROCKE platform/python root of the pinned checkout
(ck_search.ORIGIN_SHA) three more checks run: each vendored file differs from
upstream only by lines carrying a numbered patch mark, the definitions in
_compat.py are upstream's, and every strategy asks the same trials as upstream
for the same seed, warm starts (listed or repaired) and prior.

No GPU needed. Run: python3 -m pytest -q tests/test_ck_search_core.py
"""

import ast
import difflib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

DISPATCHER_PY = Path(__file__).resolve().parent.parent / "python"
PKG_DIR = DISPATCHER_PY / "ck_search"
sys.path.insert(0, str(DISPATCHER_PY))

import ck_search as cs  # noqa: E402
from _py38_check import py38_problems  # noqa: E402

VENDORED = (
    "de.py",
    "local.py",
    "params.py",
    "portfolio.py",
    "stats.py",
    "strategy.py",
    "surrogate.py",
    "trial.py",
)
CK_AUTHORED = ("__init__.py", "_compat.py", "factory.py")
PATCH_MARK = re.compile(r"# ck_search patch (\d+)")
UPSTREAM_SUBDIR = Path("rocke", "dispatch", "tuning", "search")
FORBIDDEN_MODULES = {"rocke", "lightgbm", "pandas", "torch", "hip", "ctypes_utils"}

# A toy listed GEMM tile space: a fake LDS limit makes it a sparse subset of
# the field product, as real kernel lists are.
TILES, KS = (32, 64, 128, 192, 256), (32, 64, 128)
RECORDS = [
    dict(tile_m=m, tile_n=n, tile_k=k, pipeline=p, scheduler=s)
    for m, n, k, p, s in itertools.product(
        TILES, TILES, KS, ("mem", "compv3", "compv4"), ("intrawave", "interwave")
    )
    if (m + n) * k <= 32768
]
WARM = (RECORDS[5], RECORDS[100])
# Not listed ((192 + 128) * 128 > 32768): its repair to a listed config depends
# on the warm-start rng, so it pins the rng's seed.
UNLISTED_WARM = dict(tile_m=192, tile_n=128, tile_k=128, pipeline="mem",
                     scheduler="intrawave")
TRIALS, BATCH = 40, 4


def _latency_ms(r):
    """Deterministic fake time of M=N=K=4096 on 304 CUs (lower is better)."""
    m, n, k = r["tile_m"], r["tile_n"], r["tile_k"]
    waves = math.ceil(math.ceil(4096 / m) * math.ceil(4096 / n) / 304)
    eff = {"mem": 0.6, "compv3": 0.85, "compv4": 1.0}[r["pipeline"]]
    eff *= (1.0 if r["scheduler"] == "intrawave" else 0.93) * (
        0.8 if m * n >= 65536 else 1.0
    )
    return waves * (m * n / eff + 40 * (m + n)) * 4096 * (1 + 16 / k) / 1e9


def _status(r):
    """A failing corner of the space, so failure paths are exercised."""
    return "compile_error" if r["tile_m"] == 192 and r["pipeline"] == "mem" else "ok"


def _record_key(r):
    return "{tile_m}x{tile_n}x{tile_k}:{pipeline}:{scheduler}".format(**r)


# A distorted copy of the true times: a useful but imperfect prior.
PRIOR_TABLE = {
    _record_key(r): _latency_ms(r) * (1.0 + 0.25 * (r["tile_k"] // 32 % 2))
    for r in RECORDS
}


def _drive(ns, name, seed, warm=(), prior=False, trials=TRIALS, batch=BATCH):
    """Ask/tell loop that tells every trial before the next ask. A key asked
    twice in one batch (a portfolio's members do not see each other's
    pending asks) is answered once and told again as cached, as the upstream
    evaluator does. Returns ([[uid, origin, genome, key, ms|None, cached]],
    strategy)."""
    space = ns.SearchSpace.from_configs(RECORDS, name="toy")
    tp = ns.TablePrior(PRIOR_TABLE, _record_key, scale="log") if prior else None
    strat = ns.make(name, space, seed, warm, tp)
    members = space.members
    trace = []
    while len(trace) < trials:
        asked = strat.ask(min(batch, trials - len(trace)))
        if not asked:
            break
        obs, seen = [], set()
        for t in asked:
            key = space.key(t.genome)
            rec = RECORDS[members[key]]
            status = _status(rec)
            ok = status == "ok"
            ms = _latency_ms(rec) if ok else math.inf
            obs.append(
                ns.Observation(
                    trial=t,
                    key=key,
                    status=status,
                    value=ms,
                    samples=(ms,) if ok else (),
                    cached=key in seen,
                )
            )
            trace.append(
                [t.uid, t.origin, list(t.genome), key, ms if ok else None, key in seen]
            )
            seen.add(key)
        strat.tell(obs)
    return trace, strat


def _make_vendored(name, space, seed, warm, prior):
    return cs.make_strategy(name, space, seed, warm_records=warm, prior=prior)


VENDORED_NS = SimpleNamespace(
    SearchSpace=cs.SearchSpace,
    Observation=cs.Observation,
    TablePrior=cs.TablePrior,
    make=_make_vendored,
)


# ------------------------------------------------------------ import / syntax


def test_import_pulls_in_no_rocke_gpu_or_ml_stack():
    code = (
        "import importlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "for m in sys.argv[2:]:\n"
        "    importlib.import_module('ck_search.' + m)\n"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    mods = [f[:-3] for f in VENDORED + CK_AUTHORED if f != "__init__.py"]
    out = subprocess.run(
        [sys.executable, "-B", "-c", code, str(DISPATCHER_PY)] + mods,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "ck_search" in out
    assert not FORBIDDEN_MODULES & set(out)


def test_package_files_are_the_vendored_list_plus_ck_files():
    assert sorted(p.name for p in PKG_DIR.glob("*.py")) == sorted(
        VENDORED + CK_AUTHORED
    )
    vendor_md = (PKG_DIR / "VENDOR.md").read_text()
    patches = vendor_md.split("## Local patches")[1].split("\n## ")[0]
    assert cs.ORIGIN_SHA in vendor_md
    for f in VENDORED:
        assert f"`{f}`" in vendor_md, f
        for n in PATCH_MARK.findall((PKG_DIR / f).read_text()):
            item = rf"^{n}\. [^\n]*`{re.escape(f)}`"
            assert re.search(item, patches, re.M), f"{f}: patch {n} not listed"


@pytest.mark.parametrize(
    "src, bad",
    [
        ("X = tuple[int, ...]\n", True),
        ("X = dict[str, int]\n", True),  # ast.parse(feature_version=(3, 8)) accepts it
        ("from collections.abc import Mapping\nM = Mapping[str, int]\n", True),
        ("y: int | None = None\n", True),
        ("X = int | str\n", True),
        ("ok = isinstance(1, int | float)\n", True),
        ("def f(x: bytes | bool): pass\n", True),
        ("flags = 1 | 2\n", False),
        ("s = 'ab'.removeprefix('a')\n", True),
        ("from typing import Tuple\nX = Tuple[int, ...]\n", False),
        (
            "from __future__ import annotations\n"
            "def f(x: list[int] | None, z: int | str) -> dict[str, int]:\n"
            "    y: tuple[int, ...] = ()\n",
            False,
        ),
    ],
)
def test_py38_checker_flags_runtime_aliases(src, bad):
    assert bool(py38_problems(src)) is bad


@pytest.mark.parametrize("fname", VENDORED + CK_AUTHORED)
def test_module_is_python38_compatible_and_ascii(fname):
    text = (PKG_DIR / fname).read_text(encoding="utf-8")
    assert text.isascii(), fname
    assert py38_problems(text, fname) == []


# ------------------------------------------------------------------ behaviour


@pytest.mark.parametrize("name", cs.STRATEGY_NAMES)
def test_strategy_improves_on_its_first_sample(name):
    trace, strat = _drive(VENDORED_NS, name, 0)
    fresh = [row for row in trace if not row[5]]
    assert len(trace) == TRIALS
    assert len({row[3] for row in fresh}) == len(fresh), "a told key was asked again"
    times = [row[4] for row in fresh if row[4] is not None]
    assert min(times) < times[0]
    assert strat.best()[1] == min(times)


@pytest.mark.parametrize("name", ["de", "hill", "tpe", "gp", "portfolio"])
def test_warm_records_are_asked_first(name):
    trace, strat = _drive(VENDORED_NS, name, 0, warm=WARM)
    space = strat.space
    warm_keys = [space.key(space.encode(r)) for r in WARM]
    keys = [row[3] for row in trace]
    assert keys[:2] == warm_keys  # batched: no proposal of its own in between
    if name == "portfolio":
        assert [row[1] for row in trace[:2]] == ["warm:record:0", "warm:record:1"]


@pytest.mark.parametrize("batch", [1, BATCH])
@pytest.mark.parametrize("name", ["de", "hill", "tpe", "gp", "portfolio"])
def test_every_warm_start_is_asked_before_the_strategys_own_picks(name, batch):
    # One field of 100 values: more warm starts than DE's default population (6).
    records = [{"tile_m": 16 * (i + 1)} for i in range(100)]
    space = cs.SearchSpace.from_configs(records)
    warm = records[::10]
    strat = cs.make_strategy(name, space, seed=3, warm_records=warm)
    keys = []
    while len(keys) < len(warm):
        asked = strat.ask(batch)
        assert asked
        keys += [space.key(t.genome) for t in asked]
        strat.tell([cs.Observation(t, space.key(t.genome), cs.Status.OK,
                                   float(t.genome[0] + 1)) for t in asked])
    assert keys[: len(warm)] == [space.key(space.encode(r)) for r in warm]


@pytest.mark.parametrize("name", ["gp", "portfolio"])
def test_gp_fits_a_space_with_a_singleton_ordinal_axis(name):
    # A one-level ordinal axis is active in every config; GP must not divide by
    # its zero level span (from_configs fixes such fields, a hand-built space
    # need not). Past n_startup, so the model is fitted, inside a portfolio too.
    space = cs.SearchSpace([cs.Param.ordinal("k_batch", [1]),
                            cs.Param.ordinal("tile_m", [16 * (i + 1) for i in range(40)])])
    strat = cs.make_strategy(name, space, seed=1)
    for _ in range(12):
        asked = strat.ask(BATCH)
        strat.tell([cs.Observation(t, space.key(t.genome), cs.Status.OK,
                                   abs(t.genome[1] - 25) + 1.0) for t in asked])
    gp = strat if name == "gp" else next(m for m in strat.members if m.name == "gp")
    assert gp._fits > 0


@pytest.mark.parametrize("name", ["random", "exhaustive"])
def test_random_and_exhaustive_ignore_warm_records(name):
    assert _drive(VENDORED_NS, name, 3, warm=WARM)[0] == _drive(VENDORED_NS, name, 3)[0]


def test_exhaustive_visits_every_listed_config_once():
    trace, strat = _drive(VENDORED_NS, "exhaustive", 0, trials=10**6, batch=16)
    keys = [row[3] for row in trace]
    assert sorted(keys) == sorted(strat.space.members)
    assert strat.done and strat.ask(4) == []


@pytest.mark.parametrize("batch", [1, 3, 16])
def test_exhaustive_over_a_list_asks_what_the_product_walk_asks(batch):
    def walk_product(name, space, seed, warm, prior):
        return cs.strategy.Exhaustive(space, seed)

    product_walk = SimpleNamespace(**{**vars(VENDORED_NS), "make": walk_product})
    listed, strat = _drive(VENDORED_NS, "exhaustive", 0, trials=10**6, batch=batch)
    walked, upstream = _drive(product_walk, "exhaustive", 0, trials=10**6, batch=batch)
    assert listed == walked and strat.done == upstream.done


def test_exhaustive_over_a_wide_sparse_list_does_not_walk_the_product():
    # 6 kernels differing in 14 fields: a product of 6**14 genomes, which the
    # product walk would need hours for.
    records = [{f"f{j}": (i * 7 + j) % 6 for j in range(14)} for i in range(6)]
    space = cs.SearchSpace.from_configs(records)
    assert math.prod(space.sizes) == 6**14
    strat = cs.make_strategy("exhaustive", space)
    asked = strat.ask(10)
    assert len(asked) == 6 and strat.done
    strat.tell(
        [cs.Observation(t, space.key(t.genome), cs.Status.OK, 1.0) for t in asked]
    )
    assert strat.ask(4) == []


@pytest.mark.parametrize("size", [1, 2, 6, 30])
@pytest.mark.parametrize("name", cs.STRATEGY_NAMES)
def test_empty_ask_after_telling_everything_means_done(name, size):
    space = cs.SearchSpace.from_configs(RECORDS[:: len(RECORDS) // size][:size])
    strat = cs.make_strategy(name, space, seed=1, warm_records=RECORDS[:1])
    for _ in range(10 * size + 10):
        asked = strat.ask(BATCH)
        if not asked:
            break
        strat.tell(
            [cs.Observation(t, space.key(t.genome), cs.Status.OK, 1.0) for t in asked]
        )
    assert asked == [] and strat.done and not strat.pending


def test_factory_reproduces_upstream_defaults():
    space = cs.SearchSpace.from_configs(RECORDS)
    de = cs.make_strategy("de", space)
    assert (de.variant, de.adapt) == ("current_to_pbest1bin", "shade")
    pf = cs.make_strategy("portfolio", space, seed=2, warm_records=WARM)
    assert [(m.name, m.seed) for m in pf.members] == [
        ("hill", 203),
        ("gp", 204),
        ("de", 205),
    ]
    assert pf.members[0].n_init == 0 and pf.members[0].start is None
    assert [w.source for w in pf.warm] == ["record:0", "record:1"]
    hill = cs.make_strategy("hill", space, warm_records=WARM)
    assert hill.start == space.encode(WARM[0])
    assert hill.seed_queue == [list(space.encode(WARM[1]))]
    # Warm starts are repaired with upstream's rng, seeded "warm:<seed>".
    assert space.key(space.encode(UNLISTED_WARM)) is None
    repaired = space.repair(space.encode(UNLISTED_WARM), random.Random("warm:7"))
    assert repaired is not None and space.key(repaired) is not None
    hill = cs.make_strategy("hill", space, 7, warm_records=[UNLISTED_WARM])
    assert hill.start == repaired
    prior = cs.TablePrior(PRIOR_TABLE, _record_key)
    for name in ("gp", "tpe"):
        assert cs.make_strategy(name, space, prior=prior).prior is prior
    pf = cs.make_strategy("portfolio", space, prior=prior)
    readers = [m for m in pf.members if hasattr(m, "prior")]
    assert [m.name for m in readers] == ["gp"] and readers[0].prior is prior
    for name, options in (
        ("sh", None),
        ("portfolio", {"members": ()}),
        ("portfolio", {"members": ("hill", "portfolio")}),
    ):
        with pytest.raises(ValueError):
            cs.make_strategy(name, space, options=options)


@pytest.mark.parametrize("name, floor", [("hill", 0.002), ("portfolio", 0.001)])
def test_attached_noise_model_raises_the_gain_threshold(name, floor):
    # Without a noise model a gain just above min_gain counts; the default
    # model (as upstream's run_search attaches) asks for about 1 % on one sample.
    st = cs.make_strategy(name, cs.SearchSpace.from_configs(RECORDS), 0)
    assert st.gain_threshold(floor, 1) == floor
    noise = cs.NoiseModel()
    st.attach_noise(noise)
    assert st.gain_threshold(floor, 1) > 0.009
    assert all(m.noise is noise for m in getattr(st, "members", ()))


def _hash_seed_run(hash_seed):
    """Every strategy's trace (batched, seed 5) and the exact float bits of
    each density TPE fits, from a fresh interpreter run with
    ``PYTHONHASHSEED=hash_seed``: str hashes, and so the iteration order of a
    set of config keys, change with it."""
    run = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--hash-seed-run"],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONHASHSEED=str(hash_seed)),
        timeout=300,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def _hash_seed_trace():
    densities = []
    fit = cs.surrogate.TPE._density

    def spy(self, genomes):
        d = fit(self, genomes)
        densities.append([[x.hex() for x in row] for row in d])
        return d

    cs.surrogate.TPE._density = spy
    traces = {n: _drive(VENDORED_NS, n, 5, batch=8)[0] for n in cs.STRATEGY_NAMES}
    return {"traces": traces, "tpe_densities": densities}


def test_proposals_do_not_depend_on_the_hash_seed():
    # TPE once read the batch's pending keys in set order (patch 5): the same
    # seed then fitted different float densities per process, which can flip
    # a near tie between candidates and so the whole search.
    first, second = _hash_seed_run(0), _hash_seed_run(1)
    assert first["tpe_densities"] and first == second


def test_failed_observation_is_strict_json():
    space = cs.SearchSpace.from_configs(RECORDS[:6])
    t = cs.make_strategy("random", space).ask(1)[0]
    ob = cs.Observation(t, space.key(t.genome), cs.Status.COMPILE_ERROR)
    text = json.dumps(ob.to_json(), allow_nan=False)
    back = cs.Observation.from_json(json.loads(text))
    assert back == ob and not back.ok


# ------------------------------------------------- upstream (pinned checkout)


def _rocke_skip_reason():
    if not os.environ.get("ROCKE_PY"):
        return "ROCKE_PY not set (ROCKE platform/python root of the pinned checkout)"
    root = os.environ["ROCKE_PY"]
    if not (Path(root) / UPSTREAM_SUBDIR).is_dir():
        return f"ROCKE_PY has no {UPSTREAM_SUBDIR}"
    try:
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        head = ""
    if head != cs.ORIGIN_SHA:
        return f"ROCKE_PY checkout is at {head or 'unknown'}, expected {cs.ORIGIN_SHA}"
    return ""


ROCKE_SKIP = _rocke_skip_reason()
needs_rocke = pytest.mark.skipif(bool(ROCKE_SKIP), reason=ROCKE_SKIP)

# Patch 3 changes what a batched hill ask returns while warm starts wait, so
# hill with warm starts is compared one trial per ask, where it changes nothing.
PARITY_CASES = [
    dict(name=name, seed=seed, warm=warm, prior=prior,
         batch=1 if name == "hill" and warm else BATCH)
    for name in cs.STRATEGY_NAMES
    for seed, warm, prior in (
        (0, (), False),
        (7, WARM, True),
        (7, (UNLISTED_WARM,) + WARM, True),
    )
]


@needs_rocke
@pytest.mark.parametrize("fname", VENDORED)
def test_vendored_file_differs_from_upstream_only_by_marked_patches(fname):
    up = (Path(os.environ["ROCKE_PY"]) / UPSTREAM_SUBDIR / fname).read_text()
    mine = (PKG_DIR / fname).read_text().splitlines()
    ops = difflib.SequenceMatcher(a=up.splitlines(), b=mine, autojunk=False)
    for tag, i1, i2, j1, j2 in ops.get_opcodes():
        if tag == "equal":
            continue
        assert j2 > j1, f"{fname}: upstream lines {i1 + 1}-{i2} dropped"
        # Each replaced upstream line needs a marked counterpart, so code cannot
        # disappear behind a single marked line.
        assert j2 - j1 >= i2 - i1, (
            f"{fname}: upstream lines {i1 + 1}-{i2} replaced by {j2 - j1} line(s)"
        )
        assert all(PATCH_MARK.search(line) for line in mine[j1:j2]), (
            f"{fname}: unmarked change at line {j1 + 1}"
        )


def _top_level(path, names):
    """{name: ast.dump} of the top-level definitions or assignments of ``names``."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        else:
            continue
        if name in names:
            out[name] = ast.dump(node)
    return out


@needs_rocke
def test_compat_definitions_are_upstream_copies():
    dispatch = Path(os.environ["ROCKE_PY"], "rocke", "dispatch")
    upstream = {
        **_top_level(dispatch / "core.py", {"stable_json_hash"}),
        **_top_level(dispatch / "tuning" / "axes.py", {"Knobs", "KnobAxis"}),
    }
    assert sorted(upstream) == ["KnobAxis", "Knobs", "stable_json_hash"]
    assert _top_level(PKG_DIR / "_compat.py", set(upstream)) == upstream


def _upstream_namespace(root):
    """Upstream search package with its parent packages registered as bare
    namespaces (their __init__ would import the ROCKE runtime)."""
    for name in ("rocke", "rocke.core", "rocke.dispatch", "rocke.dispatch.tuning"):
        mod = ModuleType(name)
        mod.__path__ = [os.path.join(root, *name.split("."))]
        sys.modules[name] = mod
    sys.modules["rocke.runtime"] = ModuleType("rocke.runtime")
    from rocke.dispatch.tuning import search as up
    from rocke.dispatch.tuning.search import surrogate

    # Patch 5: upstream TPE reads the pending keys in set (hash) order; give it
    # them sorted, as the vendored copy reads them, so the batched TPE cases
    # still compare every trial.
    class SortedSet(set):
        def __iter__(self):
            return iter(sorted(set.__iter__(self)))

    propose = surrogate.TPE._propose
    surrogate.TPE._propose = lambda self, taken, pos: propose(self, SortedSet(taken), pos)

    def make(name, space, seed, warm, prior):
        ws = [up.WarmStart(dict(r), f"record:{i}") for i, r in enumerate(warm)]
        return up.make_strategy(name, space, seed, warm=ws, prior=prior)

    return SimpleNamespace(
        SearchSpace=up.SearchSpace,
        Observation=up.Observation,
        TablePrior=up.TablePrior,
        make=make,
    )


@needs_rocke
def test_trials_match_upstream_for_identical_seeds():
    run = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), os.environ["ROCKE_PY"]],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        timeout=900,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    upstream = json.loads(run.stdout.strip().splitlines()[-1])
    assert len(upstream) == len(PARITY_CASES)
    for case, theirs in zip(PARITY_CASES, upstream):
        mine = json.loads(json.dumps(_drive(VENDORED_NS, **case)[0]))
        assert mine == theirs, f"{case['name']} seed={case['seed']}"


if __name__ == "__main__":
    if sys.argv[1] == "--hash-seed-run":  # test_proposals_do_not_depend_on_the_hash_seed
        print(json.dumps(_hash_seed_trace()))
        sys.exit(0)
    # The upstream side of test_trials_match_upstream_for_identical_seeds, in
    # its own interpreter so ROCKE modules never enter the test process.
    ns = _upstream_namespace(sys.argv[1])
    print(json.dumps([_drive(ns, **case)[0] for case in PARITY_CASES]))
