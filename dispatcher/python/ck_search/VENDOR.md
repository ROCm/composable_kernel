# ck_search: vendored ROCKE search core

`ck_search` is a pinned, minimal copy of the ask/tell search strategies of the
ROCKE tuning search package, used by the tile_engine adaptive kernel search.
Importing the upstream package in place loads the ROCKE runtime, and the
upstream change is not merged yet, so the strategy code is copied here with a
short list of local patches. It needs only the standard library and NumPy.

## Origin

| | |
|---|---|
| Repository | ROCm/rocm-libraries |
| Change | ROCm/rocm-libraries#14464 |
| Commit | `61501dd46f68c196f16cfe3282bd979dbd0746ae` (`ORIGIN_SHA` in `__init__.py`) |
| Path | `dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/dispatch/tuning/search/` |
| License | MIT, upstream copyright headers kept |

The pinned commit is the head of the upstream change, which is not merged yet;
a squash merge will not keep it on `develop`. Until the merge, fetch it with
`git fetch origin refs/pull/14464/head`. After the merge, re-pin to the merge
commit on `develop` (see the update procedure). Once the upstream search
package imports without the ROCKE runtime, drop this copy and import it
instead: the monorepo should not keep two copies of the search core.

## Vendored files

Copied verbatim, then patched as listed below. The blob is
`git rev-parse <commit>:<path>/<file>` at the pinned commit.

| File | Upstream lines | Upstream blob | Provides |
|---|---|---|---|
| `trial.py` | 363 | `57e7d5814b7b8b669221ac6fe12c14e2b0f9c64b` | Trial, Observation, Status, Fidelity |
| `params.py` | 938 | `f3d48dfaf10f9423da683765b8d631bdb7cf7c35` | Param, SearchSpace (`from_configs`, repair, keys) |
| `strategy.py` | 250 | `e3d4d2a5e6c09b446797c5bbed0b97c90eac3400` | Strategy base, RandomSearch, Exhaustive |
| `stats.py` | 437 | `db9db360958c53f8368564bf5ef1ea34a280eff5` | NoiseModel (used by Strategy), Race |
| `de.py` | 572 | `9b4ea6720ad4b31a90ccaa8455776de7c9f10bd2` | DifferentialEvolution (SHADE, current-to-pbest) |
| `local.py` | 354 | `9a4270e928dbca7e1a48f75c63b95d0433ce800a` | HillClimb |
| `surrogate.py` | 733 | `0f1026a178ff41187dd887652f6c1f62124086dc` | TPE, GPEI, TablePrior |
| `portfolio.py` | 264 | `572df1c78148e5b2f0f640d0f03187be0121e427` | Portfolio (bandit), WarmStart |

Total 3911 upstream lines. This is the full import closure of the strategies
`random`, `exhaustive`, `de`, `hill`, `tpe`, `gp` and `portfolio`: the files
above import only each other, the standard library, NumPy (`surrogate.py` at
import, `params.py` on use) and, outside the package, the three names
replaced by patch 1. `lightgbm` is imported only when a `LightGBMPrior` is
constructed; `ck_search` does not export it.

Not vendored: `api.py` (imports the ROCKE build, measure and database layers;
`factory.py` reproduces its `make_strategy` / `_init_genomes` for the names
above), `db.py`, `driver.py`, `measure.py`, `halving.py`, `knobspace.py`,
`replay.py`, `synthetic.py`, `__main__.py` and the upstream `__init__.py`.

## CK files

- `__init__.py`: lean exports and `ORIGIN_SHA`.
- `_compat.py`: `stable_json_hash`, `Knobs` and `KnobAxis`, copied unchanged
  (docstrings included) from upstream `rocke/dispatch/core.py` and
  `rocke/dispatch/tuning/axes.py`; the tests compare the three definitions
  with upstream when `ROCKE_PY` is set.
- `factory.py`: `STRATEGY_NAMES` and `make_strategy(name, space, seed=0, *,
  warm_records=(), prior=None, options=None)`. Same strategies, seeds and
  defaults as upstream: DE `current_to_pbest1bin` + `shade`; hill starts at
  the first warm start and measures the rest next; a portfolio of
  `("hill", "gp", "de")` with member seeds `seed * 101 + i + 1`, the climber
  without its stratified probe, and the warm starts as its first asks;
  `random` and `exhaustive` ignore warm starts; only `tpe` and `gp` use the
  prior. Interface differences: warm starts are records (or `WarmStart`
  objects) instead of `WarmStart` only, the portfolio member list is
  `options["members"]`, and there is no fidelity ladder (only the
  multi-fidelity strategies, not vendored, use it). Behaviour difference:
  with more distinct warm starts than DE's `pop_size`, upstream drops the
  extra ones; here DE's population size (the size of every generation, not
  only the first) grows to the number of distinct warm starts, so every
  requested warm start is measured before DE evolves. On a config list
  (`SearchSpace.from_configs`), `exhaustive` asks the same trials in the same
  order as upstream but steps through the sorted listed genomes instead of
  the whole product of field levels: GEMM kernel records vary in up to 19
  fields, so that product reaches millions of genomes for a few listed
  kernels and one upstream `ask` can take minutes.

## Local patches

Every patched line ends with `# ck_search patch <n>`;
`tests/test_ck_search_core.py` checks that no other line differs from
upstream (when `ROCKE_PY` points at the pinned checkout) and that every mark
is listed here.

1. `params.py`: `from ...core import stable_json_hash` and
   `from ..axes import KnobAxis, Knobs` import from `._compat` instead.
2. `params.py`, `stats.py`: the runtime aliases `Genome = tuple[int, ...]` and
   `Call = tuple[int, float]` use `typing.Tuple` (PEP 585 subscripts fail at
   runtime on Python 3.8, which CK supports). Every other PEP 585 / 604 type
   in these files is an annotation, not evaluated under
   `from __future__ import annotations`.
3. `local.py`: with warm starts beyond `start`, a batched first `ask`
   returns the start and then the warm starts, before any neighbour of the
   start (upstream fills the batch with the start's neighbours and measures
   the warm starts only in the next ask, so a budget of two trials never
   reached the second warm start). One-trial asks are unchanged.
4. `surrogate.py`: GP maps an active one-level ordinal axis to the feature
   0.0 instead of dividing by its zero level span. `from_configs` fixes
   constant fields, but a hand-built `SearchSpace` may hold such an axis, and
   GP (alone or in a portfolio) raised `ZeroDivisionError` once it fitted.
5. `surrogate.py`: TPE reads the batch's pending keys (`taken`) in sorted
   order when it fits its "bad" density, as GP already does. Upstream
   iterates the set, whose order follows the per-process str hash
   (`PYTHONHASHSEED`): the float sums of the density then differ between
   processes, a near tie between candidates can flip, and the same seed gave
   different searches (in a replay of the GEMM search, 8-10 of 100 seeds,
   by up to 12.5 points of the best kernel's speed). The parity check gives
   upstream's TPE the same keys in sorted order.

## Notes for drivers

Checked by `tests/test_ck_search_core.py`:

- Tell every trial before the next ask. Then an empty ask means the strategy
  is done (`strategy.done` is true and nothing is pending).
- Attach a timing noise model, as upstream's `run_search` does
  (`strategy.attach_noise(NoiseModel())`; a portfolio passes it to its
  members). Without one, `gain_threshold` returns its floor, so hill moves
  its center and the portfolio rewards a member on a 0.2 % or 0.1 % gain,
  well inside call-to-call timing noise.
- A portfolio may return the same config twice in one ask: its members do not
  see each other's pending asks, and upstream the evaluator answers the repeat
  from its cache. Measure a key once per batch and tell the repeat as an
  observation with `cached=True` (it does not move the bandit). No strategy
  asks again for a key it was already told.
- DE asks in whole generations, so `ask(n)` can return fewer than `n` trials.
- Failed observations carry `value=inf`; `Observation.to_json()` writes it as
  `null` (strict JSON).

## Update procedure

1. Pick the new upstream commit: a commit on `develop` (after the upstream
   change merges, its merge commit, even when no file changed), or until
   then the change's head from `refs/pull/14464/head`. Recompute the closure:
   list the imports of the vendored files (`grep -n "^from \|^import " *.py` in the upstream
   directory); any new module they import must be vendored too or covered by
   `_compat.py`.
2. Copy the files verbatim from that commit, for example
   `git show <commit>:<path>/params.py > params.py`.
3. Re-apply the patches above to the lines that still need them, each with
   its mark. Drop a patch that upstream made unnecessary and say so here
   (keep the remaining numbers).
4. Update `ORIGIN_SHA` in `__init__.py` and the table above (line counts,
   blobs).
5. Run the tests from `dispatcher/` with `ROCKE_PY` set to the new
   checkout's `dnn-providers/hip-kernel-provider/rocke/platform/python`
   directory (the one containing the `rocke` package):
   `ROCKE_PY=<dir> python3 -m pytest -q tests/test_ck_search_core.py`.
   The marked-diff check and the same-seed parity check must pass.
6. Send general fixes upstream first, so the patch list shrinks.
