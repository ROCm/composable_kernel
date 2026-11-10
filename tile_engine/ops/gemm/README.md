# CK Tile Engine GEMM Operations

## Overview

The CK Tile Engine GEMM module provides a comprehensive system for generating, building, and benchmarking GEMM (General Matrix Multiplication) kernels with various configurations. It supports multiple data types, layouts, and optimization strategies. The system has evolved from a monolithic build approach (where all kernels compile into a single executable) to a more flexible individual kernel compilation system, providing better build parallelism and targeted testing capabilities.

## Table of Contents

0. [Dispatcher Bridge Workflow](#dispatcher-bridge-workflow)
1. [Build System Architecture](#build-system-architecture)
2. [Build Instructions](#build-instructions)
3. [Running Benchmarks](#running-benchmarks)
4. [Configuration System](#configuration-system)
5. [Scripts and Tools](#scripts-and-tools)
6. [Command Line Options](#command-line-options)
7. [Understanding Kernel Names](#understanding-kernel-names)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

## Dispatcher Bridge Workflow

The **Dispatcher bridge** is the recommended path for sweeping and benchmarking
GEMM kernels. Instead of building monolithic or per-kernel executables through
CMake, Tile Engine expands a sweep config into shared `GemmKernelConfig` objects
and hands them to the Dispatcher, which codegens and compiles each into its own
`.so`. The kernel name produced by the bridge is byte-for-byte identical to the
codegen `KERNEL_NAME`, so the bridge runs exactly the same kernels the native
Tile Engine does — it only swaps the harness.

### Scripts

| Script | Role |
|---|---|
| `gemm_full_benchmark.py` | Driver: compile (Phase 1) → load problems (Phase 2) → benchmark across all visible GPUs (Phase 3). |
| `run_one_gemm_kernel.py` | Disposable worker: loads one `.so` in an isolated subprocess and times it. A GPU fault kills only the worker. |
| `gemm_search.py` | Adaptive search mode of the driver (`--search-config`), see [Adaptive kernel search](#adaptive-kernel-search---search-config). |

To measure only the kernels a trained model picks, and to score the picks
against a full sweep, see
[dispatcher/heuristics/HEURISTIC_BENCHMARKING.md](../../../dispatcher/heuristics/HEURISTIC_BENCHMARKING.md).

### Folder layout

The bridged regular-GEMM path follows the same op-root convention as the merged
`fmha/` and `grouped_conv/` bridges — driver + worker + a flat `configs/` at the
op root:

```
gemm/
├── gemm_full_benchmark.py     # bridge driver (op root)
├── run_one_gemm_kernel.py     # disposable per-kernel worker (op root)
├── configs/                   # bridged gemm_universal sweep configs (flat)
├── gemm_instance_builder.py   # shared generator for the non-bridged variants
├── gemm_benchmark.{py,hpp}, gemm_common.hpp, gemm_profiler.hpp  # shared harness
├── gemm_multi_d/   gemm_preshuffle/   grouped_gemm/             # legacy variants
└── README.md
```

`configs/` ships example sweep configs:

- `default_ci_config.json` — small CI-sized sweep (the driver's default when no
  config is passed).
- `default_config.json` — full sweep.
- `user_provided_config.json` — scratch space for custom sweeps.
- `example_problems.json` — example M/N/K problem set (used when `--problems`
  is omitted).

> The JSON used by **nightly** tests is intended to drop into the same
> `configs/` directory and be selected with a positional config — no driver
> changes needed.

The other variants keep their own per-variant `configs/` directories; the driver
selects them with `--variant`. For `gemm_preshuffle` on gfx1250, the default is
[`default_config_gfx1250.json`](gemm_preshuffle/configs/default_config_gfx1250.json),
covering [five pipelines](gemm_preshuffle/README.md) for fp16/bf16/fp8/bf8. An explicit
positional config always takes precedence.

### Running

```bash
cd tile_engine/ops/gemm

# Default: gemm_universal variant, its CI sweep + example problems,
# auto-detect and use all visible GPUs.
python gemm_full_benchmark.py

# Full sweep, fp16/rcr, restricted to 4 GPUs, custom output:
python gemm_full_benchmark.py --variant gemm_universal \
    configs/default_config.json \
    --dtype fp16 --layout rcr --devices 4 --csv gemm_results.csv

# Specific GPU ids and a custom problem file:
python gemm_full_benchmark.py --devices 0,2,5 \
    --problems configs/example_problems.json

# Correctness mode: check every kernel against an fp32 numpy reference.
python gemm_full_benchmark.py --verify --max-kernels 8
```

### Liveness vs correctness (`--verify`)

By default a measurement is reported `OK` purely on **liveness** — the kernel
ran and produced a non-zero output (`ZERO` otherwise). It is *not* a correctness
check: a numerically wrong but non-zero result still reads `OK`. Pass `--verify`
to have each worker compare its output against an fp32 numpy reference
(`A @ B`) using the global relative metric `max|out - ref| / max|ref|`. With
`--verify`, results read `VERIFY` (within `--verify-tol`, default `2e-2`) or
`MISMATCH` (counted as a failure), and the `max_rel` / `verified` columns are
populated in the CSV. This gives self-contained per-kernel confidence; the
broader numeric parity against native Tile Engine remains a separate task.

### Measurement options

All of these are off by default; without them the driver builds, pairs,
measures and reports exactly as described above.

- `--build-timeout SECONDS` — one limit for each kernel's codegen, compile and
  link step, instead of the bridge's defaults (300 s each, 1200 s to compile a
  fixed-vector-width kernel). The limits are wall-clock, so a large `--workers`
  build on a busy host can hit them; a kernel whose step times out is reported
  as a build failure (`Timeout: <step> exceeded N s`) and is absent from the
  results.
- `--failures-csv PATH` — one row per failed measurement, with `kind` one of
  `error` (the worker reported a failure, e.g. the kernel refused the problem),
  `missing` (the worker died before reporting), `worker-exit` (it exited with an
  error after reporting every kernel), `timeout`, `batch-error`, `mismatch` (with
  `--verify`) or `bad-output`; one `unsupported` row per kernel the worker
  reported as not supporting the problem (counted as skipped, not failed); and
  one `build-failed` row per kernel that did not build, with the failed step and
  the first error line. The main CSV holds every kernel that ran, including
  those that failed `--verify` (`verified` is `False`); this file is what tells
  "could not run" apart from "never tried".
- `--repeat-top K` with `--repeat-count N` (default 5) — after the sweep, re-time
  the K fastest kernels of each problem N more times, each launch in its own
  subprocess, into `--repeat-csv` (default `<csv>.repeats.csv`). The fastest of
  many single timings is biased towards a lucky timing; selecting and reporting
  on the median of the repeats removes that bias. Kernels that failed `--verify`
  or wrote no non-zero output are not re-timed.
- `--list-pairs PATH` — dry run: expand and pair as a real run would, write every
  `(problem, kernel)` pair it would measure together with each kernel's
  configuration, and exit without building. Use it to see the size of a sweep
  before running it, or as the candidate list for choosing kernels another way.
- `--pairs PATH` — measure only the listed pairs. The file maps `"MxNxK"` to a
  list of kernel names, or to `"*"`; a `--list-pairs` file is accepted. Expansion
  still uses the full config and problems files, so the fixed-vector-width
  kernels are the same as in a full run, but only what the list needs is built.
  A listed problem or kernel the run does not have is an error. A listed kernel
  is measured only if a full run would pair it with that problem when every
  kernel builds; a listed pair that cannot be measured (not built, or not paired
  by the vector-width and tile rules) is printed and, with `--failures-csv`,
  recorded as `not-measured`. A `"*"` problem is measured exactly as a full run
  measures it, including the fixed-vector-width kernels a full run falls back to
  when a native kernel fails to build, so `"*"` entries for the problems an
  interrupted sweep did not reach complete it.

The files a run writes (`--csv`, `--failures-csv`, `--repeat-csv`) and the
`--pairs` file must all be different files; the driver refuses to start
otherwise.

### Adaptive kernel search (`--search-config`)

Instead of building and measuring every eligible `(problem, kernel)` pair, a
search run lets one strategy per problem propose the next kernel to try, builds
only the proposed kernels (each at most once per run, shared by all problems),
measures them with `--verify` and learns from each verified result. At the end
it re-measures the best kernels of each problem with fresh verified calls and
exports only the confirmed winners. The coordinator is
`tile_engine/ops/common/adaptive_search.py`; the strategies are a vendored copy
of the ROCKE search core in `dispatcher/python/ck_search` (see its `VENDOR.md`).

Recommended starting point: `tpe` with `seed_policy` `shared`, warm-started
from the heuristic's top 5 kernels per problem (`select_kernels.py
--out_pairs`, `top_k` 5):

```bash
H=../../../dispatcher/heuristics
RUN="configs/default_config.json --variant gemm_universal --arch gfx950 \
    --dtype fp16 --layout rcr --problems problems.json"
# the candidates, then the heuristic model's top 5 per problem (a --pairs file)
python gemm_full_benchmark.py $RUN --list-pairs candidates.json
python $H/select_kernels.py --candidates candidates.json \
    --model_dir $H/models/gemm_universal_fp16_gfx950 --top_k 5 --out_pairs picks.json
cat > search.json <<'EOF'
{"schema_version": 1, "strategy": "tpe", "seed": 0, "seed_policy": "shared",
 "trials_per_problem": 100, "batch_per_problem": 8, "new_builds_per_run": 1200,
 "warm_start": {"pairs_file": "picks.json", "top_k": 5}}
EOF
python gemm_full_benchmark.py $RUN --devices 0, \
    --search-config search.json --search-output-dir runs/search-001

# winners.json is a --pairs file: re-check the winners with a plain run
python gemm_full_benchmark.py $RUN --pairs runs/search-001/winners.json \
    --verify --devices 0, --csv recheck.csv
```

The warm start needs a heuristic model trained for the run's dtype and arch
(`select_kernels.py` refuses another; `dispatcher/heuristics/models` has
gfx950 `gemm_universal` fp16 and fp8 models). Any `--pairs` file of good kernels
per problem, such as an earlier run's `winners.json`, can serve instead.
Without a warm start the result depends on the seed.

Evidence: an oracle replay of this coordinator (the real driver, adapter,
coordinator and strategies; builds and GPU calls answered from an exhaustive
sweep), fp16 `rcr`, 8 problems with 5520 candidates each, the settings above.
The sweep ran on CPX compute partitions of an MI350X (gfx950), each an eighth
of the GPU, and kernel rankings there differ from the full GPU's: "best" below
means a partition's best. The replay and the GPU check below used the search
code from before the deterministic TPE patch (`VENDOR.md` patch 5), with which
a seed can take another path. The figure is the geometric mean over the
problems of the confirmed winner's speed relative to the best kernel's.
Near-ties make it depend on the timing noise, so each range covers two noise
models: one calibrated on the GPU's own search runs (a persistent offset per
problem and kernel plus the GPU's run-to-run spread; 100 seeds) and
heavy-tailed Student-t noise (60 seeds).

| Setting | Mean | p10 | Min | Runs >= 99% | Builds (8 problems) |
|---|---|---|---|---|---|
| `tpe`, `shared`, no warm start | 88% | 68% | 59-62% | 14-17% | 341-358 |
| `tpe`, `shared`, warm start top 5 | 98.9-99.2% | 97.5-98.2% | 95% | 62-72% | 356-357 |

Runs >= 99% depends most on the noise model: a third GPU calibration, which
drew the offsets of kernels the GPU never measured regardless of how close
they are to the best, gave 38% with the warm start (mean 98.6%). Its drop came
mostly from one problem, whose 91%-of-best kernel drew a large speed-up and
won half the runs, and was judged an artifact of that model.

On one CPX partition, re-measured against the best kernel known for each
problem, the seeds that missed without the warm start (seed 0: 85%, seed 1:
72%) reached 100% with it at seed 1, `shared` (every problem got its
best-known kernel), and 98.9% at seed 0, `per_problem` (worst problem 92%).

With the warm start, the best kernel among each problem's first 50, 75 and
100 trials reached about 94%, 98.5% and 99.5% on average in both noise models
(81%, 86% and 89% without it); the confirmed winner is a little lower because
noise can swap near-ties. The heuristic's own top 5 never held the best kernel
(best of 5: 79%): the warm start moves the search into good regions rather
than handing it the answer. In the replay's original, lighter noise model,
`per_problem` seeds without a warm start did not help (88%, at 616 builds),
a cap of 150 builds for the 8 problems took the warm-started `per_problem`
search from 99.4% to 91%, and the deterministic TPE patch left the `shared`
means unchanged (88.9% without and 99.6% with the warm start). The runs used
about 356 of the 1200 builds allowed: keep `new_builds_per_run` from binding
by scaling it with the number of problems (the warm-started runs built about
45 kernels per problem; the bf16 ccr GPU group, whose misaligned problems
build their own fixed-width variants, about 80).

| Setting | Default | Meaning |
|---|---|---|
| `schema_version` | required | `1` |
| `strategy` | required | `random`, `exhaustive`, `de`, `hill`, `tpe`, `gp` or `portfolio` |
| `trials_per_problem` | required | measured trials per problem, failures and warm starts included |
| `new_builds_per_run` | required | distinct kernels the run may build (failed builds count); at least `trials_per_problem` |
| `seed`, `seed_policy` | `0`, `shared` | `per_problem` gives problem `i` the seed `seed + i` |
| `deadline_s` | none | seconds after which no trial is admitted; the round in flight and the confirmation still finish, so the run can end past the deadline by their time, the round's preparations included (each up to its limit, see below). Without it the trial and build budgets bound the run |
| `batch_per_problem` | `4` | trials a problem may propose per round |
| `max_stall_rounds` | `8` | rounds in a row without a new result before the run stops |
| `build_timeout_s` | `180`, native kernels only | limit for each codegen, compile and link step of a kernel's build in the search; when given, it applies to every kernel. Without it, a round that builds a fixed-vector-width kernel keeps the build function's limits, as a full run does (300 s per step, 1200 s to compile a fixed-width kernel). A kernel whose step times out is a `compile_error`. See "Build timeout" below |
| `warm_start.pairs_file`, `top_k` | none, `5` | a `--pairs` file (e.g. from `select_kernels.py --out_pairs`): the first `top_k` listed kernels of a problem are tried first |
| `warm_start.ranked_csv`, `use_prior` | none, `false` | `select_kernels.py --out_ranked` predictions as a prior (`tpe`, `gp`, `portfolio` only); give both or neither |
| `confirm.finalists`, `confirm.fresh_calls` | `2`, `3` | the winner is the lowest median among the finalists whose fresh calls all verified |
| `measurement_policy` | `full` | the only value |
| `host_cache.dir`, `max_entry_gib`, `max_total_gib` | system temp dir, `4`, a quarter of its free space | the run's cache of verification inputs and references, in a new directory under `dir` (keep it on a local filesystem) that is removed when the run ends. An entry larger than `max_entry_gib`, one past `max_total_gib` in all, or one that would leave the filesystem less than 10% free is not stored, and its calls compute as without the cache: on a filesystem already over 90% full nothing is stored, and the run warns. Both limits count the entries still being written, so they hold with any number of workers. `max_total_gib` `0` turns it off. See "Host cache" and "Host cache space" below |

Unknown keys are an error; relative warm-start and `host_cache.dir` paths are
read from the settings file's directory.

Budget: each round, every active problem in turn proposes up to
`batch_per_problem` trials (re-queued ones first), capped by its trials left
and by the build capacity left once the distinct new kernels the round already
needs are subtracted. A kernel that is already built, or planned this round,
costs no further build, so problems that share kernels leave room for others,
and such reuse goes on after the cap is spent. A proposal that would need a
build past the cap is told `invalid` at once, without a trial (it cannot be
built in this run), so a round never plans more builds than
`new_builds_per_run` allows and the strategy moves on to other kernels. Past
the cap a strategy reaches a built kernel only by proposing it, and most of its
proposals then need a build, so the slot of each one told `invalid` goes to
the problem's best untried kernel that needs no build (`refilled` in the round
line): the one closest to the best on another problem that measured it, in
latency relative to that best. Its result is told to the strategy like any
other. A cap-bound run therefore spends its trial budgets on the kernels it
built, and stops at the cap once no problem has an untried candidate that is
built; without the refill the stall guard would end it first.
A proposal of a kernel whose build already failed, for any problem, is told
that `compile_error` again without a trial.

Option rules:

- `--search-config` and `--search-output-dir` go together, for `gemm_universal`
  only. `--list-pairs`, `--repeat-top` and `--failures-csv` are rejected; the
  search writes its own records and `--csv` is not written. `--build-timeout`
  is rejected too: the search's limit is `build_timeout_s`.
- Every measurement is verified and runs one kernel per worker, whatever
  `--verify` and `--batch-size` say, so a failure always names one kernel.
  The host side of verification (random inputs, an fp32 numpy reference
  GEMM) comes from the run's host cache (see below). A search makes these
  inputs its own way, with other values than a plain run's (see "Host
  verification memory"). Before a problem's
  first measure call the search prepares it: a worker that loads no kernel
  stores the problem's host data in the cache, outside every call's timeout,
  with at most one preparation per device at a time, within the host's
  memory and never during a round's builds. A host slower than the
  rates below then makes that round longer instead of timing a kernel out. A
  preparation runs once per problem and is stopped only after 10 times the
  problem's host allowance plus 60 s; the call's JSON file under `rounds/`
  records it (`prepared`). A call's timeout is `--kernel-timeout` plus the
  largest host allowance `ceil(2*M*N*K / 5e10 + (M+N)*K * r)` s among the
  problems measured in the same round (or the same confirmation pass): the
  reference at 50 GFLOP/s, and the inputs at a rate `r` per element that
  depends on the dtype, 75 ns for fp16, bf16 and fp32 and 400 ns for fp8 and
  bf8, whose encoding costs about 4x more on the host (`HOST_S_PER_INPUT` in
  `gemm_search.py`; a dtype without a rate gets the slowest). `run.json`
  records the run's formula. A problem without an entry pays the host side in
  full inside every call. When the cache is on but did not store the entry
  (over its caps), its preparation measured that cost on this host, and its
  calls get at least twice that times the number of devices, whose calls may
  then share the CPUs the preparation had alone (so a problem over
  `max_entry_gib` is still prepared once). When the cache is off (or the
  preparation failed), they get the allowance alone. Its rates leave about 2x
  headroom over the host they were measured on for fp8 and bf8 (177-213 ns per
  element measured, 400 ns allowed) and 1.3-2.3x for the others (33-60 ns,
  75 ns): on an MI350X (gfx950) host about 2x slower per element, the host
  side of a first fp8 256x768x874159 call used 339 s of its 395 s timeout. On
  a slower host, keep the cache on (the preparation then runs outside the
  call timeout) or raise `--kernel-timeout`.
  These host times assume the workers get the node's CPUs. Inside a CPU
  allocation smaller than the node, set `OPENBLAS_NUM_THREADS` (or
  `OMP_NUM_THREADS`) to about the CPUs per device, or raise
  `--kernel-timeout`.
  `--verify-tol`, `--kernel-timeout`, `--workers` and `--devices` apply as
  usual. Discovery measurements spread over the devices; each problem's
  confirmation calls run on one device (problem `i` on device `i mod n`), so
  its finalists are compared on the same GPU.
- The candidates of a problem are exactly the kernels a full run measures with
  it when every kernel builds (its `--list-pairs` pairs). `--pairs` and
  `--max-kernels` narrow them (recorded in `run.json`); `--pairs` also limits
  the search to the problems it lists, and is not a warm start. A listed
  problem none of whose listed kernels pair with it stays unresolved. A
  problem with fewer candidates than `trials_per_problem` is measured
  exhaustively. Fixed-vector-width kernels are candidates where a full run
  pairs them: they are the only candidates of a problem whose extents the
  native widths do not divide, and with `--tune-c-vector-width` the narrower C
  widths join every problem's list. Not a candidate: a fixed-width kernel that
  a full run measures on a problem only because its native kernel failed to
  build.

Build timeout: a round measures its kernels once all its builds have ended,
so one kernel that never builds holds the whole round for the full limit.
Per-kernel compile times of the 5520 gfx950 fp16 `rcr` native kernels of
`default_config.json`, measured with 64 compiling at once (an upper bound for
a search round, which builds fewer at a time): the median was 30 s, and every
kernel outside warp tile `4x64x16` built in at most 111 s. That one class holds
every slower build and all 118 kernels (2.1%) that did not build in 300 s. The
search's default of 180 s per step keeps 1.6x headroom over 111 s and shortens
each wait for a kernel that never builds by 120 s; the price is the 2.2% of
kernels (all `4x64x16`) that build in 180-300 s, which become
`compile_error`. The default applies to a round of native kernels only. A
round that builds a fixed-vector-width kernel keeps the build function's
limits, as a full run does: 300 s per step and 1200 s to compile a fixed-width
kernel. Fixed-width compile times were not measured, and one build call has
one limit, so such a round keeps 300 s for its native kernels too; a shorter
native limit would shorten it only when a native kernel never builds and
every fixed-width one ends sooner. Such rounds are the rule in a run with a
problem the native widths do not fit (its padded fixed-width kernels also pair
with other problems where a native tile does not divide) or with
`--tune-c-vector-width`, so those runs mostly keep a full run's limits. A
`build_timeout_s` in the settings applies to every kernel of every round.
Raise it when the slow native kernels matter, or on a host that compiles much
slower. A timed-out step leaves its compiler running until it ends (the build
function stops only `hipcc`), so a run with many timeouts loads the host for a
while longer.

Outputs, in a directory that must not hold an earlier run (a path that is, or
lies under, an existing file, or whose nearest existing directory is not
writable, is refused up front with exit status `2`):

| File | Content |
|---|---|
| `winners.json` | `{"MxNxK": [kernel]}` for the confirmed winners only (a `--pairs` file) |
| `summary.json` | status (`complete`, `partial`, `no_winner`), stop reason, each winner's confirmed median, each unresolved problem with its reason, each problem's best (unconfirmed) discovery result, counts and budget |
| `run.json` | settings, build context, sources (commit, sha256 of the config and problem files), per-problem candidates, strategy and seed |
| `trials.jsonl` | one event per trial, build, result, confirmation call, re-queue, cancel and error (with its traceback) |
| `rounds/NNNN-measure.csv`, `rounds/NNNN-confirm.csv` | the usual results CSV of each measure call; the `.json` next to it maps `problem_idx` to the request and holds the call's failure records |
| `rounds/NNNN-build.json` | the kernels of each build call, its limit per step (`null`: the build function's own) and the full error of each that failed (`trials.jsonl` keeps its first line) |
| `artifacts/<kernel>.so` | the run's copy of each built kernel; only these copies are loaded, and one that changed since its build is refused |

Exit status: `0` when every problem has a confirmed winner, `1` when any has
none or the run stopped on an error, `2` for invalid options, settings or
output directory. An error (a build or worker pool that dies, a strategy
failure) ends the search, but the finalists measured before it are still
confirmed and exported; the status is then at best `partial` and
`summary.json` holds the error. When the
adapter breaks its contract (a lost or repeated outcome), nothing is confirmed.
When a write to the output directory fails (a full disk, say), the search
ends with that error and exit status `1`, also when it is one of the last
writes: nothing more is confirmed or logged to `trials.jsonl`, but
`winners.json` and `summary.json` are still tried, and `summary.json` (when it
can be written) holds the error. `130`: the run was interrupted (Ctrl-C, or
SIGTERM such as a batch system's time limit); nothing is confirmed and
`winners.json` is empty, but `summary.json` is written, with each problem's
best discovery result. A build worker that gets SIGTERM too (a batch system
signals the whole process group) stops at once, so the summary does not wait
for queued builds. Ctrl-C stops a forked build worker the same way (Linux's
default up to Python 3.13); a worker started by forkserver (the default from
Python 3.14) or spawn turns Ctrl-C into one build's error and goes on with the
queued builds, so the summary waits for the round's builds. So does a SIGTERM
sent only to the driver's pid (`kill <pid>`).

Only a verified row of the requested kernel and problem with a finite positive
latency counts as a result. A verification failure is `numeric_fail`; a blank
`verified`, a bad latency or unreadable worker output is `invalid_measurement`;
a worker error, crash or wrong kernel identity is `run_error`; a failed build is
`compile_error` for every problem that asked for the kernel in the round that
built it (one build charged, a trial each) and for any later proposal of it
(no trial); a kernel the worker never reached is retried once. Nothing falls back to an
unconfirmed result. A search run asks the worker to check each library's
identity (`check_identity`): a library whose own name
(`GpuGemmRunner.kernel_name`, the codegen `KERNEL_NAME` that
`GemmKernelConfig.name` reproduces) is not the requested kernel is refused
with an error naming both. Without `check_identity`, `search_host_data` and
`host_cache` (every run without `--search-config`) the worker's input and
output are unchanged.

Host cache: without it, every verified call regenerates the problem's random
inputs, encodes them for the kernel and recomputes the fp32 reference in its
own worker; on a 16-core host that took 15 s per call at bf16 `rcr`
8192x8192x8192 and 4.7 s at 2048x256x49152 (7.8 s in `crr`), with the
inputs a search made before "Host verification memory". In a search,
a problem's preparation computes them as a worker would and stores the encoded
A and B, the reference and its maximum as `.npy` files: one entry per key (the
problem, dtype, layout, fp8 format and input seed), written under a temp name
and renamed into place, so concurrent workers on any device read a complete
entry or none. Its calls only load them and compare: 0.37 s and 0.06 s for
the two shapes above. Measured end to end on a full MI350X (gfx950), with
`dir` on `/dev/shm` (a tmpfs) and `max_total_gib` 12, a search's measurement
or confirmation call (worker start, host side and kernel) took 0.34 s instead
of 0.62 s on the replay's 8 fp16 `rcr` shapes (1.8x) and 5.75 s instead of
17.2 s on 4 large bf16 `crr` shapes (3.0x), with the same verify outcomes and
latencies. Both are averages over all calls of the run, each problem's first
(uncached) call included: 8 of 110 calls (62 measurements, 48 confirmation
calls) for the fp16 shapes, 4 of 32 (8 and 24) for the large ones; those runs
predate the preparation step, so each problem's first call computed and stored
its entry. The default `dir` was not measured: on that host its filesystem was
too full to store anything (see "Host cache space" below). A key has one
writer: a worker that misses it while another writes it computes its own copy
and does not store it, and each writer reserves its entry's size under a lock
before writing, so the caps and the 10%-free guard hold however many workers
miss at once. A worker killed while it writes (a preparation at its limit, or
a call that timed out) keeps its reservation, and that problem computes on
every call for the rest of the run. An entry holds what each worker would compute itself,
so verification results are the same with or without it. The directory goes
when the run ends (the run prints its entries and size), also on an error,
Ctrl-C or SIGTERM; a run killed with SIGKILL leaves its
`ck_search_host_cache-*` directory behind. Storing needs POSIX file locks
(`fcntl`); without them every call computes. Keep `dir` on a local filesystem
(a tmpfs or a local disk): on NFS a worker that waits for the lock waits out
the client's lock retry backoff, inside its call's timeout (in one test, the
last of 8 workers that missed at once waited 15 s, and the last of 16, 62 s),
so a run whose cache is on NFS starts with a `Host cache: NOTE` line.

Host cache space: an entry is the fp32 reference (4*M*N bytes) plus the
encoded A and B ((M+N)*K elements), so a large-K problem needs GiB per entry:
in bf16, 1.7 GiB for 256x768x874159 and 3.7 GiB for 98304x9600x1024, close to
the 4 GiB `max_entry_gib` default. A write must also leave 10% of the whole
filesystem of `dir` free, counting the entries still being written. On a large
disk that is already over 90% full, that guard never passes (a 1.8 TB disk at
93% has 134 GB free and would need 180 GB): nothing is stored, every call pays
the full host cost, and the run starts with a `Host cache: WARNING: stores
nothing` line (its closing `Host cache:` line then counts 0 entries). Point
`host_cache.dir` at a tmpfs such as `/dev/shm` or at another local filesystem
with room, and size `max_total_gib` and `max_entry_gib` to the run's problems;
an entry past either cap is not stored, and its problem computes on every call.
A tmpfs is faster but holds the entries in RAM, so keep `max_total_gib` within
the memory the build workers leave free.

Host verification memory: a search makes each problem's inputs in the kernel's
memory order (a column-major operand is never transposed), as float32 normals
of numpy's seeded PCG64 generator, and encodes, decodes and compares them a
block of 2^20 elements at a time (`_search_host_data` in
`run_one_gemm_kernel.py`); a worker that reads a cache entry maps it instead
of copying it. A worker that computes a problem's host data (its preparation,
or a call without an entry) then holds the encoded A and B, their fp32 values
and the fp32 reference at most. Peak RSS per input element, measured through
the real worker (`256x768xK`, `rcr`, K = 16384 to 874159, the same at `rrr`,
`crr` and `ccr`; a plain run's worker is unchanged). A worker now needs a set
number of bytes per input element plus a fixed part, so its bytes per element
fall as K grows: for fp8, 10.0 B at K = 16384, 5.6 B at 131072 and 5.1 B at
874159.

| dtype | search, before | search, now | per element, now |
|---|---|---|---|
| fp8, bf8 | 58 B | 5.0 B plus about 0.08 GB | its byte and its fp32 value |
| bf16 | 16.5 B | 6.0 B plus about 0.02 GB | 2 bytes and its fp32 value |
| fp16 | 11.5 B | 6.0 B plus about 0.02 GB | 2 bytes and its fp32 value |

Before, float64 normals took 10 B per element alone, and the fp8 and bf8
encoder held about 48 B more per element in whole-array temporaries (and ran
twice: once for the kernel's buffers, once for the reference). On a 16-core
host with numpy's pip wheels (an optimized BLAS, as in the GPU container) a
search's host data now also takes about half the time: 74-79 ns per fp8 or bf8
input element and 15 ns per fp16 or bf16 one, against 145-156 and 29-31 ns.
With a reference BLAS (a distribution's own numpy), where a product with a
transposed operand is slower, it is slower than before in `rcr` and `crr`,
whose reference reads one operand transposed: bf16 at 134M input elements took
22.8 s in `rcr` and 16.0 s in `crr` against 12.3 s, and 10.2-10.6 s in `rrr`
and `ccr`. Output-heavy problems spend their time in the reference instead:
bf16 `ccr` 16384x16384x64 takes 0.8 s with the pip wheels and 7.8 s with a
reference BLAS (`rcr`: 0.4 and 13.5 s), and 16384x32768x64 2.1 and 17.2 s
(`ccr` makes its reference a block of rows at a time). The preparation runs
outside the call timeout, so with the cache on this only lengthens a round.
Per output element a worker needs at most 16 B more: the fp32 reference, C and
its comparison. So an fp8 256x768x874159 problem (895M input elements), whose
preparation was killed at 47.6 GB of RSS on a 62 GB host, now needs about
4.6 GB (the guard below plans 5.6 GB, plus its 0.9 GB entry when the cache is
on a tmpfs). A call that reads an entry adds little memory of its own; its RSS
still shows the entry's pages it maps (about 1 B per fp8 input element, 2 B
per bf16 one), which are shared, not copies.

The host memory guard: before a problem's first call, the search estimates the
bytes a worker needs for its host data, `(M+N)*K*(e+4) + 16*M*N + 1 GiB` to
compute it (`e` the encoded element size: 1 for fp8 and bf8, 2 for fp16 and
bf16, 4 otherwise; `host_compute_bytes` in `gemm_search.py`), plus the entry
itself when the cache is on a tmpfs and has room for it (within
`max_entry_gib`), and `12*M*N + 1 GiB` to read an entry
(`host_read_bytes`). The 1 GiB covers the interpreter, one block's temporaries
and the GPU runtime: a measure worker on an MI350X (gfx950) held about 0.9 GB
besides its host data. It compares that with the memory available:
`MemAvailable` from `/proc/meminfo` (or the room left under a memory limit of
the run's cgroup and its parents, when that is less: `memory.max` or
`memory.high` in cgroup v2, `memory.limit_in_bytes` in v1), less 10% of the
total. The problem's preparation must fit, and then one call per device at
once, reading its entry or, when none was stored (the cache off or over its
caps), computing it all. With the cache on a tmpfs, the calls check also keeps
room for the entries still to come (those of the run's problems neither
stored nor skipped, each within the cache's caps, up to its free room), which
will hold RAM while later rounds' calls of this problem run. A problem that
does not fit is never measured (nor prepared, when its preparation is what
does not fit; when only its calls do not, the entry its preparation stored is
removed), and one whose preparation was killed from outside (exit code -9, the
out-of-memory killer) or ran out of memory (`MemoryError`, or numpy or its
BLAS refusing memory as it starts) is not measured either: its calls would
compute the same data. A preparation stopped at its time limit is not a
memory failure: its calls compute their own data. A skipped problem's
requests get the status `host_memory`, which is not a kernel failure, so no
strategy is told anything; the problem stops, the console prints one line
(`<problem>: skipped, host_memory: needs X GiB, Y GiB available`), the run
goes on with the other problems and ends `partial` (or `no_winner` when no
problem has a winner), exit status 1, with the reason under `unresolved` in
`summary.json` and `host_memory` in its counts. The decision is taken once per
problem, when it is first measured and after that round's builds: memory freed
later does not bring it back, so one run never retries a problem that may
exhaust the host's memory; re-run its problems on a host with more memory,
with fewer devices, or with a cache that has room for the entry. Should the
host still run out of memory, a search's workers have asked to go first: each
sets its `oom_score_adj` to 1000 (a plain run's worker does not). A measure
call whose worker is then killed from outside (exit code -9: the driver kills
one only at its timeout, which it reports as `timeout`) or raises
`MemoryError` gets `host_memory` too, with the reason (`its call's worker was
killed ...`): no kernel is blamed and its problem stops, but the problem is
not skipped, so its best kernels so far are still confirmed and it can still
get a winner. Preparations run under a lock in the temp directory
(`ck_search_host.lock`): a round's builds hold it shared and its preparations
exclusive, so no preparation runs while any search run sharing that directory
(the runs of one host, or of one container) builds or prepares; a run whose
builds or preparation wait for another run prints one `waiting for` line. A
run that cannot open the lock file prints one `NOTE: cannot open the host
lock` line and goes on without it. Within one run the preparations running at
once must fit together, with the tmpfs entries of those that ended. Not
counted: the measure calls of other runs until they show in `MemAvailable`,
and other runs' builds (tens of GB with many build workers), which can start
while this run's calls compute their host data: calls take no lock.

Sizing: a preparation never runs during builds, so it shares the memory only
with the system, other processes and a tmpfs cache's entries. On a tmpfs with
room for its entry a preparation also needs that entry, so it needs `2e+4`
bytes per input element (6 for fp8 and bf8, 8 for fp16 and bf16) plus the
fixed 1 GiB, and the largest problem the guard admits has about
`(MemAvailable - 10% of MemTotal - 1 GiB) / (2e+4)` input elements, where
`MemAvailable` already lacks the tmpfs entries (for large-K shapes, whose M*N
terms are small). On the 62 GB MI350X (gfx950) host of the field failure, with
47-51 GB available when idle and 16 GiB of entries already on the tmpfs, the
guard's budget is about 24-28 GB: a preparation of about 3.7-4.4G fp8 input
elements or 2.8-3.3G bf16 ones when the cache still has room for its entry.
When it has none, a preparation needs `e+4` bytes per element (5 for fp8, 6
for bf16): about 4.5-5.3G fp8 or 3.7-4.4G bf16 input elements there. With the
cache on a disk, a preparation needs `e+4` bytes per element and no tmpfs
entries take RAM. Keep `max_total_gib` plus the build workers' memory within
RAM, as above; a disk cache costs no RAM beyond the page cache.

Console: builds print as in a full run; the search prints a line per build and
per round (`[search] round N: ...`: proposals, admissions, builds against the
cap, measurements, active problems, elapsed time and deadline) and one title
line per measure call, named like its `rounds/` file, after two lines for the
preparation of the call's new problems (how many, their limit, then how many
were stored and the time taken), and one more per problem not stored
(`<problem>: not stored (<why>); its calls compute their own`, ending with the
tail of the worker's stderr when the preparation failed) or skipped
(`<problem>: skipped, host_memory: ...`, see "Host verification memory"). A
build or preparation that waits for another search run on the host prints
`[search] waiting for ...` first, and a run that cannot open the host lock
prints one `[search] NOTE: cannot open the host lock ...` line. Search runs
of one checkout also build one at a time: the build writes each kernel's
header, `.o` and `.so` to paths from its name alone (no arch), so a run holds
the build lock (`ck_search_build.lock` in the dispatcher's `build/`) from a
round's build until its libraries are copied into the run's `artifacts/`
(`[search] NOTE: cannot open the build lock ...` when it cannot). A plain
driver run does not take it: do not run one in the same checkout as a search.
A host cache whose filesystem is already under 10% free (see "Host cache
space") prints a `Host cache: WARNING` line before the first round, and one
on NFS a `Host cache: NOTE` line; a `host_cache.dir` the run cannot create
its cache in prints `Host cache: WARNING: cannot create it ...` and the run
goes on without the cache.

Troubleshooting: for `no_winner` or `partial`, read `unresolved` in
`summary.json` and the failing calls in `trials.jsonl`; a problem stopped by
`host_memory` was never measured, or stopped when a call ran out of host
memory (see "Host verification memory"). If nearly every candidate gets
built, the problems are choosing different kernels: lower `new_builds_per_run`
or keep the default `shared` seed policy.

### Multi-GPU parallelism

Phase 3 fans the `(kernel × problem)` work out across **every visible GPU** in
parallel. One worker thread per device pulls batches from a shared queue and
spawns a disposable subprocess pinned with `HIP_VISIBLE_DEVICES`, so an N-GPU box
benchmarks roughly N× faster while keeping per-batch fault isolation. Devices are
auto-detected (`HIP_VISIBLE_DEVICES`, then `rocm-smi`/`amd-smi`); override with
`--devices`. This supersedes the serial-GPU design inherited from grouped_conv.

### Supported surface

| Axis | Supported |
|---|---|
| dtype | `fp16`, `bf16` |
| layout | `rcr`, `rrr`, `crr`, `ccr` (row-major C only — ck_tile rejects column-major C at build) |

### Variant scope

The bridge is **one shared, variant-aware driver** (`gemm_full_benchmark.py` +
`run_one_gemm_kernel.py`), not a per-variant copy of the driver. The bridged
regular-GEMM path (`gemm_universal`) uses the op-root `configs/`; `--variant`
selects a not-yet-bridged variant's own `configs/` subdirectory.

What that means for this PR:

- **Only `gemm_universal` is wired and validated through the bridge here.** It is
  the foundation variant; the dispatcher codegen path is exercised and parity-
  checked for it alone.
- The `gemm_multi_d/`, `gemm_preshuffle/`, and `grouped_gemm/` `configs/`
  directories are **scaffolding** that follows the per-variant convention so the
  layout is ready. `--variant` will select them, but the bridge does **not** yet
  produce correct kernels for those variants on this PR — do not treat their
  presence as working support.
- Grouped GEMM and stream-K go through **separate bridge efforts** (stream-K in
  #8136, grouped GEMM on its own branch), not this PR.

### Removal note

The legacy regular-GEMM standalone build path has been **removed**, and the
`gemm_universal/` folder is gone entirely. The per-config benchmark generator and
driver (`gemm_universal_instance_builder.py`, `gemm_universal_benchmark.py`,
`gemm_universal_benchmark*.{cpp,hpp}`, and `gemm_universal/CMakeLists.txt`) no
longer exist; its sweep configs were promoted to the op-root `configs/` directory
(matching the `fmha/` and `grouped_conv/` bridge convention) and are consumed by
the bridge. Regular GEMM now runs exclusively through the Dispatcher bridge
workflow above (`gemm_full_benchmark.py` / `run_one_gemm_kernel.py`). The other
variants (`gemm_multi_d/`, `gemm_preshuffle/`, `grouped_gemm/`) still use the
shared `gemm_instance_builder.py` generator.

The build-system, build-instruction, and benchmark-execution sections below
describe that removed standalone path and are retained only as historical
reference for the non-bridged variants; the `benchmark_gemm_universal_*` targets
they mention are no longer produced.

## Build System Architecture

### Individual Kernel Compilation (New Approach)

The new tile engine benchmark system compiles each kernel configuration into a separate executable. This provides:
- Better build parallelism
- Faster incremental builds
- More targeted testing
- Easier debugging of specific configurations

Each benchmark executable follows the naming pattern:
```
benchmark_gemm_<dtype>_<layout>_<config>_<tile_sizes>
```

### Monolithic Build (Legacy Approach)

The original system compiles all kernels into a single executable (`benchmark_gemm_[Datatype]_[Layout]`), which can then be filtered at runtime using command-line arguments.

## Build Instructions

### Prerequisites
- ROCm installation
- CMake 3.16 or higher
- C++17 compatible compiler

### Basic Build

```bash
# In the root of composable kernel, create build directory
mkdir build && cd build

# Configure with specific datatypes and layouts
# Replace [Arch] with your GPU architecture (e.g., gfx90a, gfx942)
# Replace [Datatype1;Datatype2;...] with datatypes (fp8, bf8, int8, fp16, bf16, fp32, fp64)
# Replace [Layout1;Layout2;...] with layouts (rcr, rrr, crr, ccr)
../script/cmake-ck-dev.sh ../ [Arch] -DGEMM_DATATYPE="[Datatype1;Datatype2]" -DGEMM_LAYOUT="[Layout1;Layout2]"

# Build specific benchmarks
make benchmark_gemm_[Datatype1]_[Layout1] -j
```

### Configuration Options

The build system supports several configuration options:

#### Using Custom Config Files
```bash
# Method 1: CMake variable (config file must be in configs/ directory)
cmake -DGEMM_CONFIG_FILE=my_custom_config.json ...

# Method 2: Environment variable (takes precedence over CMake variable)
export GEMM_CONFIG_FILE=my_custom_config.json
cmake ...
```

#### Config File Priority Order
1. **Environment variable** `GEMM_CONFIG_FILE` (highest priority)
2. **CMake variable** `GEMM_CONFIG_FILE`
3. **Default config** (default_config.json for all layouts)

**Note**: All custom config files must be placed in the `tile_engine/ops/gemm/configs/` directory.

### Example Build Commands

```bash
# Build for gfx942 with fp8 and fp16 datatypes, rcr layout
mkdir build && cd build
../script/cmake-ck-dev.sh ../ gfx942 -DGEMM_DATATYPE="fp8;fp16" -DGEMM_LAYOUT="rcr;ccr;rrr;crr"
make benchmark_gemm_universal_fp8_rcr -j
make benchmark_gemm_universal_fp16_rcr -j
```

### Building Individual Kernels

```bash
# Build a specific kernel configuration
make benchmark_gemm_universal_fp8_rcr_compv4_default_intrawave_False_False_False_False_256x256x32_1x4x1_32x32x32

# Build all fp16 benchmarks in parallel
make -j$(nproc) $(make help | grep benchmark_gemm_fp16 | awk '{print $2}')
```

### Rebuilding After Configuration Changes

If you modify the configuration file, you must rebuild:
```bash
rm -rf tile_engine/ && make benchmark_gemm_universal_[Datatype]_[Layout] -j
```

## Running Benchmarks

### Individual Kernel Execution

```bash
cd /path/to/build/directory
./bin/benchmark_gemm_universal_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16 \
    -m=512 -n=512 -k=512 -verify=1
```

### Monolithic Executable (Legacy)

```bash
# Run specific pipeline/scheduler/epilogue combination
./bin/benchmark_gemm_universal_[Datatype]_[Layout] -pipeline=compv3 -scheduler=intrawave -epilogue=default
```

### Automated Testing

Use the provided test script to run multiple benchmarks:
```bash
cd /path/to/composable_kernel/tile_engine/ops/gemm
./test_benchmark.sh [build_directory]
```

## Configuration System

### Configuration Files

The system uses JSON configuration files to specify kernel parameters:

- `configs/default_config.json` - Default configurations for various datatypes
- `configs/user_provided_config.json` - User-customizable configurations

### Configuration Structure

```json
{
    "tile_config": {
        "tile_m": {"values": [256, 128]},
        "tile_n": {"values": [256, 128]},
        "tile_k": {"values": [64, 32]},
        "warp_m": {"values": [2, 4]},
        "warp_n": {"values": [2, 1]},
        "warp_k": {"values": [1]},
        "warp_tile_m": {"values": [32, 16]},
        "warp_tile_n": {"values": [32, 16]},
        "warp_tile_k": {"values": [16, 32]}
    },
    "trait_config": {
        "pipeline": {"values": ["compv3", "compv4", "mem"]},
        "scheduler": {"values": ["intrawave", "interwave"]},
        "epilogue": {"values": ["default", "cshuffle"]},
        "pad_m": {"values": [false]},
        "pad_n": {"values": [false]},
        "pad_k": {"values": [false]},
        "persistent": {"values": [false]}
    }
}
```

## Scripts and Tools

### Python Scripts

#### gemm_instance_builder.py
**Purpose**: Shared kernel instance generator used by the non-bridged variants
(`gemm_multi_d`, `gemm_preshuffle`, `grouped_gemm`). Creates C++ kernel
implementations based on configuration files.

> The regular-GEMM subclass `gemm_universal/gemm_universal_instance_builder.py`
> has been removed; regular GEMM now goes through the Dispatcher bridge.

**Key Features**:
- Generates individual kernel header files for separate compilation
- Supports multiple data types (fp16, fp8, bf16, fp32, fp64)
- Validates tile configurations for correctness
- Creates CMake integration files

#### gemm_instance_builder_parallel.py
**Purpose**: Parallel version of the instance builder for faster generation of multiple kernel configurations.

**Features**:
- Multi-threaded kernel generation
- Improved performance for large configuration spaces

#### validation_utils.py
**Purpose**: Provides comprehensive validation functions for kernel configurations.

**Key Functions**:
- `is_tile_config_valid()` - Validates tile dimensions and alignments
- `is_trait_combination_valid()` - Checks if pipeline/epilogue/scheduler combinations are supported
- `validate_warp_tile_combination()` - GPU-specific warp tile validation
- `validate_lds_capacity()` - Ensures configurations fit in LDS memory

**Validation Checks**:
- Dimension alignment (tile dimensions must be divisible by warp dimensions)
- LDS capacity constraints
- GPU-specific warp tile support
- Unsupported trait combinations

#### test_validation.py
**Purpose**: Test suite for the validation logic to ensure correctness.

**Usage**:
```bash
python test_validation.py
```

**Tests**:
- Warp tile combination validation
- Trait combination validation
- Full tile configuration validation

#### json_config.py
**Purpose**: Configuration file parsing and management.

**Features**:
- JSON configuration loading
- Default configuration handling
- Configuration validation

#### codegen_utils.py
**Purpose**: Utility functions for code generation.

**Features**:
- Template processing
- Code formatting utilities
- File generation helpers

### Shell Scripts

#### test_benchmark.sh
**Purpose**: Automated benchmark testing script that finds and runs all built benchmark executables.

**Features**:
- Automatic build directory detection
- Batch execution of multiple benchmarks
- CSV result collection
- Colored output for easy reading
- Example command generation

**Usage**:
```bash
# Auto-detect build directory
./test_benchmark.sh

# Specify build directory
./test_benchmark.sh /path/to/build/directory
```

**What it does**:
1. Finds all benchmark executables in the build directory
2. Runs each with multiple problem sizes (512, 1024, 2048)
3. Performs GPU verification
4. Saves results to timestamped CSV file
5. Provides summary statistics

## Command Line Options

All benchmark executables support the following options:

### Matrix Dimensions
- `-m=<value>` - M dimension (default: 3840)
- `-n=<value>` - N dimension (default: 4096)
- `-k=<value>` - K dimension (default: 2048)

### Strides
- `-stride_a=<value>` - Stride for matrix A (default: 0, auto-calculated)
- `-stride_b=<value>` - Stride for matrix B (default: 0, auto-calculated)
- `-stride_c=<value>` - Stride for matrix C (default: 0, auto-calculated)

### Verification
- `-verify=<0|1|2>` - Verification mode
  - 0: No verification (default)
  - 1: CPU verification
  - 2: GPU verification

### Performance Testing
- `-warmup=<value>` - Warmup iterations (default: 50)
- `-repeat=<value>` - Benchmark iterations (default: 100)
- `-timer=<true|false>` - Use GPU timer (default: true)
- `-flush_cache=<true|false>` - Flush cache between runs (default: true)
- `-rotating_count=<value>` - Cache rotation count (default: 1000)

### Initialization
- `-init=<0|1|2>` - Tensor initialization method
  - 0: Random values [-1, 1] (default)
  - 1: Linear sequence (i % 17)
  - 2: Constant value (1.0)

### Output Options
- `-log=<true|false>` - Enable verbose logging (default: false)
- `-metric=<0|1|2>` - Performance metric
  - 0: Latency in ms (default)
  - 1: TFLOPS
  - 2: Bandwidth in GB/s
- `-json_output=<true|false>` - JSON format output (default: false)
- `-csv_filename=<filename>` - Save results to CSV
- `-csv_format=<simple|comprehensive>` - CSV format (default: comprehensive)

### Advanced Options
- `-split_k=<value>` - Split-K factor (default: 1)
- `-structured_sparsity=<true|false>` - Enable structured sparsity (default: false)
- `-pipeline=<compv3|compv4|mem>` - Pipeline type (default: compv3)
- `-scheduler=<intrawave|interwave>` - Scheduler type (default: intrawave)
- `-epilogue=<cshuffle|default>` - Epilogue type (default: cshuffle)
- `-pad_m=<true|false>` - Pad M dimension (default: false)
- `-pad_n=<true|false>` - Pad N dimension (default: false)
- `-pad_k=<true|false>` - Pad K dimension (default: false)
- `-persistent=<true|false>` - Use persistent kernel (default: false)

## Understanding Kernel Names

The kernel naming convention encodes the configuration:

```
benchmark_gemm_universal_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16
               		 ^^^^  ^^^ ^^^^^^ ^^^^^^^ ^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^ ^^^^^^^ ^^^^^^^^^
               		 |     |   |      |       |         |                       |         |       |
               		 |     |   |      |       |         Padding & flags         |         |       Warp tile
               		 |     |   |      |       Scheduler                         |         Thread tile
               		 |     |   |      Epilogue                                  Block tile
               		 |     |   Pipeline
               		 |     Layout (Row-Column-Row)
               		 Data type
```

### Components:
- **Data type**: fp16, fp32, bf16, fp8, bf8, int8
- **Layout**: rcr (Row-Column-Row), rrr, crr, ccr
- **Pipeline**: mem, compv3, compv4
- **Epilogue**: default, cshuffle
- **Scheduler**: intrawave, interwave
- **Flags**: pad_m, pad_n, pad_k, persistent (4 boolean flags)
- **Tile sizes**: BlockTile x ThreadTile x WarpTile

## Troubleshooting

### Common Issues

1. **Kernel not found**
   - Ensure the specific benchmark executable is built
   - Check the build directory bin/ folder

2. **Verification failures**
   - Try GPU verification (-verify=2) which may be more accurate
   - Check data type compatibility
   - Verify stride calculations

3. **Build failures**
   - Check GPU architecture compatibility
   - Ensure ROCm is properly installed
   - Verify configuration file syntax

4. **Performance variations**
   - Increase warmup iterations
   - Disable CPU frequency scaling
   - Use GPU timer for accurate measurements

### Debug Options

Enable verbose logging:
```bash
./bin/benchmark_gemm_... -log=true -verify=1
```

Test validation logic:
```bash
python test_validation.py
```

## Performance Tips

1. **Optimal Problem Sizes**: Use sizes that are multiples of tile dimensions
2. **Warmup**: Use at least 50-100 warmup iterations
3. **GPU Timer**: Always use `-timer=true` for accurate measurements
4. **Cache Management**: Enable cache flushing for consistent results
5. **Thread Affinity**: Set CPU affinity to reduce variation

## Integration Examples

### Python Integration

```python
import subprocess
import json

# Run benchmark with JSON output
result = subprocess.run([
    './bin/benchmark_gemm_universal_fp16_rcr_...', 
    '-m=1024', '-n=1024', '-k=1024',
    '-json_output=true'
], capture_output=True, text=True)

# Parse results
data = json.loads(result.stdout)
print(f"Performance: {data['tflops']} TFLOPS")
```

### Batch Testing Script

```bash
#!/bin/bash
SIZES="512 1024 2048 4096"
for size in $SIZES; do
    echo "Testing ${size}x${size}x${size}"
    ./bin/benchmark_gemm_... -m=$size -n=$size -k=$size \
        -verify=2 -csv_filename=results.csv
done
```

## Contributing

When adding new features or configurations:
1. Update validation logic in `validation_utils.py`
2. Add tests to `test_validation.py`
3. Update configuration examples
4. Document new command-line options

For more information about the Composable Kernel project, visit the main repository documentation.
