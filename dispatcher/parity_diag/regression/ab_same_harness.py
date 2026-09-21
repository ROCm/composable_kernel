#!/usr/bin/env python3
"""Apples-to-apples GEMM A/B: bridge kernel vs old-TE kernel, ONE harness.

Why this exists
---------------
The earlier sweep (allsweep6144rcrfp16.py) compared the bridge's dispatcher
measurement against old TE's *standalone benchmark binary*
(benchmark_gemm_universal_<stem>). That comparison is NOT apples-to-apples:
the device kernel is byte-identical, yet old TE's standalone binary reports
~18-20% lower TFLOPS at e.g. 1024^3 / compv4. rocprof shows the identical
kernel genuinely runs longer in that process -- ~+8% cycles plus a lower
sustained SCLK -- a power/clock + execution-environment artifact of that
binary, NOT a bridge speedup, compiler difference, or kernel difference.
(See diagnose.md sec.4.)

This harness removes the artifact: it builds the OLD-TE kernel into a .so from
old TE's own generated header and runs BOTH the bridge kernel and the old-TE
kernel through the SAME worker (run_one_gemm_kernel.py). Measured this way the
gap collapses to ~1%, which is the honest result.

The old-TE generated-header directory is derived per stem as
``<OLD_TE_GEN_BASE>/<dtype>/<layout>/`` (e.g. fp16/rcr, bf16/crr), so a single
run covers every dtype/layout. Set OLD_TE_GEN to pin one explicit leaf dir for
all stems (legacy behavior); set OLD_TE_GEN_BASE to relocate the base.

Usage:
  python3 ab_same_harness.py                       # default kernel list + shapes
  python3 ab_same_harness.py <stem> [<stem>...]    # explicit stems
  python3 ab_same_harness.py --stems-file F [--csv OUT]   # sweep a stems file
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# composablekernel root: .../composablekernel/dispatcher/parity_diag/regression/<this>
ROOT = Path(__file__).resolve().parents[3]
DISP = ROOT / "dispatcher"
GEN = DISP / "build" / "generated_kernels"
SRC = DISP / "bindings" / "ctypes" / "gemm_ctypes_lib.cpp"
STATIC = DISP / "build" / "libck_tile_dispatcher.a"
BR_SO_DIR = DISP / "build" / "examples"
WORKER = ROOT / "tile_engine/ops/gemm/run_one_gemm_kernel.py"
# Base dir of old-TE generated single-kernel headers; the per-stem leaf
# (<dtype>/<layout>) is appended in old_gen_dir(). Points at a sibling
# develop-parity worktree under the rocm-libraries root by default.
OLD_GEN_BASE = Path(os.environ.get(
    "OLD_TE_GEN_BASE",
    str(ROOT.parents[1] / ".claude/worktrees/develop-parity"
        "/projects/composablekernel/build/tile_engine/ops/gemm/gemm_universal"),
))
# Legacy explicit override: when set, this exact leaf dir is used for ALL stems.
OLD_GEN_PIN = os.environ.get("OLD_TE_GEN")
OUT = DISP / "parity_diag" / "regression" / "_ab_same_harness_build"
ARCH = os.environ.get("GFX_ARCH", "gfx942")
DEVICE = os.environ.get("PARITY_DEVICE", "0")
REPEATS = int(os.environ.get("AB_REPEATS", "3"))

SHAPES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
          (1024, 512, 256), (4096, 4096, 4096)]

DEFAULT_STEMS = [
    "fp16_rcr_compv4_default_intrawave_False_False_False_False_64x128x64_2x2x1_32x32x16",
    "fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_64x128x64_1x4x1_32x32x16",
    "fp16_rcr_compv4_default_intrawave_False_False_False_False_128x128x64_4x1x1_32x32x16",
]

PYPATH = os.pathsep.join([str(DISP / "python"), str(ROOT / "tile_engine/ops/gemm")])


def old_gen_dir(stem: str) -> Path:
    """Old-TE header dir for a stem: <base>/<dtype>/<layout> (or the pinned dir).

    Stems are named ``<dtype>_<layout>_...`` (e.g. fp16_rcr_..., bf16_crr_...),
    which is exactly the develop-parity gen-tree layout, so the leaf is derived
    from the stem itself -- no per-layout hardcoding.
    """
    if OLD_GEN_PIN:
        return Path(OLD_GEN_PIN)
    parts = stem.split("_")
    dtype, layout = parts[0], parts[1]
    return OLD_GEN_BASE / dtype / layout


def build_old_so(stem: str) -> Path | None:
    """Compile old TE's generated kernel header into a bridge-loadable .so.

    Cached: if the .so already exists it is reused, so a parallel --build-only
    pre-pass (CPU-bound hipcc) can be separated from the serial GPU measurement.
    """
    hdr = old_gen_dir(stem) / f"gemm_universal_single_{stem}.hpp"
    if not hdr.exists():
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    obj = OUT / f"{stem}.o"
    lib = OUT / f"libold_{stem}.so"
    if lib.exists():
        return lib
    common = [
        "-fPIC", "-O3",
        f"-I{DISP / 'include'}", f"-I{ROOT / 'include'}", f"-I{ROOT}", f"-I{GEN}",
        "-DCK_TILE_SINGLE_KERNEL_INCLUDE", f"-include{hdr}", "-D__HIP_PLATFORM_AMD__",
        f"--offload-arch={ARCH}", f'-DGFX_ARCH="{ARCH}"',
        # Match the bridge build's AMDGPU codegen flags (gemm_utils.py
        # _build_compile_jobs / _TILE_ENGINE_CODEGEN_FLAGS), which are also what
        # Tile Engine's own CMake passes. Without these the old-TE side is built
        # with a *different* instruction schedule (notably -enable-post-misched
        # defaults back on) and runs ~10-40% faster than real old-TE, making the
        # bridge look regressed when it is actually at parity. Build BOTH sides
        # identically so the A/B measures the kernel, not a flag asymmetry.
        "-mllvm", "-enable-noalias-to-md-conversion=0",
        "-mllvm", "--lsr-drop-solution=1",
        "-mllvm", "-enable-post-misched=0",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        "-fno-offload-uniform-block",
        "-Wno-undefined-func-template", "-Wno-float-equal",
    ]
    cc = subprocess.run(["/opt/rocm/bin/hipcc", "-c", *common, str(SRC), "-o", str(obj)],
                        capture_output=True)
    if cc.returncode != 0:
        return None
    ln = subprocess.run(["/opt/rocm/bin/hipcc", "-shared", "-fPIC",
                         f"--offload-arch={ARCH}", "--hip-link",
                         str(obj), str(STATIC), "-o", str(lib)], capture_output=True)
    return lib if ln.returncode == 0 else None


def meas(so: Path, M: int, N: int, K: int) -> float | None:
    """Median TFLOPS over REPEATS worker calls (each call does its own
    warmup=50/repeat=100 internally). Median, not max, to match the sweep
    methodology and stay robust to the occasional clock-warmup outlier."""
    if not so or not Path(so).exists():
        return None
    payload = json.dumps({"so_path": str(so), "problem": {"M": M, "N": N, "K": K},
                          "kernel_name": "x"})
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = DEVICE
    env["GEMM_PYPATH"] = PYPATH
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    samples = []
    for _ in range(REPEATS):
        p = subprocess.run([sys.executable, str(WORKER)], input=payload.encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        for line in p.stdout.decode().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("ok"):
                samples.append(d["tflops"])
    return statistics.median(samples) if samples else None


def meas_all(so: Path) -> dict:
    """Median TFLOPS per shape from REPEATS *batched* worker calls.

    One worker call measures ALL shapes (5x fewer python+numpy+CDLL startups
    than per-shape meas()), which is the throughput lever for a full sweep on a
    single GPU. Returns {shape_str: tflops|None}."""
    out = {f"{M}x{N}x{K}": None for (M, N, K) in SHAPES}
    if not so or not Path(so).exists():
        return out
    items = [{"so_path": str(so), "problem": {"M": M, "N": N, "K": K},
              "kernel_name": "x"} for (M, N, K) in SHAPES]
    payload = json.dumps({"items": items, "verify": False})
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = DEVICE
    env["GEMM_PYPATH"] = PYPATH
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    samples = {s: [] for s in out}
    for _ in range(REPEATS):
        p = subprocess.run([sys.executable, str(WORKER)], input=payload.encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           env=env, timeout=900)
        for line in p.stdout.decode().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = d.get("idx")
            if isinstance(idx, int) and 0 <= idx < len(SHAPES) and d.get("ok"):
                M, N, K = SHAPES[idx]
                samples[f"{M}x{N}x{K}"].append(d["tflops"])
    for s, xs in samples.items():
        if xs:
            out[s] = statistics.median(xs)
    return out


def pipeline_of(stem: str) -> str:
    for p in ("compv3", "compv4", "mem"):
        if f"_{p}_" in stem:
            return p
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stems", nargs="*", help="kernel stems to A/B")
    ap.add_argument("--stems-file", help="file with one stem per line")
    ap.add_argument("--csv", help="write results to CSV (resume-aware)")
    ap.add_argument("--build-only", action="store_true",
                    help="parallel-compile old-TE .so for all stems, then exit "
                         "(CPU pre-pass; GPU measurement reuses the cache)")
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 8, 16),
                    help="parallel compile jobs for --build-only")
    args = ap.parse_args()

    stems = list(args.stems)
    if args.stems_file:
        stems += [l.strip() for l in Path(args.stems_file).read_text().splitlines()
                  if l.strip()]
    stems = stems or DEFAULT_STEMS

    # Parallel CPU pre-compile of every old-TE .so (no GPU touched).
    if args.build_only:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ok = miss = fail = 0
        print(f"build-only: {len(stems)} stems, jobs={args.jobs}", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(build_old_so, s): s for s in stems}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    r = fut.result()
                except Exception:
                    r = None
                s = futs[fut]
                if r is None:
                    # distinguish "no header" from "compile failed"
                    if (old_gen_dir(s) / f"gemm_universal_single_{s}.hpp").exists():
                        fail += 1
                    else:
                        miss += 1
                else:
                    ok += 1
                if i % 100 == 0:
                    print(f"  [{i}/{len(stems)}] ok={ok} no_header={miss} fail={fail}",
                          flush=True)
        print(f"build-only DONE: ok={ok} no_header={miss} fail={fail}", flush=True)
        return

    # CSV sweep mode: same columns as the (now-corrected) sweep, resume-aware.
    if args.csv:
        fields = ["stem", "pipeline", "dtype", "layout", "shape",
                  "bridge_tflops", "old_tflops", "gap_pct", "oldte_built"]
        out = Path(args.csv)
        done = set()
        if out.exists():
            with open(out) as f:
                for row in csv.DictReader(f):
                    done.add((row["stem"], row["shape"]))
        mode = "a" if done else "w"
        print(f"stems={len(stems)} shapes={len(SHAPES)} resume={len(done)} -> {out}",
              flush=True)
        with open(out, mode, newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if mode == "w":
                w.writeheader()
            for stem in stems:
                todo = [(M, N, K) for (M, N, K) in SHAPES
                        if (stem, f"{M}x{N}x{K}") not in done]
                if not todo:
                    continue
                parts = stem.split("_")
                dtype, layout = parts[0], parts[1]
                old_so = build_old_so(stem)
                br_so = BR_SO_DIR / f"libgemm_{stem}.so"
                # Batched: one worker call per side covers all shapes.
                bridge = meas_all(br_so)
                old = meas_all(old_so) if old_so else {}
                for (M, N, K) in todo:
                    shape = f"{M}x{N}x{K}"
                    b = bridge.get(shape)
                    o = old.get(shape)
                    gap = (b - o) / o * 100 if (b and o) else float("nan")
                    w.writerow(dict(
                        stem=stem, pipeline=pipeline_of(stem), dtype=dtype,
                        layout=layout, shape=shape,
                        bridge_tflops=f"{b:.4f}" if b is not None else "nan",
                        old_tflops=f"{o:.4f}" if o is not None else "nan",
                        gap_pct=f"{gap:.4f}" if gap == gap else "nan",
                        oldte_built=str(old_so is not None)))
                    fh.flush()
                print(f"  done {stem[:60]}", flush=True)
        print(f"DONE -> {out}", flush=True)
        return

    # Pretty-print mode.
    print(f"{'shape':>14} {'bridge':>9} {'oldTE':>9} {'gap%':>7}  kernel")
    for stem in stems:
        old_so = build_old_so(stem)
        br_so = BR_SO_DIR / f"libgemm_{stem}.so"
        if old_so is None:
            print(f"  [skip: no old-TE header] {stem}")
            continue
        for (M, N, K) in SHAPES:
            b = meas(br_so, M, N, K)
            o = meas(old_so, M, N, K)
            gap = (b - o) / o * 100 if (b and o) else float("nan")
            print(f"{f'{M}x{N}x{K}':>14} {b or float('nan'):9.2f} "
                  f"{o or float('nan'):9.2f} {gap:7.2f}  {stem[:40]}")


if __name__ == "__main__":
    main()
