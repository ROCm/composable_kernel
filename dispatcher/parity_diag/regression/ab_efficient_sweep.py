#!/usr/bin/env python3
"""Efficient A/B sweep: bridge .so vs Old-TE binary, all layouts + fp16/bf16.

Faster successor to run_alllayout_sweep.py: the bridge side batches all shapes
for a stem into ONE run_one_gemm_kernel.py worker call (one Python+numpy+CDLL
startup per stem instead of one per measurement). Old-TE binaries are run once
per shape; their internal warmup=50/repeat=100 already yields a stable median,
matching the prior methodology.

- Bridge .so : main worktree dispatcher/build/examples (built from the FIXED source).
- Old-TE bin : develop-parity worktree build/bin (develop branch), per user instruction.

Writes allresult_fp16_bf16.csv with resume support (keyed on stem,shape).

CSV fields: stem,pipeline,dtype,layout,shape,bridge_tflops,old_tflops,gap_pct,
            bridge_verified,oldte_built
"""
import csv, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT     = Path("/home/AMD/muozturk/New_project/rocm-libraries/projects/composablekernel")
DISP     = ROOT / "dispatcher"
WORKER   = ROOT / "tile_engine/ops/gemm/run_one_gemm_kernel.py"
SO_DIR   = DISP / "build" / "examples"
GEN_DIR  = DISP / "build" / "generated_kernels"
OLD_BIN_DIR = Path(
    "/home/AMD/muozturk/New_project/rocm-libraries/.claude/worktrees"
    "/develop-parity/projects/composablekernel/build/bin"
)
REG      = DISP / "parity_diag" / "regression"
STEMS_FILE = REG / "stems_selected.txt"
CSV_OUT    = REG / "allresult_fp16_bf16.csv"

PYPATH  = os.pathsep.join([str(DISP / "python"), str(ROOT / "tile_engine/ops/gemm")])
DEVICE  = os.environ.get("PARITY_DEVICE", "0")

SHAPES  = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
           (1024, 512, 256), (4096, 4096, 4096)]

FIELDS  = ["stem", "pipeline", "dtype", "layout", "shape",
           "bridge_tflops", "old_tflops", "gap_pct",
           "bridge_verified", "oldte_built"]

_TFLOPS_RE = re.compile(r'"tflops\(TFlops\)":\s*([0-9.]+)')


def pipeline_of(stem):
    for p in ("compv3", "compv4", "mem"):
        if f"_{p}_" in stem:
            return p
    return "other"


def base_env():
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = DEVICE
    env["GEMM_PYPATH"] = PYPATH
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env


def run_bridge_all(stem):
    """One batched worker call over all SHAPES. Returns {shape_str: tflops|None}."""
    so = SO_DIR / f"libgemm_{stem}.so"
    out = {f"{M}x{N}x{K}": None for (M, N, K) in SHAPES}
    if not so.exists():
        return out
    # Staleness guard: a .so older than its generated header was built from an
    # obsolete codegen and must NOT be measured -- doing so reports phantom
    # regressions (the big 256-tile gaps in allresult_fp16_bf16_2.csv were all
    # stale binaries that recovered to parity on rebuild). Treat stale as missing.
    hdr = GEN_DIR / f"gemm_{stem}.hpp"
    if hdr.exists() and so.stat().st_mtime < hdr.stat().st_mtime:
        print(f"  STALE .so (older than header), skipping: {stem}", file=sys.stderr, flush=True)
        return out
    items = [{"so_path": str(so), "problem": {"M": M, "N": N, "K": K},
              "kernel_name": f"gemm_{stem}"} for (M, N, K) in SHAPES]
    payload = json.dumps({"items": items, "verify": False})
    try:
        p = subprocess.run([sys.executable, str(WORKER)], input=payload.encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           env=base_env(), timeout=900)
    except subprocess.TimeoutExpired:
        return out
    for line in p.stdout.decode().strip().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        idx = d.get("idx")
        if isinstance(idx, int) and 0 <= idx < len(SHAPES) and d.get("ok"):
            M, N, K = SHAPES[idx]
            out[f"{M}x{N}x{K}"] = d.get("tflops")
    return out


def run_oldte(stem, M, N, K):
    binp = OLD_BIN_DIR / f"benchmark_gemm_universal_{stem}"
    if not binp.exists():
        return None
    try:
        p = subprocess.run([str(binp), f"-m={M}", f"-n={N}", f"-k={K}",
                            "-warmup=50", "-repeat=100"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           env=base_env(), timeout=300)
    except subprocess.TimeoutExpired:
        return None
    m = _TFLOPS_RE.search(p.stdout.decode())
    return float(m.group(1)) if m else None


def main():
    stems = [l.strip() for l in STEMS_FILE.read_text().splitlines() if l.strip()]
    total = len(stems) * len(SHAPES)
    done = set()
    if CSV_OUT.exists():
        with open(CSV_OUT) as f:
            for row in csv.DictReader(f):
                done.add((row["stem"], row["shape"]))
    mode = "a" if done else "w"
    print(f"stems={len(stems)} shapes={len(SHAPES)} total={total} resume={len(done)}", flush=True)

    t0 = time.time(); n = len(done)
    with open(CSV_OUT, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()
        for stem in stems:
            shapes_todo = [(M, N, K) for (M, N, K) in SHAPES
                           if (stem, f"{M}x{N}x{K}") not in done]
            if not shapes_todo:
                continue
            parts = stem.split("_")
            dtype, layout = parts[0], parts[1]
            pipeline = pipeline_of(stem)
            oldte_built = (OLD_BIN_DIR / f"benchmark_gemm_universal_{stem}").exists()

            bridge = run_bridge_all(stem)
            for (M, N, K) in shapes_todo:
                shape = f"{M}x{N}x{K}"
                bt = bridge.get(shape)
                ot = run_oldte(stem, M, N, K)
                if bt is not None and ot not in (None, 0):
                    gap = (bt - ot) / ot * 100.0
                else:
                    gap = float("nan")
                w.writerow(dict(
                    stem=stem, pipeline=pipeline, dtype=dtype, layout=layout, shape=shape,
                    bridge_tflops=f"{bt:.4f}" if bt is not None else "nan",
                    old_tflops=f"{ot:.4f}" if ot is not None else "nan",
                    gap_pct=f"{gap:.4f}" if gap == gap else "nan",
                    bridge_verified="None", oldte_built=str(oldte_built)))
                fh.flush()
                n += 1
            el = time.time() - t0
            rate = (n - len(done)) / el if el > 0 else 0
            eta = (total - n) / rate / 3600 if rate > 0 else 0
            print(f"[{n}/{total}] {stem[:48]:48} rate={rate:.1f}/s ETA={eta:.1f}h", flush=True)
    print(f"DONE rows={n} -> {CSV_OUT}", flush=True)


if __name__ == "__main__":
    main()
