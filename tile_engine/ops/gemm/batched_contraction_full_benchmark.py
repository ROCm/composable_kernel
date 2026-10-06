#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""3-phase driver for the batched-contraction TileEngine -> Dispatcher bridge.

Phase 1  expand a tile/trait sweep (configs/*.json) into kernel configs.
Phase 2  codegen + hipcc-compile every config into a .so (parallel).
Phase 3  fan out one disposable per-GPU worker per kernel, optionally --verify
         against an fp32 numpy reference, and collect timings.

Bridge kernel name == TE config name == codegen KERNEL_NAME (byte-exact), which is
the verification anchor: we know we benchmarked exactly the requested config.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_DISP = _HERE.parents[3] / "dispatcher"
sys.path.insert(0, str(_DISP / "python"))

from batched_contraction_utils import (  # noqa: E402
    BatchedContractionProblem,
    expand_sweep,
    setup_multiple_batched_contraction_dispatchers,
)

_WORKER = _HERE.parent / "run_one_batched_contraction_kernel.py"


def _load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description="Batched-contraction bridge benchmark")
    ap.add_argument("--config", type=Path,
                    default=_HERE.parent / "batched_contraction" / "configs" / "bridge_default_ci_config.json",
                    help="TE sweep config JSON (default: bridge_default_ci_config.json; on "
                    "gfx1250/MI400 pass bridge_default_ci_config_gfx1250.json for WMMA 16x16x32)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--layout", default="rcr")
    # Leave arch unset by default so setup resolves the visible GPU architecture
    # via rocminfo; hardcoding gfx942 would compile the wrong ISA on gfx950/gfx1250.
    ap.add_argument("--arch", default=None,
                    help="GPU arch (gfx90a/gfx942/gfx950/gfx1250); default: auto-detect via rocminfo.")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/bc_bridge_bench"))
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-tol", type=float, default=2e-2)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap #kernels (0=all)")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    kernels = expand_sweep(cfg, dtype=args.dtype, layout=args.layout)
    if args.limit:
        kernels = kernels[: args.limit]
    print(f"[phase1] {len(kernels)} kernel configs")

    so_paths = setup_multiple_batched_contraction_dispatchers(
        kernels, output_dir=args.output_dir, gfx_arch=args.arch)
    built = [(k, p) for k, p in zip(kernels, so_paths) if p]
    print(f"[phase2] built {len(built)}/{len(kernels)} .so")

    prob = BatchedContractionProblem(
        g_dims=[args.g], m_dims=[args.m], n_dims=[args.n], k_dims=[args.k])

    rows = []
    for k, so in built:
        job = {"so_path": str(so), "kernel_name": k.name, "dtype": args.dtype,
               "problem": prob.to_dict(), "verify": args.verify, "verify_tol": args.verify_tol,
               "num_d_tensors": int(getattr(k, "num_d_tensors", 0)),
               "elementwise": getattr(k, "elementwise", "PassThrough")}
        r = subprocess.run([sys.executable, str(_WORKER)], input=json.dumps(job),
                           capture_output=True, text=True, timeout=600)
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            out = {"status": f"worker_crash: {r.stderr[-200:]}", "kernel_name": k.name}
        rows.append(out)
        flag = out.get("status")
        mr = out.get("max_rel")
        print(f"[phase3] {k.name}  t={out.get('time_ms')}  max_rel={mr}  {flag}"
              + ("" if out.get("name_match") in (None, True) else "  NAME_MISMATCH"))

    npass = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[done] {npass}/{len(rows)} kernels ok")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kernel_name", "time_ms", "max_rel", "status", "name_match"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        print(f"[csv] wrote {args.csv}")
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
