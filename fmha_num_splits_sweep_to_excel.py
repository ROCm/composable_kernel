#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


LINE_RE = re.compile(
    r"^(?P<kernel>[^,]+),\s*(?P<time>[0-9]*\.?[0-9]+)\s*ms,\s*"
    r"(?P<tflops>[0-9]*\.?[0-9]+)\s*TFlops,\s*(?P<gbps>[0-9]*\.?[0-9]+)\s*GB/s\s*$"
)

# Example captured part:
# _b32x256x32x128x32x128_r2x1x1_r2x1x1_w16x16x16_w16x16x16
PART_RE = re.compile(
    r"(_b\d+x\d+x\d+x\d+x\d+x\d+_r\d+x\d+x\d+_r\d+x\d+x\d+_w\d+x\d+x\d+_w\d+x\d+x\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tile_example_fmha_fwd with run_all_kernels=1 for num_splits=1..32 "
            "and export ranked results (CSV by default, optional Excel)."
        )
    )
    parser.add_argument(
        "--exe",
        default="./build/bin/tile_example_fmha_fwd",
        help="Path to tile_example_fmha_fwd executable (default: ./bin/tile_example_fmha_fwd)",
    )
    parser.add_argument(
        "--out",
        default="fmha_num_splits_1_32_ranked.csv",
        help="Output CSV file path (default: fmha_num_splits_1_32_ranked.csv)",
    )
    parser.add_argument(
        "--excel-out",
        default="",
        help="Optional Excel output path (.xlsx). Empty means CSV-only.",
    )
    parser.add_argument(
        "--min-split",
        type=int,
        default=1,
        help="Minimum num_splits value (default: 1)",
    )
    parser.add_argument(
        "--max-split",
        type=int,
        default=32,
        help="Maximum num_splits value (default: 32)",
    )
    return parser.parse_args()


def build_base_command() -> list[str]:
    # User-requested fixed config
    return [
        "-b=130",
        "-h=2",
        "-h_k=2",
        "-s=32",
        "-s_k=2048",
        "-d=128",
        "-d_v=128",
        "-prec=bf16",
        "-mode=0",
        "-iperm=1",
        "-operm=1",
        "-bias=n",
        "-mask=0",
        "-lse=0",
        "-p_drop=0",
        "-vlayout=r",
        "-kname=1",
        "-v=0",
        "-warmup=50",
        "-repeat=200",
        "-run_all_kernels=1",
    ]


def run_sweep(exe: str, min_split: int, max_split: int) -> tuple[list[dict], list[tuple[int, int, str]]]:
    rows: list[dict] = []
    errors: list[tuple[int, int, str]] = []
    base = build_base_command()

    for num_splits in range(min_split, max_split + 1):
        cmd = [exe, *base, f"-num_splits={num_splits}"]
        print(f"[run] num_splits={num_splits}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)

        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            errors.append((num_splits, proc.returncode, output[-4000:]))
            continue

        matched_any = False
        for raw_line in output.splitlines():
            line = raw_line.strip()
            m = LINE_RE.match(line)
            if not m:
                continue
            matched_any = True
            kernel_name = m.group("kernel").strip()
            part_match = PART_RE.search(kernel_name)
            kernel_part = part_match.group(1) if part_match else ""

            rows.append(
                {
                    "kernel_name": kernel_name,
                    "kernel_part": kernel_part,
                    "num_splits": num_splits,
                    "time_ms": float(m.group("time")),
                    "tflops": float(m.group("tflops")),
                    "gbps": float(m.group("gbps")),
                }
            )

        if not matched_any:
            errors.append((num_splits, proc.returncode, "No run_all result lines parsed.\n" + output[-4000:]))

        if num_splits < max_split:
            time.sleep(4)

    rows.sort(key=lambda r: r["time_ms"])
    return rows, errors


def write_excel(out_path: Path, rows: list[dict], errors: list[tuple[int, int, str]]) -> None:
    try:
        from openpyxl import Workbook
    except ModuleNotFoundError:
        print(
            "ERROR: openpyxl is not installed in this Python environment.\n"
            "Install it with: /usr/bin/python -m pip install --user openpyxl",
            file=sys.stderr,
        )
        raise

    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(["kernel name", "kernel part", "num_splits", "time", "tflops", "GB/s"])

    for r in rows:
        ws.append([r["kernel_name"], r["kernel_part"], r["num_splits"], r["time_ms"], r["tflops"], r["gbps"]])

    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"

    err_ws = wb.create_sheet("errors")
    err_ws.append(["num_splits", "exit_code", "details"])
    for num_splits, code, details in errors:
        err_ws.append([num_splits, code, details])

    wb.save(out_path)


def write_csv(out_path: Path, rows: list[dict]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel name", "kernel part", "num_splits", "time", "tflops", "GB/s"])
        for r in rows:
            writer.writerow([r["kernel_name"], r["kernel_part"], r["num_splits"], r["time_ms"], r["tflops"], r["gbps"]])


def main() -> int:
    args = parse_args()

    if args.min_split < 1 or args.max_split < args.min_split:
        print("Invalid split range", file=sys.stderr)
        return 2

    out_path = Path(args.out).resolve()
    excel_out = args.excel_out.strip()
    rows, errors = run_sweep(args.exe, args.min_split, args.max_split)

    write_csv(out_path, rows)
    print(f"Wrote CSV: {out_path}")

    if not excel_out:
        print(f"Rows (kernel results): {len(rows)}")
        print(f"Splits with errors: {len(errors)}")
        return 0

    excel_path = Path(excel_out).resolve()

    try:
        write_excel(excel_path, rows, errors)
    except ModuleNotFoundError:
        print("Excel file skipped due to missing openpyxl.", file=sys.stderr)
        return 0

    print(f"Wrote Excel: {excel_path}")
    print(f"Rows (kernel results): {len(rows)}")
    print(f"Splits with errors: {len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
