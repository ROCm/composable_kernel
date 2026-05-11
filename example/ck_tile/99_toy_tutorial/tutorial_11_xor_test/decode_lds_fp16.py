#!/usr/bin/env python3
"""
Decode ROCgdb LDS dumps (0xhhhh words) into fp16 values.

Typical usage:
  1) In rocgdb:
       set logging file /tmp/lds.txt
       set logging enabled on
       x/2048hx local#(unsigned long long)p_lds
       set logging enabled off

  2) Decode:
       python3 decode_lds_fp16.py --gdb /tmp/lds.txt --rows 64 --cols 32

You can also decode a raw binary dump:
       dump binary memory /tmp/lds.bin local#ADDR local#(ADDR+4096)
       python3 decode_lds_fp16.py --bin /tmp/lds.bin --rows 64 --cols 32
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path


def u16_to_f16(value: int) -> float:
    # ROCm and x86 host are little-endian for these dumps.
    return struct.unpack("<e", value.to_bytes(2, byteorder="little", signed=False))[0]


def parse_gdb_hx_words(text: str) -> list[int]:
    words: list[int] = []
    for line in text.splitlines():
        if ":" in line:
            _, rhs = line.split(":", 1)
        else:
            rhs = line

        for match in re.findall(r"0x([0-9a-fA-F]+)", rhs):
            value = int(match, 16)
            # Keep only 16-bit words from x/...hx output.
            if 0 <= value <= 0xFFFF:
                words.append(value)
    return words


def parse_bin_words(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 2 != 0:
        raise ValueError(f"Binary size must be multiple of 2 bytes, got {len(data)}")
    return [int.from_bytes(data[i : i + 2], "little", signed=False) for i in range(0, len(data), 2)]


def print_linear(words: list[int], start: int, limit: int) -> None:
    end = min(len(words), start + limit)
    print("idx     hex     fp16")
    print("---------------------------")
    for i in range(start, end):
        w = words[i]
        f = u16_to_f16(w)
        print(f"{i:4d}  0x{w:04x}  {f:10.6f}")


def print_matrix(words: list[int], rows: int, cols: int) -> None:
    needed = rows * cols
    if len(words) < needed:
        raise ValueError(f"Need at least {needed} words for {rows}x{cols}, got {len(words)}")

    print(f"Matrix {rows}x{cols} (fp16):")
    for r in range(rows):
        row_vals = [u16_to_f16(words[r * cols + c]) for c in range(cols)]
        print(" ".join(f"{v:8.3f}" for v in row_vals))


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode LDS dump to fp16")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gdb", type=Path, help="Text file containing ROCgdb x/...hx output")
    src.add_argument("--bin", type=Path, help="Raw binary dump from 'dump binary memory'")

    parser.add_argument("--rows", type=int, default=0, help="Optional matrix rows")
    parser.add_argument("--cols", type=int, default=0, help="Optional matrix cols")
    parser.add_argument("--start", type=int, default=0, help="Start index for linear print")
    parser.add_argument("--limit", type=int, default=256, help="How many words to print in linear mode")
    args = parser.parse_args()

    if args.gdb:
        words = parse_gdb_hx_words(args.gdb.read_text(encoding="utf-8", errors="ignore"))
    else:
        words = parse_bin_words(args.bin)

    print(f"Parsed {len(words)} x u16 words")
    if not words:
        print("No words parsed. Check that your file contains x/...hx output.")
        return 1

    if (args.rows > 0) ^ (args.cols > 0):
        raise ValueError("Provide both --rows and --cols, or neither.")

    if args.rows > 0 and args.cols > 0:
        print_matrix(words, args.rows, args.cols)
    else:
        print_linear(words, args.start, args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
