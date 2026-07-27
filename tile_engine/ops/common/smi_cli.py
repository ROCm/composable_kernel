#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CLI entry point for shell ck_smi_* delegates."""

import argparse
import sys

from smi_utils import (
    check_gpu_available,
    count_gpus,
    detect_gpu_ids,
    show_gpu_info,
    show_version,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CK GPU SMI wrapper CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-ids", help="Print GPU ids one per line")

    sub.add_parser("count", help="Print GPU count")

    show_info = sub.add_parser("show-info", help="Print GPU product/static info")
    show_info.add_argument(
        "--head",
        type=int,
        default=10,
        help="Max lines to print (0 for full output)",
    )

    sub.add_parser("check", help="Exit 0 if GPU is accessible via SMI")

    sub.add_parser("show-version", help="Print driver version info")

    args = parser.parse_args()

    if args.command == "list-ids":
        for gpu_id in detect_gpu_ids():
            print(gpu_id)
        return 0

    if args.command == "count":
        print(count_gpus())
        return 0

    if args.command == "show-info":
        head = None if args.head == 0 else args.head
        print(show_gpu_info(head=head))
        return 0

    if args.command == "check":
        return 0 if check_gpu_available() else 1

    if args.command == "show-version":
        print(show_version())
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
