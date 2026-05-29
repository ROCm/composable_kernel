#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate test .cpp files from KernelTypes definitions in
test_gemm_streamk_types.hpp.

Two modes:
  --list_files FILE   Write the list of output file paths to FILE (one per line)
                      without generating the files.  Used at CMake configure time.
  --gen_files         Actually emit the .cpp files into --output_dir.
                      Used at build time via add_custom_command.

Target selection (--target):
  extended          Kernel types containing 'Atomic' or 'Pipelines'
                    -> includes test_gemm_streamk_extended_cases.inc
  atomic_smoke      Kernel types containing 'Atomic' (not 'Pipelines')
                    -> includes test_gemm_streamk_atomic_cases.inc
  linear_smoke      Kernel types containing 'Linear' (not 'Pipelines')
                    -> includes test_gemm_streamk_reduction_cases.inc
  tree_smoke        Kernel types containing 'Tree' (not 'Pipelines')
                    -> includes test_gemm_streamk_reduction_cases.inc
  pipelines_smoke   Kernel types matching 'Pipelines'
                    -> includes test_gemm_streamk_reduction_cases.inc
                       and test_gemm_streamk_atomic_cases.inc
"""

import argparse
import os
import re
import sys

# --------------------------------------------------------------------------- #
# Template for every generated .cpp file
# --------------------------------------------------------------------------- #
CPP_TEMPLATE = """\
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class {class_name} : public TestCkTileStreamK<Tuple>
{{
}};

#define TEST_SUITE_NAME {class_name}

TYPED_TEST_SUITE({class_name}, {type_alias});

{inc_includes}

#undef TEST_SUITE_NAME
"""

# --------------------------------------------------------------------------- #
# Target definitions: filter predicate and .inc files
# --------------------------------------------------------------------------- #
TARGETS = {
    "extended": {
        "filter": lambda suffix: "Atomic" in suffix or suffix == "Pipelines",
        "inc_files": ["test_gemm_streamk_extended_cases.inc"],
    },
    "atomic_smoke": {
        "filter": lambda suffix: "Atomic" in suffix and suffix != "Pipelines",
        "inc_files": ["test_gemm_streamk_atomic_cases.inc"],
    },
    "linear_smoke": {
        "filter": lambda suffix: "Linear" in suffix and suffix != "Pipelines",
        "inc_files": ["test_gemm_streamk_reduction_cases.inc"],
    },
    "tree_smoke": {
        "filter": lambda suffix: "Tree" in suffix and suffix != "Pipelines",
        "inc_files": ["test_gemm_streamk_reduction_cases.inc"],
    },
    "pipelines_smoke": {
        "filter": lambda suffix: suffix == "Pipelines",
        "inc_files": [
            "test_gemm_streamk_reduction_cases.inc",
            "test_gemm_streamk_atomic_cases.inc",
        ],
    },
    "regression": {
        "filter": lambda suffix: suffix == "Regression",
        "inc_files": ["test_gemm_streamk_regression_cases.inc"],
    },
}

# --------------------------------------------------------------------------- #
# Mapping from CamelCase suffix fragments to file-name fragments
# --------------------------------------------------------------------------- #
KNOWN_TOKENS = [
    ("Fp16", "fp16"),
    ("Bf16", "bf16"),
    ("Fp8", "fp8"),
    ("Bf8", "bf8"),
    ("NonPersistent", "nonpersistent"),
    ("Persistent", "persistent"),
    ("Atomic", "atomic"),
    ("Linear", "linear"),
    ("Tree", "tree"),
    ("CompV3", "compv3"),
    ("Pipelines", "pipelines"),
    ("Regression", "regression"),
]


def suffix_to_file_tag(suffix: str) -> str:
    """Convert a CamelCase suffix like 'Fp16PersistentAtomicCompV3' to
    'fp16_persistent_atomic_compv3'."""
    parts: list[str] = []
    remaining = suffix
    while remaining:
        matched = False
        for token, replacement in KNOWN_TOKENS:
            if remaining.startswith(token):
                parts.append(replacement)
                remaining = remaining[len(token) :]
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Unrecognised token in KernelTypes suffix: '{remaining}' "
                f"(from '{suffix}')"
            )
    return "_".join(parts)


def parse_types_header(header_path: str, target: str) -> list[dict]:
    """Return a list of dicts with keys: type_alias, class_name, file_tag, suffix."""
    target_def = TARGETS[target]
    # Pattern matches lines like: using KernelTypesStreamKFp16PersistentAtomicCompV3 = ...
    pattern = re.compile(r"using\s+(KernelTypesStreamK(\w+))\s*=")
    entries: list[dict] = []
    with open(header_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # If the match is: using KernelTypesStreamKFp16PersistentAtomicCompV3 = ...
                # type_alias is KernelTypesStreamKFp16PersistentAtomicCompV3
                # suffix is Fp16PersistentAtomicCompV3
                type_alias = match.group(1)
                suffix = match.group(2)
                if not target_def["filter"](suffix):
                    continue
                entries.append(
                    {
                        "type_alias": type_alias,
                        "class_name": f"TestCkTileStreamK{suffix}",
                        "file_tag": suffix_to_file_tag(suffix),
                    }
                )
    return entries


def output_path(output_dir: str, entry: dict) -> str:
    return os.path.join(output_dir, f"test_gemm_streamk_{entry['file_tag']}.cpp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--types_header", required=True, help="Path to test_gemm_streamk_types.hpp"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for generated .cpp files"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=list(TARGETS.keys()),
        help="Which target to generate files for",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list_files",
        metavar="FILE",
        help="Write output file paths to FILE then exit",
    )
    group.add_argument(
        "--gen_files", action="store_true", help="Generate the .cpp files"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries = parse_types_header(args.types_header, args.target)
    if not entries:
        print(
            f"ERROR: no KernelTypesStreamK* definitions found for target "
            f"'{args.target}' in {args.types_header}",
            file=sys.stderr,
        )
        sys.exit(1)

    inc_files = TARGETS[args.target]["inc_files"]
    inc_includes = "\n".join(f'#include "{f}"' for f in inc_files)

    if args.list_files:
        os.makedirs(os.path.dirname(args.list_files) or ".", exist_ok=True)
        with open(args.list_files, "w") as f:
            for entry in entries:
                f.write(output_path(args.output_dir, entry) + "\n")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for entry in entries:
            path = output_path(args.output_dir, entry)
            content = CPP_TEMPLATE.format(
                class_name=entry["class_name"],
                type_alias=entry["type_alias"],
                inc_includes=inc_includes,
            )
            with open(path, "w") as f:
                f.write(content)


if __name__ == "__main__":
    main()
