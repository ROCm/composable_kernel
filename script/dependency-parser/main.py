#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unified CLI for Ninja Dependency Analysis and Selective Testing

Features:
- CMake pre-build dependency parsing (using compile_commands.json + clang -MM)
- Post-build dependency parsing (from build.ninja - legacy)
- Selective test filtering (between git refs)
- Code auditing (--audit)
- Build optimization (--optimize-build)
"""

import argparse
import sys


def run_cmake_dependency_analyzer(args):
    from src.cmake_dependency_analyzer import main as cmake_main

    sys.argv = ["cmake_dependency_analyzer.py"] + args
    cmake_main()


def run_dependency_parser(args):
    from src.enhanced_ninja_parser import main as ninja_main

    sys.argv = ["enhanced_ninja_parser.py"] + args
    ninja_main()


def run_selective_test_filter(args):
    from src.selective_test_filter import main as filter_main

    sys.argv = ["selective_test_filter.py"] + args
    filter_main()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Dependency Analysis & Selective Testing Tool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # CMake pre-build dependency parsing (NEW - RECOMMENDED)
    parser_cmake = subparsers.add_parser(
        "cmake-parse",
        help="[NEW] Parse compile_commands.json for pre-build dependency analysis"
    )
    parser_cmake.add_argument(
        "compile_commands",
        help="Path to compile_commands.json"
    )
    parser_cmake.add_argument(
        "build_ninja",
        help="Path to build.ninja"
    )
    parser_cmake.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root directory (default: current directory)"
    )
    parser_cmake.add_argument(
        "--output",
        default="cmake_dependency_mapping.json",
        help="Output JSON file (default: cmake_dependency_mapping.json)"
    )
    parser_cmake.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser_cmake.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser_cmake.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache is valid"
    )

    # Ninja post-build dependency parsing (LEGACY)
    parser_parse = subparsers.add_parser(
        "parse", help="[LEGACY] Parse build.ninja post-build (requires full build first)"
    )
    parser_parse.add_argument("build_ninja", help="Path to build.ninja")
    parser_parse.add_argument(
        "--ninja", help="Path to ninja executable", default="ninja"
    )
    parser_parse.add_argument(
        "--workspace-root", help="Path to workspace root", default=None
    )

    # Selective testing
    parser_test = subparsers.add_parser(
        "select", help="Selective test filtering between git refs"
    )
    parser_test.add_argument("depmap_json", help="Path to dependency mapping JSON")
    parser_test.add_argument("ref1", help="Source git ref")
    parser_test.add_argument("ref2", help="Target git ref")
    parser_test.add_argument(
        "--all", action="store_true", help="Include all executables"
    )
    parser_test.add_argument(
        "--test-prefix",
        action="store_true",
        help="Only include executables starting with 'test_'",
    )
    parser_test.add_argument(
        "--ctest-only",
        action="store_true",
        help="Only include tests registered with CTest (excludes EXCLUDE_FROM_ALL targets like benchmarks)",
    )
    parser_test.add_argument(
        "--build-dir",
        help="Build directory for ctest lookup (optional, default: inferred from depmap_json path)",
    )
    parser_test.add_argument(
        "--output", help="Output JSON file", default="tests_to_run.json"
    )

    # Code auditing
    parser_audit = subparsers.add_parser(
        "audit", help="List all files and their dependent executables"
    )
    parser_audit.add_argument("depmap_json", help="Path to dependency mapping JSON")

    # Build optimization
    parser_opt = subparsers.add_parser(
        "optimize", help="List affected executables for changed files"
    )
    parser_opt.add_argument("depmap_json", help="Path to dependency mapping JSON")
    parser_opt.add_argument("changed_files", nargs="+", help="List of changed files")

    args = parser.parse_args()

    if args.command == "cmake-parse":
        cmake_args = [args.compile_commands, args.build_ninja]
        cmake_args += ["--workspace-root", args.workspace_root]
        cmake_args += ["--output", args.output]
        cmake_args += ["--parallel", str(args.parallel)]
        if args.quiet:
            cmake_args.append("--quiet")
        if args.force:
            cmake_args.append("--force")
        run_cmake_dependency_analyzer(cmake_args)
    elif args.command == "parse":
        parse_args = [args.build_ninja, args.ninja]
        if args.workspace_root:
            parse_args.append(args.workspace_root)
        run_dependency_parser(parse_args)
    elif args.command == "select":
        filter_args = [args.depmap_json, args.ref1, args.ref2]
        if args.test_prefix:
            filter_args.append("--test-prefix")
        if args.all:
            filter_args.append("--all")
        if args.ctest_only:
            filter_args.append("--ctest-only")
        if args.build_dir:
            filter_args += ["--build-dir", args.build_dir]
        if args.output:
            filter_args += ["--output", args.output]
        run_selective_test_filter(filter_args)
    elif args.command == "audit":
        run_selective_test_filter([args.depmap_json, "--audit"])
    elif args.command == "optimize":
        run_selective_test_filter(
            [args.depmap_json, "--optimize-build"] + args.changed_files
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
