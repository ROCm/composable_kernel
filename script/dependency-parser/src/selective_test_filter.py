#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Selective Test Filter Tool

Given two git refs (branches or commit IDs), this tool:
- Identifies changed files between the refs
- Loads the enhanced dependency mapping JSON (from enhanced_ninja_parser.py)
- Maps changed files to affected test executables (optionally filtering for "test_" prefix)
- Exports the list of tests to run to tests_to_run.json

Usage:
  python selective_test_filter.py <depmap_json> <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]

Arguments:
  <depmap_json>   Path to enhanced_dependency_mapping.json
  <ref1>          Source git ref (branch or commit)
  <ref2>          Target git ref (branch or commit)

Options:
  --all           Include all executables (default)
  --test-prefix   Only include executables starting with "test_"
  --output        Output JSON file (default: tests_to_run.json)
"""

import sys
import subprocess
import json
import os
import re


def get_changed_files(ref1, ref2, project: str = None):
    """Return a set of files changed between two git refs."""
    try:
        # Don't use git path filter - it can miss files when running from subdirectories
        git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
        cmd = ["git", "-C", git_root, "diff", "--name-only", f"{ref1}...{ref2}", "--", "projects/composablekernel"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        raw_files = set(line.strip() for line in result.stdout.splitlines() if line.strip())

        if project is None:
            files = raw_files
            print(f"Identified {len(files)} modified files")
        else:
            # Strip projects/{project}/ prefix from changed files
            root = f"projects/{project}/"
            root_len = len(root)
            files = set()
            for f in raw_files:
                if f.startswith(root):
                    files.add(f[root_len:])
            print(f"Identified {len(files)} files modified in project '{project}'")

        return files
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        sys.exit(1)


def load_depmap(depmap_json):
    """Load the dependency mapping JSON."""
    with open(depmap_json, "r") as f:
        data = json.load(f)
    # Support both old and new formats
    json_project = None
    if "repo" in data:
        if data["repo"]["type"] == "monorepo":
            json_project = data["repo"]["project"]
        elif "workspace_root" in data["repo"]:
            # Extract project from workspace_root path
            workspace_root = data["repo"]["workspace_root"]
            # Convert relative path to absolute if needed
            if not os.path.isabs(workspace_root):
                depmap_dir = os.path.dirname(os.path.abspath(depmap_json))
                workspace_root = os.path.abspath(os.path.join(depmap_dir, workspace_root))
            # If workspace_root is like /path/to/projects/composablekernel, extract composablekernel
            if "/projects/" in workspace_root:
                json_project = workspace_root.split("/projects/")[1].rstrip("/").split("/")[0]
    if "file_to_executables" in data:
        return data["file_to_executables"], json_project
    return data, json_project


def get_ctest_registered_tests(build_dir=None):
    """Get list of tests registered with CTest (excludes EXCLUDE_FROM_ALL targets)."""
    try:
        cmd = ["ctest", "-N"]
        if build_dir:
            cmd.extend(["--test-dir", build_dir])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return None

        tests = set()
        # CTest formats test numbers with variable spacing:
        # Test   #1: name (3 spaces for 1-9)
        # Test  #10: name (2 spaces for 10-99)
        # Test #100: name (1 space for 100+)
        # Use regex to match all formats
        test_pattern = re.compile(r'^\s*Test\s+#\d+:\s*(.+)$')

        for line in result.stdout.splitlines():
            match = test_pattern.match(line)
            if match:
                test_name = match.group(1).strip()
                tests.add(test_name)

        return tests
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def select_tests(file_to_executables, changed_files, filter_mode, ctest_only=False, build_dir=None):
    """Return a set of test executables affected by changed files."""
    affected = set()

    ctest_tests = None
    if ctest_only:
        ctest_tests = get_ctest_registered_tests(build_dir)
        if ctest_tests is None:
            print("Warning: Could not get CTest test list, including all executables")
        else:
            print(f"Filtering to {len(ctest_tests)} CTest-registered tests (excluding EXCLUDE_FROM_ALL targets)")

    for f in changed_files:
        if f in file_to_executables:
            for exe in file_to_executables[f]:
                if filter_mode == "test_prefix" and not os.path.basename(exe).startswith("test_"):
                    continue

                if ctest_only and ctest_tests is not None:
                    test_name = exe.replace("bin/", "")
                    if test_name not in ctest_tests:
                        continue

                affected.add(exe)

    return sorted(affected)


def main():
    if "--audit" in sys.argv:
        if len(sys.argv) < 2:
            print("Usage: python selective_test_filter.py <depmap_json> --audit")
            sys.exit(1)
        depmap_json = sys.argv[1]
        if not os.path.exists(depmap_json):
            print(f"Dependency map JSON not found: {depmap_json}")
            sys.exit(1)
        file_to_executables, _ = load_depmap(depmap_json)
        for f, exes in file_to_executables.items():
            print(f"{f}: {', '.join(exes)}")
        print(f"Total files: {len(file_to_executables)}")
        sys.exit(0)

    if "--optimize-build" in sys.argv:
        if len(sys.argv) < 3:
            print(
                "Usage: python selective_test_filter.py <depmap_json> --optimize-build <changed_file1> [<changed_file2> ...]"
            )
            sys.exit(1)
        depmap_json = sys.argv[1]
        changed_files = set(sys.argv[sys.argv.index("--optimize-build") + 1 :])
        if not os.path.exists(depmap_json):
            print(f"Dependency map JSON not found: {depmap_json}")
            sys.exit(1)
        file_to_executables, _ = load_depmap(depmap_json)
        affected_executables = set()
        for f in changed_files:
            if f in file_to_executables:
                affected_executables.update(file_to_executables[f])
        print("Affected executables:")
        for exe in sorted(affected_executables):
            print(exe)
        print(f"Total affected executables: {len(affected_executables)}")
        sys.exit(0)

    if len(sys.argv) < 4:
        print(
            "Usage: python selective_test_filter.py <depmap_json> <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]"
        )
        sys.exit(1)

    depmap_json = sys.argv[1]
    ref1 = sys.argv[2]
    ref2 = sys.argv[3]
    filter_mode = "all"
    output_json = "tests_to_run.json"
    ctest_only = False
    build_dir = None

    if "--test-prefix" in sys.argv:
        filter_mode = "test_prefix"
    if "--all" in sys.argv:
        filter_mode = "all"
    if "--ctest-only" in sys.argv:
        ctest_only = True
    if "--build-dir" in sys.argv:
        idx = sys.argv.index("--build-dir")
        if idx + 1 < len(sys.argv):
            build_dir = sys.argv[idx + 1]
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_json = sys.argv[idx + 1]

    # If build_dir not specified, try to infer from depmap_json path
    if ctest_only and build_dir is None:
        depmap_dir = os.path.dirname(os.path.abspath(depmap_json))
        if os.path.basename(depmap_dir) in ["build", "."]:
            build_dir = depmap_dir
        elif os.path.exists(os.path.join(depmap_dir, "build.ninja")):
            build_dir = depmap_dir

    if not os.path.exists(depmap_json):
        print(f"Dependency map JSON not found: {depmap_json}")
        sys.exit(1)

    file_to_executables, json_project = load_depmap(depmap_json)
    changed_files = get_changed_files(ref1, ref2, json_project)
    if not changed_files:
        print("No changed files detected.")
        tests = []
    else:
        tests = select_tests(file_to_executables, changed_files, filter_mode, ctest_only, build_dir)

    # Generate ctest regex from test names
    # Split into chunks to avoid regex length limits in CTest
    regex_chunks = []
    chunk_size = 50  # Max tests per regex pattern

    if tests:
        # Extract basenames for regex (e.g., bin/test_gemm -> test_gemm)
        test_names = [os.path.basename(t) for t in tests]

        # Anchor each name with ^...$ and escape regex metacharacters so that
        # `ctest -R` does exact-name matching rather than substring matching
        # (otherwise e.g. 'test_grouped_convnd_bwd_weight' would substring-match
        # 'test_grouped_convnd_bwd_weight_bilinear' and try to run an
        # executable that was never built).
        anchored = [f"^{re.escape(n)}$" for n in test_names]

        # Split into chunks
        for i in range(0, len(anchored), chunk_size):
            chunk = anchored[i:i + chunk_size]
            regex_chunks.append("|".join(chunk))

        # Keep single regex for backward compatibility (but may be too long)
        regex = "|".join(anchored)
    else:
        regex = ""

    # Output format matches Jenkinsfile usage and documentation
    output = {
        "tests_to_run": tests,  # For backward compatibility and length check
        "executables": tests,  # Used by Jenkinsfile for ninja build
        "regex": regex,  # Used by Jenkinsfile for ctest (deprecated for large test sets)
        "regex_chunks": regex_chunks,  # Multiple regex patterns for large test sets
        "changed_files": sorted(changed_files),
        "statistics": {
            "total_changed_files": len(changed_files),
            "total_affected_executables": len(tests),
            "num_regex_chunks": len(regex_chunks),
        },
    }

    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Exported {len(tests)} tests to run to {output_json}")

    # Print changed files for visibility
    if changed_files:
        print(f"\nChanged files ({len(changed_files)}):")
        for f in sorted(changed_files):
            print(f"  - {f}")
    else:
        print("\nNo files changed.")


if __name__ == "__main__":
    main()
