#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Enhanced Ninja Dependency Parser

This script combines ninja build file parsing with ninja -t deps to create a comprehensive
mapping that includes both source files AND header files, and properly handles files
used by multiple executables.
"""

import re
import os
import sys
import subprocess
from collections import defaultdict
import json


class EnhancedNinjaDependencyParser:
    def __init__(self, build_file_path, ninja_executable="ninja"):
        self.build_file_path = build_file_path
        self.build_dir = os.path.dirname(build_file_path) or "."
        self.ninja_executable = ninja_executable

        # Core data structures
        self.executable_to_objects = {}  # exe -> [object_files]
        self.object_to_source = {}  # object -> primary_source
        self.object_to_all_deps = {}  # object -> [all_dependencies]
        self.file_to_executables = defaultdict(set)  # file -> {executables}

    def parse_dependencies(self):
        """Main method to parse all dependencies."""
        print(f"Parsing ninja dependencies from: {self.build_file_path}")

        # Step 1: Parse build file for executable -> object mappings
        self._parse_build_file()

        # Step 2: Get all object files and their dependencies
        print(f"Found {len(self.object_to_source)} object files")
        print("Extracting detailed dependencies for all object files...")
        self._extract_object_dependencies()

        # Step 3: Build the final file -> executables mapping
        self._build_file_to_executable_mapping()

    def _parse_build_file(self):
        """Parse the ninja build file to extract executable -> object mappings."""
        print("Parsing ninja build file...")

        with open(self.build_file_path, "r") as f:
            content = f.read()
        # Parse executable build rules
        exe_pattern = r"^build (bin/[^:]+):\s+\S+\s+([^|]+)"
        obj_pattern = r"^build ([^:]+\.(?:cpp|cu|hip)\.o):\s+\S+\s+([^\s|]+)"

        lines = content.split("\n")

        for line in lines:
            # Match executable rules
            exe_match = re.match(exe_pattern, line)
            if exe_match and (
                "EXECUTABLE" in line
                or "test_" in exe_match.group(1)
                or "example_" in exe_match.group(1)
            ):
                exe = exe_match.group(1)
                deps_part = exe_match.group(2).strip()

                object_files = []
                for dep in deps_part.split():
                    if dep.endswith(".o") and not dep.startswith("/"):
                        object_files.append(dep)

                self.executable_to_objects[exe] = object_files
                continue

            # Match object compilation rules
            obj_match = re.match(obj_pattern, line)
            if obj_match:
                object_file = obj_match.group(1)
                source_file = obj_match.group(2)
                self.object_to_source[object_file] = source_file

        print(f"Found {len(self.executable_to_objects)} executables")
        print(f"Found {len(self.object_to_source)} object-to-source mappings")

    def _extract_object_dependencies(self):
        """Extract detailed dependencies for all object files using a single ninja -t deps call.

        Previous implementation spawned ninja -t deps per object file (29K+ subprocesses),
        each re-parsing the full build.ninja. For large monorepo builds (246MB+ build.ninja),
        each call takes 2-14 seconds, making the total ~54 minutes.

        This implementation calls ninja -t deps once (no args) to dump ALL deps in ~2 seconds,
        then parses the output and filters to only the objects we care about.
        """
        object_files = set(self.object_to_source.keys())
        if not object_files:
            print("No object files found - skipping dependency extraction")
            return

        print(f"Running single 'ninja -t deps' call for all built objects...")

        try:
            cmd = [self.ninja_executable, "-t", "deps"]
            result = subprocess.run(
                cmd, cwd=self.build_dir, capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0 and not result.stdout:
                print(f"Warning: ninja -t deps returned code {result.returncode}")
                if result.stderr:
                    print(f"  stderr: {result.stderr.strip()}")
                return

            # Parse the combined output: each block starts with an object name line,
            # followed by indented dependency lines
            ws_root = getattr(self, "workspace_root", "..")
            ws_prefix = ws_root.rstrip("/") + "/"

            current_obj = None
            current_deps = []
            matched = 0

            for line in result.stdout.split("\n"):
                if not line:
                    continue
                if not line.startswith(" ") and not line.startswith("\t"):
                    # This is an object header line like:
                    #   some/path/foo.cpp.o: #deps 42, deps mtime ... (VALID)
                    # Save the previous block if it was relevant
                    if current_obj and current_obj in object_files:
                        self.object_to_all_deps[current_obj] = current_deps
                        matched += 1
                        if matched % 100 == 0:
                            print(f"  Matched {matched} objects so far...")
                    # Parse the new object name (everything before the colon)
                    colon_pos = line.find(":")
                    if colon_pos > 0:
                        current_obj = line[:colon_pos].strip()
                        current_deps = []
                    else:
                        current_obj = None
                        current_deps = []
                else:
                    # Indented dependency line
                    if current_obj is not None:
                        dep_file = line.strip()
                        if dep_file and not dep_file.startswith("#"):
                            # Strip workspace root prefix from absolute paths
                            if dep_file.startswith(ws_prefix):
                                dep_file = dep_file[len(ws_prefix):]
                            current_deps.append(dep_file)

            # Don't forget the last block
            if current_obj and current_obj in object_files:
                self.object_to_all_deps[current_obj] = current_deps
                matched += 1

        except subprocess.TimeoutExpired:
            print("Error: ninja -t deps timed out after 120 seconds")
            return
        except Exception as e:
            print(f"Error running ninja -t deps: {e}")
            return

        print(
            f"Completed dependency extraction for {len(self.object_to_all_deps)} "
            f"of {len(object_files)} object files"
        )

    def _build_file_to_executable_mapping(self):
        """Build the final mapping from files to executables."""
        print("Building file-to-executable mapping...")

        # For monorepo, truncate the path before and including projects/<project_name>
        # This regex matches both absolute and relative monorepo paths
        self.project = None
        rl_regex = rf"(?:^|.*[\\/])projects[\\/]+([^\\/]+)[\\/]+(.*)"
        for exe, object_files in self.executable_to_objects.items():
            for obj_file in object_files:
                # Add all dependencies of this object file
                if obj_file in self.object_to_all_deps:
                    for dep_file in self.object_to_all_deps[obj_file]:
                        match = re.search(rl_regex, dep_file, re.IGNORECASE)
                        if match:
                            dep_file = match.group(2)
                            if not self.project:
                                print(f"Found rocm-libraries project: '{match.group(1)}'")
                                self.project = match.group(1)
                        # Filter out system files and focus on project files
                        if self._is_project_file(dep_file):
                            self.file_to_executables[dep_file].add(exe)

        print(f"Built mapping for {len(self.file_to_executables)} files")

        # Show statistics
        multi_exe_files = {
            f: exes for f, exes in self.file_to_executables.items() if len(exes) > 1
        }
        print(f"Files used by multiple executables: {len(multi_exe_files)}")

        if multi_exe_files:
            print("Sample files with multiple dependencies:")
            for f, exes in sorted(multi_exe_files.items())[:5]:
                print(f"  {f}: {len(exes)} executables")

    def _is_project_file(self, file_path):
        """Determine if a file is part of the project (not system files).

        Handles both standalone-style paths (e.g., include/ck/...) and
        monorepo-style paths (e.g., projects/composablekernel/include/ck/...).
        """
        # Exclude system files first (absolute paths to system dirs)
        if any(
            file_path.startswith(prefix)
            for prefix in ["/usr/", "/opt/rocm", "/lib/", "/system/", "/local/"]
        ):
            return False

        # Project directory prefixes (without monorepo prefix).
        # These match paths relative to the CK project root.
        project_dirs = [
            "include/",
            "library/",
            "test/",
            "example/",
            "src/",
            "profiler/",
            "build/include/",
            "build/_deps/gtest",
            "client_example",
            "codegen",
            "tile_engine",
            "dispatcher",
            "experimental",
            "tutorial",
        ]

        # Check both stripped paths (relative to CK root) and
        # monorepo-prefixed paths (relative to monorepo root)
        if any(file_path.startswith(prefix) for prefix in project_dirs):
            return True

        # Also check monorepo-style paths (projects/composablekernel/...)
        if any(
            file_path.startswith(f"projects/composablekernel/{prefix}")
            for prefix in project_dirs
        ):
            return True

        # Include files with common source/header extensions that weren't
        # excluded as system files above
        if file_path.endswith(
            (".cpp", ".hpp", ".h", ".c", ".cc", ".cxx", ".cu", ".hip", ".inc")
        ):
            return True

        return False

    def export_to_csv(self, output_file):
        """Export the file-to-executable mapping to CSV with proper comma separation."""
        print(f"Exporting mapping to {output_file}")

        with open(output_file, "w") as f:
            f.write("source_file,executables\n")
            for file_path in sorted(self.file_to_executables.keys()):
                executables = sorted(self.file_to_executables[file_path])
                # Use semicolon to separate multiple executables within the field
                exe_list = ";".join(executables)
                f.write(f'"{file_path}","{exe_list}"\n')

    def export_to_json(self, output_file):
        """Export the complete mapping to JSON."""
        print(f"Exporting complete mapping to {output_file}")

        # Build reverse mapping (executable -> files)
        exe_to_files = defaultdict(set)
        for file_path, exes in self.file_to_executables.items():
            for exe in exes:
                exe_to_files[exe].add(file_path)

        mapping_data = {
            "repo": {
                "type": "monorepo" if self.project else "component",
                "project": self.project
            },
            "file_to_executables": {
                file_path: list(exes)
                for file_path, exes in self.file_to_executables.items()
            },
            "executable_to_files": {
                exe: sorted(files) for exe, files in exe_to_files.items()
            },
            "statistics": {
                "total_files": len(self.file_to_executables),
                "total_executables": len(self.executable_to_objects),
                "total_object_files": len(self.object_to_source),
                "files_with_multiple_executables": len(
                    [f for f, exes in self.file_to_executables.items() if len(exes) > 1]
                ),
            },
        }

        with open(output_file, "w") as f:
            json.dump(mapping_data, f, indent=2)

    def print_summary(self):
        """Print a summary of the parsed dependencies."""
        print("\n=== Enhanced Dependency Mapping Summary ===")
        print(f"Total executables: {len(self.executable_to_objects)}")
        print(f"Total files mapped: {len(self.file_to_executables)}")
        print(f"Total object files processed: {len(self.object_to_all_deps)}")

        # Files by type
        cpp_files = sum(
            1 for f in self.file_to_executables.keys() if f.endswith(".cpp")
        )
        hpp_files = sum(
            1 for f in self.file_to_executables.keys() if f.endswith(".hpp")
        )
        h_files = sum(1 for f in self.file_to_executables.keys() if f.endswith(".h"))

        print("\nFile types:")
        print(f"  .cpp files: {cpp_files}")
        print(f"  .hpp files: {hpp_files}")
        print(f"  .h files: {h_files}")

        # Multi-executable files
        multi_exe_files = {
            f: exes for f, exes in self.file_to_executables.items() if len(exes) > 1
        }
        print(f"\nFiles used by multiple executables: {len(multi_exe_files)}")

        if multi_exe_files:
            print("\nTop files with most dependencies:")
            sorted_multi = sorted(
                multi_exe_files.items(), key=lambda x: len(x[1]), reverse=True
            )
            for file_path, exes in sorted_multi[:10]:
                print(f"  {file_path}: {len(exes)} executables")


def main():
    # Accept: build_file, ninja_path, workspace_root
    default_workspace_root = ".."
    if len(sys.argv) > 3:
        build_file = sys.argv[1]
        ninja_path = sys.argv[2]
        workspace_root = sys.argv[3]
    elif len(sys.argv) > 2:
        build_file = sys.argv[1]
        ninja_path = sys.argv[2]
        workspace_root = default_workspace_root
    elif len(sys.argv) > 1:
        build_file = sys.argv[1]
        ninja_path = "ninja"
        workspace_root = default_workspace_root
    else:
        build_file = f"{default_workspace_root}/build/build.ninja"
        ninja_path = "ninja"
        workspace_root = default_workspace_root

    if not os.path.exists(build_file):
        print(f"Error: Build file not found: {build_file}")
        sys.exit(1)

    try:
        subprocess.run([ninja_path, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Error: ninja executable not found: {ninja_path}")
        sys.exit(1)

    parser = EnhancedNinjaDependencyParser(build_file, ninja_path)
    parser.workspace_root = workspace_root  # Attach for use in _extract_object_dependencies
    parser.parse_dependencies()
    parser.print_summary()

    # Export results
    output_dir = os.path.dirname(build_file)
    csv_file = os.path.join(output_dir, "enhanced_file_executable_mapping.csv")
    json_file = os.path.join(output_dir, "enhanced_dependency_mapping.json")

    parser.export_to_csv(csv_file)
    parser.export_to_json(json_file)

    print("\nResults exported to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")


if __name__ == "__main__":
    main()
