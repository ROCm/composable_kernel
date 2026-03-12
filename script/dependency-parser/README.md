# Dependency-based Selective Test Filtering using Static Analysis of Ninja Builds for C++ Projects

## Overview

This tool provides advanced dependency-based selective test filtering and build optimization for large C++ monorepos using static parsing of Ninja build files. By analyzing both source and header dependencies, it enables precise identification of which tests and executables are affected by code changes, allowing for efficient CI/CD workflows and faster incremental builds.

The parser:
- Identifies all executables in the Ninja build.
- Maps object files to their source and header dependencies using `ninja -t deps`.
- Constructs a reverse mapping from each file to all dependent executables.
- Automatically detects monorepo structure (`projects/<name>/`) and scopes analysis accordingly.
- Exports results in CSV and JSON formats for integration with other tools.

## Features

- **Comprehensive Dependency Tracking**: Captures direct source file dependencies and, critically, all included header files via `ninja -t deps`.
- **Executable to Object Mapping**: Parses the `build.ninja` file to understand how executables are linked from object files.
- **Batch Dependency Extraction**: Runs a single `ninja -t deps` call (no arguments) to dump all dependency information at once, then filters in-memory. This avoids the massive overhead of per-object subprocess calls on large build files (e.g., a 246MB `build.ninja` with 29K+ objects completes in ~2 seconds instead of ~54 minutes).
- **Monorepo Awareness**: Automatically detects `projects/<project>/` paths, strips them to project-relative paths, and scopes `git diff` to only the relevant subtree.
- **File to Executable Inversion**: Inverts the dependency graph to map each file to the set of executables that depend on it.
- **Filtering**: Filters out system files (`/usr/`, `/opt/rocm/`, etc.) and focuses on project-specific dependencies.
- **Multiple Output Formats**:
    - **CSV**: `enhanced_file_executable_mapping.csv` - Each row lists a file and a semicolon-separated list of dependent executables.
    - **JSON**: `enhanced_dependency_mapping.json` - Includes file-to-executable mapping, executable-to-file mapping, repo metadata, and statistics.
- **Robust Error Handling**: Includes error handling for missing files and failed subprocess commands.

## Prerequisites

- **Python 3.7+**
- **Ninja build system**: The `ninja` executable must be in the system's PATH or its path provided as an argument.
- A **Ninja build directory** containing a `build.ninja` file. The project should have been built at least once (even partially) so that `ninja -t deps` has dependency data.

## Quick Start with launch_tests.sh

The easiest way to use this tool is via the `launch_tests.sh` wrapper script:

```bash
# From the monorepo root (or anywhere):
script/launch_tests.sh /path/to/build-dir

# Uses default build dir (<CK_ROOT>/build) if no argument given:
script/launch_tests.sh
```

This script:
1. Discovers the git root (monorepo root) automatically.
2. Runs the dependency parser against `build.ninja`.
3. Runs `git diff` between `origin/develop` and the current branch (scoped to CK files only).
4. Maps changed files to affected tests/examples.
5. Runs the affected tests via `ctest` in chunks.

Environment variables:
- `CTEST_CHUNK_SIZE`: Number of tests per ctest invocation (default: 10).
- `CTEST_FAIL_FAST`: Set to `true` to stop on first failure (default: `false`).

## Using CMake with Ninja

To use this tool effectively, your C++ project should be configured with CMake to generate Ninja build files:

1. **Configure CMake to use Ninja:**
    ```bash
    cmake -G Ninja \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_TARGETS="gfx942" \
      /path/to/composablekernel
    ```

2. **Build your project (full or partial):**
    ```bash
    # Full build
    ninja

    # Or build specific targets
    ninja example_gemm_xdl_fp16 example_gemm_xdl_fp16_v3
    ```
    The parser only extracts dependencies for objects that were actually built.

3. **Run the dependency parser:**
    ```bash
    python main.py parse /path/to/build/build.ninja --workspace-root /path/to/monorepo-root
    ```

**Note:** `--workspace-root` should point to the **git root** (monorepo root) for correct monorepo detection. If omitted, it defaults to `..` relative to the build directory.

## Usage

All features are available via the unified `main.py` CLI:

```bash
# Dependency parsing
python main.py parse /path/to/build.ninja --workspace-root /path/to/monorepo-root

# Selective test filtering (between git refs)
python main.py select enhanced_dependency_mapping.json <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]

# Code auditing (list all files and their dependent executables)
python main.py audit enhanced_dependency_mapping.json

# Build optimization (list affected executables for specific changed files)
python main.py optimize enhanced_dependency_mapping.json <changed_file1> [<changed_file2> ...]
```

### Parse arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `build_ninja` | Yes | Path to the `build.ninja` file |
| `--workspace-root` | No | Root of the workspace/monorepo (default: `..`) |
| `--ninja` | No | Path to the ninja executable (default: `ninja`) |

### Select arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `depmap_json` | Yes | Path to `enhanced_dependency_mapping.json` |
| `ref1` | Yes | Source git ref (branch or commit SHA) |
| `ref2` | Yes | Target git ref (branch or commit SHA) |
| `--all` | No | Include all affected executables (default) |
| `--test-prefix` | No | Only include executables starting with `test_` |
| `--output` | No | Output JSON file (default: `tests_to_run.json`) |

## How It Works

1.  **Build File Parsing (`_parse_build_file`)**:
    *   Reads the `build.ninja` file (~246MB for the full CK monorepo build).
    *   Uses regular expressions to identify executable link rules and object compilation rules.
    *   Populates `executable_to_objects` and `object_to_source` mappings.

2.  **Batch Dependency Extraction (`_extract_object_dependencies`)**:
    *   Runs a single `ninja -t deps` command (no arguments) which dumps all dependency information for every built object file.
    *   Parses the output and filters to only the objects found in `object_to_source`.
    *   Strips the workspace root prefix from absolute paths to produce project-relative paths.

3.  **Monorepo Path Detection (`_build_file_to_executable_mapping`)**:
    *   Applies a regex to detect `projects/<project_name>/` in dependency paths.
    *   Strips the monorepo prefix so paths are relative to the CK project root (e.g., `include/ck/ck.hpp`).
    *   Records the detected project name for use by the selective test filter.

4.  **File Filtering (`_is_project_file`)**:
    *   Excludes system files (`/usr/`, `/opt/rocm/`, etc.).
    *   Includes files in known CK directories (`include/`, `library/`, `test/`, `example/`, etc.).
    *   Also recognizes monorepo-prefixed paths (`projects/composablekernel/include/`, etc.).

5.  **Selective Test Filtering (`selective_test_filter.py`)**:
    *   Loads the dependency mapping JSON.
    *   Runs `git diff --name-only` between two refs, scoped to `projects/<project>/` when in monorepo mode.
    *   Strips the monorepo prefix from changed file paths.
    *   Looks up each changed file in the dependency map to find affected executables.
    *   Exports the list of tests to run as JSON.

## Output Files

Running the parser generates two files in the build directory:

-   **`enhanced_file_executable_mapping.csv`**:
    ```csv
    source_file,executables
    "include/ck/ck.hpp","bin/example_gemm_xdl_fp16;bin/example_gemm_xdl_fp16_v3"
    "example/01_gemm/gemm_xdl_fp16.cpp","bin/example_gemm_xdl_fp16"
    ```

-   **`enhanced_dependency_mapping.json`**:
    ```json
    {
      "repo": {
        "type": "monorepo",
        "project": "composablekernel"
      },
      "file_to_executables": {
        "include/ck/ck.hpp": ["bin/example_gemm_xdl_fp16", "bin/example_gemm_xdl_fp16_v3"],
        "example/01_gemm/gemm_xdl_fp16.cpp": ["bin/example_gemm_xdl_fp16"]
      },
      "executable_to_files": {
        "bin/example_gemm_xdl_fp16": ["include/ck/ck.hpp", "example/01_gemm/gemm_xdl_fp16.cpp"]
      },
      "statistics": {
        "total_files": 180,
        "total_executables": 20403,
        "total_object_files": 29530,
        "files_with_multiple_executables": 140
      }
    }
    ```

## Use Cases

-   **Selective CI/CD Testing**: Run only the tests affected by a PR's changes, cutting CI time dramatically.
-   **Impact Analysis**: Determine which executables need to be rebuilt when a header changes.
-   **Build Optimization**: Identify which targets are affected by a set of file changes.
-   **Code Auditing**: Get a clear overview of how files are used across different executables.

## Limitations

-   Relies on the accuracy of Ninja's dependency information (`ninja -t deps`). If the build system doesn't correctly generate `.d` (dependency) files, the header information might be incomplete.
-   Only objects that have been **actually built** will have dependency data. A partial build means partial coverage of the dependency map.
-   The definition of "project file" vs. "system file" is based on a path-based heuristic and might need adjustment for other project structures.

## Troubleshooting

-   **"ninja: command not found"**: Ensure `ninja` is installed and in your PATH, or provide the full path via `--ninja`.
-   **"build.ninja not found"**: Double-check the path to your `build.ninja` file.
-   **Empty or Incomplete Output**:
    *   Make sure the project has been successfully built at least once. `ninja -t deps` relies on information generated during the build.
    *   Verify that your CMake is configured to generate dependency files for Ninja (`-G Ninja`).
-   **JSON shows `"type": "component"` instead of `"monorepo"`**: Ensure `--workspace-root` points to the **git/monorepo root**, not the CK project root. The parser needs to see `projects/<name>/` in the dependency paths to detect monorepo mode.
