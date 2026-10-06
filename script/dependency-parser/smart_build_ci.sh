#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Build CI Script
#
# This script orchestrates the smart-build process:
# 1. Runs ci_safety_check.sh to determine if selective build is safe
# 2. Generates dependency map using cmake-parse
# 3. Selects affected tests
# 4. Outputs build targets to a file for Jenkins to consume
#
# Exit codes:
#   0 = Success (selective build targets generated)
#   1 = Full build required (run ninja check)
#
# Output files:
#   tests_to_run.json - Selected tests and executables
#   build_targets.txt - Space-separated list of ninja targets to build
#   build_mode.env - Environment variables (SMART_BUILD_MODE=selective|full)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$(pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd ${BUILD_DIR}/.. && pwd)}"
PARALLEL="${PARALLEL:-32}"
BASE_BRANCH="${BASE_BRANCH:-develop}"

echo "========================================="
echo "Smart Build CI"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
echo "BASE_BRANCH: ${BASE_BRANCH}"
echo "PARALLEL: ${PARALLEL}"
echo "-----------------------------------------"

# Step 1: Run CI safety check
echo "Step 1: Running CI safety check..."
cd "${BUILD_DIR}"

if ! bash "${SCRIPT_DIR}/ci_safety_check.sh"; then
    echo "CI safety check failed - full build required"
    echo "full" > build_targets.txt
    exit 1
fi

echo "✓ CI safety check passed - selective build enabled"

# Step 2: Generate dependency map
echo ""
echo "Step 2: Generating dependency map..."
if [ ! -f "compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in ${BUILD_DIR}"
    echo "Make sure cmake configure has been run with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    exit 1
fi

if [ ! -f "build.ninja" ]; then
    echo "Error: build.ninja not found in ${BUILD_DIR}"
    echo "Make sure cmake configure has been run with -G Ninja"
    exit 1
fi

python3 "${SCRIPT_DIR}/main.py" cmake-parse \
    compile_commands.json \
    build.ninja \
    --workspace-root "${WORKSPACE_ROOT}" \
    --parallel ${PARALLEL} \
    --output enhanced_dependency_mapping.json

if [ ! -f "enhanced_dependency_mapping.json" ]; then
    echo "Error: Failed to generate enhanced_dependency_mapping.json"
    exit 1
fi

echo "✓ Dependency map generated"

# Step 3: Select affected tests
echo ""
echo "Step 3: Selecting affected tests..."
python3 "${SCRIPT_DIR}/main.py" select \
    enhanced_dependency_mapping.json \
    origin/${BASE_BRANCH} \
    HEAD \
    --ctest-only \
    --output tests_to_run.json

if [ ! -f "tests_to_run.json" ]; then
    echo "Error: Failed to generate tests_to_run.json"
    exit 1
fi

# Step 3b: Force-include the ck_tile FMHA test if its example changed
# The dependency parser only traces sources in the test build graph, so changes
# to example/ck_tile/01_fmha/ (and subfolders) are not mapped to any test.
# Explicitly build and run test_ck_tile_fmha when those C++/Python files change.
echo ""
echo "Step 3b: Checking ck_tile FMHA example for changes..."
FMHA_EXAMPLE_PATTERN='projects/composablekernel/example/ck_tile/01_fmha/.*\.(cpp|cc|cxx|hpp|hxx|h|py)$'
CHANGED_FILES=$(git diff --name-only origin/${BASE_BRANCH}...HEAD 2>/dev/null || echo "")

if echo "${ARCH_NAME}" | grep -qE "gfx12|gfx9" \
    && echo "${CHANGED_FILES}" | grep -qE "${FMHA_EXAMPLE_PATTERN}"; then
    echo "FMHA example changes detected - forcing test_ck_tile_fmha build and run"
    # test_ck_tile_fmha is a ninja umbrella target (builds all fwd/bwd fmha tests)
    # and, unanchored, matches every registered test_ck_tile_fmha_* ctest name.
    tmp_json=$(mktemp)
    jq --arg t "test_ck_tile_fmha" '
        .executables = (.executables + [$t] | unique)
        | .tests_to_run = (.tests_to_run + [$t] | unique)
        | .regex_chunks = (.regex_chunks + [$t] | unique)
        | .regex = (if (.regex | length) > 0 then .regex + "|" + $t else $t end)
    ' tests_to_run.json > "${tmp_json}" && mv "${tmp_json}" tests_to_run.json
else
    echo "No ck_tile FMHA example changes detected"
fi

# Step 3c: Force-include the ck_tile example build tests if their CMakeLists changed
# CMakeLists.txt is never a key in the dependency map (it is built from the
# .cpp/.cc/.cu/.hip translation units and their header closures), so a change
# that adds, removes or re-guards an example target maps to no test at all and
# the whole stage is skipped. Build and run the example set in that case.
echo ""
echo "Step 3c: Checking ck_tile example CMakeLists for changes..."
CK_TILE_EXAMPLE_CMAKE_PATTERN='projects/composablekernel/example/ck_tile/.*CMakeLists\.txt$'

if ! echo "${CHANGED_FILES}" | grep -qE "${CK_TILE_EXAMPLE_CMAKE_PATTERN}"; then
    echo "No ck_tile example CMakeLists changes detected"
elif ! ninja -t query tile_examples >/dev/null 2>&1; then
    # example/ is only added when BUILD_CK_EXAMPLES is on, so the umbrella
    # target does not always exist. Forcing it then would fail the build.
    echo "ck_tile example CMakeLists changed, but tile_examples is not configured - nothing to force"
else
    echo "ck_tile example CMakeLists changes detected - forcing tile_examples build and run"
    # tile_examples is a ninja umbrella target depending on every registered
    # example, so it covers whichever subset this architecture configured.
    # The ctest names are the target names, hence the separate regex.
    tmp_json=$(mktemp)
    jq --arg t "tile_examples" --arg r "^tile_example_" '
        .executables = (.executables + [$t] | unique)
        | .tests_to_run = (.tests_to_run + [$t] | unique)
        | .regex_chunks = (.regex_chunks + [$r] | unique)
        | .regex = (if (.regex | length) > 0 then .regex + "|" + $r else $r end)
    ' tests_to_run.json > "${tmp_json}" && mv "${tmp_json}" tests_to_run.json
fi

# Step 4: Check if any tests were selected
num_tests=$(jq -r '.tests_to_run | length' tests_to_run.json 2>/dev/null || echo "0")
echo "✓ Selected ${num_tests} tests"

if [ "${num_tests}" -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Result: No tests affected by changes"
    echo "========================================="
    echo "none" > build_targets.txt
    exit 0
fi

# Step 5: Extract build targets (executables)
echo ""
echo "Step 4: Extracting build targets..."
jq -r '.executables[]' tests_to_run.json | tr '\n' ' ' > build_targets.txt

num_targets=$(jq -r '.executables | length' tests_to_run.json)
echo "✓ Generated ${num_targets} build targets"

# Display summary
echo ""
echo "========================================="
echo "Smart Build Summary"
echo "========================================="
echo "Tests to run: ${num_tests}"
echo "Build targets: ${num_targets}"
echo "Output files:"
echo "  - tests_to_run.json (test selection)"
echo "  - build_targets.txt (ninja targets)"
echo "  - build_mode.env (SMART_BUILD_MODE=selective)"
echo "========================================="

# Show first few targets for verification
echo ""
echo "Sample build targets (first 5):"
head -1 build_targets.txt | tr ' ' '\n' | head -5

echo ""
echo "✓ Smart build preparation complete"
exit 0
