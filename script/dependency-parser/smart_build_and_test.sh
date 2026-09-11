#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Build and Test Execution Script
#
# This script handles the complete smart-build workflow:
# 1. Runs smart_build_ci.sh to determine build mode and targets
# 2. Builds only affected targets (selective mode) or everything (full mode)
# 3. Runs affected tests using ctest with regex filtering
# 4. Optionally processes ninja build traces
#
# Exit codes:
#   0 = Success
#   1 = Build or test failure
#
# Environment variables:
#   WORKSPACE_ROOT - Path to workspace root
#   BUILD_DIR - Build directory (defaults to current directory)
#   PARALLEL - Number of parallel jobs for dependency analysis (default: 32)
#   NINJA_JOBS - Number of ninja parallel jobs (required)
#   ARCH_NAME - Architecture name for trace files (required if PROCESS_NINJA_TRACE=true)
#   PROCESS_NINJA_TRACE - Set to "true" to process ninja build traces (default: false)
#   NINJA_FTIME_TRACE - Set to "true" to run ClangBuildAnalyzer (default: false)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$(pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd ${BUILD_DIR}/.. && pwd)}"
PARALLEL="${PARALLEL:-32}"
PROCESS_NINJA_TRACE="${PROCESS_NINJA_TRACE:-false}"
NINJA_FTIME_TRACE="${NINJA_FTIME_TRACE:-false}"

# Validate required parameters
if [ -z "$NINJA_JOBS" ]; then
    echo "Error: NINJA_JOBS environment variable is required"
    exit 1
fi

if [ "$PROCESS_NINJA_TRACE" = "true" ] && [ -z "$ARCH_NAME" ]; then
    echo "Error: ARCH_NAME environment variable is required when PROCESS_NINJA_TRACE=true"
    exit 1
fi

echo "========================================="
echo "Smart Build and Test Execution"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
echo "NINJA_JOBS: ${NINJA_JOBS}"
echo "PROCESS_NINJA_TRACE: ${PROCESS_NINJA_TRACE}"
echo "NINJA_FTIME_TRACE: ${NINJA_FTIME_TRACE}"
echo "-----------------------------------------"

cd "${BUILD_DIR}"

# Step 1: Run smart-build CI script
echo "🚀 Using Smart Build System"
echo ""

export WORKSPACE_ROOT
export PARALLEL

if ! bash "${SCRIPT_DIR}/smart_build_ci.sh"; then
    # Full build required (exit code 1 from smart_build_ci.sh)
    echo "⚠ Full build mode - building and testing everything"
    ninja -j${NINJA_JOBS} check

    # Process ninja build trace if requested
    if [ "$PROCESS_NINJA_TRACE" = "true" ]; then
        echo ""
        echo "Processing ninja build trace..."
        python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace_${ARCH_NAME}.json
        python3 ../script/parse_ninja_trace.py ck_build_trace_${ARCH_NAME}.json

        if [ "$NINJA_FTIME_TRACE" = "true" ]; then
            echo "Running ClangBuildAnalyzer..."
            /ClangBuildAnalyzer/build/ClangBuildAnalyzer --all . clang_build.log
            /ClangBuildAnalyzer/build/ClangBuildAnalyzer --analyze clang_build.log > clang_build_analysis_${ARCH_NAME}.log
        fi
    fi

    exit 0
fi

# Step 2: Selective build mode - read targets
BUILD_TARGETS=$(cat build_targets.txt)

if [ "$BUILD_TARGETS" = "none" ]; then
    echo "✓ No tests affected by changes - skipping build and test execution"
    exit 0
fi

# Step 3: Build only affected targets
echo "✓ Selective build - building only affected targets"
echo "Building targets: ${BUILD_TARGETS}"
ninja -j${NINJA_JOBS} ${BUILD_TARGETS}

# Process ninja build trace if requested
if [ "$PROCESS_NINJA_TRACE" = "true" ]; then
    echo ""
    echo "Processing ninja build trace..."
    python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace_${ARCH_NAME}.json
    python3 ../script/parse_ninja_trace.py ck_build_trace_${ARCH_NAME}.json

    if [ "$NINJA_FTIME_TRACE" = "true" ]; then
        echo "Running ClangBuildAnalyzer..."
        /ClangBuildAnalyzer/build/ClangBuildAnalyzer --all . clang_build.log
        /ClangBuildAnalyzer/build/ClangBuildAnalyzer --analyze clang_build.log > clang_build_analysis_${ARCH_NAME}.log
    fi
fi

# Step 4: Run affected tests using regex_chunks
echo ""
echo "Running affected tests..."

NUM_CHUNKS=$(jq -r '.regex_chunks | length' tests_to_run.json)
echo "Running ${NUM_CHUNKS} test chunk(s)"

if [ "$NUM_CHUNKS" -eq 1 ]; then
    TEST_REGEX=$(jq -r '.regex_chunks[0]' tests_to_run.json)
    CTEST_PARALLEL_LEVEL=4 ctest --output-on-failure -R "${TEST_REGEX}"
else
    for ((i=0; i<NUM_CHUNKS; i++)); do
        TEST_REGEX=$(jq -r ".regex_chunks[$i]" tests_to_run.json)
        echo "Running test chunk $((i+1))/${NUM_CHUNKS}"
        CTEST_PARALLEL_LEVEL=4 ctest --output-on-failure -R "${TEST_REGEX}"
    done
fi

echo ""
echo "✓ Smart build and test execution complete"
exit 0
