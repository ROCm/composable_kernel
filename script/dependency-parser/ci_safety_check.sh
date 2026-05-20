#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# CI Safety Check for Smart Build System
#
# This script determines when to force full builds vs selective builds.
# Integrates with existing Jenkins infrastructure (FORCE_CI, BRANCH_NAME, etc.)
#
# Exit codes:
#   0 = Selective build OK (use smart build)
#   1 = Full build required
#
# Environment variables (set by Jenkins):
#   FORCE_CI - Set to "true" for nightly/scheduled builds
#   BRANCH_NAME - Git branch name
#   CHANGE_ID - PR number (set by Jenkins Multibranch Pipeline for PRs)
#   CHANGE_TARGET - Base branch for PR builds (set by Jenkins Multibranch Pipeline)
#
# Note: CHANGE_ID may not be set even for PR builds if Jenkins job is not
# configured as Multibranch Pipeline. Script uses three-dot git diff syntax
# to detect only PR-specific changes (excluding merged commits from base branch).
#
# Manual override (set by developer/admin if needed):
#   DISABLE_SMART_BUILD - Set to "true" to force full build
#   BASE_BRANCH - Override base branch (default: "develop")

set -e

# Configuration
FORCE_FULL_BUILD=false
REASON=""
BASE_BRANCH="${CHANGE_TARGET:-${BASE_BRANCH:-develop}}"

# 1. Check if this is a nightly/scheduled build
# Existing Jenkins infrastructure sets FORCE_CI=true for cron-triggered builds
if [ "$FORCE_CI" = "true" ]; then
    FORCE_FULL_BUILD=true
    REASON="nightly/scheduled build (FORCE_CI=true from Jenkins cron)"
fi

# 2. Manual override to disable smart build
# Set DISABLE_SMART_BUILD=true in Jenkins job parameters if you want to force a full build
if [ "$DISABLE_SMART_BUILD" = "true" ]; then
    FORCE_FULL_BUILD=true
    REASON="manual override (DISABLE_SMART_BUILD=true)"
fi

# 3. Force full build if CMakeLists.txt or cmake/ configuration changed
# Always compare against base branch (not consecutive commits) to avoid false positives from merge commits
# Three-dot syntax (...) shows only changes unique to the current branch (excludes merged commits from base)
# This prevents false positives when the PR branch has merged in commits from develop
CHANGED_FILES=$(git diff --name-only origin/${BASE_BRANCH}...HEAD 2>/dev/null || echo "")

# Comprehensive pattern for build/infrastructure files that require full build:
# Scoped to composablekernel-specific paths only to avoid false positives from other projects
# - CMake: CMakeLists.txt, *.cmake, *.cmake.in within projects/composablekernel/
# - Scripts: Only build-critical scripts (dependency-parser, cmake utilities)
# - Compiler: .clang-format, .clang-tidy within projects/composablekernel/
# - Python: setup.py, pyproject.toml within projects/composablekernel/
BUILD_INFRA_PATTERN="(projects/composablekernel/.*CMakeLists\.txt"
BUILD_INFRA_PATTERN="${BUILD_INFRA_PATTERN}|projects/composablekernel/.*\.cmake$|projects/composablekernel/.*\.cmake\.in$"
BUILD_INFRA_PATTERN="${BUILD_INFRA_PATTERN}|projects/composablekernel/script/dependency-parser/"
BUILD_INFRA_PATTERN="${BUILD_INFRA_PATTERN}|projects/composablekernel/script/cmake/"
BUILD_INFRA_PATTERN="${BUILD_INFRA_PATTERN}|projects/composablekernel/setup\.py|projects/composablekernel/pyproject\.toml)"

if echo "$CHANGED_FILES" | grep -qE "${BUILD_INFRA_PATTERN}"; then
    FORCE_FULL_BUILD=true
    REASON="build system configuration changed"
fi

# 4. Force full build if dependency cache is older than 7 days
CACHE_FILE="cmake_dependency_mapping.json"
if [ -f "$CACHE_FILE" ]; then
    # Different stat command for Linux vs macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CACHE_MTIME=$(stat -f %m "$CACHE_FILE")
    else
        CACHE_MTIME=$(stat -c %Y "$CACHE_FILE")
    fi
    CURRENT_TIME=$(date +%s)
    CACHE_AGE_DAYS=$(( ($CURRENT_TIME - $CACHE_MTIME) / 86400 ))

    if [ $CACHE_AGE_DAYS -gt 7 ]; then
        FORCE_FULL_BUILD=true
        REASON="dependency cache older than 7 days"
    fi
fi

# Output decision
echo "========================================="
echo "Smart Build Safety Check"
echo "========================================="
echo "FORCE_CI: ${FORCE_CI:-false}"
echo "BRANCH_NAME: ${BRANCH_NAME:-unknown}"
echo "BASE_BRANCH: ${BASE_BRANCH}"
echo "CHANGE_ID: ${CHANGE_ID:-<not a PR>}"
echo "DISABLE_SMART_BUILD: ${DISABLE_SMART_BUILD:-false}"
echo "-----------------------------------------"

if [ "$FORCE_FULL_BUILD" = true ]; then
    echo "Decision: 🔴 FULL BUILD REQUIRED"
    echo "Reason: $REASON"
    echo "========================================="
    echo "export SMART_BUILD_MODE=full" > build_mode.env
    exit 1  # Exit with error to signal full build needed
else
    echo "Decision: 🟢 SELECTIVE BUILD ENABLED"
    echo "Using smart build for faster CI"
    echo "========================================="
    echo "export SMART_BUILD_MODE=selective" > build_mode.env
    exit 0  # Exit success to signal selective build OK
fi
