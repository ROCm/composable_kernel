#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Common utilities for CK Docker tools
# Shared configuration and helper functions

# Find project root using relative path (fallback)
get_project_root() {
    local script_dir="$1"
    cd "${script_dir}/../.." && pwd
}

# Detect git branch and sanitize for Docker naming
get_sanitized_branch() {
    local project_root="$1"
    local branch

    branch=$(cd "${project_root}" && git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '_' | tr -cd 'a-zA-Z0-9_-' || echo "")
    branch=${branch:-unknown}

    # Handle detached HEAD state
    if [ "${branch}" = "HEAD" ]; then
        branch="detached"
    fi

    echo "${branch}"
}

# Get username with fallback
get_username() {
    echo "${USER:-$(whoami 2>/dev/null || echo "user")}"
}

# Generate default container name: ck_<username>_<branch>
get_default_container_name() {
    local project_root="$1"
    local user_name
    local git_branch

    user_name=$(get_username)
    git_branch=$(get_sanitized_branch "${project_root}")

    echo "ck_${user_name}_${git_branch}"
}

# Get container name (respects CK_CONTAINER_NAME env var)
get_container_name() {
    local project_root="$1"
    local default_name

    default_name=$(get_default_container_name "${project_root}")
    echo "${CK_CONTAINER_NAME:-${default_name}}"
}

# Get Docker image (respects CK_DOCKER_IMAGE env var)
get_docker_image() {
    echo "${CK_DOCKER_IMAGE:-rocm/composable_kernel:ck_ub24.04_rocm7.0.1}"
}

# Check if container exists (exact match)
container_exists() {
    local name="$1"
    docker ps -a --filter "name=^${name}$" --format '{{.Names}}' | grep -q "^${name}$"
}

# Check if container is running (exact match)
container_is_running() {
    local name="$1"
    docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q "^${name}$"
}

# Detect GPU target in container
detect_gpu_target() {
    local container="$1"

    # Allow override via CK_GPU_TARGET environment variable
    if [ -n "${CK_GPU_TARGET:-}" ]; then
        echo "${CK_GPU_TARGET}"
        return 0
    fi

    docker exec "${container}" bash -c "
        rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | head -1 || echo 'gfx950'
    " | tr -d '\r\n'
}

# Ensure container is running, start if needed
ensure_container_running() {
    local container="$1"
    local script_dir="$2"

    if ! container_is_running "${container}"; then
        echo "Container '${container}' not running. Starting with ck-docker..."
        "${script_dir}/ck-docker" start "${container}"
    fi
}

# ============================================================================
# Native (non-Docker) utilities
# ============================================================================

# Output utilities
info()  { echo "[info] $*"; }
warn()  { echo "[warn] $*" >&2; }
error() { echo "[error] $*" >&2; }

# Require argument for option (validates $2 exists and is not another flag)
require_arg() {
    local option="$1"
    local value="$2"
    if [ -z "$value" ] || [[ "$value" == -* ]]; then
        error "Option $option requires an argument"
        exit 1
    fi
}

# Native GPU detection (no Docker required)
detect_gpu_native() {
    # Allow override via CK_GPU_TARGET environment variable
    if [ -n "${CK_GPU_TARGET:-}" ]; then
        echo "${CK_GPU_TARGET}"
        return 0
    fi

    # Try rocminfo if available
    if command -v rocminfo &>/dev/null; then
        local gpu
        gpu=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | head -1)
        if [ -n "$gpu" ]; then
            echo "$gpu"
            return 0
        fi
    fi

    # Fallback
    echo "gfx950"
}

# Get build directory (respects CK_BUILD_DIR env var)
get_build_dir() {
    local project_root="${1:-$(get_project_root "$(dirname "${BASH_SOURCE[0]}")")}"
    echo "${CK_BUILD_DIR:-${project_root}/build}"
}

# Check if build is configured (build.ninja exists)
is_build_configured() {
    local build_dir="${1:-$(get_build_dir)}"
    [ -f "${build_dir}/build.ninja" ]
}

# Find project root from any subdirectory (walks up to find .ck-project-root)
find_project_root() {
    local dir="${1:-$(pwd)}"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/.ck-project-root" ]; then
            echo "$dir"
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

# List available CMake presets
list_cmake_presets() {
    local project_root="${1:-$(find_project_root)}"
    local presets_file="${project_root}/CMakePresets.json"

    if [ ! -f "$presets_file" ]; then
        return 1
    fi

    # Extract non-hidden preset names
    if command -v jq &>/dev/null; then
        jq -r '.configurePresets[] | select(.hidden != true) | .name' "$presets_file" 2>/dev/null
    else
        # Fallback: sed-based extraction (more portable than grep -P)
        sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$presets_file" | grep -v '^use-'
    fi
}
