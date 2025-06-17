#!/bin/bash

run_and_check() {
    "$@"
    status=$?
    if [ $status -ne 0 ]; then
        echo "Error with \"$@\": Exited with status $status"
        exit $status
    fi
    return $status
}

# Function to detect if running in Docker
is_docker() {
    # Method 1: Check for .dockerenv file (most reliable)
    if [ -f /.dockerenv ]; then
        return 0
    fi
    
    # Method 2: Check cgroup for docker (fallback)
    if [ -f /proc/1/cgroup ] && grep -q docker /proc/1/cgroup 2>/dev/null; then
        return 0
    fi
    
    # Method 3: Check for container environment variable
    if [ -n "${container}" ] || [ -n "${DOCKER_CONTAINER}" ]; then
        return 0
    fi
    
    return 1
}

echo "I: Installing tools required for pre-commit checks..."
run_and_check apt install clang-format-12

echo "I: Installing pre-commit itself..."

# Check if running in Docker and handle pip installation accordingly
if is_docker; then
    echo "I: Docker environment detected - using --break-system-packages"
    run_and_check pip3 install --break-system-packages pre-commit
else
    echo "I: Host environment detected - using standard pip install"
    run_and_check pip3 install pre-commit
fi

run_and_check pre-commit install

echo "I: Installation successful."
