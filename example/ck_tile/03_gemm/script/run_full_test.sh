#!/bin/bash 
#
# in order to run this script you'd first need to build the tile_example_gemm executables in ../build/bin/
#
# run the script as "./run_full_test.sh <tag for your test environment> <branch name> <host name> <gpu_arch>
# input arguments: 
# environment tag  : a string describing the specifics of your test environment
# branch name      : name of the branch in git repo (git status | grep -e 'On branch')
# host name        : $hostname
# gpu architecture: e.g., gfx90a, or gfx942, etc.

# get the command line arguments:
export env_type=$1
echo 'Environment type: ' $env_type
export branch=$2
echo 'Branch name: ' $branch
export host_name=$3
echo 'Host name: ' $host_name
export GPU_arch=$4
echo 'GPU_arch: ' $GPU_arch

# run verification tests
example/ck_tile/03_gemm/script/smoke_test.sh

# We do not have a performance benchmark for gemm yet. Will add it in the future.