# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# CK Tile Dispatcher Tests (C++ and Python)
# =============================================================================

cmake_minimum_required(VERSION 3.16)

# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Bridge on-device GPU correctness test (SKIPs cleanly without a GPU/built lib).
add_test(
    NAME dispatcher_test_preshuffle_gpu_correctness
    # Run the script directly: this module is script-style (a bare function plus
    # a main() guard), not unittest.TestCase, so `unittest discover` collects 0
    # tests -- and unittest exits NONZERO on "NO TESTS RAN", turning an
    # unrunnable box into a hard ctest failure. main() skips (exit 77) without a
    # GPU/built lib and returns nonzero only on a real correctness failure.
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_preshuffle_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_preshuffle_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    # Without this a CPU-only runner reports a green PASS for a test that never
    # reached the GPU; 77 makes ctest record it as skipped instead.
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Bridge on-device GPU correctness test (SKIPs cleanly without a GPU/built lib).
add_test(
    NAME dispatcher_test_batched_gemm_gpu_correctness
    # Run the script directly: this module is script-style (a parameterized
    # function + main() guard), not unittest.TestCase, so `unittest discover`
    # would collect 0 tests and pass vacuously. main() skips (exit 77) without a
    # GPU/built lib and returns nonzero only on a real correctness failure.
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_batched_gemm_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_batched_gemm_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    # Without this a CPU-only runner reports a green PASS for a test that never
    # reached the GPU; 77 makes ctest record it as skipped instead.
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Bridge on-device GPU correctness test (SKIPs cleanly without a GPU/built lib).
add_test(
    NAME dispatcher_test_batched_contraction_gpu_correctness
    # Run the script directly: this module is script-style (a parameterized
    # function + main() guard), not unittest.TestCase, so `unittest discover`
    # would collect 0 tests and pass vacuously. main() skips (exit 77) without a
    # GPU/built lib and returns nonzero only on a real correctness failure.
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_batched_contraction_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_batched_contraction_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    # Without this a CPU-only runner reports a green PASS for a test that never
    # reached the GPU; 77 makes ctest record it as skipped instead.
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Bridge on-device GPU correctness test (SKIPs cleanly without a GPU/built lib).
add_test(
    NAME dispatcher_test_tensor_quant_gpu_correctness
    # Run the script directly rather than under pytest: pytest exits 0 when every
    # test skips, so a CPU-only or non-fp8 runner reported a green PASS for work
    # that never reached the GPU. main() exits 77 on those boxes instead.
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_quant_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_tensor_quant_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Bridge on-device GPU correctness test (SKIPs cleanly without a GPU/hipcc).
add_test(
    NAME dispatcher_test_contraction_multi_abd_gpu_correctness
    # Run the script directly: this module is script-style (a parameterized
    # function + main() guard), not unittest.TestCase, so `unittest discover`
    # would collect 0 tests and pass vacuously. main() skips (exit 77) without a
    # GPU/hipcc and returns nonzero only on a real correctness failure.
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_contraction_multi_abd_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_contraction_multi_abd_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    # Builds a kernel from source via hipcc before running; needs more than the
    # 600s the sibling bridges use.
    TIMEOUT 1800
    # Without this a CPU-only runner reports a green PASS for a test that never
    # reached the GPU; 77 makes ctest record it as skipped instead.
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# =============================================================================
# Python Tests
# =============================================================================

# Auto-correction and validation stress test
add_test(
    NAME dispatcher_test_autocorrect
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_autocorrect.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_autocorrect PROPERTIES
    LABELS "dispatcher;python;validation"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Operator tile-constraints table regression test
add_test(
    NAME dispatcher_test_arch_filter_constraints
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_arch_filter_constraints.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_arch_filter_constraints PROPERTIES
    LABELS "dispatcher;python;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# codegen_common shared-infrastructure regression test. Covers normalize_gfx_arch()
# and the EXACT-gfx1250 RowColQuant/TensorQuant tile selector: a family-wide match
# there would hand gfx1200/gfx1201 a tile that compiles cleanly and returns garbage,
# which no downstream check catches.
add_test(
    NAME dispatcher_test_codegen_common
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/test_codegen_common.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_codegen_common PROPERTIES
    LABELS "dispatcher;python;codegen;validation"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Arch-string handling in tile_engine's gemm_validation_utils, which seven GEMM
# instance builders and the dispatcher's own warp gate all share. Registered on the
# dispatcher side because the bridge/Old-TE instance-set parity that depends on it
# is a dispatcher property; tile_engine has no CTest registration of its own.
add_test(
    NAME dispatcher_test_gemm_validation_arch_normalization
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_validation_arch_normalization.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_gemm_validation_arch_normalization PROPERTIES
    LABELS "dispatcher;python;validation"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Arch-gate predicate used by the RowColQuant/TensorQuant GPU tests. CPU-only: it
# exercises the string predicate, not a device. Worth its own registration because
# the failure mode it guards is a SKIP -- CI stays green while covering nothing.
add_test(
    NAME dispatcher_test_gpu_test_arch_gate
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_test_arch_gate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_gpu_test_arch_gate PROPERTIES
    LABELS "dispatcher;python;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# dispatcher_common unit tests: path helpers, validators, and -- since this is
# the only place they are covered -- the fp8-encoding arch predicates
# (fp8_uses_ocp / ocp_arch_defines / arch_feature_defines). Those decide whether
# -DCK_USE_OCP_FP8 reaches hipcc; getting them wrong makes host and device
# disagree on the fp8 encoding with no build error, so they need a test that
# actually runs. This file was previously not registered at all.
add_test(
    NAME dispatcher_test_dispatcher_common
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p test_dispatcher_common.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_dispatcher_common PROPERTIES
    LABELS "dispatcher;python;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Architecture-aware LDS staging budget regression test
add_test(
    NAME dispatcher_test_lds_capacity_arch_aware
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_lds_capacity_arch_aware.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_lds_capacity_arch_aware PROPERTIES
    LABELS "dispatcher;python;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# gemm_utils host-helper + runner-shape regression test. Also acts as an import
# canary: a merge that truncates python/gemm_utils.py (see #9308) makes this fail.
add_test(
    NAME dispatcher_test_gemm_utils
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p test_gemm_utils.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_gemm_utils PROPERTIES
    LABELS "dispatcher;python;gemm;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# =============================================================================
# Bridge on-device GPU correctness tests (merged ops). Each builds the op's
# dispatcher .so, runs on-device via its Gpu*Runner, and checks output vs a
# numpy fp32 reference (scale-relative tolerance). They SKIP cleanly (no fail)
# when no GPU / hipcc / static lib is present, so they cost nothing on a
# CPU-only runner and only run on a GPU runner. mx arch-gates to gfx950.
#
# These three are unittest.TestCase-based and skip themselves in setUp, so they
# are driven through run_unittest_77.py rather than `python -m unittest
# discover` directly. discover exits 0 when every test skipped, which made ctest
# report Passed for a run that asserted nothing -- a CPU-only runner showed the
# same green tick as a real GPU run. The driver keeps the discovery semantics and
# only maps an all-skipped run onto 77, which SKIP_RETURN_CODE turns into
# "Skipped". multi_d is script-style-77 in its own right and is registered below.
# =============================================================================
foreach(_gpu_test
        test_grouped_gemm_gpu_correctness
        test_multi_abd_gpu_correctness
        test_mx_gemm_gpu_correctness)
    add_test(
        NAME dispatcher_${_gpu_test}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_unittest_77.py ${_gpu_test}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(dispatcher_${_gpu_test} PROPERTIES
        LABELS "dispatcher;python;bridge;gpu"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
    )
endforeach()

# Multi-D GEMM: E = MultiDAdd(A @ B, D0, D1) over all four A/B layouts.
# Invoked as a script, not via `unittest discover`: it returns 77 on a missing
# GPU / static lib / hipcc, which SKIP_RETURN_CODE turns into ctest "Skipped".
# The discover path above exits 0 on an all-skipped run and would report Passed.
#
# TIMEOUT is 1800, not the 600 its neighbours use: the layout sweep is four
# separate hipcc builds (setup_multiple_gemm_dispatchers runs once per layout),
# so the wall-clock budget is roughly 4x theirs.
add_test(
    NAME dispatcher_test_multi_d_gpu_correctness
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_d_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_multi_d_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 1800
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Config search-space sweep: enumerate configs for a variant, build and run a
# sampled subset, verify each against a numpy reference. Previously registered
# nowhere -- it was reachable only from the Jenkins dispatcher lane, so a local
# `ctest` never exercised it and neither did any pre-merge gate.
#
# --budget 16, one per (dtype x layout) stratum: the sampler splits the budget
# evenly across the 16 strata the standard variant enumerates, so anything below
# 16 leaves whole strata unsampled on any given run. At --budget 4 the run was
# not a small slice of the space but a slice of a quarter of it, and no single
# run ever compiled an fp8 or bf8 config. Each sampled config is its own hipcc
# build, hence the 1800s timeout.
#
# The runner returns 77 when no GPU is present and --arch is not given, so
# SKIP_RETURN_CODE keeps a CPU-only runner honest instead of green. That skip
# path and the sampler's stratum coverage are both pinned by
# test_search_space_sampling.py, which needs no GPU.
add_test(
    NAME dispatcher_test_gemm_search_space
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_search_space.py --budget 16
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_gemm_search_space PROPERTIES
    LABELS "dispatcher;python;gemm;gpu"
    TIMEOUT 1800
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Unit tests for the search-space runner's sampler and skip path. Pure CPU, no
# hipcc and no built kernels, so unlike the sweep above these run on every
# machine and gate every build.
#
# Registered separately because neither behaviour is observable from the sweep's
# own result: a sampler that draws from only a quarter of the strata still
# passes every config it does run, and a broken skip path shows up as a green
# GPU lane, not a red one. Both regressed silently once already.
add_test(
    NAME dispatcher_test_search_space_sampling
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/test_search_space_sampling.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_search_space_sampling PROPERTIES
    LABELS "dispatcher;python;gemm;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# =============================================================================
# RowColQuant / TensorQuant bridge tests.
#
# These are pytest-style (bare test functions and the tmp_path fixture), not
# unittest.TestCase, so they are invoked via `-m pytest` like the examples
# integration tests below rather than via `unittest discover` -- discovery would
# silently collect zero tests and report success.
#
# The *_utils suites are pure Python (naming, config, codegen text) and run
# anywhere, and stay on `-m pytest`.
#
# The *_gpu_correctness suites are dual-mode: importable by pytest, but also
# runnable as scripts with a `main()` that returns 77 to skip. ctest invokes
# them as scripts, because `-m pytest` reports exit 0 -- i.e. PASSED -- when
# every test in the file skips, which would hide a missing GPU behind a green
# result. Run as scripts with SKIP_RETURN_CODE 77 they report Skipped instead.
# =============================================================================
foreach(_quant_test
        test_grouped_gemm_rowcolquant_utils
        test_grouped_gemm_tensorquant_utils
        test_quant_review_regressions)
    add_test(
        NAME dispatcher_${_quant_test}
        COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/${_quant_test}.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(dispatcher_${_quant_test} PROPERTIES
        LABELS "dispatcher;python;quant;validation"
        TIMEOUT 120
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
    )
endforeach()

foreach(_quant_gpu_test
        test_rowcolquant_gpu_correctness
        test_tensorquant_gpu_correctness)
    add_test(
        NAME dispatcher_${_quant_gpu_test}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${_quant_gpu_test}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(dispatcher_${_quant_gpu_test} PROPERTIES
        LABELS "dispatcher;python;bridge;quant;gpu"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
    )
endforeach()

# Verbose version of the test
add_test(
    NAME dispatcher_test_autocorrect_verbose
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_autocorrect.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_autocorrect_verbose PROPERTIES
    LABELS "dispatcher;python;validation;verbose"
    TIMEOUT 180
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Individual Python Test Categories
add_test(
    NAME dispatcher_test_gemm_validation
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestGemmValidation test_autocorrect.TestGemmExpansion -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_gemm_validation PROPERTIES
    LABELS "dispatcher;python;gemm;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_python_autocorrect
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestPythonAutoCorrect -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_python_autocorrect PROPERTIES
    LABELS "dispatcher;python;autocorrect"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_stress
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestStressRandom -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_stress PROPERTIES
    LABELS "dispatcher;python;stress"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_arch_support
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestArchitectureSupport -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_arch_support PROPERTIES
    LABELS "dispatcher;python;arch"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_fmha_utils
    COMMAND ${Python3_EXECUTABLE} -m unittest test_fmha_utils -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_fmha_utils PROPERTIES
    LABELS "dispatcher;python;fmha;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_fmha_codegen
    COMMAND ${Python3_EXECUTABLE} -m unittest test_fmha_codegen -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_fmha_codegen PROPERTIES
    LABELS "dispatcher;python;fmha;codegen"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_fmha_rules
    COMMAND ${Python3_EXECUTABLE} -m unittest test_fmha_rules -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_fmha_rules PROPERTIES
    LABELS "dispatcher;python;fmha;rules"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# FMHA parity test (requires GPU)
add_test(
    NAME dispatcher_test_fmha_parity
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fmha_parity.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_fmha_parity PROPERTIES
    LABELS "dispatcher;python;fmha;parity;gpu"
    TIMEOUT 600
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Stream-K deep-core registry test (requires GPU + hipcc; SKIPs otherwise).
# Pass the gfx target CMake already configured with so the test does not have to
# detect it at runtime (no rocminfo dependency); it falls back to ROCm env vars
# / amdgpu-arch if none is set here.
set(_streamk_test_arch "")
if(GPU_TARGETS)
    list(GET GPU_TARGETS 0 _streamk_test_arch)
elseif(CMAKE_HIP_ARCHITECTURES)
    list(GET CMAKE_HIP_ARCHITECTURES 0 _streamk_test_arch)
elseif(AMDGPU_TARGETS)
    list(GET AMDGPU_TARGETS 0 _streamk_test_arch)
endif()
set(_streamk_arch_arg "")
if(_streamk_test_arch)
    set(_streamk_arch_arg --arch ${_streamk_test_arch})
endif()

add_test(
    NAME dispatcher_test_streamk_registry
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_streamk_registry.py ${_streamk_arch_arg}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_streamk_registry PROPERTIES
    LABELS "dispatcher;python;streamk;gpu"
    # 96 cases (4 dtypes x 4 layouts x 3 strategies x 2 shapes), each a separate
    # hipcc build of the driver -- compile time dominates, so the budget scales
    # with the matrix, not with GPU speed. Measured end-to-end at ~55 min on a
    # local box; 900 was not survivable by any arch.
    TIMEOUT 4200
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# GPU numeric-correctness tests for the quantized / Stream-K bridges.
#
# These are script-style (argparse + `return 77` to skip), not unittest, so each
# needs its own add_test + SKIP_RETURN_CODE pair -- they cannot join the
# `unittest discover` foreach above, which has no skip code and a fixed timeout.
#
# Deliberately NOT given a --gfx argument: each test hipcc-compiles kernels with
# --offload-arch and then runs them, so the arch must match the PHYSICAL GPU.
# rocminfo auto-detection is the right source for that; the CMake-configured
# GPU_TARGETS may legitimately differ from the card in the box. Every one of
# these exits 77 (Skipped, not Passed) when no GPU is present or the detected
# arch is not the one the operator supports.

# Stream-K: atomic / linear / tree reduction through the ctypes bridge (gfx942).
# 12 hipcc builds plus a 4096-cubed reference, hence the larger timeout.
add_test(
    NAME dispatcher_test_streamk_gpu_correctness
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_streamk_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_streamk_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 900
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# AQuant grouped GEMM: fp8 / bf8 per-(m,k)-group A-side scales (gfx950).
add_test(
    NAME dispatcher_test_aquant_gpu_correctness
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_aquant_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_aquant_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# BQuant grouped GEMM: C4 (fp8/bf8) + H3 (mx_*) B-side scales (gfx950).
# The test already existed but ran only from the Jenkins gfx950 lane; this makes
# it reachable from ctest too.
add_test(
    NAME dispatcher_test_bquant_gpu_correctness
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bquant_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_bquant_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# ABQuant grouped GEMM: fp8 / bf8 scales on BOTH operands, compv3 + the
# gfx950-native eightwaves pipeline (gfx950).
add_test(
    NAME dispatcher_test_abquant_gpu_correctness
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_abquant_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_abquant_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Stress Test Script
add_test(
    NAME dispatcher_stress_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/stress_test_autocorrect.py
        --arch gfx942 --samples 30 --seed 42
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_stress_test PROPERTIES
    LABELS "dispatcher;python;stress;integration"
    TIMEOUT 180
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# =============================================================================
# Integration Tests (mimic examples)
# =============================================================================

# Full integration test suite
add_test(
    NAME dispatcher_integration_tests
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_examples_integration.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_integration_tests PROPERTIES
    LABELS "dispatcher;python;integration;examples"
    TIMEOUT 600
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Quick integration test (utilities only)
add_test(
    NAME dispatcher_integration_quick
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_examples_integration.py::TestUtilityImports -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_integration_quick PROPERTIES
    LABELS "dispatcher;python;integration;quick"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# GEMM examples integration
add_test(
    NAME dispatcher_integration_gemm
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_examples_integration.py::TestGemmPythonExamples -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_integration_gemm PROPERTIES
    LABELS "dispatcher;python;integration;gemm"
    TIMEOUT 300
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# =============================================================================
# C++ Tests (Google Test)
# =============================================================================

# Include Google Test setup
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gtest.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gtest.cmake)
else()
    include(gtest)
endif()

# Mock kernel instance for testing (shared across tests)
add_library(dispatcher_test_utils STATIC
    test_mock_kernel.cpp
)

target_include_directories(dispatcher_test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(dispatcher_test_utils PRIVATE
    ck_tile_dispatcher
)

# Test executables using Google Test
set(TEST_SOURCES
    # Core unit tests
    test_kernel_key.cpp
    test_problem.cpp
    test_registry.cpp
    test_dispatcher.cpp
    test_tile_backend.cpp
    test_fmha_problem.cpp
    test_fmha_dispatcher.cpp
    test_fmha_registry.cpp
    
    # Extended unit tests (more comprehensive coverage)
    test_kernel_key_extended.cpp
    test_problem_extended.cpp
    test_registry_extended.cpp
    test_dispatcher_extended.cpp
    
    # Regression tests (known issues and edge cases)
    test_regression.cpp
    
    # JSON export tests
    test_json_export.cpp
)

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    
    add_executable(${test_name} ${test_source})
    
    target_link_libraries(${test_name} PRIVATE
        ck_tile_dispatcher
        dispatcher_test_utils
        GTest::gtest_main
    )
    
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )
    
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "dispatcher;cpp;unit")
endforeach()

# Standalone integration tests (with their own main())
set(STANDALONE_TESTS
    test_minimal.cpp
    test_grouped_conv_config.cpp
    test_grouped_conv_problem.cpp
    test_grouped_conv_kernel_decl.cpp
    test_grouped_conv_registry.cpp
    test_fmha_kernel_decl.cpp
)

foreach(test_source ${STANDALONE_TESTS})
    get_filename_component(test_name ${test_source} NAME_WE)
    
    add_executable(${test_name} ${test_source})
    
    target_link_libraries(${test_name} PRIVATE
        ck_tile_dispatcher
        dispatcher_test_utils
    )
    
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )
    
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "dispatcher;cpp;integration")
endforeach()

# =============================================================================
# Real Kernel Tests (requires generated kernels)
# =============================================================================

set(KERNEL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../generated_kernels")
set(KERNEL_REGISTRATION_HEADER "${KERNEL_OUTPUT_DIR}/dispatcher_wrappers/register_all_kernels.hpp")
set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py")

option(BUILD_DISPATCHER_REAL_KERNEL_TESTS "Build tests with real GPU kernels" ON)

if(BUILD_DISPATCHER_REAL_KERNEL_TESTS AND EXISTS "${CODEGEN_SCRIPT}")
    message(STATUS "Setting up real kernel test generation")
    
    add_custom_command(
        OUTPUT ${KERNEL_REGISTRATION_HEADER}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_OUTPUT_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CODEGEN_SCRIPT}
            --output-dir ${KERNEL_OUTPUT_DIR}
            --datatype fp16
            --layout rcr
            --gpu-target gfx942
            --preselected fp16_rcr_essential
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
        COMMENT "Generating CK Tile kernels for real kernel tests..."
        VERBATIM
    )
    
    add_custom_target(generate_test_kernels DEPENDS ${KERNEL_REGISTRATION_HEADER})
    
    set(SINGLE_KERNEL_HEADER "${KERNEL_OUTPUT_DIR}/gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp")
    
    set(REAL_KERNEL_TESTS
        test_real_kernel_simple
        test_real_kernel_multi_size
        test_real_kernel_performance
        test_real_kernel_correctness
        test_sanity_ck_tile
    )
    
    if(EXISTS "${SINGLE_KERNEL_HEADER}")
        foreach(test_name ${REAL_KERNEL_TESTS})
            add_executable(${test_name} ${test_name}.cpp)
            
            add_dependencies(${test_name} generate_test_kernels)
            
            target_link_libraries(${test_name} PRIVATE
                ck_tile_dispatcher
            )
            
            target_include_directories(${test_name} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                ${KERNEL_OUTPUT_DIR}
            )
            
            target_compile_options(${test_name} PRIVATE
                -include ${SINGLE_KERNEL_HEADER}
                -mllvm -enable-noalias-to-md-conversion=0
                -Wno-undefined-func-template
                -Wno-float-equal
                --offload-compress
            )
            
            if(hip_FOUND)
                target_link_libraries(${test_name} PRIVATE hip::device hip::host)
            endif()
            
            add_test(NAME ${test_name} COMMAND ${test_name})
            set_tests_properties(${test_name} PROPERTIES LABELS "dispatcher;cpp;gpu;kernel")
        endforeach()
    endif()
endif()

# =============================================================================
# Custom Targets
# =============================================================================

add_custom_target(run_dispatcher_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L dispatcher --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all dispatcher tests"
)

add_custom_target(test_dispatcher_python
    COMMAND ${CMAKE_CTEST_COMMAND} -L "dispatcher;python" --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Python dispatcher tests"
)

add_custom_target(test_dispatcher_cpp
    COMMAND ${CMAKE_CTEST_COMMAND} -L "dispatcher;cpp" --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running C++ dispatcher tests"
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "Dispatcher tests configured:")
message(STATUS "  Run all:        ctest -L dispatcher")
message(STATUS "  Run Python:     ctest -L 'dispatcher;python' or make test_dispatcher_python")
message(STATUS "  Run C++:        ctest -L 'dispatcher;cpp' or make test_dispatcher_cpp")
message(STATUS "  Run verbose:    ctest -R dispatcher_test_autocorrect_verbose")


# Bridge on-device GPU correctness test (SKIPs cleanly without a GPU/built lib).
# NON-GROUPED gemm_rowcolquant. The GROUPED grouped_gemm_rowcolquant GPU test is
# registered above in the _quant_gpu_test foreach as
# dispatcher_test_rowcolquant_gpu_correctness -- distinct test, distinct file.
add_test(
    NAME dispatcher_test_gemm_rowcolquant_gpu_correctness
    # Run the script directly rather than under pytest: pytest exits 0 when every
    # test skips, so a CPU-only or non-fp8 runner reported a green PASS for work
    # that never reached the GPU. main() exits 77 on those boxes instead.
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_rowcolquant_gpu_correctness.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_gemm_rowcolquant_gpu_correctness PROPERTIES
    LABELS "dispatcher;python;bridge;gpu"
    TIMEOUT 600
    SKIP_RETURN_CODE 77
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Same shape, for the NON-GROUPED gemm_{a,ab,b}quant bridges. The GROUPED
# counterparts are registered above in the _quant_gpu_test foreach as
# dispatcher_test_{a,ab,b}quant_gpu_correctness -- distinct tests, distinct
# files, distinct bridges. The names differ by one token; deleting a
# "duplicate-looking" entry here silently drops a whole operator.
#
# No --gfx argument, deliberately: ctest must autodetect via rocminfo so the
# JIT build matches the physical GPU (see the note above the C++ GPU tests).
# --gfx is for the Jenkins lane only, where the arch is known from the node
# label. Each script self-skips with 77 on an unsupported arch -- gfx942/gfx950
# for aquant and abquant, gfx950 only for bquant.
foreach(_gemm_quant_gpu_test
        test_gemm_aquant_gpu_correctness
        test_gemm_abquant_gpu_correctness
        test_gemm_bquant_gpu_correctness)
    add_test(
        NAME dispatcher_${_gemm_quant_gpu_test}
        # Run the script directly rather than under pytest: pytest exits 0 when
        # every test skips, so a CPU-only or non-fp8 runner would report a green
        # PASS for work that never reached the GPU. main() exits 77 instead.
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${_gemm_quant_gpu_test}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(dispatcher_${_gemm_quant_gpu_test} PROPERTIES
        LABELS "dispatcher;python;bridge;gpu"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
    )
endforeach()

# =============================================================================
# Block-scale quant bridge tests (per-op unit tests + self-tests)
#
# The *_gpu_correctness counterparts are registered above, next to the other
# script-style tests that need SKIP_RETURN_CODE 77.
# =============================================================================

# Per-op bridge name-contract / codegen / scope unit tests (CPU-only, unittest).
foreach(_bs_op tensor_quant rowcolquant aquant abquant bquant)
    add_test(
        NAME dispatcher_test_${_bs_op}_bridge
        COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p test_${_bs_op}_bridge.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(dispatcher_test_${_bs_op}_bridge PROPERTIES
        LABELS "dispatcher;python;bridge"
        TIMEOUT 120
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
    )
endforeach()

# gemm_bquant CPU self-test (no GPU / no hipcc required).
add_test(
    NAME dispatcher_test_gemm_bquant_selftest
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_bquant_selftest.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_gemm_bquant_selftest PROPERTIES
    LABELS "dispatcher;python;bridge"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# grouped_gemm_bquant utils tests (CPU, pytest).
add_test(
    NAME dispatcher_test_grouped_gemm_bquant_utils
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/test_grouped_gemm_bquant_utils.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(dispatcher_test_grouped_gemm_bquant_utils PROPERTIES
    LABELS "dispatcher;python;bridge"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)
