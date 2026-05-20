# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# CShuffleLds LDS store/load microbenchmark suite
# Measures LDS bandwidth and bank conflicts for different MFMA configurations

set(GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_SOURCE_DIR}")

# Core function: generate and build a benchmark executable
function(add_cshuffle_lds_benchmark NAME A_TYPE B_TYPE ACC_TYPE O_TYPE M N M_WAVE N_WAVE M_XDL N_XDL K_XDL CONFIG_NAME)
    set(GENERATED_SOURCE "${GENERATED_SOURCE_DIR}/${NAME}.cpp")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/benchmark_template.cpp.in" "${GENERATED_SOURCE}" @ONLY)
    set_source_files_properties(${GENERATED_SOURCE} PROPERTIES LANGUAGE HIP)
    add_executable(${NAME} ${GENERATED_SOURCE})
    set_property(TARGET ${NAME} PROPERTY HIP_ARCHITECTURES ${SUPPORTED_GPU_TARGETS})
    target_include_directories(${NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/test ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${NAME} PRIVATE hip::device)
    if(CK_USE_OCP_FP8)
        target_compile_options(${NAME} PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endfunction()

# Type-specific wrappers (derive name and config from parameters)
function(add_fp16_benchmark M N M_WAVE N_WAVE M_XDL N_XDL K_XDL)
    set(NAME "bench_lds_fp16_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}")
    set(CONFIG "FP16_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}")
    add_cshuffle_lds_benchmark(${NAME} "ck_tile::half_t" "ck_tile::half_t" "float" "ck_tile::half_t"
        ${M} ${N} ${M_WAVE} ${N_WAVE} ${M_XDL} ${N_XDL} ${K_XDL} ${CONFIG})
endfunction()

function(add_fp8_fp16_benchmark M N M_WAVE N_WAVE M_XDL N_XDL K_XDL)
    set(NAME "bench_lds_fp8_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}_fp16")
    set(CONFIG "FP8_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}_fp16")
    add_cshuffle_lds_benchmark(${NAME} "ck_tile::fp8_t" "ck_tile::fp8_t" "float" "ck_tile::half_t"
        ${M} ${N} ${M_WAVE} ${N_WAVE} ${M_XDL} ${N_XDL} ${K_XDL} ${CONFIG})
endfunction()

function(add_fp8_fp8_benchmark M N M_WAVE N_WAVE M_XDL N_XDL K_XDL)
    set(NAME "bench_lds_fp8_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}_fp8")
    set(CONFIG "FP8_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}_fp8")
    add_cshuffle_lds_benchmark(${NAME} "ck_tile::fp8_t" "ck_tile::fp8_t" "float" "ck_tile::fp8_t"
        ${M} ${N} ${M_WAVE} ${N_WAVE} ${M_XDL} ${N_XDL} ${K_XDL} ${CONFIG})
endfunction()

function(add_fp32_benchmark M N M_WAVE N_WAVE M_XDL N_XDL K_XDL)
    set(NAME "bench_lds_fp32_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}")
    set(CONFIG "FP32_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}")
    add_cshuffle_lds_benchmark(${NAME} "float" "float" "float" "float"
        ${M} ${N} ${M_WAVE} ${N_WAVE} ${M_XDL} ${N_XDL} ${K_XDL} ${CONFIG})
endfunction()

function(add_bf16_benchmark M N M_WAVE N_WAVE M_XDL N_XDL K_XDL)
    set(NAME "bench_lds_bf16_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}")
    set(CONFIG "BF16_${M_XDL}x${N_XDL}x${K_XDL}_${M_WAVE}x${N_WAVE}")
    add_cshuffle_lds_benchmark(${NAME} "ck_tile::bf16_t" "ck_tile::bf16_t" "float" "ck_tile::bf16_t"
        ${M} ${N} ${M_WAVE} ${N_WAVE} ${M_XDL} ${N_XDL} ${K_XDL} ${CONFIG})
endfunction()

# Helper to add benchmarks for all wave layouts of a given MFMA tile
# Block tile M = M_XDL * M_WAVE, N = N_XDL * N_WAVE (must be divisible, here we use single iteration)
macro(add_benchmarks_for_mfma FUNC M_XDL N_XDL K_XDL)
    foreach(WAVE_LAYOUT "4;1" "2;2" "1;4")
        list(GET WAVE_LAYOUT 0 M_WAVE)
        list(GET WAVE_LAYOUT 1 N_WAVE)
        math(EXPR M "${M_XDL} * ${M_WAVE}")
        math(EXPR N "${N_XDL} * ${N_WAVE}")
        cmake_language(CALL ${FUNC} ${M} ${N} ${M_WAVE} ${N_WAVE} ${M_XDL} ${N_XDL} ${K_XDL})
    endforeach()
endmacro()

#
# FP32 benchmarks
#
# MFMA tiles: 32x32x4, 32x32x8, 16x16x4, 16x16x8, 16x16x16
add_benchmarks_for_mfma(add_fp32_benchmark 32 32 4)
add_benchmarks_for_mfma(add_fp32_benchmark 32 32 8)
add_benchmarks_for_mfma(add_fp32_benchmark 16 16 4)
add_benchmarks_for_mfma(add_fp32_benchmark 16 16 8)
add_benchmarks_for_mfma(add_fp32_benchmark 16 16 16)

#
# FP16 benchmarks
#
# MFMA tiles: 32x32x8, 32x32x16, 16x16x16, 4x64x16, 64x4x16
add_benchmarks_for_mfma(add_fp16_benchmark 32 32 8)
add_benchmarks_for_mfma(add_fp16_benchmark 32 32 16)
add_benchmarks_for_mfma(add_fp16_benchmark 16 16 16)
add_benchmarks_for_mfma(add_fp16_benchmark 4 64 16)
add_benchmarks_for_mfma(add_fp16_benchmark 64 4 16)

#
# FP8 -> FP16 benchmarks
#
# MFMA tiles: 32x32x16, 16x16x32
add_benchmarks_for_mfma(add_fp8_fp16_benchmark 32 32 16)
add_benchmarks_for_mfma(add_fp8_fp16_benchmark 16 16 32)

#
# FP8 -> FP8 benchmarks
#
# MFMA tiles: 32x32x16, 16x16x32
add_benchmarks_for_mfma(add_fp8_fp8_benchmark 32 32 16)
add_benchmarks_for_mfma(add_fp8_fp8_benchmark 16 16 32)

#
# gfx950-only configurations
#
if(SUPPORTED_GPU_TARGETS MATCHES "gfx950")
    # FP16: 16x16x32
    add_benchmarks_for_mfma(add_fp16_benchmark 16 16 32)

    # BF16: 16x16x64 (gfx950-only, uses 16x16x32 base instruction)
    # Other BF16 tiles have same LDS behavior as FP16 since both are 2-byte types
    add_benchmarks_for_mfma(add_bf16_benchmark 16 16 64)

    # FP8 -> FP16: 32x32x32, 32x32x64, 16x16x64, 16x16x128
    add_benchmarks_for_mfma(add_fp8_fp16_benchmark 32 32 32)
    add_benchmarks_for_mfma(add_fp8_fp16_benchmark 32 32 64)
    add_benchmarks_for_mfma(add_fp8_fp16_benchmark 16 16 64)
    add_benchmarks_for_mfma(add_fp8_fp16_benchmark 16 16 128)

    # FP8 -> FP8: 32x32x32, 32x32x64, 16x16x64, 16x16x128
    add_benchmarks_for_mfma(add_fp8_fp8_benchmark 32 32 32)
    add_benchmarks_for_mfma(add_fp8_fp8_benchmark 32 32 64)
    add_benchmarks_for_mfma(add_fp8_fp8_benchmark 16 16 64)
    add_benchmarks_for_mfma(add_fp8_fp8_benchmark 16 16 128)
endif()
