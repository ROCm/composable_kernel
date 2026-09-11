# SPDX-License-Identifier: MIT
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
#list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm -enable-noalias-to-md-conversion=0")
#list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -Wno-unused-local-typedef)
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS --save-temps=obj -Wno-gnu-line-marker)
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -amdgpu-hard-clause-length-limit=1)
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -Xarch_device -mllvm=-amdgpu-kernarg-preload-count=16)
#list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS "SHELL: -Rpass-analysis=kernel-resource-usage ")

set(GEMM_XDL_BENCHMARK gemm_xdl_benchmark)
set(GEMM_XDL_BENCHMARK_SRC gemm_xdl_benchmark.cpp)

set(MX_GEMM_XDL_BENCHMARK mx_gemm_xdl_benchmark)
set(MX_GEMM_XDL_BENCHMARK_SRC mx_gemm_xdl_benchmark.cpp)

set(WP_GEMM_XDL_BENCHMARK wp_gemm_xdl_benchmark)
set(WP_GEMM_XDL_BENCHMARK_SRC wp_gemm_xdl_benchmark.cpp)

set(MX_WP_GEMM_XDL_BENCHMARK mx_wp_gemm_xdl_benchmark)
set(MX_WP_GEMM_XDL_BENCHMARK_SRC mx_wp_gemm_xdl_benchmark.cpp)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

generate_sharded_instantiations(
  INSTANCES_NAME gemm_xdl_benchmark_instances
  TEMPLATE_FILE gemm_xdl_benchmark_instances.in
  NUM_SHARDS 160
  SRC_LIST GEMM_XDL_BENCHMARK_SRC
  OUTPUT_DIR ${GENERATED_DIR}
)

generate_sharded_instantiations(
  INSTANCES_NAME mx_gemm_xdl_benchmark_instances
  TEMPLATE_FILE mx_gemm_xdl_benchmark_instances.in
  NUM_SHARDS 60
  SRC_LIST MX_GEMM_XDL_BENCHMARK_SRC
  OUTPUT_DIR ${GENERATED_DIR}
)

generate_sharded_instantiations(
  INSTANCES_NAME wp_gemm_xdl_benchmark_instances
  TEMPLATE_FILE wp_gemm_xdl_benchmark_instances.in
  NUM_SHARDS 60
  SRC_LIST WP_GEMM_XDL_BENCHMARK_SRC
  OUTPUT_DIR ${GENERATED_DIR}
)

generate_sharded_instantiations(
  INSTANCES_NAME mx_wp_gemm_xdl_benchmark_instances
  TEMPLATE_FILE mx_wp_gemm_xdl_benchmark_instances.in
  NUM_SHARDS 50
  SRC_LIST MX_WP_GEMM_XDL_BENCHMARK_SRC
  OUTPUT_DIR ${GENERATED_DIR}
)

add_custom_target(example_${GEMM_XDL_BENCHMARK})
add_custom_target(example_${MX_GEMM_XDL_BENCHMARK})
add_custom_target(example_${WP_GEMM_XDL_BENCHMARK})
add_custom_target(example_${MX_WP_GEMM_XDL_BENCHMARK})

function(add_benchmark GemmBenchMark PrecDataType ALayout BLayout)
    set(EXE_NAME example_${GemmBenchMark}_${PrecDataType}_${ALayout}_${BLayout})
    foreach(source IN LISTS ARGN)
        set(FILE_NAME ${FILE_NAME} ${source})
    endforeach()
    add_example_executable(${EXE_NAME} ${FILE_NAME})
    add_example_dependencies(example_${GemmBenchMark} ${EXE_NAME})
    example_compile_options(${EXE_NAME} PRIVATE -Wno-global-constructors)
    example_compile_options(${EXE_NAME} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    if (TARGET  ${EXE_NAME})
       target_compile_definitions(${EXE_NAME} PRIVATE PREC_DATATYPE=${PrecDataType})
       target_compile_definitions(${EXE_NAME} PRIVATE A_LAYOUT=${ALayout})
       target_compile_definitions(${EXE_NAME} PRIVATE B_LAYOUT=${BLayout})
    endif()
endfunction(add_benchmark)

add_benchmark(${GEMM_XDL_BENCHMARK} fp16 Row Col ${GEMM_XDL_BENCHMARK_SRC} )
add_benchmark(${GEMM_XDL_BENCHMARK}  fp16 Row Row ${GEMM_XDL_BENCHMARK_SRC})
#add_benchmark(${GEMM_XDL_BENCHMARK} fp16 Col Col ${GEMM_XDL_BENCHMARK_SRC} )
#add_benchmark(${GEMM_XDL_BENCHMARK} fp16 Col Row  ${GEMM_XDL_BENCHMARK_SRC})
add_benchmark(${GEMM_XDL_BENCHMARK} i8 Row Col ${GEMM_XDL_BENCHMARK_SRC} )
add_benchmark(${GEMM_XDL_BENCHMARK}  fp8 Row Col ${GEMM_XDL_BENCHMARK_SRC})
add_benchmark(${GEMM_XDL_BENCHMARK}  fp8 Row Row ${GEMM_XDL_BENCHMARK_SRC})
add_benchmark(${GEMM_XDL_BENCHMARK}  pk_i4 Row Col ${GEMM_XDL_BENCHMARK_SRC})
#add_benchmark(${GEMM_XDL_BENCHMARK}  pk_fp4 Row Col ${GEMM_XDL_BENCHMARK_SRC})

add_benchmark(${MX_GEMM_XDL_BENCHMARK}  fp8 Row Col ${MX_GEMM_XDL_BENCHMARK_SRC})
# ck async load/direct load doesn't support transposed layout
#add_benchmark(${MX_GEMM_XDL_BENCHMARK}  fp8 Row Row ${MX_GEMM_XDL_BENCHMARK_SRC})
#add_benchmark(${MX_GEMM_XDL_BENCHMARK}  fp8 Col Col ${MX_GEMM_XDL_BENCHMARK_SRC})
#add_benchmark(${MX_GEMM_XDL_BENCHMARK}  fp8 Col Row ${MX_GEMM_XDL_BENCHMARK_SRC})
add_benchmark(${MX_GEMM_XDL_BENCHMARK}  pk_fp4 Row Col ${MX_GEMM_XDL_BENCHMARK_SRC})

add_benchmark(${WP_GEMM_XDL_BENCHMARK}  fp8 Row Col ${WP_GEMM_XDL_BENCHMARK_SRC})
add_benchmark(${WP_GEMM_XDL_BENCHMARK}  fp16 Row Col ${WP_GEMM_XDL_BENCHMARK_SRC})

add_benchmark(${MX_WP_GEMM_XDL_BENCHMARK}  fp8 Row Col ${MX_WP_GEMM_XDL_BENCHMARK_SRC})
add_benchmark(${MX_WP_GEMM_XDL_BENCHMARK}  pk_fp4 Row Col ${MX_WP_GEMM_XDL_BENCHMARK_SRC})

add_example_executable(example_gemm_xdl_ck_tile_wrap_fp16 gemm_xdl_ck_tile_wrap_fp16.cpp)
example_compile_options(example_gemm_xdl_ck_tile_wrap_fp16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
add_example_executable(example_gemm_xdl_ck_tile_wrap_fp16_async gemm_xdl_ck_tile_wrap_fp16_async.cpp)
example_compile_options(example_gemm_xdl_ck_tile_wrap_fp16_async PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
add_example_executable(example_gemm_xdl_ck_tile_wrap_fp16_tdm gemm_xdl_ck_tile_wrap_fp16_tdm.cpp)
example_compile_options(example_gemm_xdl_ck_tile_wrap_fp16_tdm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

add_example_executable(example_gemm_xdl_ck_tile_wrap_mx_tdm gemm_xdl_ck_tile_wrap_mx_tdm.cpp)
example_compile_options(example_gemm_xdl_ck_tile_wrap_mx_tdm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})