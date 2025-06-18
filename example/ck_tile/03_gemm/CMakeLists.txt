add_executable(tile_example_gemm_basic EXCLUDE_FROM_ALL gemm_basic.cpp)
add_executable(tile_example_gemm_aquant_basic EXCLUDE_FROM_ALL gemm_aquant_basic.cpp)
add_executable(tile_example_gemm_universal EXCLUDE_FROM_ALL universal_gemm.cpp)
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)
target_compile_options(tile_example_gemm_basic PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
target_compile_options(tile_example_gemm_universal PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
target_compile_options(tile_example_gemm_aquant_basic PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

# BQuant example
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DUSING_MFMA_16x16x32=1 -Wno-unused-local-typedef)
#list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DUSING_MFMA_32x32x16=1 -Wno-unused-local-typedef)
add_executable(tile_example_gemm_bquant_basic EXCLUDE_FROM_ALL gemm_bquant_basic.cpp)
target_compile_options(tile_example_gemm_bquant_basic PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
