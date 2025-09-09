add_executable(tile_example_flatmm_basic EXCLUDE_FROM_ALL flatmm_basic.cpp)
add_executable(tile_example_mixed_prec_flatmm EXCLUDE_FROM_ALL mixed_prec/mixed_prec_flatmm.cpp)

set(EXAMPLE_FLATMM_COMPILE_OPTIONS)

list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-unused-local-typedef -Wno-unused-variable -Wno-unused-parameter)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS --save-temps -Wno-nrvo)
#list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-32 -mllvm -enable-noalias-to-md-conversion=0")
#list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-32 -mllvm --disable-schedmodel-in-sched-mi=1 -mllvm --amdgpu-sched-strategy=gcn-iterative-max-occupancy-experimental -mllvm --misched-bottomup=1")
target_compile_options(tile_example_flatmm_basic PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
target_compile_options(tile_example_mixed_prec_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
