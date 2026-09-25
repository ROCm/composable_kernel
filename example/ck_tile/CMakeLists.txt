# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(01_fmha)
add_subdirectory(02_layernorm2d)
add_subdirectory(03_gemm)
add_subdirectory(04_img2col)
add_subdirectory(05_reduce)
add_subdirectory(06_permute)
add_subdirectory(09_topk_softmax)
add_subdirectory(10_rmsnorm2d)
add_subdirectory(11_add_rmsnorm2d_rdquant)
add_subdirectory(12_smoothquant)
add_subdirectory(13_moe_sorting)
add_subdirectory(14_moe_smoothquant)
add_subdirectory(15_fused_moe)
add_subdirectory(16_batched_gemm)
add_subdirectory(17_grouped_gemm)
add_subdirectory(18_flatmm)
add_subdirectory(19_gemm_multi_d)
add_subdirectory(20_grouped_convolution)
add_subdirectory(21_elementwise)
add_subdirectory(22_gemm_multi_abd)
add_subdirectory(35_batched_transpose)
add_subdirectory(36_pooling)
add_subdirectory(38_block_scale_gemm)
add_subdirectory(40_streamk_gemm)
add_subdirectory(41_batched_contraction)
add_subdirectory(42_mx_gemm)
add_subdirectory(49_sageattention)
add_subdirectory(50_sparse_attn)
add_subdirectory(51_tile_distr_enc_reg_map)
if(BUILD_CK_TILE_CSHUFFLE_LDS_BENCHMARKS)
    add_subdirectory(52_cshuffle_lds)
endif()

# Collect the tile_example_* executables configured above. Each example
# directory decides for itself which targets a given architecture gets, so the
# set has to be discovered rather than listed. This skips the tile_* driver
# targets and the 52_cshuffle_lds bench_lds_* microbenchmarks.
#
# EXCLUDE_FROM_ALL examples are skipped too. They are held out of the default
# build on purpose because they are expensive to compile, and registering them
# would defeat that: `check` runs every test regardless of label. That covers
# FMHA, which has its own CI stage (RUN_CK_TILE_FMHA_TESTS), along with the
# multi_reduce, sageattention and sparse_attn examples.
function(ck_tile_example_targets out_var)
  set(targets)
  set(pending ${CMAKE_CURRENT_SOURCE_DIR})
  while(pending)
    list(POP_FRONT pending dir)
    get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    list(APPEND pending ${subdirs})
    get_property(dir_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS dir_targets)
      get_property(target_type TARGET ${target} PROPERTY TYPE)
      get_property(target_excluded TARGET ${target} PROPERTY EXCLUDE_FROM_ALL)
      if(target_type STREQUAL "EXECUTABLE" AND NOT target_excluded
         AND target MATCHES "^tile_example_")
        list(APPEND targets ${target})
      endif()
    endforeach()
  endwhile()
  list(SORT targets)
  set(${out_var} ${targets} PARENT_SCOPE)
endfunction()

ck_tile_example_targets(CK_TILE_EXAMPLES)

# Register a CTest entry per example so the selective-test filter in
# script/dependency-parser can reach them: it intersects the targets affected by
# a PR with the output of `ctest -N`, and unregistered targets are dropped.
#
# The test compiles the example instead of running it. The examples default to
# benchmark problem sizes and iteration counts (gemm_quant, for instance,
# defaults to 3840x4096x2048 with 50 warmup and 1000 timed iterations), so
# running them is far too slow for PR CI, and several are gated on hardware the
# PR runners do not have.
set(CK_TILE_EXAMPLE_BUILD_TIMEOUT 7200)

add_custom_target(tile_examples)

foreach(target IN LISTS CK_TILE_EXAMPLES)
  # The test name must match the executable name exactly: the filter maps
  # bin/<exe> to a CTest name by stripping the bin/ prefix.
  add_test(NAME ${target}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target ${target}
  )
  # Concurrent builds in one build tree race, so let these tests run alongside
  # other tests but never alongside each other.
  set_tests_properties(${target} PROPERTIES
    LABELS CK_TILE_EXAMPLE
    RESOURCE_LOCK ck_tile_example_build
    TIMEOUT ${CK_TILE_EXAMPLE_BUILD_TIMEOUT}
  )

  add_dependencies(tile_examples ${target})
endforeach()

list(LENGTH CK_TILE_EXAMPLES CK_TILE_EXAMPLES_COUNT)
message(STATUS "Registered ${CK_TILE_EXAMPLES_COUNT} ck_tile example build tests (ctest -L CK_TILE_EXAMPLE)")
