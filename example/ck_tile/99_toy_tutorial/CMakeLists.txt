# CK_TILE Tutorial Examples
# Educational examples for learning ck_tile API

include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

# Tutorial Series - Tensor View API
# Each tutorial builds on the previous one

# Tutorial 01: Tensor Fundamentals - Basic tensor concepts
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_01_tensor_fundamentals/CMakeLists.txt)
    add_subdirectory(tutorial_01_tensor_fundamentals)
endif()

# Tutorial 02: Tensor Adaptors - Advanced layout transformations
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_02_tensor_adaptors/CMakeLists.txt)
    add_subdirectory(tutorial_02_tensor_adaptors)
endif()

# Tutorial 03: Padding with Tile Windows
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_03_padding_and_tiles/CMakeLists.txt)
    add_subdirectory(tutorial_03_padding_and_tiles)
endif()

# Tutorial 04: Descriptor vs Adaptor - Understanding the differences
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_04_descriptor_vs_adaptor/CMakeLists.txt)
    add_subdirectory(tutorial_04_descriptor_vs_adaptor)
endif()

# Tutorial 05: Basic Distributed GEMM
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_05_basic_distributed_gemm/CMakeLists.txt)
    add_subdirectory(tutorial_05_basic_distributed_gemm)
endif()

# Tutorial 06: Tile Sweeping GEMM
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_06_tile_sweeping_gemm/CMakeLists.txt)
    add_subdirectory(tutorial_06_tile_sweeping_gemm)
endif()

# Tutorial 07: Tile Sweeping with Y-Dimension Repetition
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_07_tile_sweeping_with_y_repetition/CMakeLists.txt)
    add_subdirectory(tutorial_07_tile_sweeping_with_y_repetition)
endif()

# Tutorial 08: Simple LDS Staging - Basic shared memory usage
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_08_lds_staging/CMakeLists.txt)
    add_subdirectory(tutorial_08_lds_staging)
endif()

# Tutorial 09: Optimized LDS Staging - Separate copy distributions and optimizations
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_09_optimized_lds/CMakeLists.txt)
    add_subdirectory(tutorial_09_optimized_lds)
endif()

# Tutorial 10: Padded LDS Layout - Reducing LDS bank conflicts via padding
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_10_xor_lds/CMakeLists.txt)
    add_subdirectory(tutorial_10_xor_lds)
endif()

# Tutorial 11: XOR Test - Minimal test to understand XOR swizzling
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_11_xor_test/CMakeLists.txt)
    add_subdirectory(tutorial_11_xor_test)
endif()

# Tutorial 12: XOR-Based Bank Conflict-Free LDS (Correct [N,K] Layout)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_12_xor_correct/CMakeLists.txt)
    add_subdirectory(tutorial_12_xor_correct)
endif()

# Tutorial 13: Production-Style XOR LDS with [N,K] B Layout
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_13_production_xor/CMakeLists.txt)
    add_subdirectory(tutorial_13_production_xor)
endif()

# Tutorial 14: Bank Conflict Scenarios - Step-by-step comparison of storage layouts
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_14_bank_conflict_scenarios/CMakeLists.txt)
    add_subdirectory(tutorial_14_bank_conflict_scenarios)
endif()

# Tutorial 15: Three Ways to Call a GEMM - BlockGemm vs Pipeline vs Kernel
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_15_calling_gemm/CMakeLists.txt)
    add_subdirectory(tutorial_15_calling_gemm)
endif()

# Tutorial 16: Row Reduction — Warp Reduce vs Block Reduce
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_16_row_reduction/CMakeLists.txt)
    add_subdirectory(tutorial_16_row_reduction)
endif()

message(STATUS "CK_TILE Tutorial examples configured")
