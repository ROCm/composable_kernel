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

message(STATUS "CK_TILE Tutorial examples configured")
