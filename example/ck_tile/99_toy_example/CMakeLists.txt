# CK_TILE Tutorial Examples
# These are educational examples for learning ck_tile API

include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

# Tutorial stages for learning ck_tile
# Each stage builds on the previous one

# Stage 00: Hello ck_tile - First program
#add_subdirectory(stage_00_hello_ck_tile)

# Stage 01: Tile Distribution - Understanding thread-to-data mapping
#add_subdirectory(stage_01_tile_distribution)

# Stage 02: 2D Tensors and Windows
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_02_2d_tensors/CMakeLists.txt)
    add_subdirectory(stage_02_2d_tensors)
endif()

# Stage 03: Moving Windows
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_03_moving_windows/CMakeLists.txt)
    add_subdirectory(stage_03_moving_windows)
endif()

# Stage 04: Multiple Tile Operations
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_04_multi_tile/CMakeLists.txt)
    add_subdirectory(stage_04_multi_tile)
endif()

# Stage 05: Naive GEMM (Baseline)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_05_naive_gemm/CMakeLists.txt)
    add_subdirectory(stage_05_naive_gemm)
endif()

# Stage 06: Tiled GEMM
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_06_tiled_gemm/CMakeLists.txt)
    add_subdirectory(stage_06_tiled_gemm)
endif()

# Stage 09: Single MFMA - Simplest MFMA GEMM (key example)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_09_single_mfma/CMakeLists.txt)
    add_subdirectory(stage_09_single_mfma)
endif()

# Stage 10: MFMA with K-Loop
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_10_mfma_k_loop/CMakeLists.txt)
    add_subdirectory(stage_10_mfma_k_loop)
endif()

# Stage 10b: Distributed HGEMM (Testing distributions)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_10b_distributed_hgemm/CMakeLists.txt)
    add_subdirectory(stage_10b_distributed_hgemm)
endif()

# Stage 11: Sweep Tile Playground - Experimenting with sweep_tile, sweep_tile_span, static_ford
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_11_sweep_tile_playground/CMakeLists.txt)
    add_subdirectory(stage_11_sweep_tile_playground)
endif()

# Stage 11b: 2x2 MFMA Tiles
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_11_2x2_mfma/CMakeLists.txt)
    add_subdirectory(stage_11_2x2_mfma)
endif()

# Keep CT04 as reference (will become Tutorial 02 later)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stage_CT04_basic_transforms/CMakeLists.txt)
    add_subdirectory(stage_CT04_basic_transforms)
endif()

# Future stages to be implemented:
# Stage 03: Moving Windows
# Stage 04: Multiple Tile Operations
# Stage 05: Naive GEMM (No MFMA)
# Stage 06: Tiled GEMM (No MFMA)
# Stage 07: Shared Memory GEMM (No MFMA)
# Stage 08: Warp-Level GEMM (No MFMA)
# Stage 10: MFMA with K-Loop
# Stage 11: 2x2 MFMA Tiles
# Stage 12: 4x4 MFMA Tiles
# ... and more

# Original examples (preserved for reference)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/01_add)
    add_subdirectory(01_add)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/02_gemm)
    add_subdirectory(02_gemm)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/03_flash_attention_fwd)
    add_subdirectory(03_flash_attention_fwd)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/04_codegen_flash_attention_fwd)
    add_subdirectory(04_codegen_flash_attention_fwd)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/05_simple_gemm)
    add_subdirectory(05_simple_gemm)
endif()

message(STATUS "CK_TILE Tutorial examples configured")
