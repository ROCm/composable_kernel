# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(01_naive_gemm)
add_subdirectory(02_padding_k_first)
add_subdirectory(03_mfma_16x16x16)
