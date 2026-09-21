# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(InstPrefetchPatch)

add_test_executable(test_s_prefetch_inst_op s_prefetch_inst_op.cpp)
target_compile_options(test_s_prefetch_inst_op PRIVATE -mllvm -pragma-unroll-threshold=16384)
target_link_libraries(test_s_prefetch_inst_op PRIVATE utility)
ck_inst_prefetch_patch(test_s_prefetch_inst_op)
