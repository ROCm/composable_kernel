# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
add_executable(tile_example_permute permute.cpp)

if(NOT DEFINED PERMUTE_USE_ALTERNATIVE_IMPL)
# set(PERMUTE_USE_ALTERNATIVE_IMPL false)
set(PERMUTE_USE_ALTERNATIVE_IMPL true)
endif()
# The alternative implementation (matrix_core_swizzle) is MFMA-only. permute.cpp already
# guards its usage with `#if !CK_TILE_USE_WMMA`, so on WMMA targets (gfx11/gfx12) compiling
# this source is both unused and pathological: the MFMA layout degenerates to a 4096-wide
# static_for whose fold expression overflows -fbracket-depth. Gate the source to match the
# code guard so it is only built for non-WMMA (MFMA) targets.
if(PERMUTE_USE_ALTERNATIVE_IMPL AND NOT CK_TILE_USE_WMMA)
target_compile_options(tile_example_permute PRIVATE -DPERMUTE_USE_ALTERNATIVE_IMPL)
target_sources(tile_example_permute PRIVATE alternative_impl/matrix_core_swizzle.cpp)
endif()
# target_compile_options(tile_example_permute PRIVATE -v --save-temps -Wno-gnu-line-marker)
