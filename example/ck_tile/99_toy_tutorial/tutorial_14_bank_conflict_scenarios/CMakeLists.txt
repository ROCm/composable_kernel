# Tutorial 14: LDS Bank Conflict Avoidance Tricks
#
# Walks through storage-layout variants for a MxK -> KxM transpose
# through LDS, so the reader can see each conflict-avoidance trick in
# isolation and compare.
#
#   01 row_major              : baseline, 4-way bank conflicts on transpose read
#   02 column_major           : conflict-free, but changes physical layout
#   03 row_major_padded       : classic stride padding (coprime with bank count)
#   04 row_major_xor          : XOR swizzle via make_xor_transform
#                               (has residual conflicts -- see 06 and 07)
#   05 xor_plus_padding       : XOR + padding combined
#   06 xor_register_transpose : single [M,K] XOR descriptor with K-vectorized
#                               reads + register-level transpose_tile2d
#                               (LDS is a pass-through in this example;
#                                transpose happens in registers)
#   07 xor_cross_warp_lds_transpose : same [M,K] XOR descriptor, but the
#                               LDS READ distribution differs from the WRITE
#                               distribution so LDS genuinely shuffles data
#                               between warps. Reads are logically
#                               "transposed" (would conflict naively) but
#                               stay K-vectorized, so XOR still spreads
#                               accesses across all 32 banks.
#   08 xor_cross_warp_window_reinterpret : same LDS pattern as 07 but
#                               drops `transpose_tile2d` -- uses an
#                               [M,K]-shaped view over the [K,M] global
#                               buffer instead. Proof that the "transpose"
#                               is really done in LDS, and the
#                               transpose_tile2d in 07 was only a
#                               type-system bridge that compiled to nothing.
#
#   10 m_vector_store         : rotates the READ distribution so each
#                               thread holds 8 M values at 1 K (vector
#                               on M). Turns the global store into
#                               buffer_store_dwordx4 at the cost of
#                               narrow + conflicting ds_read on the
#                               LDS side. Experiment to see if moving
#                               the narrow op from HBM to LDS wins.
#
# Plus one utility unrelated to the trick itself:
#   09 minimal_buffer_view_lds : simplest buffer_view<lds> usage for rocgdb
#
# Investigation / debugging files that were produced while tracking down
# the "XOR still has conflicts" issue live under ./investigation and are
# not part of the tutorial build.

add_executable(aa_tutorial_14_01_row_major 01_row_major.cpp)
target_include_directories(aa_tutorial_14_01_row_major PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_01_row_major_fp32 01_row_major_fp32.cpp)
target_include_directories(aa_tutorial_14_01_row_major_fp32 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_02_column_major 02_column_major.cpp)
target_include_directories(aa_tutorial_14_02_column_major PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_03_row_major_padded 03_row_major_padded.cpp)
target_include_directories(aa_tutorial_14_03_row_major_padded PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_04_row_major_xor 04_row_major_xor.cpp)
target_include_directories(aa_tutorial_14_04_row_major_xor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_04_row_major_xor_fp32 04_row_major_xor_fp32.cpp)
target_include_directories(aa_tutorial_14_04_row_major_xor_fp32 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_05_xor_plus_padding 05_xor_plus_padding.cpp)
target_include_directories(aa_tutorial_14_05_xor_plus_padding PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_06_xor_register_transpose 06_xor_register_transpose.cpp)
target_include_directories(aa_tutorial_14_06_xor_register_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_07_xor_cross_warp_transpose 07_xor_cross_warp_lds_transpose.cpp)
target_include_directories(aa_tutorial_14_07_xor_cross_warp_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_08_xor_cross_warp_window_reinterpret 08_xor_cross_warp_window_reinterpret.cpp)
target_include_directories(aa_tutorial_14_08_xor_cross_warp_window_reinterpret PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_09_minimal_buffer_view_lds 09_minimal_buffer_view_lds.cpp)
target_include_directories(aa_tutorial_14_09_minimal_buffer_view_lds PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_10_m_vector_store 10_m_vector_store.cpp)
target_include_directories(aa_tutorial_14_10_m_vector_store PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_11_sfc_and_tile_distribution_intro 11_sfc_and_tile_distribution_intro.cpp)
target_include_directories(aa_tutorial_14_11_sfc_and_tile_distribution_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_12_sweep_tile_intro 12_sweep_tile_intro.cpp)
target_include_directories(aa_tutorial_14_12_sweep_tile_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_13_scatter_gather_intro 13_scatter_gather_intro.cpp)
target_include_directories(aa_tutorial_14_13_scatter_gather_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_14_shuffle_tile_intro 14_shuffle_tile_intro.cpp)
target_include_directories(aa_tutorial_14_14_shuffle_tile_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_15_static_array_and_thread_buffer_intro 15_static_array_and_thread_buffer_intro.cpp)
target_include_directories(aa_tutorial_14_15_static_array_and_thread_buffer_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_16_sync_primitives_intro 16_sync_primitives_intro.cpp)
target_include_directories(aa_tutorial_14_16_sync_primitives_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_17_sync_wrappers_intro 17_sync_wrappers_intro.cpp)
target_include_directories(aa_tutorial_14_17_sync_wrappers_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_18_global_to_lds_paths 18_global_to_lds_paths.cpp)
target_include_directories(aa_tutorial_14_18_global_to_lds_paths PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_19_async_load_tile_packed_lds 19_async_load_tile_packed_lds.cpp)
target_include_directories(aa_tutorial_14_19_async_load_tile_packed_lds PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_20_static_distributed_tensor_intro 20_static_distributed_tensor_intro.cpp)
target_include_directories(aa_tutorial_14_20_static_distributed_tensor_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_21_load_tile_intro 21_load_tile_intro.cpp)
target_include_directories(aa_tutorial_14_21_load_tile_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(aa_tutorial_14_22_y_sliced_thread_data_intro 22_y_sliced_thread_data_intro.cpp)
target_include_directories(aa_tutorial_14_22_y_sliced_thread_data_intro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)

message(STATUS "Tutorial 14: Bank conflict avoidance tricks configured (9 tricks + 1 debug helper + SFC intro + sweep intro + scatter_gather intro + shuffle_tile intro + static_array/thread_buffer intro + sync primitives intro + sync wrappers intro + global->LDS load families + async_load_tile variant of 08 + minimal load_tile for debugging + Y-slice intro)")
