# Tutorial 16: Row Reduction — Warp Reduce vs Block Reduce
# Compares warp-only vs cross-warp (LDS) row reduction using ck_tile primitives

add_executable(aa_tutorial_16_row_reduction row_reduction.cpp)

target_include_directories(aa_tutorial_16_row_reduction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

message(STATUS "Added Tutorial 16: Row Reduction — Warp vs Block Reduce")
