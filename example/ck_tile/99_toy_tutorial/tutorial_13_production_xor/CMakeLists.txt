# Tutorial 13: Production-Style XOR LDS with [N,K] B Layout
# Fixes bank conflicts by storing B as [N,K] in LDS (matching production GEMM)

add_executable(aa_tutorial_13_production_xor production_xor_gemm.cpp)

target_include_directories(aa_tutorial_13_production_xor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

message(STATUS "Added Tutorial 13: Production XOR LDS with [N,K] B Layout")
