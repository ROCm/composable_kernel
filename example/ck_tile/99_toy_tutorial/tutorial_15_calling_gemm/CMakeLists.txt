# Tutorial 15: Three Ways to Call a GEMM
# Compares BlockGemm vs GemmPipeline vs GemmKernel abstraction levels

add_compile_options(--save-temps) 
add_executable(aa_tutorial_15_calling_gemm calling_gemm.cpp)

target_include_directories(aa_tutorial_15_calling_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

message(STATUS "Added Tutorial 15: Three Ways to Call a GEMM")
