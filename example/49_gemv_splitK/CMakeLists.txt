
add_custom_target(example_splitK_gemv)
add_example_executable(example_splitK_gemv_fp16 splitK_gemv_fp16.cpp)
add_dependencies(example_splitK_gemv
              example_splitK_gemv_fp16)