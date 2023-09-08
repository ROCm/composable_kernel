
add_custom_target(example_gemv_splitk)
add_example_executable(example_gemv_splitk_fp16 gemv_splitk_fp16.cpp)
add_dependencies(example_gemv_splitk
              example_gemv_splitk_fp16)