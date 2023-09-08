if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    add_example_executable(example_layernorm_fp16 layernorm_fp16.cpp)
    add_example_executable(example_layernorm_splitk_fp16 layernorm_splitk_fp16.cpp)
endif()
