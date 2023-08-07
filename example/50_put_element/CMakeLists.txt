if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    add_example_executable(example_put_element_fp16 put_element_fp16.cpp)
endif()
