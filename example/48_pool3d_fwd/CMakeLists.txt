if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    add_example_executable(example_pool3d_fwd_fp16 pool3d_fwd_fp16.cpp)
endif()
