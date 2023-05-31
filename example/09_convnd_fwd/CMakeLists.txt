if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR GPU_TARGETS MATCHES "gfx940")
    add_example_executable(example_convnd_fwd_xdl_fp32 convnd_fwd_xdl_fp32.cpp)
    add_example_executable(example_convnd_fwd_xdl_fp16 convnd_fwd_xdl_fp16.cpp)
    add_example_executable(example_convnd_fwd_xdl_bf16 convnd_fwd_xdl_bf16.cpp)
    add_example_executable(example_convnd_fwd_xdl_int8 convnd_fwd_xdl_int8.cpp)
    # FIXME: re-enable this exampe as test when SWDEV-335738 is fixed
    add_example_executable_no_testing(example_convnd_fwd_xdl_fp64 convnd_fwd_xdl_fp64.cpp)
endif()
add_example_executable(example_convnd_fwd_dl_fp16 convnd_fwd_dl_fp16.cpp)
add_example_executable(example_convnd_fwd_dl_fp32 convnd_fwd_dl_fp32.cpp)
add_example_executable(example_convnd_fwd_dl_int8 convnd_fwd_dl_int8.cpp)

