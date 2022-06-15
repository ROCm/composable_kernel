add_example_executable(example_cpu_conv2d_fwd cpu_conv2d_fwd.cpp)
target_link_libraries(example_cpu_conv2d_fwd PRIVATE ck_cpu_instance)

set_target_properties(example_cpu_conv2d_fwd PROPERTIES LINK_FLAGS "${OMP_LINK_FLAG}")
target_link_libraries(example_cpu_conv2d_fwd PRIVATE "${OMP_LIBRARY}")
target_compile_options(example_cpu_conv2d_fwd PRIVATE "${OMP_CXX_FLAG}")
