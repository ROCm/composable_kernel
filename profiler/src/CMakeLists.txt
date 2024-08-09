# ckProfiler
set(PROFILER_SOURCES
    profiler.cpp
)
list(APPEND PROFILER_SOURCES profile_grouped_conv_fwd.cpp)


set(PROFILER_EXECUTABLE ckProfiler)

add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)

target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE utility getopt::getopt)

target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv1d_fwd_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv2d_fwd_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv3d_fwd_instance)


rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)
