# ckProfiler
set(PROFILER_SOURCES
    profiler.cpp
)

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9")
  if(SUPPORTED_GPU_TARGETS MATCHES "gfx94")
    list(APPEND PROFILER_SOURCES profile_gemm_multiply_multiply_wp.cpp)
  endif()
endif()

set(PROFILER_EXECUTABLE ckProfiler)

add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)
# flags to compress the library
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
  message(STATUS "Adding --offload-compress flag for ${PROFILER_EXECUTABLE}")
  target_compile_options(${PROFILER_EXECUTABLE} PRIVATE --offload-compress)
endif()

target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE utility getopt::getopt)

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9")
  if(SUPPORTED_GPU_TARGETS MATCHES "gfx94")
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_multiply_multiply_wp_instance)
  endif()
endif()

rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)
