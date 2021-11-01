include_directories(BEFORE
    include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/device/include
    ${PROJECT_SOURCE_DIR}/host/driver_offline/include
    ${PROJECT_SOURCE_DIR}/profiler/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/external/rocm/include
)

# device_gemm_instance
set(DEVICE_GEMM_INSTANCE_SOURCE 
    device_gemm_xdl_instance_vol1.cpp;
    device_gemm_xdl_instance_vol2.cpp;
)

add_library(device_gemm_instance SHARED ${DEVICE_GEMM_INSTANCE_SOURCE}) 
target_include_directories(device_gemm_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

target_compile_features(device_gemm_instance PUBLIC)
set_target_properties(device_gemm_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS device_gemm_instance LIBRARY DESTINATION lib) 

# ck_profiler
set(PROFILER_SOURCE driver_profiler.cpp)
add_executable(ckProfiler ${PROFILER_SOURCE})

target_link_libraries(ckProfiler PRIVATE host_tensor)
target_link_libraries(ckProfiler PRIVATE device_gemm_instance)
