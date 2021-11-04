include_directories(BEFORE
    include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/device/include
    ${PROJECT_SOURCE_DIR}/device_operation/include
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
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f32_f32_f32_mk_kn_mn.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f32_f32_f32_mk_nk_mn.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f32_f32_f32_km_kn_mn.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f32_f32_f32_km_nk_mn.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f16_f16_f16_mk_kn_mn.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f16_f16_f16_mk_nk_mn.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f16_f16_f16_km_kn_mn.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_gemm_xdl_instance_f16_f16_f16_km_nk_mn.cpp;
) 

add_library(device_gemm_instance SHARED ${DEVICE_GEMM_INSTANCE_SOURCE}) 
target_include_directories(device_gemm_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_compile_features(device_gemm_instance PUBLIC)
set_target_properties(device_gemm_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS device_gemm_instance LIBRARY DESTINATION lib) 

## device_conv_instance
#set(DEVICE_CONV_INSTANCE_SOURCE 
#   ${PROJECT_SOURCE_DIR}/device_operation/device_conv_xdl_instance_f32_f32_f32_nhwc_kyxc_nhwk.cpp;
#) 
#
#add_library(device_conv_instance SHARED ${DEVICE_CONV_INSTANCE_SOURCE}) 
#target_include_directories(device_conv_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
#target_compile_features(device_conv_instance PUBLIC)
#set_target_properties(device_conv_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
#install(TARGETS device_conv_instance LIBRARY DESTINATION lib) 

# ck_profiler
#set(PROFILER_SOURCE profiler.cpp gemm_profiler.cpp conv_profiler.cpp)
set(PROFILER_SOURCE profiler.cpp gemm_profiler.cpp)
#set(PROFILER_SOURCE profiler.cpp conv_profiler.cpp)
add_executable(ckProfiler ${PROFILER_SOURCE})

target_link_libraries(ckProfiler PRIVATE host_tensor)
target_link_libraries(ckProfiler PRIVATE device_gemm_instance)
#target_link_libraries(ckProfiler PRIVATE device_conv_instance)
