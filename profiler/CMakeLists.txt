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

# device_conv2d_fwd_instance
set(DEVICE_CONV2D_FWD_INSTANCE_SOURCE 
   ${PROJECT_SOURCE_DIR}/device_operation/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_p0_f16_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_s1_p0_f16_instance.cpp;
) 

add_library(device_conv2d_fwd_instance SHARED ${DEVICE_CONV2D_FWD_INSTANCE_SOURCE}) 
target_include_directories(device_conv2d_fwd_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_compile_features(device_conv2d_fwd_instance PUBLIC)
set_target_properties(device_conv2d_fwd_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS device_conv2d_fwd_instance LIBRARY DESTINATION lib) 

# device_conv2d_fwd_bias_relu_instance
set(DEVICE_CONV2D_FWD_BIAS_RELU_INSTANCE_SOURCE 
   ${PROJECT_SOURCE_DIR}/device_operation/device_conv2d_fwd_xdl_bias_relu_nhwc_kyxc_nhwk_f16_instance.cpp;
) 

add_library(device_conv2d_fwd_bias_relu_instance SHARED ${DEVICE_CONV2D_FWD_BIAS_RELU_INSTANCE_SOURCE}) 
target_include_directories(device_conv2d_fwd_bias_relu_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_compile_features(device_conv2d_fwd_bias_relu_instance PUBLIC)
set_target_properties(device_conv2d_fwd_bias_relu_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS device_conv2d_fwd_bias_relu_instance LIBRARY DESTINATION lib) 

# device_conv2d_fwd_bias_relu_add_instance
set(DEVICE_CONV2D_FWD_BIAS_RELU_ADD_INSTANCE_SOURCE 
   ${PROJECT_SOURCE_DIR}/device_operation/device_conv2d_fwd_xdl_bias_relu_add_nhwc_kyxc_nhwk_f16_instance.cpp;
) 

add_library(device_conv2d_fwd_bias_relu_add_instance SHARED ${DEVICE_CONV2D_FWD_BIAS_RELU_ADD_INSTANCE_SOURCE}) 
target_include_directories(device_conv2d_fwd_bias_relu_add_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_compile_features(device_conv2d_fwd_bias_relu_add_instance PUBLIC)
set_target_properties(device_conv2d_fwd_bias_relu_add_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS device_conv2d_fwd_bias_relu_add_instance LIBRARY DESTINATION lib) 

# ck_profiler
set(PROFILER_SOURCE profiler.cpp profile_gemm.cpp profile_conv_fwd.cpp profile_conv_fwd_bias_relu.cpp profile_conv_fwd_bias_relu_add.cpp)
add_executable(ckProfiler ${PROFILER_SOURCE})

target_link_libraries(ckProfiler PRIVATE host_tensor)
target_link_libraries(ckProfiler PRIVATE device_gemm_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_instance)
target_link_libraries(ckProfiler PRIVATE device_conv2d_fwd_bias_relu_add_instance)
