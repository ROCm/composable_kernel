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
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f32_f32_f32_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f32_f32_f32_mk_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f32_f32_f32_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f32_f32_f32_km_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_mk_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_km_nk_mn_instance.cpp;
)

# device_gemm_bias_2d_instance
set(DEVICE_GEMM_BIAS_2D_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_km_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_mk_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_mk_nk_mn_instance.cpp;
)

# device_gemm_bias_relu_instance
set(DEVICE_GEMM_BIAS_RELU_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_f16_f16_f16_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_f16_f16_f16_mk_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_f16_f16_f16_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_f16_f16_f16_km_nk_mn_instance.cpp;
)

# device_gemm_bias_relu_add_instance
set(DEVICE_GEMM_BIAS_RELU_ADD_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_mk_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_mk_nk_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_kn_mn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_nk_mn_instance.cpp;
)

set(DEVICE_BATCHED_GEMM_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instance.cpp;
)

# device_conv2d_fwd_instance
set(DEVICE_CONV2D_FWD_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instance.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instance.cpp;
)

# device_conv2d_fwd_bias_relu_instance
set(DEVICE_CONV2D_FWD_BIAS_RELU_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instance.cpp;
)

# device_conv2d_fwd_bias_relu_add_instance
set(DEVICE_CONV2D_FWD_BIAS_RELU_ADD_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_c_shuffle_bias_relu_add_nhwc_kyxc_nhwk_f16_instance.cpp;
)

# device_conv2d_fwd_bias_relu_atomic_add_instance
set(DEVICE_CONV2D_FWD_BIAS_RELU_ATOMIC_ADD_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_conv2d_fwd_xdl_c_shuffle_bias_relu_atomic_add_nhwc_kyxc_nhwk_f16_instance.cpp;
)

# device_reduce_instance
set(DEVICE_REDUCE_INSTANCE_SOURCE
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_f16_f16_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_f16_f32_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_f32_f32_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_f32_f64_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_f64_f64_f64.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_threadwise_f16_f16_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_threadwise_f16_f32_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_threadwise_f32_f32_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_threadwise_f32_f64_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_threadwise_f64_f64_f64.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_second_call_f16_f16_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_second_call_f32_f32_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_second_call_f32_f32_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_second_call_f64_f64_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_blockwise_second_call_f64_f64_f64.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_atomic_add_f16_f32_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_atomic_add_f32_f32_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_atomic_add_f32_f64_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_two_call_f16_f16_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_two_call_f16_f32_f16.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_two_call_f32_f32_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_two_call_f32_f64_f32.cpp;
   ${PROJECT_SOURCE_DIR}/device_operation/src/device_reduce_instance_multiblock_two_call_f64_f64_f64.cpp;
)

add_library(device_gemm_instance SHARED ${DEVICE_GEMM_INSTANCE_SOURCE})
add_library(device_gemm_bias_2d_instance SHARED ${DEVICE_GEMM_BIAS_2D_INSTANCE_SOURCE})
add_library(device_gemm_bias_relu_instance SHARED ${DEVICE_GEMM_BIAS_RELU_INSTANCE_SOURCE})
add_library(device_gemm_bias_relu_add_instance SHARED ${DEVICE_GEMM_BIAS_RELU_ADD_INSTANCE_SOURCE})
add_library(device_batched_gemm_instance SHARED ${DEVICE_BATCHED_GEMM_INSTANCE_SOURCE})
add_library(device_conv2d_fwd_instance SHARED ${DEVICE_CONV2D_FWD_INSTANCE_SOURCE})
add_library(device_conv2d_fwd_bias_relu_instance SHARED ${DEVICE_CONV2D_FWD_BIAS_RELU_INSTANCE_SOURCE})
add_library(device_conv2d_fwd_bias_relu_add_instance SHARED ${DEVICE_CONV2D_FWD_BIAS_RELU_ADD_INSTANCE_SOURCE})
add_library(device_conv2d_fwd_bias_relu_atomic_add_instance SHARED ${DEVICE_CONV2D_FWD_BIAS_RELU_ATOMIC_ADD_INSTANCE_SOURCE})
add_library(device_reduce_instance SHARED ${DEVICE_REDUCE_INSTANCE_SOURCE}) 

target_include_directories(device_gemm_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_gemm_bias_2d_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_gemm_bias_relu_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_gemm_bias_relu_add_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_batched_gemm_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_conv2d_fwd_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_conv2d_fwd_bias_relu_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_conv2d_fwd_bias_relu_add_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_conv2d_fwd_bias_relu_atomic_add_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_include_directories(device_reduce_instance SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

target_compile_features(device_gemm_instance PUBLIC)
target_compile_features(device_gemm_bias_2d_instance PUBLIC)
target_compile_features(device_gemm_bias_relu_instance PUBLIC)
target_compile_features(device_gemm_bias_relu_add_instance PUBLIC)
target_compile_features(device_batched_gemm_instance PUBLIC)
target_compile_features(device_conv2d_fwd_instance PUBLIC)
target_compile_features(device_conv2d_fwd_bias_relu_instance PUBLIC)
target_compile_features(device_conv2d_fwd_bias_relu_add_instance PUBLIC)
target_compile_features(device_conv2d_fwd_bias_relu_atomic_add_instance PUBLIC)
target_compile_features(device_reduce_instance PUBLIC)

set_target_properties(device_gemm_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_gemm_bias_2d_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_gemm_bias_relu_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_gemm_bias_relu_add_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_batched_gemm_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_conv2d_fwd_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_conv2d_fwd_bias_relu_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_conv2d_fwd_bias_relu_add_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_conv2d_fwd_bias_relu_atomic_add_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(device_reduce_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS device_gemm_instance LIBRARY DESTINATION lib)
install(TARGETS device_gemm_bias_2d_instance LIBRARY DESTINATION lib)
install(TARGETS device_gemm_bias_relu_instance LIBRARY DESTINATION lib)
install(TARGETS device_gemm_bias_relu_add_instance LIBRARY DESTINATION lib)
install(TARGETS device_batched_gemm_instance LIBRARY DESTINATION lib)
install(TARGETS device_conv2d_fwd_instance LIBRARY DESTINATION lib)
install(TARGETS device_conv2d_fwd_bias_relu_instance LIBRARY DESTINATION lib)
install(TARGETS device_conv2d_fwd_bias_relu_add_instance LIBRARY DESTINATION lib)
install(TARGETS device_conv2d_fwd_bias_relu_atomic_add_instance LIBRARY DESTINATION lib)
install(TARGETS device_reduce_instance LIBRARY DESTINATION lib) 
