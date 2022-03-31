include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_description
    ${PROJECT_SOURCE_DIR}/include/ck/tensor
    ${PROJECT_SOURCE_DIR}/include/ck/problem_transform
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/device
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/grid
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/block
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/warp
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/thread
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/element
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/tensor_operation_instance
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/tensor_operation_instance/gpu/reduce
    ${PROJECT_SOURCE_DIR}/external/include/half
)

function(add_instance_library INSTANCE_NAME)
    message("adding instance ${INSTANCE_NAME}")
    add_library(${INSTANCE_NAME} SHARED ${ARGN}) 
    target_compile_features(${INSTANCE_NAME} PUBLIC)
    set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction(add_instance_library INSTANCE_NAME)

add_subdirectory(gemm)
add_subdirectory(gemm_bias2d)
add_subdirectory(gemm_bias_relu)
add_subdirectory(gemm_bias_relu_add)
add_subdirectory(gemm_reduce)
add_subdirectory(batched_gemm)
add_subdirectory(conv1d_fwd)
add_subdirectory(conv2d_fwd)
add_subdirectory(conv3d_fwd)
add_subdirectory(conv2d_fwd_bias_relu)
add_subdirectory(conv2d_fwd_bias_relu_add)
add_subdirectory(conv2d_fwd_bias_relu_atomic_add)
add_subdirectory(conv2d_bwd_data)
add_subdirectory(reduce)
add_subdirectory(convnd_bwd_data)
add_subdirectory(grouped_gemm)
add_subdirectory(conv2d_bwd_weight)
add_subdirectory(batched_gemm_reduce)
