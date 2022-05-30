include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/include/ck/host_utility
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
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/reference_tensor_operation/cpu
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/reference_tensor_operation/gpu
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/utility
    ${PROJECT_SOURCE_DIR}/external/include/half
)

add_custom_target(examples)

function(add_example_executable EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE host_tensor)
    add_test(NAME ${EXAMPLE_NAME} COMMAND $<TARGET_FILE:${EXAMPLE_NAME}> ${ARGN})
    add_dependencies(examples ${EXAMPLE_NAME})
    add_dependencies(check ${EXAMPLE_NAME})
endfunction(add_example_executable EXAMPLE_NAME)

function(add_example_executable_no_testing EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE host_tensor)
    add_dependencies(examples ${EXAMPLE_NAME})
endfunction(add_example_executable_no_testing EXAMPLE_NAME)

add_subdirectory(01_gemm)
add_subdirectory(02_gemm_alpha_beta)
add_subdirectory(03_gemm_bias_relu)
add_subdirectory(04_gemm_bias_relu_add)
add_subdirectory(06_conv2d_fwd_bias_relu)
add_subdirectory(07_conv2d_fwd_bias_relu_add)
add_subdirectory(09_convnd_fwd)
add_subdirectory(10_conv2d_bwd_data)
add_subdirectory(11_conv2d_bwd_weight)
add_subdirectory(12_reduce)
add_subdirectory(13_pool2d_fwd)
add_subdirectory(14_gemm_xdl_requant_relu_requant)
add_subdirectory(15_grouped_gemm)
add_subdirectory(16_gemm_reduce)
add_subdirectory(17_convnd_bwd_data_xdl)
add_subdirectory(18_batched_gemm_reduce)
add_subdirectory(19_binary_elementwise)
add_subdirectory(20_convnd_bwd_weight_xdl)
add_subdirectory(21_gemm_layernorm)
add_subdirectory(22_cgemm)
