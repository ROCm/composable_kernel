function(add_instance_library INSTANCE_NAME)
    message("adding instance ${INSTANCE_NAME}")
    add_library(${INSTANCE_NAME} OBJECT ${ARGN})
    target_compile_features(${INSTANCE_NAME} PUBLIC)
    set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    clang_tidy_check(${INSTANCE_NAME})
endfunction(add_instance_library INSTANCE_NAME)

add_subdirectory(gemm)
add_subdirectory(gemm_splitk)
add_subdirectory(gemm_bilinear)
add_subdirectory(gemm_add_add_fastgelu)
add_subdirectory(gemm_reduce)
add_subdirectory(gemm_bias_add_reduce)
add_subdirectory(batched_gemm)
add_subdirectory(batched_gemm_reduce)
add_subdirectory(grouped_gemm)
add_subdirectory(contraction_scale)
add_subdirectory(contraction_bilinear)
add_subdirectory(conv1d_fwd)
add_subdirectory(conv2d_fwd)
add_subdirectory(conv3d_fwd)
add_subdirectory(conv2d_fwd_bias_relu)
add_subdirectory(conv2d_fwd_bias_relu_add)
add_subdirectory(conv2d_bwd_data)
add_subdirectory(convnd_bwd_data)
add_subdirectory(conv2d_bwd_weight)
add_subdirectory(convnd_bwd_weight)
add_subdirectory(reduce)
add_subdirectory(normalization)
add_subdirectory(elementwise)

add_library(device_operations STATIC
    $<TARGET_OBJECTS:device_gemm_instance>
    $<TARGET_OBJECTS:device_gemm_splitk_instance>
    $<TARGET_OBJECTS:device_gemm_bilinear_instance>
    $<TARGET_OBJECTS:device_gemm_add_add_fastgelu_instance>
    $<TARGET_OBJECTS:device_gemm_bias_add_reduce_instance>
    $<TARGET_OBJECTS:device_batched_gemm_instance>
    $<TARGET_OBJECTS:device_batched_gemm_reduce_instance>
    $<TARGET_OBJECTS:device_grouped_gemm_instance>
    $<TARGET_OBJECTS:device_contraction_scale_instance>
    $<TARGET_OBJECTS:device_contraction_bilinear_instance>
    $<TARGET_OBJECTS:device_conv1d_fwd_instance>
    $<TARGET_OBJECTS:device_conv2d_fwd_instance>
    $<TARGET_OBJECTS:device_conv3d_fwd_instance>
    $<TARGET_OBJECTS:device_conv2d_fwd_bias_relu_instance>
    $<TARGET_OBJECTS:device_conv2d_fwd_bias_relu_add_instance>
    $<TARGET_OBJECTS:device_conv2d_bwd_data_instance>
    $<TARGET_OBJECTS:device_convnd_bwd_data_instance>
    $<TARGET_OBJECTS:device_conv2d_bwd_weight_instance>
    $<TARGET_OBJECTS:device_convnd_bwd_weight_instance>
    $<TARGET_OBJECTS:device_reduce_instance>
    $<TARGET_OBJECTS:device_normalization_instance>
    $<TARGET_OBJECTS:device_elementwise_instance>
)
add_library(composablekernels::device_operations ALIAS device_operations)


set(DEV_OPS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include/ck/
    ${PROJECT_SOURCE_DIR}/library/include/ck/
)

target_compile_features(device_operations PUBLIC)
set_target_properties(device_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(device_operations PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/utility>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_description>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/problem_transform>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/grid>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/block>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/warp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/thread>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/element>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/host_tensor>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/reduce>
)

#once new arches are enabled make this an option on the main cmake file
# and pass down here to be exported
target_compile_options(device_operations PRIVATE
    --offload-arch=gfx908
    --offload-arch=gfx90a
)

# install(TARGETS device_operations LIBRARY DESTINATION lib)
rocm_install(TARGETS device_operations
        EXPORT device_operationsTargets)

rocm_install(DIRECTORY ${DEV_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck)
rocm_install(EXPORT device_operationsTargets
        FILE composable_kerneldevice_operationsTargets.cmake
        NAMESPACE composable_kernel::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)
