include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_description
    ${PROJECT_SOURCE_DIR}/include/ck/tensor
    ${PROJECT_SOURCE_DIR}/include/ck/problem_transform
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/cpu/device
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/cpu/grid
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/cpu/block
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/cpu/thread
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/cpu/element
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/tensor_operation_instance
    ${PROJECT_SOURCE_DIR}/external/include/half
)

function(add_instance_library INSTANCE_NAME)
    message("adding instance ${INSTANCE_NAME}")
    add_library(${INSTANCE_NAME} SHARED ${ARGN}) 
    target_compile_features(${INSTANCE_NAME} PUBLIC)
    set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction(add_instance_library INSTANCE_NAME)

add_subdirectory(conv2d_fwd)
