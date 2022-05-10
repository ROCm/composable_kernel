include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/device
    ${PROJECT_SOURCE_DIR}/include/ck/tensor_operation/gpu/element
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/reference_tensor_operation/cpu
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/utility
)

set(CONV_UTIL_SOURCE
    conv_util.cpp
)

add_library(conv_util SHARED ${CONV_UTIL_SOURCE})
target_link_libraries(conv_util PRIVATE host_tensor)
target_compile_features(conv_util PUBLIC)
set_target_properties(conv_util PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(conv_util SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

clang_tidy_check(conv_util)
