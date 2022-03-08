## host_tensor
include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
)

set(HOST_TENSOR_SOURCE
    device.cpp
    host_tensor.cpp
)

add_library(host_tensor SHARED ${HOST_TENSOR_SOURCE})
target_compile_features(host_tensor PUBLIC)
set_target_properties(host_tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS host_tensor LIBRARY DESTINATION lib)

clang_tidy_check(host_tensor)
