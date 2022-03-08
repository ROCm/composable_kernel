## host_tensor
include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/ck
    ${PROJECT_SOURCE_DIR}/include/ck/utility
    ${PROJECT_SOURCE_DIR}/library/include/ck/library/host_tensor
)

file(GLOB_RECURSE HOST_TENSOR_SOURCE "./*.cpp")
add_library(host_tensor SHARED ${HOST_TENSOR_SOURCE})
target_link_libraries(host_tensor PRIVATE hip::device)
target_link_libraries(host_tensor INTERFACE hip::host)
target_compile_features(host_tensor PUBLIC)
set_target_properties(host_tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS host_tensor LIBRARY DESTINATION lib)

clang_tidy_check(host_tensor)
