set(CONV_UTIL_SOURCE
    conv_util.cpp
)

add_library(conv_util SHARED ${CONV_UTIL_SOURCE})
target_link_libraries(conv_util PRIVATE host_tensor)
target_compile_features(conv_util PUBLIC)
set_target_properties(conv_util PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(conv_util SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

clang_tidy_check(conv_util)
