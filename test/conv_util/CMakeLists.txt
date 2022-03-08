set(CONV_UTIL_SOURCE
    conv_util.cpp
)

get_filename_component(CONV_UTIL ${CONV_UTIL_SOURCE} NAME_WE)
add_test_executable(test_${CONV_UTIL} ${CONV_UTIL_SOURCE})
target_link_libraries(test_${CONV_UTIL} PRIVATE host_tensor)
