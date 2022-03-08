set(REFERENCE_CONV_FWD_SOURCE
    reference_conv_fwd.cpp
)

get_filename_component(REFERENCE_CONV_FWD ${REFERENCE_CONV_FWD_SOURCE} NAME_WE)
add_test_executable(test_${REFERENCE_CONV_FWD} ${REFERENCE_CONV_FWD_SOURCE})
target_link_libraries(test_${REFERENCE_CONV_FWD} PRIVATE host_tensor)
