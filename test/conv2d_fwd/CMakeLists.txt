set(CONV2D_FWD_SOURCE
    conv2d_fwd.cpp
)

get_filename_component(CONV2D_FWD ${CONV2D_FWD_SOURCE} NAME_WE)
add_test_executable(test_${CONV2D_FWD} ${CONV2D_FWD_SOURCE})
target_link_libraries(test_${CONV2D_FWD} PRIVATE host_tensor)
target_link_libraries(test_${CONV2D_FWD} PRIVATE device_conv2d_fwd_instance)
