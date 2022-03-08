set(CONV2D_BWD_DATA_SOURCE
    conv2d_bwd_data.cpp
)

get_filename_component(CONV2D_BWD_DATA ${CONV2D_BWD_DATA_SOURCE} NAME_WE)
add_test_executable(test_${CONV2D_BWD_DATA} ${CONV2D_BWD_DATA_SOURCE})
target_link_libraries(test_${CONV2D_BWD_DATA} PRIVATE host_tensor)
target_link_libraries(test_${CONV2D_BWD_DATA} PRIVATE device_conv2d_bwd_data_instance)
