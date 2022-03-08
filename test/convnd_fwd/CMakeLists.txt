set(CONVND_FWD_SOURCE
    convnd_fwd.cpp
)

get_filename_component(CONVND_FWD ${CONVND_FWD_SOURCE} NAME_WE)
add_test_executable(test_${CONVND_FWD} ${CONVND_FWD_SOURCE})
target_link_libraries(test_${CONVND_FWD} PRIVATE host_tensor)
