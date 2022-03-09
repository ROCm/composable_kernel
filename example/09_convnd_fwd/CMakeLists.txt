set(CONVND_FWD_XDL_SOURCE convnd_fwd_xdl.cpp)
add_executable(convnd_fwd_xdl ${CONVND_FWD_XDL_SOURCE})
target_link_libraries(convnd_fwd_xdl PRIVATE host_tensor)
