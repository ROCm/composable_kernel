set(CONV3D_FWD_XDL_SOURCE conv3d_fwd_xdl.cpp)
add_executable(conv3d_fwd_xdl ${CONV3D_FWD_XDL_SOURCE})
target_link_libraries(conv3d_fwd_xdl PRIVATE host_tensor)
