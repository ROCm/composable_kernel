set(POOL2D_FWD_SOURCE pool2d_fwd.cpp)
add_executable(pool2d_fwd ${POOL2D_FWD_SOURCE})
target_link_libraries(pool2d_fwd PRIVATE host_tensor)
