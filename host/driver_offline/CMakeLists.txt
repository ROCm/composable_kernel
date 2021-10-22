include_directories(BEFORE
    include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/device/include
    ${PROJECT_SOURCE_DIR}/host/solver/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/driver
    ${PROJECT_SOURCE_DIR}/external/rocm/include
)

set(CONV_FWD_DRIVER_OFFLINE_SOURCE src/conv_fwd_driver_offline.cpp)
set(CONV_FWD_DRIVER_OFFLINE_NCHWC_SOURCE src/conv_fwd_driver_offline_nchwc.cpp)
set(CONV_ADD_FWD_DRIVER_OFFLINE_NCHWC_SOURCE src/conv_add_fwd_driver_offline_nchwc.cpp)
set(CONV_MAXPOOL_FWD_DRIVER_OFFLINE_NCHWC_SOURCE src/conv_maxpool_fwd_driver_offline_nchwc.cpp)
set(CONV_BWD_DRIVER_OFFLINE_SOURCE src/conv_bwd_driver_offline.cpp)
set(CONV_WRW_DRIVER_OFFLINE_SOURCE src/conv_wrw_driver_offline.cpp)
set(GEMM_DRIVER_OFFLINE_SOURCE src/gemm_driver_offline.cpp)

set(CONV3D_FWD_DRIVER_OFFLINE_SOURCE src/conv3d_fwd_driver_offline.cpp)

add_executable(conv_fwd_driver_offline ${CONV_FWD_DRIVER_OFFLINE_SOURCE})
add_executable(conv_fwd_driver_offline_nchwc ${CONV_FWD_DRIVER_OFFLINE_NCHWC_SOURCE})
add_executable(conv_add_fwd_driver_offline_nchwc ${CONV_ADD_FWD_DRIVER_OFFLINE_NCHWC_SOURCE})
add_executable(conv_maxpool_fwd_driver_offline_nchwc ${CONV_MAXPOOL_FWD_DRIVER_OFFLINE_NCHWC_SOURCE})
add_executable(conv_bwd_driver_offline ${CONV_BWD_DRIVER_OFFLINE_SOURCE})
add_executable(conv_wrw_driver_offline ${CONV_WRW_DRIVER_OFFLINE_SOURCE})
add_executable(gemm_driver_offline ${GEMM_DRIVER_OFFLINE_SOURCE})

add_executable(conv3d_fwd_driver_offline ${CONV3D_FWD_DRIVER_OFFLINE_SOURCE})

target_link_libraries(conv_fwd_driver_offline PRIVATE host_tensor)
target_link_libraries(conv_fwd_driver_offline_nchwc PRIVATE host_tensor)
target_link_libraries(conv_add_fwd_driver_offline_nchwc PRIVATE host_tensor)
target_link_libraries(conv_maxpool_fwd_driver_offline_nchwc PRIVATE host_tensor)
target_link_libraries(conv_bwd_driver_offline PRIVATE host_tensor)
target_link_libraries(conv_wrw_driver_offline PRIVATE host_tensor)
target_link_libraries(gemm_driver_offline PRIVATE host_tensor)

target_link_libraries(conv3d_fwd_driver_offline PRIVATE host_tensor)
