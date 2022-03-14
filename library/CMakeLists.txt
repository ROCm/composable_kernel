add_subdirectory(src/host_tensor)
add_subdirectory(src/tensor_operation_instance/gpu)

# rocm_install_targets(
#   TARGETS device_operations host_tensor
#   INCLUDE
#     ${PROJECT_SOURCE_DIR}/include
#     ${PROJECT_SOURCE_DIR}/library/include
# )