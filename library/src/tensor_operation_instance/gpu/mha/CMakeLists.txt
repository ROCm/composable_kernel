set(DEVICE_FMHA_INSTANCES)

# python stuff
find_package(PythonInterp 3.10 REQUIRED)
if(${PYTHON_VERSION_STRING} VERSION_GREATER_EQUAL 3.10)
    message(STATUS "Python version is greater than or equal to 3.10: ${PYTHON_VERSION_STRING}")
else()
    message(FATAL_ERROR "Python version is less than 3.10: ${PYTHON_VERSION_STRING}")
endif()

# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/composable_kernel_sub_module/example/ck_tile/01_fmha/generate.py
  --list_blobs ${CMAKE_CURRENT_SOURCE_DIR}/blob_list.txt
)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/blob_list.txt FMHA_FWD_GEN_BLOBS)

# actually generate the cpp files
add_custom_command(
  OUTPUT ${FMHA_FWD_GEN_BLOBS}
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/composable_kernel_sub_module/example/ck_tile/01_fmha/generate.py
  --output_dir ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mha kernel (cpp) files now ..."
  VERBATIM
)

add_custom_target(generate_cpp_files
    DEPENDS FMHA_FWD_GEN_BLOBS)

add_instance_library(device_fmha_instance ${FMHA_FWD_GEN_BLOBS})
