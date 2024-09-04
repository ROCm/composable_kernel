set(FMHA_CPP_FOLDER ${CMAKE_CURRENT_BINARY_DIR})
set(FMHA_SRC_FOLDER ${CMAKE_SOURCE_DIR}/example/ck_tile/01_fmha/)
set(CK_TILE_SRC_FOLDER ${CMAKE_SOURCE_DIR}/include/ck_tile/)
# python stuff
find_package(PythonInterp 3 REQUIRED)

rocm_install(DIRECTORY ${CK_TILE_SRC_FOLDER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck_tile)

file(GLOB MHA_HEADERS "${FMHA_SRC_FOLDER}/*.hpp")
rocm_install(FILES ${MHA_HEADERS} DESTINATION include/ck_tile/ops)
# headers for building lib
file(COPY ${MHA_HEADERS} DESTINATION ${FMHA_CPP_FOLDER})

# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --list_blobs ${FMHA_CPP_FOLDER}/blob_list.txt
)
file(STRINGS ${FMHA_CPP_FOLDER}/blob_list.txt FMHA_FWD_GEN_BLOBS)

# actually generate the cpp files
add_custom_command(
  OUTPUT ${FMHA_FWD_GEN_BLOBS}
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
  --output_dir ${FMHA_CPP_FOLDER}
  COMMENT "Generating mha kernel (cpp) files now ..."
  VERBATIM
)

# This is done to remove path info and just
# have filename. Since, it was cauing the cmake
# to throw "File name too long"
set(device_files)
foreach(filepath IN LISTS FMHA_FWD_GEN_BLOBS)
    get_filename_component(filename ${filepath} NAME)
    # Append the filename to the device_files list
    list(APPEND device_files ${filename})
endforeach()
add_custom_target(generate_cpp_files DEPENDS ${FMHA_FWD_GEN_BLOBS})

add_instance_library(device_mha_instance ${device_files})



if (TARGET device_mha_instance)
  add_dependencies(device_mha_instance generate_cpp_files)
endif()

