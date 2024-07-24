set(FMHA_CPP_FOLDER ${CMAKE_CURRENT_BINARY_DIR})
set(FMHA_SRC_FOLDER ${CMAKE_SOURCE_DIR}/example/ck_tile/01_fmha/)

# python stuff
find_package(PythonInterp 3 REQUIRED)

file(COPY ${FMHA_SRC_FOLDER}/fmha_fwd.hpp DESTINATION ${FMHA_CPP_FOLDER})
file(COPY ${FMHA_SRC_FOLDER}/bias.hpp DESTINATION ${FMHA_CPP_FOLDER})
file(COPY ${FMHA_SRC_FOLDER}/mask.hpp DESTINATION ${FMHA_CPP_FOLDER})

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
add_custom_target(generate_cpp_files DEPENDS ${FMHA_FWD_GEN_BLOBS})

add_instance_library(device_mha_instance ${FMHA_FWD_GEN_BLOBS})

add_dependencies(device_mha_instance generate_cpp_files)


