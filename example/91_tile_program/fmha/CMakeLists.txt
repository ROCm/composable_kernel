# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate_kernels.py
  --list_kernels ${CMAKE_CURRENT_BINARY_DIR}/kernel_list.txt
)

# NOTE: for cmake, the FMHA_FWD_GEN_KERNELS files must be in the same directory
#       as current cmake list, otherwise will not figure out the dependency properly
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/kernel_list.txt FMHA_FWD_GEN_KERNELS)

add_custom_command(
  OUTPUT ${FMHA_FWD_GEN_KERNELS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate_kernels.py
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
)

add_example_executable(example_fmha_fwd fmha_fwd.cpp)
target_include_directories(example_fmha_fwd PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(example_fmha_fwd PRIVATE ${FMHA_FWD_GEN_KERNELS})

# NOTE: this is dangerous since will change the whole kernel to flush denormals
#       WIP with compiler team for an exp2 intrinsic..., then remove this
if(NOT DEFINED FMHA_FWD_FAST_EXP2)
    set(FMHA_FWD_FAST_EXP2 true)
endif()

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
#       ... because they are auto-generated
if(FMHA_FWD_FAST_EXP2)
target_compile_options(example_fmha_fwd PRIVATE "-Wno-undefined-func-template;-DCK_FMHA_FWD_FAST_EXP2=1;-fgpu-flush-denormals-to-zero")
else()
target_compile_options(example_fmha_fwd PRIVATE "-Wno-undefined-func-template;-DCK_FMHA_FWD_FAST_EXP2=0")
endif()
