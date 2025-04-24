set(FLASH_ATTENTION_FWD_KNOWN_APIS "fwd")
set(FLASH_ATTENTION_FWD_ENABLE_APIS "fwd" CACHE STRING
    "semicolon-separated list of APIs to generate (${FLASH_ATTENTION_FWD_KNOWN_APIS}) & link, or \"all\".")
if(FLASH_ATTENTION_FWD_ENABLE_APIS STREQUAL "all")
  set(FLASH_ATTENTION_FWD_ENABLE_APIS ${FLASH_ATTENTION_FWD_KNOWN_APIS})
endif()

option(TOY_FA_FWD_OPT "Enable toy flash attention forward optimization" ON)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api ${FLASH_ATTENTION_FWD_ENABLE_APIS}
  --working_path ${CMAKE_CURRENT_BINARY_DIR}
  --list_blobs
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "Failed to list Flash Attention kernels via Python. ${ret}")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/flash_attention_fwd_blobs.txt FLASH_ATTENTION_FWD_GEN_BLOBS)

add_custom_command(
  OUTPUT ${FLASH_ATTENTION_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api ${FLASH_ATTENTION_FWD_ENABLE_APIS}
  --working_path ${CMAKE_CURRENT_BINARY_DIR}
  --gen_blobs
)

set(EXAMPLE_FA "codegen_basic_flash_attention_fwd")
message("adding example ${EXAMPLE_FA}")

add_executable(${EXAMPLE_FA}
  EXCLUDE_FROM_ALL
  flash_attention_fwd.cpp
)

target_compile_definitions(${EXAMPLE_FA}
  PRIVATE
  $<$<BOOL:${TOY_FA_FWD_OPT}>:TOY_FA_FWD_OPT=1>
)

target_include_directories(${EXAMPLE_FA}
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
)

target_sources(${EXAMPLE_FA} PRIVATE ${FLASH_ATTENTION_FWD_GEN_BLOBS})

message("FLASH_ATTENTION_FWD_GEN_BLOBS = ${FLASH_ATTENTION_FWD_GEN_BLOBS}")

set(EXAMPLE_FA_COMPILE_OPTIONS)
list(APPEND EXAMPLE_FA_COMPILE_OPTIONS
  -Wno-undefined-func-template
  -Wno-float-equal
  --offload-compress
)

target_compile_options(${EXAMPLE_FA}
  PRIVATE
  ${EXAMPLE_FA_COMPILE_OPTIONS}
)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)