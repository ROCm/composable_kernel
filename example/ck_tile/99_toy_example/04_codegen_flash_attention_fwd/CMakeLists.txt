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

set(EXAMPLE_REDUCE "codegen_basic_flash_attention_fwd")
message("adding example ${EXAMPLE_REDUCE}")

add_executable(${EXAMPLE_REDUCE}
  EXCLUDE_FROM_ALL
  flash_attention_fwd.cpp
)

target_compile_definitions(${EXAMPLE_REDUCE}
  PRIVATE
  $<$<BOOL:${TOY_FA_FWD_OPT}>:TOY_FA_FWD_OPT=1>
)

target_include_directories(${EXAMPLE_REDUCE}
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
)

target_sources(${EXAMPLE_REDUCE} PRIVATE ${FLASH_ATTENTION_FWD_GEN_BLOBS})

message("FLASH_ATTENTION_FWD_GEN_BLOBS = ${FLASH_ATTENTION_FWD_GEN_BLOBS}")

set(EXAMPLE_REDUCE_COMPILE_OPTIONS)
list(APPEND EXAMPLE_REDUCE_COMPILE_OPTIONS
  -Wno-undefined-func-template
  -Wno-float-equal
  --offload-compress
)

# option(ENABLE_TOY_FA_FWD_OPT "Enable toy FA fwd optimization" ON)
# if(ENABLE_TOY_FA_FWD_OPT)
#     message("Compiling with toy FA fwd optimization")
#     target_compile_definitions(${EXAMPLE_REDUCE} PRIVATE TOY_FA_FWD_OPT)
# endif()

target_compile_options(${EXAMPLE_REDUCE}
  PRIVATE
  ${EXAMPLE_REDUCE_COMPILE_OPTIONS}
)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)