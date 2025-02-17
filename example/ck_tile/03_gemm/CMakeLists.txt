function (add_gemm_example TARGET_NAME MAIN_SRC)
  message("adding ${TARGET_NAME}")
  # not using add_example_executable() to add target, since we don't want this to have
  # to be included in "make all/install/check"
  add_executable(${TARGET_NAME} EXCLUDE_FROM_ALL ${MAIN_SRC})
  target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

  foreach(source IN LISTS ARGN)
      list(APPEND INSTANCE_SRCS ${source})
  endforeach()

  target_sources(${TARGET_NAME} PRIVATE ${INSTANCE_SRCS})

  set(COMPILE_OPTIONS)
  # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
  list(APPEND COMPILE_OPTIONS -Wno-undefined-func-template)

  target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})
endfunction(add_gemm_example TARGET_NAME MAIN_SRC)

file(GLOB INSTANCE_SRCS instances/*.cpp)
add_gemm_example(tile_example_gemm_universal universal_gemm.cpp ${INSTANCE_SRCS})
target_compile_options(tile_example_gemm_universal PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)

add_executable(tile_example_gemm_basic EXCLUDE_FROM_ALL gemm_basic.cpp)
