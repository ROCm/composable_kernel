add_executable(tile_example_flatmm_basic flatmm_basic.cpp)
rocm_install(TARGETS tile_example_flatmm_basic COMPONENT examples)


set(EXAMPLE_FLATMM_COMPILE_OPTIONS)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-unused-variable -Wno-unused-parameter)
# list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-unused-local-typedef)
target_compile_options(tile_example_flatmm_basic PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
