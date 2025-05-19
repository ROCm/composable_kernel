# 1D elementwise target
set(TARGET_NAME_1D tile_example_elementwise_1D)
add_executable(${TARGET_NAME_1D} elementwise_example_1D_basecase.cpp)
rocm_install(TARGETS ${TARGET_NAME_1D} COMPONENT examples)
target_include_directories(${TARGET_NAME_1D} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# 2D elementwise target
set(TARGET_NAME_2D tile_example_elementwise_2D)
add_executable(${TARGET_NAME_2D} elementwise_example_2D_basecase.cpp)
rocm_install(TARGETS ${TARGET_NAME_2D} COMPONENT examples)
target_include_directories(${TARGET_NAME_2D} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# 2D elementwise 3D input target
set(TARGET_NAME_2D_3D_INPUT tile_example_elementwise_2D_3D_input)
add_executable(${TARGET_NAME_2D_3D_INPUT} elementwise_example_2D_basecase_3D_input.cpp)
rocm_install(TARGETS ${TARGET_NAME_2D_3D_INPUT} COMPONENT examples)
target_include_directories(${TARGET_NAME_2D_3D_INPUT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# N-ary elementwise target
set(TARGET_NAME_N_ARY tile_example_elementwise_nary)
add_executable(${TARGET_NAME_N_ARY} elementwise_example_nary.cpp)
rocm_install(TARGETS ${TARGET_NAME_N_ARY} COMPONENT examples)
target_include_directories(${TARGET_NAME_N_ARY} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# N-ary elementwise target with 2D input
set(TARGET_NAME_N_ARY_2D_INPUT tile_example_elementwise_nary_2D_input)
add_executable(${TARGET_NAME_N_ARY_2D_INPUT} elementwise_example_nary_1D_basecase_2D_input.cpp)
rocm_install(TARGETS ${TARGET_NAME_N_ARY_2D_INPUT} COMPONENT examples)
target_include_directories(${TARGET_NAME_N_ARY_2D_INPUT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -Wno-unused-variable -Rpass-analysis=kernel-resource-usage)
# list(APPEND EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(${TARGET_NAME_1D} PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
target_compile_options(${TARGET_NAME_2D} PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
target_compile_options(${TARGET_NAME_2D_3D_INPUT} PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
target_compile_options(${TARGET_NAME_N_ARY} PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
target_compile_options(${TARGET_NAME_N_ARY_2D_INPUT} PRIVATE ${EXAMPLE_ELEMENTWISE_COMPILE_OPTIONS})
