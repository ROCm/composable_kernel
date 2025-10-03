
set(EXAMPLE_NAME builder_example_conv_fwd_2d)
set(EXAMPLE_SOURCES ckb_example_conv_fwd_2d.cpp)

add_example_executable(${EXAMPLE_NAME} "${EXAMPLE_SOURCES}")

if(TARGET ${EXAMPLE_NAME})
    target_include_directories(${EXAMPLE_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/experimental/builder/include
    )
    target_compile_options(${EXAMPLE_NAME} PRIVATE
        -Wno-global-constructors
        -Wno-c++20-compat
    )
endif()