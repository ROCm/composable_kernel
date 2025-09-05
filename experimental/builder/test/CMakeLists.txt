set(CMAKE_CXX_STANDARD 20)

include(gtest)

# Helper function to create a gtest executable with common properties
function(add_ck_builder_test test_name)
    add_executable(${test_name} ${ARGN})
    target_include_directories(${test_name} PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/include"
        "${PROJECT_SOURCE_DIR}/include"
    )
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-c++20-compat
    )
    target_link_libraries(${test_name} PRIVATE GTest::gtest GTest::gtest_main)
endfunction()

add_ck_builder_test(test_conv_builder
    test_conv_builder.cpp
    test_conv_instances.cpp)

add_ck_builder_test(test_builder_utils test_builder_utils.cpp)