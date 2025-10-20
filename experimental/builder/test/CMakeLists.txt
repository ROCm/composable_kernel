
include(gtest)

# Helper function to create a gtest executable with common properties
function(add_ck_builder_test test_name)
    add_executable(${test_name} ${ARGN})
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    target_include_directories(${test_name} PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/include"
        "${PROJECT_SOURCE_DIR}/include"
    )
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-c++20-compat
    )
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main GTest::gmock)
endfunction()

add_ck_builder_test(test_conv_builder
    test_conv_builder.cpp)
