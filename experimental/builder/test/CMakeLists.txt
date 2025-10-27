include(gtest)

# Helper function to create a gtest executable with common properties
function(add_ck_builder_test test_name)
    add_executable(${test_name} ${ARGN})
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    target_include_directories(${test_name} PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/include"
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-c++20-compat
    )
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main GTest::gmock)
endfunction()

add_ck_builder_test(test_conv_builder
    test_conv_builder.cpp
    test_instance_traits.cpp
    testing_utils.cpp)

add_ck_builder_test(test_get_instance_string
    test_get_instance_string.cpp)

add_ck_builder_test(test_inline_diff test_inline_diff.cpp testing_utils.cpp)

add_ck_builder_test(test_ckb_build_fwd_instances
    conv/test_ckb_conv_fwd_2d_bf16.cpp
    conv/test_ckb_conv_fwd_2d_fp16.cpp
    conv/test_ckb_conv_fwd_2d_fp32.cpp
    conv/test_ckb_conv_fwd_3d_bf16.cpp
    conv/test_ckb_conv_fwd_3d_fp16.cpp
    conv/test_ckb_conv_fwd_3d_fp32.cpp)