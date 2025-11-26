include(gtest)

# Helper function to create a gtest executable with common properties
function(add_ck_builder_test test_name)
    add_executable(${test_name} ${ARGN} testing_utils.cpp)
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

# The test_ckb_conv_builder target has all the unit tests (each test should run < 10 ms)
add_ck_builder_test(test_ckb_conv_builder
    test_conv_builder.cpp
    test_fwd_instance_traits.cpp
    test_bwd_weight_instance_traits.cpp
    test_bwd_data_instance_traits.cpp
    test_instance_traits_util.cpp)

add_ck_builder_test(test_ckb_inline_diff test_inline_diff.cpp)

# Testing the virtual GetInstanceString methods requires kernel compilation.
add_ck_builder_test(test_ckb_get_instance_string
    test_get_instance_string_fwd_grp_conv_v3.cpp
    test_get_instance_string_fwd_grp_conv.cpp
    test_get_instance_string_fwd_grp_conv_large_tensor.cpp
    test_get_instance_string_fwd_grp_conv_wmma.cpp
    test_get_instance_string_fwd_grp_conv_dl.cpp
    test_get_instance_string_bwd_weight_grp_conv_xdl.cpp)

# Testing the fwd convolution builder requires kernel compilation.
# To enable parallel compilation, the individual tests are split into separate files.
add_ck_builder_test(test_ckb_build_fwd_instances
    conv/test_ckb_conv_fwd_1d_fp16.cpp
    conv/test_ckb_conv_fwd_1d_bf16.cpp
    conv/test_ckb_conv_fwd_1d_i8.cpp
    conv/test_ckb_conv_fwd_2d_fp8.cpp
    conv/test_ckb_conv_fwd_2d_bf16.cpp
    conv/test_ckb_conv_fwd_2d_fp16.cpp
    conv/test_ckb_conv_fwd_2d_fp32.cpp
    conv/test_ckb_conv_fwd_2d_dl_fp16.cpp
    conv/test_ckb_conv_fwd_2d_large_tensor_fp16.cpp
    conv/test_ckb_conv_fwd_3d_bf16.cpp
    conv/test_ckb_conv_fwd_3d_fp16.cpp
    conv/test_ckb_conv_fwd_3d_fp32.cpp
    )

# Factory tests attempt to build all the kernels need by MIOpen.
# This is only for regression testing and development, the builds are too expensive for regular use in CI.
function(add_ck_factory_test test_name)
    add_ck_builder_test(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE composablekernels::device_conv_operations)
endfunction()

# TODO: add these tests back in once we have CI working across all GPU architectures.
add_ck_factory_test(test_ckb_testing_utils test_testing_utils.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward test_ck_factory_grouped_convolution_forward.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_clamp test_ck_factory_grouped_convolution_forward_clamp.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_convscale test_ck_factory_grouped_convolution_forward_convscale.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bilinear test_ck_factory_grouped_convolution_forward_bilinear.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scale test_ck_factory_grouped_convolution_forward_scale.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scaleadd_ab test_ck_factory_grouped_convolution_forward_scaleadd_ab.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bias_clamp test_ck_factory_grouped_convolution_forward_bias_clamp.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bias_bnorm_clamp test_ck_factory_grouped_convolution_forward_bias_bnorm_clamp.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scaleadd_scaleadd_relu test_ck_factory_grouped_convolution_forward_scaleadd_scaleadd_relu.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_dynamic_op test_ck_factory_grouped_convolution_forward_dynamic_op.cpp)

add_ck_builder_test(test_ckb_conv_traits
    conv/test_conv_traits.cpp)

add_ck_builder_test(test_ckb_conv_description
    test_conv_description.cpp)

# Register tests with CTest and assign labels
include(CTest)

# Smoke test: fast-compiling unit test
add_test(NAME test_ckb_conv_builder COMMAND test_ckb_conv_builder)
set_tests_properties(test_ckb_conv_builder PROPERTIES LABELS "BUILDER_SMOKE")

# Regression tests: all other tests that require kernel compilation
set(CKB_REGRESSION_TESTS
    test_ckb_inline_diff
    test_ckb_get_instance_string
    test_ckb_build_fwd_instances
    test_ckb_testing_utils
    test_ckb_factory_grouped_convolution_forward_convscale
    test_ckb_factory_grouped_convolution_forward_scaleadd_ab
    test_ckb_factory_grouped_convolution_forward_scaleadd_scaleadd_relu
    test_ckb_factory_grouped_convolution_forward_dynamic_op
    test_ckb_conv_traits
    test_ckb_conv_description
)

foreach(test_target ${CKB_REGRESSION_TESTS})
    add_test(NAME ${test_target} COMMAND ${test_target})
    set_tests_properties(${test_target} PROPERTIES LABELS "BUILDER_REGRESSION")
endforeach()

# Helper target to build all regression tests
add_custom_target(build-regression-builder DEPENDS ${CKB_REGRESSION_TESTS})

# Target to run only smoke tests (builds only test_ckb_conv_builder)
add_custom_target(smoke-builder
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "BUILDER_SMOKE"
    DEPENDS test_ckb_conv_builder
    USES_TERMINAL
    COMMENT "Running experimental builder smoke tests..."
)

# Target to run only regression tests (builds all regression test executables)
add_custom_target(regression-builder
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "BUILDER_REGRESSION"
    DEPENDS build-regression-builder
    USES_TERMINAL
    COMMENT "Running experimental builder regression tests..."
)

# Target to run all builder tests (builds all test executables)
add_custom_target(check-builder
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -R "^test_ckb"
    DEPENDS test_ckb_conv_builder build-regression-builder
    USES_TERMINAL
    COMMENT "Running all experimental builder tests..."
)

# Print summary of test organization
message(STATUS "CK Builder test organization:")
message(STATUS "  Smoke test: test_ckb_conv_builder")
message(STATUS "  Regression tests: ${CKB_REGRESSION_TESTS}")
