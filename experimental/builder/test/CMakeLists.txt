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

function(add_ck_factory_test test_name)
    add_ck_builder_test(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE composablekernels::device_conv_operations)
endfunction()

add_ck_factory_test(test_ckb_testing_utils test_testing_utils.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward test_ck_factory_grouped_convolution_forward.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_clamp test_ck_factory_grouped_convolution_forward_clamp.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_convscale test_ck_factory_grouped_convolution_forward_convscale.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bilinear test_ck_factory_grouped_convolution_forward_bilinear.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scale test_ck_factory_grouped_convolution_forward_scale.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scaleadd_ab test_ck_factory_grouped_convolution_forward_scaleadd_ab.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bias_clamp test_ck_factory_grouped_convolution_forward_bias_clamp.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bias_bnorm_clamp test_ck_factory_grouped_convolution_forward_bias_bnorm_clamp.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scaleadd_scaleadd_relu test_ck_factory_grouped_convolution_forward_scaleadd_scaleadd_relu.cpp)
add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_dynamic_op test_ck_factory_grouped_convolution_forward_dynamic_op.cpp)

add_ck_builder_test(test_ckb_conv_traits
    conv/test_conv_traits.cpp)

add_ck_builder_test(test_ckb_conv_description
    test_conv_description.cpp)

# Function to add all test_ckb targets to a list
function(collect_test_ckb_targets result_var)
    # Get all targets in current directory
    get_directory_property(all_targets BUILDSYSTEM_TARGETS)
    
    set(test_ckb_targets)
    foreach(target ${all_targets})
        # Check if target name starts with "test_ckb"
        string(REGEX MATCH "^test_ckb" match_result ${target})
        if(match_result)
            list(APPEND test_ckb_targets ${target})
        endif()
    endforeach()
    
    set(${result_var} ${test_ckb_targets} PARENT_SCOPE)
endfunction()

# Create the custom target
collect_test_ckb_targets(TEST_CKB_TARGETS)
add_custom_target(test_ckb_all)
add_dependencies(test_ckb_all ${TEST_CKB_TARGETS})

# Optional: Print the collected targets for verification
message(STATUS "Found following CK Builder test targets: ${TEST_CKB_TARGETS}")
