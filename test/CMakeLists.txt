# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/profiler/include
)

include(gtest)

add_custom_target(tests)

# list of tests that are labelled as REGRESSION_TEST for make regression (runtime more than 60 seconds on emulator)
# all other tests are labelled as SMOKE_TEST
set(REGRESSION_TESTS
    test_gemm_standalone_xdl_fp16
    test_gemm_fp16
    test_gemm_splitk
    test_gemm_universal_wmma_fp16
    test_gemm_universal_xdl_fp16
    test_batched_gemm_softmax_gemm_fp16
    test_batched_gemm_softmax_gemm_permute_fp16
    test_batched_gemm_bias_softmax_gemm_permute_fp16
    test_batched_gemm_softmax_gemm_permute_bf16
    test_batched_gemm_bias_softmax_gemm_permute_bf16
    test_grouped_gemm_splitk
    test_batched_gemm_b_scale_wmma
    test_reduce_no_index
    test_reduce_with_index
    test_convnd_fwd
    test_convnd_bwd_data
    test_grouped_convnd_fwd
    test_grouped_convnd_fwd_bias_bnorm_clamp
    test_grouped_convnd_fwd_scaleadd_ab
    test_grouped_convnd_bwd_weight
    test_softmax_rank3
    test_softmax_rank4
    test_batchnorm_fwd_rank_4
    test_batchnorm_bwd_rank_4
    test_grouped_convnd_bwd_data_xdl
    test_grouped_convnd_bwd_data_wmma
    test_grouped_convnd_bwd_data_wmma_large_cases
    test_grouped_conv_bwd_data_wmma_scale
    test_conv_tensor_rearrange
    test_gemm_mx
    test_ck_tile_batched_transpose
    test_ck_tile_fmha_bwd_fp32
    test_ck_tile_fmha_bwd_bf16
    test_ck_tile_fmha_bwd_fp16
    test_ck_tile_fmha_fwd_fp32
    test_ck_tile_fmha_fwd_bf16
    test_ck_tile_fmha_fwd_fp16
    test_ck_tile_fmha_fwd_fp8bf16
    test_ck_tile_fmha_fwd_mxfp8
    test_ck_tile_fmha_fwd_mxfp4
    test_ck_tile_streamk_extended
    test_amdgcn_mma_layout_gfx1250
    test_gemm_multiply_multiply_wmma
    test_gemm_multiply_multiply_wp_fp8
    test_gemm_universal_fp16
    test_gemm_universal_bf16
    test_gemm_universal_preshuffle_fp8
    test_gemm_ab_scale
    test_batched_contraction
    test_grouped_convnd_fwd_bilinear
    test_grouped_convnd_fwd_dynamic_op
    test_grouped_convnd_fwd_bias_clamp
    test_grouped_convnd_fwd_gk_bias_clamp
    test_grouped_convnd_fwd_clamp
    test_grouped_convnd_fwd_scale
    test_grouped_convnd_fwd_gk_bias_bnorm_clamp
    test_grouped_convnd_fwd_scaleadd_scaleadd_relu
    test_contraction
    test_avg_pool3d_fwd
    test_max_pool3d_fwd
    test_grouped_convnd_bwd_data
    test_monitor_mwait
    test_ck_tile_gemm_pipeline_tdm_wmma
    test_ck_tile_gemm_pipeline_wp
    test_ck_tile_gemm_pipeline_wp_tdm
    test_ck_tile_grouped_gemm_multi_d
    test_ck_tile_grouped_gemm_preshuffle
    test_memory_copy
    test_fill
    test_ck_tile_pooling
    test_magic_number_division
    test_gemm_fp32
    test_gemm_bf16
    test_gemm_int8
    test_gemm_add
    test_gemm_add_relu
    test_gemm_add_silu
    test_gemm_add_fastgelu
    test_gemm_fastgelu_wmma
    test_gemm_add_add_fastgelu_wmma
    test_gemm_add_multiply_wmma
    test_gemm_multiply_add_wmma
    test_gemm_bilinear_wmma
    test_gemm_add_relu_add_layernorm_fp16
    test_gemm_multi_abd
    test_gemm_universal_fp8
    test_gemm_b_scale
    test_batched_gemm
    test_batched_gemm_gemm_fp16
    test_batched_gemm_gemm_bf16_wmma_cshuffle_v3
    test_grouped_gemm_fastgelu
    test_grouped_gemm_fixed_nk
    test_grouped_gemm_fixed_nk_bias
    test_grouped_gemm_tile_loop_vanilla
    test_grouped_gemm_tile_loop_multiply
    test_grouped_convnd_fwd_wavelet
    test_grouped_convnd_bwd_weight_bilinear
    test_grouped_convnd_bwd_weight_scale
    test_layernorm2d_fwd_fp32
    test_groupnorm_fwd_fp32
    test_layernorm2d_fwd_fp16
    test_elementwise_layernorm_fp16
    test_batchnorm_infer_rank_4
    test_avg_pool3d_bwd
    test_avg_pool2d_fwd
    test_max_pool2d_fwd
    test_gemm_quantization
)
# List of tests currently failing on gfx1250 emulator.
# All of the failing tests must be fixed asap.
set(KNOWN_FAILING_TESTS
    test_ck_tile_mx_scale
    test_tile_gemm_quant_bquant_microscale_rcr_1d_64
    test_tile_gemm_quant_bquant_microscale_ccr_1d_64
    test_tile_gemm_quant_bquant_microscale_rcr_1d_128
    test_tile_gemm_quant_bquant_microscale_ccr_1d_128
    test_ck_tile_mx_gemm_pipeline_tdm_wmma
    test_gemm_blockscale_wp_fp8
    test_grouped_gemm_multi_abd_fixed_nk
    test_fp6
    test_bf6
    test_mx_fp4
    test_mx_fp6
    test_mx_bf6
    test_mx_fp8_pk4scale
    test_mx_bf8_pk4scale
)

function(add_test_executable TEST_NAME)
    message(DEBUG "adding test ${TEST_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
        foreach(source IN LISTS ARGN)
            get_filename_component(source_name ${source} NAME)
            set(test 0)
            if((source_name MATCHES "_fp16|_f16") AND NOT "fp16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp32|_f32") AND NOT "fp32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_tf32|_tf32") AND NOT "tf32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp64|_f64") AND NOT "fp64" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp8|_f8") AND NOT "fp8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf8|_bf8") AND NOT "bf8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf16|_b16") AND NOT "bf16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_int8|_i8") AND NOT "int8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if(test EQUAL 1)
                message(DEBUG "removing test ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endforeach()
    endif()

    set(TEST_TARGETS ${SUPPORTED_GPU_TARGETS})

    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        if(NOT DEFINED DPP_KERNELS AND source_name MATCHES "_dpp")
            message(DEBUG "removing dpp test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT DEFINED DL_KERNELS AND source_name MATCHES "_dl")
            message(DEBUG "removing dl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx9|gfx11|gfx12" AND source_name MATCHES "xdl")
            message(DEBUG "removing xdl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx11|gfx12" AND source_name MATCHES "wmma")
            message(DEBUG "removing wmma test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()
    #only continue if there are some source files left on the list
    set(source_name_list "")
    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        list(APPEND source_name_list ${source_name})
    endforeach()
    if(ARGN)
        if(source_name_list MATCHES "_xdl")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx10-3-generic amdgcnspirv)
        elseif(source_name_list MATCHES "_wmma")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx950 amdgcnspirv)
        elseif(source_name_list MATCHES "_smfmac")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx908 gfx90a gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic gfx1250 amdgcnspirv)
        endif()
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
        add_executable(${TEST_NAME} ${ARGN})
        set_property(TARGET ${TEST_NAME} PROPERTY HIP_ARCHITECTURES ${TEST_TARGETS} )
        target_link_libraries(${TEST_NAME} PRIVATE getopt::getopt)
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
        add_dependencies(tests ${TEST_NAME})
        add_dependencies(check ${TEST_NAME})
        rocm_install(TARGETS ${TEST_NAME} COMPONENT tests)
        set(result 0)
    endif()
    message(DEBUG "add_test returns ${result}")
    set(result ${result} PARENT_SCOPE)
    if(result EQUAL 0 AND NOT "${TEST_NAME}" IN_LIST REGRESSION_TESTS AND NOT "${TEST_NAME}" IN_LIST KNOWN_FAILING_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "SMOKE_TEST")
        add_dependencies(smoke ${TEST_NAME})
    elseif(result EQUAL 0 AND "${TEST_NAME}" IN_LIST REGRESSION_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "REGRESSION_TEST")
        add_dependencies(regression ${TEST_NAME})
    endif()
endfunction()

function(add_gtest_executable TEST_NAME)
    message(DEBUG "adding gtest ${TEST_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
        foreach(source IN LISTS ARGN)
            get_filename_component(source_name ${source} NAME)
            set(test 0)
            if((source_name MATCHES "_fp16|_f16") AND NOT "fp16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp32|_f32") AND NOT "fp32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_tf32|_tf32") AND NOT "tf32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp64|_f64") AND NOT "fp64" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp8|_f8") AND NOT "fp8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf8|_bf8") AND NOT "bf8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf16|_b16") AND NOT "bf16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_int8|_i8") AND NOT "int8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if(test EQUAL 1)
                message(DEBUG "removing gtest ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endforeach()
    endif()

    set(TEST_TARGETS ${SUPPORTED_GPU_TARGETS})

    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        if(NOT DEFINED DL_KERNELS AND source_name MATCHES "_dl")
            message(DEBUG "removing dl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx9|gfx11|gfx12" AND source_name MATCHES "xdl")
            message(DEBUG "removing xdl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx95|gfx125" AND source_name MATCHES "mx_")
            message(DEBUG "removing microscaling test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx11|gfx12" AND source_name MATCHES "wmma")
            message(DEBUG "removing wmma test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()

    foreach(source IN LISTS ARGN)
        if((source MATCHES "monitor_mwait" OR source MATCHES "async_lds_load_store") AND NOT TEST_TARGETS MATCHES "gfx125" )
             message(DEBUG "removing synchronization test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()

    #only continue if there are some source files left on the list
    set(source_name_list "")
    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        list(APPEND source_name_list ${source_name})
    endforeach()
    if(ARGN)
        if(source_name_list MATCHES "_xdl")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx10-3-generic amdgcnspirv)
        elseif(source_name_list MATCHES "_wmma")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx950 amdgcnspirv)
        elseif(source_name_list MATCHES "monitor_mwait" OR source_name_list MATCHES "async_lds_load_store") #only build sync tests for gfx1250+
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx950 gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic amdgcnspirv)
        elseif(source_name_list MATCHES "_smfmac")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx908 gfx90a gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic gfx1250 amdgcnspirv)
        elseif(source_name_list MATCHES "_mx") #only build mx example for gfx950 and gfx125x
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic amdgcnspirv)
             if(ARGN MATCHES "_pk4scale") #only build pk4scale for gfx1250
                list(REMOVE_ITEM TEST_TARGETS gfx950)
             endif()
        endif()
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
        add_executable(${TEST_NAME} ${ARGN})
        set_property(TARGET ${TEST_NAME} PROPERTY HIP_ARCHITECTURES ${TEST_TARGETS} )
        add_dependencies(tests ${TEST_NAME})
        add_dependencies(check ${TEST_NAME})

        # suppress gtest warnings
        target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
        target_link_libraries(${TEST_NAME} PRIVATE gtest_main getopt::getopt)
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
        rocm_install(TARGETS ${TEST_NAME} COMPONENT tests)
        set(result 0)
    endif()
    message(DEBUG "add_gtest returns ${result}")
    set(result ${result} PARENT_SCOPE)
    if(result EQUAL 0 AND NOT "${TEST_NAME}" IN_LIST REGRESSION_TESTS AND NOT "${TEST_NAME}" IN_LIST KNOWN_FAILING_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "SMOKE_TEST")
        add_dependencies(smoke ${TEST_NAME})
    elseif(result EQUAL 0 AND "${TEST_NAME}" IN_LIST REGRESSION_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "REGRESSION_TEST")
        add_dependencies(regression ${TEST_NAME})
    endif()
endfunction()

add_compile_options(-Wno-c++20-extensions)
add_subdirectory(ck_tile)
add_subdirectory(magic_number_division)
add_subdirectory(space_filling_curve)
add_subdirectory(threadwise_transfer_helper)
add_subdirectory(conv_util)
add_subdirectory(reference_conv_fwd)
add_subdirectory(gemm)
add_subdirectory(gemm_add)
add_subdirectory(gemm_bias_add_reduce)
add_subdirectory(gemm_blockscale_wp)
add_subdirectory(gemm_layernorm)
add_subdirectory(gemm_multi_abd)
add_subdirectory(gemm_multiply_multiply_wp)
add_subdirectory(gemm_split_k)
add_subdirectory(gemm_universal)
add_subdirectory(gemm_universal_preshuffle)
add_subdirectory(gemm_ab_scale)
add_subdirectory(gemm_b_scale)
add_subdirectory(gemm_reduce)
add_subdirectory(gemm_universal_reduce)
add_subdirectory(batched_contraction)
add_subdirectory(batched_gemm)
add_subdirectory(batched_gemm_reduce)
add_subdirectory(batched_gemm_gemm)
add_subdirectory(batched_gemm_multiple_d_gemm_multiple_d)
add_subdirectory(batched_gemm_softmax_gemm)
add_subdirectory(batched_gemm_softmax_gemm_permute)
add_subdirectory(batched_gemm_b_scale)
add_subdirectory(grouped_gemm)
add_subdirectory(grouped_gemm_tile_loop)
add_subdirectory(reduce)
add_subdirectory(convnd_fwd)
add_subdirectory(convnd_bwd_data)
add_subdirectory(grouped_convnd_fwd)
add_subdirectory(grouped_convnd_fwd_activation)
add_subdirectory(grouped_convnd_bwd_weight)
add_subdirectory(block_to_ctile_map)
add_subdirectory(softmax)
add_subdirectory(normalization_fwd)
add_subdirectory(normalization_bwd_data)
add_subdirectory(normalization_bwd_gamma_beta)
add_subdirectory(data_type)
add_subdirectory(elementwise_normalization)
add_subdirectory(batchnorm)
add_subdirectory(contraction)
add_subdirectory(pool)
add_subdirectory(batched_gemm_multi_d)
add_subdirectory(grouped_convnd_bwd_data)
add_subdirectory(conv_tensor_rearrange)
add_subdirectory(transpose)
add_subdirectory(permute_scale)
add_subdirectory(wrapper)
add_subdirectory(quantization)
add_subdirectory(device_memory)
if(SUPPORTED_GPU_TARGETS MATCHES "gfx11|gfx12")
    add_subdirectory(wmma_op)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx942|gfx950") # smfmac needs ROCm6.2
    add_subdirectory(smfmac_op)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx950")
    add_subdirectory(mx_mfma_op)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_subdirectory(mx_wmma_op)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx950|gfx125")
    add_subdirectory(gemm_mx)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx12")
    add_subdirectory(s_prefetch_op)
    add_subdirectory(s_prefetch_inst_op)
endif()

if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_subdirectory(prefetch_op)
endif()

add_subdirectory(cluster_load)
add_subdirectory(cluster_launch)

add_subdirectory(position_embedding)
add_subdirectory(scatter_gather)
add_subdirectory(synchronization)
add_subdirectory(gpu_reference)
add_subdirectory(util)
add_subdirectory(gpu_verification)
add_subdirectory(swiglu_oai_activation)
