list(APPEND gpu_list_xdl gfx908 gfx90a gfx940 gfx941 gfx942 gfx950)
list(APPEND gpu_list_wmma gfx1100 gfx1101 gfx1102 gfx1200 gfx1201)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list_xdl AND target EQUAL 0)
        add_custom_target(example_grouped_conv_bwd_weight)
        add_example_executable(example_grouped_conv_bwd_weight_xdl_fp16 grouped_conv_bwd_weight_xdl_fp16.cpp)
        add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_fp16)

        add_example_executable(example_grouped_conv_bwd_weight_xdl_bf16 grouped_conv_bwd_weight_xdl_bf16.cpp)
        add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_bf16)

        add_example_executable(example_grouped_conv_bwd_weight_xdl_fp16_comp_bf8_fp8 grouped_conv_bwd_weight_xdl_fp16_comp_bf8_fp8.cpp)
        add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_fp16_comp_bf8_fp8)
        set(target 1)
    endif()

    if(gpu IN_LIST gpu_list_wmma AND target EQUAL 0)
        add_custom_target(example_grouped_conv_bwd_weight)
        add_example_executable(example_grouped_conv_bwd_weight_wmma_fp16 grouped_conv_bwd_weight_wmma_fp16.cpp)
        add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_wmma_fp16)
        set(target 1)
    endif()
endforeach()

add_custom_target(example_grouped_conv_bwd_weight_dl)

add_example_executable(example_grouped_conv_bwd_weight_dl_fp16 grouped_conv_bwd_weight_dl_fp16.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight_dl example_grouped_conv_bwd_weight_dl_fp16)
