# ONLY XDL_KERNELS
set(GEMM_BLOCKSCALE_WP_INSTANCES)

list(APPEND GEMM_BLOCKSCALE_WP_INSTANCES 
        device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instance.cpp
        device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instance.cpp
        device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp
        device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instance.cpp
        )

set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--slp-threshold=-32;-mllvm;--misched-bottomup=1")
set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--slp-threshold=-32;-mllvm;--misched-bottomup=1")
set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--slp-threshold=-32;-mllvm;--misched-bottomup=1")
set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--slp-threshold=-32;-mllvm;--misched-bottomup=1")

add_instance_library(device_gemm_blockscale_wp_instance ${GEMM_BLOCKSCALE_WP_INSTANCES})
