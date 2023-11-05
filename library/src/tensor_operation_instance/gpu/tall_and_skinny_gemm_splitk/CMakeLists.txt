list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
 if(gpu IN_LIST gpu_list AND target EQUAL 0)

    set(TALL_AND_SKINNY_GEMM_SPLITK_INSTANCES)


    if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
      list(APPEND TALL_AND_SKINNY_GEMM_SPLITK_INSTANCES device_tall_and_skinny_gemm_splitk_f16_f16_f16_mk_kn_mn_instance.cpp)
      list(APPEND TALL_AND_SKINNY_GEMM_SPLITK_INSTANCES device_tall_and_skinny_gemm_splitk_f16_f16_f16_mk_nk_mn_instance.cpp)
    endif()


    add_instance_library(device_tall_and_skinny_gemm_splitk_instance ${TALL_AND_SKINNY_GEMM_SPLITK_INSTANCES})
    set(target 1)
 endif()
 endforeach()