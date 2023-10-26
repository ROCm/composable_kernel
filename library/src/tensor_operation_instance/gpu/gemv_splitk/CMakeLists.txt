set(GEMV_SPLITK_INSTANCES)


if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
  list(APPEND GEMV_SPLITK_INSTANCES device_gemv_splitk_f16_f16_f16_mk_kn_mn_instance.cpp)
  list(APPEND GEMV_SPLITK_INSTANCES device_gemv_splitk_f16_f16_f16_mk_nk_mn_instance.cpp)
endif()


add_instance_library(device_gemv_splitk_instance ${GEMV_SPLITK_INSTANCES})
