get_filename_component(CURRENT_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

set(INSTANCE_FILES
   ${CURRENT_DIR}/km_kn_mn_add_instance.cpp
   ${CURRENT_DIR}/km_kn_mn_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/km_kn_mn_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/km_kn_mn_default_pipeline_v2_opt_instance.cpp
   ${CURRENT_DIR}/km_kn_mn_interwave_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/km_kn_mn_irregular_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/km_kn_mn_irregular_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/km_kn_mn_irregular_interwave_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_add_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_default_pipeline_v2_opt_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_interwave_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_irregular_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_irregular_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/km_nk_mn_irregular_interwave_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_add_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_default_pipeline_v2_opt_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_interwave_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_irregular_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_irregular_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/mk_kn_mn_irregular_interwave_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_add_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_default_pipeline_v2_opt_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_interwave_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_irregular_default_pipeline_v1_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_irregular_default_pipeline_v2_instance.cpp
   ${CURRENT_DIR}/mk_nk_mn_irregular_interwave_pipeline_v1_instance.cpp
)

set(GEMM_INSTANCES "${GEMM_INSTANCES};${INSTANCE_FILES}" PARENT_SCOPE)
