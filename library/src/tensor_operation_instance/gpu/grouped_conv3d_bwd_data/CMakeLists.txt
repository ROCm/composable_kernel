# ONLY XDL_AND_WMMA_KERNELS
set(GROUPED_CONV3D_BWD_DATA)

include(ShardInstantiation)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.in
  NUM_SHARDS 15
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_bf16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_bf16_instance.in
  NUM_SHARDS 15
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f32_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.in
  NUM_SHARDS 12
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16_16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16_16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16_16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16_16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16_16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16_16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_f16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_f16_instance.in
  NUM_SHARDS 1
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_bf16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_bf16_instance.in
  NUM_SHARDS 1
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_f32_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_f32_instance.in
  NUM_SHARDS 1
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_16_16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_16_16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_16_16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_16_16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_16_16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_16_16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_vec_transpose_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_vec_transpose_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_vec_transpose_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_vec_transpose_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_vec_transpose_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_vec_transpose_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_optimized_loads_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_optimized_loads_instance.in
  NUM_SHARDS 18
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_optimized_loads_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_optimized_loads_instance.in
  NUM_SHARDS 18
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_optimized_loads_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_optimized_loads_instance.in
  NUM_SHARDS 12
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_instance.in
  NUM_SHARDS 18
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instance.in
  NUM_SHARDS 18
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_instance.in
  NUM_SHARDS 15
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instance.in
  NUM_SHARDS 15
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instance.in
  NUM_SHARDS 18
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instance.in
  NUM_SHARDS 18
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instance.in
  NUM_SHARDS 15
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instances
  TEMPLATE_FILE wmma/device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instance.in
  NUM_SHARDS 15
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/wmma
)

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "bf8" AND DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
      
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_input_f16_comp_bf8_f8_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_input_f16_comp_bf8_f8_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV3D_BWD_DATA
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
endif()

add_instance_library(device_grouped_conv3d_bwd_data_instance ${GROUPED_CONV3D_BWD_DATA})
