# ONLY XDL_KERNELS
set(GROUPED_CONV2D_BWD_DATA_CLAMP)
include(ShardInstantiation)

   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_instances
  TEMPLATE_FILE xdl/device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV2D_BWD_DATA_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_16x16_instances
  TEMPLATE_FILE xdl/device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_16x16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV2D_BWD_DATA_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_instances
  TEMPLATE_FILE xdl/device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV2D_BWD_DATA_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_16x16_instances
  TEMPLATE_FILE xdl/device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_16x16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV2D_BWD_DATA_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_instances
  TEMPLATE_FILE xdl/device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_instance.in
  NUM_SHARDS 17
  SRC_LIST GROUPED_CONV2D_BWD_DATA_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_16x16_instances
  TEMPLATE_FILE xdl/device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_16x16_instance.in
  NUM_SHARDS 6
  SRC_LIST GROUPED_CONV2D_BWD_DATA_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

add_instance_library(device_grouped_conv2d_bwd_data_clamp_instance ${GROUPED_CONV2D_BWD_DATA_CLAMP})
