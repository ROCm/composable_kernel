# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(ShardInstantiation)

# NGCDHW layout instances for grouped_conv3d_fwd
set(GROUPED_CONV3D_FWD_NGCDHW
   xdl/comp/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_comp_2x_instance.cpp
   xdl/comp/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_comp_part2_instance.cpp
   xdl/comp/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_comp_2x_instance.cpp
   xdl/comp/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_comp_part2_instance.cpp
   xdl/comp/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_comp_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_16x16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_16x16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_16x16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_instance.cpp
   xdl/merged_groups/device_grouped_conv3d_fwd_xdl_merged_groups_ngcdhw_gkczyx_ngkdhw_bf16_instance.cpp
   xdl/merged_groups/device_grouped_conv3d_fwd_xdl_merged_groups_ngcdhw_gkczyx_ngkdhw_f16_instance.cpp
   xdl/merged_groups/device_grouped_conv3d_fwd_xdl_merged_groups_ngcdhw_gkczyx_ngkdhw_f32_instance.cpp
)


# Generate sharded instances from templates
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_comp_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_comp_instance.in
  NUM_SHARDS 12
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_comp_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_comp_instance.in
  NUM_SHARDS 12
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instance.in
  NUM_SHARDS 8
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_instance.in
  NUM_SHARDS 8
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_mem_inter_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_mem_inter_instance.in
  NUM_SHARDS 10
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_mem_intra_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_mem_intra_instance.in
  NUM_SHARDS 10
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_mem_inter_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_mem_inter_instance.in
  NUM_SHARDS 10
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_mem_intra_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_mem_intra_instance.in
  NUM_SHARDS 10
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_mem_inter_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_mem_inter_instance.in
  NUM_SHARDS 10
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_mem_intra_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_mem_intra_instance.in
  NUM_SHARDS 10
  SRC_LIST GROUPED_CONV3D_FWD_NGCDHW
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

add_instance_library(device_grouped_conv3d_fwd_ngcdhw_instance ${GROUPED_CONV3D_FWD_NGCDHW})
