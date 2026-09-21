#!/bin/bash

# Get flag --update-test-configs-only to skip running the CK profiler and update tests based on the existing profiler configs
UPDATE_TEST_CONFIGS_ONLY=false
for arg in "$@"; do
    if [ "$arg" == "--update-test-configs-only" ]; then
        UPDATE_TEST_CONFIGS_ONLY=true
    fi
done

if [ "$UPDATE_TEST_CONFIGS_ONLY" = false ]; then

  ProfilerPath="../../../build/bin/ckProfiler"

  # Layout: NHWGC-GKYXC-NHWGK (channels last)
  fwd_layout=1
  bwd_weight_layout=2
  bwd_data_layout=1

  # FWD configs
  mkdir -p forward/profiler

  # 2D
  dim=2
  $ProfilerPath grouped_conv_fwd 0 $fwd_layout $dim --instances > forward/profiler/nhwgc_fp32.conf 
  $ProfilerPath grouped_conv_fwd 1 $fwd_layout $dim --instances > forward/profiler/nhwgc_fp16.conf 
  $ProfilerPath grouped_conv_fwd 2 $fwd_layout $dim --instances > forward/profiler/nhwgc_bf16.conf

  # 3D
  dim=3
  $ProfilerPath grouped_conv_fwd 2 $fwd_layout $dim --instances > forward/profiler/ndhwgc_bf16.conf 
  $ProfilerPath grouped_conv_fwd 1 $fwd_layout $dim --instances > forward/profiler/ndhwgc_fp16.conf 
  $ProfilerPath grouped_conv_fwd 0 $fwd_layout $dim --instances > forward/profiler/ndhwgc_fp32.conf

  # BWD weight configs
  mkdir -p backward_weight/profiler

  # 2D
  dim=2
  $ProfilerPath grouped_conv_bwd_weight 0 $bwd_weight_layout $dim --instances > backward_weight/profiler/nhwgc_fp32.conf 
  $ProfilerPath grouped_conv_bwd_weight 1 $bwd_weight_layout $dim --instances > backward_weight/profiler/nhwgc_fp16.conf 
  $ProfilerPath grouped_conv_bwd_weight 5 $bwd_weight_layout $dim --instances > backward_weight/profiler/nhwgc_bf16.conf

  #3D
  dim=3
  $ProfilerPath grouped_conv_bwd_weight 5 $bwd_weight_layout $dim --instances > backward_weight/profiler/ndhwgc_bf16.conf 
  $ProfilerPath grouped_conv_bwd_weight 1 $bwd_weight_layout $dim --instances > backward_weight/profiler/ndhwgc_fp16.conf 
  $ProfilerPath grouped_conv_bwd_weight 0 $bwd_weight_layout $dim --instances > backward_weight/profiler/ndhwgc_fp32.conf

  # BWD data configs
  mkdir -p backward_data/profiler

  # 2D
  dim=2
  $ProfilerPath grouped_conv_bwd_data 0 $bwd_data_layout $dim --instances > backward_data/profiler/nhwgc_fp32.conf 
  $ProfilerPath grouped_conv_bwd_data 1 $bwd_data_layout $dim --instances > backward_data/profiler/nhwgc_fp16.conf 
  $ProfilerPath grouped_conv_bwd_data 2 $bwd_data_layout $dim --instances > backward_data/profiler/nhwgc_bf16.conf

  #3D
  dim=3
  $ProfilerPath grouped_conv_bwd_data 2 $bwd_data_layout $dim --instances > backward_data/profiler/ndhwgc_bf16.conf 
  $ProfilerPath grouped_conv_bwd_data 1 $bwd_data_layout $dim --instances > backward_data/profiler/ndhwgc_fp16.conf 
  $ProfilerPath grouped_conv_bwd_data 0 $bwd_data_layout $dim --instances > backward_data/profiler/ndhwgc_fp32.conf

fi

mkdir -p forward/tests
mkdir -p backward_weight/tests
mkdir -p backward_data/tests

# For FWD, generate new test configs by taking 20% of the profiler configs for each data type and layout
for layout in nhwgc ndhwgc; do
    for dtype in fp32 fp16 bf16; do
        profiler_config="forward/profiler/${layout}_${dtype}.conf"
        test_config="forward/tests/${layout}_${dtype}.conf"
        awk 'NR % 5 == 0' $profiler_config > $test_config # 20% of lines in the profiler configs
    done
done

# For BWD weight, generate new test configs by taking 20% of the profiler configs for each data type and layout
for layout in nhwgc ndhwgc; do
    for dtype in fp32 fp16 bf16; do
        profiler_config="backward_weight/profiler/${layout}_${dtype}.conf"
        test_config="backward_weight/tests/${layout}_${dtype}.conf"
        awk 'NR % 5 == 0' $profiler_config > $test_config # 20% of lines in the profiler configs
    done
done

# For BWD data, generate new test configs by taking 20% of the profiler configs for each data type and layout
for layout in nhwgc ndhwgc; do
    for dtype in fp32 fp16 bf16; do
        profiler_config="backward_data/profiler/${layout}_${dtype}.conf"
        test_config="backward_data/tests/${layout}_${dtype}.conf"
        awk 'NR % 5 == 0' $profiler_config > $test_config # 20% of lines in the profiler configs
    done
done