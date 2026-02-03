#!/bin/bash

ProfilerPath="../../../build/bin/ckProfiler"

# Layout: NHWGC-GKYXC-NHWGK (channels last)
fwd_layout=1
bwd_weight_layout=2
bwd_data_layout=1

# FWD configs
mkdir -p fwd/profiler

# 2D
dim=2
$ProfilerPath grouped_conv_fwd 0 $fwd_layout $dim --instances > fwd/profiler/nhwgc_fp32.conf 
$ProfilerPath grouped_conv_fwd 1 $fwd_layout $dim --instances > fwd/profiler/nhwgc_fp16.conf 
$ProfilerPath grouped_conv_fwd 2 $fwd_layout $dim --instances > fwd/profiler/nhwgc_bf16.conf

# 3D
dim=3
$ProfilerPath grouped_conv_fwd 2 $fwd_layout $dim --instances > fwd/profiler/ndhwgc_bf16.conf 
$ProfilerPath grouped_conv_fwd 1 $fwd_layout $dim --instances > fwd/profiler/ndhwgc_fp16.conf 
$ProfilerPath grouped_conv_fwd 0 $fwd_layout $dim --instances > fwd/profiler/ndhwgc_fp32.conf

# BWD weight configs
mkdir -p bwd_weight/profiler

# 2D
dim=2
$ProfilerPath grouped_conv_bwd_weight 0 $bwd_weight_layout $dim --instances > bwd_weight/profiler/nhwgc_fp32.conf 
$ProfilerPath grouped_conv_bwd_weight 1 $bwd_weight_layout $dim --instances > bwd_weight/profiler/nhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_weight 5 $bwd_weight_layout $dim --instances > bwd_weight/profiler/nhwgc_bf16.conf

#3D
dim=3
$ProfilerPath grouped_conv_bwd_weight 5 $bwd_weight_layout $dim --instances > bwd_weight/profiler/ndhwgc_bf16.conf 
$ProfilerPath grouped_conv_bwd_weight 1 $bwd_weight_layout $dim --instances > bwd_weight/profiler/ndhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_weight 0 $bwd_weight_layout $dim --instances > bwd_weight/profiler/ndhwgc_fp32.conf

# BWD data configs
mkdir -p bwd_data/profiler

# 2D
dim=2
$ProfilerPath grouped_conv_bwd_data 0 $bwd_data_layout $dim --instances > bwd_data/profiler/nhwgc_fp32.conf 
$ProfilerPath grouped_conv_bwd_data 1 $bwd_data_layout $dim --instances > bwd_data/profiler/nhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_data 2 $bwd_data_layout $dim --instances > bwd_data/profiler/nhwgc_bf16.conf

#3D
dim=3
$ProfilerPath grouped_conv_bwd_data 2 $bwd_data_layout $dim --instances > bwd_data/profiler/ndhwgc_bf16.conf 
$ProfilerPath grouped_conv_bwd_data 1 $bwd_data_layout $dim --instances > bwd_data/profiler/ndhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_data 0 $bwd_data_layout $dim --instances > bwd_data/profiler/ndhwgc_fp32.conf
