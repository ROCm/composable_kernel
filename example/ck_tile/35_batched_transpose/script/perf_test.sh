#!/bin/sh

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

EXE=./build/bin/tile_example_batched_transpose

for pr in "fp8" "fp16" "bf16"; do
for pipeline in "0" "1"; do

$EXE -pipeline=$pipeline -pr=$pr -N=1 -C=64 -H=1 -W=64 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pipeline=$pipeline -pr=$pr -N=1 -C=1024 -H=1 -W=1024 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pipeline=$pipeline -pr=$pr -N=1 -C=1024 -H=1 -W=2048 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pipeline=$pipeline -pr=$pr -N=1 -C=4096 -H=1 -W=2048 -layout_in='NCHW' -layout_out='NHWC'

done
done
