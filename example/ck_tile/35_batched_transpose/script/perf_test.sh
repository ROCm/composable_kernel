#!/bin/sh

EXE=./build/bin/tile_example_batched_transpose

for pr in "fp8" "fp16" "bf16"; do
$EXE -pr=$pr -N=1 -C=64 -H=1 -W=64 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -N=1 -C=1024 -H=1 -W=1024 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -N=1 -C=1024 -H=1 -W=2048 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -N=1 -C=4096 -H=1 -W=2048 -layout_in='NCHW' -layout_out='NHWC'

done