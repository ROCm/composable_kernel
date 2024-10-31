#!/bin/sh

EXE=./build/bin/tile_example_batched_transpose

# for pr_i in "fp32" "fp16" "bf16" "int8" ; do
for pr in "fp16" ; do
$EXE -pr_i=$pr -N=1 -C=32 -H=1 -W=32 -stride_dim0=1024 -stride_dim1=32 -stride_dim2=32 -stride_dim3=1
$EXE -pr_i=$pr -N=1 -C=12 -H=1 -W=32 -stride_dim0=384 -stride_dim1=32 -stride_dim2=32 -stride_dim3=1
$EXE -pr_i=$pr -N=1 -C=134 -H=1 -W=32 -stride_dim0=4224 -stride_dim1=32 -stride_dim2=32 -stride_dim3=1
$EXE -pr_i=$pr -N=1 -C=27 -H=1 -W=32 -stride_dim0=864 -stride_dim1=32 -stride_dim2=32 -stride_dim3=1
$EXE -pr_i=$pr -N=1 -C=1234 -H=1 -W=12 -stride_dim0=14808 -stride_dim1=12 -stride_dim2=12 -stride_dim3=1
done
