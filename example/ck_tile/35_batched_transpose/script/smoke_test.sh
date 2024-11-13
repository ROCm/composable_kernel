#!/bin/sh

EXE=./build/bin/tile_example_batched_transpose

for pr in "fp32" "fp16" "int8" ; do
$EXE -pr=$pr -N=1 -C=32 -H=1 -W=32 -stride_dim0=1024 -stride_dim1=32 -stride_dim2=32 -stride_dim3=1
$EXE -pr=$pr -N=2 -C=12 -H=1 -W=32 -stride_dim0=384 -stride_dim1=32 -stride_dim2=32 -stride_dim3=1
$EXE -pr=$pr -N=3 -C=1334 -H=1 -W=37 -stride_dim0=49358 -stride_dim1=37 -stride_dim2=37 -stride_dim3=1
$EXE -pr=$pr -N=4 -C=27 -H=1 -W=32 -stride_dim0=864 -stride_dim1=32 -stride_dim2=32 -stride_dim3=1
$EXE -pr=$pr -N=5 -C=1234 -H=1 -W=12 -stride_dim0=14808 -stride_dim1=12 -stride_dim2=12 -stride_dim3=1
done
