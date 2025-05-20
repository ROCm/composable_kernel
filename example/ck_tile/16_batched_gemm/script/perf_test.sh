#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/tile_example_batched_gemm 

$EXE -v=0 -prec=fp16   -m=960   -n=1024 -k=1024  -batch_count=8
$EXE -v=0 -prec=fp16   -m=1920  -n=2048 -k=2048  -batch_count=8
$EXE -v=0 -prec=fp16   -m=3840  -n=4096 -k=4096  -batch_count=4
$EXE -v=0 -prec=fp16   -m=7680  -n=8192 -k=8192  -batch_count=2
$EXE -v=0 -prec=fp16   -m=1024  -n=1024 -k=1024  -stride_a=1056    -stride_b=1056    -stride_c=1056   -batch_count=8
$EXE -v=0 -prec=fp16   -m=2048  -n=2048 -k=2048  -stride_a=2080    -stride_a=2080    -stride_c=2080    -batch_count=8
$EXE -v=0 -prec=fp16   -m=4096  -n=4096 -k=4096  -stride_a=4128    -stride_a=4128    -stride_c=4128    -batch_count=4
$EXE -v=0 -prec=fp16   -m=8192  -n=8192 -k=8192  -stride_a=8224    -stride_a=8224    -stride_c=8224    -batch_count=2

#$EXE -v=0 -prec=fp8   -m=960   -n=1024 -k=1024  -batch_count=8
#$EXE -v=0 -prec=fp8   -m=1920  -n=2048 -k=2048  -batch_count=8
#$EXE -v=0 -prec=fp8   -m=3840  -n=4096 -k=4096  -batch_count=4
#$EXE -v=0 -prec=fp8   -m=7680  -n=8192 -k=8192  -batch_count=2
#$EXE -v=0 -prec=fp8   -m=1024  -n=1024 -k=1024  -stride_a=1056    -stride_b=1056    -stride_c=1056   -batch_count=8
#$EXE -v=0 -prec=fp8   -m=2048  -n=2048 -k=2048  -stride_a=2080    -stride_a=2080    -stride_c=2080    -batch_count=8
#$EXE -v=0 -prec=fp8   -m=4096  -n=4096 -k=4096  -stride_a=4128    -stride_a=4128    -stride_c=4128    -batch_count=4
#$EXE -v=0 -prec=fp8   -m=8192  -n=8192 -k=8192  -stride_a=8224    -stride_a=8224    -stride_c=8224    -batch_count=2


