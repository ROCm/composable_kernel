#!/bin/sh

EXE=./build/bin/tile_example_elementwise

$EXE -pr_i=fp16 -pr_o=fp32 -n=2043904
$EXE -pr_i=fp16 -pr_o=fp32 -n=992256
$EXE -pr_i=fp16 -pr_o=fp32 -n=846304
$EXE -pr_i=fp16 -pr_o=fp32 -n=434176
$EXE -pr_i=fp16 -pr_o=fp32 -n=159424
$EXE -pr_i=fp16 -pr_o=fp32 -n=98304
$EXE -pr_i=fp16 -pr_o=fp32 -n=73728
$EXE -pr_i=fp16 -pr_o=fp32 -n=17408
$EXE -pr_i=fp16 -pr_o=fp32 -n=512
$EXE -pr_i=fp16 -pr_o=fp32 -n=256

echo "-------------------------------------"

$EXE -pr_i=fp32 -pr_o=fp16 -n=2043904
$EXE -pr_i=fp32 -pr_o=fp16 -n=992256
$EXE -pr_i=fp32 -pr_o=fp16 -n=846304
$EXE -pr_i=fp32 -pr_o=fp16 -n=434176
$EXE -pr_i=fp32 -pr_o=fp16 -n=159424
$EXE -pr_i=fp32 -pr_o=fp16 -n=98304
$EXE -pr_i=fp32 -pr_o=fp16 -n=73728
$EXE -pr_i=fp32 -pr_o=fp16 -n=17408
$EXE -pr_i=fp32 -pr_o=fp16 -n=512
$EXE -pr_i=fp32 -pr_o=fp16 -n=256
