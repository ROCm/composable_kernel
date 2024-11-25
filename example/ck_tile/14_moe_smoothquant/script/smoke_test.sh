#!/bin/sh
EXE=build/bin/tile_example_moe_smoothquant

for pr_i in "fp16" "bf16" ; do
$EXE -prec=$pr_i -t=99  -h=13
$EXE -prec=$pr_i -t=17  -h=16
$EXE -prec=$pr_i -t=1   -h=100
$EXE -prec=$pr_i -t=4   -h=128
$EXE -prec=$pr_i -t=80  -h=127
$EXE -prec=$pr_i -t=22  -h=255 -stride=256
$EXE -prec=$pr_i -t=7   -h=599
$EXE -prec=$pr_i -t=19  -h=512
$EXE -prec=$pr_i -t=33  -h=313 -stride=1000
$EXE -prec=$pr_i -t=11  -h=510
$EXE -prec=$pr_i -t=171 -h=676 -stride=818
$EXE -prec=$pr_i -t=91  -h=636
$EXE -prec=$pr_i -t=12  -h=768 -stride=800
$EXE -prec=$pr_i -t=100 -h=766 -stride=812
$EXE -prec=$pr_i -t=31  -h=1024
$EXE -prec=$pr_i -t=64  -h=1000 -stride=1004
$EXE -prec=$pr_i -t=8   -h=1501
$EXE -prec=$pr_i -t=3   -h=1826
$EXE -prec=$pr_i -t=5   -h=2040
$EXE -prec=$pr_i -t=7   -h=2734
$EXE -prec=$pr_i -t=1   -h=3182
$EXE -prec=$pr_i -t=9   -h=4096
$EXE -prec=$pr_i -t=3   -h=8192
$EXE -prec=$pr_i -t=1   -h=10547
$EXE -prec=$pr_i -t=3   -h=17134
done
