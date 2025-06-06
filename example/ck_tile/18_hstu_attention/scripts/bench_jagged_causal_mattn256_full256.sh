#!/bin/bash

set +x
BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

dtype="bf16"

set -x

target8="10,10,14,17,16,12,14,9"

## seqlen 1024
$EXE -v=0 -prec=$dtype -b=8 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=1024 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target8 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 2048
$EXE -v=0 -prec=$dtype -b=8 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=2048 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target8 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 3072
$EXE -v=0 -prec=$dtype -b=8 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=3072 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target8 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 4096
$EXE -v=0 -prec=$dtype -b=8 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=4096 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target8 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 8192
$EXE -v=0 -prec=$dtype -b=8 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=8192 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target8 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 16384
$EXE -v=0 -prec=$dtype -b=8 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=16384 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target8 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

target16="13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10"

## seqlen 1024
$EXE -v=0 -prec=$dtype -b=16 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=1024 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target16 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 2048
$EXE -v=0 -prec=$dtype -b=16 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=2048 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target16 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 3072
$EXE -v=0 -prec=$dtype -b=16 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=3072 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target16 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 4096
$EXE -v=0 -prec=$dtype -b=16 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=4096 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target16 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 8192
$EXE -v=0 -prec=$dtype -b=16 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=8192 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target16 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 16384
$EXE -v=0 -prec=$dtype -b=16 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=16384 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target16 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

target32="13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10,11,0,4,8,2,10,20,14,11,7,4,6,9,7,14,17"

## seqlen 1024
$EXE -v=0 -prec=$dtype -b=32 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=1024 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target32 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 2048
$EXE -v=0 -prec=$dtype -b=32 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=2048 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target32 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 3072
$EXE -v=0 -prec=$dtype -b=32 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=3072 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target32 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 4096
$EXE -v=0 -prec=$dtype -b=32 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=4096 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target32 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 8192
$EXE -v=0 -prec=$dtype -b=32 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=8192 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target32 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

## seqlen 16384
$EXE -v=0 -prec=$dtype -b=32 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=16384 -causal=1 -local_len=256 -context_len=0 -minfull_len=256 -targets=$target32 -max_target=20 -perf=1 -alpha=2.0
echo -e ""

set +x

