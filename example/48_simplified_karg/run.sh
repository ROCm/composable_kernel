#!/bin/sh
EXE=./bin/simplified_karg_gemm

# below split works on gfx90a
#             M    N    K       split
$EXE 0 1 1    1 1104 4608 0 0 0 32
$EXE 0 1 1   16 1104 4608 0 0 0 32
$EXE 0 1 1 1335 1104 4608 0 0 0 1
$EXE 0 1 1    1 4608  320 0 0 0 8
$EXE 0 1 1   16 4608  320 0 0 0 8
$EXE 0 1 1 1335 4608  320 0 0 0 1
$EXE 0 1 1    1   16 4608 0 0 0 32
$EXE 0 1 1   16   16 4608 0 0 0 32
$EXE 0 1 1 1335   16 4608 0 0 0 32
$EXE 0 1 1    1  768 4608 0 0 0 32
$EXE 0 1 1   16  768 4608 0 0 0 32
$EXE 0 1 1 1335  768 4608 0 0 0 2
$EXE 0 1 1    1 4608  768 0 0 0 24
$EXE 0 1 1   16 4608  768 0 0 0 32
$EXE 0 1 1 1335 4608  768 0 0 0 1
