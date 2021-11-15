#!/bin/bash

## GPU visibility
 export HIP_VISIBLE_DEVICES=1

 make -j gemm_xdl

 DRIVER="./example/gemm_xdl"

VERIFY=$1
INIT=$2
LOG=$3
REPEAT=$4

######### verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC
#$DRIVER $VERIFY $INIT $LOG $REPEAT   960 1024 1024     1024    1024    1024
#$DRIVER $VERIFY $INIT $LOG $REPEAT  1024 1024 1024     1024    1024    1024
#$DRIVER $VERIFY $INIT $LOG $REPEAT  1920 2048 2048     2048    2048    2048
 $DRIVER $VERIFY $INIT $LOG $REPEAT  3840 4096 4096     4096    4096    4096
#$DRIVER $VERIFY $INIT $LOG $REPEAT  7680 8192 8192     8192    8192    8192
