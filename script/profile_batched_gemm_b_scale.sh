#!/bin/bash
## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"

OP=$1
DATATYPE=$2
LAYOUT=$3
VERIFY=$4
INIT=$5
LOG=$6
TIME=$7
 
########  op  datatype  layout B_block_tile verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC  BatchCount KSplit 
$DRIVER $OP $DATATYPE $LAYOUT 1           $VERIFY $INIT $LOG $TIME  960  1024 1024       -1     -1      -1          8   1
$DRIVER $OP $DATATYPE $LAYOUT 1           $VERIFY $INIT $LOG $TIME 1920  2048 2048       -1     -1      -1          8   1
$DRIVER $OP $DATATYPE $LAYOUT 1           $VERIFY $INIT $LOG $TIME 3840  4096 4096       -1     -1      -1          4   1
$DRIVER $OP $DATATYPE $LAYOUT 1           $VERIFY $INIT $LOG $TIME 7680  8192 8192       -1     -1      -1          2   1

##todo: B_block_tile is incorect profile_batched_gemm_b_scale
