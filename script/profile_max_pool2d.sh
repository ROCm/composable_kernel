#!/bin/bash
## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"
OP=$1
DATATYPE=$2
VERIFY=$3
INIT=$4
LOG=$5
TIME=$6
 
########  op  datatype  verify  init  log  time  length(NCHW)
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 2 32 30 30
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 64 32 256 256
