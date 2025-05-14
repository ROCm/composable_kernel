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
 
########  op  datatype  verify  init  log  time   length
$DRIVER $OP $DATATYPE  $VERIFY $INIT $LOG $TIME  --length 1 16 16 32 40
$DRIVER $OP $DATATYPE  $VERIFY $INIT $LOG $TIME  --length 2 64 64 64 64
$DRIVER $OP $DATATYPE  $VERIFY $INIT $LOG $TIME  --length 2 64 64 128 128