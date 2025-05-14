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
 
########  op  datatype  verify  init  log  time   N  C  D  H   W
 $DRIVER $OP $DATATYPE  $VERIFY $INIT $LOG $TIME  4  8  8  512 512
