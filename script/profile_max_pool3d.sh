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
 
########  op  datatype  verify  init  log  time  length(NCDHW)              window size(YX)     stride      dilation         left pad      right pad
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 2 32 30 30 30     --wsize 2 2 2 --wstride 2 2 2 --wdilation 1 1 1 --pad1 1 1 1 --pad2 1 1 1
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 8 16 32 256 256   --wsize 2 2 2 --wstride 2 2 2 --wdilation 1 1 1 --pad1 1 1 1 --pad2 1 1 1
