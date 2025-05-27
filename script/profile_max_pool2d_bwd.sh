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
 
########  op  datatype  verify  init  log  time  length(NCHW)          window size(YX) stride      dilation       left pad  right pad
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 2 32 30 30    --wsize 2 2 --wstride 2 2 --wdilation 1 1 --pad1 1 1 --pad2 1 1 --dmmy 28 29 30 31 32
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 64 32 256 256 --wsize 2 2 --wstride 2 2 --wdilation 1 1 --pad1 1 1 --pad2 1 1 --dmmy 28 29 30 31 32
