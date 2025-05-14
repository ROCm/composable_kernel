#!/bin/bash
## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"
OP=$1
DATATYPE=$2
VERIFY="-v $3"
INIT=$4
TIME=$5
USE=$6
 
########  op  datatype UseSavedMean  verify  init  log  time   inOutLengths(nhwc)   reduceDims  verify
 $DRIVER $OP $DATATYPE $USE             $VERIFY $INIT $LOG $TIME  -D 64,64,280,82       -R 0        $VERIFY