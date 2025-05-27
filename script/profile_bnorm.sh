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
 
########  op  datatype UseSavedMean  init   time   inOutLengths(nhwc)   reduceDims  verify
 $DRIVER $OP $DATATYPE $USE          $INIT  $TIME  -D 64,64,280,82       -R 1,2,3        $VERIFY