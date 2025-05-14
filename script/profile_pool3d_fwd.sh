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
REDUCEOP=$7
 
########  op  datatype  verify  init  log  time return_index reduce_op  length(NCDHW)
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  0            $REDUCEOP --length 2 32 30 30 30
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  0            $REDUCEOP --length 8 16 32 256 256
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  1            $REDUCEOP --length 2 32 30 30 30
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  1            $REDUCEOP --length 8 16 32 256 256