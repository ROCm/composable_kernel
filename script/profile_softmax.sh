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
 
########  op  datatype  verify  init  log  time   length                stride                                reduce
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 8 4 256        --stride 1024 256 1                   --reduce 2
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 2 8 128 1024   --stride 2097152 1048576 131072 1     --reduce 2
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 2 8 128 1024   --stride 2097152 1048576 131072 1     --reduce 3
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 2 32 512 4096  --stride 134217728 67108864 2097152 1 --reduce 2
$DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  --length 2 32 512 4096  --stride 134217728 67108864 2097152 1 --reduce 3

