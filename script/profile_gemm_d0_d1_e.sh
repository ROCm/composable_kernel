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
 
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideD0 StrideD1 StrideE
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1280  1408 1024       -1     -1      -1   -1       -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1280  2816 2048       -1     -1      -1   -1       -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2560  1408 2048       -1     -1      -1   -1       -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2560  2816 2048       -1     -1      -1   -1       -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 5120  5632 4096       -1     -1      -1   -1       -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 7040  8192 8192       -1     -1      -1   -1       -1