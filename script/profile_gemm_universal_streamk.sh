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
STRATEGY=$8


# 120 CU
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC  STRATEGY_ GridSize
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  960  1024 1024       -1     -1      -1   $STRATEGY -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  960  2048 2048       -1     -1      -1   $STRATEGY -1
 
# 104 CU
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC STRATEGY_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  832  1024 1024       -1     -1      -1  $STRATEGY -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  832  2048 2048       -1     -1      -1  $STRATEGY -1
 
# 110 CU
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC STRATEGY_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1280  1408 1024       -1     -1      -1  $STRATEGY -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1280  2816 2048       -1     -1      -1  $STRATEGY -1

# testing different strides
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC STRATEGY_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024	    1024   1024    1024  $STRATEGY -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048	    2048   2048    2048  $STRATEGY -1
 
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024	    1056   1056    1056  $STRATEGY -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048	    2080   2080    2080  $STRATEGY -1
 
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024	    1088   1088    1088  $STRATEGY -1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048	    2112   2112    2112  $STRATEGY -1

