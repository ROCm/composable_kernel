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
 
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC   BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  960  1024 1024       -1     -1      -1              8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1920  2048 2048       -1     -1      -1            8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 3840  4096 4096       -1     -1      -1               4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 7680  8192 8192       -1     -1      -1              2
 
 #######  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC   BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024     1024    1024    1024           8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048     2048    2048    2048          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 4096  4096 4096     4096    4096    4096         4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 8192  8192 8192     8192    8192    8192          2
 
 #######  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC   BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024     1056    1056    1056          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048     2080    2080    2080            8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 4096  4096 4096     4128    4128    4128               4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 8192  8192 8192     8224    8224    8224             2
 
 #######  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC   BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024     1088    1088    1088         8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048     2112    2112    2112          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 4096  4096 4096     4160    4160    4160          4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 8192  8192 8192     8256    8256    8256         2
