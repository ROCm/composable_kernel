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
REPEAT=$7
 
OP=$1
DATATYPE=$2
LAYOUT=$3
VERIFY=$4
INIT=$5
LOG=$6
REPEAT=$7
 
########  op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC BatchStrideA BatchStrideB BatchStrideC BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT  960  1024 1024       -1     -1      -1            -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 1920  2048 2048       -1     -1      -1            -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 3840  4096 4096       -1     -1      -1            -1           -1           -1          4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 7680  8192 8192       -1     -1      -1            -1           -1           -1          2
 
 #######  op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC BatchStrideA BatchStrideB BatchStrideC BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 1024  1024 1024     1024    1024    1024           -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 2048  2048 2048     2048    2048    2048           -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 4096  4096 4096     4096    4096    4096           -1           -1           -1          4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 8192  8192 8192     8192    8192    8192           -1           -1           -1          2
 
 #######  op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC BatchStrideA BatchStrideB BatchStrideC BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 1024  1024 1024     1056    1056    1056           -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 2048  2048 2048     2080    2080    2080           -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 4096  4096 4096     4128    4128    4128           -1           -1           -1          4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 8192  8192 8192     8224    8224    8224           -1           -1           -1          2
 
 #######  op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC BatchStrideA BatchStrideB BatchStrideC BatchCount
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 1024  1024 1024     1088    1088    1088           -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 2048  2048 2048     2112    2112    2112           -1           -1           -1          8
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 4096  4096 4096     4160    4160    4160           -1           -1           -1          4
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $REPEAT 8192  8192 8192     8256    8256    8256           -1           -1           -1          2