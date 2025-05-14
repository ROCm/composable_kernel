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
 
N=$8
########  op  datatype  layout  verify  init  log  time  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME   $N  256 1024 1 1   14   14     1 1       1 1      0 0       0 0
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME   $N  512 1024 1 1   14   14     1 1       1 1      0 0       0 0
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  128 3 3   28   28     1 1       1 1      1 1       1 1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  128 1 1   28   28     1 1       1 1      0 0       0 0
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  128 3 3   56   56     2 2       1 1      1 1       1 1
