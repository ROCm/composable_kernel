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
 
########  op  datatype  compute_datatype  num_dim layout   verify   init   log       time     alpha    M0  M1  N0  N1  K0  K1
 $DRIVER $OP  $DATATYPE   $DATATYPE        2      $LAYOUT  $VERIFY  $INIT    $LOG     $TIME    1.0     128 128 128 128 128 128

