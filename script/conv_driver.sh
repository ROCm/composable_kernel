#!/bin/bash

## GPU visibility
 export HIP_VISIBLE_DEVICES=0

 make -j conv_fwd_driver_offline
#make -j conv_bwd_driver_offline
#make -j conv_wrw_driver_offline

 DRIVER="./host/driver_offline/conv_fwd_driver_offline"
#DRIVER="./host/driver_offline/conv_bwd_driver_offline"
#DRIVER="./host/driver_offline/conv_wrw_driver_offline"

LAYOUT=$1
ALGO=$2
VERIFY=$3
INIT=$4
LOG=$5
REPEAT=$6

 DESIRED_GRID_SIZE=$7

######### layout  algo  verify  init  log  repeat  N__ K___ C___ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads  Desired_grid_size__
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  128  192 3 3  71   71     2 2       1 1      1 1       1 1    $DESIRED_GRID_SIZE
 $DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256 1024 1 7  17   17     1 1       1 1      0 3       0 3    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  14   14     1 1       1 1      1 1       1 1    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  128  128 3 3  14   14     1 1       1 1      1 1       1 1    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3   7    7     1 1       1 1      1 1       1 1    $DESIRED_GRID_SIZE
                                                                                                                      $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  512  192 3 3  35   35     2 2       1 1      0 0       0 0    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  30   30     2 2       1 1      0 0       0 0    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3  16   16     2 2       1 1      0 0       0 0    $DESIRED_GRID_SIZE
                                                                                                                      $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256 2048 1024 1 1  14   14     2 2       1 1      0 0       0 0    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256 1024 1 1  14   14     1 1       1 1      0 0       0 0    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512 2048 1 1   7    7     1 1       1 1      0 0       0 0    $DESIRED_GRID_SIZE
                                                                                                                      $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  14   14     1 1       1 1      1 1       1 1    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 1 1  14   14     1 1       1 1      0 0       0 0    $DESIRED_GRID_SIZE
                                                                                                                      $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128   32  256 3 3   1    1     1 1       1 1      1 1       1 1    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128   32  256 1 1   1    1     1 1       1 1      0 0       0 0    $DESIRED_GRID_SIZE
                                                                                                                      $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256   64 1 1   2    2     1 1       1 1      0 0       0 0    $DESIRED_GRID_SIZE
#$DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256  128 1 1   2    2     1 1       1 1      0 0       0 0    $DESIRED_GRID_SIZE

# Resnet50
######### layout  algo  verify  init  log  repeat  N__ K___ C___ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads  Desired_grid_size__
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256 2048 1024 1 1  14   14    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256 1024 1 1  14   14    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512 1024 1 1  14   14    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  128  128 3 3  28   28    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  128 1 1  28   28    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  128  128 3 3  58   58    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512 2048 1 1   7    7    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256 1024  256 1 1  14   14    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  14   14    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  30   30    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  128  256 1 1  56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  256 1 1  56   56    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256   64  256 1 1  56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3  16   16    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256 1024  512 1 1  28   28    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  128  512 1 1  28   28    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  512 1 1  28   28    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256 2048  512 1 1   7    7    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3   7    7    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256   64 1 1  56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256   64   64 1 1  56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
##DRIVER $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256   64   64 3 3  56   56    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
