#!/bin/bash

## GPU visibility
 export HIP_VISIBLE_DEVICES=0

 make -j $1

DRIVER=example/$1
VERIFY=$2
INIT=$3
REPEAT=$4

# test
########  verify  init  repeat  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads  Desired_grid_size__
 $DRIVER $VERIFY $INIT $REPEAT  128  256  192 3 3   71   71     2 2       1 1      1 1       1 1   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT  128  256   64 1 1    1    1     1 1       1 1      0 0       0 0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT  256   64    3 7 7  230  230     2 2       1 1      0 0       0 0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT  128  512  512 3 3    7    7     1 1       1 1      1 1       1 1   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT  256   64    3 7 7  224 224    2   2     1   1    3   3     3   3

 N=$5

# Resnet50
########  verify  init  repeat  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads  Desired_grid_size__
#$DRIVER $VERIFY $INIT $REPEAT   $N 2048 1024 1 1   14   14    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  256 1024 1 1   14   14    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  512 1024 1 1   14   14    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  128  128 3 3   28   28    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  512  128 1 1   28   28    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  128  128 3 3   58   58    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  512 2048 1 1    7    7    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N 1024  256 1 1   14   14    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  256  256 3 3   14   14    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  256  256 3 3   30   30    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  128  256 1 1   56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  512  256 1 1   56   56    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N   64  256 1 1   56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  512  512 3 3   16   16    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N 1024  512 1 1   28   28    2  2      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  128  512 1 1   28   28    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  256  512 1 1   28   28    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N 2048  512 1 1    7    7    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  512  512 3 3    7    7    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N  256   64 1 1   56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N   64   64 1 1   56   56    1  1      1  1     0  0      0  0   $DESIRED_GRID_SIZE
#$DRIVER $VERIFY $INIT $REPEAT   $N   64   64 3 3   56   56    1  1      1  1     1  1      1  1   $DESIRED_GRID_SIZE
