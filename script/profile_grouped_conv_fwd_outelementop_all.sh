#!/bin/bash

## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="gfx90a-release/bin/ckProfiler"

OP=grouped_conv_fwd_outelementop
N=4


# ConvScale
#######  op    datatype  OUTELEMENTOP  layout   verify   init   log   time  Ndims    G    N    K     C   Z   Y   X   Di  Hi   Wi  Sz  Sy  Sx  Dz  Dy  Dx  Left Pz LeftPy  LeftPx  RightPz RightPy  RightPx
echo $DRIVER $OP           0             0       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           0             0       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           0             0       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           0             0       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo
echo $DRIVER $OP           1             0       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           1             0       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           1             0       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           1             0       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo
echo $DRIVER $OP           2             0       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           2             0       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           2             0       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           2             0       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo
echo $DRIVER $OP           3             0       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           3             0       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           3             0       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           3             0       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo
echo
# ConvInvScale
#######  op    datatype  OUTELEMENTOP  layout   verify   init   log   time  Ndims    G    N    K     C   Z   Y   X   Di  Hi   Wi  Sz  Sy  Sx  Dz  Dy  Dx  Left Pz LeftPy  LeftPx  RightPz RightPy  RightPx
echo $DRIVER $OP           0             1       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           0             1       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           0             1       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           0             1       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo
echo $DRIVER $OP           1             1       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           1             1       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           1             1       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           1             1       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo
echo $DRIVER $OP           2             1       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           2             1       1        1      2     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           2             1       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           2             1       1        0      1     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo
echo $DRIVER $OP           3             1       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           3             1       1        1      1     0      1      3   32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
echo $DRIVER $OP           3             1       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP           3             1       1        0      2     0      1      3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1


#$DRIVER $OP   $DATATYPE $OUTELEMENTOP $LAYOUT  $VERIFY  $INIT  $LOG  $TIME     3   32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
