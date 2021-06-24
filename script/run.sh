#!/bin/bash


 export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

 export OLC_DEBUG_HIP_VERBOSE=1
 export OLC_DEBUG_HIP_DUMP=1
 export OLC_DEBUG_SAVE_TEMP_DIR=1

#make -j conv_driver
#make -j conv_driver_v2
 make -j conv_driver_v2_olc

 rm -rf /root/_hip_binary_kernels_/
 rm -rf /tmp/olCompile*

LAYOUT=$1
ALGO=$2
VERIFY=$3
INIT=$4
LOG=$5
REPEAT=$6

######################  layout  algo  verify  init  log  repeat  N__ K__ C__ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128 128 192 3 3  71   71     2 2       1 1      1 1       1 1
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128 384 192 3 3  35   35     2 2       1 1      0 0       0 0
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128 128 128 1 7  17   17     1 1       1 1      0 3       0 3
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128 256 256 3 3  14   14     1 1       1 1      1 1       1 1

#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128 128 192 3 3  71   71     2 2       1 1      1 1       1 1
 ./conv_driver_v2_olc  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128 128 192 3 3  71   71     2 2       1 1      1 1       1 1
