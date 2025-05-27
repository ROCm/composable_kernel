#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/tile_example_img2col 


N=256
######    Dim G N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads
$EXE 1 2 1 2  1 $N  256  64   1 1   64   64    1 1       1 1      0 0       0 0
$EXE 1 2 1 2  1 $N  512  64   1 1   64   64    1 1       1 1      0 0       0 0
$EXE 1 2 1 2  1 $N  128  64   3 3   64   64    1 1       1 1      1 1       1 1



