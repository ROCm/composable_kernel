#!/bin/bash
EXAMPLE="../build/bin/example_grouped_conv_bwd_weight_dl_v4_fp16"

set -x

#               G    N   K C Y X H   W   Sy Sx Dy Dx Pad                
$EXAMPLE 1 2 1 2 480  128 1 1 5 5 28  28  1 1   1  1  2 2 2 2
$EXAMPLE 1 2 1 2 960  128 1 1 5 5 14  14  1 1   1  1  2 2 2 2
$EXAMPLE 1 2 1 2 1344 128 1 1 5 5 14  14  1 1   1  1  2 2 2 2
$EXAMPLE 1 2 1 2 2304 128 1 1 5 5 7   7   1 1   1  1  2 2 2 2
$EXAMPLE 1 2 1 2 288  128 1 1 5 5 56  56  2 2   1  1  2 2 2 2
$EXAMPLE 1 2 1 2 1344 128 1 1 5 5 14  14  2 2   1  1  2 2 2 2

$EXAMPLE 1 2 1 2 288  128 1 1 3 3 56  56  1 1   1  1  1 1 1 1
$EXAMPLE 1 2 1 2 64   128 1 1 3 3 112 112 1 1   1  1  1 1 1 1
$EXAMPLE 1 2 1 2 32   128 1 1 3 3 112 112 1 1   1  1  1 1 1 1
$EXAMPLE 1 2 1 2 960  128 1 1 3 3 14  14  1 1   1  1  1 1 1 1
$EXAMPLE 1 2 1 2 2304 128 1 1 3 3 7   7   1 1   1  1  1 1 1 1
$EXAMPLE 1 2 1 2 3840 128 1 1 3 3 7   7   1 1   1  1  1 1 1 1
$EXAMPLE 1 2 1 2 480  128 1 1 3 3 28  28  2 2   1  1  1 1 1 1
$EXAMPLE 1 2 1 2 192  128 1 1 3 3 112 112 2 2   1  1  1 1 1 1

set +x

