#!/bin/bash

PRECISION=--half

if test -n $PRECISION && test "$PRECISION" = "--half"; then 
   CTYPE="-C 1"
else
   CTYPE=""
fi

WTYPE=

if [ $# -ge 1 ] ; then
    NREPEAT=$1
else
    NREPEAT=1
fi

Operation=7

for op in $Operation; do
    set -x
    ./bin/ckProfiler reduce $PRECISION -D 256,14,14,1024 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT 
    ./bin/ckProfiler reduce $PRECISION -D 256,28,28,128 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,58,58,128 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,7,7,2048 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,14,14,256 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,30,30,256 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,56,56,256 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,16,16,512 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,28,28,512 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,7,7,512 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,56,56,64 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 256,230,230,3 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,14,14,1024 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,28,28,128 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,58,58,128 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,7,7,2048 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,14,14,256 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,30,30,256 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,56,56,256 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,16,16,512 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,28,28,512 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,7,7,512 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    ./bin/ckProfiler reduce $PRECISION -D 128,56,56,64 -R 0,1,2 -O $op $CTYPE $WTYPE -v 1 1 $NREPEAT
    set +x
done 
