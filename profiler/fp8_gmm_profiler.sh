EXE="$(find . -name ckProfiler -type f | head -n 1)"
op="gemm_multiply_multiply"

loopFunc() {
    N=$1
    K=$2
    $EXE $op 7 1 0 2 0 1 1 $N $K -1 -1 0 0 -1 1 40 500 4096
    for ((M=32; M<=20480;M*=2))
    do
        # echo "M = $M, N = $N, K = $K"
        $EXE $op 7 1 0 2 0 1 $M $N $K -1 -1 0 0 -1 1 40 500 4096
    done
    $EXE $op 7 1 0 2 0 1 20480 $N $K -1 -1 0 0 -1 1 40 500 4096
}


N=4608
K=3584
loopFunc $N $K

N=3584
K=3584
loopFunc $N $K

N=3584
K=20480
loopFunc $N $K

N=40960
K=3584
loopFunc $N $K
