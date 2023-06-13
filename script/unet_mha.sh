#!/bin/bash
while getopts e: flag
do
    case "${flag}" in
        e) executable=${OPTARG};;
    esac
done
echo "CK-NAVI31 Performance Test: MHA for AITemplate"

VERIFICATION=0
INITIALIZE=1
TIMING=1

ALL_TEST_CASE=0
SELF_ATTENTION=1
CROSS_ATTENTION=0
CAUSAL_MASK=0
# self attention with causal mask
if  [ $ALL_TEST_CASE -eq 1 ] || { [ $SELF_ATTENTION -eq 1 ] && [ $CAUSAL_MASK -eq 1 ]; }; then
    echo "Test launched: self attention with causal mask"
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING 4096 4096  40  40 2 8 0.158113881945610 1 1
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING 1024 1024  80  80 2 8 0.111803397536277 1 1
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING  256  256 160 160 2 8 0.079056940972805 1 1
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING   64   64 160 160 2 8 0.079056940972805 1 1
fi

# cross attention with causal mask
if [ $ALL_TEST_CASE -eq 1 ] || { [ $CROSS_ATTENTION -eq 1 ] && [ $CAUSAL_MASK -eq 1 ]; }; then
    echo "Test launched: cross attention with causal mask"
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING 4096   64  40  40 2 8 0.158113881945610 1 1
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING 1024   64  80  80 2 8 0.111803397536277 1 1
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING  256   64 160 160 2 8 0.079056940972805 1 1
    ./bin/example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16 $VERIFICATION 1 $TIMING   64   64 160 160 2 8 0.079056940972805 1 1
fi

# self attention without causal mask
if [ $ALL_TEST_CASE -eq 1 ] || { [ $SELF_ATTENTION -eq 1 ] && [ $CAUSAL_MASK -eq 0 ]; }; then
    echo "Test launched: self attention without causal mask"
    $executable $VERIFICATION $INITIALIZE $TIMING 4096 4096  64  64 2  5 0.125 1 1
    $executable $VERIFICATION $INITIALIZE $TIMING 1024 1024  64  64 2 10 0.125 1 1
    $executable $VERIFICATION $INITIALIZE $TIMING  256  256  64  64 2 20 0.125 1 1
    $executable $VERIFICATION $INITIALIZE $TIMING   64   64  64  64 2 20 0.125 1 1
fi

# cross attention without causal mask
if [ $ALL_TEST_CASE -eq 1 ] || { [ $CROSS_ATTENTION -eq 1 ] && [ $CAUSAL_MASK -eq 0 ]; }; then
    echo "Test launched: cross attention without causal mask"
    $executable $VERIFICATION 1 $TIMING 4096   64  40  40 2 8 0.158113881945610 1 1
    $executable $VERIFICATION 1 $TIMING 1024   64  80  80 2 8 0.111803397536277 1 1
    $executable $VERIFICATION 1 $TIMING  256   64 160 160 2 8 0.079056940972805 1 1
    $executable $VERIFICATION 1 $TIMING   64   64 160 160 2 8 0.079056940972805 1 1
fi