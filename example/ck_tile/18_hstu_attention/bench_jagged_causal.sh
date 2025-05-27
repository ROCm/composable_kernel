#!/bin/bash

set +x
BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

dtype="bf16"
hdim=128
num_batch=32
num_head=4
target=20

add_comma()
{
   x=$*

   y=""
   for len in $x; do
       if test -z $y; then
          y="$len"
       else
          y="$y,$len"
       fi; 
   done   

   echo $y 
}

sl1024="889 602 63 923 219 400 572 297 896 115 792 313 134 605 424 582 376 975 67 50 41 582 306 580 803 680 44 117 141 688 579 958" 
sl2048="34 822 1581 415 1458 408 1897 968 176 640 1148 623 521 1734 135 874 662 1132 1907 283 679 818 1679 1723 1601 655 1774 1810 317 507 1347 1127"
sl4096="1497 2516 3179 2891 190 3572 640 3025 464 1824 712 1519 2727 2621 1135 704 1752 1665 384 1796 2567 2329 1926 2911 3787 2185 17 898 2186 3725 719 1515"
sl8192="4571 3202 270 1540 8169 3365 6055 7181 2942 4213 2717 3593 7748 4646 5502 4489 6525 2481 7397 2983 5667 1003 7926 3659 6129 6647 3758 6244 4175 2327 849 5261" 
sl16384="6956 7177 338 13755 10382 13392 10150 15592 15929 5256 6825 3804 5197 13415 14099 12418 13772 13659 5998 3715 9862 9183 11826 12964 6041 6712 12846 475 4672 7690 12280 10175"

s_sl1024=`add_comma $sl1024`
s_sl2048=`add_comma $sl2048`
s_sl4096=`add_comma $sl4096`
s_sl8192=`add_comma $sl8192`
s_sl16384=`add_comma $sl16384`

set -x 

$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl1024 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""
$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl1024 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""

$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl2048 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""
$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl2048 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""

$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl4096 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""
$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl4096 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""

$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl8192 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""
$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl8192 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""

$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl16384 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""
$EXE -v=0 -prec=$dtype -b=$num_batch -jagged=1 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$s_sl16384 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=$target -perf=1
echo -e ""

set +x

