#!/bin/bash
# TODO: run this script from CK root or build directory
#EXE="/code/composable_kernel/build/bin/tile_example_fmha_fwd"
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n 1)"
KNAME=1


COMMON_ARGS='-v=1 -warmup=0 -repeat=1'


$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=16 -d_v=32 -s=8 -s_k=8 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=c -num_splits=1 -page_block_size=0 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=t:2,0,2

# window_size[2,0], sink_size = 2

#    x=1/y=3                 
#    1 * * * * * * *           1 * * * * * * *  
#    1 1 * * * * * *           1 1 * * * * * *
#    1 1 1 * * * * *   ---->   1 1 1 * * * * * 
#    * 1 1 1 * * * *           1 1 1 1 * * * * 
#    * * 1 1 1 * * *           1 1 1 1 1 * * * 
#    * * * 1 1 1 * *           1 1 * 1 1 1 * * 
#    * * * * 1 1 1 *           1 1 * * 1 1 1 *
#    * * * * * 1 1 1           1 1 * * * 1 1 1
#    l=2/r=0(tl)               l=2/r=0/s=2(tl)

$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=16 -d_v=32 -s=5 -s_k=8 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=c -num_splits=1 -page_block_size=0 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=t:0,3,2 #-mask=b:3,0,2

#    x=4/y=1                   
#    1 1 1 1 * * * *           1 1 1 1 * * * * 
#    * 1 1 1 1 * * *           1 1 1 1 1 * * *
#    * * 1 1 1 1 * *   ---->   1 1 1 1 1 1 * *
#    * * * 1 1 1 1 *           1 1 * 1 1 1 1 *
#    * * * * 1 1 1 1           1 1 * * 1 1 1 1 
#    l=0/r=3(tl)               l=0/r=3/s=2(tl)
#    l=3/r=0(br)               l=3/r=0/s=2(br)  


$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=16 -d_v=32 -s=5 -s_k=8 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=c -num_splits=1 -page_block_size=0 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=b:1,0,2 

#    x=4/y=-1          
#    * * 1 1 * * * *           1 1 1 1 * * * * 
#    * * * 1 1 * * *           1 1 * 1 1 * * *
#    * * * * 1 1 * *   ---->   1 1 * * 1 1 * *
#    * * * * * 1 1 *           1 1 * * * 1 1 *
#    * * * * * * 1 1           1 1 * * * * 1 1 
#    l=1/r=0(br)               l=1/r=0/s=2(br)


$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=16 -d_v=32 -s=8 -s_k=6 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=c -num_splits=1 -page_block_size=0 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=b:2,0,2

#    x=-1/y=5 
     
#    * * * * * *               * * * * * *
#    * * * * * *               * * * * * *
#    1 * * * * *               1 * * * * *
#    1 1 * * * *               1 1 * * * *
#    1 1 1 * * *       ---->   1 1 1 * * *
#    * 1 1 1 * *               1 1 1 1 * *
#    * * 1 1 1 *               1 1 1 1 1 *  
#    * * * 1 1 1               1 1 * 1 1 1
#    l=2/r=0(br)               l=2/r=0/s=2(br)


$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=16 -d_v=32 -s=8 -s_k=5 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=c -num_splits=1 -page_block_size=0 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=b:-1,1,2
#      x=-1/y=8
#    * * * * *               * * * * *    
#    * * * * *               * * * * * 
#    1 * * * *      ---->    1 * * * * 
#    1 1 * * *               1 1 * * * 
#    1 1 1 * *               1 1 1 * * 
#    1 1 1 1 *               1 1 1 1 * 
#    1 1 1 1 1               1 1 1 1 1 
#    1 1 1 1 1               1 1 1 1 1 
#    l=2/r=0(br)             l=2/r=0/s=2(br)
     