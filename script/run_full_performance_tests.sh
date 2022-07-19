#!/bin/bash 
#
# in order to run this script you'd first need to build the ckProfiler executable in ../build/bin/
# you would also need to set up some environment variables in order to 
# post your new test results to the database and compare them to the baseline
# please contact Illia.Silin@amd.com for more details
#
# run the script as "./run_full_performance_tests.sh <verification> <tag for your test environment> <gpu_arch>
# input arguments: 
# verification = 0 : do not verify result correctness on CPU
#              = 1 : verifuy correctness on CPU (may take a long time)
# environment tag : a string describing the specifics of your test environment
# gpu_arch        : a string for GPU architecture, e.g. "gfx908" or "gfx90a".

#get the command line arguments:
export verify=$1
echo 'Verification: ' $verify
export env_type=$2
echo 'Environment type: ' $env_type
export gpu_arch=$3
echo 'GPU architecture: ' $gpu_arch

function print_log_header(){
	rm -f $1;
	GIT='git --git-dir='/composable_kernel'/.git'
	$GIT status | grep -e 'On branch' &> $1;
	echo -n 'Node name: ' >> $1; hostname >> $1;
	#get GPU_arch and number of compute units from rocminfo
	echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1;
	rocminfo | grep "Compute Unit:" >> $1;
	hipcc --version | grep -e 'HIP version'  >> $1;
	echo 'Environment type: ' $2 >> $1;
	/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1;
}

#run gemm tests
export gemm_log="perf_gemm_${gpu_arch}.log"
print_log_header $gemm_log $env_type
./profile_gemm.sh gemm 0 0 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 0 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 0 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 0 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 0 1 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 1 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 1 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 1 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 0 2 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 2 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 2 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 2 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 0 3 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 3 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 3 $verify 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 3 $verify 1 0 5 | tee -a $gemm_log

#run resnet50 tests
export resnet256_log="perf_resnet50_N256_${gpu_arch}.log"
print_log_header $resnet256_log $env_type
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 2 0 1 256 | tee -a $resnet256_log
export resnet4_log="perf_resnet50_N4_${gpu_arch}.log"
print_log_header $resnet4_log $env_type
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 2 0 1 4 | tee -a $resnet4_log

#run batched_gemm tests
export batched_gemm_log="perf_batched_gemm_${gpu_arch}.log"
print_log_header $batched_gemm_log $env_type
./profile_batched_gemm.sh batched_gemm 0 0 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 1 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 2 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 3 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 0 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 1 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 2 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 3 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 0 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 1 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 2 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 3 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 0 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 1 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 2 $verify 2 0 5 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 3 $verify 2 0 5 | tee -a $batched_gemm_log

#run grouped_gemm tests
export grouped_gemm_log="perf_grouped_gemm_${gpu_arch}.log"
print_log_header $grouped_gemm_log $env_type
./profile_grouped_gemm.sh grouped_gemm 1 0 $verify 2 0 5 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 1 $verify 2 0 5 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 2 $verify 2 0 5 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 3 $verify 2 0 5 | tee -a $grouped_gemm_log

#run fwd_conv tests
export fwd_conv_log="perf_fwd_conv_${gpu_arch}.log"
print_log_header $fwd_conv_log $env_type
./profile_conv.sh conv_fwd 0 1 $verify 2 0 5 2 256 | tee -a $fwd_conv_log
./profile_conv.sh conv_fwd 1 1 $verify 2 0 5 2 256 | tee -a $fwd_conv_log
./profile_conv.sh conv_fwd 2 1 $verify 2 0 5 2 256 | tee -a $fwd_conv_log
./profile_conv.sh conv_fwd 3 1 $verify 2 0 5 2 256 | tee -a $fwd_conv_log

#run bwd_conv tests
export bwd_conv_log="perf_bwd_conv_${gpu_arch}.log"
print_log_header $bwd_conv_log $env_type
./profile_conv.sh conv2d_bwd_data 0 1 1 1 $verify 2 0 5 128 | tee -a $bwd_conv_log
./profile_conv.sh conv2d_bwd_data 1 1 1 1 $verify 2 0 5 128 | tee -a $bwd_conv_log
./profile_conv.sh conv2d_bwd_data 2 1 1 1 $verify 2 0 5 128 | tee -a $bwd_conv_log
./profile_conv.sh conv2d_bwd_data 3 1 1 1 $verify 2 0 5 128 | tee -a $bwd_conv_log

#run fusion tests
export fusion_log="perf_fusion_${gpu_arch}.log"
print_log_header $fusion_log $env_type
./profile_gemm_bilinear.sh gemm_bilinear 1 0 $verify 2 0 1 | tee -a $fusion_log
./profile_gemm_bilinear.sh gemm_bilinear 1 1 $verify 2 0 1 | tee -a $fusion_log
./profile_gemm_bilinear.sh gemm_bilinear 1 2 $verify 2 0 1 | tee -a $fusion_log
./profile_gemm_bilinear.sh gemm_bilinear 1 3 $verify 2 0 1 | tee -a $fusion_log

#run reduction tests
export reduction_log="perf_reduction_${gpu_arch}.log"
print_log_header $reduction_log $env_type
./profile_reduce_with_index.sh $verify 2 10 --half | tee -a $reduction_log
./profile_reduce_no_index.sh $verify 2 10 --half | tee -a $reduction_log

#process results
#python3 process_perf_data.py $gemm_log
#python3 process_perf_data.py $resnet256_log
#python3 process_perf_data.py $resnet4_log
#python3 process_perf_data.py $batched_gemm_log
#python3 process_perf_data.py $grouped_gemm_log
#python3 process_perf_data.py $fwd_conv_log
#python3 process_perf_data.py $bwd_conv_log
#python3 process_perf_data.py $fusion_log
#python3 process_perf_data.py $reduction_log