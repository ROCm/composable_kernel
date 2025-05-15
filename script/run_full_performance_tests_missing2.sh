#!/bin/bash 
#
# in order to run this script you'd first need to build the ckProfiler executable in ../build/bin/
# you would also need to set up some environment variables in order to 
# post your new test results to the database and compare them to the baseline
# please contact Illia.Silin@amd.com for more details
#
# run the script as "./run_full_performance_tests.sh <verification> <tag for your test environment> <branch name> <node name>
# input arguments: 
# verification = 0 : do not verify result correctness on CPU
#              = 1 : verifuy correctness on CPU (may take a long time)
# environment tag  : a string describing the specifics of your test environment
# branch name      : name of the branch in git repo (git status | grep -e 'On branch')
# node name        : $hostname

#get the command line arguments:
export verify=$1
echo 'Verification: ' $verify
export env_type=$2
echo 'Environment type: ' $env_type
export branch=$3
echo 'Branch name: ' $branch
export host_name=$4
echo 'Host name: ' $host_name
function print_log_header(){
	rm -f $1;
	echo 'On branch ' $3 &> $1;
	echo 'Node name: ' $4 >> $1;
	#get GPU_arch and number of compute units from rocminfo
	echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1;
	rocminfo | grep "Compute Unit:" >> $1;
	hipcc --version | grep -e 'HIP version'  >> $1;
	echo 'Environment type: ' $2 >> $1;
	/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1;
}

#run grouped_bwd_data tests
export grouped_conv_bwd_data_log="perf_grouped_conv_bwd_data.log"
print_log_header $grouped_conv_bwd_data_log $env_type $branch $host_name
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 0 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 1 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 2 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 0 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 2 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log

#run batched_gemm_b_scale tests	
export batched_gemm_b_scale_log="perf_batched_gemm_b_scale.log"	
print_log_header $batched_gemm_b_scale_log $env_type $branch $host_name	
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 0 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 1 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 2 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 3 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 4 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 5 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 6 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 7 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log

#run batched_gemm_reduce tests	
export batched_gemm_reduce_log="perf_batched_gemm_reduce.log"	
print_log_header $batched_gemm_reduce_log $env_type $branch $host_name	
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log

#run conv_fwd_bias_relu_add tests	
export conv_fwd_bias_relu_add_log="perf_conv_fwd_bias_relu_add.log"	
print_log_header $conv_fwd_bias_relu_add_log $env_type $branch $host_name	
./profile_conv_fwd_bias_relu_add.sh conv_fwd_bias_relu_add 1 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_bias_relu_add_log

#run conv_tensor_rearrange tests	
export conv_tensor_rearrange_log="perf_conv_tensor_rearrange.log"	
print_log_header $conv_tensor_rearrange_log $env_type $branch $host_name	
./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 0 0 $verify 1 0 1 0 256 2>&1 | tee -a $conv_tensor_rearrange_log
./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 1 0 $verify 1 0 1 0 256 2>&1 | tee -a $conv_tensor_rearrange_log
./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 2 0 $verify 1 0 1 0 256 2>&1 | tee -a $conv_tensor_rearrange_log
./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 3 0 $verify 1 0 1 0 256 2>&1 | tee -a $conv_tensor_rearrange_log

./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 1 1 $verify 1 0 1 0 256 2>&1 | tee -a $conv_tensor_rearrange_log
./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 3 1 $verify 1 0 1 0 256 2>&1 | tee -a $conv_tensor_rearrange_log

./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 1 0 $verify 1 0 1 1 256 2>&1 | tee -a $conv_tensor_rearrange_log
./profile_conv_tensor_rearrange.sh conv_tensor_rearrange 1 1 $verify 1 0 1 1 256 2>&1 | tee -a $conv_tensor_rearrange_log

#run gemm_ab_scale tests	
export gemm_ab_scale_log="perf_gemm_ab_scale.log"	
print_log_header $gemm_ab_scale_log $env_type $branch $host_name	
./profile_gemm_b_scale.sh gemm_ab_scale 7 1 $verify 1 0 1  2>&1 | tee -a $gemm_ab_scale_log

#run gemm_add tests	
export gemm_add_log="perf_gemm_add.log"	
print_log_header $gemm_add_log $env_type $branch $host_name	
./profile_gemm_d0_e.sh gemm_add 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_log
./profile_gemm_d0_e.sh gemm_add 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_log
./profile_gemm_d0_e.sh gemm_add 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_log
./profile_gemm_d0_e.sh gemm_add 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_log

#run gemm_b_scale tests	
export gemm_b_scale_log="perf_gemm_b_scale.log"	
print_log_header $gemm_b_scale_log $env_type $branch $host_name	
./profile_gemm_b_scale.sh gemm_b_scale 0 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 1 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 2 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 3 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 4 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 5 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 6 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 7 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 8 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log

#run gemm_multiply_multiply_weight_preshuffle tests	
export gemm_multiply_multiply_weight_preshuffle_log="perf_gemm_multiply_multiply_weight_preshuffle.log"	
print_log_header $gemm_multiply_multiply_weight_preshuffle_log $env_type $branch $host_name	
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply_weight_preshuffle 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_weight_preshuffle_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply_weight_preshuffle 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_weight_preshuffle_log

#run gemm_multiply_multiply tests	
export gemm_multiply_multiply_log="perf_gemm_multiply_multiply.log"	
print_log_header $gemm_multiply_multiply_log $env_type $branch $host_name	
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 4 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 5 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 6 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 7 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 8 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 9 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 10 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 4 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 5 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 6 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 7 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 8 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 9 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log
./profile_gemm_d0_d1_e.sh gemm_multiply_multiply 10 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_multiply_log

#run grouped_gemm_fixed_nk tests	
export grouped_gemm_fixed_nk_log="perf_grouped_gemm_fixed_nk.log"	
print_log_header $grouped_gemm_fixed_nk_log $env_type $branch $host_name	
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 0 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 1 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 2 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 3 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 0 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 1 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 2 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log
./profile_grouped_gemm_fixed_nk.sh grouped_gemm_fixed_nk 3 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fixed_nk_log

#run grouped_gemm_tile_loop tests	
export grouped_gemm_tile_loop_log="perf_grouped_gemm_tile_loop.log"	
print_log_header $grouped_gemm_tile_loop_log $env_type $branch $host_name	
./profile_grouped_gemm.sh grouped_gemm_tile_loop 0 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_tile_loop_log
./profile_grouped_gemm.sh grouped_gemm_tile_loop 0 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_tile_loop_log

#run avg_pool2d_bwd tests	
export avg_pool2d_bwd_log="perf_avg_pool2d_bwd.log"	
print_log_header $avg_pool2d_bwd_log $env_type $branch $host_name	
./profile_avg_pool2d_bwd.sh avg_pool2d_bwd 0 $verify 1 0 1 2>&1 | tee -a $avg_pool2d_bwd_log
./profile_avg_pool2d_bwd.sh avg_pool2d_bwd 1 $verify 1 0 1 2>&1 | tee -a $avg_pool2d_bwd_log
./profile_avg_pool2d_bwd.sh avg_pool2d_bwd 3 $verify 1 0 1 2>&1 | tee -a $avg_pool2d_bwd_log
./profile_avg_pool2d_bwd.sh avg_pool2d_bwd 5 $verify 1 0 1 2>&1 | tee -a $avg_pool2d_bwd_log
./profile_avg_pool2d_bwd.sh avg_pool2d_bwd 7 $verify 1 0 1 2>&1 | tee -a $avg_pool2d_bwd_log

#run avg_pool3d_bwd tests	
export avg_pool3d_bwd_log="perf_avg_pool3d_bwd.log"	
print_log_header $avg_pool3d_bwd_log $env_type $branch $host_name	
./profile_avg_pool3d_bwd.sh avg_pool3d_bwd 0 $verify 1 0 1 2>&1 | tee -a $avg_pool3d_bwd_log
./profile_avg_pool3d_bwd.sh avg_pool3d_bwd 1 $verify 1 0 1 2>&1 | tee -a $avg_pool3d_bwd_log
./profile_avg_pool3d_bwd.sh avg_pool3d_bwd 5 $verify 1 0 1 2>&1 | tee -a $avg_pool3d_bwd_log

#run bnorm_bwd tests	
export bnorm_bwd_log="perf_bnorm_bwd.log"	
print_log_header $bnorm_bwd_log $env_type $branch $host_name	
./profile_bnorm.sh bnorm_bwd 0 $verify 0 1 0 2>&1 | tee -a $bnorm_bwd_log
./profile_bnorm.sh bnorm_bwd 1 $verify 0 1 0 2>&1 | tee -a $bnorm_bwd_log
./profile_bnorm.sh bnorm_bwd 5 $verify 0 1 0 2>&1 | tee -a $bnorm_bwd_log
./profile_bnorm.sh bnorm_bwd 6 $verify 0 1 0 2>&1 | tee -a $bnorm_bwd_log

#run bnorm_fwd tests	
export bnorm_fwd_log="perf_bnorm_fwd.log"
print_log_header $bnorm_fwd_log $env_type $branch $host_name	
./profile_bnorm_fwd.sh bnorm_fwd 0 $verify 0 1 0 2>&1 | tee -a $bnorm_fwd_log
./profile_bnorm_fwd.sh bnorm_fwd 1 $verify 0 1 0 2>&1 | tee -a $bnorm_fwd_log
./profile_bnorm_fwd.sh bnorm_fwd 5 $verify 0 1 0 2>&1 | tee -a $bnorm_fwd_log
./profile_bnorm_fwd.sh bnorm_fwd 6 $verify 0 1 0 2>&1 | tee -a $bnorm_fwd_log

#run bnorm_infer tests	
export bnorm_infer_log="perf_bnorm_infer.log"	
print_log_header $bnorm_infer_log $env_type $branch $host_name	
./profile_bnorm.sh bnorm_infer 0 $verify 0 1 0 2>&1 | tee -a $bnorm_infer_log
./profile_bnorm.sh bnorm_infer 1 $verify 0 1 0 2>&1 | tee -a $bnorm_infer_log
./profile_bnorm.sh bnorm_infer 5 $verify 0 1 0 2>&1 | tee -a $bnorm_infer_log
./profile_bnorm.sh bnorm_infer 6 $verify 0 1 0 2>&1 | tee -a $bnorm_infer_log

#run max_pool2d_bwd tests	
export max_pool2d_bwd_log="perf_max_pool2d_bwd.log"	
print_log_header $max_pool2d_bwd_log $env_type $branch $host_name	
./profile_max_pool2d_bwd.sh max_pool2d_bwd 0  $verify 1 0 1 2>&1 | tee -a $max_pool2d_bwd_log
./profile_max_pool2d_bwd.sh max_pool2d_bwd 1  $verify 1 0 1 2>&1 | tee -a $max_pool2d_bwd_log
./profile_max_pool2d_bwd.sh max_pool2d_bwd 3  $verify 1 0 1 2>&1 | tee -a $max_pool2d_bwd_log
./profile_max_pool2d_bwd.sh max_pool2d_bwd 5  $verify 1 0 1 2>&1 | tee -a $max_pool2d_bwd_log

#run max_pool2d_fwd tests	
export max_pool2d_fwd_log="perf_max_pool2d_fwd.log"	
print_log_header $max_pool2d_fwd_log $env_type $branch $host_name	
./profile_max_pool2d_fwd.sh max_pool2d_fwd 0 $verify 1 0 1 2>&1 | tee -a $max_pool2d_fwd_log
./profile_max_pool2d_fwd.sh max_pool2d_fwd 1 $verify 1 0 1 2>&1 | tee -a $max_pool2d_fwd_log
./profile_max_pool2d_fwd.sh max_pool2d_fwd 2 $verify 1 0 1 2>&1 | tee -a $max_pool2d_fwd_log
./profile_max_pool2d_fwd.sh max_pool2d_fwd 3 $verify 1 0 1 2>&1 | tee -a $max_pool2d_fwd_log
./profile_max_pool2d_fwd.sh max_pool2d_fwd 4 $verify 1 0 1 2>&1 | tee -a $max_pool2d_fwd_log

#run max_pool3d_bwd tests	
export max_pool3d_bwd_log="perf_max_pool3d_bwd.log"	
print_log_header $max_pool3d_bwd_log $env_type $branch $host_name	
./profile_max_pool3d.sh max_pool3d_bwd 0 $verify 1 0 1 2>&1 | tee -a $max_pool3d_bwd_log
./profile_max_pool3d.sh max_pool3d_bwd 1 $verify 1 0 1 2>&1 | tee -a $max_pool3d_bwd_log
./profile_max_pool3d.sh max_pool3d_bwd 5 $verify 1 0 1 2>&1 | tee -a $max_pool3d_bwd_log

#run pool3d_fwd tests	
export pool3d_fwd_log="perf_pool3d_fwd.log"	
print_log_header $pool3d_fwd_log $env_type $branch $host_name	
./profile_pool3d_fwd.sh pool3d_fwd 0 $verify 1 0 1 0 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 1 $verify 1 0 1 0 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 3 $verify 1 0 1 0 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 5 $verify 1 0 1 0 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 7 $verify 1 0 1 0 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 0 $verify 1 0 1 1 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 1 $verify 1 0 1 1 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 3 $verify 1 0 1 1 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 5 $verify 1 0 1 1 2>&1 | tee -a $pool3d_fwd_log
./profile_pool3d_fwd.sh pool3d_fwd 7 $verify 1 0 1 1 2>&1 | tee -a $pool3d_fwd_log

