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

#run gemm tests
export gemm_log="perf_gemm.log"
print_log_header $gemm_log $env_type $branch $host_name
./profile_gemm.sh gemm 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 3 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 3 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 3 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 3 $verify 1 0 1 2>&1 | tee -a $gemm_log

#run batched_gemm tests
export batched_gemm_log="perf_batched_gemm.log"
print_log_header $batched_gemm_log $env_type $branch $host_name
./profile_batched_gemm.sh batched_gemm 0 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log

#run grouped_gemm tests
export grouped_gemm_log="perf_grouped_gemm.log"
print_log_header $grouped_gemm_log $env_type $branch $host_name
./profile_grouped_gemm.sh grouped_gemm 1 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 2 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 3 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log

#run GEMM+Bilinear tests
export gemm_bilinear_log="perf_gemm_bilinear.log"
print_log_header $gemm_bilinear_log $env_type $branch $host_name
./profile_gemm_bilinear.sh gemm_bilinear 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log
./profile_gemm_bilinear.sh gemm_bilinear 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log
./profile_gemm_bilinear.sh gemm_bilinear 1 2 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log
./profile_gemm_bilinear.sh gemm_bilinear 1 3 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log

#run grouped_fwd tests
export grouped_conv_fwd_log="perf_grouped_conv_fwd.log"
print_log_header $grouped_conv_fwd_log $env_type $branch $host_name
./profile_grouped_conv_fwd.sh grouped_conv_fwd 0 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_log
./profile_grouped_conv_fwd.sh grouped_conv_fwd 1 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_log
./profile_grouped_conv_fwd.sh grouped_conv_fwd 2 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_log

#run grouped_bwd_data tests
export grouped_conv_bwd_data_log="perf_grouped_conv_bwd_data.log"
print_log_header $grouped_conv_bwd_data_log $env_type $branch $host_name
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 0 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 1 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 2 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 0 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log
./profile_grouped_conv_bwd_data.sh grouped_conv_bwd_data 2 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_bwd_data_log

#run grouped_bwd_weight tests
export grouped_conv_bwd_weight_log="perf_grouped_conv_bwd_weight.log"
print_log_header $grouped_conv_bwd_weight_log $env_type $branch $host_name
./profile_grouped_conv_bwd_weight.sh grouped_conv_bwd_weight 0 2 $verify 1 0 1 256 1 2>&1 | tee -a $grouped_conv_bwd_weight_log
./profile_grouped_conv_bwd_weight.sh grouped_conv_bwd_weight 1 2 $verify 1 0 1 256 1 2>&1 | tee -a $grouped_conv_bwd_weight_log
./profile_grouped_conv_bwd_weight.sh grouped_conv_bwd_weight 2 2 $verify 1 0 1 256 1 2>&1 | tee -a $grouped_conv_bwd_weight_log
./profile_grouped_conv_bwd_weight.sh grouped_conv_bwd_weight 1 2 $verify 1 0 1 256 4 2>&1 | tee -a $grouped_conv_bwd_weight_log

#run resnet50 tests
export resnet256_log="perf_resnet50_N256.log"
print_log_header $resnet256_log $env_type $branch $host_name
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 1 0 1 256 2>&1 | tee -a $resnet256_log
export resnet4_log="perf_resnet50_N4.log"
print_log_header $resnet4_log $env_type $branch $host_name
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 1 0 1 4 2>&1 | tee -a $resnet4_log

#run reduction tests
export reduction_log="perf_reduction.log"
print_log_header $reduction_log $env_type $branch $host_name
./profile_reduce_with_index.sh $verify 2 10 --half 2>&1 | tee -a $reduction_log
./profile_reduce_no_index.sh $verify 2 10 --half 2>&1 | tee -a $reduction_log

#run splitK_gemm tests, first correctness verification, then performance
export splitK_gemm_log="perf_splitK_gemm.log"
print_log_header $splitK_gemm_log $env_type $branch $host_name
./profile_splitK_gemm.sh gemm_splitk 0 0 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 0 1 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 0 2 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 0 3 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 0 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 1 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 2 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 3 $verify 1 0 1 4 2>&1 | tee -a $splitK_gemm_log

#run ONNX gemm tests
export onnx_log="perf_onnx_gemm.log"
print_log_header $onnx_log $env_type $branch $host_name
./profile_onnx_gemm.sh gemm 0 0 $verify 1 0 1 2>&1 | tee -a $onnx_log
./profile_onnx_gemm.sh gemm 1 0 $verify 1 0 1 2>&1 | tee -a $onnx_log

#run mixed fp16/fp8 and fp8/fp16 gemm tests
export mixed_gemm_log="perf_mixed_gemm.log"
print_log_header $mixed_gemm_log $env_type $branch $host_name
./profile_mixed_gemm.sh gemm_splitk 4 0 $verify 2 0 1 16 2>&1 | tee -a $mixed_gemm_log
./profile_mixed_gemm.sh gemm_splitk 5 0 $verify 2 0 1 16 2>&1 | tee -a $mixed_gemm_log

#run batched_gemm_add_relu_gemm_add tests	
export batched_gemm_add_relu_gemm_add_log="perf_batched_gemm_add_relu_gemm_add.log"	
print_log_header $batched_gemm_add_relu_gemm_add_log $env_type $branch $host_name	
./profile_batched_gemm_gemm.sh batched_gemm_add_relu_gemm_add 1 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_add_relu_gemm_add_log
./profile_batched_gemm_gemm.sh batched_gemm_add_relu_gemm_add 1 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_add_relu_gemm_add_log

#run batched_gemm_b_scale tests	
export batched_gemm_b_scale_log="perf_batched_gemm_b_scale.log"	
print_log_header $batched_gemm_b_scale_log $env_type $branch $host_name	
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 0 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 1 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 2 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 3 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 4 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 5 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 6 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 7 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log
#./profile_batched_gemm_b_scale.sh batched_gemm_b_scale 8 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_b_scale_log

#run batched_gemm_gemm tests	
export batched_gemm_gemm_log="perf_batched_gemm_gemm.log"	
print_log_header $batched_gemm_gemm_log $env_type $branch $host_name	
./profile_batched_gemm_gemm.sh batched_gemm_gemm 1 0 $verify 1 0 1  2>&1 | tee -a $batched_gemm_gemm_log
./profile_batched_gemm_gemm.sh batched_gemm_gemm 1 1 $verify 1 0 1  2>&1 | tee -a $batched_gemm_gemm_log

#run batched_gemm_multi_d tests	
export batched_gemm_multi_d_log="perf_batched_gemm_multi_d.log"	
print_log_header $batched_gemm_multi_d_log $env_type $branch $host_name	
./profile_batched_gemm.sh batched_gemm_multi_d 0 0 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log
./profile_batched_gemm.sh batched_gemm_multi_d 0 1 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log
./profile_batched_gemm.sh batched_gemm_multi_d 0 2 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log
./profile_batched_gemm.sh batched_gemm_multi_d 0 3 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log
./profile_batched_gemm.sh batched_gemm_multi_d 1 0 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log
./profile_batched_gemm.sh batched_gemm_multi_d 1 1 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log
./profile_batched_gemm.sh batched_gemm_multi_d 1 2 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log
./profile_batched_gemm.sh batched_gemm_multi_d 1 3 $verify 1 0 1  2>&1 | tee -a $batched_gemm_multi_d_log

#run batched_gemm_reduce tests	
export batched_gemm_reduce_log="perf_batched_gemm_reduce.log"	
print_log_header $batched_gemm_reduce_log $env_type $branch $host_name	
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log
./profile_batched_gemm_reduce.sh batched_gemm_reduce 1 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_reduce_log

#run contraction_bilinear tests	
export contraction_bilinear_log="perf_contraction_bilinear.log"	
print_log_header $contraction_bilinear_log $env_type $branch $host_name	
./profile_contraction_bilinear.sh contraction_bilinear 0 0 $verify 1 0 1 2>&1 | tee -a $contraction_bilinear_log
./profile_contraction_bilinear.sh contraction_bilinear 1 0 $verify 1 0 1 2>&1 | tee -a $contraction_bilinear_log
#./profile_contraction_bilinear.sh contraction_bilinear 2 0 $verify 1 0 1 2>&1 | tee -a $contraction_bilinear_log
#./profile_contraction_bilinear.sh contraction_bilinear 3 0 $verify 1 0 1 2>&1 | tee -a $contraction_bilinear_log

#run contraction_scale tests	
export contraction_scale_log="perf_contraction_scale.log"	
print_log_header $contraction_scale_log $env_type $branch $host_name	
./profile_contraction_scale.sh contraction_scale 0 0 $verify 1 0 1 2>&1 | tee -a $contraction_scale_log
./profile_contraction_scale.sh contraction_scale 1 0 $verify 1 0 1 2>&1 | tee -a $contraction_scale_log
#./profile_contraction_scale.sh contraction_scale 2 0 $verify 1 0 1 2>&1 | tee -a $contraction_scale_log
#./profile_contraction_scale.sh contraction_scale 3 0 $verify 1 0 1 2>&1 | tee -a $contraction_scale_log

#run conv_bwd_data tests	
export conv_bwd_data_log="perf_conv_bwd_data.log"	
print_log_header $conv_bwd_data_log $env_type $branch $host_name	
#./profile_conv.sh conv_bwd_data 0 0 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
#./profile_conv.sh conv_bwd_data 1 0 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
#./profile_conv.sh conv_bwd_data 2 0 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
#./profile_conv.sh conv_bwd_data 3 0 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
./profile_conv.sh conv_bwd_data 0 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
./profile_conv.sh conv_bwd_data 1 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
./profile_conv.sh conv_bwd_data 2 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
./profile_conv.sh conv_bwd_data 3 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log

#run conv_fwd_bias_relu_add tests	
export conv_fwd_bias_relu_add_log="perf_conv_fwd_bias_relu_add.log"	
print_log_header $conv_fwd_bias_relu_add_log $env_type $branch $host_name	
./profile_conv_fwd_bias_relu_add.sh conv_fwd_bias_relu_add 1 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_bias_relu_add_log

#run conv_fwd tests	
export conv_fwd_log="perf_conv_fwd.log"	
print_log_header $conv_fwd_log $env_type $branch $host_name	
./profile_conv.sh conv_fwd 0 0 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv.sh conv_fwd 1 0 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv.sh conv_fwd 2 0 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv.sh conv_fwd 3 0 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv.sh conv_fwd 0 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv.sh conv_fwd 1 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv.sh conv_fwd 2 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv.sh conv_fwd 3 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log

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

#run gemm_add_add_fastgelu tests	
export gemm_add_add_fastgelu_log="perf_gemm_add_add_fastgelu.log"	
print_log_header $gemm_add_add_fastgelu_log $env_type $branch $host_name	
#./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log
./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log
#./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log
#./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log
#./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log
./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log
#./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log
#./profile_gemm_d0_d1_e.sh gemm_add_add_fastgelu 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_add_fastgelu_log

#run gemm_add_fastgelu tests	
export gemm_add_fastgelu_log="perf_gemm_add_fastgelu.log"	
print_log_header $gemm_add_fastgelu_log $env_type $branch $host_name	
#./profile_gemm_d0_e.sh gemm_add_fastgelu 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
./profile_gemm_d0_e.sh gemm_add_fastgelu 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
./profile_gemm_d0_e.sh gemm_add_fastgelu 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
#./profile_gemm_d0_e.sh gemm_add_fastgelu 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
#./profile_gemm_d0_e.sh gemm_add_fastgelu 4 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
#./profile_gemm_d0_e.sh gemm_add_fastgelu 5 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
#./profile_gemm_d0_e.sh gemm_add_fastgelu 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
./profile_gemm_d0_e.sh gemm_add_fastgelu 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
./profile_gemm_d0_e.sh gemm_add_fastgelu 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
#./profile_gemm_d0_e.sh gemm_add_fastgelu 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
#./profile_gemm_d0_e.sh gemm_add_fastgelu 4 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log
#./profile_gemm_d0_e.sh gemm_add_fastgelu 5 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_fastgelu_log

#run gemm_add_multiply tests	
export gemm_add_multiply_log="perf_gemm_add_multiply.log"	
print_log_header $gemm_add_multiply_log $env_type $branch $host_name	
#./profile_gemm_d0_d1_e.sh gemm_add_multiply 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log
./profile_gemm_d0_d1_e.sh gemm_add_multiply 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log
#./profile_gemm_d0_d1_e.sh gemm_add_multiply 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log
#./profile_gemm_d0_d1_e.sh gemm_add_multiply 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log
#./profile_gemm_d0_d1_e.sh gemm_add_multiply 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log
./profile_gemm_d0_d1_e.sh gemm_add_multiply 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log
#./profile_gemm_d0_d1_e.sh gemm_add_multiply 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log
#./profile_gemm_d0_d1_e.sh gemm_add_multiply 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_multiply_log

#run gemm_add_relu_add_layernorm tests	
export gemm_add_relu_add_layernorm_log="perf_gemm_add_relu_add_layernorm.log"	
print_log_header $gemm_add_relu_add_layernorm_log $env_type $branch $host_name	
#./profile_gemm_d0_d1_e.sh gemm_add_relu_add_layernorm 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_add_layernorm_log
./profile_gemm_d0_d1_e.sh gemm_add_relu_add_layernorm 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_add_layernorm_log
#./profile_gemm_d0_d1_e.sh gemm_add_relu_add_layernorm 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_add_layernorm_log
#./profile_gemm_d0_d1_e.sh gemm_add_relu_add_layernorm 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_add_layernorm_log
./profile_gemm_d0_d1_e.sh gemm_add_relu_add_layernorm 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_add_layernorm_log
#./profile_gemm_d0_d1_e.sh gemm_add_relu_add_layernorm 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_add_layernorm_log

#run gemm_add_relu tests	
export gemm_add_relu_log="perf_gemm_add_relu.log"	
print_log_header $gemm_add_relu_log $env_type $branch $host_name	
./profile_gemm_d0_e.sh gemm_add_relu 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_log
./profile_gemm_d0_e.sh gemm_add_relu 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_log
#./profile_gemm_d0_e.sh gemm_add_relu 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_log
#./profile_gemm_d0_e.sh gemm_add_relu 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_relu_log

#run gemm_add_silu tests	
export gemm_add_silu_log="perf_gemm_add_silu.log"	
print_log_header $gemm_add_silu_log $env_type $branch $host_name	
./profile_gemm_d0_e.sh gemm_add_silu 0 0 $verify  1 0 1 2>&1 | tee -a $gemm_add_silu_log
./profile_gemm_d0_e.sh gemm_add_silu 1 0 $verify  1 0 1 2>&1 | tee -a $gemm_add_silu_log
./profile_gemm_d0_e.sh gemm_add_silu 0 1 $verify  1 0 1 2>&1 | tee -a $gemm_add_silu_log
./profile_gemm_d0_e.sh gemm_add_silu 1 1 $verify  1 0 1 2>&1 | tee -a $gemm_add_silu_log

#run gemm_add tests	
export gemm_add_log="perf_gemm_add.log"	
print_log_header $gemm_add_log $env_type $branch $host_name	
#./profile_gemm_d0_e.sh gemm_add 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_log
./profile_gemm_d0_e.sh gemm_add 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_add_log
#./profile_gemm_d0_e.sh gemm_add 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_log
./profile_gemm_d0_e.sh gemm_add 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_add_log

#run gemm_b_scale tests	
export gemm_b_scale_log="perf_gemm_b_scale.log"	
print_log_header $gemm_b_scale_log $env_type $branch $host_name	
#./profile_gemm_b_scale.sh gemm_b_scale 0 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
#./profile_gemm_b_scale.sh gemm_b_scale 1 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
#./profile_gemm_b_scale.sh gemm_b_scale 2 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
#./profile_gemm_b_scale.sh gemm_b_scale 3 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
#./profile_gemm_b_scale.sh gemm_b_scale 4 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
#./profile_gemm_b_scale.sh gemm_b_scale 5 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
#./profile_gemm_b_scale.sh gemm_b_scale 6 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
#./profile_gemm_b_scale.sh gemm_b_scale 7 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log
./profile_gemm_b_scale.sh gemm_b_scale 8 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_b_scale_log

#run gemm_bias_add_reduce tests	
export gemm_bias_add_reduce_log="perf_gemm_bias_add_reduce.log"	
print_log_header $gemm_bias_add_reduce_log $env_type $branch $host_name	
#./profile_gemm_d0_e.sh gemm_bias_add_reduce 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_bias_add_reduce_log
./profile_gemm_d0_e.sh gemm_bias_add_reduce 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_bias_add_reduce_log
#./profile_gemm_d0_e.sh gemm_bias_add_reduce 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_bias_add_reduce_log
./profile_gemm_d0_e.sh gemm_bias_add_reduce 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_bias_add_reduce_log

#run gemm_fastgelu tests	
export gemm_fastgelu_log="perf_gemm_fastgelu.log"	
print_log_header $gemm_fastgelu_log $env_type $branch $host_name	
#./profile_gemm.sh gemm_fastgelu 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log
./profile_gemm.sh gemm_fastgelu 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log
#./profile_gemm.sh gemm_fastgelu 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log
#./profile_gemm.sh gemm_fastgelu 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log
#./profile_gemm.sh gemm_fastgelu 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log
./profile_gemm.sh gemm_fastgelu 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log
#./profile_gemm.sh gemm_fastgelu 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log
#./profile_gemm.sh gemm_fastgelu 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_fastgelu_log

#run gemm_multiply_add tests	
export gemm_multiply_add_log="perf_gemm_multiply_add.log"	
print_log_header $gemm_multiply_add_log $env_type $branch $host_name	
./profile_gemm_d0_d1_e.sh gemm_multiply_add 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_add_log
./profile_gemm_d0_d1_e.sh gemm_multiply_add 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_add_log
./profile_gemm_d0_d1_e.sh gemm_multiply_add 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_add_log
./profile_gemm_d0_d1_e.sh gemm_multiply_add 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_multiply_add_log

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

#run gemm_reduce tests	
export gemm_reduce_log="perf_gemm_reduce.log"	
print_log_header $gemm_reduce_log $env_type $branch $host_name	
#./profile_splitK_gemm.sh gemm_reduce 0 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_reduce_log
./profile_splitK_gemm.sh gemm_reduce 1 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_reduce_log
#./profile_splitK_gemm.sh gemm_reduce 0 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_reduce_log
./profile_splitK_gemm.sh gemm_reduce 1 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_reduce_log

#run gemm_streamk tests	
export gemm_streamk_log="perf_gemm_streamk.log"	
print_log_header $gemm_streamk_log $env_type $branch $host_name	
./profile_gemm.sh gemm_streamk 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log
./profile_gemm.sh gemm_streamk 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log
./profile_gemm.sh gemm_streamk 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log
./profile_gemm.sh gemm_streamk 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log
./profile_gemm.sh gemm_streamk 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log
./profile_gemm.sh gemm_streamk 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log
./profile_gemm.sh gemm_streamk 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log
./profile_gemm.sh gemm_streamk 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_streamk_log

#run gemm_universal_batched tests	
export gemm_universal_batched_log="perf_gemm_universal_batched.log"	
print_log_header $gemm_universal_batched_log $env_type $branch $host_name	
./profile_gemm_universal_batched.sh gemm_universal_batched 0 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_batched_log
./profile_gemm_universal_batched.sh gemm_universal_batched 1 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_batched_log
./profile_gemm_universal_batched.sh gemm_universal_batched 0 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_batched_log
./profile_gemm_universal_batched.sh gemm_universal_batched 1 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_batched_log

#run gemm_universal_reduce tests	
export gemm_universal_reduce_log="perf_gemm_universal_reduce.log"	
print_log_header $gemm_universal_reduce_log $env_type $branch $host_name	
./profile_splitK_gemm.sh gemm_universal_reduce 0 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 1 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 2 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 3 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 4 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 5 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 6 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 0 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 1 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 2 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 3 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 4 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 5 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log
./profile_splitK_gemm.sh gemm_universal_reduce 6 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_reduce_log

#run gemm_universal_streamk tests	
export gemm_universal_streamk_log="perf_gemm_universal_streamk.log"	
print_log_header $gemm_universal_streamk_log $env_type $branch $host_name	
./profile_gemm_universal_streamk.sh gemm_universal_streamk 0 0 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 1 0 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 2 0 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 3 0 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 4 0 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 5 0 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 6 0 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 0 1 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 1 1 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 2 1 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 3 1 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 4 1 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 5 1 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 6 1 $verify 1 0 1 0 2>&1 | tee -a $gemm_universal_streamk_log

./profile_gemm_universal_streamk.sh gemm_universal_streamk 0 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 1 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 2 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 3 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 4 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 5 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 6 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 0 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 1 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 2 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 3 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 4 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 5 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 6 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_streamk_log

./profile_gemm_universal_streamk.sh gemm_universal_streamk 0 0 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 1 0 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 2 0 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 3 0 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 4 0 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 5 0 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 6 0 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 0 1 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 1 1 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 2 1 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 3 1 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 4 1 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 5 1 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log
./profile_gemm_universal_streamk.sh gemm_universal_streamk 6 1 $verify 1 0 1 2 2>&1 | tee -a $gemm_universal_streamk_log

#run gemm_universal tests	
export gemm_universal_log="perf_gemm_universal.log"	
print_log_header $gemm_universal_log $env_type $branch $host_name	
./profile_splitK_gemm.sh gemm_universal 0 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 1 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 2 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 3 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 4 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 5 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 6 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 7 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 8 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 9 0 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log

./profile_splitK_gemm.sh gemm_universal 0 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 1 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 2 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 3 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 4 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 5 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 6 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 7 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 8 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log
./profile_splitK_gemm.sh gemm_universal 9 1 $verify 1 0 1 1 2>&1 | tee -a $gemm_universal_log

#run grouped_conv_fwd_outelementop tests	
export grouped_conv_fwd_outelementop_log="perf_grouped_conv_fwd_outelementop.log"	
print_log_header $grouped_conv_fwd_outelementop_log $env_type $branch $host_name	
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 0 0 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 1 0 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 2 0 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 3 0 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 0 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 1 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 2 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
#./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 3 1 0 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 0 0 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 1 0 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 2 0 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 3 0 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 0 1 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 1 1 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 2 1 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log
./profile_grouped_conv_fwd_outelementop.sh grouped_conv_fwd_outelementop 3 1 1 $verify 1 0 1 256 2>&1 | tee -a $grouped_conv_fwd_outelementop_log

#run grouped_gemm_fastgelu tests	
export grouped_gemm_fastgelu_log="perf_grouped_gemm_fastgelu.log"	
print_log_header $grouped_gemm_fastgelu_log $env_type $branch $host_name	
#./profile_grouped_gemm.sh grouped_gemm_fastgelu 0 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log
./profile_grouped_gemm.sh grouped_gemm_fastgelu 1 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log
#./profile_grouped_gemm.sh grouped_gemm_fastgelu 2 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log
#./profile_grouped_gemm.sh grouped_gemm_fastgelu 3 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log
#./profile_grouped_gemm.sh grouped_gemm_fastgelu 0 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log
./profile_grouped_gemm.sh grouped_gemm_fastgelu 1 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log
#./profile_grouped_gemm.sh grouped_gemm_fastgelu 2 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log
#./profile_grouped_gemm.sh grouped_gemm_fastgelu 3 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_fastgelu_log

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

#run grouped_gemm_multiply_tile_loop tests	
export grouped_gemm_multiply_tile_loop_log="perf_grouped_gemm_multiply_tile_loop.log"	
print_log_header $grouped_gemm_multiply_tile_loop_log $env_type $branch $host_name	
./profile_grouped_gemm.sh grouped_gemm_multiply_tile_loop 0 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_multiply_tile_loop_log

#run grouped_gemm_tile_loop tests	
export grouped_gemm_tile_loop_log="perf_grouped_gemm_tile_loop.log"	
print_log_header $grouped_gemm_tile_loop_log $env_type $branch $host_name	
./profile_grouped_gemm.sh grouped_gemm_tile_loop 0 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_tile_loop_log
./profile_grouped_gemm.sh grouped_gemm_tile_loop 0 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_tile_loop_log

#run groupnorm tests	
export groupnorm_log="perf_groupnorm.log"	
print_log_header $groupnorm_log $env_type $branch $host_name	
./profile_groupnorm.sh groupnorm 0 $verify 1 0 1 2>&1 | tee -a $groupnorm_log
./profile_groupnorm.sh groupnorm 1 $verify 1 0 1 2>&1 | tee -a $groupnorm_log

#run permute_scale tests	
export permute_scale_log="perf_permute_scale.log"	
print_log_header $permute_scale_log $env_type $branch $host_name	
./profile_permute_scale.sh permute_scale 0 $verify 1 0 1 2>&1 | tee -a $permute_scale_log
./profile_permute_scale.sh permute_scale 1 $verify 1 0 1 2>&1 | tee -a $permute_scale_log

#run transpose tests	
export transpose_log="perf_transpose.log"	
print_log_header $transpose_log $env_type $branch $host_name	
./profile_transpose.sh transpose 0 $verify 1 0 1 2>&1 | tee -a $transpose_log
./profile_transpose.sh transpose 1 $verify 1 0 1 2>&1 | tee -a $transpose_log

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

#run groupnorm_bwd_data tests	
export groupnorm_bwd_data_log="perf_groupnorm_bwd_data.log"	
print_log_header $groupnorm_bwd_data_log $env_type $branch $host_name	
#./profile_groupnorm.sh groupnorm_bwd_data 0 $verify 1 0 1 2>&1 | tee -a $groupnorm_bwd_data_log
./profile_groupnorm.sh groupnorm_bwd_data 1 $verify 1 0 1 2>&1 | tee -a $groupnorm_bwd_data_log

#run groupnorm_bwd_gamma_beta tests	
export groupnorm_bwd_gamma_beta_log="perf_groupnorm_bwd_gamma_beta.log"	
print_log_header $groupnorm_bwd_gamma_beta_log $env_type $branch $host_name	
./profile_groupnorm.sh groupnorm_bwd_gamma_beta 0 $verify 1 0 1 2>&1 | tee -a $groupnorm_bwd_gamma_beta_log
./profile_groupnorm.sh groupnorm_bwd_gamma_beta 1 $verify 1 0 1 2>&1 | tee -a $groupnorm_bwd_gamma_beta_log

#run layernorm_bwd_data tests	
export layernorm_bwd_data_log="perf_layernorm_bwd_data.log"	
print_log_header $layernorm_bwd_data_log $env_type $branch $host_name	
./profile_layernorm.sh layernorm_bwd_data 0 $verify 1 0 1 2>&1 | tee -a $layernorm_bwd_data_log
./profile_layernorm.sh layernorm_bwd_data 1 $verify 1 0 1 2>&1 | tee -a $layernorm_bwd_data_log

#run layernorm_bwd_gamma_beta tests	
export layernorm_bwd_gamma_beta_log="perf_layernorm_bwd_gamma_beta.log"	
print_log_header $layernorm_bwd_gamma_beta_log $env_type $branch $host_name	
./profile_layernorm.sh layernorm_bwd_gamma_beta 0  $verify 1 0 1 2>&1 | tee -a $layernorm_bwd_gamma_beta_log
./profile_layernorm.sh layernorm_bwd_gamma_beta 1  $verify 1 0 1 2>&1 | tee -a $layernorm_bwd_gamma_beta_log

#run layernorm_fwd tests	
export layernorm_fwd_log="perf_layernorm_fwd.log"	
print_log_header $layernorm_fwd_log $env_type $branch $host_name	
./profile_layernorm.sh layernorm_fwd 0 $verify 1 0 1 2>&1 | tee -a $layernorm_fwd_log
./profile_layernorm.sh layernorm_fwd 1 $verify 1 0 1 2>&1 | tee -a $layernorm_fwd_log

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

#run softmax tests	
export softmax_log="perf_softmax.log"	
print_log_header $softmax_log $env_type $branch $host_name	
./profile_softmax.sh softmax 0 $verify 1 0 1 2>&1 | tee -a $softmax_log
./profile_softmax.sh softmax 1 $verify 1 0 1 2>&1 | tee -a $softmax_log
./profile_softmax.sh softmax 2 $verify 1 0 1 2>&1 | tee -a $softmax_log
./profile_softmax.sh softmax 3 $verify 1 0 1 2>&1 | tee -a $softmax_log