#!/bin/bash 

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

cd ..

#run mha performance benchmarks
export fmha_fwd_log="perf_fmha_fwd_$GPU_arch.log"
print_log_header $fmha_fwd_log $env_type $branch $host_name
example/ck_tile/01_fmha/script/benchmark_fwd.sh 2>&1 | tee -a $fmha_fwd_log

export fmha_bwd_log="perf_fmha_bwd_$GPU_arch.log"
print_log_header $fmha_bwd_log $env_type $branch $host_name
example/ck_tile/01_fmha/script/benchmark_bwd.sh 2>&1 | tee -a $fmha_bwd_log

#run layernorm2d
export layernorm2d_log="perf_layernorm2d_$GPU_arch.log"
print_log_header  $layernorm2d_log $env_type $branch $host_name
example/ck_tile/02_layernorm2d/script/perf_test.sh 2>&1 | tee -a $layernorm2d_log

#run gemm
example/ck_tile/03_gemm/script/run_full_test.sh

#run image2col
export img2col_log="perf_img2col_$GPU_arch.log"
print_log_header  $layernorm2d_log $env_type $branch $host_name
./build/bin/tile_example_img2col 2>&1 | tee -a $img2col_log

#run reduce
export reduce_log="perf_reduce_$GPU_arch.log"
print_log_header  $reduce_log $env_type $branch $host_name
./build/bin/tile_example_reduce 2>&1 | tee -a $reduce_log

#run permute
export permute_log="perf_permute_$GPU_arch.log"
print_log_header  $permute_log $env_type $branch $host_name
example/ck_tile/06_permute/script/smoke_test.sh 2>&1 | tee -a $permute_log

#run topk_softmax
export topk_softmax_log="perf_topk_softmax_$GPU_arch.log"
print_log_header  $topk_softmax_log $env_type $branch $host_name
example/ck_tile/09_topk_softmax/script/smoke_test.sh 2>&1 | tee -a $topk_softmax_log

#run rmsnorm2d
export rmsnorm2d_log="perf_rmsnorm2d_$GPU_arch.log"
print_log_header  $rmsnorm2d_log $env_type $branch $host_name
example/ck_tile/10_rmsnorm2d/script/perf_test.sh 2>&1 | tee -a $rmsnorm2d_log

#run add_rmsnorm2d_rdquant
export add_rmsnorm2d_rdquant_log="perf_add_rmsnorm2d_rdquant_$GPU_arch.log"
print_log_header  $add_rmsnorm2d_rdquant_log $env_type $branch $host_name
example/ck_tile/11_add_rmsnorm2d_rdquant/script/perf_test.sh 2>&1 | tee -a $add_rmsnorm2d_rdquant_log

#run smoothquant
export smoothquant_log="perf_smoothquant_$GPU_arch.log"
print_log_header  $smoothquant_log $env_type $branch $host_name
example/ck_tile/12_smoothquant/script/perf_test.sh 2>&1 | tee -a $smoothquant_log

#run moe_sorting
export moe_sorting_log="perf_moe_sorting_$GPU_arch.log"
print_log_header  $moe_sorting_log $env_type $branch $host_name
example/ck_tile/13_moe_sorting/script/smoke_test.sh 2>&1 | tee -a $moe_sorting_log

#run moe_smoothquant
export moe_smoothquant_log="perf_moe_smoothquant_$GPU_arch.log"
print_log_header  $moe_smoothquant_log $env_type $branch $host_name
example/ck_tile/14_moe_sorting/script/perf_test.sh 2>&1 | tee -a $moe_smoothquant_log

#run fused_moe
export fused_moe_log="perf_fused_moe_$GPU_arch.log"
print_log_header  $fused_moe_log $env_type $branch $host_name
#example/ck_tile/14_moe_sorting/script/perf_test.sh 2>&1 | tee -a $fused_moe_log

#run batched_gemm


#run grouped_gemm


#run flatmm
export flatmm_log="perf_flatmm_$GPU_arch.log"
print_log_header  $flatmm_log $env_type $branch $host_name
example/ck_tile/18_flatmm/script/smoke_test_basic.sh 2>&1 | tee -a $flatmm_log

#run batched_transpose
export batched_transpose_log="perf_batched_transpose_$GPU_arch.log"
print_log_header  $batched_transpose_log $env_type $branch $host_name
example/ck_tile/35_batched_transpose/script/smoke_test_basic.sh 2>&1 | tee -a $batched_transpose_log
