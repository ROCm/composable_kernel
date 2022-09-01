#!/bin/bash 
#
# in order to run this script you'd need the following python packages:

pip3 install --upgrade pip
pip3 install sqlalchemy pymysql pandas sshtunnel

# you would also need to set up some environment variables in order to 
# post your new test results to the database and compare them to the baseline
# please contact Illia.Silin@amd.com for more details

#process results
gpu_arch=$1
python3 process_perf_data.py perf_gemm_"$gpu_arch".log
python3 process_perf_data.py perf_resnet50_N256_"$gpu_arch".log
python3 process_perf_data.py perf_resnet50_N4_"$gpu_arch".log
python3 process_perf_data.py perf_batched_gemm_"$gpu_arch".log
python3 process_perf_data.py perf_grouped_gemm_"$gpu_arch".log
python3 process_perf_data.py perf_conv_fwd_"$gpu_arch".log
python3 process_perf_data.py perf_conv_bwd_data_"$gpu_arch".log
python3 process_perf_data.py perf_gemm_bilinear_"$gpu_arch".log
python3 process_perf_data.py perf_reduction_"$gpu_arch".log
#python3 process_perf_data.py perf_splitK_gemm_"$gpu_arch".log
#python3 process_perf_data.py perf_onnx_gemm_"$gpu_arch".log
