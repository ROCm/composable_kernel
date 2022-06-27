#!/bin/bash 
#
# in order to run this script you'd first need to build the ckProfiler executable in ../build/bin/
# and make sure the following python packages are installed in your environment:

pip3 install --upgrade pip
pip3 install sqlalchemy pymysql pandas sshtunnel

# you would also need to set up some environment variables in order to 
# post your new test results to the database and compare them to the baseline
# please contact Illia.Silin@amd.com for more details
#
# run the script as "./run_performance_tests.sh <tag for your test environment>

#get the test environment type:
export env_type=$1
echo 'Environment type ' $env_type

function print_log_header(){
	rm -f $1;
	git status | grep -e 'On branch' > $1;
	echo -n 'Node name: ' >>$1; hostname >> $1;
	#get GPU_arch and number of compute units from rocminfo
	echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1;
	rocminfo | grep "Compute Unit:" >> $1;
	hipcc --version | grep -e 'HIP version'  >> $1;
	echo 'Environment type: ' $2 >>$1;
	/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1;
}
#run gemm tests
export gemm_log="perf_gemm.log"
print_log_header $gemm_log $env_type
./profile_gemm.sh gemm 0 0 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 0 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 0 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 0 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 0 1 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 1 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 1 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 1 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 0 2 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 2 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 2 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 2 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 0 3 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 1 3 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 2 3 0 1 0 5 | tee -a $gemm_log
./profile_gemm.sh gemm 3 3 0 1 0 5 | tee -a $gemm_log
python3 parse_perf_data.py $gemm_log

#run resnet50 test
export resnet_log="perf_resnet50.log"
print_log_header $resnet_log $env_type
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 0 2 0 1 256 | tee -a $resnet_log
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 0 2 0 1 4 | tee -a $resnet_log
#the script will put the results from N=256 and N=4 runs into separate tables
python3 parse_perf_data.py $resnet_log
