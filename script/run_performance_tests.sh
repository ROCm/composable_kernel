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

export gemm_log="perf_gemm.log"
rm -f $gemm_log
git status | grep -e 'On branch' > ${gemm_log}
echo -n 'Node name: ' >>${gemm_log}; hostname >> ${gemm_log}
#get GPU_arch and number of compute units from rocminfo
echo -n "GPU_arch: " >> ${gemm_log}; rocminfo | grep "Name:" | grep "gfx" >> ${gemm_log} 
rocminfo | grep "Compute Unit:" >> ${gemm_log} 
hipcc --version | grep -e 'HIP version'  >> ${gemm_log}
/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> ${gemm_log}
./profile_gemm.sh gemm 0 0 0 1 0 5 | tee -a ${gemm_log}
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

python3 parse_perf_data.py ${gemm_log}

#run resnet50 test
export resnet_log="perf_resnet50.log"
rm -f $resnet_log
git status | grep -e 'On branch' > ${resnet_log}
echo -n 'Node name: '>>${resnet_log}; hostname >>${resnet_log}
#get GPU_arch and number of compute units from rocminfo
echo -n "GPU_arch: " >> ${resnet_log}; rocminfo | grep "Name:" | grep "gfx" >> ${resnet_log}
rocminfo | grep "Compute Unit:" >> ${resnet_log} 
hipcc --version | grep -e 'HIP version'  >> ${resnet_log}
/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> ${resnet_log}
#first run tests with N=256
./profile_conv.sh conv_fwd_bias_relu 1 1 1 1 0 2 0 1 256 | tee -a ${resnet_log}
#then run with N=4
./profile_conv.sh conv_fwd_bias_relu 1 1 1 1 0 2 0 1 4 | tee -a ${resnet_log}
#the script will put the results from N=256 and N=4 runs into separate tables
python3 parse_perf_data.py ${resnet_log}
