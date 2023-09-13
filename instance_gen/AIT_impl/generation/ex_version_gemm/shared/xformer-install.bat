git config --global --add safe.directory /root/workspace/xformers-rocm
git config --global --add safe.directory /root/workspace/xformers-rocm/third_party/composable_kernel
git config --global --add safe.directory /root/workspace/xformers-rocm/third_party/cutlass
git config --global --add safe.directory /root/workspace/xformers-rocm/third_party/flash-attention
git submodule update --init --recursive
pip install -r requirements.txt
pip install -U matplotlib
pip install pandas
pip install seaborn
pip install triton
pip install -e  ./