make tile_example_layernorm2d_bwd -j 200
./bin/tile_example_layernorm2d_bwd -m=2048 -n=2048
rocprofv2 --kernel-trace -d /home/dteng/PerfProf/out -o kernel_trace
rocprofv2 -i /home/dteng/PerfProf/input.txt --plugin att auto -d /home/dteng/PerfProf/out
rocprofv2 -i /home/dteng/PerfProf/input.txt --plugin att auto --mode csv -d /home/dteng/PerfProf/out