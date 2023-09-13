import enum
import os.path
import shutil
import functools
import operator
import collections
import subprocess
import re
import gemm_op
from gemm_op import *
import user

def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


class EmitMake:
    def __init__(self):
        self.make_template =  """

CFLAGS=-I ~/workspace/composable_kernel/include -I /opt/workspace/rocm-5.1.1/hip/include -I ~/workspace/composable_kernel/include/ -I ~/workspace/composable_kernel/include/ck/ -I ~/workspace/composable_kernel/example/01_gemm/ -I ~/workspace/composable_kernel/library/include/  -I ~/workspace/composable_kernel/library/src/utility/ -I ~/workspace/composable_kernel/include/ck/problem_transform/ -I ~/workspace/composable_kernel/include/ck/tensor/ -I ~/workspace/composable_kernel/include/ck/tensor_description/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/block/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/device/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/device/impl/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/element/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/grid/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/thread/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/warp/ -I ~/workspace/composable_kernel/include/ck/host_utility -I /external/include/half/ -I ~/workspace/composable_kernel/library/include/ck/library/host/ -I ~/workspace/composable_kernel/library/include/ck/library/host_tensor/ -I ~/workspace/composable_kernel/library/include/ck/library/obselete_driver_offline/ -I ~/workspace/composable_kernel/library/include/ck/library/reference_tensor_operation/cpu/ -I ~/workspace/composable_kernel/library/include/ck/library/reference_tensor_operation/gpu/ -I ~/workspace/composable_kernel/library/include/ck/library/tensor_operation_instance/ -I ~/workspace/composable_kernel/library/include/ck/library/tensor_operation_instance/gpu/" + "reduce/ -I ~/workspace/composable_kernel/library/include/ck/library/tensor_op/ -I ~/workspace/composable_kernel/library/include/ck/library/utility/ -I ~/workspace/composable_kernel/profiler/include/

CXXFLAGS = -std=c++17

device_memory.o: ../../../../../library/src/utility/device_memory.cpp
	hipcc -fPIC -fvisibility=hidden $(CXXFLAGS) $(CFLAGS) -c ../../../../../library/src/utility/device_memory.cpp

host_tensor.o: ../../../../../library/src/utility/host_tensor.cpp
	hipcc -fPIC -fvisibility=hidden $(CXXFLAGS) $(CFLAGS) -c ../../../../../library/src/utility/host_tensor.cpp

main.o: main.cpp
	hipcc -fPIC -fvisibility=hidden $(CXXFLAGS) -w $(CFLAGS) -L/opt/rocm-5.3.0/rocrand -lrocrand -x hip -c  main.cpp

obj_files = 256_128_128_8_2.o 256_128_128_16_2.o 128_32_128_8_2.o 128_64_32_8_2.o 128_64_128_8_2.o 128_128_32_8_2.o 128_128_64_8_2.o 256_64_128_8_2.o 256_128_64_8_2.o

%.o : %.cpp
	hipcc -fPIC -fvisibility=hidden $(CXXFLAGS) -w $(CFLAGS) -L/opt/rocm-5.3.0/rocrand -lrocrand -x hip -c $<

done: libtest.so
	cp libtest.so /lib

main: main.o device_memory.o host_tensor.o $(obj_files)
	hipcc $(CXXFLAGS) $(CFLAGS) main.o host_tensor.o device_memory.o $(obj_files) -o main

libtest.so: $(obj_files) host_tensor.o device_memory.o
	hipcc -shared $(CXXFLAGS) $(CFLAGS) -o $@ $(obj_files) host_tensor.o device_memory.o

all: done main.o
	hipcc $(CXXFLAGS) $(CFLAGS) -L/root/workspace/composable_kernel/python/ait_impl/generation/ex/shared -l test main.o -o example

clean:
	rm -f *.o libtest.so example
    

    """

    def emit(self, instances):
        obj_files = instances
        values = {
            'obj_files' : str(instances)
        }
        m_template = self.make_template
        cf = open("Makefile", 'w')
        cf.write(SubstituteTemplate(m_template, values))
        cf.close()

        PIPE = -1
        STDOUT = -2

        proc = subprocess.Popen(
        ["make all"],
        shell=True,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )

        out, err = proc.communicate()