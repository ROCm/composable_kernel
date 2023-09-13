import os
import re
from hashlib import sha1
from typing import Any, Dict, OrderedDict

import jinja2

#from ...target import Target

#templating

FUNC_CALL_PARAM_TEMPLATE = jinja2.Template("(void *)({{name}})")

INSTANCE_TEMPLATE = jinja2.Template(
    """
using {{name}} = {{ config_name }};
"""
)

ARGS_PARSE_TEMPLATE = jinja2.Template(
    """
{% for idx in range(rank) %}
  const int64_t in_{{idx}} = std::stoi(argv[{{ idx + 1 }}]);
{% endfor %}
"""
)

STRUCTS_DEF_TEMPLATE = jinja2.Template(
    """

struct ProfilerMemoryPool {
  ProfilerMemoryPool() {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
  }
  ~ProfilerMemoryPool() {
    for(int i = 0; i < ptrs.size(); i++){
      hipFree(ptrs[i]);
    }
  }

  template <typename DType>
  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    DType *d_x;
    hipMalloc(&d_x, length);

    float mean = 0.0f;
    float stddev = 1.0f;
    uint64_t seed = uniform_dist(gen);
    rocrand_set_seed(generator, seed);
    rocrand_generate_normal(generator, reinterpret_cast<float*>(d_x), size, mean, stddev);
    return d_x;
  }

  ck::half_t* AllocateHalfGaussianTensor(int64_t size) {
    return reinterpret_cast<ck::half_t*>(
        AllocateGaussianTensor<ck::half_t>(size));
  }

  int AllocateHalfTensor(int64_t size, int64_t copy) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    auto ptr = AllocateHalfGaussianTensor(size * copy);
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  ck::half_t* RequestHalfTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    ck::half_t* ptr = reinterpret_cast<ck::half_t*>(ptrs.at(idx));
    ptr += offset;
    offset += stride;
    if (offset == copy * stride) {
        offset = 0;
    }
    offsets[idx] = offset;
    return ptr;
  }
  std::vector<int64_t> offsets;
  std::vector<int64_t> strides;
  std::vector<int64_t> copies;
  std::vector<void*> ptrs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
  rocrand_generator generator;
};

// hack for DeviceMem linking error
// TODO fix this by making CK a header-only lib
// <<< hack begin
DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
  hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}
void* DeviceMem::GetDeviceBuffer() const { return mpDeviceBuf; }
void DeviceMem::ToDevice(const void* p) const
{
  hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}
void DeviceMem::FromDevice(void* p) const
{
  hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}
DeviceMem::~DeviceMem() { hipGetErrorString(hipFree(mpDeviceBuf)); }
struct KernelTimerImpl
{
  KernelTimerImpl() {
    hipGetErrorString(hipEventCreate(&mStart));
    hipGetErrorString(hipEventCreate(&mEnd));
  }
  ~KernelTimerImpl() {
    hipGetErrorString(hipEventDestroy(mStart));
    hipGetErrorString(hipEventDestroy(mEnd));
  }
  void Start() {
    hipGetErrorString(hipDeviceSynchronize());
    hipGetErrorString(hipEventRecord(mStart, nullptr));
  }
  void End() {
    hipGetErrorString(hipEventRecord(mEnd, nullptr));
    hipGetErrorString(hipEventSynchronize(mEnd));
  }
  float GetElapsedTime() const {
    float time;
    hipGetErrorString(hipEventElapsedTime(&time, mStart, mEnd));
    return time;
  }
  hipEvent_t mStart, mEnd;
};
// >>> hack end

"""
)

FUNC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <random>
#include <rocrand/rocrand.h>
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/reduction_operator.hpp"
{{extra_headers}}

{{extra_code}}

{{instances_decl}}

{{func_signature}}
{
{{shape_eval}}
{{exec_paths}}
}
    """
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{input}},
{{indent}}   {{output}},
{% for name in input_dim_names %}
{{indent}}    const_cast<int64_t *>(&{{name}}),
{% endfor %}
{{indent}}   stream
{{indent}});
    """
)

PROFILER_TEMPLATE = jinja2.Template(
    """
size_t GLOBAL_WORKSPACE_SIZE = 0;
{{op_func}}

{{structs_def}}

int main(int argc, char** argv) {
  {{args_parse}}
  auto memory_pool = std::make_unique<ProfilerMemoryPool>();
  hipStream_t stream = nullptr;
  {{tensor_decl}}
  // warmup
  for(int i = 0; i < 3; ++i) {
    {{func_call}}
  }
  // run
  KernelTimerImpl timer;
  timer.Start();
  for(int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  timer.End();
  std::cout << "WS:" <<GLOBAL_WORKSPACE_SIZE<<std::endl;
  std::cout << "TIME:" << timer.GetElapsedTime() << std::endl;
}
"""
)

# rendering (messy, need to modularize and organize)

# def gen_profiler(
#     shape_eval_template: jinja2.Template,
#     exec_template: jinja2.Template,
#     tensor_decl_template: jinja2.Template,
#     extra_header_template: jinja2.Template,
#     get_func_signature: Any,
#     extra_code: str = "",
#     func_call_template: jinja2.Template = FUNC_CALL_TEMPLATE,
#     indent: str = "  ",
# ) -> str:
    # shape_eval_template: jinja2.Template
    # exec_template: jinja2.Template
    # tensor_decl_template: jinja2.Template
#extra_header_template: jinja2.Template
get_func_signature: Any
extra_code: str = ""
func_call_template: jinja2.Template = FUNC_CALL_TEMPLATE
indent: str = "  "

    # shape_eval = shape_eval_template.render(rank=2) #if shape_eval_template else ""
    # exe_path = exec_template.render(instance="DeviceInstance",dtype="void",reduce_dims=1,rank=2,eps=eps,)

instances = INSTANCE_TEMPLATE.render(
            name="DeviceInstance", config_name= "ck::tensor_operation::device::DeviceLayernormImpl",)

op_func = FUNC_TEMPLATE.render(
            instances_decl=instances,
            #func_signature=get_func_signature(func_attrs),
            #shape_eval=shape_eval,
            #exec_paths=exe_path,
            #extra_headers=extra_header_template.render(),
            extra_code=extra_code,)

structs_def = STRUCTS_DEF_TEMPLATE.render()
args_parse = ARGS_PARSE_TEMPLATE.render(rank=2)
#tensor_decl = tensor_decl_template.render(rank=2)

input_dim_names = [f"in_{i}" for i in range(2)]
func_call = func_call_template.render(
            func_name="norm",
            input="(void *) memory_pool->RequestHalfTensorByIdx(0)",
            gamma="(void *) memory_pool->RequestHalfTensorByIdx(2)",
            beta="(void *) memory_pool->RequestHalfTensorByIdx(3)",
            output="(void *) memory_pool->RequestHalfTensorByIdx(1)",
            input_dim_names=input_dim_names,
            indent=indent,
)

code = PROFILER_TEMPLATE.render(
            op_func=op_func,
            structs_def=structs_def,
            args_parse=args_parse,
            #tensor_decl=tensor_decl,
            func_call=func_call,
)

# print(instances)
# print(args_parse)
# print(structs_def)
#print(func_call)
#print(op_func)
print(code)


