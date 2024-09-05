// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HSACO "kernel.co"
#define HSA_KERNEL "kernel_func"

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;

    unsigned int _p1;
};
struct __attribute__((packed)) fmha_bwd_asm_args
{
    void* ptr_dq;
    p2 _p0;
    void* ptr_dk;
    p2 _p1;
    void* ptr_dv;
    p2 _p2;
    void* ptr_q;
    p2 _p3;
    void* ptr_k;
    p2 _p4;
    void* ptr_v;
    p2 _p5;
    void* ptr_do;
    p2 _p6;
    void* ptr_lse;
    p2 _p7;
    void* ptr_odo;
    p2 _p8;
    float scalar;
    p3 _p9;
    float log2e;
    p3 _p10;
    unsigned int seq_len;
    p3 _p11;
    unsigned int Ts;
    p3 _p12;
    unsigned int Hs;
    p3 _p13;
    unsigned int BAs;
    p3 _p14;
#ifdef ASM_PRINT
    void* print;
#endif
};

#define HIP_CALL(call)                                                              \
    do                                                                              \
    {                                                                               \
        hipError_t err = call;                                                      \
        if(err != hipSuccess)                                                       \
        {                                                                           \
            printf("[hiperror](%d) fail to call %s", static_cast<int>(err), #call); \
            exit(0);                                                                \
        }                                                                           \
    } while(0)

struct fmha_bwd_ext_traits
{
    int b;
    int h;
    int s;
    int d;

    int atm_f32;
    int skip_dq_rd;
    int mask;
    int mask_kb;
    int ts_qo;
    int ts_kv;
    int dump_result;
};


class fmha_bwd_ext_kernel
{
    public:
    // fmha_bwd_ext_kernel(const char* image, const std::string& name)
    // {
    //     HIP_CALL(hipModuleLoad(&module, image));
    //     HIP_CALL(hipModuleGetFunction(&kernel_func, module, name.c_str()));
    // }

    fmha_bwd_ext_kernel(const std::string& name, unsigned char buffer[])
    {
        HIP_CALL(hipModuleLoadData(&module, buffer));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name.c_str()));
    }

    void
    launch_kernel(fmha_bwd_ext_traits fmha_ext_traits, fmha_bwd_asm_args args, hipStream_t stream)
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = fmha_ext_traits.s / fmha_ext_traits.ts_kv;
        int gdy = fmha_ext_traits.h;
        int gdz = fmha_ext_traits.b;
        if((fmha_ext_traits.mask == 1) && (fmha_ext_traits.mask_kb == 1))
        {
            int num_tg = fmha_ext_traits.s / fmha_ext_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       stream,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }

    private:
    hipModule_t module;
    hipFunction_t kernel_func;
};

int fmha_bwd_ext(fmha_bwd_ext_traits fmha_ext_traits, fmha_bwd_asm_args args, hipStream_t stream);
// int fmha_bwd_ext(fmha_bwd_ext_traits fmha_ext_traits);
