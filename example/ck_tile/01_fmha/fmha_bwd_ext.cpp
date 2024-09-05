// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <stdio.h>
#include <vector>
#include <functional>
#include <iostream>
#include <string>
#include "fmha_bwd_ext.hpp"
#include "hsaco/fmha_hsaco.h"

int fmha_bwd_ext(fmha_bwd_ext_traits fmha_ext_traits, fmha_bwd_asm_args args, hipStream_t stream)
{
    hipEvent_t evt_00, evt_11;

    HIP_CALL(hipSetDevice(0));
    // fmha_bwd_ext_kernel impl(HSACO, HSA_KERNEL);
    fmha_bwd_ext_kernel impl(HSA_KERNEL, bwd_fp16_a32);

    int b           = fmha_ext_traits.b;
    int h           = fmha_ext_traits.h;
    int s           = fmha_ext_traits.s;
    int d           = fmha_ext_traits.d;
    int atm_f32     = fmha_ext_traits.atm_f32;
    int skip_dq_rd  = fmha_ext_traits.skip_dq_rd;
    // int ts_qo       = fmha_ext_traits.ts_qo;
    // int ts_kv       = fmha_ext_traits.ts_kv;
    int dump_result = fmha_ext_traits.dump_result;

    std::cout << "b:" << b << std::endl;
    std::cout << "h:" << h << std::endl;
    std::cout << "s:" << s << std::endl;
    std::cout << "d:" << d << std::endl;
    std::cout << "dump_result:" << dump_result << std::endl;
    std::cout << "atm_f32:" << atm_f32 << std::endl;
    std::cout << "skip_dq_rd:" << skip_dq_rd << std::endl;

    size_t arg_size = sizeof(args);
    printf("argsize: %zu\n", arg_size);

#ifdef ASM_PRINT
    int max_i = 256;
    HIP_CALL(hipMemcpy(host_print, print, 8 * max_i, hipMemcpyDeviceToHost));
    for(int i = 0; i < max_i; i++)
    {
        if(((uint32_t*)host_print)[2 * i + 1] != 0x5c005c00)
            printf("Thread%d, PrintVal:0x%x\n",
                   ((int*)host_print)[2 * i],
                   ((uint32_t*)host_print)[2 * i + 1]);
        // std::cout<<"Thread"<<((int*) host_print)[2*i]<<",
        // PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
        //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
    }
#endif

    HIP_CALL(hipEventCreate(&evt_00));
    HIP_CALL(hipEventCreate(&evt_11));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipEventRecord(evt_00, NULL));

    impl.launch_kernel(fmha_ext_traits, args, stream);

    std::cout << "we are done" << std::endl;
    float elapsed_ms;
    HIP_CALL(hipEventRecord(evt_11, NULL));
    HIP_CALL(hipEventSynchronize(evt_11));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
    HIP_CALL(hipEventDestroy(evt_00));
    HIP_CALL(hipEventDestroy(evt_11));

    // float time_per_loop = elapsed_ms / total_loop;
    // float gflops        = static_cast<float>(2.0) * 5 * b * h * d * s * s / time_per_loop / (1e6);
    // printf("b:%d,h:%d,s:%d,d:%d, time: %.3f, gflops:%.3f\n", b, h, s, d, time_per_loop, gflops);
    printf("b:%d,h:%d,s:%d,d:%d\n", b, h, s, d);

#ifdef ASM_PRINT
    free(host_print);
    HIP_CALL(hipFree(print));
#endif
    // printf("CU:%d, TIPS:%.3f(2x:%.3f, 4x:%.3f), cost:%fms per loop\n", num_cu, tips, 2*tips,
    // 4*tips, time_per_loop);

    return 0;
}
