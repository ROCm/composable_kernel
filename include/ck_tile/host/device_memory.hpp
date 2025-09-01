// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {
#ifndef KL_MODEL
template <typename T>
__global__ void set_buffer_value(T* p, T x, uint64_t buffer_element_size)
{
    for(uint64_t i = threadIdx.x; i < buffer_element_size; i += blockDim.x)
    {
        p[i] = x;
    }
}

/**
 * @brief Manages device memory allocation and host-device data transfers
 *
 * DeviceMem encapsulates GPU memory management operations using HIP runtime API.
 * It provides functionality for allocating device memory, transferring data between
 * host and device, and performing basic memory operations.
 *
 * Key features:
 * - Automatic memory allocation and deallocation
 * - Host-to-device and device-to-host data transfers
 * - Memory initialization operations
 * - Integration with HostTensor for simplified data handling
 *
 * Usage example:
 * ```
 * // Allocate device memory
 * BHostTensor<float> AHostData({256});
 * DeviceMem d_mem(BHostData.get_element_space_size_in_bytes());
 *
 * // Transfer data to device
 * HostTensor<float> AHostTensor({256});
 * d_mem.ToDevice(AHostData.data());
 *
 * // Retrieve data from device
 * HostTensor<float> ResultHostTensor({256});
 * d_mem.FromDevice(ResultHostTensor.data());
 * ```
 */
struct DeviceMem
{
    DeviceMem(void * kl) : mpDeviceBuf(nullptr), mMemSize(0) {}
    DeviceMem(void * kl, std::size_t mem_size) : mMemSize(mem_size)
    {
        if(mMemSize != 0)
        {
            HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
        }
        else
        {
            mpDeviceBuf = nullptr;
        }
    }
    template <typename T>
    DeviceMem(void * kl, const HostTensor<T>& t) : mMemSize(t.get_element_space_size_in_bytes())
    {
        if(mMemSize != 0)
        {
            HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
        }
        else
        {
            mpDeviceBuf = nullptr;
        }
        ToDevice(t.data());
    }
    void Realloc(std::size_t mem_size)
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipFree(mpDeviceBuf));
        }
        mMemSize = mem_size;
        if(mMemSize != 0)
        {
            HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
        }
        else
        {
            mpDeviceBuf = nullptr;
        }
    }
    void* GetDeviceBuffer() const { return mpDeviceBuf; }
    std::size_t GetBufferSize() const { return mMemSize; }
    void ToDevice(const void* p) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(
                hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
        }
        // else
        // {
        //     throw std::runtime_error("ToDevice with an empty pointer");
        // }
    }
    void ToDevice(const void* p, const std::size_t cpySize) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(
                hipMemcpy(mpDeviceBuf, const_cast<void*>(p), cpySize, hipMemcpyHostToDevice));
        }
    }
    void FromDevice(void* p) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
        }
        // else
        // {
        //     throw std::runtime_error("FromDevice with an empty pointer");
        // }
    }
    void FromDevice(void* p, const std::size_t cpySize) const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipMemcpy(p, mpDeviceBuf, cpySize, hipMemcpyDeviceToHost));
        }
    }

    // construct a host tensor with type T
    template <typename T>
    HostTensor<T> ToHost(std::size_t cpySize)
    {
        // TODO: host tensor could be slightly larger than the device tensor
        // we just copy all data from GPU buffer
        std::size_t host_elements = (cpySize + sizeof(T) - 1) / sizeof(T);
        HostTensor<T> h_({host_elements});
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipMemcpy(h_.data(), mpDeviceBuf, cpySize, hipMemcpyDeviceToHost));
        }
        return h_;
    }
    template <typename T>
    HostTensor<T> ToHost()
    {
        return ToHost<T>(mMemSize);
    }

    void SetZero() const
    {
        if(mpDeviceBuf)
        {
            HIP_CHECK_ERROR(hipMemset(mpDeviceBuf, 0, mMemSize));
        }
    }
    template <typename T>
    void SetValue(T x) const
    {
        if(mpDeviceBuf)
        {
            if(mMemSize % sizeof(T) != 0)
            {
                throw std::runtime_error("wrong! not entire DeviceMem will be set");
            }

            // TODO: call a gpu kernel to set the value (?)
            set_buffer_value<T><<<1, 1024>>>(static_cast<T*>(mpDeviceBuf), x, mMemSize / sizeof(T));
        }
    }
    ~DeviceMem()
    {
        if(mpDeviceBuf)
        {
            try
            {
                HIP_CHECK_ERROR(hipFree(mpDeviceBuf));
            }
            catch(std::runtime_error& re)
            {
                std::cerr << re.what() << std::endl;
            }
        }
    }

    void* mpDeviceBuf;    ///< pointer to device buffer
    std::size_t mMemSize; ///< size of device buffer in bytes
};

#else

#include "cm9_kernel_launch.hpp"
//using namespace MI_KERNEL;

struct DeviceMem
{
    DeviceMem(std::shared_ptr<MI_KERNEL::Kernel_Launcher> _kl) 
        : kl(_kl)
        , mKbuff(0)
        , mMemSize(0) 
    {}
    DeviceMem(std::shared_ptr<MI_KERNEL::Kernel_Launcher> _kl, std::size_t mem_size) 
        : kl(_kl)
        , mMemSize(mem_size)
    {
        if(mMemSize != 0)
        {
            mKbuff = kl->create_buffer(MI_KERNEL::TYPE_READ_WRITE, mMemSize);
        }
    }
    template <typename T>
    DeviceMem(std::shared_ptr<MI_KERNEL::Kernel_Launcher> _kl, const HostTensor<T>& t) 
        : kl(_kl)
        , mMemSize(t.get_element_space_size_in_bytes())
    {
        if(mMemSize != 0)
        {
            mKbuff = kl->create_buffer(MI_KERNEL::TYPE_READ_WRITE, mMemSize);
        }
        ToDevice(t.data());
    }
    void Realloc(std::size_t mem_size)
    {
        if(mKbuff)
        {
            kl->release_buffer(mKbuff);
            mKbuff = 0;
        }
        mMemSize = mem_size;
        if(mMemSize != 0)
        {
            kl->create_buffer(MI_KERNEL::TYPE_READ_WRITE, mMemSize);
        }
    }
    void* GetDeviceBuffer() const { return (void *)(&mKbuff); }
    std::size_t GetBufferSize() const { return mMemSize; }
    void ToDevice(const void* p) const
    {
        if(mKbuff)
        {
            printf("mKbuff, 0x%ld\n", mKbuff);
            KL_CHECK_ERROR(kl->write_buffer(mKbuff,	mMemSize, (void*)p));
        }
    }
    void ToDevice(const void* p, const std::size_t cpySize) const
    {
        if(mKbuff)
        {
            KL_CHECK_ERROR(kl->write_buffer(mKbuff,	cpySize, (void*)p));
        }
    }
    void FromDevice(void* p) const
    {
        if(mKbuff)
        {
			KL_CHECK_ERROR(kl->read_buffer(mKbuff, mMemSize, (void*)p));
        }
    }
    void FromDevice(void* p, const std::size_t cpySize) const
    {
        if(mKbuff)
        {
			KL_CHECK_ERROR(kl->read_buffer(mKbuff, cpySize, (void*)p));
        }
    }

    // construct a host tensor with type T
    template <typename T>
    HostTensor<T> ToHost(std::size_t cpySize)
    {
        // TODO: host tensor could be slightly larger than the device tensor
        // we just copy all data from GPU buffer
        std::size_t host_elements = (cpySize + sizeof(T) - 1) / sizeof(T);
        HostTensor<T> h_({host_elements});
        if(mKbuff)
        {
			KL_CHECK_ERROR(kl->read_buffer(mKbuff, cpySize, (void*)(h_.data())));
        }
        return h_;
    }
    template <typename T>
    HostTensor<T> ToHost()
    {
        return ToHost<T>(mMemSize);
    }

    void SetZero() const
    {
        if(mKbuff)
        {
            void * tmp = malloc(mMemSize);
            memset(tmp, 0, mMemSize);
            KL_CHECK_ERROR(kl->write_buffer(mKbuff,	mMemSize, (void*)tmp));
            free(tmp);
        }
    }
    template <typename T>
    void SetValue(T x) const
    {
        if(mKbuff)
        {
            if(mMemSize % sizeof(T) != 0)
            {
                throw std::runtime_error("wrong! not entire DeviceMem will be set");
            }

            void * tmp = malloc(mMemSize);
            T * tmpT = (T*)tmp;
            for (int i = 0; i < mMemSize % sizeof(T); i++)
                tmpT[i] = x;

            KL_CHECK_ERROR(kl->write_buffer(mKbuff,	mMemSize, (void*)tmp));
            free(tmp);
        }
    }
    ~DeviceMem()
    {
        if(mKbuff)
        {
            try
            {
                kl->release_buffer(mKbuff);
                mKbuff = 0;
            }
            catch(std::runtime_error& re)
            {
                std::cerr << re.what() << std::endl;
            }
        }
    }

    std::size_t mMemSize;
    std::shared_ptr<MI_KERNEL::Kernel_Launcher> kl;
    MI_KERNEL::K_Buffer mKbuff = 0; // PM4_Buf *
};
#endif  // KL_MODEL
} // namespace ck_tile
