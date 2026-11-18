// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {

// This structs are fully based on ck tile that already supports TDM

struct TDMLdsPaddingConfig
{
    uint32_t pad_interval{0};
    uint32_t pad_amount{0};
};

// context switch is not supported on gfx1250
struct TDMConfig
{
    // 0 - not a context-restore descriptor. 1 - from a context-restore
    // never set by CK, always 0
    bool is_restore{false};
    // 0 - don't send an atomic barrier op. 1 - send an atomic barrier op.
    bool atomic_barrier_enable{false};
    uint16_t atomic_barrier_address{0}; // only consumed when atomic_barrier_enable
    // 0 - padding disabled. 1 - add padding to lds destination address.
    bool pad_enable{false};
    TDMLdsPaddingConfig pad_config; // padding config
};

namespace detail {

union TDM_GROUP0
{
    __device__ TDM_GROUP0(uintptr_t lds_addr_in, uintptr_t global_addr_in) : bitfield{0, 0, 0, 0}
    {
        count    = 1;
        lds_addr = static_cast<uint32_t>(lds_addr_in);
        globalAddr(global_addr_in);
        type              = 2; // set to 2 for spg
        gather_index_size = 0;
        gather_mode       = 0;
    }

    __device__ void globalAddr(uintptr_t value)
    {
        global_addr_lo = value & 0xFFFFFFFF;
        global_addr_hi = (value >> 32);
    }

    struct
    {
        union
        {
            struct
            {
                uint32_t count : 2;
                uint32_t is_restore : 1;
                uint32_t is_store : 1;
                uint32_t nv : 1;
                uint32_t scope_trait : 2;
                uint32_t th : 3;
                uint32_t reserved_space : 20;
                uint32_t gather_index_size : 1;
                uint32_t gather_mode : 1;
            };
            uint32_t reserved0;
        };
        uint32_t lds_addr;
        uint32_t global_addr_lo;
        union
        {
            struct
            {
                uint32_t global_addr_hi : 25;
                uint32_t reserved2 : 5;
                uint32_t type : 2;
            };
            uint32_t sgpr3;
        };
    };
    int32x4_t bitfield;
};

union TDM_GROUP1
{
    __device__ constexpr TDM_GROUP1() : bitfield{0, 0, 0, 0, 0, 0, 0, 0} {}

    struct
    {
        union
        {
            struct
            {
                uint32_t workgroup_mask : 16;
                uint32_t data_size : 2;
                uint32_t atomic_barrier_enable : 1;
                uint32_t iterate_enable : 1;
                uint32_t pad_enable : 1;
                uint32_t early_timeout : 1;
                uint32_t pad_interval : 3;
                uint32_t pad_amount : 7;
            };
            uint32_t sgpr0;
        };
        union
        {
            struct
            {
                uint32_t atomic_barrier_address : 16;
                uint32_t tensor_dim0_lo : 16;
            };
            uint32_t sgpr1;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim0_hi : 16;
                uint32_t tensor_dim1_lo : 16;
            };
            uint32_t sgpr2;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim1_hi : 16;
                uint32_t tile_dim0 : 16;
            };
            uint32_t sgpr3;
        };
        union
        {
            struct
            {
                uint32_t tile_dim1 : 16;
                uint32_t tile_dim2 : 16;
            };
            uint32_t sgpr4;
        };
        union
        {
            uint32_t tensor_dim0_stride_lo;
            uint32_t sgpr5;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim0_stride_hi : 16;
                uint32_t tensor_dim1_stride_lo : 16;
            };
            uint32_t sgpr6;
        };
        union
        {
            uint32_t tensor_dim1_stride_hi;
            uint32_t sgpr7;
        };
    };
    int32x8_t bitfield;

    __device__ void tensorDim0(uint32_t value)
    {
        tensor_dim0_lo = value & 0xFFFF;
        tensor_dim0_hi = (value >> 16);
    }
    __device__ void tensorDim1(uint32_t value)
    {
        tensor_dim1_lo = value & 0xFFFF;
        tensor_dim1_hi = (value >> 16);
    }
    __device__ void tensorDim0Stride(uint64_t value)
    {
        tensor_dim0_stride_lo = value & 0xFFFFFFFF;
        tensor_dim0_stride_hi = (value >> 32);
    }
    __device__ void tensorDim1Stride(uint64_t value)
    {
        tensor_dim1_stride_lo = value & 0xFFFF;
        tensor_dim1_stride_hi = (value >> 16);
    }
    __device__ void tensorDim(uint32_t index, uint32_t value)
    {
        if(index == 0)
            tensorDim0(value);
        else if(index == 1)
            tensorDim1(value);
    }
    __device__ void tensorDimStride(uint32_t index, uint64_t value)
    {
        if(index == 0)
            tensorDim0Stride(value);
        else if(index == 1)
            tensorDim1Stride(value);
    }
    __device__ void tileDim(uint32_t index, uint16_t value)
    {
        if(index == 0)
            tile_dim0 = value;
        else if(index == 1)
            tile_dim1 = value;
        else if(index == 2)
            tile_dim2 = value;
    }
};

union TDM_GROUP2
{
    __device__ constexpr TDM_GROUP2() : bitfield{0, 0, 0, 0} {}

    struct
    {
        uint32_t tensor_dim2;
        uint32_t tensor_dim3;
        uint32_t tensor_dim2_stride_lo;
        union
        {
            struct
            {
                uint32_t tensor_dim2_stride_hi : 16;
                uint32_t tile_dim3 : 16;
            };
            uint32_t sgpr3;
        };
    };

    __device__ void tensorDim2Stride(uint64_t value)
    {
        tensor_dim2_stride_lo = value & 0xFFFFFFFF;
        tensor_dim2_stride_hi = value >> 32;
    }

    int32x4_t bitfield;
};

union TDM_GROUP3
{
    __device__ constexpr TDM_GROUP3() : bitfield{0, 0, 0, 0} {}

    struct
    {
        uint32_t tensor_dim3_stride_lo;
        union
        {
            struct
            {
                uint32_t tensor_dim3_stride_hi : 16;
                uint32_t tensor_dim_4_lo : 16;
            };
            uint32_t sgpr1;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim_4_hi : 16;
                uint32_t tile_dim4 : 16;
            };
            uint32_t sgpr2;
        };
        uint32_t sgpr3_reserved;
    };

    __device__ void tensorDim3Stride(uint64_t value)
    {
        tensor_dim3_stride_lo = value & 0xFFFFFFFF;
        tensor_dim3_stride_hi = value >> 32;
    }

    __device__ void tensorDim4(uint32_t value)
    {
        tensor_dim_4_lo = value & 0xFFFF;
        tensor_dim_4_hi = value >> 16;
    }

    int32x4_t bitfield;
};

struct TDMDescriptorPack
{
    int32x4_t g0;
    int32x8_t g1;
    int32x4_t g2;
    int32x4_t g3;
    int32x8_t g4;
};

template <typename DataType, index_t TensorRank>
__device__ TDMDescriptorPack MakeTDMDescriptorPack(const void* global_address,
                                                   void* local_address,
                                                   const uint32_t* tensor_dims,
                                                   const uint64_t* global_strides,
                                                   const uint16_t* box_dims,
                                                   const TDMConfig& tdm_config)
{
    TDM_GROUP0 group0{reinterpret_cast<uintptr_t>(local_address),
                      reinterpret_cast<uintptr_t>(global_address)};

    TDM_GROUP1 group1;
    group1.workgroup_mask = 0; // no cluster multicast
    group1.data_size =
        sizeof(DataType) == 8 ? 3 : (sizeof(DataType) == 4 ? 2 : (sizeof(DataType) == 2 ? 1 : 0));
    group1.atomic_barrier_enable  = tdm_config.atomic_barrier_enable;
    group1.atomic_barrier_address = tdm_config.atomic_barrier_address;
    group1.iterate_enable         = 0;
    group1.pad_enable             = tdm_config.pad_enable;
    group1.early_timeout          = 0;
    group1.pad_interval           = tdm_config.pad_config.pad_interval;
    group1.pad_amount             = tdm_config.pad_config.pad_amount;

    static_for<0, 2, 1>{}([&](auto i) {
        if constexpr(i < TensorRank)
        {
            group1.tensorDim(i, tensor_dims[i]);
            group1.tensorDimStride(i, global_strides[i]);
        }
    });

    static_for<0, 3, 1>{}([&](auto i) {
        if constexpr(i < TensorRank)
        {
            group1.tileDim(i, box_dims[i]);
        }
    });

    TDM_GROUP2 group2;
    if constexpr(TensorRank > 2)
    {
        group2.tensor_dim2 = tensor_dims[2];
        group2.tensorDim2Stride(global_strides[2]);
    }
    if constexpr(TensorRank > 3)
    {
        group2.tensor_dim3 = tensor_dims[3];
        group2.tile_dim3   = box_dims[3];
    }

    TDM_GROUP3 group3;
    if constexpr(TensorRank > 3)
    {
        group3.tensorDim3Stride(global_strides[3]);
    }
    if constexpr(TensorRank > 4)
    {
        group3.tensorDim4(tensor_dims[4]);
        group3.tile_dim4 = box_dims[4];
    }

    int32x8_t zeros = {};
    return TDMDescriptorPack{
        group0.bitfield, group1.bitfield, group2.bitfield, group3.bitfield, zeros};
}

} // namespace detail

} // namespace ck
