// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

// E = Relu(C + D0 + D1)
struct AddAddRelu
{
    __host__ __device__ void
    operator()(ck::half_t& e, const ck::half_t& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const ck::half_t x = c + d0 + d1;

        ck::tensor_operation::element_wise::Relu{}.operator()(e, x);
    }
    __host__ __device__ void
    operator()(float& e, const float& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const float x = c + ck::type_convert<float>(d0) + ck::type_convert<float>(d1);

        ck::tensor_operation::element_wise::Relu{}.operator()(e, x);
    }
};

// E = Gelu(C + D0 + D1)
struct AddAddGelu
{
    __host__ __device__ void
    operator()(ck::half_t& e, const ck::half_t& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const ck::half_t x = c + d0 + d1;

        ck::tensor_operation::element_wise::Gelu{}.template operator()<ck::half_t, ck::half_t>(e,
                                                                                               x);
    }

    __host__ __device__ void
    operator()(float& e, const float& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const float x = c + ck::type_convert<float>(d0) + ck::type_convert<float>(d1);

        ck::tensor_operation::element_wise::Gelu{}.template operator()<float, float>(e, x);
    }
};

// E = FastGelu(C + D0 + D1)
struct AddAddFastGelu
{
    __host__ __device__ void
    operator()(float& e, const float& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const float x = c + ck::type_convert<float>(d0) + ck::type_convert<float>(d1);

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(e, x);
    }
};
