// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>

#include "ck/stream_config.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct BaseArgument
{
    BaseArgument()                    = default;
    BaseArgument(const BaseArgument&) = default;
    BaseArgument& operator=(const BaseArgument&) = default;

    virtual ~BaseArgument() {}

    void* p_workspace_ = nullptr;
};

struct BaseInvoker
{
    BaseInvoker()                   = default;
    BaseInvoker(const BaseInvoker&) = default;
    BaseInvoker& operator=(const BaseInvoker&) = default;

    virtual float Run(const BaseArgument*, const StreamConfig& = StreamConfig{})
    {
        return float{0};
    }

    virtual ~BaseInvoker() {}
};

struct BaseParameters
{
    BaseParameters()                   = default;
    BaseParameters(const BaseParameters&) = default;
    BaseParameters& operator=(const BaseParameters&) = default;

    virtual void SetAElementOp(const std::string&) {}

    virtual void SetBElementOp(const std::string&) {}

    virtual void SetCDEElementOp(const std::string&) {}

    virtual void SetDsLayout(const std::string&) {}

    virtual void SetDsDataType(const std::string&) {}

    virtual void SetGemmSpec(const index_t, const index_t, const index_t) {}

    virtual index_t GetGridSize(const index_t, const index_t) 
    {
        return 0;
    }

    virtual index_t GetBlockSize()
    {
        return 0;
    }

    virtual std::string GetParametersString()
    {
        return "";
    }

    virtual ~BaseParameters() {}
};

struct BaseOperator
{
    BaseOperator()                    = default;
    BaseOperator(const BaseOperator&) = default;
    BaseOperator& operator=(const BaseOperator&) = default;

    virtual bool IsSupportedArgument(const BaseArgument*) { return false; }
    virtual std::string GetTypeString() const { return ""; }

    virtual std::string GetTypeIdName() const { return typeid(*this).name(); }

    virtual std::string GetTypeIdHashCode() const
    {
        std::ostringstream oss;

        oss << std::hex << typeid(*this).hash_code();

        return oss.str();
    };

    virtual size_t GetWorkSpaceSize(const BaseArgument*) const { return 0; }

    virtual void SetWorkSpacePointer(BaseArgument* p_arg, void* p_workspace) const
    {
        assert(p_arg);
        p_arg->p_workspace_ = p_workspace;
    }

    virtual ~BaseOperator() {}
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
