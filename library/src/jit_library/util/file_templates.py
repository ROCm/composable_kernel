out_file_with_quant = """// SPDX-License-Identifier: MIT 
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

namespace ck {{
namespace host {{
namespace instance {{

struct {op_name}_instances
{{
    static inline std::vector<std::string> {col_row_name}  = 
    {{
{col_row_instances}
    }};

    static inline std::vector<std::string> {col_col_name}  = 
    {{
{col_col_instances}
    }};

    static inline std::vector<std::string> {row_row_name}  = 
    {{
{row_row_instances}
    }};

    static inline std::vector<std::string> {row_col_name}  = 
    {{
{row_col_instances}
    }};

    static inline std::vector<std::string> {int8_col_row_name}  = 
    {{
{int8_col_row_instances}
    }};

    static inline std::vector<std::string> {int8_col_col_name}  = 
    {{
{int8_col_col_instances}
    }};

    static inline std::vector<std::string> {int8_row_row_name}  = 
    {{
{int8_row_row_instances}
    }};

    static inline std::vector<std::string> {int8_row_col_name}  = 
    {{
{int8_row_col_instances}
    }};

    static auto get_col_row_instances(const bool quantize)
    {{
        return quantize ? {int8_col_row_name} : 
                          {col_row_name};
    }}

    static auto get_col_col_instances(const bool quantize)
    {{
        return quantize ? {int8_col_col_name} :
                          {col_col_name};
    }}

    static auto get_row_row_instances(const bool quantize)
    {{
        return quantize ? {int8_row_row_name} : 
                          {row_row_name};
    }}

    static auto get_row_col_instances(const bool quantize)
    {{
        return quantize ? {int8_row_col_name} :
                          {row_col_name};
    }}

    static auto get_include_header()
    {{
        return "{include_header}";
    }}
}};

}} // namespace instance
}} // namespace host
}} // namespace ck
"""

out_file_no_quant = """// SPDX-License-Identifier: MIT 
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

namespace ck {{
namespace host {{
namespace instance {{

struct {op_name}_instances
{{
    static inline std::vector<std::string> {instances_name}  = 
    {{
{instances}
    }};

    static auto get_instances()
    {{
        return {instances_name};
    }}

    static auto get_include_header()
    {{
        return "{include_header}";
    }}
}};

}} // namespace instance
}} // namespace host
}} // namespace ck
"""


def get_device_gemm_multiple_d_file(op_name,
                                col_row_name,
                                col_row_instances,
                                col_col_name,
                                col_col_instances,
                                row_row_name,
                                row_row_instances,
                                row_col_name,
                                row_col_instances,
                                int8_col_row_name,
                                int8_col_row_instances,
                                int8_col_col_name,
                                int8_col_col_instances,
                                int8_row_row_name,
                                int8_row_row_instances,
                                int8_row_col_name,
                                int8_row_col_instances,
                                include_header):
    return out_file_with_quant.format(
            op_name=op_name,
            col_row_name=col_row_name,
            col_row_instances=col_row_instances,
            col_col_name=col_col_name,
            col_col_instances=col_col_instances,
            row_row_name=row_row_name,
            row_row_instances=row_row_instances,
            row_col_name=row_col_name,
            row_col_instances=row_col_instances,
            int8_col_row_name=int8_col_row_name,
            int8_col_row_instances=int8_col_row_instances,
            int8_col_col_name=int8_col_col_name,
            int8_col_col_instances=int8_col_col_instances,
            int8_row_row_name=int8_row_row_name,
            int8_row_row_instances=int8_row_row_instances,
            int8_row_col_name=int8_row_col_name,
            int8_row_col_instances=int8_row_col_instances,
            include_header=include_header)

def get_device_gemm_softmax_gemm_file(op_name,
                                instances_name,
                                instances,
                                include_header):
    return out_file_no_quant.format(
            op_name=op_name,
            instances_name=instances_name,
            instances=instances,
            include_header=include_header)



