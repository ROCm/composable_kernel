import argparse, re, json, os

out_file = """// SPDX-License-Identifier: MIT 
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

namespace ck {{
namespace tensor_operation {{
namespace device {{
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
}} // namespace device
}} // namespace tensor_operation
}} // namespace ck
"""

def strip_sequences(str):
    matches = re.findall(r'S<\d+(?:,\s*\d+)*>', str)
    for match in matches:
        str = str.replace(match, match.replace(' ', ''))
    str = str.replace('S<', "ck::Sequence<")
    
    return str

def remove_commas_and_brackets(string):
    regex_matches = re.findall(r'ck::Sequence<.*?>', string)
    for match in regex_matches:
        string = string.replace(match, match.replace(',', '|').replace('<', '%').replace('>', '$'))
    
    string = string.replace(',', '').replace('<', '').replace('>', '')
    for match in regex_matches:
        string = string.replace(match.replace(',', '|').replace('<', '%').replace('>', '$'), match)

    return string

def get_int8_instances(src, file, template_name):
    aliases = {"Empty_Tuple": "ck::Tuple<>",
               "Row": "ck::tensor_layout::gemm::RowMajor",
               "Col": "ck::tensor_layout::gemm::ColumnMajor",
               "OutElementOp": "PassThrough"}
    instances = {"row_row": [],
                 "row_col": [],
                 "col_col": [],
                 "col_row": [],
                 "row_row_name": [],
                 "row_col_name": [],
                 "col_col_name": [],
                 "col_row_name": []}
    path = src + file
    with open(path) as f:
        for line in f:
            if "impl" in line:
                include_header = line.replace("#include \"", "").replace("\"", "").replace("\n", "")
            elif "using" in line:
                if bool(re.search(".*mk.*kn.*", line)):
                    instances["row_row_name"] = re.search("device_gemm.*instance", line).group()
                elif bool(re.search(".*mk.*nk.*", line)):
                    instances["row_col_name"] = re.search("device_gemm.*instance", line).group()
                elif bool(re.search(".*km.*nk.*", line)):
                    instances["col_col_name"] = re.search("device_gemm.*instance", line).group()
                elif bool(re.search(".*km.*kn.*", line)):
                    instances["col_row_name"] = re.search("device_gemm.*instance", line).group()
                
            elif template_name in line:
                # Turn all whitespace into single spaces
                new_line = " ".join(line.split())
                # Remove whitespace from S<*>
                new_line = strip_sequences(new_line)
                new_line = remove_commas_and_brackets(new_line)
                last_char = "\n"
                if new_line[-1] == ",":
                    last_char = ",\n"
                    new_line = new_line[:-1]
                new_line = '        "ck::tensor_operation::device::' + new_line + '",'
                versions = []
                for key in aliases:
                    new_line = new_line.replace(key, aliases[key])
                
                versions.append(new_line.replace("GemmPipeline", "ck::PipelineVersion::v1").replace("GemmLoopScheduler", "ck::LoopScheduler::Default"))
                versions.append(new_line.replace("GemmPipeline", "ck::PipelineVersion::v1").replace("GemmLoopScheduler", "ck::LoopScheduler::Interwave"))
                versions.append(new_line.replace("GemmPipeline", "ck::PipelineVersion::v2").replace("GemmLoopScheduler", "ck::LoopScheduler::Default"))
                if "ck::tensor_layout::gemm::RowMajor ck::tensor_layout::gemm::RowMajor" in new_line:
                    instances["row_row"].extend(versions)
                elif "ck::tensor_layout::gemm::RowMajor ck::tensor_layout::gemm::ColumnMajor" in new_line:
                    instances["row_col"].extend(versions)
                elif "ck::tensor_layout::gemm::ColumnMajor ck::tensor_layout::gemm::ColumnMajor" in new_line:
                    instances["col_col"].extend(versions)
                elif "ck::tensor_layout::gemm::ColumnMajor ck::tensor_layout::gemm::RowMajor" in new_line:
                    instances["col_row"].extend(versions)
    
    instances["row_row"][-1] = instances["row_row"][-1][:-1]
    instances["row_col"][-1] = instances["row_col"][-1][:-1]
    instances["col_col"][-1] = instances["col_col"][-1][:-1]
    instances["col_row"][-1] = instances["col_row"][-1][:-1]
    return instances

def parse_instances(source):
    out_dir = os.path.join(source, "../../../src/jit_library/solution_instances")
    aliases = {"F16_F16_Tuple": "ck::Tuple<F16,F16>",
               "Row_Row_Tuple": "ck::Tuple<Row,Row>",
               "Empty_Tuple": "ck::Tuple<>",
               "LoopScheduler": "ck::LoopScheduler",
               "PipelineVersion": "ck::PipelineVersion",
               "Row": "ck::tensor_layout::gemm::RowMajor",
               "Col": "ck::tensor_layout::gemm::ColumnMajor",
               "F16": "ck::half_t",
               "F32": "float",
               "OutElementOp": "PassThrough"}
    device_ops = {"gemm_add_add_fastgelu": "DeviceGemmMultipleD_Xdl_CShuffle",
                  #"batched_gemm_softmax_gemm": "DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle"
                 }
    
    for root_, dirs_, files_ in os.walk(source):
        for dir in dirs_:
            op_name = os.path.split(dir)[-1]
            if op_name not in device_ops:
                continue
            col_row_name = ""
            col_col_name = ""
            row_row_name = ""
            row_col_name = ""
            row_row_instances = [] 
            col_row_instances = []
            row_col_instances = []
            col_col_instances = []
            for root, dirs, files in os.walk(os.path.join(root_, dir)):
                for file in files:
                    if not file.endswith(".cpp"):
                        continue;
                    file_name = os.path.split(file)[-1]
                    is_row_row = bool(re.search(".*mk.*kn.*", file_name))
                    is_col_row = bool(re.search(".*km.*kn.*", file_name))
                    is_row_col = bool(re.search(".*mk.*nk.*", file_name))
                    is_col_col = bool(re.search(".*km.*nk.*", file_name))
                    if is_row_row:
                        row_row_name = file_name[:-4]
                    if is_col_row:
                        col_row_name = file_name[:-4]
                    if is_row_col:
                        row_col_name = file_name[:-4]
                    if is_col_col:
                        col_col_name = file_name[:-4]
                    instances_list = []
                    template_name = device_ops[op_name]
                    include_header = ""
                    with open(os.path.join(root, file)) as f:
                        for line in f:
                            if "impl" in line:
                                include_header = line.replace("#include \"", "").replace("\"", "").replace("\n", "")
                            elif template_name in line:
                                # Turn all whitespace into single spaces
                                new_line = " ".join(line.split())
                                # Remove whitespace from S<*>
                                new_line = strip_sequences(new_line)
                                new_line = remove_commas_and_brackets(new_line)
                                last_char = "\n"
                                if new_line[-1] == ",":
                                    last_char = ",\n"
                                    new_line = new_line[:-1]
                                new_line = '        "ck::tensor_operation::device::' + new_line + '",'
                                for key in aliases:
                                    new_line = new_line.replace(key, aliases[key])
                                instances_list.append(new_line)
                    instances_list[-1] = instances_list[-1][:-1]
                    if is_row_row:
                        row_row_instances = instances_list
                    if is_col_row:
                        col_row_instances = instances_list
                    if is_row_col:
                        row_col_instances = instances_list
                    if is_col_col:
                        col_col_instances = instances_list
                out_file_name = op_name + "_instances.hpp"
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                int8_file = "/quantization/gemm/device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_instance.hpp"
                int8_instances = get_int8_instances(source, int8_file, "DeviceGemmMultipleD_Xdl_CShuffle")
                with open(os.path.join(out_dir, out_file_name), "w+") as f:
                    f.write(out_file.format(op_name=op_name,
                                col_row_name=col_row_name,
                                col_row_instances="\n".join(col_row_instances),
                                col_col_name=col_col_name,
                                col_col_instances="\n".join(col_col_instances),
                                row_row_name=row_row_name,
                                row_row_instances="\n".join(row_row_instances),
                                row_col_name=row_col_name,
                                row_col_instances="\n".join(row_col_instances),
                                int8_col_row_name=int8_instances["col_row_name"],
                                int8_col_row_instances="\n".join(int8_instances["col_row"]),
                                int8_col_col_name=int8_instances["col_col_name"],
                                int8_col_col_instances="\n".join(int8_instances["col_col"]),
                                int8_row_row_name=int8_instances["row_row_name"],
                                int8_row_row_instances="\n".join(int8_instances["row_row"]),
                                int8_row_col_name=int8_instances["row_col_name"],
                                int8_row_col_instances="\n".join(int8_instances["row_col"]),
                                include_header=include_header))

def run():
    source = "/code/composable_kernel/library/src/tensor_operation_instance/gpu"
    parse_instances(source)

if __name__ == '__main__':
    run()