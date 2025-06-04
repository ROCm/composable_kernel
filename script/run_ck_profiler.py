#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-

def parse_args():
    """
    Parse command-line arguments
    -   --shapes_csv : input csv file with M, N, K integer columns
    -   --best       : if True, store only the result reported by the best instance.
                       if False, store results from all instances
    -   -o           : output csv file
    -   --build_dir  : path to directory where CMake stores all the build artifacts.
                       The profiler binary is bin/ckProfiler relative to this directory.
    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--shapes_csv", required=True)
    parser.add_argument("--best", action="store_true")
    parser.add_argument("-o", default="out.csv")
    parser.add_argument("--build_dir", default=".")

    return vars(parser.parse_args())


def tuples(filename):
    """
    Parse M, N, K integers from the input csv file
    """
    lines = []
    with open(filename, "r", newline="") as f:
        import csv

        reader = csv.reader(f)
        for line in reader:
            try:
                m, n, k = map(int, line)
                lines.append((m, n, k))
            except:
                pass
    return lines


def parse_result(line):
    """
    Parse the ckProfiler stdout line.
    Result: a dict with the instance metadata and performance results
    """
    words = line.split()
    fields = dict()
    if "Perf:" in words or "Perf" in words:
        for key in ("ms", "TFlops", "GB/s"):
            fields[key] = words[words.index(key + ",") - 1]
    for key in (
        "BlkSize:",
        "BlkTile:",
        "WaveTile:",
        "WaveMap:",
        "VmemReadVec:",
        "BlkGemmPipelineScheduler:",
        "BlkGemmPipelineVersion:",
        "BlkGemmPipelinePrefetchStages:",
    ):
        fields[key.strip(":")] = words[words.index(key) + 1].strip(",")
    if "KBatch" in words:
        key = "KBatch"
        fields[key] = words[words.index(key) + 1]

    return fields


def run_shape(shape, profiler_bin):
    """
    Launch ckProfiler in subprocess and collect its stdout
    """
    import subprocess

    m, n, k = shape
    op_name = "gemm_multiply_multiply_weight_preshuffle"
    meta_args = map(str, [1, 0, 0, 2, 0, 1])
    shape_args = map(str, [m, n, k, k, k, 0, 0, n])
    control_args = map(str, [1, 50, 10, 4096])

    cmd = [profiler_bin, op_name, *meta_args, *shape_args, *control_args]
    print(" ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    ).stdout

    return result.splitlines()


def filter_output_line(result_line, best_only):
    """
    Filter out ckProfiler output lines which don't report performance results
    """
    if "DeviceGemmXdlUniversal" in result_line:
        if best_only:
            if "Best Perf" in result_line:
                return True
        else:
            if "Best Perf" not in result_line:
                return True
    return False


def write_results(filename, results):
    """
    Write out the performance results to a csv file
    """
    if not results:
        return
    with open(filename, "w", newline="") as f:
        import csv

        fields = list(results[0].keys())
        writer = csv.DictWriter(f, dialect="unix", fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def add_shape_to_metadata(shape, metadata):
    """
    Adds M, N, K to the parsed profiler results
    """
    m, n, k = shape
    return metadata | {"M": m, "N": n, "K": k}


def main():
    """
    Main driver: 
    - parses command line arguments
    - parses input shapes to run ckProfiler with
    - for each shape,
       - runs ckProfiler
       - parses the ckProfiler output
    - writes out the results for all shapes 
    """
    args = parse_args()
    filename = args["shapes_csv"]
    shapes = tuples(filename)

    all_results = []
    from tqdm import tqdm
    from functools import partial
    from os import path

    profiler_bin = path.join(args["build_dir"], "bin", "ckProfiler")

    for s in tqdm(shapes):
        run_shape_stdout_lines = run_shape(s, profiler_bin)
        results_single_shape = map(
            lambda r: add_shape_to_metadata(s, r),
            map(
                parse_result,
                filter(
                    partial(filter_output_line, best_only=args["best"]), run_shape_stdout_lines
                ),
            ),
        )
        all_results.extend(list(results_single_shape))

    write_results(args["o"], all_results)


if __name__ == "__main__":
    main()
