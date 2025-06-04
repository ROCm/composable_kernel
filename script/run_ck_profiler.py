def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--shapes_csv", required=True)
    parser.add_argument("--best", action="store_true")
    parser.add_argument("-o", default="out.csv")

    return vars(parser.parse_args())


def tuples(filename):
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


def run_shape(shape):
    import subprocess

    m, n, k = shape
    bin_name = "./bin/ckProfiler"
    op_name = "gemm_multiply_multiply_weight_preshuffle"
    meta_args = map(str, [1, 0, 0, 2, 0, 1])
    shape_args = map(str, [m, n, k, k, k, 0, 0, n])
    control_args = map(str, [1, 50, 10, 4096])

    cmd = [bin_name, op_name, *meta_args, *shape_args, *control_args]
    print(" ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    ).stdout

    return result.splitlines()


def filter_output_line(result_line, best_only):
    if "DeviceGemmXdlUniversal" in result_line:
        if best_only:
            if "Best Perf" in result_line:
                return True
        else:
            if "Best Perf" not in result_line:
                return True
    return False


def write_results(filename, results):
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
    m, n, k = shape
    return metadata | {"M": m, "N": n, "K": k}


def main():
    args = parse_args()
    filename = args["shapes_csv"]
    shapes = tuples(filename)

    all_results = []
    from tqdm import tqdm
    from functools import partial

    for s in tqdm(shapes):
        results_single_shape = map(
            lambda r: add_shape_to_metadata(s, r),
            map(
                parse_result,
                filter(
                    partial(filter_output_line, best_only=args["best"]), run_shape(s)
                ),
            ),
        )
        all_results.extend(list(results_single_shape))

    write_results(args["o"], all_results)


if __name__ == "__main__":
    main()
