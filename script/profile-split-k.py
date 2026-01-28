#!/usr/bin/env python3

import subprocess
import csv
import os

split_k_values = [1, 2, 4, 8, 16, 32, 64, 128]
results = []

tile_example_path = os.path.join("..", "build", "bin", "tile_example_grouped_conv_bwd_weight")

for split_k in split_k_values:
    output_file = f"./profile_split_k_{split_k}.csv"
    
    # Run your GEMM with specific split-K value
    cmd = [
        "rocprofv3",
        #"--input", "split_k_profile.txt",
        "-o", output_file,
        "--",
        tile_example_path,
        f"-split_k={split_k}" 
    ]
    
    print(f"Profiling Split-K = {split_k}...")
    subprocess.run(cmd)
    
    # Parse results
    # with open(output_file, 'r') as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         row['split_k'] = split_k
    #         results.append(row)

# Save combined results
# with open('all_split_k_results.csv', 'w') as f:
#     if results:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)