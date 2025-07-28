#!/usr/bin/env python3

import os
import argparse
import sys
import pandas as pd
import csv
import matplotlib
from collections import defaultdict
import numpy as np

matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze convolution test results.")
    parser.add_argument("--csv-file", type=str, dest="csv_file", required=True, help="Path to the CSV file containing test cases.")
    parser.add_argument("--output-dir", type=str, dest="output_dir", required=True, help="Directory to save output plots.")
    parser.add_argument("--label", type=str, dest="label", default="", help="Label for the figure names.")
    parser.add_argument("--old-format", action="store_true", dest="old_format", default=False, help="Old format of the CSV files")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def calculate_ranking_numbers(best_split_k_ranks, num_ops):
  """Calculate ranking numbers based on best split-k ranks and number of operations."""
  best_split_k_ranking_numbers = []
  for i in range(len(best_split_k_ranks)):
      rank = int(best_split_k_ranks.iloc[i])
      total_ops = int(num_ops.iloc[i])
      ranking = 100.0 * (total_ops - rank + 1) / total_ops
      best_split_k_ranking_numbers.append(ranking)

  return best_split_k_ranking_numbers

def plot_ranking_histogram(best_split_k_ranking_numbers, file_name, explanation):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figure(figsize=(10, 6))
    plt.hist(best_split_k_ranking_numbers, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Optimized Split-K Ranking Numbers')
    plt.xlabel('Ranking (%)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0.05, 0.8, explanation, transform=plt.gca().transAxes, fontsize=9,
          verticalalignment='bottom', bbox=props)
    plt.savefig(file_name)

def plot_local_ranking_bar_chart(best_split_k_ranking_numbers, file_name, explanation):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # Count the occurrences of each ranking
    rankings_count = {}
    for ranking in best_split_k_ranking_numbers:
        rankings_count[ranking] = rankings_count.get(ranking, 0) + 1
    
    # Ensure all ranks 1-9 are represented
    max_rank = 9
    all_ranks = list(range(1, max_rank+1))  # Ranks 1 through 9
    
    # Create a list of counts, with 0 for missing ranks
    counts = [rankings_count.get(rank, 0) for rank in all_ranks]
    
    # Check that there are not other ranks than 1-9
    if any(rank < 1 or rank > max_rank for rank in rankings_count.keys()):
        raise f"Error: Found ranks outside the range 1-9:"

    plt.figure(figsize=(10, 6))
    
    # Create bar chart with consistent coloring
    bars = plt.bar(
        all_ranks,              # X positions (1-9)
        counts,                 # Heights (frequencies)
        color='skyblue',
        edgecolor='black',
        alpha=0.7,
        width=0.6
    )
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add labels for non-zero bars
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{int(height)}',
                ha='center', 
                va='bottom',
                fontweight='bold'
            )
    
    # Set x-tick positions and labels
    plt.xticks(
        all_ranks,              # Positions (1-9)
        [f"{rank}" for rank in all_ranks],  # Labels
        fontsize=11
    )
    
    # Add labels and title
    plt.title('Distribution of Optimal Split-K Rankings', fontsize=14, fontweight='bold')
    plt.xlabel('Ranking (1=Best, 9=Worst)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')  # Grid lines only on y-axis
    
    # Add explanation text
    plt.text(0.2, 0.85, explanation, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    # Add statistics
    total_instances = sum(counts)
    stats_text = (f"Total instances: {total_instances}\n"
                 f"Best performing (Rank 1): {counts[0]} ({counts[0]/total_instances:.1%})\n"
                 f"Worst performing (Rank 9): {counts[7]} ({counts[8]/total_instances:.1%})")
    
    plt.text(0.65, 0.675, stats_text, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(file_name)

def plot_local_performance_histogram(local_performance, file_name, explanation):
    import numpy as np
    mean_val = np.mean(local_performance)
    median_val = np.median(local_performance)
    std_val = np.std(local_performance)
    min_val = np.min(local_performance)
    max_val = np.max(local_performance)
    count = len(local_performance)
    
    # Create statistics text
    stats_text = (f"Statistics:\n"
                  f"Count: {count}\n"
                  f"Mean: {mean_val:.2f}%\n"
                  f"Median: {median_val:.2f}%\n"
                  f"Std Dev: {std_val:.2f}%\n"
                  f"Min: {min_val:.2f}%\n"
                  f"Max: {max_val:.2f}%")
    
    # Create figure and plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(local_performance, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Local Performance of Split-K Values')
    plt.xlabel('Performance (%)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanation text box (on the left)
    plt.text(0.05, 0.85, explanation, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add statistics text box (on the right)
    plt.text(0.05, 0.55, stats_text, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Save figure
    plt.savefig(file_name)
    plt.close()

def plot_best_split_k_values(standard_counts, optimized_count, 
                            standard_equal_optimized_counts, suffix, args):
    # Prepare data for plotting
    categories = list(standard_counts.keys()) + ['Optimized Split-K']

    # Calculate total counts (standard counts + cases where standard equals optimized)
    total_standard_counts = []
    equal_counts = []

    # First, collect data for all standard values
    for key in standard_counts.keys():
        # Get the count where standard equals optimized (default to 0 if key doesn't exist)
        equal_count = standard_equal_optimized_counts.get(key, 0)
        equal_counts.append(equal_count)
        
        # Total is the standard count
        total_standard_counts.append(standard_counts[key] + equal_count)

    # Add the optimized count as the last category
    total_counts = total_standard_counts + [optimized_count]
    equal_counts.append(0)  # No "equals optimized" for the optimized category itself

    # Calculate the "non-equal" portion (what will show at the bottom of each stack)
    non_equal_counts = [total - equal for total, equal in zip(total_counts, equal_counts)]

    # Create figure
    plt.figure(figsize=(14, 7))

    # Create the base bars (non-equal counts)
    base_bars = plt.bar(
        range(len(categories)),  # X positions
        non_equal_counts,        # Heights (counts without the "equals optimized" portion)
        color='skyblue',         # Base color
        edgecolor='black',
        alpha=0.8,
        width=0.6,
        label='Standard Split-K (1,2,4,8,16,32,64,128)'
    )

    # Create the stacked bars for the "equals optimized" portion
    equal_bars = plt.bar(
        range(len(categories)),  # X positions
        equal_counts,            # Heights (just the "equals optimized" counts)
        bottom=non_equal_counts, # Start these bars where the base bars end
        color='orange',          # Different color to highlight this portion
        edgecolor='black',
        alpha=0.8,
        width=0.6,
        label='Standard = Optimized'
    )

    # Add value labels for total height of each bar
    for i, (total, equal) in enumerate(zip(total_counts, equal_counts)):
        if total > 0:  # Only add label if there's a value
            # Position the text at the top of the stacked bar
            plt.text(
                i,                      # X position (bar index)
                total + 0.5,            # Y position (just above the top)
                f'{int(total)}',        # Total count as text
                ha='center', 
                va='bottom',
                fontweight='bold'
            )
            
            # If there's a significant "equals optimized" portion, add a label inside that section
            if equal > 5:  # Only add for larger values to avoid clutter
                plt.text(
                    i,                      # X position (bar index)
                    non_equal_counts[i] + equal/2,  # Y position (middle of orange section)
                    f'{int(equal)}',        # Equal count as text
                    ha='center', 
                    va='center',
                    fontweight='bold',
                    color='black'
                )

    base_bars[-1].set_color('green')
    base_bars[-1].set_label('Optimized Split-K')

    plt.xticks(
        range(len(categories)),  
        categories,              
        rotation=45 if len(categories) > 8 else 0,  
        fontsize=11,
        ha='right' if len(categories) > 8 else 'center'  
    )

    plt.title('Best Split-K Values', fontsize=16, fontweight='bold')
    plt.xlabel('Split-K Value', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')  
    plt.legend(fontsize=12)

    explanation = "Orange sections represent cases where optimized\nsplit-K equals to one of the fixed split-K values"
    plt.text(
        0.02, 0.95,                     
        explanation,
        transform=plt.gca().transAxes,  
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    plt.tight_layout()

    split_k_distribution_path = os.path.join(args.output_dir, f'best_split_k_values{suffix}.png')
    plt.savefig(split_k_distribution_path)
    print(f"Saved best split-K values chart to: {split_k_distribution_path}")
    
    plt.close()

def plot_perf(perf_difference, output_dir, suffix="", op_name="", label=""):
    """Plot the performance differences as a histogram with statistics."""
    import numpy as np
    
    mean_val = np.mean(perf_difference)
    median_val = np.median(perf_difference)
    std_val = np.std(perf_difference)
    min_val = np.min(perf_difference)
    max_val = np.max(perf_difference)
    p25 = np.percentile(perf_difference, 25)
    p75 = np.percentile(perf_difference, 75)
    count = len(perf_difference)
    
    min_edge = np.floor(min_val / 5) * 5
    max_edge = np.ceil(max_val / 5) * 5
    bin_edges = np.arange(min_edge, max_edge + 5, 5)
    
    plt.figure(figsize=(12, 6))
    
    below_100 = [x for x in perf_difference if x < 100]
    above_100 = [x for x in perf_difference if x >= 100]
    
    if below_100:
        counts_below, _ = np.histogram(below_100, bins=bin_edges)
    else:
        counts_below = np.zeros(len(bin_edges) - 1)
        
    if above_100:
        counts_above, _ = np.histogram(above_100, bins=bin_edges)
    else:
        counts_above = np.zeros(len(bin_edges) - 1)
    
    if below_100:
        plt.hist(below_100, bins=bin_edges, color='red', 
                alpha=0.7, edgecolor='black', label='Below 100%')
    
    if above_100:
        plt.hist(above_100, bins=bin_edges, color='green', 
                alpha=0.7, edgecolor='black', label='Above 100%')
    
    total_counts = counts_below + counts_above
    
    for i in range(len(bin_edges) - 1):
        if total_counts[i] > 0:
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            
            plt.text(
                bin_center,                 
                total_counts[i] + 0.5,     
                f'{int(total_counts[i])}',  
                ha='center',                
                va='bottom',                
                fontweight='bold',          
                fontsize=9                 
            )
    
    stats_text = (f"Statistics:\n"
                  f"Count: {count}\n"
                  f"Mean: {mean_val:.2f}%\n"
                  f"Median: {median_val:.2f}%\n"
                  f"Std Dev: {std_val:.2f}%\n"
                  f"Min: {min_val:.2f}%\n"
                  f"Max: {max_val:.2f}%\n"
                  f"25th Percentile: {p25:.2f}%\n"
                  f"75th Percentile: {p75:.2f}%")
    
    title = op_name if op_name else "Performance of autodeducted Split-K vs best standard Split-K"
    size = 12 if op_name else 14
    plt.title(title, 
              fontsize=size, fontweight='bold')
    plt.xlabel('Performance (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(bin_edges)
    plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.axvline(x=100, color='black', linestyle='--', alpha=0.9, linewidth=2, 
                label='100% Threshold')
    
    below_count = len(below_100)
    above_count = len(above_100)
    below_percent = (below_count / count) * 100 if count > 0 else 0
    above_percent = (above_count / count) * 100 if count > 0 else 0

    legend =plt.legend([
        f'Below 100% ({below_count}, {below_percent:.1f}%)',
        f'Above 100% ({above_count}, {above_percent:.1f}%)',
        '100% Threshold'
    ])
    legend.set_bbox_to_anchor((0.225, 0.65))
    
    plt.tight_layout()
 
    file_name = os.path.join(output_dir, f'performance{suffix}{label}.png')
    plt.savefig(file_name, dpi=150)
    print(f"Saved performance chart to: {file_name}")
    
    plt.close()

def plot_split_k_distribution(non_standard_counts, optimized_count, args, suffix):
    sorted_items = sorted(non_standard_counts.items(), key=lambda x: int(x[0]))
    opt_values = [x[0] for x in sorted_items]
    opt_counts = [x[1] for x in sorted_items]
    
    plt.figure(figsize=(10, max(6, len(opt_values) * 0.4)))  
    bars = plt.barh(
        range(len(opt_values)),  
        opt_counts,             
        color='green',
        edgecolor='black',
        alpha=0.8,
        height=0.6
    )
    
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{int(width)}',
            va='center',
            fontweight='bold'
        )
    
    plt.yticks(
        range(len(opt_values)),  
        opt_values,             
        fontsize=10
    )
    
    plt.title('Distribution of Optimized Split-K Values', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Count)', fontsize=12)
    plt.ylabel('Split-K Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    stats_text = (f"Total Optimized Values: {optimized_count}\n"
                  f"Unique Values: {len(opt_values)}\n"
                  f"Min: {min(map(int, opt_values))}\n"
                  f"Max: {max(map(int, opt_values))}")
    
    plt.text(0.75, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    opt_plot_path = os.path.join(args.output_dir, f'optimized_split_k_distribution{suffix}.png')
    plt.savefig(opt_plot_path)
    print(f"Saved optimized split-K distribution chart to: {opt_plot_path}")

def plot_subscription_factor(gemm_k_values, subs_factor_values, output_dir, suffix="", key=""):
    """Plot the subscription factor distribution in relation to gemm_k."""
    import numpy as np
    from scipy import stats
    
    suffix = f"{suffix}-{key}"

    plt.figure(figsize=(10, 6))
    plt.scatter(gemm_k_values, subs_factor_values, 
                alpha=0.7, color='blue', edgecolor='black')
    
    size = 10 if key else 14
    title = key if key else "Subscription factor vs GEMM K Dimension for best instance"
    plt.title(title, fontsize=size, fontweight='bold')
    plt.xlabel('GEMM K Dimension', fontsize=12)
    plt.ylabel('Subscription Factor', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    mode_result = stats.mode(subs_factor_values)
    mode_value = mode_result.mode
    if mode_value > 1:
        print(f"NOTE: Operator {key} has a mode subscription factor of {mode_value}, which is greater than 1.")
    mode_count = np.sum(np.array(subs_factor_values) == mode_value) 
    stats_text = (f"Statistics for Subscription Factor:\n"
                  f"Count: {len(subs_factor_values)}\n"
                  f"Mean: {np.mean(subs_factor_values):.2f}\n"
                  f"Median: {np.median(subs_factor_values):.2f}\n"
                  f"Min: {np.min(subs_factor_values):.2f}\n"
                  f"Max: {np.max(subs_factor_values):.2f}\n"
                  f"Most Common: {mode_value} (occurs {mode_count} times)")
    
    plt.text(0.6, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    file_name = os.path.join(output_dir, f'subscription_factor{suffix}.png')
    plt.savefig(file_name)
    
    plt.close()

def plot_subscription_factor_per_instance(kgemm_to_subscription_per_instance, output_dir, suffix):
    """Plot the subscription factor distribution for all instances in the same figure with different colors."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10.colors
    color_index = 0
    legend_handles = []

    for op, data_points in kgemm_to_subscription_per_instance.items():
        if not data_points:
            continue
            
        # Skip if the op name doesn't start with "Device"
        if not op.startswith("Device"):
            continue

        kgemm_values = []
        subs_values = []
        for p in data_points:
            if p[0] == "N/A" or pd.isna(p[0]) or p[1] == "N/A" or pd.isna(p[1]):
                continue

            kgemm_values.append(int(p[0]))
            subs_values.append(int(p[1]))
        
        current_color = colors[color_index % len(colors)]
        color_index += 1
        
        scatter = plt.scatter(kgemm_values, subs_values, 
                             alpha=0.7, 
                             color=current_color, 
                             edgecolor='black',
                             label=op)
        
        legend_handles.append(scatter)
    
    plt.title('Subscription Factor vs GEMM K for All Instances', fontsize=14)
    plt.xlabel('GEMM K Dimension', fontsize=12)
    plt.ylabel('Subscription Factor', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(handles=legend_handles, 
           loc='upper center',
           bbox_to_anchor=(0.5, -0.1),  
           fontsize=9,
           title='Operation Names') 
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    file_name = os.path.join(output_dir, f'subscription_factor_all_instances{suffix}.png')
    plt.savefig(file_name, dpi=150)
    plt.close()

def plot_performance(fixed_split_k_tflops, best_occupancy_split_k_tflops, gemm_m, gemm_n, gemm_k, 
                     arithmetic_intensity, output_dir, suffix, op_name):
    """Plot the performance of fixed split-k vs best occupancy split-k."""
    plt.figure(figsize=(12, 8))
    
    # Convert to float for plotting
    fixed_split_k_tflops = fixed_split_k_tflops.astype(float).values
    best_occupancy_split_k_tflops = best_occupancy_split_k_tflops.astype(float).values
    gemm_m_arr = gemm_m.astype(float).values
    gemm_n_arr = gemm_n.astype(float).values
    gemm_k_arr = gemm_k.astype(float).values
    ai_arr = arithmetic_intensity.astype(float).values
    
    perf = (best_occupancy_split_k_tflops / fixed_split_k_tflops) * 100.0

    x_values = np.log10(gemm_k_arr) 
    y_values = np.log10(gemm_m_arr * gemm_n_arr)

    # Heat map with axis gemm_m * gemm_n and gemm_k
    scatter = plt.scatter(x_values, y_values, 
                c=perf,
                cmap='bwr',
                edgecolor='black',
                alpha=0.7,
                s=40,  # Size of the points
                norm=plt.Normalize(vmin=0, vmax=200))  # Normalize colors: blue (<100%), red (>100%)
    
    title = op_name if op_name else 'Performance of Best Occupancy Split-K vs Fixed Split-K'
    title_size = 14 if op_name else 16

    plt.colorbar(label='Performance (%)')
    plt.title(title, fontsize=title_size)
    plt.xlabel('log(K)', fontsize=14)
    plt.ylabel('log(M * N)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    file_name = os.path.join(output_dir, f'performance_heatmap_k_mn{suffix}.png')
    plt.savefig(file_name, dpi=150)
    print(f"Saved performance heatmap to: {file_name}")

    # Heat map with axis log(gemm_k) and log(ai_arr)
    y_values = np.log(ai_arr)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(x_values, y_values,
                c=perf,
                cmap='bwr',
                edgecolor='black',
                alpha=0.7,
                s=40,  # Size of the points
                norm=plt.Normalize(vmin=0, vmax=200))  # Normalize colors: blue (<100%), red (>100%)
    plt.colorbar(label='Performance (%)')
    plt.title(title, fontsize=title_size)
    plt.xlabel('$\\log(K_{\\mathrm{gemm}})$', fontsize=14)
    plt.ylabel('log(Arithmetic Intensity)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    fp16_ridge_point = np.log10(1307.4 / 5.3)
    fp32_ridge_point = np.log10(653.7 / 5.3) 
    plt.axhline(y=fp16_ridge_point, color='green', linestyle='--', label='FP16/BF16 Ridge Point')
    plt.axhline(y=fp32_ridge_point, color='black', linestyle='--', label='FP32 Ridge Point')
    plt.legend()

    plt.text(0.05, 0.05, "Memory bound", transform=plt.gca().transAxes, 
             fontsize=12, color='black', verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.05, 0.95, "Compute bound", transform=plt.gca().transAxes, 
             fontsize=12, color='black', verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    file_name = os.path.join(output_dir, f'performance_heatmap_k_ai{suffix}.png')
    plt.savefig(file_name, dpi=150)
    print(f"Saved performance heatmap to: {file_name}")

def plot_split_k_value_comparison(fixed_split_k_values, best_occupancy_split_k_values, gemm_k, arithmetic_intensity, output_dir, suffix, op_name):
    """Plot the comparison of fixed split-k values vs best occupancy split-k values."""
    plt.figure(figsize=(12, 8))
    
    # Convert to float for plotting
    fixed_split_k_values = fixed_split_k_values.astype(float).values
    best_occupancy_split_k_values = best_occupancy_split_k_values.astype(float).values
    gemm_k_arr = gemm_k.astype(float).values
    ai_arr = arithmetic_intensity.astype(float).values
    
    ratio = (fixed_split_k_values / best_occupancy_split_k_values)

    x_values = np.log(gemm_k_arr) 
    y_values = np.log(ai_arr)

    # Heat map with axis gemm_k and arithmetic intensity
    scatter = plt.scatter(x_values, y_values, 
                c=ratio,
                cmap='viridis',
                edgecolor='black',
                alpha=0.7,
                s=40,  # Size of the points
                norm=plt.Normalize(vmin=0.0, vmax=2.0))

    fp16_ridge_point = np.log10(1307.4 / 5.3)
    fp32_ridge_point = np.log10(653.7 / 5.3) 
    plt.axhline(y=fp16_ridge_point, color='green', linestyle='--', label='FP16/BF16 Ridge Point')
    plt.axhline(y=fp32_ridge_point, color='black', linestyle='--', label='FP32 Ridge Point')
    
    title = op_name if op_name else 'Comparison of Fixed Split-K vs Best Occupancy Split-K'
    title_size = 14 if op_name else 16

    plt.colorbar(label='best fixed split-K / best occupancy split-K')
    plt.title(title, fontsize=title_size)
    plt.xlabel('log(K)', fontsize=14)
    plt.ylabel('log(Arithmetic Intensity)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    file_name = os.path.join(output_dir, f'split_k_value_comparison{suffix}.png')
    plt.savefig(file_name, dpi=150)
    print(f"Saved split-k value comparison heatmap to: {file_name}")

def get_convolution_shapes(profiler_commands):
    """Extract convolution shapes from profiler commands."""
    G, N, K, C, Y, X, Ho, Wo = [], [], [], [], [], [], [], []
    
    for command in profiler_commands:
        parts = command.split()
        g = int(parts[9])
        n = int(parts[10])
        k = int(parts[11])
        c = int(parts[12])
        y = int(parts[13])
        x = int(parts[13])
        hi = int(parts[14])
        wi = int(parts[15])
        sy = int(parts[16])
        sx = int(parts[17])
        dy = int(parts[18])
        dx = int(parts[19])
        left_py = int(parts[20])
        left_px = int(parts[21])
        right_py = int(parts[22])
        right_px = int(parts[23])

        effective_y = dy * (y - 1) + 1
        effective_x = dx * (x - 1) + 1
        
        total_pad_y = left_py + right_py
        total_pad_x = left_px + right_px
        
        ho = (hi + total_pad_y - effective_y) // sy + 1
        wo = (wi + total_pad_x - effective_x) // sx + 1
     
        G.append(g)
        N.append(n)
        K.append(k)
        C.append(c)
        Y.append(y)
        X.append(x)
        Ho.append(ho)
        Wo.append(wo)
    
    return G, N, K, C, Y, X, Ho, Wo

def plot_tSNE_performance(G, N, K, C, Y, X, Ho, Wo, fixed_split_k_tflops, best_occupancy_split_k_tflops, output_dir, suffix="", op_name=""):
    """Plot t-SNE performance of fixed split-k vs best occupancy split-k."""
    from sklearn.manifold import TSNE
    
    # Prepare data for t-SNE
    data = np.array([G, N, K, C, Y, X, Ho, Wo]).T
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data)

    perf = (best_occupancy_split_k_tflops / fixed_split_k_tflops) * 100.0

    plt.figure(figsize=(12, 8))
    
    # Scatter plot of t-SNE results
    scatter = plt.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        c=perf, 
        cmap='bwr', 
        edgecolor='black', 
        alpha=0.7,
        s=30,
        norm=plt.Normalize(vmin=0, vmax=200))
    
    plt.colorbar(scatter, label='Performance (%)')
    
    title = op_name if op_name else 't-SNE Performance of Fixed Split-K vs Best Occupancy Split-K'
    title_size = 14 if op_name else 16

    plt.title(title, fontsize=title_size)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    file_name = os.path.join(output_dir, f'tSNE_performance{suffix}.png')
    plt.savefig(file_name, dpi=150)
    print(f"Saved t-SNE performance chart to: {file_name}")
    
    plt.close()

def get_statistics(fixed_split_k_values, fixed_split_k_times, fixed_split_k_ops, best_occupancy_split_k_values, best_occupancy_split_k_times, best_occupancy_split_k_ops):
    # Find indices where split-k is not in the standard set
    standard_split_k = ['1', '2', '4', '8', '16', '32', '64', '128']
    non_standard_indices = [i for i in range(len(best_occupancy_split_k_values)) 
                            if best_occupancy_split_k_values.iloc[i] not in standard_split_k]

    non_standard_split_k_values = []

    for i in non_standard_indices:
        try:
            non_standard_split_k_values.append(best_occupancy_split_k_values.iloc[i])
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not process non-standard row {i}: {e}")

    fixed_split_k_counts = defaultdict(int)
    best_occupancy_split_k_count = 0
    fixed_equal_best_occupancy_counts = defaultdict(int)
    perf_change = []

    # Initialize counts for standard split-k values
    for sk in standard_split_k:
        fixed_split_k_counts[sk] = 0
        fixed_equal_best_occupancy_counts[sk] = 0

    assert len(fixed_split_k_values) == len(best_occupancy_split_k_values), \
        "Length of fixed split-k values and best occupancy split-k values must match."

    for i in range(len(fixed_split_k_values)):
        fixed_split_k_time = float(fixed_split_k_times.iloc[i])
        best_occ_split_k_time = float(best_occupancy_split_k_times.iloc[i])
        fixed_split_k_value = fixed_split_k_values.iloc[i]
        best_occ_split_k_value = best_occupancy_split_k_values.iloc[i]
        fixed_split_k_op = fixed_split_k_ops.iloc[i]
        best_occ_split_k_op = best_occupancy_split_k_ops.iloc[i]

        if best_occ_split_k_op:
            tol = 1e-7  # Tolerance for floating point comparison
            perf = 100.0 * (fixed_split_k_time / best_occ_split_k_time) if best_occ_split_k_time > tol else 0.0

            if best_occ_split_k_value == fixed_split_k_value and best_occ_split_k_op == fixed_split_k_op:
                fixed_equal_best_occupancy_counts[fixed_split_k_value] += 1

            elif best_occ_split_k_time < fixed_split_k_time and best_occ_split_k_time > tol:
                best_occupancy_split_k_count += 1
                perf_change.append(min(150.0, perf)) # Cap to 150% to make visualization better.
            elif best_occ_split_k_time > fixed_split_k_time and fixed_split_k_time > tol:
                fixed_split_k_counts[fixed_split_k_value] += 1
                perf_change.append(min(150.0, perf)) # Cap to 150% to make visualization better.

            if best_occ_split_k_time < tol and fixed_split_k_time > tol:
                print(f"WARNING: Optimized time is very small for row {i}. Split-K (opt): {best_occ_split_k_value}, Split-K (standard): {fixed_split_k_value}")
            elif best_occ_split_k_time > tol and fixed_split_k_time < tol:
                print(f"WARNING: Non-optimized time is very small for row {i}. Split-K (opt): {best_occ_split_k_value}, Split-K (stardard): {fixed_split_k_value}")
            elif best_occ_split_k_time < tol and fixed_split_k_time < tol:
                print(f"WARNING: Both optimized and non-optimized times are too small for row {i}, skipping this. Split-K (opt): {best_occ_split_k_value}, Split-K (stardard): {fixed_split_k_value}")

    return perf_change, fixed_split_k_counts, fixed_equal_best_occupancy_counts, best_occupancy_split_k_count, non_standard_indices

def plot_perf_for_all_solvers(solvers_per_conv_shape, output_dir, suffix, op_name):
    
    perf_difference = []
    ranking = []
    for _, values in solvers_per_conv_shape.items():
        if not values:
            continue
        
        for _, fixed_split_k_tflops, _, best_occ_split_k_tflops, rank  in values:
            perf_diff = (best_occ_split_k_tflops / fixed_split_k_tflops) * 100.0 if fixed_split_k_tflops > 0 else 0.0
            perf_difference.append(min(150.0, perf_diff))
            ranking.append(rank)

    plot_perf(perf_difference, output_dir, suffix=suffix, op_name=op_name, label="-all_instances")

    # Create a bar chart for the ranking distribution
    title = op_name if op_name else "Ranking Distribution of All Instances"
    title_size = 14 if op_name else 16
    plt.figure(figsize=(10, 6))
    
    # Define the bins edges
    bin_edges = range(1, max(ranking) + 2)
    
    # Create histogram
    counts, bins, patches = plt.hist(ranking, bins=bin_edges, 
             color='skyblue', edgecolor='black', alpha=0.7)
    
    # Calculate the center of each bin for x-ticks
    bin_centers = [bins[i] + (bins[i+1] - bins[i])/2 for i in range(len(bins)-1)]
    
    plt.title(title, fontsize=title_size, fontweight='bold')
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add explanation text middle top
    # y_loc = 0.9*max(counts)
    # explanation = "Candidate split-K values ['best occupancy', 1, 2, 4, 8, 16, 32, 64, 128].\n" \
    #               "Ranking of 'best occupancy' value for each solver instance\n" \
    #               "Rank 1 is the best, rank 2 is second best, etc."
    # plt.text(2.5, y_loc, explanation)

    # Set x-ticks at the center of each bar
    plt.xticks(bin_centers, range(1, max(ranking) + 1))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    rank_distribution_path = os.path.join(output_dir, f'ranking_distribution{suffix}.png')
    plt.savefig(rank_distribution_path, dpi=150)
    print(f"Saved ranking distribution chart to: {rank_distribution_path}")

def main():
    args = parse_cli_args()

    csv.register_dialect('PipeDialect', delimiter=';')
    with open(args.csv_file) as csvfile:
        data = [row for row in csv.reader(csvfile, 'PipeDialect')]

    df = pd.DataFrame(data = data)

    print(f"Loaded {len(df)} rows.")
    print(df.head())

    if args.old_format:
        fixed_split_k_ops = df[0]
        fixed_split_k_times = df[1]
        fixed_split_k_values = df[2]
        best_occupancy_split_k_ops = df[3]
        best_occupancy_split_k_times = df[4]
        best_occupancy_split_k_values = df[5]
    else:
        # The dataframe may row that that contain only one column. 
        # These are the shapes where no instance of the solver was applicable.
        # Separate these into a separate dataframe.
        non_null_counts = df.count(axis=1)
        no_applicable_op_found = df[non_null_counts == 1].copy()
        df = df[non_null_counts > 1].copy()

        valid_mask1 = df[11] == "SplitKStrategy::FixedSplitK"
        valid_mask2 = df[17] == "SplitKStrategy::BestOccupancy"
        valid_mask = valid_mask1 & valid_mask2

        profiler_commands = df[0][valid_mask]
        gemm_m = df[1][valid_mask]
        gemm_n = df[2][valid_mask]
        gemm_k = df[3][valid_mask]
        arithmetic_intensity = df[4][valid_mask]
        data_type = df[5][valid_mask]

        fixed_split_k_ops = df[6][valid_mask]
        fixed_split_k_times = df[7][valid_mask]
        fixed_split_k_tflops = df[8][valid_mask]
        fixed_split_k_values = df[8][valid_mask]
        # 10 - rank
        # 11 - strategy
    
        best_occupancy_split_k_ops = df[12][valid_mask]
        best_occupancy_split_k_times = df[13][valid_mask]
        best_occupancy_split_k_tflops = df[14][valid_mask]
        best_occupancy_split_k_values = df[15][valid_mask]
        # 16 - rank
        # 17 - strategy
        # 18 - total number of candidate ops.

    # Columns 19-30 are
    # 19: op_name
    # 20: fixed_split_k_time
    # 21: fixed_split_k_tflops
    # 22: fixed_split_k_value
    # 23: rank_fixed_split_k
    # 24: strategy (FixedSplitK)
    # 25: best_occupancy_split_k_time
    # 26: best_occupancy_split_k_tflops
    # 27: best_occupancy_split_k_value
    # 28: rank_best_occupancy_split_k
    # 29: strategy (BestOccupancy)
    # 30: total number of candidate values
    # This repeats for size=12 blocks, i.e., the next 12 elemnts from 31-42 have the same structure if they are not null.
    # Collect  these elents into a dictionary
    # where each key is the profiler_command and the value is a list of tuples containing the values for each block.
    solvers_per_conv_shape = defaultdict(list)
    offset = 18
    size = 12
    for i in range(len(profiler_commands)):
        profiler_command = profiler_commands.iloc[i]
        #print(f"Processing profiler command: {profiler_command}, row: {i}")
        if pd.isna(profiler_command):
            continue
        if profiler_command not in solvers_per_conv_shape:
            solvers_per_conv_shape[profiler_command] = []
        for j in range(0, len(df.columns) - size - offset, size):
            op_name = df.iloc[i, offset + j + 1]
            if pd.isna(op_name):
                continue

            try:
                loc_fixed_split_k_time = float(df.iloc[i, offset + j + 2])
                loc_fixed_split_k_tflops = float(df.iloc[i, offset + j + 3])
                loc_fixed_split_k_value = int(df.iloc[i, offset + j + 4])
                loc_rank_fixed_split_k = int(df.iloc[i, offset + j + 5])
                loc_strategy_fixed_split_k = df.iloc[i, offset + j + 6]
                loc_best_occupancy_split_k_time = float(df.iloc[i, offset + j + 7])
                loc_best_occupancy_split_k_tflops = float(df.iloc[i, offset + j + 8])
                loc_best_occupancy_split_k_value = int(df.iloc[i, offset + j + 9])
                loc_rank_best_occupancy_split_k = int(df.iloc[i, offset + j + 10])
                loc_strategy_best_occupancy_split_k = df.iloc[i, offset + j + 11]
                loc_num_candidates = int(df.iloc[i, offset + j + 12])

                assert loc_strategy_fixed_split_k == "SplitKStrategy::FixedSplitK", \
                    f"Expected strategy_fixed_split_k to be 'SplitKStrategy::FixedSplitK', got {loc_strategy_fixed_split_k}."
                assert loc_strategy_best_occupancy_split_k == "SplitKStrategy::BestOccupancy", \
                    f"Expected strategy_best_occupancy_split_k to be 'SplitKStrategy::BestOccupancy', got {loc_strategy_best_occupancy_split_k}."
                # Candidates: {-1, 1, 2, 4, 8, 16, 32, 64, 128}
                # Sometime the split-K value can be incompatible with the V3 pipeline and we have may less than 9 candidates.
                assert loc_num_candidates <= 9 and loc_num_candidates > 1, \
                    f"Expected num_candidates to be 9, got {loc_num_candidates}." 
                assert loc_rank_best_occupancy_split_k >= 1 and loc_rank_best_occupancy_split_k <= 9, \
                        f"Expected rank_best_occupancy_split_k to be between 1 and 9, got {loc_rank_best_occupancy_split_k}."

                solvers_per_conv_shape[profiler_command].append(
                    (loc_fixed_split_k_value, loc_fixed_split_k_tflops, loc_best_occupancy_split_k_value, loc_best_occupancy_split_k_tflops, loc_rank_best_occupancy_split_k))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not process row {i}, block {j}: {e}. Skipping this block.")
                continue

    op_name = fixed_split_k_ops.iloc[0].split("<")[0]
    suffix = f"_{args.label}" if args.label else ""

    plot_perf_for_all_solvers(solvers_per_conv_shape, args.output_dir, suffix, op_name)

    G, N, K, C, Y, X, Ho, Wo = get_convolution_shapes(profiler_commands)
    plot_tSNE_performance(G,N,K,C,Y,X,Ho,Wo, fixed_split_k_tflops.astype(float).values, best_occupancy_split_k_tflops.astype(float).values, args.output_dir, suffix, op_name)

    perf_change, fixed_split_k_counts, fixed_equal_best_occupancy_counts, best_occupancy_split_k_count, non_standard_indices = get_statistics(
        fixed_split_k_values, fixed_split_k_times, fixed_split_k_ops,
        best_occupancy_split_k_values, best_occupancy_split_k_times, best_occupancy_split_k_ops)

    plot_perf(perf_change, args.output_dir, suffix, op_name)

    plot_best_split_k_values(
        fixed_split_k_counts, best_occupancy_split_k_count, 
        fixed_equal_best_occupancy_counts, suffix, args)

    non_standard_values = [best_occupancy_split_k_values.iloc[i] for i in non_standard_indices]
    non_standard_counts = {}
    for val in non_standard_values:
        non_standard_counts[val] = non_standard_counts.get(val, 0) + 1
    
    plot_split_k_distribution(non_standard_counts, best_occupancy_split_k_count, args, suffix)

    plot_performance(fixed_split_k_tflops, best_occupancy_split_k_tflops, gemm_m, gemm_n, gemm_k, arithmetic_intensity, args.output_dir, suffix, op_name)

    plot_split_k_value_comparison(fixed_split_k_values, best_occupancy_split_k_values, gemm_k, arithmetic_intensity, args.output_dir, suffix, op_name)

if __name__ == "__main__":
    main()