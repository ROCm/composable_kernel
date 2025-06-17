#!/usr/bin/env python3

import os
import argparse
import sys
import pandas as pd
import csv
import matplotlib
from collections import defaultdict

matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze convolution test results.")
    parser.add_argument("--csv-file", type=str, dest="csv_file", required=True, help="Path to the CSV file containing test cases.")
    parser.add_argument("--output-dir", type=str, dest="output_dir", required=True, help="Directory to save output plots.")
    parser.add_argument("--label", type=str, dest="label", default="", help="Label for the figure names.")
    
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
        total_standard_counts.append(standard_counts[key])

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

    # Highlight the optimized category with a different color
    base_bars[-1].set_color('green')
    base_bars[-1].set_label('Optimized Split-K')

    # Set x-tick positions and labels
    plt.xticks(
        range(len(categories)),  # Positions
        categories,              # Labels
        rotation=45 if len(categories) > 8 else 0,  # Rotate if many categories
        fontsize=11,
        ha='right' if len(categories) > 8 else 'center'  # Align rotated labels
    )

    # Add labels, title, and legend
    plt.title('Best Split-K Values', fontsize=16, fontweight='bold')
    plt.xlabel('Split-K Value', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')  # Grid lines only on y-axis
    plt.legend(fontsize=12)

    # Add explanation text for the orange portion
    explanation = "Orange sections represent cases where optimized\nsplit-K equals to one of the fixed split-K values"
    plt.text(
        0.02, 0.95,                     # Position in axes coordinates (top-left)
        explanation,
        transform=plt.gca().transAxes,  # Use axes coordinates
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    split_k_distribution_path = os.path.join(args.output_dir, f'best_split_k_values{suffix}.png')
    plt.savefig(split_k_distribution_path)
    print(f"Saved best split-K values chart to: {split_k_distribution_path}")
    
    plt.close()

def plot_perf(perf_difference, output_dir, suffix=""):
    """Plot the performance differences as a histogram with statistics."""
    import numpy as np
    
    # Calculate statistics
    mean_val = np.mean(perf_difference)
    median_val = np.median(perf_difference)
    std_val = np.std(perf_difference)
    min_val = np.min(perf_difference)
    max_val = np.max(perf_difference)
    p25 = np.percentile(perf_difference, 25)
    p75 = np.percentile(perf_difference, 75)
    count = len(perf_difference)
    
    # Determine bin edges at 5% intervals
    min_edge = np.floor(min_val / 5) * 5
    max_edge = np.ceil(max_val / 5) * 5
    bin_edges = np.arange(min_edge, max_edge + 5, 5)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Split data into below and above 100%
    below_100 = [x for x in perf_difference if x < 100]
    above_100 = [x for x in perf_difference if x >= 100]
    
    # Plot histogram for values below 100% (red)
    if below_100:
        plt.hist(below_100, bins=bin_edges, color='red', 
                alpha=0.7, edgecolor='black', label='Below 100%')
    
    # Plot histogram for values above or equal to 100% (green)
    if above_100:
        plt.hist(above_100, bins=bin_edges, color='green', 
                alpha=0.7, edgecolor='black', label='Above 100%')
    
    # Create statistics text
    stats_text = (f"Statistics:\n"
                  f"Count: {count}\n"
                  f"Mean: {mean_val:.2f}%\n"
                  f"Median: {median_val:.2f}%\n"
                  f"Std Dev: {std_val:.2f}%\n"
                  f"Min: {min_val:.2f}%\n"
                  f"Max: {max_val:.2f}%\n"
                  f"25th Percentile: {p25:.2f}%\n"
                  f"75th Percentile: {p75:.2f}%")
    
    plt.title('Performance of Optimized Split-K value vs Best Standard Split-K value', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Performance (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add gridlines aligned with bin edges
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure x-axis ticks align with bin edges
    plt.xticks(bin_edges)
    
    # Add statistics text box
    plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add a vertical line at x=100 to highlight the threshold
    plt.axvline(x=100, color='black', linestyle='--', alpha=0.9, linewidth=2, 
                label='100% Threshold')
    
    plt.legend(loc='upper center')
    plt.tight_layout()
 
    file_name = os.path.join(output_dir, f'performance{suffix}.png')
    plt.savefig(file_name, dpi=150)
    print(f"Saved performance chart to: {file_name}")
    
    plt.close()

def plot_split_k_distribution(non_standard_counts, optimized_count, args, suffix):
    # Sort the values numerically
    sorted_items = sorted(non_standard_counts.items(), key=lambda x: int(x[0]))
    opt_values = [x[0] for x in sorted_items]
    opt_counts = [x[1] for x in sorted_items]
    
    # Create figure for optimized values
    plt.figure(figsize=(10, max(6, len(opt_values) * 0.4)))  # Adjust height based on number of items
    
    # Create horizontal bar chart
    bars = plt.barh(
        range(len(opt_values)),  # Y positions
        opt_counts,              # Widths (counts)
        color='green',
        edgecolor='black',
        alpha=0.8,
        height=0.6
    )
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{int(width)}',
            va='center',
            fontweight='bold'
        )
    
    # Set y-tick positions and labels
    plt.yticks(
        range(len(opt_values)),  # Positions
        opt_values,              # Labels
        fontsize=10
    )
    
    # Add labels and title
    plt.title('Distribution of Optimized Split-K Values', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Count)', fontsize=12)
    plt.ylabel('Split-K Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')  # Grid lines only on x-axis
    
    # Add summary statistics as a text box
    stats_text = (f"Total Optimized Values: {optimized_count}\n"
                  f"Unique Values: {len(opt_values)}\n"
                  f"Min: {min(map(int, opt_values))}\n"
                  f"Max: {max(map(int, opt_values))}")
    
    plt.text(0.75, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    opt_plot_path = os.path.join(args.output_dir, f'optimized_split_k_distribution{suffix}.png')
    plt.savefig(opt_plot_path)
    print(f"Saved optimized split-K distribution chart to: {opt_plot_path}")

def main():
  args = parse_cli_args()

  csv.register_dialect('PipeDialect', delimiter=';')
  with open(args.csv_file) as csvfile:
    data = [row for row in csv.reader(csvfile, 'PipeDialect')]

  df = pd.DataFrame(data = data)

  print(f"Loaded {len(df)} rows.")
  print(df.head())

  non_opt_split_k_ops = df[0]
  non_opt_split_k_times = df[1]
  non_opt_split_k_value = df[2]
  opt_split_k_ops = df[3]
  opt_split_k_times = df[4]
  opt_split_k_values = df[5]

  suffix = f"_{args.label}" if args.label else ""
  
  # Find indices where split-k is not in the standard set
  standard_split_k = ['1', '2', '4', '8', '16', '32', '64', '128']
  non_standard_indices = [i for i in range(len(opt_split_k_values)) 
                          if opt_split_k_values.iloc[i] not in standard_split_k]
  
  print(f"Found {len(non_standard_indices)} cases with non-standard split-k values")
  
  if non_standard_indices:
    non_standard_split_k_values = []
    
    for i in non_standard_indices:
        try:
            non_standard_split_k_values.append(opt_split_k_values.iloc[i])
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not process non-standard row {i}: {e}")

    standard_counts = defaultdict(int)
    optimized_count = 0
    standard_equal_optimized_counts = defaultdict(int)
    perf_change = []

    # Initialize counts for standard split-k values
    for sk in standard_split_k:
        standard_counts[sk] = 0
        standard_equal_optimized_counts[sk] = 0

    assert len(non_opt_split_k_value) == len(opt_split_k_values), \
        "Length of non-opt split-k values and optimized split-k values must match."

    for i in range(len(non_opt_split_k_value)):
        non_opt_time = float(non_opt_split_k_times.iloc[i])
        opt_time = float(opt_split_k_times.iloc[i])
        non_opt_value = non_opt_split_k_value.iloc[i]
        opt_value = opt_split_k_values.iloc[i]
        non_opt_op = non_opt_split_k_ops.iloc[i]
        opt_op = opt_split_k_ops.iloc[i]

        perf = 100.0 * (non_opt_time / opt_time) if opt_time > 1e-5 else 0.0
        perf_change.append(perf)

        if opt_value == non_opt_value and opt_op == non_opt_op:
            standard_equal_optimized_counts[non_opt_value] += 1
 
        elif opt_time < non_opt_time:
            optimized_count += 1
        elif opt_time > non_opt_time:
            standard_counts[non_opt_value] += 1


    plot_perf(perf_change, args.output_dir, suffix)

    plot_best_split_k_values(
        standard_counts, optimized_count, 
        standard_equal_optimized_counts, suffix, args)

    # Display the detailed breakdown
    print("\nFrequency of standard Split-K values:")
    for k, count in standard_counts.items():
        print(f"  Split-K = {k}: {count} instances")

    print("\nFrequency of standard = optimized Split-K values:")
    for k, count in standard_equal_optimized_counts.items():
        print(f"  Split-K = {k}: {count} instances")

    print(f"\nOptimized Split-K: {optimized_count} instances")

    # If optimized count is non-zero, show the distribution of optimized values
    if optimized_count > 0:
        non_standard_values = [opt_split_k_values.iloc[i] for i in non_standard_indices]
        non_standard_counts = {}
        for val in non_standard_values:
            non_standard_counts[val] = non_standard_counts.get(val, 0) + 1
        
        print("\nBreakdown of optimized Split-K values:")
        for k, count in sorted(non_standard_counts.items(), key=lambda x: int(x[0])):
            print(f"  Split-K = {k}: {count} instances")

        plot_split_k_distribution(non_standard_counts, optimized_count, args, suffix)
    
if __name__ == "__main__":
    main()