#!/usr/bin/env python3

import os
import argparse
import sys
import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze convolution test results.")
    parser.add_argument("--csv-file", type=str, dest="csv_file", required=True, help="Path to the CSV file containing test cases.")
    parser.add_argument("--output-dir", type=str, dest="output_dir", required=True, help="Directory to save output plots.")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def main():
  args = parse_cli_args()

  csv.register_dialect('PipeDialect', delimiter=';')
  with open(args.csv_file) as csvfile:
    data = [row for row in csv.reader(csvfile, 'PipeDialect')]

  df = pd.DataFrame(data = data)

  print(f"Loaded {len(df)} rows.")
  print(df.head())

  best_ops = df[0]
  best_times = df[1]
  best_split_k = df[2]
  best_split_k_ops = df[3]
  best_split_k_values = df[4]
  best_split_k_ranks = df[5]
  num_ops = df[6]
  
  print(f"Column stats:")
  print(f"- Best split-k values unique count: {best_split_k.nunique()}")
  print(f"- Best split-k values: {', '.join(best_split_k.unique().tolist()[:10])}...")
  
  # Calculate ranking numbers
  best_split_k_ranking_numbers = []
  for i in range(len(best_split_k_ranks)):
      try:
          rank = int(best_split_k_ranks.iloc[i])
          total_ops = int(num_ops.iloc[i])
          ranking = 100.0 * (total_ops - rank + 1) / total_ops
          best_split_k_ranking_numbers.append(ranking)
      except (ValueError, TypeError) as e:
          print(f"Warning: Could not process row {i}: {e}")
          best_split_k_ranking_numbers.append(0)  # Default value

  # Plot the ranking number as a historgram
  explanation = """For each shape, all supported instances were benchmarked 
with split-K values ["optimized", 1, 2, 4, 8, 16, 32, 128]. 
Ranking 100% means that best instance had optimized split-K value, 
lower values mean that the best instance had one of the fixed split-K values."""
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  plt.figure(figsize=(10, 6))
  plt.hist(best_split_k_ranking_numbers, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
  plt.title('Optimized Split-K Ranking Numbers')
  plt.xlabel('Ranking (%)')
  plt.ylabel('Frequency')
  plt.grid(True, linestyle='--', alpha=0.7)
  plt.text(0.05, 0.8, explanation, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='bottom', bbox=props)
  hist_plot_path = os.path.join(args.output_dir, 'ranking_histogram.png')
  plt.savefig(hist_plot_path)
  print(f"Saved ranking histogram to: {hist_plot_path}")
  
  # Find indices where split-k is not in the standard set
  standard_split_k = ['1', '2', '4', '8', '16', '32', '64', '128']
  non_standard_indices = [i for i in range(len(best_split_k)) 
                          if best_split_k.iloc[i] not in standard_split_k]
  
  print(f"Found {len(non_standard_indices)} cases with non-standard split-k values")
  
  if non_standard_indices:
      # Calculate ranking for non-standard split-k values
      non_standard_split_k_ranking_numbers = []
      non_standard_split_k_values = []
      
      for i in non_standard_indices:
          try:
              rank = int(best_split_k_ranks.iloc[i])
              total_ops = int(num_ops.iloc[i])
              ranking = 100.0 * (total_ops - rank + 1) / total_ops
              non_standard_split_k_ranking_numbers.append(ranking)
              non_standard_split_k_values.append(best_split_k.iloc[i])
          except (ValueError, TypeError) as e:
              print(f"Warning: Could not process non-standard row {i}: {e}")
  
      # Define standard split-K values
      standard_split_k = ['1', '2', '4', '8', '16', '32', '64', '128']

      # Count occurrences
      standard_counts = {}
      optimized_count = 0

      # Initialize standard counts with zeros
      for sk in standard_split_k:
          standard_counts[sk] = 0

      # Count occurrences in your data
      for i in range(len(best_split_k)):
          value = best_split_k.iloc[i]
          if value in standard_split_k:
              standard_counts[value] += 1
          else:
              optimized_count += 1

      # Create ordered categories for the plot
      categories = list(standard_counts.keys()) + ['Optimized Split-K']
      counts = list(standard_counts.values()) + [optimized_count]

      # Create figure
      plt.figure(figsize=(14, 7))

      # Create bar chart with different colors for standard vs optimized
      colors = ['skyblue'] * len(standard_counts) + ['crimson']
      bars = plt.bar(
          range(len(categories)),  # X positions
          counts,                  # Heights (counts)
          color=colors,
          edgecolor='black',
          alpha=0.8,
          width=0.6
      )

      # Add value labels on top of each bar
      for bar in bars:
          height = bar.get_height()
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
          range(len(categories)),  # Positions
          categories,              # Labels
          rotation=0,              # No rotation needed for few categories
          fontsize=11
      )

      # Add labels and title
      plt.title('Distribution of Best Split-K Values', fontsize=16, fontweight='bold')
      plt.xlabel('Split-K Value', fontsize=14)
      plt.ylabel('Frequency (Count)', fontsize=14)
      plt.grid(True, linestyle='--', alpha=0.7, axis='y')  # Grid lines only on y-axis

      # Add a legend
      from matplotlib.patches import Patch
      legend_elements = [
          Patch(facecolor='skyblue', edgecolor='black', label='Standard Values'),
          Patch(facecolor='crimson', edgecolor='black', label='Optimized Values')
      ]
      plt.legend(handles=legend_elements, loc='upper center', fontsize=12)

      # Adjust layout to prevent label cutoff
      plt.tight_layout()

      # Save the plot
      bar_plot_path = os.path.join(args.output_dir, 'split_k_distribution.png')
      plt.savefig(bar_plot_path)
      print(f"Saved split-K distribution chart to: {bar_plot_path}")
      print(f"You can view it with: \"$BROWSER\" {os.path.abspath(bar_plot_path)}")

      # Display the detailed breakdown
      print("\nFrequency of Split-K values:")
      for k, count in standard_counts.items():
          print(f"  Split-K = {k}: {count} instances")
      print(f"  Optimized Split-K: {optimized_count} instances")

      # If optimized count is non-zero, show the distribution of optimized values
      if optimized_count > 0:
          non_standard_values = [best_split_k.iloc[i] for i in range(len(best_split_k)) 
                                if best_split_k.iloc[i] not in standard_split_k]
          non_standard_counts = {}
          for val in non_standard_values:
              non_standard_counts[val] = non_standard_counts.get(val, 0) + 1
          
          print("\nBreakdown of optimized Split-K values:")
          for k, count in sorted(non_standard_counts.items(), key=lambda x: int(x[0])):
              print(f"  Split-K = {k}: {count} instances")


  if optimized_count > 0:
    non_standard_values = [best_split_k.iloc[i] for i in range(len(best_split_k)) 
                          if best_split_k.iloc[i] not in standard_split_k]
    non_standard_counts = {}
    for val in non_standard_values:
        non_standard_counts[val] = non_standard_counts.get(val, 0) + 1
    
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
        color='crimson',
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
    opt_plot_path = os.path.join(args.output_dir, 'optimized_split_k_distribution.png')
    plt.savefig(opt_plot_path)
    print(f"Saved optimized split-K distribution chart to: {opt_plot_path}")

if __name__ == "__main__":
    main()