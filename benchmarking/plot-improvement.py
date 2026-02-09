#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

tflops_baseline = [2.85567, 10.3518, 15.4196, 849.773, 990.991]
tflops_improved = [8.09047, 19.8867, 20.4604, 952.806, 1091.89]

# Plot tflops for both baseline and improved
plt.figure(figsize=(10, 6))
indices = np.arange(len(tflops_baseline))
bar_width = 0.35
plt.bar(indices, tflops_baseline, bar_width, label='Baseline', color='b')
plt.bar(indices + bar_width, tflops_improved, bar_width, label='Improved', color='g')
plt.xlabel('Test Cases')
plt.ylabel('TFLOPS')
plt.title('TFLOPS Comparison: Baseline vs Improved')
plt.xticks(indices + bar_width / 2, ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5'])
plt.yscale('log')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('tflops_comparison.png')
plt.close()

# plot improvement factor
improvement_factor = [100 *(improved / baseline)for improved, baseline in zip(tflops_improved, tflops_baseline)]
improvement_factor = [factor - 100 for factor in improvement_factor]  # Convert to percentage improvement
plt.figure(figsize=(10, 6))
plt.bar(indices, improvement_factor, color='orange')
plt.xlabel('Test Cases')
plt.ylabel('Improvement (%)')
plt.title('Improvement')
plt.xticks(indices, ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5'])
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('improvement.png')
plt.close()