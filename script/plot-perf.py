
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Sample data for Split-K values and TFops values
split_k_values = [1, 2, 4, 8, 16, 32, 64, 128]
tfops_values = [3.50016, 7.175, 13.7204, 26.6367, 41.2162, 48.5583, 51.9327, 51.8869]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(split_k_values, tfops_values, color='blue', marker='o')
plt.title('Split-K Values vs TFops Values')
plt.xlabel('Split-K Values')
plt.ylabel('TFops Values')
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)  

# Save the plot as a PNG file
plt.savefig('split_k_vs_tfops.png')
plt.show()