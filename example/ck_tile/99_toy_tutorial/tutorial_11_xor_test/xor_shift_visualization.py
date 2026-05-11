#!/usr/bin/env python3
"""
Visualization of XOR transform: how logical [row, col] maps to physical column.
Physical: (row, col ^ (row % 8))
Row stays the same; column gets XOR'd with (row % 8).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# XOR parameters (from xor_test.cpp)
ROWS = 32   # kM / MLdsLayer
COLS = 8    # kK / kKPack * MLdsLayer

def xor_physical_col(logical_row, logical_col):
    """Physical column = logical_col ^ (logical_row % COLS)"""
    return logical_col ^ (logical_row % COLS)

# Build the mapping grid
# logical_grid[r, c] = physical column that logical (r,c) maps to
logical_to_physical_col = np.zeros((ROWS, COLS), dtype=int)
for r in range(ROWS):
    for c in range(COLS):
        logical_to_physical_col[r, c] = xor_physical_col(r, c)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 10))

# --- Left: Logical grid colored by physical column ---
# Each cell (r,c) shows where it goes: same row, but column = c ^ (r % 8)
ax1 = axes[0]
im1 = ax1.imshow(logical_to_physical_col, cmap='tab10', vmin=0, vmax=9, aspect='auto')
ax1.set_xlabel('Logical column (pack index)')
ax1.set_ylabel('Logical row (bank row index)')
ax1.set_title('Physical column destination\n(cell at logical [r,c] → physical col = c ^ (r % 8))')
ax1.set_xticks(range(COLS))
ax1.set_yticks(range(0, ROWS, 4))
ax1.set_xticklabels(range(COLS))
ax1.set_yticklabels(range(0, ROWS, 4))

# Add text annotations for first few rows to show the pattern
for r in range(min(8, ROWS)):
    for c in range(COLS):
        phys = logical_to_physical_col[r, c]
        ax1.text(c, r, f'{phys}', ha='center', va='center', fontsize=8, color='white', weight='bold')

# Colorbar
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('Physical column')

# --- Right: "Shift" amount (how much each column moved) ---
# shift[r,c] = physical_col - logical_col (can be negative)
shift = logical_to_physical_col - np.arange(COLS)[np.newaxis, :]
ax2 = axes[1]
im2 = ax2.imshow(shift, cmap='RdBu_r', vmin=-7, vmax=7, aspect='auto')
ax2.set_xlabel('Logical column')
ax2.set_ylabel('Logical row')
ax2.set_title('Column shift (physical_col - logical_col)\nXOR permutes columns differently per row')
ax2.set_xticks(range(COLS))
ax2.set_yticks(range(0, ROWS, 4))
ax2.set_xticklabels(range(COLS))
ax2.set_yticklabels(range(0, ROWS, 4))

for r in range(min(8, ROWS)):
    for c in range(COLS):
        s = shift[r, c]
        color = 'white' if abs(s) > 3 else 'black'
        ax2.text(c, r, f'{s:+d}', ha='center', va='center', fontsize=8, color=color, weight='bold')

cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Shift amount')

plt.suptitle('XOR Transform: logical [row, col] → physical [row, col ^ (row % 8)]', fontsize=12)
plt.tight_layout()
plt.savefig('xor_shift_visualization.png', dpi=150, bbox_inches='tight')
print('Saved xor_shift_visualization.png')
plt.show()
