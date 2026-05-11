#!/usr/bin/env python3
"""
XOR transform visualization - pure Python, no matplotlib/numpy.
Generates xor_shift_visualization.svg
"""

ROWS, COLS = 32, 8
CELL_W, CELL_H = 36, 22

def xor_physical_col(r, c):
    return c ^ (r % COLS)

# Colors for physical column 0-7
COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
    '#f58231', '#911eb4', '#46f0f0', '#f032e6',
]

def main():
    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="800" viewBox="0 0 500 800">')
    svg.append('<rect width="500" height="800" fill="#1a1a2e"/>')
    svg.append('<text x="10" y="25" fill="#eee" font-family="sans-serif" font-size="14">XOR: physical_col = logical_col ^ (row % 8)</text>')
    svg.append('<text x="10" y="42" fill="#888" font-family="sans-serif" font-size="11">Same column in different rows → different physical cols → different LDS banks</text>')

    ox, oy = 50, 60
    # Column headers
    for c in range(COLS):
        x = ox + 40 + c * CELL_W
        svg.append(f'<text x="{x+CELL_W/2-4}" y="{oy-5}" fill="#888" font-size="10" text-anchor="middle">c{c}</text>')
    # Grid
    for r in range(ROWS):
        y = oy + r * CELL_H
        svg.append(f'<text x="{ox-5}" y="{y+CELL_H/2+4}" fill="#666" font-size="9" text-anchor="end">r{r}</text>')
        for c in range(COLS):
            phys = xor_physical_col(r, c)
            x = ox + 40 + c * CELL_W
            svg.append(f'<rect x="{x}" y="{y}" width="{CELL_W-2}" height="{CELL_H-2}" rx="3" fill="{COLORS[phys]}"/>')
            svg.append(f'<text x="{x+CELL_W/2-4}" y="{y+CELL_H/2+4}" font-size="10" text-anchor="middle" fill="black" font-weight="bold">{phys}</text>')

    # Legend
    ly = oy + ROWS * CELL_H + 25
    svg.append(f'<text x="{ox}" y="{ly}" fill="#7fdbff" font-size="12">Physical column:</text>')
    for i in range(COLS):
        lx = ox + 100 + i * 45
        svg.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" rx="2" fill="{COLORS[i]}"/>')
        svg.append(f'<text x="{lx+18}" y="{ly-1}" fill="#eee" font-size="11">={i}</text>')

    svg.append('</svg>')

    out = '\n'.join(svg)
    with open('xor_shift_visualization.svg', 'w') as f:
        f.write(out)
    print('Saved xor_shift_visualization.svg')

if __name__ == '__main__':
    main()
