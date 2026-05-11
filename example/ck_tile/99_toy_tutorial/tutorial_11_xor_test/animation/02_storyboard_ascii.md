# XOR Tile Window FP16 Storyboard (ASCII)

This storyboard is the direct pre-production script for the HTML/JS animation.

## Visual language

- Literal matrix requirement: every matrix shown as a full grid (no tensor cubes).
- Higher-than-2D states: shown as tiled 2D panels.
- `kKPack` merge: shown as block merge (4 blocks width 8 become one width 32).
- Highlight token used across all scenes: `v*` (same tracked element through transforms).

---

## Scene 0: Setup and constants

Caption:

```text
We start from a logical LDS tile of shape MxK = 64x32 (fp16).
```

ASCII:

```text
Logical tile [M,K] = [64,32]

M (rows)
^
|   +--------------------------------------------------------------+
|   | [ ][ ][ ][ ] ... [ ]  <- K=32 columns                       |
|   | [ ][ ][ ][ ] ... [ ]                                        |
|   | [ ][ ][ ][ ] ... [ ]                                        |
|   |          ... total 64 rows ...                              |
|   | [ ][ ][ ][ ] ... [ ]                                        |
|   +--------------------------------------------------------------+ ---> K

fp16 constants:
kM=64, kK=32, kKPack=8, DataTypeSize=2
MLdsLayer = max(1, 32*4/(32*2)) = 2
```

Transition cue:

```text
Split K into (K0=4 blocks, KPack=8 each), and route through A/B/C indexing.
```

---

## Scene 1: Step 1 reshape to [A,B,C] = [8,32,8]

Caption:

```text
Reshape [64,32] into 8 panels. Each panel is a full 32x8 grid.
```

ASCII:

```text
Step1: shape [A,B,C] = [8,32,8]

A0                  A1                  A2                  A3
+----------------+  +----------------+  +----------------+  +----------------+
| 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |
| (rows B, colsC)|  | (rows B, colsC)|  | (rows B, colsC)|  | (rows B, colsC)|
+----------------+  +----------------+  +----------------+  +----------------+

A4                  A5                  A6                  A7
+----------------+  +----------------+  +----------------+  +----------------+
| 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |
| (rows B, colsC)|  | (rows B, colsC)|  | (rows B, colsC)|  | (rows B, colsC)|
+----------------+  +----------------+  +----------------+  +----------------+

Example tracked cell:
v* at (a,b,c) = (5, 9, 3)
```

Transition cue:

```text
Apply XOR to panel index and row index coupling: b' = b xor a.
```

---

## Scene 2: Step 2 XOR permute (rows within each A panel)

Caption:

```text
Panel count stays 8. Each panel remains 32x8. Only row placement is permuted.
```

ASCII:

```text
Step2 mapping:
(a,b,c) -> (a, b xor a, c)

Before (Step1 panels)                  After (XOR-permuted panels)

A0: rows 0..31                         A0: rows XOR with a=0 (unchanged)
A1: rows 0..31                         A1: rows XOR with a=1
A2: rows 0..31                         A2: rows XOR with a=2
...
A7: rows 0..31                         A7: rows XOR with a=7

Tracked element:
v*: (a,b,c)=(5,9,3) -> (5, 9 xor 5, 3) = (5,12,3)
```

Transition cue:

```text
Now split A=8 into two factors L=2 and K0=4.
```

---

## Scene 3: Step 3 unmerge A -> [L,K0], shape [2,32,4,8]

Caption:

```text
No cubes: show as 2 layer groups, each with 4 tiled 32x8 panels.
```

ASCII:

```text
Step3: [A,B,C]=[8,32,8] -> [L,B,K0,C]=[2,32,4,8]
where A = L*4 + K0

Layer L0:
K0_0                K0_1                K0_2                K0_3
+----------------+  +----------------+  +----------------+  +----------------+
| 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |
+----------------+  +----------------+  +----------------+  +----------------+

Layer L1:
K0_0                K0_1                K0_2                K0_3
+----------------+  +----------------+  +----------------+  +----------------+
| 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |  | 32x8 full grid |
+----------------+  +----------------+  +----------------+  +----------------+

Tracked element:
a=5 => (l,k0) = (1,1), so
v*: (5,12,3) -> (l=1, b=12, k0=1, c=3)
```

Transition cue:

```text
Merge vertical blocks for M and horizontal blocks for K.
```

---

## Scene 4: Step 4 merge back to [M,K] = [64,32]

Caption:

```text
M merge is vertical (B with L). K merge is horizontal (K0 with KPack).
```

ASCII:

```text
Step4 equations:
m = b*2 + l
k = k0*8 + c

K merge (block view):
[K0_0 width8][K0_1 width8][K0_2 width8][K0_3 width8] -> width 32

M merge (stack view):
[L0 rows 0..31] stacked with [L1 rows 0..31] -> 64 rows

Final [64x32] full grid:
+--------------------------------------------------------------+
| [ ][ ][ ][ ] ... [ ]                                        |
| [ ][ ][ ][ ] ... [ ]                                        |
| ...                                                          |
| [ ][ ][ ][ ] ... [ ]                                        |
+--------------------------------------------------------------+

Tracked element:
v*: (l=1,b=12,k0=1,c=3) -> (m,k)=(12*2+1, 1*8+3)=(25,11)
```

Transition cue:

```text
Overlay with tile_window usage to close the loop.
```

---

## Scene 5: Verification overlay (context in kernel)

Caption:

```text
This transformed descriptor is exactly what the tile_window LDS view uses.
```

ASCII:

```text
global_in_window --load_tile--> reg_tile --store_tile--> lds_window(lds_desc XOR)
                                                    block_sync_lds
global_out_window <--store_tile-- reg_tile_out <--load_tile-- lds_window(lds_desc XOR)

Key point:
logical 64x32 shape is preserved for operations,
while physical LDS placement is XOR-permuted.
```

End card:

```text
Same logical tile.
Different physical layout.
Fewer bank hot-spots for strided patterns.
```
