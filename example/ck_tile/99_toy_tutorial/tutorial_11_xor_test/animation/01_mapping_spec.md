# FP16 Mapping Spec for XOR Tile Window (Step 1-4)

This document fixes the exact constants and index equations used by the animation for:

- `example/ck_tile/99_toy_tutorial/tutorial_11_xor_test/xor_test_with_tile_window.cpp`
- Descriptor transform block at Step 1 to Step 4

## Constants (fp16 case)

- `kM = 64`
- `kK = 32`
- `kKPack = 8`
- `DataTypeSize = 2`
- `MLdsLayer = max(1, 32*4/(kK*DataTypeSize)) = max(1, 128/64) = 2`

Derived factors:

- `A = kK/kKPack * MLdsLayer = 8`
- `B = kM/MLdsLayer = 32`
- `C = kKPack = 8`
- `L = MLdsLayer = 2`
- `K0 = kK/kKPack = 4`

## Step 1: `lds_desc_0` reshape

Shape is `[A, B, C] = [8, 32, 8]` with strides:

- `stride_A = 8`
- `stride_B = 64`
- `stride_C = 1`

Address expression:

- `offset_step1(a,b,c) = a*8 + b*64 + c`

In the animation, this is shown as 8 tiled panels (`A0..A7`), each panel a literal `32x8` grid (`B x C`).

## Step 2: XOR transform on `(B, A)`

The visualization uses the XOR permutation on the pair `(a,b)`:

- `b_xor = b xor a`
- `a_xor = a`
- `c_xor = c`

So the displayed mapping is:

- `(a,b,c) -> (a, b xor a, c)`

The panel count and panel shape stay the same (`8` panels of `32x8`), but rows are permuted inside each panel.

## Step 3: Unmerge `A=8` into `(L=2, K0=4)`

From Step 2 tuple `(a_xor, b_xor, c)`:

- `l = floor(a_xor / K0) = floor(a_xor / 4)` in `[0,1]`
- `k0 = a_xor % K0` in `[0,3]`

Output tuple order in this file is `[L, B, K0, C]`, so:

- `(l, b_xor, k0, c)`

Visualization layout:

- Two layer groups (`L0`, `L1`)
- Each layer contains four `K0` panels
- Each panel is still literal `32x8` (`B x C`)

## Step 4: Merge back to `[M, K]`

Merge operations in code:

- `M = merge(B, L)` via `sequence<1,0>`
- `K = merge(K0, C)` via `sequence<2,3>`

Animation equations:

- `m = b_xor * L + l = b_xor*2 + l` in `[0,63]`
- `k = k0 * C + c = k0*8 + c` in `[0,31]`

Final tuple:

- `(m,k)` with shape `[64,32]`

`kKPack` merge is shown as horizontal block merge:

- 4 blocks (`K0`) each width 8 (`C`) -> final width 32.

## Deterministic value labels used by animation

To track elements across scenes, each original `(m,k)` gets:

- `value = m*100 + k` (compact numeric label)

This keeps every cell unique while remaining readable.
