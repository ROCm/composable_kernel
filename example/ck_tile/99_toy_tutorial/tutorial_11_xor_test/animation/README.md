# XOR Tile Window FP16 Animation

This folder contains a 3b1b-style HTML/JS animation for the descriptor transforms in:

- `example/ck_tile/99_toy_tutorial/tutorial_11_xor_test/xor_test_with_tile_window.cpp`
- Block: Step 1 to Step 4 (`lds_desc_0`, XOR permute, unmerge, merge)

## Files

- `index.html` - app shell and scene controls
- `styles.css` - visual theme and grid/panel styling
- `app.js` - deterministic data generation, exact mapping, and scene rendering
- `01_mapping_spec.md` - exact fp16 equations used by animation
- `02_storyboard_ascii.md` - scene-by-scene ASCII storyboard

## Run

Open directly:

- Open `index.html` in a browser

or run a local static server from this folder:

```bash
python3 -m http.server 8000
```

Then browse:

- `http://localhost:8000/index.html`

## Scene guide

- Scene 0: Initial logical tile `[64 x 32]`
- Scene 1: First transform = combine `kKPack` (`64x32 -> 64x4` block matrix)
- Scene 2: XOR impact shown as color shuffle (`before XOR` vs `after XOR`) in block mode
- Scene 3: Unmerge shown as tiled grid (`L0/L1`, each `32x4`)
- Scene 4: `MLdsLayer=2` shown in same column lane (top `L0`, bottom `L1`)
- Scene 5: Merge back to final `[64,32]` matrix

## Mapping constants (fp16)

- `kM=64`
- `kK=32`
- `kKPack=8`
- `DataTypeSize=2`
- `MLdsLayer=2`

Derived:

- `A=8`, `B=32`, `C=8`, `L=2`, `K0=4`

## Quick verification checklist

- [ ] Early scenes are uncluttered (single matrix flow, no panel overload)
- [ ] First transformation explicitly combines `kKPack` into `64x4` block mode
- [ ] XOR impact is visible as shuffle via colors in block mode
- [ ] Unmerge is shown as a tiled grid for `(L, Bxor, K0)`
- [ ] `MLdsLayer=2` appears as two row slots in the same lane cell (not separate lower grid)
- [ ] Final scene returns to a clear `64x32` matrix view
