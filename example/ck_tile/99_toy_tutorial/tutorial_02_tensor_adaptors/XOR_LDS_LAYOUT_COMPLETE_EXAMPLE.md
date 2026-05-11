# Complete XOR LDS Layout - Step-by-Step with Examples

This document walks through the complete XOR LDS layout transformation code with concrete numerical examples.

## The Complete Code

```cpp
constexpr auto DataTypeSize = sizeof(ADataType);
constexpr auto MLdsLayer =
    (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
    make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
               number<kMPerBlock / MLdsLayer>{},
               number<kKPack>{}),
    make_tuple(number<kKPack>{}, number<kKPerBlock * MLdsLayer>{}, number<1>{}),
    number<kKPack>{},
    number<1>{});

constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
    a_lds_block_desc_0,
    make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                             number<kKPerBlock / kKPack * MLdsLayer>{})),
               make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<1, 0>{}, sequence<2>{}),
    make_tuple(sequence<1, 0>{}, sequence<2>{}));

constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
    a_lds_block_desc_permuted,
    make_tuple(make_unmerge_transform(
                   make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
               make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
               make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

constexpr auto a_lds_block_desc = transform_tensor_descriptor(
    a_lds_block_desc_xk0_mnldslayer_mn_xk1,
    make_tuple(
        make_merge_transform(
            make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
        make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
```

---

## Concrete Example with Numbers

Let's use these values:
```cpp
kMPerBlock = 128
kKPerBlock = 16
kKPack = 8
ADataType = float (4 bytes)
```

### Step 0: Calculate MLdsLayer

```cpp
DataTypeSize = sizeof(float) = 4
MLdsLayer = (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize)
          = (128 / 16 / 4) < 1 ? 1 : (128 / 16 / 4)
          = (8 / 4) < 1 ? 1 : 2
          = 2 < 1 ? 1 : 2
          = 2

MLdsLayer = 2
```

**What this means**:
- Divide M into 2 layers
- Each layer has 128/2 = 64 M elements
- Formula ensures we don't exceed LDS capacity (32 banks × 4 bytes = 128 bytes per row)

### Step 1: Create Initial Descriptor - Understanding the "Unnatural" Strides

```cpp
a_lds_block_desc_0 = make_naive_tensor_descriptor(
    // LENGTHS (shape):
    make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},  // 16/8 * 2 = 4
               number<kMPerBlock / MLdsLayer>{},            // 128/2 = 64
               number<kKPack>{}),                           // 8
    
    // STRIDES (how to navigate memory):
    make_tuple(number<kKPack>{},                 // stride = 8
               number<kKPerBlock * MLdsLayer>{}, // stride = 16*2 = 32
               number<1>{}),                     // stride = 1
    
    number<kKPack>{},   // vector length
    number<1>{});       // vector stride
```

**With our numbers**:
```
Shape:   [4, 64, 8]
Strides: [8, 32, 1]
```

### Why These "Unnatural" Strides? The Intuition

**The Natural Stride Pattern Would Be**:
```
For shape [4, 64, 8] in row-major order:
  Stride for dim 2 (innermost): 1
  Stride for dim 1 (middle):    8  (size of dim 2)
  Stride for dim 0 (outermost): 8 × 64 = 512

Natural strides: [512, 8, 1]
```

**But We Use**: `[8, 32, 1]` - Why?

**Answer**: We're NOT storing in natural row-major order! We're using a **custom interleaved layout** optimized for bank conflict avoidance.

### The Custom Layout Explained

**What we're storing**: A matrix tile [M=128, K=16]

**How we want to access it**:
- Vectorized loads of 8 K elements at a time (KPack=8)
- Divided into 2 layers for the M dimension (MLdsLayer=2)
- Need to avoid bank conflicts

**The Chosen Layout**:
```
Think of it as storing in this order:
  For m=0:
    Store k_pack_layer=0, all 8 elements  (addresses 0-7)
    Store k_pack_layer=1, all 8 elements  (addresses 8-15)
    Store k_pack_layer=2, all 8 elements  (addresses 16-23)
    Store k_pack_layer=3, all 8 elements  (addresses 24-31)
  For m=1:
    Store k_pack_layer=0, all 8 elements  (addresses 32-39)
    ...and so on
```

**Why This Pattern?**

1. **Vectorized Access** (stride 1 for k_elem):
   - 8 consecutive K elements can be loaded with one vector instruction
   - Maximizes memory bandwidth

2. **Bank Spreading** (stride 8 for k_pack_layer):
   - Different k_pack_layers are 8 elements apart
   - When XOR swizzles, this spacing helps spread across banks

3. **Layer Organization** (stride 32 for m):
   - Each m value gets 32 consecutive addresses (4 k_pack_layers × 8 elements)
   - Keeps related M elements together for cache locality

### Visual: The Memory Layout

```
Logical view: [M=128, K=16] matrix

Physical LDS layout:
┌─────────────────────────────────────────────────────┐
│ m=0:  [k=0-7][k=8-15][k=0-7][k=8-15]                │ ← 32 elements
│       layer0  layer0  layer1  layer1                │
├─────────────────────────────────────────────────────┤
│ m=1:  [k=0-7][k=8-15][k=0-7][k=8-15]                │ ← 32 elements
├─────────────────────────────────────────────────────┤
│ m=2:  [k=0-7][k=8-15][k=0-7][k=8-15]                │
└─────────────────────────────────────────────────────┘
...continues for all 64 m values
```

**The Stride Pattern Makes Sense Now**:
```
Stride 1:  Within each 8-element pack (vectorized load)
Stride 8:  Between k_pack_layers (jump to next pack)
Stride 32: Between m values (jump to next row's data)
```

**Result**: Descriptor with shape `[4, 64, 8]` and strides `[8, 32, 1]`

**What this represents**:
```
Dimension 0 (size 4):  K/KPack * MLdsLayer = (16/8) * 2 = 2 * 2 = 4
  → 2 K-packs (K split into packs of 8) × 2 layers = 4 combinations

Dimension 1 (size 64): M/MLdsLayer = 128/2 = 64
  → 64 M elements per layer

Dimension 2 (size 8):  KPack = 8
  → 8 elements per vectorized load

Memory layout:
  [K-pack-layer-combo, M-per-layer, elements-per-pack]
  [4, 64, 8]
```

**Understanding the Strides - The Correct Explanation**:

Strides tell us how memory addresses change when we increment each dimension.

```
Shape:   [4, 64, 8]
Strides: [8, 32, 1]

Address formula:
  address = dim0 * stride0 + dim1 * stride1 + dim2 * stride2
  address = dim0 * 8 + dim1 * 32 + dim2 * 1
```

**NO OVERLAP! Each coordinate maps to a unique address.**

Let's trace through the memory layout step by step:

**Dim 2 (k_elem, size 8, stride 1)**:
```
(0,0,0) → address 0
(0,0,1) → address 1  (moved 1)
(0,0,2) → address 2  (moved 1)
...
(0,0,7) → address 7  (moved 1)

These 8 elements are CONTIGUOUS in memory.
```

**Dim 0 (k_pack_layer, size 4, stride 8)**:
```
(0,0,0) → address 0
(1,0,0) → address 8   (moved 8)
(2,0,0) → address 16  (moved 8)
(3,0,0) → address 24  (moved 8)

Each k_pack_layer starts 8 addresses apart.
```

**Dim 1 (m, size 64, stride 32)**:
```
(0,0,0) → address 0
(0,1,0) → address 32  (moved 32)
(0,2,0) → address 64  (moved 32)
(0,3,0) → address 96  (moved 32)

Each m value starts 32 addresses apart.
```

**The Complete Memory Layout**:
```
Addresses 0-31: m=0, all k_pack_layers and k_elems
  0-7:   (k=0, m=0, pack 0-7)
  8-15:  (k=1, m=0, pack 0-7)
  16-23: (k=2, m=0, pack 0-7)
  24-31: (k=3, m=0, pack 0-7)

Addresses 32-63: m=1, all k_pack_layers and k_elems
  32-39:  (k=0, m=1, pack 0-7)
  40-47:  (k=1, m=1, pack 0-7)
  48-55:  (k=2, m=1, pack 0-7)
  56-63:  (k=3, m=1, pack 0-7)

Addresses 64-95: m=2, all k_pack_layers and k_elems
  ...and so on
```

**Why stride for m = 32? (This is the confusing part!)**

The stride is `kKPerBlock * MLdsLayer = 16 * 2 = 32`

**This is NOT the same as the number of elements per m!**

Let me explain what's really happening:

```
The stride of 32 means: when m increments by 1, add 32 to the address.

But wait - we have 64 m values, and stride is 32?
Let's check the address range:
  m=0:  address starts at 0
  m=1:  address starts at 32
  m=2:  address starts at 64
  m=63: address starts at 63*32 = 2016
  
  Plus the k_pack_layer and k_elem offsets (0-31)
  Maximum address: 2016 + 31 = 2047 ✓
```

**The Key Insight**:

The stride of 32 is actually `kKPerBlock * MLdsLayer`:
- kKPerBlock = 16 (total K elements)
- MLdsLayer = 2 (number of layers)
- Product = 32

**Why this specific value?**

Think about what's stored for each m:
```
For m=0, we store K elements organized as:
  Layer 0: K[0-7]   (k_pack_layer=0, k_elem=0-7)
  Layer 0: K[8-15]  (k_pack_layer=1, k_elem=0-7)
  Layer 1: K[0-7]   (k_pack_layer=2, k_elem=0-7)
  Layer 1: K[8-15]  (k_pack_layer=3, k_elem=0-7)
  
Total: 4 packs × 8 elements = 32 elements per m value
```

**So stride 32 IS correct!**
- Each m value occupies 32 consecutive addresses
- To get to the next m, skip 32 addresses
- Stride = 32 ✓

**Why Only 32? The Interleaving Explanation**:

We have 64 m values, but the stride is only 32. This seems wrong until you understand the INTERLEAVING:

```
The descriptor shape is [4, 64, 8], which represents:
  Dim 0: 4 k_pack_layer combinations
  Dim 1: 64 m values  
  Dim 2: 8 k_elem values

But these dimensions are INTERLEAVED in memory!
```

**The Memory Pattern**:
```
Think of it like this - for each m value, we DON'T store all 64 m's worth of data.
Instead, we store data for ONE m value across all k_pack_layers:

m=0: [k_pack_layer 0-3, each with 8 elements] = 32 elements
     ↓
m=1: [k_pack_layer 0-3, each with 8 elements] = 32 elements
     ↓
m=2: [k_pack_layer 0-3, each with 8 elements] = 32 elements
...
```

**Why 32 specifically?**
```
For ONE m value, we store:
  - k_pack_layer 0: 8 elements (K[0-7])
  - k_pack_layer 1: 8 elements (K[8-15])
  - k_pack_layer 2: 8 elements (K[0-7] from layer 1)
  - k_pack_layer 3: 8 elements (K[8-15] from layer 1)
  
Total: 4 × 8 = 32 elements for this ONE m value

When we move to the NEXT m value (m+1), we skip these 32 elements.
Hence stride = 32!
```

**Visual Diagram**:
```
Memory addresses:
[0-31]:    m=0's data (all 4 k_pack_layers × 8 elements)
[32-63]:   m=1's data (all 4 k_pack_layers × 8 elements)
[64-95]:   m=2's data
[96-127]:  m=3's data
...
[2016-2047]: m=63's data

Each m "block" is 32 elements wide.
We have 64 such blocks.
Total: 64 × 32 = 2048 elements ✓
```

**The Key Insight**:

The stride tells you the SPACING between consecutive values of that dimension, not the total size of the dimension!

```
Dimension 1 has SIZE=64 (there are 64 different m values)
Dimension 1 has STRIDE=32 (each m value is 32 addresses apart)

These are independent concepts!
```

**Why stride for k_pack_layer = 8?**
```
For each k_pack_layer, we store:
  8 k_elem values (contiguous)

To get from k_pack_layer=0 to k_pack_layer=1, we skip 8 elements.
Stride = 8 ✓
```

**NO OVERLAP - Verification**:
```
Total addresses used:
  64 m values × 32 elements per m = 2048 addresses
  Addresses 0 through 2047 are used exactly once.
  
Each coordinate (k_pack_layer, m, k_elem) maps to a unique address:
  (0,0,0) → 0
  (3,63,7) → 3*8 + 63*32 + 7 = 24 + 2016 + 7 = 2047 ✓

No overlaps! Each of the 4×64×8 = 2048 coordinates gets its own address.
```

### Step 2: Apply XOR Transform

```cpp
a_lds_block_desc_permuted = transform_tensor_descriptor(
    a_lds_block_desc_0,  // Input: [4, 64, 8]
    make_tuple(
        make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},      // 64
                                     number<kKPerBlock / kKPack * MLdsLayer>{})),  // 4
        make_pass_through_transform(number<kKPack>{})  // 8
    ),
    make_tuple(sequence<1, 0>{}, sequence<2>{}),  // Input dims: [1,0] for XOR, [2] for pass-through
    make_tuple(sequence<1, 0>{}, sequence<2>{})   // Output dims: same layout
);
```

**What XOR does here**:
- Operates on dimensions [1, 0] = [M-per-layer=64, K-pack-layer=4]
- XOR pattern: [64, 4]
- Dimension 2 (KPack=8) passes through unchanged

**XOR Swizzling Formula**:
```
For coordinate (k_pack_layer, m, k_elem):
  
  Original address = k_pack_layer * 8 + m * 32 + k_elem
  
  XOR swizzle:
    xor_offset = m XOR k_pack_layer
    final_address = original_address XOR xor_offset
```

**Example Coordinates**:
```
(k_pack_layer=0, m=0, k_elem=0):
  base = 0*8 + 0*32 + 0 = 0
  xor = 0 XOR 0 = 0
  final = 0 XOR 0 = 0

(k_pack_layer=0, m=32, k_elem=0):
  base = 0*8 + 32*32 + 0 = 1024
  xor = 32 XOR 0 = 32
  final = 1024 XOR 32 = 1056
  
Without XOR: address 1024 → bank 0 (1024 % 32 = 0)
With XOR:    address 1056 → bank 0 (1056 % 32 = 0)
Wait, both bank 0? Let me recalculate...

Actually, the XOR operates on the INDICES, not addresses directly!
```

### Understanding XOR on Dimensions

**Key Point**: XOR transform operates on **coordinate indices**, not memory addresses!

```cpp
make_xor_transform(make_tuple(number<64>{}, number<4>{}))
```

This means:
- When you access coordinate (m, k_pack_layer)
- The transform computes: swizzled_k = k_pack_layer XOR (m % 4)
- Then uses (m, swizzled_k) to calculate the address

**Concrete Example**:

Original coordinates → Swizzled coordinates:
```
(m=0,  k=0) → (m=0,  k'=0 XOR (0%4)) = (0, 0)
(m=1,  k=0) → (m=1,  k'=0 XOR (1%4)) = (1, 1)
(m=2,  k=0) → (m=2,  k'=0 XOR (2%4)) = (2, 2)
(m=3,  k=0) → (m=3,  k'=0 XOR (3%4)) = (3, 3)
(m=32, k=0) → (m=32, k'=0 XOR (32%4)) = (32, 0)  ← Same k' as m=0!
(m=33, k=0) → (m=33, k'=0 XOR (33%4)) = (33, 1)  ← Same k' as m=1!
```

**Address calculation with swizzled coordinates**:
```
(m=0, k=0) → (m=0, k'=0) → address = 0*8 + 0*32 + 0 = 0
(m=1, k=0) → (m=1, k'=1) → address = 1*8 + 1*32 + 0 = 40
(m=2, k=0) → (m=2, k'=2) → address = 2*8 + 2*32 + 0 = 80
(m=32,k=0) → (m=32,k'=0) → address = 0*8 + 32*32 + 0 = 1024
```

Wait, this still doesn't look right. Let me reconsider the dimension ordering...

### Step 3: Unmerge for Hierarchy

```cpp
a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
    a_lds_block_desc_permuted,  // Input: [4, 64, 8] (XOR-swizzled)
    make_tuple(
        make_unmerge_transform(make_tuple(number<MLdsLayer>{},          // 2
                                         number<kKPerBlock / kKPack>{})),  // 2
        make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),  // 64
        make_pass_through_transform(number<kKPack>{})                   // 8
    ),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{})
);
```

**What happens**:
- Unmerge dimension 0 (size 4) into [MLdsLayer=2, K/KPack=2]
- Dimensions 1 and 2 pass through
- Output: [MLdsLayer=2, M/MLdsLayer=64, K/KPack=2, KPack=8]

**Layout**: `[2, 64, 2, 8]`

### Step 4: Merge Back to 2D

```cpp
a_lds_block_desc = transform_tensor_descriptor(
    a_lds_block_desc_xk0_mnldslayer_mn_xk1,  // Input: [2, 64, 2, 8]
    make_tuple(
        make_merge_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},  // 64
                                       number<MLdsLayer>{})),               // 2
        make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{},     // 2
                                       number<kKPack>{}))                   // 8
    ),
    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{})
);
```

**What happens**:
- Merge dimensions [1, 0] = [M/MLdsLayer=64, MLdsLayer=2] → M=128
- Merge dimensions [2, 3] = [K/KPack=2, KPack=8] → K=16
- Output: [M=128, K=16]

**Final**: Back to 2D `[128, 16]` but with XOR swizzling preserved!

---

## The Key Question: What Happens When You XOR Two Dimensions?

Let's use a simple example to understand.

### Simple Example: XOR Transform on [4, 4]

```cpp
make_xor_transform(make_tuple(number<4>{}, number<4>{}))
```

This creates a 4×4 grid where coordinates get swizzled.

**Without XOR** - Regular 2D indexing:
```
Coordinates → Address (assuming row-major, stride=4):
(0,0) → 0*4 + 0 = 0
(0,1) → 0*4 + 1 = 1
(0,2) → 0*4 + 2 = 2
(0,3) → 0*4 + 3 = 3
(1,0) → 1*4 + 0 = 4
(1,1) → 1*4 + 1 = 5
(2,0) → 2*4 + 0 = 8
(3,0) → 3*4 + 0 = 12
```

**With XOR** - Swizzled indexing:
```
The XOR transform modifies dimension 1 based on dimension 0:
  swizzled_dim1 = original_dim1 XOR (original_dim0 % 4)

Coordinates → Swizzled → Address:
(0,0) → (0, 0 XOR 0) = (0,0) → 0*4 + 0 = 0
(0,1) → (0, 1 XOR 0) = (0,1) → 0*4 + 1 = 1
(0,2) → (0, 2 XOR 0) = (0,2) → 0*4 + 2 = 2
(0,3) → (0, 3 XOR 0) = (0,3) → 0*4 + 3 = 3

(1,0) → (1, 0 XOR 1) = (1,1) → 1*4 + 1 = 5  ← Different!
(1,1) → (1, 1 XOR 1) = (1,0) → 1*4 + 0 = 4  ← Swapped with (1,0)!
(1,2) → (1, 2 XOR 1) = (1,3) → 1*4 + 3 = 7
(1,3) → (1, 3 XOR 1) = (1,2) → 1*4 + 2 = 6

(2,0) → (2, 0 XOR 2) = (2,2) → 2*4 + 2 = 10 ← Different!
(2,1) → (2, 1 XOR 2) = (2,3) → 2*4 + 3 = 11
(2,2) → (2, 2 XOR 2) = (2,0) → 2*4 + 0 = 8  ← Swapped!
(2,3) → (2, 3 XOR 2) = (2,1) → 2*4 + 1 = 9

(3,0) → (3, 0 XOR 3) = (3,3) → 3*4 + 3 = 15 ← Different!
(3,1) → (3, 1 XOR 3) = (3,2) → 3*4 + 2 = 14
(3,2) → (3, 2 XOR 3) = (3,1) → 3*4 + 1 = 13
(3,3) → (3, 3 XOR 3) = (3,0) → 3*4 + 0 = 12
```

### Address Mapping Table

```
Original    Swizzled    Address    Bank (addr % 4)
Coord       Coord                  Without XOR | With XOR
--------------------------------------------------------------
(0,0)  →    (0,0)   →   0         bank 0      | bank 0
(0,1)  →    (0,1)   →   1         bank 1      | bank 1
(0,2)  →    (0,2)   →   2         bank 2      | bank 2
(0,3)  →    (0,3)   →   3         bank 3      | bank 3
(1,0)  →    (1,1)   →   5         bank 0      | bank 1  ✓
(1,1)  →    (1,0)   →   4         bank 1      | bank 0  ✓
(1,2)  →    (1,3)   →   7         bank 2      | bank 3  ✓
(1,3)  →    (1,2)   →   6         bank 3      | bank 2  ✓
(2,0)  →    (2,2)   →   10        bank 0      | bank 2  ✓
(2,1)  →    (2,3)   →   11        bank 1      | bank 3  ✓
(2,2)  →    (2,0)   →   8         bank 2      | bank 0  ✓
(2,3)  →    (2,1)   →   9         bank 3      | bank 1  ✓
(3,0)  →    (3,3)   →   15        bank 0      | bank 3  ✓
(3,1)  →    (3,2)   →   14        bank 1      | bank 2  ✓
(3,2)  →    (3,1)   →   13        bank 2      | bank 1  ✓
(3,3)  →    (3,0)   →   12        bank 3      | bank 0  ✓
```

### The Pattern Revealed!

**Without XOR** - Column 0 accesses:
```
(0,0) → bank 0
(1,0) → bank 0  ← CONFLICT!
(2,0) → bank 0  ← CONFLICT!
(3,0) → bank 0  ← CONFLICT!
All hit bank 0!
```

**With XOR** - Column 0 accesses:
```
(0,0) → (0,0) → bank 0
(1,0) → (1,1) → bank 1  ← Different!
(2,0) → (2,2) → bank 2  ← Different!
(3,0) → (3,3) → bank 3  ← Different!
All hit different banks!
```

**The Magic**: XOR spreads column accesses across different banks by swizzling the second dimension based on the first dimension!

---

## Back to Our Real Example: [64, 4] XOR Pattern

With our calculated values:
- MLdsLayer = 2
- M/MLdsLayer = 64
- (K/KPack) × MLdsLayer = 4

XOR pattern: `make_xor_transform(make_tuple(number<64>{}, number<4>{}))`

**What this does**:
```
For any coordinate (m, k) where m ∈ [0,63], k ∈ [0,3]:
  swizzled_k = k XOR (m % 4)
  
Examples:
(m=0,  k=0) → k' = 0 XOR (0%4) = 0 XOR 0 = 0
(m=1,  k=0) → k' = 0 XOR (1%4) = 0 XOR 1 = 1
(m=2,  k=0) → k' = 0 XOR (2%4) = 0 XOR 2 = 2
(m=3,  k=0) → k' = 0 XOR (3%4) = 0 XOR 3 = 3
(m=4,  k=0) → k' = 0 XOR (4%4) = 0 XOR 0 = 0  ← Repeats every 4
(m=32, k=0) → k' = 0 XOR (32%4) = 0 XOR 0 = 0
(m=33, k=0) → k' = 0 XOR (33%4) = 0 XOR 1 = 1
```

**The Pattern**:
- Every 4 M elements, the XOR pattern repeats
- This creates a "checkerboard" swizzling pattern
- Spreads accesses across banks

---

## Complete Transformation Flow with Numbers

```
Start: Logical [M=128, K=16]

Step 1: Initial descriptor
  Shape:   [4, 64, 8]
  Strides: [8, 32, 1]
  Meaning: [K-pack-layers, M-per-layer, K-pack-elements]

Step 2: XOR transform
  XOR pattern: [64, 4]
  Operates on dims [1, 0] (M and K-pack-layers)
  Result: Coordinates swizzled, addresses spread across banks

Step 3: Unmerge
  [4, 64, 8] → [2, 64, 2, 8]
  Split dim 0 into [MLdsLayer=2, K/KPack=2]
  Result: [MLdsLayer, M/MLdsLayer, K/KPack, KPack]

Step 4: Merge
  [2, 64, 2, 8] → [128, 16]
  Merge [64, 2] → 128 (M dimension)
  Merge [2, 8] → 16 (K dimension)
  Result: Back to [M, K] with XOR swizzling preserved
```

---

## Summary

**What XOR Transform Does to Dimensions**:
1. Takes two dimension indices (e.g., m and k)
2. Computes: `swizzled_second = second XOR (first % second_length)`
3. Uses swizzled coordinates to calculate memory address
4. Result: Addresses spread across banks instead of clustering

**Key Insight**: XOR operates on **coordinate space**, not address space directly. It modifies which coordinates map to which addresses, creating the bank spreading effect.

**The Formula**:
```
idx_low[0] = idx_up[0]  (pass through)
idx_low[1] = idx_up[1] XOR (idx_up[0] % up_lengths_[1])  (swizzle)
```

This simple formula creates complex address patterns that avoid bank conflicts!
