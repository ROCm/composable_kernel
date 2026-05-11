# Understanding chain_tensor_adaptors - A Deep Dive

## Overview

`chain_tensor_adaptors` composes two tensor adaptors sequentially, where the output dimensions of the first adaptor become the input dimensions of the second adaptor.

```
Adaptor0: [Bottom0] -> [Top0]
Adaptor1: [Bottom1] -> [Top1]
Chained:  [Bottom0] -> [Top1]
```

**Key Constraint**: `Top0` must match `Bottom1` in number of dimensions.

---

## The Hidden Dimension ID System

Each tensor adaptor uses a system of "hidden dimension IDs" to track dimensions through transformations:

- **Bottom dimensions**: Input dimensions (e.g., original [M, K])
- **Top dimensions**: Output dimensions (e.g., transformed [M0, M1, K0, K1])
- **Hidden dimensions**: Internal dimension IDs used to track transformations

### Example: Simple Adaptor
```cpp
// Adaptor: [M, K] -> [M0, M1, K]
// Hidden IDs might be:
//   Bottom: [0, 1]        (M=0, K=1)
//   Top:    [2, 3, 1]     (M0=2, M1=3, K=1)
```

The hidden ID system allows tracking which dimensions come from which transformations.

---

## The Challenge: Merging Two Adaptors

When chaining two adaptors, we need to:

1. **Combine all transformations** from both adaptors
2. **Ensure unique hidden IDs** (no ID conflicts between adaptors)
3. **Match connecting dimensions** (Top0 = Bottom1)
4. **Preserve bottom and top** (Bottom0 and Top1)

### The Problem: ID Conflicts

```
Adaptor0 hidden IDs: [0, 1, 2, 3]
Adaptor1 hidden IDs: [0, 1, 2, 3, 4]  ← Conflicts with Adaptor0!
```

We need to shift Adaptor1's IDs to avoid conflicts.

---

## Step-by-Step Algorithm

### Step 1: Find Maximum Hidden ID in Adaptor0

```cpp
constexpr index_t adaptor0_max_hidden_id = [&]() {
    index_t adaptor0_max_hidden_id_ = numeric<index_t>::min();
    
    // Scan all transforms in adaptor0
    static_for<0, TensorAdaptor0::get_num_of_transform(), 1>{}([&](auto itran) {
        // Check all lower dimension IDs
        static_for<0, ndim_low, 1>{}([&](auto idim_low) {
            adaptor0_max_hidden_id_ = max(
                adaptor0_max_hidden_id_,
                TensorAdaptor0::get_lower_dimension_hidden_idss()[itran][idim_low].value
            );
        });
        
        // Check all upper dimension IDs
        static_for<0, ndim_up, 1>{}([&](auto idim_up) {
            adaptor0_max_hidden_id_ = max(
                adaptor0_max_hidden_id_,
                TensorAdaptor0::get_upper_dimension_hidden_idss()[itran][idim_up].value
            );
        });
    });
    
    return adaptor0_max_hidden_id_;
}();
```

**Purpose**: Find the highest hidden ID used in Adaptor0.

**Example**: If Adaptor0 uses IDs [0, 1, 2, 3], then `adaptor0_max_hidden_id = 3`.

---

### Step 2: Find Minimum Hidden ID in Adaptor1 (Excluding Bottom)

```cpp
constexpr index_t adaptor1_min_hidden_id = [&]() {
    index_t adaptor1_min_hidden_id_ = numeric<index_t>::max();
    
    static_for<0, TensorAdaptor1::get_num_of_transform(), 1>{}([&](auto itran) {
        // Check lower dimensions (but skip bottom dimensions)
        static_for<0, ndim_low, 1>{}([&](auto idim_low) {
            constexpr index_t low_dim_hidden_id = 
                TensorAdaptor1::get_lower_dimension_hidden_idss()[itran][idim_low].value;
            
            bool is_bottom_dim = false;
            static_for<0, TensorAdaptor1::get_num_of_bottom_dimension(), 1>{}([&](auto i) {
                if constexpr(low_dim_hidden_id == 
                             TensorAdaptor1::get_bottom_dimension_hidden_ids()[i]) {
                    is_bottom_dim = true;
                }
            });
            
            if(!is_bottom_dim) {
                adaptor1_min_hidden_id_ = min(adaptor1_min_hidden_id_, low_dim_hidden_id);
            }
        });
        
        // Check all upper dimensions
        static_for<0, ndim_up, 1>{}([&](auto idim_up) {
            adaptor1_min_hidden_id_ = min(
                adaptor1_min_hidden_id_,
                TensorAdaptor1::get_upper_dimension_hidden_idss()[itran][idim_up].value
            );
        });
    });
    
    return adaptor1_min_hidden_id_;
}();
```

**Purpose**: Find the lowest hidden ID in Adaptor1 that's NOT a bottom dimension.

**Why exclude bottom dimensions?** Bottom dimensions will be matched with Top0 dimensions, so they don't need shifting.

**Example**: If Adaptor1 uses IDs [0, 1, 2, 3, 4] where [0, 1] are bottom dims, then `adaptor1_min_hidden_id = 2`.

---

### Step 3: Calculate the Shift Amount

```cpp
constexpr index_t adaptor1_hidden_id_shift = 
    adaptor0_max_hidden_id + 1 - adaptor1_min_hidden_id;
```

**Purpose**: Calculate how much to shift Adaptor1's IDs so its minimum non-bottom ID starts right after Adaptor0's maximum ID.

**Why subtract `adaptor1_min_hidden_id`?**

The key insight is that we want to **relocate** Adaptor1's ID range, not just shift everything by a fixed amount.

**Concrete Example**:

```
Adaptor0 uses IDs: [0, 1, 2, 3]
  - max_id = 3
  - Next available ID = 4

Adaptor1 original IDs: [0, 1, 5, 6, 7]
  - Bottom IDs: [0, 1] (will be matched, not shifted)
  - Non-bottom IDs: [5, 6, 7]
  - min_non_bottom = 5

Goal: Move Adaptor1's non-bottom IDs [5, 6, 7] to start at 4
```

**Without the subtraction** (naive approach):
```
shift = adaptor0_max_hidden_id + 1 = 4
Adaptor1 IDs after shift: [4, 5, 9, 10, 11]
                                   ↑   ↑   ↑
                                   Starts at 9, not 4!
                                   Wastes IDs 4-8
```

**With the subtraction** (correct approach):
```
shift = adaptor0_max_hidden_id + 1 - adaptor1_min_hidden_id
      = 3 + 1 - 5
      = -1

Adaptor1 IDs after shift: [-1, 0, 4, 5, 6]
                           ↑   ↑  ↑  ↑  ↑
                           Bottom dims (will be matched)
                           Non-bottom starts at 4 ✓
```

**The Formula Explained**:
```
new_id = old_id + shift
new_id = old_id + (adaptor0_max + 1 - adaptor1_min)

For adaptor1_min:
  new_id = adaptor1_min + (adaptor0_max + 1 - adaptor1_min)
         = adaptor0_max + 1  ✓

This ensures the minimum non-bottom ID lands exactly at the first available slot!
```

**Another Example**:
```
Adaptor0 IDs: [0, 1, 2]  → max = 2
Adaptor1 IDs: [0, 1, 10, 11, 12]  → min_non_bottom = 10

shift = 2 + 1 - 10 = -7

After shift: [0-7, 1-7, 10-7, 11-7, 12-7]
           = [-7, -6, 3, 4, 5]
             ↑   ↑   ↑  ↑  ↑
             Bottom (matched)  Non-bottom starts at 3 ✓
```

**Why This Matters**:
- Keeps hidden IDs **compact** and **sequential**
- Avoids wasting ID space
- Works regardless of what IDs Adaptor1 originally used
- The subtraction "normalizes" Adaptor1's ID range to start where Adaptor0 ended

---

### Step 4: Process Adaptor1's Lower Dimension IDs (THE CRITICAL MATCHING STEP)

This is where the two adaptors get connected! We need to:
1. First shift all IDs to avoid conflicts
2. Then **replace** bottom dimension IDs with the corresponding Top0 IDs

```cpp
constexpr auto low_dim_hidden_idss_1 = generate_tuple(
    [&](auto itran) {
        constexpr auto low_dim_hidden_ids_1 = 
            TensorAdaptor1::get_lower_dimension_hidden_idss()[itran];
        
        constexpr auto low_dim_hidden_ids_1_mod = [&]() constexpr {
            auto ids = to_multi_index(low_dim_hidden_ids_1);
            
            // Step 4a: Shift all IDs
            static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                ids(idim_low_1) += adaptor1_hidden_id_shift;
            });
            
            // Step 4b: Match bottom dimensions with Top0
            static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                static_for<0, ndim_bottom_1, 1>{}([&](auto idim_bottom_1) {
                    if constexpr(low_dim_hidden_ids_1[idim_low_1] == 
                                 TensorAdaptor1::get_bottom_dimension_hidden_ids()[idim_bottom_1]) {
                        // This is a bottom dim - match it with Top0
                        ids(idim_low_1) = 
                            TensorAdaptor0::get_top_dimension_hidden_ids()[idim_bottom_1];
                    }
                });
            });
            
            return ids;
        }();
        
        return generate_sequence_v2(
            [&](auto i) constexpr { return number<low_dim_hidden_ids_1_mod[i]>{}; },
            number<ndim_low_1>{}
        );
    },
    number<TensorAdaptor1::get_num_of_transform()>{}
);
```

---

## Deep Dive: The Bottom ID Matching Process

### Why Do We Need Matching?

When chaining adaptors, the **output of Adaptor0 must feed into the input of Adaptor1**. This means:

```
Adaptor0 produces: [M0, M1, K]  (Top0)
                    ↓   ↓   ↓
Adaptor1 expects:  [M0, M1, K]  (Bottom1)
```

These must refer to the **same dimensions** in the combined adaptor!

### The Matching Algorithm - Step by Step

Let's use a concrete example:

**Setup**:
```
Adaptor0: [M, K] -> [M0, M1, K]
  Bottom IDs: [0, 1]
  Top IDs:    [2, 3, 1]  ← M0=2, M1=3, K=1

Adaptor1: [M0, M1, K] -> [M0, M1, K0, K1]
  Bottom IDs: [0, 1, 2]  ← M0=0, M1=1, K=2
  Lower IDss for transforms: [[0], [1], [2]]
    - Transform 0 (PassThrough M0) uses lower ID 0
    - Transform 1 (PassThrough M1) uses lower ID 1
    - Transform 2 (Unmerge K) uses lower ID 2

Shift calculated: 1
```

**Step 4a: Initial Shift**
```
Original lower IDs: [[0], [1], [2]]
After shift by 1:   [[1], [2], [3]]
```

**Step 4b: The Matching Loop**

For each transform in Adaptor1, for each lower dimension ID:

**Transform 0, Lower ID = 0 (after shift = 1)**:
```
Check: Is original ID 0 a bottom dimension?
  → Yes! It's Bottom[0]
  
Action: Replace shifted ID with Adaptor0's Top[0]
  → ID 1 becomes ID 2 (Adaptor0's Top[0])
  
Why? Because Adaptor1's Bottom[0] (M0) should connect to Adaptor0's Top[0] (M0)
```

**Transform 1, Lower ID = 1 (after shift = 2)**:
```
Check: Is original ID 1 a bottom dimension?
  → Yes! It's Bottom[1]
  
Action: Replace shifted ID with Adaptor0's Top[1]
  → ID 2 becomes ID 3 (Adaptor0's Top[1])
  
Why? Because Adaptor1's Bottom[1] (M1) should connect to Adaptor0's Top[1] (M1)
```

**Transform 2, Lower ID = 2 (after shift = 3)**:
```
Check: Is original ID 2 a bottom dimension?
  → Yes! It's Bottom[2]
  
Action: Replace shifted ID with Adaptor0's Top[2]
  → ID 3 becomes ID 1 (Adaptor0's Top[2])
  
Why? Because Adaptor1's Bottom[2] (K) should connect to Adaptor0's Top[2] (K)
```

**Final Result**:
```
Lower IDs after matching: [[2], [3], [1]]
```

### Visual Representation of Matching

```
BEFORE MATCHING (after shift):
Adaptor1 Transform 0: uses lower ID 1 (shifted from 0)
Adaptor1 Transform 1: uses lower ID 2 (shifted from 1)
Adaptor1 Transform 2: uses lower ID 3 (shifted from 2)

MATCHING PROCESS:
Original ID 0 is Bottom[0] → connects to Top0[0] = 2
  Transform 0: ID 1 → ID 2 ✓

Original ID 1 is Bottom[1] → connects to Top0[1] = 3
  Transform 1: ID 2 → ID 3 ✓

Original ID 2 is Bottom[2] → connects to Top0[2] = 1
  Transform 2: ID 3 → ID 1 ✓

AFTER MATCHING:
Adaptor1 Transform 0: uses lower ID 2 (matched!)
Adaptor1 Transform 1: uses lower ID 3 (matched!)
Adaptor1 Transform 2: uses lower ID 1 (matched!)
```

### Why This Creates the Connection

After matching, when we trace through the combined adaptor:

```
Input coordinate [M=0, K=1]
  ↓
Adaptor0 Transform 0: Unmerge M (ID 0) → produces M0 (ID 2), M1 (ID 3)
Adaptor0 Transform 1: PassThrough K (ID 1) → produces K (ID 1)
  ↓
Intermediate state: [M0=2, M1=3, K=1]
  ↓
Adaptor1 Transform 0: PassThrough M0 (ID 2) → uses ID 2 ✓ (matched!)
Adaptor1 Transform 1: PassThrough M1 (ID 3) → uses ID 3 ✓ (matched!)
Adaptor1 Transform 2: Unmerge K (ID 1) → uses ID 1 ✓ (matched!)
  ↓
Output: [M0, M1, K0, K1]
```

The matching ensures that Adaptor1's transforms operate on the **exact same dimensions** that Adaptor0 produced!

### What Would Happen Without Matching?

```
Without matching, after shift:
Adaptor1 Transform 2 would use ID 3 for K

But Adaptor0 produces K at ID 1!

Result: Adaptor1 would try to read from ID 3, which doesn't contain K
→ BROKEN! The adaptors wouldn't connect properly.
```

### Key Takeaway

**Matching is the "glue"** that connects the two adaptors:
- Bottom IDs in Adaptor1 are **placeholders** saying "I need these inputs"
- Top IDs in Adaptor0 say "I produce these outputs"
- Matching **replaces the placeholders** with the actual IDs where those outputs live
- This creates a seamless data flow from Adaptor0's outputs to Adaptor1's inputs

---

## Code Walkthrough: Where Matching Happens in tensor_adaptor.hpp

Let me show you the exact code with detailed annotations:

```cpp
// This is inside chain_tensor_adaptors function in tensor_adaptor.hpp
// Around line 420-470

constexpr auto low_dim_hidden_idss_1 = generate_tuple(
    // For each transform in Adaptor1
    [&](auto itran) {
        // Get the original lower dimension IDs for this transform
        constexpr auto ndim_low_1 = 
            TensorAdaptor1::get_lower_dimension_hidden_idss()[itran].size();
        
        constexpr auto low_dim_hidden_ids_1 = 
            TensorAdaptor1::get_lower_dimension_hidden_idss()[itran];
        
        // Example: For transform 2 in Adaptor1 (Unmerge K)
        // low_dim_hidden_ids_1 = sequence<2>{}  (original K is at ID 2)
        
        constexpr auto low_dim_hidden_ids_1_mod = [&]() constexpr {
            auto low_dim_hidden_ids_1_mod_ = to_multi_index(low_dim_hidden_ids_1);
            
            // ============================================================
            // STEP 1: SHIFT ALL IDs (including bottom dims temporarily)
            // ============================================================
            static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                low_dim_hidden_ids_1_mod_(idim_low_1) += adaptor1_hidden_id_shift;
            });
            
            // After this step:
            // Transform 0: ID 0 → ID 1 (shift by 1)
            // Transform 1: ID 1 → ID 2 (shift by 1)
            // Transform 2: ID 2 → ID 3 (shift by 1)
            
            // ============================================================
            // STEP 2: MATCHING - Replace bottom IDs with Top0 IDs
            // ============================================================
            static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                // For each lower dimension in this transform
                
                static_for<0, ndim_bottom_1, 1>{}([&](auto idim_bottom_1) {
                    // Check each bottom dimension
                    
                    // THE MATCHING CONDITION:
                    if constexpr(low_dim_hidden_ids_1[idim_low_1] == 
                                 TensorAdaptor1::get_bottom_dimension_hidden_ids()[idim_bottom_1])
                    {
                        // *** THIS IS WHERE MATCHING HAPPENS! ***
                        
                        // If this lower ID matches a bottom dimension ID,
                        // replace it with the corresponding Top0 ID
                        
                        low_dim_hidden_ids_1_mod_(idim_low_1) = 
                            TensorAdaptor0::get_top_dimension_hidden_ids()[idim_bottom_1];
                        
                        // Example for Transform 2:
                        // - low_dim_hidden_ids_1[0] = 2 (original K ID)
                        // - Bottom[2] = 2 (K is the 3rd bottom dimension)
                        // - Condition is TRUE!
                        // - Replace: ID 3 (shifted) → ID 1 (Top0[2])
                        //   Because Adaptor0's Top[2] is K at ID 1
                    }
                });
            });
            
            // After matching:
            // Transform 0: ID 1 → ID 2 (matched with Top0[0])
            // Transform 1: ID 2 → ID 3 (matched with Top0[1])
            // Transform 2: ID 3 → ID 1 (matched with Top0[2])
            
            return low_dim_hidden_ids_1_mod_;
        }();
        
        return generate_sequence_v2(
            [&](auto i) constexpr { return number<low_dim_hidden_ids_1_mod[i]>{}; },
            number<ndim_low_1>{}
        );
    },
    number<TensorAdaptor1::get_num_of_transform()>{}
);
```

### Detailed Trace for Transform 2 (Unmerge K)

Let's trace exactly what happens for Adaptor1's Transform 2:

```cpp
// BEFORE PROCESSING:
// Transform 2 in Adaptor1: Unmerge(K -> K0, K1)
// Lower ID: [2]  (K is at position 2 in Bottom1)

// STEP 1: SHIFT
idim_low_1 = 0  (first and only lower dimension for this transform)
low_dim_hidden_ids_1[0] = 2  (original K ID)
low_dim_hidden_ids_1_mod_(0) = 2 + 1 = 3  (after shift)

// STEP 2: MATCHING LOOP
// Outer loop: idim_low_1 = 0
//   Inner loop: idim_bottom_1 = 0
//     Check: low_dim_hidden_ids_1[0] == Bottom[0]?
//            2 == 0? NO
//   
//   Inner loop: idim_bottom_1 = 1
//     Check: low_dim_hidden_ids_1[0] == Bottom[1]?
//            2 == 1? NO
//   
//   Inner loop: idim_bottom_1 = 2
//     Check: low_dim_hidden_ids_1[0] == Bottom[2]?
//            2 == 2? YES! ← MATCH FOUND!
//     
//     Action: low_dim_hidden_ids_1_mod_(0) = Top0[2]
//            = 1  (Adaptor0's Top[2] is K at ID 1)

// RESULT:
// Transform 2 now uses lower ID [1] instead of [3]
// This connects it to Adaptor0's K output!
```

### Why Each Check Matters

```cpp
if constexpr(low_dim_hidden_ids_1[idim_low_1] == 
             TensorAdaptor1::get_bottom_dimension_hidden_ids()[idim_bottom_1])
```

This condition asks: **"Is this lower dimension ID one of Adaptor1's bottom dimensions?"**

- `low_dim_hidden_ids_1[idim_low_1]`: The ORIGINAL (pre-shift) ID
- `Bottom[idim_bottom_1]`: One of Adaptor1's bottom dimension IDs

**Why use original ID?** Because we're checking which dimension this was in Adaptor1's original interface.

**When TRUE**: This dimension is an input to Adaptor1, so it must connect to Adaptor0's output.

**Action**: Replace the shifted ID with the actual ID where Adaptor0 produces this dimension.

### Complete Matching Table

```
Adaptor1 Transform | Original Lower ID | Is Bottom? | Bottom Index | Top0 ID | Final ID
-------------------|-------------------|------------|--------------|---------|----------
Transform 0        | 0                 | YES        | 0            | 2       | 2
Transform 1        | 1                 | YES        | 1            | 3       | 3
Transform 2        | 2                 | YES        | 2            | 1       | 1
```

Each bottom dimension gets matched with its corresponding Top0 dimension, creating the connection between the two adaptors.

---

### Step 5: Process Adaptor1's Upper Dimension IDs

```cpp
constexpr auto up_dim_hidden_idss_1 = generate_tuple(
    [&](auto itran) {
        constexpr auto up_dim_hidden_ids_1 = 
            TensorAdaptor1::get_upper_dimension_hidden_idss()[itran];
        
        constexpr auto up_dim_hidden_ids_1_mod = [&]() constexpr {
            auto ids = to_multi_index(up_dim_hidden_ids_1);
            
            // Simply shift all upper IDs
            static_for<0, ndim_up_1, 1>{}([&](auto idim_up_1) {
                ids(idim_up_1) += adaptor1_hidden_id_shift;
            });
            
            return ids;
        }();
        
        return generate_sequence_v2(
            [&](auto i) constexpr { return number<up_dim_hidden_ids_1_mod[i]>{}; },
            number<ndim_up_1>{}
        );
    },
    number<TensorAdaptor1::get_num_of_transform()>{}
);
```

**Purpose**: Shift all upper dimension IDs by the calculated shift amount.

**Example**:
```
Adaptor1 Upper IDs before: [3, 4]
After shift (by 2):        [5, 6]
```

---

### Step 6: Combine Everything

```cpp
// Concatenate all transforms
const auto all_transforms = 
    container_concat(adaptor0.get_transforms(), adaptor1.get_transforms());

// Concatenate all lower dimension ID sequences
constexpr auto all_low_dim_hidden_idss = 
    container_concat(TensorAdaptor0::get_lower_dimension_hidden_idss(), 
                     low_dim_hidden_idss_1);

// Concatenate all upper dimension ID sequences
constexpr auto all_up_dim_hidden_idss = 
    container_concat(TensorAdaptor0::get_upper_dimension_hidden_idss(), 
                     up_dim_hidden_idss_1);

// Bottom stays from Adaptor0
constexpr auto bottom_dim_hidden_ids = 
    TensorAdaptor0::get_bottom_dimension_hidden_ids();

// Top comes from Adaptor1 (shifted)
constexpr auto top_dim_hidden_ids = 
    TensorAdaptor1::get_top_dimension_hidden_ids() + number<adaptor1_hidden_id_shift>{};
```

---

## Complete Example Walkthrough

### Input Adaptors

**Adaptor0**: `[M, K] -> [M0, M1, K]`
```
Transforms: [Unmerge(M -> M0,M1), PassThrough(K)]
Bottom IDs: [0, 1]           (M=0, K=1)
Top IDs:    [2, 3, 1]        (M0=2, M1=3, K=1)
Lower IDss: [[0], [1]]       (transform 0 uses dim 0, transform 1 uses dim 1)
Upper IDss: [[2, 3], [1]]    (transform 0 produces dims 2,3; transform 1 produces dim 1)
```

**Adaptor1**: `[M0, M1, K] -> [M0, M1, K0, K1]`
```
Transforms: [PassThrough(M0), PassThrough(M1), Unmerge(K -> K0,K1)]
Bottom IDs: [0, 1, 2]        (M0=0, M1=1, K=2)
Top IDs:    [0, 1, 3, 4]     (M0=0, M1=1, K0=3, K1=4)
Lower IDss: [[0], [1], [2]]
Upper IDss: [[0], [1], [3, 4]]
```

### Step-by-Step Execution

**Step 1**: Find `adaptor0_max_hidden_id`
- Scan all IDs in Adaptor0: [0, 1, 2, 3]
- Maximum = **3**

**Step 2**: Find `adaptor1_min_hidden_id` (excluding bottom)
- Adaptor1 all IDs: [0, 1, 2, 3, 4]
- Bottom IDs: [0, 1, 2]
- Non-bottom IDs: [3, 4]
- Minimum non-bottom = **3**

**Step 3**: Calculate shift
```
shift = 3 + 1 - 3 = 1
```

**Step 4**: Process Adaptor1's lower IDs
```
Original lower IDss: [[0], [1], [2]]

After shift by 1:    [[1], [2], [3]]

After matching with Top0 [2, 3, 1]:
  - ID 0 is bottom[0] -> match with Top0[0] = 2
  - ID 1 is bottom[1] -> match with Top0[1] = 3
  - ID 2 is bottom[2] -> match with Top0[2] = 1
  
Final lower IDss:    [[2], [3], [1]]
```

**Step 5**: Process Adaptor1's upper IDs
```
Original upper IDss: [[0], [1], [3, 4]]

After shift by 1:    [[1], [2], [4, 5]]
```

**Step 6**: Combine
```
All transforms: [Unmerge(M), PassThrough(K), PassThrough(M0), PassThrough(M1), Unmerge(K)]
                 ↑ Adaptor0 transforms ↑    ↑      Adaptor1 transforms        ↑

All lower IDss: [[0], [1], [2], [3], [1]]
                 ↑ Adaptor0 ↑   ↑ Adaptor1 (matched) ↑

All upper IDss: [[2, 3], [1], [1], [2], [4, 5]]
                 ↑ Adaptor0 ↑     ↑ Adaptor1 (shifted) ↑

Bottom IDs: [0, 1]           (from Adaptor0)
Top IDs:    [1, 2, 4, 5]     (from Adaptor1, shifted by 1)
```

---

## Why This Works

### 1. **Unique IDs**
The shift ensures all hidden IDs are unique:
- Adaptor0 uses IDs: [0, 1, 2, 3]
- Adaptor1 uses IDs: [1, 2, 4, 5] (after shift and matching)
- Combined unique IDs: [0, 1, 2, 3, 4, 5]

### 2. **Proper Connection**
Bottom dimensions of Adaptor1 are matched with Top dimensions of Adaptor0:
```
Adaptor0 Top:    [M0=2, M1=3, K=1]
                  ↓     ↓      ↓
Adaptor1 Bottom: [M0=2, M1=3, K=1]  (after matching)
```

### 3. **Correct Data Flow**
```
Input [M=0, K=1]
  ↓ Adaptor0 transforms
Intermediate [M0=2, M1=3, K=1]
  ↓ Adaptor1 transforms (using matched IDs)
Output [M0=1, M1=2, K0=4, K1=5]
```

---

## Visual Example

```
Adaptor0: [M, K] -> [M0, M1, K]
          [0, 1] -> [2,  3,  1]

Adaptor1: [M0, M1, K] -> [M0, M1, K0, K1]
          [0,  1,  2] -> [0,  1,  3,  4]

After chaining:
          [M, K] -> [M0, M1, K0, K1]
          [0, 1] -> [1,  2,  4,  5]

Hidden ID mapping:
  0: M (bottom)
  1: K (bottom) 
  2: M0 (from Adaptor0, becomes intermediate, matched with Adaptor1's bottom[0])
  3: M1 (from Adaptor0, becomes intermediate, matched with Adaptor1's bottom[1])
  1: K (from Adaptor0, becomes intermediate, matched with Adaptor1's bottom[2])
  4: K0 (from Adaptor1, shifted from 3)
  5: K1 (from Adaptor1, shifted from 4)
```

---

## Key Insights

1. **Hidden IDs are internal bookkeeping** - They track dimension flow through transformations

2. **Shifting prevents conflicts** - Each adaptor's internal dimensions get unique IDs

3. **Matching connects adaptors** - Bottom1 IDs are replaced with Top0 IDs

4. **Bottom and Top define interface** - Only these are exposed to users

5. **Zero-copy composition** - All this is compile-time metadata manipulation

---

## Common Patterns

### Pattern 1: Sequential Tiling
```cpp
// A: [M] -> [M0, M1]
// B: [M0, M1] -> [M0, M1_0, M1_1]
// Chained: [M] -> [M0, M1_0, M1_1]
```

### Pattern 2: Pad then Tile
```cpp
// A: [M_raw] -> [M_padded]
// B: [M_padded] -> [M0, M1]
// Chained: [M_raw] -> [M0, M1]
```

### Pattern 3: Multi-dimensional Tiling
```cpp
// A: [M, K] -> [M0, M1, K]
// B: [M0, M1, K] -> [M0, M1, K0, K1]
// Chained: [M, K] -> [M0, M1, K0, K1]
```

---

## Summary

`chain_tensor_adaptors` performs these key operations:

1. **Find max ID in Adaptor0** - Determines where Adaptor1's IDs should start
2. **Find min non-bottom ID in Adaptor1** - Determines baseline for shifting
3. **Calculate shift** - Ensures unique IDs across both adaptors
4. **Shift and match lower IDs** - Connects the two adaptors properly
5. **Shift upper IDs** - Maintains uniqueness for output dimensions
6. **Combine all metadata** - Creates unified adaptor with all transformations

The result is a single tensor adaptor that applies both transformations sequentially, with proper dimension tracking throughout.
