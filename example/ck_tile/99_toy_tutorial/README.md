# CK Tile Tutorials

This directory contains step-by-step tutorials for learning the CK Tile API, progressing from fundamental concepts to production-ready optimizations.

## Tutorial Overview

### Tutorial 01: Tensor Fundamentals
Introduction to CK Tile's core tensor concepts.

**Files:** `tutorial_01_tensor_fundamentals/`

### Tutorial 02: Tensor Adaptors
Learn how to transform tensor layouts using adaptors.

**Files:** `tutorial_02_tensor_adaptors/`
**Documentation:** `tutorial_02_tensor_adaptors/XOR_TRANSFORM_EXPLAINED.md`

### Tutorial 03: Padding and Tiles
Understanding tile operations and padding strategies.

**Files:** `tutorial_03_padding_and_tiles/`

### Tutorial 04: Descriptor vs Adaptor
Deep dive into the differences between descriptors and adaptors.

**Files:** `tutorial_04_descriptor_vs_adaptor/`
**Documentation:** `tutorial_04_descriptor_vs_adaptor/DESCRIPTOR_VS_ADAPTOR.md`

### Tutorial 05: Basic Distributed GEMM
Introduction to distributed matrix multiplication.

**Files:** `tutorial_05_basic_distributed_gemm/`

### Tutorial 06: Tile Sweeping GEMM
Optimized GEMM using tile sweeping techniques.

**Files:** `tutorial_06_tile_sweeping_gemm/`

### Tutorial 07: Tile Sweeping with Y Repetition
Advanced tile sweeping with dimension repetition.

**Files:** `tutorial_07_tile_sweeping_with_y_repetition/`
**Documentation:** `tutorial_07_tile_sweeping_with_y_repetition/Y_REPETITION_EXPLAINED.md`

### Tutorial 08: LDS Staging
Introduction to Local Data Share (shared memory) staging.

**Files:** `tutorial_08_lds_staging/`

### Tutorial 09: Optimized LDS
Advanced LDS optimization techniques.

**Files:** `tutorial_09_optimized_lds/`

### Tutorial 10: XOR LDS
First introduction to XOR swizzling for bank conflict reduction.

**Files:** `tutorial_10_xor_lds/`

### Tutorial 11: Bank Conflicts and XOR Swizzling ⭐

**Complete guide to understanding and eliminating LDS bank conflicts on AMD GPUs.**

This tutorial provides comprehensive coverage of bank conflicts, from theory to implementation.

#### Files:
- `tutorial_11_xor_test/xor_test_plain_only.cpp` - Baseline transpose (no XOR)
- `tutorial_11_xor_test/xor_test_production_transpose.cpp` - XOR optimized transpose

#### Documentation:
- **[BANK_CONFLICT_TUTORIAL.md](BANK_CONFLICT_TUTORIAL.md)** - Comprehensive guide (START HERE!)
- `tutorial_11_xor_test/BANK_CONFLICT_SUMMARY.md` - Quick reference
- `tutorial_11_xor_test/XOR_TRANSPOSE_SUMMARY.md` - Implementation details

#### Scripts:
- `scripts/profile_bank_conflicts.sh` - Automated profiling
- `scripts/analyze_bank_conflicts.py` - Results analysis

#### What You'll Learn:
- **LDS bank conflict architecture** on AMD MI300 GPUs
- **Constraint satisfaction problem (CSP) framing** of optimization
- **Measuring conflicts** with rocprofv3 profiling tools
- **XOR swizzling technique** in CK Tile API
- **Trade-offs** between different optimization approaches
- **Mathematical limits** of bank conflict elimination

#### Key Results:
```
Plain LDS:  1,244% conflict rate (12.4 conflicts per instruction)
XOR LDS:      533% conflict rate (5.3 conflicts per instruction)
Improvement:  57% reduction in bank conflicts

Theoretical minimum: 2-way conflicts (64 threads / 32 banks)
Gap to optimal: 2.5× (good practical result!)
```

#### Quick Start:

**1. Build the tutorials:**
```bash
cd relbuild
cmake --build . --target aa_tutorial_11_plain_transpose -j$(nproc)
cmake --build . --target aa_tutorial_11_production_transpose -j$(nproc)
```

**2. Run baseline (plain transpose):**
```bash
./bin/aa_tutorial_11_plain_transpose
```

**3. Run optimized (XOR transpose):**
```bash
./bin/aa_tutorial_11_production_transpose
```

**4. Profile and analyze (requires rocprofv3):**
```bash
bash ../example/ck_tile/99_toy_tutorial/scripts/profile_bank_conflicts.sh
```

This will:
- Build both versions
- Profile with AMD performance counters
- Generate comprehensive analysis report
- Show 57% conflict reduction

#### Prerequisites:
- Basic GPU programming knowledge (threads, blocks, wavefronts)
- Understanding of shared memory concepts
- Tutorial 08 (LDS staging) recommended
- AMD GPU with ROCm for profiling

#### Next Steps:
After completing this tutorial, you can:
- Apply XOR swizzling to your own kernels
- Experiment with different tile sizes (32×32 for near-zero conflicts)
- Explore advanced optimizations (double buffering, padding)
- Read the complete [BANK_CONFLICT_TUTORIAL.md](BANK_CONFLICT_TUTORIAL.md) for deep dive

---

### Tutorial 12: XOR Correct
Verification and testing of XOR implementations.

**Files:** `tutorial_12_xor_correct/`

### Tutorial 13: Production XOR
Production-ready XOR swizzling implementation.

**Files:** `tutorial_13_production_xor/`

---

## Learning Path

### Beginner (Start Here)
1. Tutorial 01: Tensor Fundamentals
2. Tutorial 02: Tensor Adaptors
3. Tutorial 03: Padding and Tiles
4. Tutorial 04: Descriptor vs Adaptor

### Intermediate (GEMM Basics)
5. Tutorial 05: Basic Distributed GEMM
6. Tutorial 06: Tile Sweeping GEMM
7. Tutorial 07: Tile Sweeping with Y Repetition

### Advanced (Performance Optimization)
8. Tutorial 08: LDS Staging
9. Tutorial 09: Optimized LDS
10. Tutorial 10: XOR LDS
11. **Tutorial 11: Bank Conflicts (Comprehensive)** ⭐
12. Tutorial 12: XOR Correct
13. Tutorial 13: Production XOR

---

## Building Tutorials

All tutorials can be built using CMake from the repository root:

```bash
# Create build directory
mkdir -p relbuild && cd relbuild

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DGPU_TARGETS="gfx942" \
      ..

# Build specific tutorial (example)
cmake --build . --target aa_tutorial_11_plain_transpose -j$(nproc)

# Or build all tutorials
cmake --build . -j$(nproc)
```

## Profiling Tutorials

For performance analysis of Tutorial 11 (bank conflicts):

```bash
# Use the automated profiling script
bash example/ck_tile/99_toy_tutorial/scripts/profile_bank_conflicts.sh relbuild /tmp/my_analysis

# Or manually profile a specific tutorial
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/profile_output \
          -- ./bin/aa_tutorial_11_plain_transpose
```

## Documentation

Each tutorial may include:
- **Source code** with detailed comments
- **README** or **markdown docs** explaining concepts
- **CMakeLists.txt** for building
- **Analysis scripts** for performance evaluation

**Comprehensive guides:**
- [BANK_CONFLICT_TUTORIAL.md](BANK_CONFLICT_TUTORIAL.md) - Complete bank conflict guide

## Contributing

When adding new tutorials:
1. Follow the naming convention: `tutorial_XX_descriptive_name/`
2. Include clear comments in source code
3. Add documentation for complex concepts
4. Update this README with tutorial summary
5. Ensure tutorials build successfully

## Getting Help

- See individual tutorial README files for specific guidance
- Refer to CK Tile API documentation
- Check the main repository README for general setup
- Open issues on GitHub for bugs or questions

---

**Happy Learning!**

For questions or feedback about these tutorials, please refer to the [CK Tile documentation](https://github.com/ROCm/composable_kernel) or open an issue.
