╔═══════════════════════════════════════════════════════════════════════════╗
║                   PHASE GROUPING & BANK CONFLICTS                         ║
║                        Complete Understanding                             ║
╔═══════════════════════════════════════════════════════════════════════════╗

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. HARDWARE BANDWIDTH CONSTRAINT                                        │
└─────────────────────────────────────────────────────────────────────────┘

LDS Bandwidth:  32 banks × 4 bytes = 128 bytes/cycle
Wavefront:      64 lanes
Instruction:    ds_read_b128 / ds_write_b128 (16 bytes per lane)
Total demand:   64 lanes × 16 bytes = 1024 bytes

Problem: 1024 bytes > 128 bytes! 
Solution: Execute in phases: 1024 ÷ 128 = 8 phases

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. WRITE PHASES (Sequential)                                            │
└─────────────────────────────────────────────────────────────────────────┘

Cycle 1: Phase 0 → Lanes 0-7    execute (128 bytes)
Cycle 2: Phase 1 → Lanes 8-15   execute (128 bytes)
Cycle 3: Phase 2 → Lanes 16-23  execute (128 bytes)
Cycle 4: Phase 3 → Lanes 24-31  execute (128 bytes)
Cycle 5: Phase 4 → Lanes 32-39  execute (128 bytes)
Cycle 6: Phase 5 → Lanes 40-47  execute (128 bytes)
Cycle 7: Phase 6 → Lanes 48-55  execute (128 bytes)
Cycle 8: Phase 7 → Lanes 56-63  execute (128 bytes)

Minimum write time: 8 cycles (if no bank conflicts)

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. READ PHASES (Non-Sequential - AMD MI300X)                            │
└─────────────────────────────────────────────────────────────────────────┘

Cycle 1: Phase 0 → Lanes {0,1,2,3,20,21,22,23}       execute
Cycle 2: Phase 1 → Lanes {4,5,6,7,16,17,18,19}       execute
Cycle 3: Phase 2 → Lanes {8,9,10,11,28,29,30,31}     execute
Cycle 4: Phase 3 → Lanes {12,13,14,15,24,25,26,27}   execute
Cycle 5: Phase 4 → Lanes {32,33,34,35,52,53,54,55}   execute
Cycle 6: Phase 5 → Lanes {36,37,38,39,48,49,50,51}   execute
Cycle 7: Phase 6 → Lanes {40,41,42,43,60,61,62,63}   execute
Cycle 8: Phase 7 → Lanes {44,45,46,47,56,57,58,59}   execute

Minimum read time: 8 cycles (if no bank conflicts)

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. BANK CONFLICT DETECTION: PER-PHASE, PER-LANE                         │
└─────────────────────────────────────────────────────────────────────────┘

Bank conflicts are checked:
  ✓ Within each phase (only those 8 lanes matter)
  ✓ Within each lane's 16-byte access (hits 4 banks)
  ✗ NOT across different phases (they execute at different times)

Example - Write Phase 0:
  ┌──────────────────────────────────────────────┐
  │ Lane 0: addr  0-15  → banks  0- 3   │  ← Different banks
  │ Lane 1: addr 16-31  → banks  4- 7   │  ← Different banks
  │ Lane 2: addr 32-47  → banks  8-11   │  ← Different banks
  │ Lane 3: addr 48-63  → banks 12-15   │  ← Different banks
  │ Lane 4: addr 64-79  → banks 16-19   │  ← Different banks
  │ Lane 5: addr 80-95  → banks 20-23   │  ← Different banks
  │ Lane 6: addr 96-111 → banks 24-27   │  ← Different banks
  │ Lane 7: addr112-127 → banks 28-31   │  ← Different banks
  └──────────────────────────────────────────────┘
  Result: NO CONFLICTS ✓ → Executes in 1 cycle

┌─────────────────────────────────────────────────────────────────────────┐
│ 5. TRANSPOSE PROBLEM: WITHIN-LANE CONFLICTS                             │
└─────────────────────────────────────────────────────────────────────────┘

Reading column 0 (transposed row 0), Lane 0:

  Lane 0 needs to read 8 FP16 from column 0, rows 0-7:
  
  ┌─────────────────────────────────────────────────────┐
  │ Element  │ Address │ Bank │ Pattern                │
  ├──────────┼─────────┼──────┼────────────────────────┤
  │ row 0    │   0     │  0   │                        │
  │ row 1    │  64     │ 16   │                        │
  │ row 2    │ 128     │  0   │ ← Same as row 0! (4x)  │
  │ row 3    │ 192     │ 16   │ ← Same as row 1! (4x)  │
  │ row 4    │ 256     │  0   │ ← CONFLICT!            │
  │ row 5    │ 320     │ 16   │ ← CONFLICT!            │
  │ row 6    │ 384     │  0   │ ← CONFLICT!            │
  │ row 7    │ 448     │ 16   │ ← CONFLICT!            │
  └─────────────────────────────────────────────────────┘

  Pattern: {0, 16, 0, 16, 0, 16, 0, 16}
  
  Lane 0's SINGLE ds_read_b128 instruction needs:
    - Bank 0:  4 times
    - Bank 16: 4 times
  
  Hardware must serialize:
    Cycle 1: Read bank 0  (gets element from row 0)
    Cycle 2: Read bank 0  (gets element from row 2)
    Cycle 3: Read bank 0  (gets element from row 4)
    Cycle 4: Read bank 0  (gets element from row 6)
    ... same for bank 16 ...
    
  Total for Lane 0: 4 cycles instead of 1!
  
  This is a 4-WAY BANK CONFLICT within a single lane!

┌─────────────────────────────────────────────────────────────────────────┐
│ 6. WHY STRIDE = 64 BYTES CAUSES CONFLICTS                               │
└─────────────────────────────────────────────────────────────────────────┘

Bank assignment: bank = (address_bytes ÷ 4) % 32

Stride (row width) = 32 elements × 2 bytes = 64 bytes
Bank stride = 64 ÷ 4 = 16 banks

Pattern for column 0:
  Row 0:  address 0   → bank  0
  Row 1:  address 64  → bank 16  (offset by 16 banks)
  Row 2:  address 128 → bank  0  (wraps: 32 % 32 = 0)
  Row 3:  address 192 → bank 16  (wraps: 48 % 32 = 16)
  ...

Banks wrap every 128 bytes (32 banks × 4 bytes)
Our stride (64 bytes) is exactly HALF the wrap period
→ Alternates between only 2 banks!

┌─────────────────────────────────────────────────────────────────────────┐
│ 7. XOR SWIZZLING SOLUTION                                               │
└─────────────────────────────────────────────────────────────────────────┘

XOR descriptor permutes physical addresses:
  physical_address = XOR(row_component, col_component)

WITHOUT XOR:
  Column 0: banks {0, 16, 0, 16, 0, 16, 0, 16} → only 2 banks!

WITH XOR:
  Column 0: banks {0, 5, 11, 14, 16, 21, 27, 30, ...} → spreads across many!

Effect: Reduces 4-way conflicts to ~2-way conflicts
Result: 57% improvement in bank conflict efficiency

┌─────────────────────────────────────────────────────────────────────────┐
│ 8. SUMMARY: KEY INSIGHTS                                                │
└─────────────────────────────────────────────────────────────────────────┘

✓ Only 8 lanes execute per cycle (bandwidth limit)
✓ 64 lanes → 8 phases minimum
✓ Write phases: sequential (0-7, 8-15, ...)
✓ Read phases: non-sequential (hardware-specific pattern)
✓ Conflicts checked PER-PHASE only
✓ Transpose creates WITHIN-LANE conflicts (lane vs itself)
✓ Strided access → same banks repeatedly
✓ XOR swizzling → spreads accesses across all banks

The problem is NOT lanes conflicting with each other.
The problem is EACH LANE conflicting with ITSELF due to stride pattern!

╚═══════════════════════════════════════════════════════════════════════════╝
