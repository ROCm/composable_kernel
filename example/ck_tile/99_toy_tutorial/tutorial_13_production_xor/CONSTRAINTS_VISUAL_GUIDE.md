╔═══════════════════════════════════════════════════════════════════════════╗
║              LDS ACCESS CONSTRAINTS - QUICK REFERENCE                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────────┐
│ Q1: How many banks can EACH LANE use in ONE CYCLE?                       │
└───────────────────────────────────────────────────────────────────────────┘

Answer: 4 banks (for ds_read_b128 / ds_write_b128)

BUT CRITICALLY: These must be 4 CONSECUTIVE banks!

Example:
  Lane at address 0:   uses banks {0, 1, 2, 3}      ✓ OK
  Lane at address 16:  uses banks {4, 5, 6, 7}      ✓ OK
  Lane at address 64:  uses banks {16, 17, 18, 19}  ✓ OK

CANNOT do:
  Lane uses banks {0, 5, 12, 20}  ✗ Not consecutive!

┌───────────────────────────────────────────────────────────────────────────┐
│ Q2: How much data can EACH LANE read/write in ONE INSTRUCTION?           │
└───────────────────────────────────────────────────────────────────────────┘

Answer: 16 bytes (for ds_read_b128 / ds_write_b128)

This equals: 8 FP16 elements OR 4 FP32 elements

These 16 bytes span 4 consecutive banks (4 bytes per bank)

┌───────────────────────────────────────────────────────────────────────────┐
│ Q3: How many lanes can execute in ONE CYCLE?                             │
└───────────────────────────────────────────────────────────────────────────┘

Answer: 8 lanes (for ds_read_b128 / ds_write_b128)

Calculation:
  LDS bandwidth:  128 bytes/cycle
  Per lane:       16 bytes
  Lanes/cycle:    128 / 16 = 8 lanes

Therefore: 64 lanes / 8 lanes per cycle = 8 cycles minimum

┌───────────────────────────────────────────────────────────────────────────┐
│ Q4: What happens if a lane needs the SAME BANK multiple times?           │
└───────────────────────────────────────────────────────────────────────────┘

Answer: BANK CONFLICT! Hardware serializes the accesses.

Example - Lane reading column 0:
  ┌─────────────────────────────────────────────┐
  │ Address 0   → bank 0   ◄─┐                 │
  │ Address 64  → bank 16  ◄─┼─┐               │
  │ Address 128 → bank 0   ◄─┘ │  Lane needs   │
  │ Address 192 → bank 16  ◄───┘  each bank    │
  │ Address 256 → bank 0   ◄─┐    4 times!     │
  │ Address 320 → bank 16  ◄─┼─┐  CONFLICT!    │
  │ Address 384 → bank 0   ◄─┘ │                │
  │ Address 448 → bank 16  ◄───┘                │
  └─────────────────────────────────────────────┘

Hardware must make 4 separate trips to bank 0:
  Cycle 1: Get address 0
  Cycle 2: Get address 128
  Cycle 3: Get address 256
  Cycle 4: Get address 384

Result: 4-way conflict = 4× slower!

┌───────────────────────────────────────────────────────────────────────────┐
│ Q5: What determines which banks a lane uses?                             │
└───────────────────────────────────────────────────────────────────────────┘

Answer: The STARTING ADDRESS of the read/write

Formula: starting_bank = (address_in_bytes / 4) % 32

Then lane uses: {starting_bank, starting_bank+1, starting_bank+2, starting_bank+3}

Example:
  Address 0:   starting_bank = 0   → uses banks {0, 1, 2, 3}
  Address 16:  starting_bank = 4   → uses banks {4, 5, 6, 7}
  Address 64:  starting_bank = 16  → uses banks {16, 17, 18, 19}
  Address 128: starting_bank = 0   → uses banks {0, 1, 2, 3}  (wrapped!)

┌───────────────────────────────────────────────────────────────────────────┐
│ Q6: COMPLETE CONSTRAINT SUMMARY                                          │
└───────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┬─────────────┬──────────────────────────┐
│ Parameter                │ Value       │ Notes                    │
├──────────────────────────┼─────────────┼──────────────────────────┤
│ LDS HARDWARE             │             │                          │
│  Total banks             │ 32          │ Fixed hardware           │
│  Bank width              │ 4 bytes     │ Per bank                 │
│  Total bandwidth         │ 128 bytes   │ Per cycle                │
│                          │             │                          │
│ PER LANE (ds_read_b128)  │             │                          │
│  Data per instruction    │ 16 bytes    │ Per lane                 │
│  Banks per instruction   │ 4 banks     │ CONSECUTIVE only         │
│  Which banks?            │ addr/4 % 32 │ Start bank, then +1,+2,+3│
│                          │             │                          │
│ PHASE EXECUTION          │             │                          │
│  Lanes per phase         │ 8 lanes     │ For 16-byte instruction  │
│  Phases per wavefront    │ 8 phases    │ 64 lanes / 8             │
│  Cycles per phase (min)  │ 1 cycle     │ If no conflicts          │
│  Cycles if N-way conflict│ N cycles    │ Serialization            │
│                          │             │                          │
│ BANK CONFLICTS           │             │                          │
│  Checked between         │ Phase lanes │ 8 lanes in same phase    │
│  Checked within          │ Each lane   │ Lane's 4 bank accesses   │
│  NOT checked between     │ Phases      │ Execute at different time│
└──────────────────────────┴─────────────┴──────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q7: WHY TRANSPOSE HAS CONFLICTS - STEP BY STEP                           │
└───────────────────────────────────────────────────────────────────────────┘

Matrix: 64 rows × 32 cols, FP16
Row width: 32 × 2 bytes = 64 bytes

Reading COLUMN 0 (transpose):
  Need: elem[0][0], elem[1][0], elem[2][0], ..., elem[63][0]
  Addresses: 0, 64, 128, 192, ... (stride = 64 bytes)

Lane 0 reads first 8 elements of column 0:

Step 1: Calculate addresses
  elem[0][0]: row 0, col 0  → address = 0 × 64 + 0 × 2 = 0
  elem[1][0]: row 1, col 0  → address = 1 × 64 + 0 × 2 = 64
  elem[2][0]: row 2, col 0  → address = 2 × 64 + 0 × 2 = 128
  ...

Step 2: Calculate which bank each address maps to
  Address 0:   (0/4) % 32   = 0
  Address 64:  (64/4) % 32  = 16
  Address 128: (128/4) % 32 = 32 % 32 = 0   ← REPEATS!
  Address 192: (192/4) % 32 = 48 % 32 = 16  ← REPEATS!
  Address 256: (256/4) % 32 = 64 % 32 = 0   ← REPEATS!
  Address 320: (320/4) % 32 = 80 % 32 = 16  ← REPEATS!
  Address 384: (384/4) % 32 = 96 % 32 = 0   ← REPEATS!
  Address 448: (448/4) % 32 = 112 % 32 = 16 ← REPEATS!

Step 3: Count bank usage
  Bank 0:  used 4 times (addresses 0, 128, 256, 384)
  Bank 16: used 4 times (addresses 64, 192, 320, 448)
  
Step 4: Conflict!
  Lane 0 needs bank 0 FOUR times in ONE instruction
  → Hardware must serialize into 4 accesses
  → 4-way conflict!

┌───────────────────────────────────────────────────────────────────────────┐
│ Q8: WHY WRITE HAS NO CONFLICTS                                           │
└───────────────────────────────────────────────────────────────────────────┘

Writing ROW 0 (sequential):
  Need: elem[0][0], elem[0][1], elem[0][2], ..., elem[0][31]
  Addresses: 0, 2, 4, 6, 8, ... (stride = 2 bytes for FP16)

Lane 0 writes first 8 elements of row 0:

Step 1: Calculate addresses
  elem[0][0]: address = 0
  elem[0][1]: address = 2
  elem[0][2]: address = 4
  ...
  elem[0][7]: address = 14

Step 2: Calculate banks (lane writes 16 bytes starting at address 0)
  Bytes 0-3:   bank 0
  Bytes 4-7:   bank 1
  Bytes 8-11:  bank 2
  Bytes 12-15: bank 3

Step 3: Count bank usage
  Bank 0:  1 time
  Bank 1:  1 time
  Bank 2:  1 time
  Bank 3:  1 time

Step 4: No conflict!
  Each bank used only once → executes in 1 cycle ✓

╚═══════════════════════════════════════════════════════════════════════════╝

KEY INSIGHT:
The constraint is NOT "each lane can only use 1 bank per cycle"
The constraint is "each lane can use 4 banks, but they must be CONSECUTIVE,
                   and if the lane needs the SAME bank multiple times within
                   its 16-byte access, that causes a conflict and serialization!"
