#!/usr/bin/env node
// CORRECTED: Understanding that ds_read_b128 = 16 bytes = 4 consecutive banks

console.log("=".repeat(80));
console.log("CORRECTED: XOR Bank Access (16 bytes = 4 consecutive banks)");
console.log("=".repeat(80));
console.log();

const M = 64;
const K = 32;
const ELEM_SIZE = 2;  // FP16

function getAddress_XOR(row, col) {
    const xorCol = col ^ (row % 8);
    return row * (K * ELEM_SIZE) + xorCol * ELEM_SIZE;
}

function getBanks_FromAddress(addr) {
    // ds_read_b128 reads 16 bytes starting at addr
    // This spans 4 consecutive banks
    const starting_bank = Math.floor(addr / 4) % 32;
    const banks = [];
    for (let i = 0; i < 4; i++) {
        banks.push((starting_bank + i) % 32);
    }
    return banks;
}

console.log("KEY FACT: ds_read_b128 = 16 bytes = 4 consecutive banks");
console.log("Starting from address A, uses banks {A/4, A/4+1, A/4+2, A/4+3} (mod 32)");
console.log();

const READ_PHASES = [
    [0, 1, 2, 3, 20, 21, 22, 23],
];

console.log("=".repeat(80));
console.log("Phase 0 Lane Analysis (XOR swizzle):");
console.log("=".repeat(80));
console.log();

const phase0Lanes = READ_PHASES[0];

for (const lane of phase0Lanes) {
    const K2_idx = lane % 8;  // which column
    const M0_idx = Math.floor(lane / 8); // which M group
    const m_start = M0_idx * 8;

    // This lane reads ONE ds_read_b128 instruction
    // But wait - does it read from ONE address, or multiple?
    // Let me check what the distribution actually gives...

    // If lane reads "column K2_idx, rows [m_start to m_start+7]",
    // these are 8 DIFFERENT addresses (NOT contiguous!)
    // So this can't be a single ds_read_b128!

    console.log(`Lane ${lane}: col=${K2_idx}, M group=${M0_idx}`);
    console.log(`  The question: What does this lane actually READ in one instruction?`);
    console.log();
}

console.log("=".repeat(80));
console.log("REALIZATION: I need to understand the DISTRIBUTION better!");
console.log("=".repeat(80));
console.log();
console.log("The distribution determines:");
console.log("  1. What LOGICAL coordinates each lane maps to");
console.log("  2. Whether those coordinates are CONTIGUOUS in physical memory");
console.log();
console.log("For transpose read [K,M] from physical [M,K] XOR storage:");
console.log("  - Logical [K,M][k,m] element");
console.log("  - Maps to physical [M,K][m,k] with XOR");
console.log("  - physical_col = k XOR (m % 8)");
console.log();
console.log("If a lane reads M1=8 consecutive M values (for fixed k),");
console.log("these map to 8 different rows in physical storage.");
console.log("These are NOT contiguous! (64 bytes apart)");
console.log();
console.log("So either:");
console.log("  A) The distribution gives each lane CONTIGUOUS elements");
console.log("  B) The vector load is NOT a single ds_read_b128");
console.log();
console.log("Let me check the actual distribution code...");
console.log();

console.log("From tutorial_13: MakeACopyDistribution() for [M, K]:");
console.log("  M splits: M0=1, M1=4, M2=16");
console.log("  K splits: K0=2, K1=8");
console.log("  K1=8 is the vectorized dimension (16 bytes = 8 FP16)");
console.log();
console.log("This means each lane gets 8 CONSECUTIVE K elements (same row)!");
console.log("NOT 8 different M elements (different rows)!");
console.log();

console.log("=".repeat(80));
console.log("AH! I had the distribution BACKWARDS!");
console.log("=".repeat(80));
console.log();
console.log("For GEMM READ from [K,M] LDS (transpose read):");
console.log("  The distribution likely gives:");
console.log("    - Each lane: fixed K value, consecutive M values");
console.log("    - Vector dimension: M (the 8 consecutive elements)");
console.log();
console.log("But in Tutorial 14 (simple transpose, not GEMM),");
console.log("the distribution I calculated had M as the vector dimension.");
console.log();
console.log("Let me look at Tutorial 14's actual distribution...");
console.log();
