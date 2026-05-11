#!/usr/bin/env node
// FINAL CORRECT VERSION: Each lane reads M1=8 consecutive M elements

const M = 64;
const K = 32;
const ELEM_SIZE = 2;  // FP16

// Distribution for [K, M]:
// K splits: K0=1, K1=4, K2=8
// M splits: M0=8, M1=8  ← M1=8 is vectorized!
//
// Lane mapping:
// lane_id ranges 0-255 (256 threads)
// Within one wave (64 threads):
//   K2 = lane % 8
//   M0 = lane / 8
//   Each lane reads: [K2, M0*8 : M0*8+8] (8 consecutive M elements)

function getAddress_XOR(row, col) {
    // Physical [M,K] coordinates with XOR
    const xorCol = col ^ (row % 8);
    return row * (K * ELEM_SIZE) + xorCol * ELEM_SIZE;
}

console.log("=".repeat(80));
console.log("FINAL CORRECT: XOR Bank Access with M1=8 Vectorization");
console.log("=".repeat(80));
console.log();
console.log("Key insight: M1=8 is the vectorized dimension");
console.log("Each lane reads [K, M] where:");
console.log("  - K is FIXED (one of 8 values: 0-7)");
console.log("  - M is a range of 8 CONSECUTIVE values");
console.log();

const READ_PHASES = [[0, 1, 2, 3, 20, 21, 22, 23]];
const phase0 = READ_PHASES[0];

console.log("Phase 0 lanes: [0, 1, 2, 3, 20, 21, 22, 23]");
console.log();

for (const lane of phase0) {
    const K2 = lane % 8;
    const M0 = Math.floor(lane / 8);
    const m_start = M0 * 8;

    console.log(`Lane ${lane}: K=${K2}, M=[${m_start}-${m_start+7}]`);
    console.log("-".repeat(70));

    // These are 8 consecutive M elements in logical [K,M] space
    // They map to physical [M,K] as [m, K] where m varies
    //
    // Physical addresses for m=m_start to m_start+7, k=K2:
    const addresses = [];
    const banks = [];

    for (let m = 0; m < 8; m++) {
        const row = m_start + m;  // physical row (M dimension)
        const col = K2;            // physical col (K dimension)

        const addr = getAddress_XOR(row, col);
        addresses.push(addr);

        // Each ds_read reads 16 bytes = 4 banks
        // But we have 8 elements = 16 bytes total
        // Are they contiguous?
        console.log(`  M=${row}: addr=${addr}`);
    }

    console.log();

    // Check if addresses are contiguous
    const isContiguous = addresses.every((addr, i) =>
        i === 0 || addr === addresses[i-1] + ELEM_SIZE
    );

    if (isContiguous) {
        console.log(`  Addresses ARE contiguous! (${addresses[0]} to ${addresses[7]})`);
        console.log(`  Single ds_read_b128 reads 16 bytes starting at ${addresses[0]}`);

        const starting_bank = Math.floor(addresses[0] / 4) % 32;
        const banks_used = [];
        for (let i = 0; i < 4; i++) {
            banks_used.push((starting_bank + i) % 32);
        }

        console.log(`  Banks: [${banks_used.join(', ')}]`);
        console.log(`  → NO intra-lane conflict (4 different banks) ✓`);
    } else {
        console.log(`  Addresses are NOT contiguous!`);
        console.log(`  This is NOT a single ds_read_b128!`);

        // Calculate banks for each element
        for (let i = 0; i < addresses.length; i++) {
            const bank = Math.floor(addresses[i] / 4) % 32;
            banks.push(bank);
        }

        console.log(`  Banks: [${banks.join(', ')}]`);

        // Count bank usage
        const bankCounts = {};
        for (const bank of banks) {
            bankCounts[bank] = (bankCounts[bank] || 0) + 1;
        }

        const maxCount = Math.max(...Object.values(bankCounts));
        if (maxCount > 1) {
            console.log(`  → ${maxCount}-way intra-lane conflict`);
        } else {
            console.log(`  → NO intra-lane conflict ✓`);
        }
    }

    console.log();
}

console.log("=".repeat(80));
console.log("ANSWER TO USER'S QUESTION:");
console.log("=".repeat(80));
console.log();
console.log("Are 8 elements packed together?");
console.log("  → YES if they're contiguous in physical memory");
console.log("  → Then ds_read_b128 reads them as 16 bytes = 4 consecutive banks");
console.log();
console.log("For XOR with M consecutive:");
console.log("  → Need to check if consecutive M values (same K) are contiguous");
console.log("  → This depends on the storage layout after XOR");
console.log();
