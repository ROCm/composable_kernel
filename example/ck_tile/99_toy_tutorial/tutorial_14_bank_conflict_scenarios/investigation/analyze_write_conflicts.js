#!/usr/bin/env node
// Analyze WRITE conflicts for Tutorial 14 padded scenario

const M = 64;
const K = 32;
const K_PADDED = 34;
const ELEM_SIZE = 2;  // FP16

// WRITE distribution for [M, K]
// M2=16, M1=4, M0=1, K0=4, K1=8
// Thread mapping for WRITE (row-wise to [M,K]):
function getThreadMapping_Write(lane) {
    // Distribution: [M0, M1, M2], [K0, K1]
    // M0=1, M1=4, M2=16, K0=4, K1=8
    const K1_idx = lane % 8;          // 0-7
    const K0_idx = Math.floor((lane / 8) % 4);  // 0-3
    const M2_idx = Math.floor((lane / 32) % 16); // 0-15
    const M1_idx = Math.floor(lane / (32 * 16)) % 4; // 0-3 (for 256 threads)

    const k = K0_idx * 8 + K1_idx;
    const m = M2_idx;  // Simplified for 64 threads per wave

    // Each thread writes K1=8 elements along K
    const elements = [];
    for (let k1 = 0; k1 < 8; k1++) {
        const k_value = K0_idx * 8 + k1;
        if (k_value < K) {
            elements.push({ row: m, col: k_value });
        }
    }

    return { m, k_start: K0_idx * 8, elements };
}

function getAddress_Padded(row, col) {
    return row * (K_PADDED * ELEM_SIZE) + col * ELEM_SIZE;
}

function getBank(address) {
    return Math.floor(address / 4) % 32;
}

function analyzeWriteConflicts(lane) {
    const mapping = getThreadMapping_Write(lane);
    const bankCounts = {};
    const banks = [];

    for (const elem of mapping.elements) {
        const addr = getAddress_Padded(elem.row, elem.col);
        const bank = getBank(addr);
        banks.push(bank);
        bankCounts[bank] = (bankCounts[bank] || 0) + 1;
    }

    const maxAccesses = Math.max(...Object.values(bankCounts));
    const uniqueBanks = Object.keys(bankCounts).length;
    const conflictBanks = Object.entries(bankCounts)
        .filter(([_, count]) => count > 1)
        .map(([bank, count]) => ({ bank: parseInt(bank), count }));

    return { lane, mapping, banks, maxAccesses, uniqueBanks, conflictBanks };
}

console.log("=".repeat(80));
console.log("WRITE Conflict Analysis - Padded Row-Major");
console.log("=".repeat(80));
console.log("Configuration: M=64, K=32, K_PADDED=34, FP16");
console.log("Operation: WRITE [M,K] to LDS (row-wise access)");
console.log();

// Analyze first wave lanes (0-63)
for (let lane = 0; lane < 64; lane += 8) {
    const result = analyzeWriteConflicts(lane);
    console.log(`Lane ${lane}: m=${result.mapping.m}, k=[${result.mapping.k_start}-${result.mapping.k_start + 7}]`);
    console.log(`  Banks: [${result.banks.join(', ')}]`);
    console.log(`  Unique banks: ${result.uniqueBanks}`);
    if (result.conflictBanks.length > 0) {
        const conflictStr = result.conflictBanks.map(c => `bank ${c.bank}: ${c.count}x`).join(', ');
        console.log(`  Conflict: ${result.maxAccesses}-way (${conflictStr})`);
    } else {
        console.log(`  No intra-lane conflicts`);
    }
    console.log();
}

console.log("=".repeat(80));
console.log("Conclusion:");
console.log("=".repeat(80));
console.log("WRITE operation: Threads write CONSECUTIVE K values in same row");
console.log("Padded row stride = 68 bytes, elements are consecutive -> few conflicts");
console.log();
console.log("The 2,048 measured conflicts must come from:");
console.log("1. Inter-lane conflicts (multiple threads in same phase)");
console.log("2. Accumulation across multiple iterations");
console.log("3. Both WRITE + READ operations combined");
console.log();
