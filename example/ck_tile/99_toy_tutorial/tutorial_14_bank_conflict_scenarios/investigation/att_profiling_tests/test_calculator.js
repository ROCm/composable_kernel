#!/usr/bin/env node
// Test script to verify calculator results match tutorial 14 scenarios

// Configuration constants
const M = 64;  // rows
const K = 32;  // cols
const ELEM_SIZE = 2;  // FP16 = 2 bytes

// Address calculation functions
function getAddress_RowMajor(row, col) {
    return row * (K * ELEM_SIZE) + col * ELEM_SIZE;
}

function getAddress_ColumnMajor(row, col) {
    return col * (M * ELEM_SIZE) + row * ELEM_SIZE;
}

function getAddress_Padded(row, col) {
    const K_PADDED = 34;  // 32 + 2 padding elements
    return row * (K_PADDED * ELEM_SIZE) + col * ELEM_SIZE;
}

function getAddress_XOR(row, col) {
    const xorCol = col ^ (row % 8);
    return row * (K * ELEM_SIZE) + xorCol * ELEM_SIZE;
}

function getBank(address) {
    return Math.floor(address / 4) % 32;
}

// Thread distribution for transpose read (column-wise access)
// Reading [K,M] from physical [M,K] storage
// [K,M][k,m] maps to physical [M,K][m,k]
function getThreadMapping_TransposeRead(lane) {
    const K2_idx = lane % 8;      // which k value (0-7) - this is the COLUMN in [M,K]
    const M0_idx = Math.floor(lane / 8); // which M group (0-7) - these are ROWS in [M,K]
    const m_start = M0_idx * 8;

    const elements = [];
    // Reading [K,M][K2_idx, m_start:m_start+8]
    // = physical [M,K][m_start:m_start+8, K2_idx]
    // = column K2_idx, rows m_start to m_start+7
    for (let m1 = 0; m1 < 8; m1++) {
        const m_value = m_start + m1;
        // Physical [M,K] coordinates: row=m_value, col=K2_idx
        elements.push({ row: m_value, col: K2_idx });
    }
    return { k: K2_idx, m_start, elements };
}

// Analyze intra-lane conflicts for a single lane
function analyzeIntraLaneConflicts(lane, addressFunc) {
    const mapping = getThreadMapping_TransposeRead(lane);
    const bankCounts = {};
    const banks = [];

    for (const elem of mapping.elements) {
        const addr = addressFunc(elem.row, elem.col);
        const bank = getBank(addr);
        banks.push(bank);
        bankCounts[bank] = (bankCounts[bank] || 0) + 1;
    }

    const maxAccesses = Math.max(...Object.values(bankCounts));
    const uniqueBanks = Object.keys(bankCounts).length;
    const conflictBanks = Object.entries(bankCounts)
        .filter(([_, count]) => count > 1)
        .map(([bank, count]) => ({ bank: parseInt(bank), count }));

    return { lane, banks, maxAccesses, uniqueBanks, conflictBanks };
}

// Test all Phase 0 lanes
const READ_PHASES = [
    [0, 1, 2, 3, 20, 21, 22, 23],
    [4, 5, 6, 7, 16, 17, 18, 19],
    [8, 9, 10, 11, 28, 29, 30, 31],
    [12, 13, 14, 15, 24, 25, 26, 27],
    [32, 33, 34, 35, 52, 53, 54, 55],
    [36, 37, 38, 39, 48, 49, 50, 51],
    [40, 41, 42, 43, 60, 61, 62, 63],
    [44, 45, 46, 47, 56, 57, 58, 59]
];

console.log("=".repeat(80));
console.log("LDS Bank Calculator Verification - Tutorial 14 Scenarios");
console.log("=".repeat(80));
console.log();

// Test Scenario 1: Row-Major (baseline)
console.log("Scenario 1: Row-Major Baseline");
console.log("-".repeat(80));
console.log("Configuration: M=64, K=32, FP16, Row-major storage, Transpose read");
console.log();

const phase0Lanes = READ_PHASES[0];
for (const lane of phase0Lanes) {
    const result = analyzeIntraLaneConflicts(lane, getAddress_RowMajor);
    const mapping = getThreadMapping_TransposeRead(lane);
    console.log(`Lane ${lane}: k=${result.lane < 8 ? result.lane : result.lane - 16}, m=[${mapping.m_start}-${mapping.m_start + 7}]`);
    console.log(`  Banks: [${result.banks.join(', ')}]`);
    console.log(`  Unique banks: ${result.uniqueBanks}`);
    if (result.conflictBanks.length > 0) {
        const conflictStr = result.conflictBanks.map(c => `bank ${c.bank}: ${c.count}x`).join(', ');
        console.log(`  Conflict: ${result.maxAccesses}-way (${conflictStr})`);
    } else {
        console.log(`  No conflicts`);
    }
    console.log();
}

console.log("Expected: 4-way conflicts with pattern {0, 16, 0, 16, 0, 16, 0, 16}");
console.log();
console.log("=".repeat(80));
console.log();

// Test Scenario 2: Column-Major
console.log("Scenario 2: Column-Major");
console.log("-".repeat(80));
console.log("Configuration: M=64, K=32, FP16, Column-major storage, Transpose read");
console.log();

for (const lane of phase0Lanes.slice(0, 2)) {  // Just first 2 lanes for brevity
    const result = analyzeIntraLaneConflicts(lane, getAddress_ColumnMajor);
    const mapping = getThreadMapping_TransposeRead(lane);
    console.log(`Lane ${lane}: k=${result.lane < 8 ? result.lane : result.lane - 16}, m=[${mapping.m_start}-${mapping.m_start + 7}]`);
    console.log(`  Banks: [${result.banks.join(', ')}]`);
    console.log(`  Unique banks: ${result.uniqueBanks}`);
    if (result.conflictBanks.length > 0) {
        const conflictStr = result.conflictBanks.map(c => `bank ${c.bank}: ${c.count}x`).join(', ');
        console.log(`  Conflict: ${result.maxAccesses}-way (${conflictStr})`);
    } else {
        console.log(`  No conflicts`);
    }
    console.log();
}

console.log("Expected: 2-way conflicts (each bank hit twice)");
console.log();
console.log("=".repeat(80));
console.log();

// Test Scenario 3: Padded
console.log("Scenario 3: Row-Major with Padding");
console.log("-".repeat(80));
console.log("Configuration: M=64, K=32, FP16, Row-major + 4-byte padding, Transpose read");
console.log();

for (const lane of phase0Lanes.slice(0, 2)) {  // Just first 2 lanes
    const result = analyzeIntraLaneConflicts(lane, getAddress_Padded);
    const mapping = getThreadMapping_TransposeRead(lane);
    console.log(`Lane ${lane}: k=${result.lane < 8 ? result.lane : result.lane - 16}, m=[${mapping.m_start}-${mapping.m_start + 7}]`);
    console.log(`  Banks: [${result.banks.join(', ')}]`);
    console.log(`  Unique banks: ${result.uniqueBanks}`);
    if (result.conflictBanks.length > 0) {
        const conflictStr = result.conflictBanks.map(c => `bank ${c.bank}: ${c.count}x`).join(', ');
        console.log(`  Conflict: ${result.maxAccesses}-way (${conflictStr})`);
    } else {
        console.log(`  No conflicts`);
    }
    console.log();
}

console.log("Expected: Reduced conflicts (stride = 17 banks hits more unique banks)");
console.log();
console.log("=".repeat(80));
console.log();

// Test Scenario 4: XOR Swizzle
console.log("Scenario 4: XOR Swizzle");
console.log("-".repeat(80));
console.log("Configuration: M=64, K=32, FP16, XOR swizzle, Transpose read");
console.log();

for (const lane of phase0Lanes.slice(0, 2)) {  // Just first 2 lanes
    const result = analyzeIntraLaneConflicts(lane, getAddress_XOR);
    const mapping = getThreadMapping_TransposeRead(lane);
    console.log(`Lane ${lane}: k=${result.lane < 8 ? result.lane : result.lane - 16}, m=[${mapping.m_start}-${mapping.m_start + 7}]`);
    console.log(`  Banks: [${result.banks.join(', ')}]`);
    console.log(`  Unique banks: ${result.uniqueBanks}`);
    if (result.conflictBanks.length > 0) {
        const conflictStr = result.conflictBanks.map(c => `bank ${c.bank}: ${c.count}x`).join(', ');
        console.log(`  Conflict: ${result.maxAccesses}-way (${conflictStr})`);
    } else {
        console.log(`  No conflicts - all banks unique!`);
    }
    console.log();
}

console.log("Expected: NO conflicts (all 8 banks unique per lane)");
console.log();
console.log("=".repeat(80));
