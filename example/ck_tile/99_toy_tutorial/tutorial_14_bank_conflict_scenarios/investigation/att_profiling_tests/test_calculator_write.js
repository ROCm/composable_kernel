#!/usr/bin/env node
// Test WRITE operation to verify calculator shows correct patterns

const M = 64;
const K = 32;
const ELEM_SIZE = 2;  // FP16

// WRITE distribution for [M, K] - row-wise access
// Distribution: [M0, M1, M2], [K0, K1] where M2=16, M1=4, M0=1, K0=4, K1=8
const WRITE_PHASES = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31],
    [32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43, 44, 45, 46, 47],
    [48, 49, 50, 51, 52, 53, 54, 55],
    [56, 57, 58, 59, 60, 61, 62, 63]
];

function getAddress_RowMajor(row, col) {
    return row * (K * ELEM_SIZE) + col * ELEM_SIZE;
}

function getAddress_Padded(row, col) {
    const K_PADDED = 34;
    return row * (K_PADDED * ELEM_SIZE) + col * ELEM_SIZE;
}

function getBank(address) {
    return Math.floor(address / 4) % 32;
}

// Thread mapping for WRITE (row-wise to [M,K])
// For row-wise access, the calculator should use:
// row_idx = lane % rows, col_start = (lane / rows) * 8
// But for the actual distribution, it's more complex
// Let's use simplified mapping for first wave
function getThreadMapping_Write_Simple(lane) {
    // Simplified: assume lane maps to row and reads consecutive cols
    const row = Math.floor(lane / 4);  // 4 lanes per row (4*8=32 cols)
    const col_start = (lane % 4) * 8;

    const elements = [];
    for (let c = 0; c < 8; c++) {
        const col = col_start + c;
        if (col < K && row < M) {
            elements.push({ row, col });
        }
    }
    return { row, col_start, elements };
}

function analyzeWriteConflicts(lane, addressFunc) {
    const mapping = getThreadMapping_Write_Simple(lane);
    const banks = [];
    const bankCounts = {};

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

    return { lane, mapping, banks, maxAccesses, uniqueBanks, conflictBanks };
}

console.log("=".repeat(80));
console.log("WRITE Operation - Intra-Lane Conflict Analysis");
console.log("=".repeat(80));
console.log();

console.log("Row-Major WRITE:");
console.log("-".repeat(80));
const phase0 = WRITE_PHASES[0];
for (const lane of phase0.slice(0, 4)) {
    const result = analyzeWriteConflicts(lane, getAddress_RowMajor);
    console.log(`Lane ${lane}: row=${result.mapping.row}, cols=[${result.mapping.col_start}-${result.mapping.col_start + 7}]`);
    console.log(`  Banks: [${result.banks.join(', ')}]`);
    if (result.conflictBanks.length > 0) {
        const conflictStr = result.conflictBanks.map(c => `bank ${c.bank}: ${c.count}x`).join(', ');
        console.log(`  Conflict: ${result.maxAccesses}-way (${conflictStr})`);
    } else {
        console.log(`  No conflicts`);
    }
    console.log();
}

console.log("Expected: 2-way conflicts from FP16 pairing (two elements per 4-byte bank slot)");
console.log();

console.log("=".repeat(80));
console.log("Padded WRITE:");
console.log("-".repeat(80));
for (const lane of phase0.slice(0, 4)) {
    const result = analyzeWriteConflicts(lane, getAddress_Padded);
    console.log(`Lane ${lane}: row=${result.mapping.row}, cols=[${result.mapping.col_start}-${result.mapping.col_start + 7}]`);
    console.log(`  Banks: [${result.banks.join(', ')}]`);
    if (result.conflictBanks.length > 0) {
        const conflictStr = result.conflictBanks.map(c => `bank ${c.bank}: ${c.count}x`).join(', ');
        console.log(`  Conflict: ${result.maxAccesses}-way (${conflictStr})`);
    } else {
        console.log(`  No conflicts`);
    }
    console.log();
}

console.log("Expected: STILL 2-way conflicts (padding doesn't help row-wise WRITE)");
console.log();
console.log("=".repeat(80));
console.log("Calculator should show WRITE with 'Row-wise' access pattern");
console.log("This corresponds to the [M,K] WRITE operation in the kernels");
console.log("=".repeat(80));
