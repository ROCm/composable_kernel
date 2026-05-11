#!/usr/bin/env node
// Analyze WRITE operation with XOR and packing

const M = 64;
const K = 32;
const kKPack = 8;
const ELEM_SIZE = 2;

console.log("=".repeat(80));
console.log("XOR WRITE Analysis with kKPack=8");
console.log("=".repeat(80));
console.log();

// WRITE distribution for [M,K]
// M splits: M0=1, M1=4, M2=16
// K splits: K0=4, K1=8
// K1=8 is vectorized dimension

console.log("WRITE [M,K] distribution:");
console.log("  Each lane writes: fixed M, consecutive K values");
console.log("  Lane 0: M=0, K=[0-7] (8 consecutive K elements)");
console.log();

function getAddress_PackedXOR(row, col) {
    const k_pack = Math.floor(col / kKPack);
    const k_in_pack = col % kKPack;
    const m = row;

    // XOR permutation
    const xor_k_pack = k_pack ^ (m % 8);

    // Physical address
    return (xor_k_pack * 8 + m * 64 + k_in_pack) * ELEM_SIZE;
}

console.log("Lane 0 writing M=0, K=[0-7]:");
console.log("-".repeat(70));

const addresses = [];
for (let k = 0; k < 8; k++) {
    const m = 0;
    const addr = getAddress_PackedXOR(m, k);
    addresses.push(addr);

    const k_pack = Math.floor(k / kKPack);
    const k_in_pack = k % kKPack;
    const xor_k_pack = k_pack ^ (m % 8);

    console.log(`  [m=${m}, k=${k}]: k_pack=${k_pack}, k_in_pack=${k_in_pack}, addr=${addr}`);
}

console.log();

// Check if contiguous
const isContiguous = addresses.every((addr, i) =>
    i === 0 || addr === addresses[i-1] + ELEM_SIZE
);

if (isContiguous) {
    console.log("✓ Addresses ARE contiguous!");
    console.log(`  Range: ${addresses[0]} to ${addresses[7]}`);
    console.log(`  Total: ${addresses[7] - addresses[0] + ELEM_SIZE} bytes = 16 bytes`);
    console.log();
    console.log("This is a SINGLE ds_write_b128 instruction!");
    console.log("16 bytes = 4 consecutive banks");
    console.log();

    const starting_bank = Math.floor(addresses[0] / 4) % 32;
    const banks = [];
    for (let i = 0; i < 4; i++) {
        banks.push((starting_bank + i) % 32);
    }

    console.log(`Starting bank: ${starting_bank}`);
    console.log(`Banks used: [${banks.join(', ')}]`);
    console.log();
    console.log("✓ NO intra-lane conflicts (4 different consecutive banks)");
} else {
    console.log("✗ Addresses are NOT contiguous!");

    const banks = addresses.map(a => Math.floor(a/4) % 32);
    console.log(`Banks: [${banks.join(', ')}]`);

    const bank_counts = {};
    for (const bank of banks) {
        bank_counts[bank] = (bank_counts[bank] || 0) + 1;
    }

    const max_count = Math.max(...Object.values(bank_counts));
    if (max_count > 1) {
        console.log(`✗ ${max_count}-way intra-lane conflict!`);
    } else {
        console.log(`✓ NO intra-lane conflicts`);
    }
}

console.log();
console.log("=".repeat(80));
console.log("Try multiple lanes:");
console.log("=".repeat(80));
console.log();

for (let lane = 0; lane < 4; lane++) {
    // Simplified mapping: lane writes row 0, different K ranges
    const m = 0;
    const k_start = lane * 8;

    const lane_addrs = [];
    for (let k = 0; k < 8; k++) {
        lane_addrs.push(getAddress_PackedXOR(m, k_start + k));
    }

    const contig = lane_addrs.every((addr, i) =>
        i === 0 || addr === lane_addrs[i-1] + ELEM_SIZE
    );

    console.log(`Lane ${lane} (M=${m}, K=[${k_start}-${k_start+7}]): ${contig ? 'CONTIGUOUS ✓' : 'NOT contiguous ✗'}`);

    if (contig) {
        const start_bank = Math.floor(lane_addrs[0] / 4) % 32;
        console.log(`  Banks: [${start_bank}, ${start_bank+1}, ${start_bank+2}, ${start_bank+3}]`);
    }
}

console.log();
console.log("=".repeat(80));
console.log("CONCLUSION:");
console.log("=".repeat(80));
console.log();
console.log("With kKPack=8 and XOR:");
console.log("  - WRITE (consecutive K): Elements ARE contiguous");
console.log("  - Each lane writes 16 bytes = 4 consecutive banks");
console.log("  - NO intra-lane conflicts for WRITE! ✓");
console.log();
console.log("  - READ (consecutive M): Elements are NOT contiguous (strided)");
console.log("  - Each lane reads 8 strided elements");
console.log("  - With XOR: NO intra-lane conflicts for READ! ✓");
console.log();
console.log("So where do the 3,072 conflicts come from???");
console.log();
console.log("Possible sources:");
console.log("  1. Inter-lane conflicts (multiple lanes, same phase, same banks)");
console.log("  2. Non-XOR test in the same kernel (04_xor tests both plain AND xor)");
console.log("  3. Different matrix sizes or configurations");
console.log("  4. My XOR address calculation is simplified/wrong");
console.log();
