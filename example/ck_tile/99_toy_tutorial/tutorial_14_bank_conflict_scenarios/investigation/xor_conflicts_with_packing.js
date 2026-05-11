#!/usr/bin/env node
// Understanding XOR conflicts WITH kKPack=8

const M = 64;
const K = 32;
const kKPack = 8;
const ELEM_SIZE = 2;  // FP16

console.log("=".repeat(80));
console.log("XOR with kKPack=8: Where do conflicts come from?");
console.log("=".repeat(80));
console.log();

console.log("Physical layout: [K/Pack, M, Pack] = [4, 64, 8]");
console.log("Strides (elements): [8, 64, 1]");
console.log();
console.log("This means:");
console.log("  - Pack dimension is CONTIGUOUS (stride 1)");
console.log("  - M dimension has stride 64 elements");
console.log("  - K/Pack dimension has stride 8 elements");
console.log();

// Physical address calculation for packed layout
function getAddress_Packed(row, col) {
    // [M,K] logical → [K/Pack, M, Pack] physical
    const k_pack = Math.floor(col / kKPack);
    const k_in_pack = col % kKPack;
    const m = row;

    // Physical address (BEFORE XOR)
    // addr = k_pack * 8 + m * 64 + k_in_pack
    return (k_pack * 8 + m * 64 + k_in_pack) * ELEM_SIZE;
}

// With XOR (simplified - actual XOR is more complex)
function getAddress_PackedXOR(row, col) {
    const k_pack = Math.floor(col / kKPack);
    const k_in_pack = col % kKPack;
    const m = row;

    // XOR permutation on k_pack dimension
    const MLdsLayer = 2;
    const m_layer = Math.floor(m / (M / MLdsLayer));
    const xor_k_pack = k_pack ^ (m % 8);

    // Physical address with XOR
    return (xor_k_pack * 8 + m * 64 + k_in_pack) * ELEM_SIZE;
}

console.log("=".repeat(80));
console.log("QUESTION: What does Lane 0 read for TRANSPOSE?");
console.log("=".repeat(80));
console.log();

console.log("Transpose read [K,M]: Each lane reads M1=8 consecutive M elements");
console.log("Lane 0: K=0, M=[0-7]");
console.log();

console.log("WITHOUT XOR:");
console.log("-".repeat(70));
for (let m = 0; m < 8; m++) {
    const k = 0;
    const addr = getAddress_Packed(m, k);
    console.log(`  [m=${m}, k=${k}]: addr=${addr} (${addr/2} elements)`);
}

console.log();
console.log("Observations:");
console.log("  - k=0 means k_pack=0, k_in_pack=0");
console.log("  - All elements are at position 0 within pack 0");
console.log("  - BUT they're in different M locations!");
console.log("  - M stride = 64 elements = 128 bytes");
console.log("  - Addresses: 0, 128, 256, 384, 512, 640, 768, 896");
console.log("  - These are NOT contiguous!");
console.log();

const addrs_without_xor = [];
for (let m = 0; m < 8; m++) {
    addrs_without_xor.push(getAddress_Packed(m, 0));
}

console.log("Banks (without XOR):");
const banks_without_xor = addrs_without_xor.map(a => Math.floor(a/4) % 32);
console.log(`  ${banks_without_xor.join(', ')}`);
console.log();

console.log("=".repeat(80));
console.log("WITH XOR:");
console.log("-".repeat(70));
for (let m = 0; m < 8; m++) {
    const k = 0;
    const addr = getAddress_PackedXOR(m, k);
    const k_pack = Math.floor(k / kKPack);
    const xor_k_pack = k_pack ^ (m % 8);
    console.log(`  [m=${m}, k=${k}]: k_pack=${k_pack}, XOR=${m%8}, xor_k_pack=${xor_k_pack}, addr=${addr}`);
}

console.log();
const addrs_with_xor = [];
for (let m = 0; m < 8; m++) {
    addrs_with_xor.push(getAddress_PackedXOR(m, 0));
}

console.log("Banks (with XOR):");
const banks_with_xor = addrs_with_xor.map(a => Math.floor(a/4) % 32);
console.log(`  ${banks_with_xor.join(', ')}`);
console.log();

// Check for conflicts
const bank_counts = {};
for (const bank of banks_with_xor) {
    bank_counts[bank] = (bank_counts[bank] || 0) + 1;
}

const max_count = Math.max(...Object.values(bank_counts));
if (max_count > 1) {
    console.log(`CONFLICT: ${max_count}-way intra-lane conflict!`);
    console.log("Conflicting banks:");
    for (const [bank, count] of Object.entries(bank_counts)) {
        if (count > 1) {
            console.log(`  Bank ${bank}: accessed ${count} times`);
        }
    }
} else {
    console.log("NO intra-lane conflicts!");
}

console.log();
console.log("=".repeat(80));
console.log("KEY INSIGHT:");
console.log("=".repeat(80));
console.log();
console.log("The packing helps for:");
console.log("  ✓ WRITE [M,K] row-wise: consecutive K values ARE in same pack");
console.log("  ✓ GEMM compute: reading consecutive elements along one dimension");
console.log();
console.log("The packing DOESN'T help for:");
console.log("  ✗ TRANSPOSE read [K,M]: consecutive M values are NOT in same pack!");
console.log("  ✗ They're strided by 64 elements (M stride)");
console.log();
console.log("For transpose, each lane still reads STRIDED elements,");
console.log("just like the non-packed model!");
console.log();
console.log("So the calculator's analysis (8 strided elements) is closer");
console.log("to correct for TRANSPOSE operations.");
console.log();

console.log("=".repeat(80));
console.log("WHERE DO THE 3,072 XOR CONFLICTS COME FROM?");
console.log("=".repeat(80));
console.log();
console.log("Let me recalculate based on strided access...");
console.log();

// Check if there are actually intra-lane conflicts with strided access
const unique_banks = new Set(banks_with_xor);
console.log(`Unique banks accessed: ${unique_banks.size}`);
console.log(`Total accesses: ${banks_with_xor.length}`);

if (unique_banks.size < banks_with_xor.length) {
    console.log("There ARE intra-lane conflicts!");
} else {
    console.log("There are NO intra-lane conflicts!");
}
console.log();

console.log("So the conflicts must come from:");
console.log("  1. WRITE operation (FP16 pairing)");
console.log("  2. Inter-lane conflicts (if any)");
console.log("  3. Other phases");
console.log("  4. Accumulation across iterations");
console.log();
