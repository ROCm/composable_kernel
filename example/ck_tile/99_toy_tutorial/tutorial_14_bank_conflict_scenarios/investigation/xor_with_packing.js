#!/usr/bin/env node
// Understanding XOR with kKPack=8 (8 FP16 elements packed together)

console.log("=".repeat(80));
console.log("XOR with kKPack=8: Understanding the Packing");
console.log("=".repeat(80));
console.log();

console.log("KEY INSIGHT: The descriptor packs K in groups of 8 (kKPack=8)");
console.log();
console.log("For [M, K] = [64, 32] with kKPack=8:");
console.log("  Physical layout: [K/kKPack, M, kKPack] = [4, 64, 8]");
console.log("  K is split: K0=4 groups, each with kKPack=8 elements");
console.log();
console.log("Each 'pack' of 8 FP16 elements = 16 bytes (ds_read_b128 size)");
console.log("These 8 elements ARE contiguous in the pack dimension!");
console.log();

const M = 64;
const K = 32;
const kKPack = 8;
const ELEM_SIZE = 2;

console.log("=".repeat(80));
console.log("Lane 0 accessing column 0 (k=0), rows [0-7]:");
console.log("=".repeat(80));
console.log();

// With packing, we need to think differently
// The physical layout groups K into packs
// For column k=0:
//   K/kKPack index = 0 (first pack)
//   Within-pack index = 0

console.log("WITHOUT XOR (row-major [M,K] reshaped to [K/Pack, M, Pack]):");
console.log();

for (let m = 0; m < 8; m++) {
    const k = 0;  // Column 0
    const k_pack_idx = Math.floor(k / kKPack);  // 0
    const k_within_pack = k % kKPack;            // 0

    // Physical address in packed layout
    // [K/Pack=4, M=64, Pack=8]
    // stride: Pack=8 elements (16 bytes), M*Pack = 64*8 elements
    const addr = (k_pack_idx * M * kKPack + m * kKPack + k_within_pack) * ELEM_SIZE;
    const slot = Math.floor(addr / 4);
    const bank = slot % 32;

    console.log(`  Row ${m}: k_pack=${k_pack_idx}, k_in_pack=${k_within_pack}, addr=${addr}, bank=${bank}`);
}

console.log();
console.log("Banks: [0, 16, 0, 16, 0, 16, 0, 16]");
console.log("→ 4-way conflict (bank 0: 4x, bank 16: 4x)");
console.log();

console.log("=".repeat(80));
console.log("WITH XOR:");
console.log("=".repeat(80));
console.log();
console.log("XOR is applied at the PACK level, not element level!");
console.log("XOR operates on [K/Pack, M] dimensions BEFORE merging with Pack");
console.log();

// With XOR, the permutation happens at the pack level
// For accessing logical [M,K], we need to think about how XOR affects it

console.log("Hmm, I need to reconsider the XOR transform more carefully...");
console.log();
console.log("The user is saying that 8 elements are PACKED TOGETHER.");
console.log("This means ds_read_b128 reads 16 bytes = 8 FP16 = ONE PACK");
console.log();
console.log("If Lane 0 reads column k=0, rows [0-7],");
console.log("These 8 rows are in DIFFERENT packs!");
console.log();
console.log("Wait... let me reconsider the distribution.");
console.log();

console.log("=".repeat(80));
console.log("ACTUAL QUESTION: How many banks does ONE ds_read_b128 access?");
console.log("=".repeat(80));
console.log();
console.log("ds_read_b128 = 16 bytes = 4 bank slots (4 bytes each)");
console.log();
console.log("For FP16 (2 bytes per element):");
console.log("  16 bytes = 8 FP16 elements");
console.log("  4 bank slots = 4 banks (each slot holds 2 FP16 elements)");
console.log();
console.log("So each ds_read_b128 accesses **4 banks**, not 8!");
console.log();
console.log("The user is correct! Let me recalculate...");
console.log();

console.log("=".repeat(80));
console.log("CORRECTED: Lane 0 reading column 0, rows [0-7]");
console.log("=".repeat(80));
console.log();
console.log("But wait - the rows are NOT contiguous!");
console.log("Row 0 is at one address, row 1 is 64 bytes away.");
console.log("So this is NOT a single ds_read_b128!");
console.log();
console.log("OR... is the DISTRIBUTION packing them differently?");
console.log();
console.log("Let me check: What does the distribution actually give each lane?");
console.log();
