#!/usr/bin/env node
// Verify XOR bank access pattern - CORRECTED to match real CK descriptor

const M = 64;
const K = 32;
const kKPack = 8;
const ELEM_SIZE = 2;  // FP16
const MLdsLayer = 2;  // (32 * 4 / 32 / 2) = 2

// Correct XOR offset calculation matching RealKernelDescriptor
function getOffset_XOR(m, k) {
    // The descriptor transforms:
    // 1. Reshape to [K/kKPack*MLdsLayer, M/MLdsLayer, kKPack] = [8, 32, 8]
    //    with strides [kKPack=8, K*MLdsLayer=64, 1]
    // 2. XOR transform on [M/MLdsLayer, K/kKPack*MLdsLayer]
    // 3. Unmerge and merge back to [M, K]

    const m_hi = Math.floor(m / MLdsLayer);  // 0-31
    const m_lo = m % MLdsLayer;              // 0-1
    const k_hi = Math.floor(k / kKPack);     // 0-3
    const k_lo = k % kKPack;                 // 0-7

    // Index into reshaped [K/kKPack*MLdsLayer, M/MLdsLayer, kKPack]
    const idx0 = m_lo * (K / kKPack) + k_hi;  // 0-7
    const idx1 = m_hi;                         // 0-31
    const idx2 = k_lo;                         // 0-7

    // XOR transform
    const xor_idx0 = idx1 ^ idx0;
    const xor_idx1 = idx1;

    // Offset with strides [8, 64, 1]
    const offset = xor_idx0 * kKPack + xor_idx1 * (K * MLdsLayer) + idx2;
    return offset;
}

// Plain (no XOR) offset
function getOffset_Plain(m, k) {
    return m * K + k;
}

function getBank(offset) {
    const byte_offset = offset * ELEM_SIZE;
    return Math.floor(byte_offset / 4) % 32;
}

function getSlot(offset) {
    const byte_offset = offset * ELEM_SIZE;
    return Math.floor(byte_offset / 4);
}

console.log("=".repeat(80));
console.log("XOR Bank Access Verification - Corrected");
console.log("=".repeat(80));
console.log();

// Phase 0 lanes for READ (transpose)
const phase0Lanes = [0, 1, 2, 3, 20, 21, 22, 23];

console.log("Phase 0 READ lanes: [0, 1, 2, 3, 20, 21, 22, 23]");
console.log("Each lane reads 8 M values for its assigned K column");
console.log();

function analyzePhase(useXor, k_base = 0) {
    const getOffset = useXor ? getOffset_XOR : getOffset_Plain;
    const label = useXor ? "WITH XOR" : "WITHOUT XOR";

    console.log("=".repeat(80));
    console.log(`${label} - k_base=${k_base}`);
    console.log("=".repeat(80));
    console.log();

    // Track all slot accesses by bank for inter-lane analysis
    const bankToSlots = {};
    for (let b = 0; b < 32; b++) bankToSlots[b] = new Set();

    let totalIntraConflicts = 0;

    for (const lane of phase0Lanes) {
        const K2_idx = lane % 8;
        const M0_idx = Math.floor(lane / 8);
        const k = k_base + K2_idx;
        const m_start = M0_idx * 8;

        const banks = [];
        const slots = [];

        // Track slots per bank for this lane (intra-lane)
        const laneBankToSlots = {};

        for (let dm = 0; dm < 8; dm++) {
            const m = m_start + dm;
            const offset = getOffset(m, k);
            const slot = getSlot(offset);
            const bank = slot % 32;

            banks.push(bank);
            slots.push(slot);

            // Track for intra-lane
            if (!laneBankToSlots[bank]) laneBankToSlots[bank] = new Set();
            laneBankToSlots[bank].add(slot);

            // Track for inter-lane
            bankToSlots[bank].add(slot);
        }

        // Count intra-lane conflicts (multiple different slots in same bank)
        let laneIntraConflicts = 0;
        for (const [bank, slotSet] of Object.entries(laneBankToSlots)) {
            if (slotSet.size > 1) {
                laneIntraConflicts += slotSet.size - 1;
            }
        }
        totalIntraConflicts += laneIntraConflicts;

        const uniqueBanks = [...new Set(banks)].sort((a,b) => a-b);
        console.log(`Lane ${String(lane).padStart(2)}: k=${k}, m=[${m_start}-${m_start+7}]`);
        console.log(`  Banks: [${banks.join(',')}]`);
        console.log(`  Unique banks: ${uniqueBanks.length} → [${uniqueBanks.join(',')}]`);
        console.log(`  Intra-lane conflicts: ${laneIntraConflicts}`);
        console.log();
    }

    // Count inter-lane conflicts (multiple different slots in same bank across all lanes)
    let totalInterConflicts = 0;
    console.log("Inter-lane analysis (slots per bank across all 8 lanes):");
    for (let bank = 0; bank < 32; bank++) {
        const numSlots = bankToSlots[bank].size;
        if (numSlots > 0) {
            const conflicts = numSlots > 1 ? numSlots - 1 : 0;
            totalInterConflicts += conflicts;
            if (numSlots > 1) {
                console.log(`  Bank ${String(bank).padStart(2)}: ${numSlots} slots → ${conflicts} conflicts`);
            }
        }
    }

    console.log();
    console.log(`SUMMARY for k_base=${k_base}:`);
    console.log(`  Intra-lane conflicts: ${totalIntraConflicts}`);
    console.log(`  Inter-lane conflicts: ${totalInterConflicts}`);
    console.log();

    return { intra: totalIntraConflicts, inter: totalInterConflicts };
}

// Analyze all k_base values (0, 8, 16, 24)
console.log("\n" + "=".repeat(80));
console.log("FULL ANALYSIS - All k_base iterations");
console.log("=".repeat(80) + "\n");

let plainTotal = { intra: 0, inter: 0 };
let xorTotal = { intra: 0, inter: 0 };

for (const k_base of [0, 8, 16, 24]) {
    const plain = analyzePhase(false, k_base);
    plainTotal.intra += plain.intra;
    plainTotal.inter += plain.inter;

    const xor = analyzePhase(true, k_base);
    xorTotal.intra += xor.intra;
    xorTotal.inter += xor.inter;
}

console.log("=".repeat(80));
console.log("GRAND TOTALS (Phase 0 only, one tile)");
console.log("=".repeat(80));
console.log();
console.log("WITHOUT XOR:");
console.log(`  Intra-lane: ${plainTotal.intra}`);
console.log(`  Inter-lane: ${plainTotal.inter}`);
console.log(`  Total: ${plainTotal.intra + plainTotal.inter}`);
console.log();
console.log("WITH XOR:");
console.log(`  Intra-lane: ${xorTotal.intra}`);
console.log(`  Inter-lane: ${xorTotal.inter}`);
console.log(`  Total: ${xorTotal.intra + xorTotal.inter}`);
console.log();

// Scale to match profiler
const numPhases = 8;
const numKIterations = 4;
const numBlocks = 4;

console.log("=".repeat(80));
console.log(`SCALED (×${numPhases} phases × ${numKIterations} K-iters × ${numBlocks} blocks)`);
console.log("=".repeat(80));
console.log();

const scale = numPhases * numKIterations * numBlocks;
console.log("WITHOUT XOR:");
console.log(`  Scaled total: ${(plainTotal.intra + plainTotal.inter) * scale}`);
console.log();
console.log("WITH XOR:");
console.log(`  Scaled total: ${(xorTotal.intra + xorTotal.inter) * scale}`);
console.log();
console.log("PROFILER TARGETS:");
console.log("  WITHOUT XOR: 7168");
console.log("  WITH XOR: 3072");
