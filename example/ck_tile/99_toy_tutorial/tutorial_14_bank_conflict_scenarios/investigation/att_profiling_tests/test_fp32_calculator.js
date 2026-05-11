#!/usr/bin/env node
// Test FP32 vs FP16 bank patterns

const M = 8;
const K = 32;

console.log("=".repeat(80));
console.log("FP16 vs FP32: Bank Conflict Comparison");
console.log("=".repeat(80));
console.log();

// FP16: 2 bytes per element
console.log("FP16 (2 bytes/element):");
console.log("-".repeat(70));
const fp16_row_stride_bytes = K * 2;
const fp16_bank_stride = fp16_row_stride_bytes / 4;

console.log(`Row stride: ${fp16_row_stride_bytes} bytes = ${fp16_bank_stride} banks`);
console.log();
console.log("Reading column 0, rows [0-7]:");
for (let row = 0; row < 8; row++) {
    const addr = row * fp16_row_stride_bytes;
    const bank = Math.floor(addr / 4) % 32;
    console.log(`  Row ${row}: addr ${addr}, bank ${bank}`);
}

const fp16_banks = [];
for (let row = 0; row < 8; row++) {
    const addr = row * fp16_row_stride_bytes;
    fp16_banks.push(Math.floor(addr / 4) % 32);
}
console.log(`Banks: [${fp16_banks.join(', ')}]`);

const fp16_unique = new Set(fp16_banks);
const fp16_max_count = Math.max(...[...fp16_unique].map(b =>
    fp16_banks.filter(x => x === b).length
));
console.log(`Unique banks: ${fp16_unique.size}, max count: ${fp16_max_count}-way`);

console.log();
console.log("=".repeat(80));
console.log();

// FP32: 4 bytes per element
console.log("FP32 (4 bytes/element):");
console.log("-".repeat(70));
const fp32_row_stride_bytes = K * 4;
const fp32_bank_stride = fp32_row_stride_bytes / 4;

console.log(`Row stride: ${fp32_row_stride_bytes} bytes = ${fp32_bank_stride} banks`);
console.log();
console.log("Reading column 0, rows [0-7]:");
for (let row = 0; row < 8; row++) {
    const addr = row * fp32_row_stride_bytes;
    const bank = Math.floor(addr / 4) % 32;
    console.log(`  Row ${row}: addr ${addr}, bank ${bank}`);
}

const fp32_banks = [];
for (let row = 0; row < 8; row++) {
    const addr = row * fp32_row_stride_bytes;
    fp32_banks.push(Math.floor(addr / 4) % 32);
}
console.log(`Banks: [${fp32_banks.join(', ')}]`);

const fp32_unique = new Set(fp32_banks);
const fp32_max_count = Math.max(...[...fp32_unique].map(b =>
    fp32_banks.filter(x => x === b).length
));
console.log(`Unique banks: ${fp32_unique.size}, max count: ${fp32_max_count}-way`);

console.log();
console.log("=".repeat(80));
console.log("COMPARISON:");
console.log("=".repeat(80));
console.log();
console.log(`FP16: ${fp16_unique.size} unique banks, ${fp16_max_count}-way conflict`);
console.log(`FP32: ${fp32_unique.size} unique banks, ${fp32_max_count}-way conflict`);
console.log();

if (fp32_max_count > fp16_max_count) {
    console.log("⚠️  FP32 has WORSE conflicts than FP16 for transpose READ!");
    console.log("    This is because stride = 32 banks wraps to same bank");
} else if (fp32_max_count < fp16_max_count) {
    console.log("✓ FP32 has better conflicts than FP16");
} else {
    console.log("= FP32 and FP16 have similar conflicts");
}

console.log();
console.log("But FP32 should have NO pairing conflicts for WRITE!");
console.log("So total conflicts depend on WRITE vs READ balance.");
console.log();
