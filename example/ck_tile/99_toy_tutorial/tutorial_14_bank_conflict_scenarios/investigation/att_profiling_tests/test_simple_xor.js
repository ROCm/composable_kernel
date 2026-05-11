#!/usr/bin/env node
// Quick test that simple XOR address calculation works

const M = 8;
const K = 32;
const ELEM_SIZE = 2;

function getAddress_XOR(row, col) {
    const xorCol = col ^ (row % 8);
    return row * (K * ELEM_SIZE) + xorCol * ELEM_SIZE;
}

function getBank(addr) {
    return Math.floor(addr / 4) % 32;
}

console.log("Testing simple XOR (calculator version):");
console.log("Lane 0: col=0, rows=[0-7]");
console.log();

const banks = [];
for (let row = 0; row < 8; row++) {
    const col = 0;
    const addr = getAddress_XOR(row, col);
    const bank = getBank(addr);
    banks.push(bank);
    console.log(`  Row ${row}: addr=${addr}, bank=${bank}`);
}

console.log();
console.log(`Banks: [${banks.join(', ')}]`);

const unique = new Set(banks);
if (unique.size === banks.length) {
    console.log("✓ All unique - NO conflicts!");
} else {
    console.log("✗ Conflicts detected!");
}
