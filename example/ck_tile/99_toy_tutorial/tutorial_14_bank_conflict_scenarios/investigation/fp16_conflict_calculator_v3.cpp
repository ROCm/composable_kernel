/*
 * FP16 Conflict Calculator v3 - Match profiler exactly
 *
 * Key observations from testing:
 * 1. test_exact_kernel_pattern (1 block, 1 iteration) = 192 conflicts
 * 2. test_repeated_exact_pattern (4 blocks × 4 iters) = 3072 conflicts (XOR target!)
 *
 * But real kernel with XOR shows 3072 also!
 * And real kernel WITHOUT XOR shows 7168.
 *
 * Our test pattern matches XOR kernel perfectly.
 * The difference: test only does reads, real kernel does write+read.
 *
 * From profiling:
 * - test_write_only = 192 conflicts (write only, row-major)
 * - test_read_only = 192 conflicts (read only, transpose)
 * - test_write_then_read = 384 conflicts (both)
 *
 * Real kernel ratios:
 * - No XOR: 7168 / 16 iterations = 448 per iteration
 * - With XOR: 3072 / 16 iterations = 192 per iteration
 *
 * Our test "write + read" = 384 per iteration
 * Our test "read only" = 192 per iteration = matches XOR kernel!
 *
 * So XOR eliminates WRITE conflicts, not read conflicts!
 *
 * Let me verify this hypothesis:
 * - XOR kernel: only read conflicts = 192 × 16 = 3072 ✓
 * - Plain kernel: write + read conflicts = (256 + 192) × 16 = 7168?
 *   Actually (448 per iter) = 256 write + 192 read = 448
 *   448 × 16 = 7168 ✓
 *
 * So the model is:
 * - Write conflicts (row-major): depends on distribution
 * - Read conflicts (transpose): 192 per iteration for both XOR and plain
 * - XOR: eliminates write conflicts, keeps read conflicts
 * - Plain: has both write and read conflicts
 */

#include <iostream>
#include <set>
#include <map>

constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kKPack = 8;
constexpr int kBlockSize = 256;
constexpr int kWavefronts = 4;
constexpr int MLdsLayer = 2;

int calc_xor_offset(int m, int k) {
    int m_div = m / MLdsLayer;
    int layer = m % MLdsLayer;
    int k_div = k / kKPack;
    int k_pack = k % kKPack;
    int dim0 = k_div * MLdsLayer + layer;
    int dim1 = m_div;
    int xor_dim1 = dim1 ^ dim0;
    return dim0 * kKPack + xor_dim1 * (kK * MLdsLayer) + k_pack;
}

int calc_plain_offset(int m, int k) {
    return m * kK + k;
}

// Calculate write conflicts (store_tile pattern)
// Store is vectorized (ds_write_b128 = 8 FP16 elements per instruction)
// Thread writes 8 consecutive K elements at same M row
int calc_write_conflicts(bool use_xor) {
    auto calc_offset = use_xor ? calc_xor_offset : calc_plain_offset;
    int conflicts = 0;

    // For vectorized B128 write, all 8 elements go to consecutive bytes
    // Calculate conflicts per half-wavefront (32 lanes)
    for (int wf = 0; wf < kWavefronts; wf++) {
        for (int half = 0; half < 2; half++) {
            // Each ds_write_b128 writes 16 bytes = 4 slots
            // Count which banks get hit by multiple writes
            std::map<int, int> bank_write_count;

            for (int lane_in_half = 0; lane_in_half < 32; lane_in_half++) {
                int lane = half * 32 + lane_in_half;
                int k1 = wf, k2 = lane % 8, m0 = lane / 8;
                int k = k1 * 8 + k2;
                int m = m0 * 8 + 0;  // First dm value

                // Write 8 consecutive K values (but as vector, not scalar)
                // The base address determines which banks are hit
                int base_offset = calc_offset(m, k);
                int base_byte = base_offset * 2;

                // ds_write_b128 = 16 bytes = 4 slots
                for (int s = 0; s < 4; s++) {
                    int slot = (base_byte + s * 4) / 4;
                    int bank = slot % 32;
                    bank_write_count[bank]++;
                }
            }

            // Each bank can serve one 4-byte write per cycle
            // Conflicts = sum of (count - 1) for each bank
            for (const auto& [bank, count] : bank_write_count) {
                if (count > 1) {
                    conflicts += count - 1;
                }
            }
        }
    }

    return conflicts;
}

// Calculate read conflicts (transpose read pattern)
// Each thread reads 8 scalar ds_read_u16 instructions
int calc_read_conflicts(bool use_xor) {
    auto calc_offset = use_xor ? calc_xor_offset : calc_plain_offset;
    int conflicts = 0;

    for (int wf = 0; wf < kWavefronts; wf++) {
        for (int dm = 0; dm < 8; dm++) {
            for (int half = 0; half < 2; half++) {
                std::map<int, std::set<int>> bank_slots;

                for (int lane_in_half = 0; lane_in_half < 32; lane_in_half++) {
                    int lane = half * 32 + lane_in_half;
                    int k1 = wf, k2 = lane % 8, m0 = lane / 8;
                    int k = k1 * 8 + k2;
                    int m = m0 * 8 + dm;

                    int offset = calc_offset(m, k);
                    int byte_off = offset * 2;
                    int slot = byte_off / 4;
                    int bank = slot % 32;
                    bank_slots[bank].insert(slot);
                }

                for (const auto& [bank, slots] : bank_slots) {
                    if (slots.size() > 1) {
                        conflicts += slots.size() - 1;
                    }
                }
            }
        }
    }

    return conflicts;
}

int main() {
    std::cout << "=== FP16 Conflict Calculator v3 ===\n\n";

    int num_blocks = 4;
    int num_k_iters = 4;

    std::cout << "Per-iteration analysis (1 block × 1 k-iter):\n\n";

    int write_plain = calc_write_conflicts(false);
    int write_xor = calc_write_conflicts(true);
    int read_plain = calc_read_conflicts(false);
    int read_xor = calc_read_conflicts(true);

    std::cout << "WITHOUT XOR:\n";
    std::cout << "  Write conflicts: " << write_plain << "\n";
    std::cout << "  Read conflicts:  " << read_plain << "\n";
    std::cout << "  Total per iter:  " << (write_plain + read_plain) << "\n\n";

    std::cout << "WITH XOR:\n";
    std::cout << "  Write conflicts: " << write_xor << "\n";
    std::cout << "  Read conflicts:  " << read_xor << "\n";
    std::cout << "  Total per iter:  " << (write_xor + read_xor) << "\n\n";

    int total_iters = num_blocks * num_k_iters;
    std::cout << "Total (" << num_blocks << " blocks × " << num_k_iters << " k-iters = "
              << total_iters << " iterations):\n\n";

    int total_plain = (write_plain + read_plain) * total_iters;
    int total_xor = (write_xor + read_xor) * total_iters;

    std::cout << "WITHOUT XOR:\n";
    std::cout << "  Calculated: " << total_plain << "\n";
    std::cout << "  Target:     7168\n";
    std::cout << "  Match: " << (total_plain == 7168 ? "YES" : "NO") << "\n\n";

    std::cout << "WITH XOR:\n";
    std::cout << "  Calculated: " << total_xor << "\n";
    std::cout << "  Target:     3072\n";
    std::cout << "  Match: " << (total_xor == 3072 ? "YES" : "NO") << "\n";

    return 0;
}
