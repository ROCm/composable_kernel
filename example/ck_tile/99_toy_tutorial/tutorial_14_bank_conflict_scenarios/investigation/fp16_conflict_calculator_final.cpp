/*
 * FP16 LDS Bank Conflict Calculator - Final Version
 *
 * Targets:
 * - RowMajorTransposeKernel (no XOR): 7,168 conflicts
 * - ProductionTransposeKernel (XOR): 3,072 conflicts
 *
 * Configuration: M=256, K=128, Tile=[64,32]
 * 4 M-blocks × 4 K-iterations = 16 total iterations
 *
 * Empirical finding from profiler tests:
 * - Read-only pattern (transpose): 192 conflicts per iteration
 * - Write-only pattern (row-major): 192 conflicts per iteration
 * - Combined: 384 per iteration
 *
 * BUT real kernels show:
 * - No XOR: 7168 / 16 = 448 per iteration
 * - XOR: 3072 / 16 = 192 per iteration
 *
 * The XOR kernel matches read-only! So XOR eliminates write conflicts.
 * The no-XOR kernel has 448 = 192 (read) + 256 (write)
 *
 * The discrepancy suggests that the real kernels have different write patterns
 * than our simple tests. Let me model the actual conflict sources.
 */

#include <iostream>
#include <set>
#include <map>
#include <cmath>

// Constants for RowMajorTransposeKernel
constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kBlockSize = 256;
constexpr int kWavefronts = kBlockSize / 64;  // 4
constexpr int DataTypeSize = 2;  // FP16

// Calculate plain row-major offset
int plain_offset(int m, int k) {
    return m * kK + k;
}

// Analyze conflicts for one ds_read_u16 instruction
// Returns conflict count for one wavefront, one dm value
int analyze_one_read(int wf, int dm, bool verbose = false) {
    int conflicts = 0;

    // Process 64 lanes in two halves of 32
    for (int half = 0; half < 2; half++) {
        std::map<int, std::set<int>> bank_slots;

        for (int lane_in_half = 0; lane_in_half < 32; lane_in_half++) {
            int lane = half * 32 + lane_in_half;

            // Thread distribution:
            // k1 = wf (wavefront index, 0-3)
            // k2 = lane % 8 (0-7)
            // m0 = lane / 8 (0-7)
            int k1 = wf;
            int k2 = lane % 8;
            int m0 = lane / 8;

            int k = k1 * 8 + k2;  // Full K coordinate
            int m = m0 * 8 + dm;  // Full M coordinate

            int offset = plain_offset(m, k);
            int byte_addr = offset * DataTypeSize;
            int slot = byte_addr / 4;
            int bank = slot % 32;

            bank_slots[bank].insert(slot);
        }

        // Count conflicts
        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                conflicts += slots.size() - 1;
            }
        }

        if (verbose) {
            std::cout << "  Half " << half << ": ";
            for (const auto& [bank, slots] : bank_slots) {
                std::cout << "B" << bank << "(" << slots.size() << ") ";
            }
            std::cout << "-> " << conflicts << " total\n";
        }
    }

    return conflicts;
}

// Calculate total read conflicts for one iteration (one block, one k-iter)
int calc_read_conflicts_per_iter() {
    int total = 0;

    // 4 wavefronts
    for (int wf = 0; wf < kWavefronts; wf++) {
        // 8 dm values (8 reads per thread)
        for (int dm = 0; dm < 8; dm++) {
            total += analyze_one_read(wf, dm);
        }
    }

    return total;
}

// Analyze write conflicts for ds_write_b128
// Each thread writes 8 FP16 elements as one 128-bit write
int calc_write_conflicts_per_iter() {
    int total = 0;

    // For write, each thread writes to consecutive K addresses (row-major)
    // The 128-bit write spans 4 consecutive 4-byte slots

    for (int wf = 0; wf < kWavefronts; wf++) {
        // For each dm value written (but write is vectorized, so all 8 at once?
        // Actually, looking at assembly: ds_write_b128 writes 16 bytes = 8 FP16
        // But the thread only writes to ONE m value per iteration!

        // Thread writes: for fixed (k, m0*8+dm) where dm loops 0..7
        // But with ds_write_b128, it writes 8 consecutive K values at once

        // Let me model what the store_tile does:
        // Each thread has coordinates based on distribution
        // Thread writes 8 consecutive K values at its M row

        for (int half = 0; half < 2; half++) {
            std::map<int, std::set<int>> bank_slots;

            for (int lane_in_half = 0; lane_in_half < 32; lane_in_half++) {
                int lane = half * 32 + lane_in_half;

                int k1 = wf;
                int k2 = lane % 8;
                int m0 = lane / 8;

                // Thread writes 8 K values starting at k = k1*8 + k2
                // Wait no - the ds_write_b128 is for consecutive memory
                // Actually thread writes to its specific (m, k) position

                // For store_tile with distribution:
                // thread_(m0, k1, k2) writes to m=m0*8+dm, k=k1*8+k2 for dm=0..7
                // These are NOT consecutive in memory!

                // Let me check the actual store pattern...
                // For ONE ds_write_b128: thread writes 8 FP16 = 16 bytes
                // These should be 8 consecutive K values at fixed M

                // So for each m0 row:
                int m = m0 * 8;  // Base M for this thread group

                // The write covers k values: k1*8 + k2 for k2=0..7
                // But that's across multiple threads, not one thread!

                // For ONE thread's write:
                // k = k1*8 + k2 (fixed)
                // m = m0*8 + dm for dm=0..7 (8 values)
                // Offsets: (m0*8+dm)*32 + k1*8+k2 for dm=0..7

                // These are NOT consecutive in memory!
                // offset[dm] = (m0*8 + dm)*32 + k = m0*256 + dm*32 + k
                // Stride = 32 elements = 64 bytes

                // So actually the write is probably split into 8 separate writes
                // or uses a scatter pattern. Let me check assembly again...

                // Actually I need to reconsider. The assembly shows:
                // v_perm_b32 v14, v21, v20, s16
                // ds_write_b128 v12, v[14:17]
                // This writes 4 registers (v14-v17) = 16 bytes as consecutive data

                // So the data is packed into 4 consecutive slots (16 bytes)
                // The BASE address is in v12

                // For store_tile, the data is likely from global load which is
                // consecutive in K. So write is row-major: consecutive K values.

                // For thread with (k, m), writes K values k, k+1, ..., k+7? No...
                // Actually each thread handles specific elements based on distribution.

                // Let's simplify: assume write is to consecutive K addresses
                int k = k1 * 8 + k2;
                int base_offset = plain_offset(m, k);
                int base_byte = base_offset * DataTypeSize;

                // ds_write_b128 = 16 bytes = 4 slots
                for (int s = 0; s < 4; s++) {
                    int slot = (base_byte + s * 4) / 4;
                    int bank = slot % 32;
                    bank_slots[bank].insert(slot);
                }
            }

            for (const auto& [bank, slots] : bank_slots) {
                if (slots.size() > 1) {
                    total += slots.size() - 1;
                }
            }
        }
    }

    return total;
}

int main() {
    std::cout << "=== FP16 Conflict Calculator - Final ===\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Tile: [" << kM << ", " << kK << "]\n";
    std::cout << "  Wavefronts: " << kWavefronts << "\n\n";

    int read_per_iter = calc_read_conflicts_per_iter();
    int write_per_iter = calc_write_conflicts_per_iter();

    std::cout << "Per-iteration conflicts:\n";
    std::cout << "  Read (transpose): " << read_per_iter << "\n";
    std::cout << "  Write (row-major): " << write_per_iter << "\n";
    std::cout << "  Total: " << (read_per_iter + write_per_iter) << "\n\n";

    int num_blocks = 4;
    int num_k_iters = 4;
    int total_iters = num_blocks * num_k_iters;

    int total_no_xor = (read_per_iter + write_per_iter) * total_iters;
    int total_xor = read_per_iter * total_iters;  // XOR eliminates write conflicts

    std::cout << "Total (" << total_iters << " iterations):\n";
    std::cout << "  No XOR: " << total_no_xor << " (target: 7168)\n";
    std::cout << "  XOR:    " << total_xor << " (target: 3072)\n\n";

    std::cout << "Match status:\n";
    std::cout << "  No XOR: " << (total_no_xor == 7168 ? "MATCH" : "NO MATCH") << "\n";
    std::cout << "  XOR:    " << (total_xor == 3072 ? "MATCH" : "NO MATCH") << "\n\n";

    // Debug: show one iteration in detail
    std::cout << "=== Detailed analysis for WF0, dm=0 ===\n";
    analyze_one_read(0, 0, true);

    return 0;
}
