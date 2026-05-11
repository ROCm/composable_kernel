/*
 * FP16 Conflict Calculator - Phase-based model
 *
 * Hypothesis: Counter counts per-phase (8 lanes), not per-half (32 lanes)
 *
 * Target for test_exact_kernel_pattern (1 block):
 * - 4 WFs × 8 dm × ? per phase = 192 total
 * - 192 / 32 = 6 conflicts per WF per dm (for 8 phases)
 * - That's ~0.75 conflicts per phase? Doesn't make sense...
 *
 * Let me try: 192 = 4 WFs × 8 dm × 6 conflicts
 * So 6 conflicts per WF per dm total
 * Split into 8 phases: 6/8 = 0.75 per phase (fractional)
 *
 * Alternative: maybe profiler aggregates differently.
 * Let me just count unique slots per bank per phase.
 */

#include <iostream>
#include <set>
#include <map>

constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kBlockSize = 256;
constexpr int kWavefronts = kBlockSize / 64;  // 4
constexpr int DataTypeSize = 2;

int plain_offset(int m, int k) {
    return m * kK + k;
}

// Count conflicts using 8-lane phases
int analyze_phase_based(int wf, int dm, bool verbose = false) {
    int conflicts = 0;

    // 8 phases of 8 lanes each
    for (int phase = 0; phase < 8; phase++) {
        std::map<int, std::set<int>> bank_slots;

        for (int lane_in_phase = 0; lane_in_phase < 8; lane_in_phase++) {
            int lane = phase * 8 + lane_in_phase;

            int k1 = wf;
            int k2 = lane % 8;
            int m0 = lane / 8;

            int k = k1 * 8 + k2;
            int m = m0 * 8 + dm;

            int offset = plain_offset(m, k);
            int byte_addr = offset * DataTypeSize;
            int slot = byte_addr / 4;
            int bank = slot % 32;

            bank_slots[bank].insert(slot);
        }

        int phase_conflicts = 0;
        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                phase_conflicts += slots.size() - 1;
            }
        }
        conflicts += phase_conflicts;

        if (verbose && phase_conflicts > 0) {
            std::cout << "  Phase " << phase << ": ";
            for (const auto& [bank, slots] : bank_slots) {
                std::cout << "B" << bank << "(" << slots.size() << ") ";
            }
            std::cout << "-> " << phase_conflicts << "\n";
        }
    }

    return conflicts;
}

// Count conflicts using 32-lane halves
int analyze_half_based(int wf, int dm, bool verbose = false) {
    int conflicts = 0;

    for (int half = 0; half < 2; half++) {
        std::map<int, std::set<int>> bank_slots;

        for (int lane_in_half = 0; lane_in_half < 32; lane_in_half++) {
            int lane = half * 32 + lane_in_half;

            int k1 = wf;
            int k2 = lane % 8;
            int m0 = lane / 8;

            int k = k1 * 8 + k2;
            int m = m0 * 8 + dm;

            int offset = plain_offset(m, k);
            int byte_addr = offset * DataTypeSize;
            int slot = byte_addr / 4;
            int bank = slot % 32;

            bank_slots[bank].insert(slot);
        }

        int half_conflicts = 0;
        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                half_conflicts += slots.size() - 1;
            }
        }
        conflicts += half_conflicts;

        if (verbose) {
            std::cout << "  Half " << half << ": ";
            for (const auto& [bank, slots] : bank_slots) {
                std::cout << "B" << bank << "(" << slots.size() << ") ";
            }
            std::cout << "-> " << half_conflicts << "\n";
        }
    }

    return conflicts;
}

int main() {
    std::cout << "=== FP16 Conflict Calculator - Phase vs Half ===\n\n";

    // Test for one block, one iteration
    int phase_total = 0;
    int half_total = 0;

    for (int wf = 0; wf < kWavefronts; wf++) {
        for (int dm = 0; dm < 8; dm++) {
            phase_total += analyze_phase_based(wf, dm);
            half_total += analyze_half_based(wf, dm);
        }
    }

    std::cout << "One block (4 WFs × 8 dm):\n";
    std::cout << "  Phase-based (8 lanes): " << phase_total << "\n";
    std::cout << "  Half-based (32 lanes): " << half_total << "\n";
    std::cout << "  Target (profiler): 192\n\n";

    std::cout << "Per WF-dm analysis (phase-based):\n";
    std::cout << "  Total: " << phase_total << " / 32 = " << (phase_total / 32.0) << " per (WF, dm)\n\n";

    // Detailed analysis
    std::cout << "=== Detailed: WF0, dm=0 (phase-based) ===\n";
    int p0 = analyze_phase_based(0, 0, true);
    std::cout << "  Total: " << p0 << "\n\n";

    std::cout << "=== Detailed: WF0, dm=0 (half-based) ===\n";
    int h0 = analyze_half_based(0, 0, true);
    std::cout << "  Total: " << h0 << "\n\n";

    // What if we only count conflicts WITHIN phases but NOT across phases?
    // I.e., phase 0 and phase 1 accessing same bank don't conflict
    // because they execute sequentially.
    std::cout << "=== Realization ===\n";
    std::cout << "Phase-based and Half-based show same result because\n";
    std::cout << "phases execute sequentially, not in parallel.\n";
    std::cout << "The 32-lane half is what executes in parallel.\n";

    // Let me try: what if ONLY same-slot pairs count as 0?
    // And each additional slot costs 1?
    std::cout << "\n=== Alternative model: unique slots per bank per half ===\n";
    std::cout << "WF0, dm=0:\n";
    std::cout << "  4 banks, each with 8 accesses, 4 unique slots per bank\n";
    std::cout << "  Conflicts = 4 banks × (4-1) = 12 per half\n";
    std::cout << "  Total = 12 × 2 = 24 per (WF, dm)\n";
    std::cout << "  Total per block = 24 × 4 WFs × 8 dm = 768\n";
    std::cout << "  But profiler shows 192!\n\n";

    std::cout << "=== Ratio ===\n";
    std::cout << "Our model: 768, Profiler: 192, Ratio: " << (768.0/192.0) << "\n";
    std::cout << "Factor of 4x off.\n\n";

    // Maybe FP16 same-slot means 4 elements, not 2?
    std::cout << "=== Hypothesis: FP16 4-element slot ===\n";
    std::cout << "If 4 FP16 fit in one conflict-free unit (8 bytes):\n";
    std::cout << "  8 accesses / 4 per unit = 2 units per bank\n";
    std::cout << "  Conflicts = 4 banks × (2-1) = 4 per half\n";
    std::cout << "  Total = 4 × 2 = 8 per (WF, dm)\n";
    std::cout << "  Total per block = 8 × 4 WFs × 8 dm = 256\n";
    std::cout << "  Still not 192...\n\n";

    // Maybe conflicts are counted per-wavefront, not per-half?
    std::cout << "=== Hypothesis: per-wavefront counting ===\n";
    std::cout << "If entire 64-lane wavefront conflicts are counted as one:\n";
    std::cout << "  4 banks × (8 slots - 1) = 28 per (WF, dm)\n";
    std::cout << "  Wait, we have 8 unique slots per bank over 64 lanes\n";
    std::cout << "  Conflicts = 4 banks × (8-1) = 28\n";
    std::cout << "  Hmm but that's not 192 either...\n\n";

    // 192 / (4 WF × 8 dm) = 6 conflicts per (WF, dm)
    std::cout << "=== Target breakdown ===\n";
    std::cout << "192 total / (4 WF × 8 dm) = 6 conflicts per (WF, dm)\n";
    std::cout << "6 conflicts = ?\n";
    std::cout << "  Maybe: 2 banks × (4-1) = 6? Banks {0,16} each with 4 unique slots?\n";
    std::cout << "  Check: dm=0 hits rows m=0,8,16,24,32,40,48,56\n";
    std::cout << "  Offsets: 0,256,512,768,1024,1280,1536,1792\n";
    std::cout << "  Bytes: 0,512,1024,1536,2048,2560,3072,3584\n";
    std::cout << "  Slots: 0,128,256,384,512,640,768,896\n";
    std::cout << "  Banks: 0,0,0,0,0,0,0,0 (all bank 0!)\n";
    std::cout << "  Wait that's 8 accesses to bank 0 = 7 conflicts\n\n";

    // Hmm let me reconsider the m0 distribution
    std::cout << "=== Re-examine m0 distribution ===\n";
    std::cout << "For WF0 (k1=0), lanes 0-63:\n";
    std::cout << "  Lane 0: k2=0, m0=0\n";
    std::cout << "  Lane 8: k2=0, m0=1\n";
    std::cout << "  Lane 16: k2=0, m0=2\n";
    std::cout << "  ...\n";
    std::cout << "  Lane 56: k2=0, m0=7\n\n";

    std::cout << "For dm=0, the 64 lanes access:\n";
    std::cout << "  m = m0*8 + 0 = {0, 8, 16, 24, 32, 40, 48, 56}\n";
    std::cout << "  k = 0*8 + k2 = {0, 1, 2, 3, 4, 5, 6, 7}\n\n";

    std::cout << "So each m0 group (8 lanes) accesses:\n";
    std::cout << "  Same m row, different k columns\n";
    std::cout << "  Offsets: m*32 + k = m*32 + {0,1,2,3,4,5,6,7}\n";
    std::cout << "  These are consecutive -> no conflicts within m0 group!\n\n";

    std::cout << "Conflicts arise between DIFFERENT m0 groups:\n";
    std::cout << "  m0=0: offsets {0,1,2,3,4,5,6,7}\n";
    std::cout << "  m0=1: offsets {256+0,1,2,...,7} = {256,257,...,263}\n";
    std::cout << "  Banks for m0=0: {0,0,0,0,1,1,1,1} (slots 0,0,1,1,2,2,3,3)\n";
    std::cout << "  Banks for m0=1: {0,0,0,0,1,1,1,1} (slots 128,128,129,129,...)\n";
    std::cout << "  Same banks but DIFFERENT slots -> conflicts!\n\n";

    // Refined model
    std::cout << "=== Refined conflict counting ===\n";
    std::cout << "For half 0 (m0=0,1,2,3 and all k2):\n";
    std::cout << "  4 m0 values × 8 k2 values = 32 lanes\n";
    std::cout << "  Each (m0, k2) pair has unique (m, k) -> unique slot\n";
    std::cout << "  Bank = (offset * 2 / 4) % 32 = (m*32 + k) / 2 % 32\n\n";

    // Let me trace actual banks for half 0
    std::cout << "Half 0 bank analysis:\n";
    for (int m0 = 0; m0 < 4; m0++) {
        int m = m0 * 8;
        std::cout << "  m0=" << m0 << " (m=" << m << "): banks { ";
        for (int k2 = 0; k2 < 8; k2++) {
            int k = k2;
            int offset = m * 32 + k;
            int byte_addr = offset * 2;
            int slot = byte_addr / 4;
            int bank = slot % 32;
            std::cout << bank << " ";
        }
        std::cout << "}\n";
    }

    return 0;
}
