"""
A pedagogical Python model of GPU global-memory coalescing on AMD CDNA.

Run:
    python3 coalescing_model.py

Debug:
    Set breakpoints inside `execute_wave_load` to watch how the memory unit:
      1. collects all lane addresses,
      2. groups them by 64-byte cache line,
      3. issues ONE transaction per unique line,
      4. scatters bytes back to lanes.

This is not the real silicon pipeline. It is a faithful *logical* model of the
coalescing step, intended only for intuition building.
"""

from collections import defaultdict
from dataclasses import dataclass, field


CACHE_LINE = 64   # bytes. AMD CDNA global memory transaction granularity.
WAVE_SIZE  = 64   # lanes per wavefront on AMD.


@dataclass
class WaveLoadResult:
    """What one wave-level load produced."""
    lane_bytes:       list        # one bytes object per lane
    num_transactions: int         # how many 64B HBM fetches were issued
    useful_bytes:     int         # bytes the lanes actually wanted
    fetched_bytes:    int         # bytes the HBM bus actually moved
    line_map:         dict = field(default_factory=dict)  # line_id -> contributors

    @property
    def efficiency(self) -> float:
        return self.useful_bytes / self.fetched_bytes


def execute_wave_load(lane_addresses, lane_nbytes, HBM):
    """
    Model one wavefront executing a single global-memory load.

    Parameters
    ----------
    lane_addresses : list[int]
        Byte addresses one per lane. Must have length WAVE_SIZE.
    lane_nbytes : int
        How many bytes EACH lane wants to read this instruction.
        4  = ds/global_load_b32  (one dword)
        8  = ..._b64             (two dwords)
        16 = ..._b128            (four dwords = float4 / uint4)
    HBM : bytearray
        Pretend global memory.

    Returns
    -------
    WaveLoadResult
    """
    assert len(lane_addresses) == WAVE_SIZE, \
        f"expected {WAVE_SIZE} lane addresses, got {len(lane_addresses)}"

    # ------------------------------------------------------------------
    # STEP 1: every lane's address has already been computed in parallel.
    #         SIMT means all 64 lanes execute the same load instruction
    #         but with their own address value.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # STEP 2: The memory unit walks every lane's request and breaks it
    #         into pieces that each fit inside a single 64B cache line.
    #
    #         line_requests[line_id] is a list of (lane, offset_in_line, nbytes)
    #         contributions from lanes that want data in that line.
    # ------------------------------------------------------------------
    line_requests = defaultdict(list)

    for lane, addr in enumerate(lane_addresses):
        remaining = lane_nbytes
        cur = addr
        while remaining > 0:
            line_id = cur // CACHE_LINE
            offset  = cur %  CACHE_LINE
            chunk   = min(remaining, CACHE_LINE - offset)
            line_requests[line_id].append((lane, offset, chunk))
            cur       += chunk
            remaining -= chunk

    # ------------------------------------------------------------------
    # STEP 3: THE COALESCING STEP.
    #         Issue exactly ONE 64B HBM transaction per unique cache line.
    #         This is where "64 lane requests -> N transactions" happens.
    # ------------------------------------------------------------------
    fetched_lines = {}
    for line_id in line_requests:
        start = line_id * CACHE_LINE
        fetched_lines[line_id] = bytes(HBM[start : start + CACHE_LINE])

    num_transactions = len(fetched_lines)

    # ------------------------------------------------------------------
    # STEP 4: Scatter the fetched bytes back to the lanes that wanted them.
    #         A single wide lane (e.g. b128 straddling a line) reassembles
    #         its bytes from multiple line contributions in address order.
    # ------------------------------------------------------------------
    # Collect per-lane pieces along with their original address so we can
    # reassemble in order (important for lanes that straddle lines).
    per_lane_pieces = defaultdict(list)  # lane -> list of (addr, bytes)

    for line_id, reqs in line_requests.items():
        line_bytes = fetched_lines[line_id]
        line_base  = line_id * CACHE_LINE
        for lane, offset, chunk in reqs:
            piece_addr  = line_base + offset
            piece_bytes = line_bytes[offset : offset + chunk]
            per_lane_pieces[lane].append((piece_addr, piece_bytes))

    lane_bytes = [b""] * WAVE_SIZE
    for lane, pieces in per_lane_pieces.items():
        pieces.sort(key=lambda p: p[0])
        lane_bytes[lane] = b"".join(p[1] for p in pieces)

    # ------------------------------------------------------------------
    # STEP 5: Accounting.
    # ------------------------------------------------------------------
    useful_bytes  = WAVE_SIZE * lane_nbytes
    fetched_bytes = num_transactions * CACHE_LINE

    return WaveLoadResult(
        lane_bytes       = lane_bytes,
        num_transactions = num_transactions,
        useful_bytes     = useful_bytes,
        fetched_bytes    = fetched_bytes,
        line_map         = dict(line_requests),
    )


# ======================================================================
# Scenarios
# ======================================================================

def make_hbm(size_bytes: int) -> bytearray:
    """A recognisable HBM: byte i holds (i & 0xFF)."""
    return bytearray(i & 0xFF for i in range(size_bytes))


def report(title, result: WaveLoadResult):
    print(f"--- {title} ---")
    print(f"  transactions : {result.num_transactions}")
    print(f"  useful bytes : {result.useful_bytes}")
    print(f"  fetched bytes: {result.fetched_bytes}")
    print(f"  efficiency   : {result.efficiency*100:.2f}%")
    unique_lines = sorted(result.line_map.keys())
    print(f"  unique lines : {len(unique_lines)}  "
          f"(first few: {unique_lines[:6]}{' ...' if len(unique_lines) > 6 else ''})")
    print()


def scenario_1_coalesced_b32(HBM):
    """Every lane reads 4 bytes, lane i -> addr 4*i. Fully contiguous."""
    addrs = [4 * lane for lane in range(WAVE_SIZE)]
    return execute_wave_load(addrs, 4, HBM)


def scenario_2_strided_b32(HBM):
    """Every lane reads 4 bytes but strides by one cache line. Worst case."""
    addrs = [256 * lane for lane in range(WAVE_SIZE)]  # 256B stride
    return execute_wave_load(addrs, 4, HBM)


def scenario_3_coalesced_b128(HBM):
    """Every lane reads 16 bytes, lane i -> addr 16*i. Contiguous float4s."""
    addrs = [16 * lane for lane in range(WAVE_SIZE)]
    return execute_wave_load(addrs, 16, HBM)


def scenario_4_strided_b128(HBM):
    """Wide loads but scattered: width does NOT rescue bad patterns."""
    addrs = [1024 * lane for lane in range(WAVE_SIZE)]
    return execute_wave_load(addrs, 16, HBM)


def scenario_5_misaligned_b128(HBM):
    """
    Contiguous b128, but the base is shifted by 4 bytes so every lane
    straddles a cache-line boundary. Shows how misalignment inflates
    transaction count.
    """
    base = 4
    addrs = [base + 16 * lane for lane in range(WAVE_SIZE)]
    return execute_wave_load(addrs, 16, HBM)


def scenario_6_column_of_rowmajor(HBM):
    """
    Reading a column of a row-major fp32 matrix with row length = 1024 floats.
    Each lane -> different row, same column -> different 64B line each.
    """
    ROW_FLOATS = 1024
    addrs = [lane * ROW_FLOATS * 4 for lane in range(WAVE_SIZE)]
    return execute_wave_load(addrs, 4, HBM)


def scenario_7_row_of_rowmajor_with_vec(HBM):
    """
    Reading a row of a row-major fp16 tile with b128 per lane.
    Each lane owns 8 fp16 elements; 64 lanes cover 512 fp16 = 1024 B contiguous.
    """
    addrs = [16 * lane for lane in range(WAVE_SIZE)]  # 16B per lane, contiguous
    return execute_wave_load(addrs, 16, HBM)


def main():
    HBM = make_hbm(1 << 20)  # 1 MiB

    print(f"CACHE_LINE = {CACHE_LINE} B,  WAVE_SIZE = {WAVE_SIZE} lanes\n")

    report("1. coalesced b32 (contiguous 4B per lane)",
           scenario_1_coalesced_b32(HBM))

    report("2. strided b32 (256B stride per lane)",
           scenario_2_strided_b32(HBM))

    report("3. coalesced b128 (contiguous 16B per lane)",
           scenario_3_coalesced_b128(HBM))

    report("4. strided b128 (1024B stride per lane)",
           scenario_4_strided_b128(HBM))

    report("5. misaligned b128 (contiguous but base=4, straddles lines)",
           scenario_5_misaligned_b128(HBM))

    report("6. column of row-major fp32 matrix",
           scenario_6_column_of_rowmajor(HBM))

    report("7. row of row-major fp16 tile with b128 per lane",
           scenario_7_row_of_rowmajor_with_vec(HBM))


if __name__ == "__main__":
    main()
