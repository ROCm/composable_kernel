#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Two-pass instruction-prefetch offset patcher.

Round 1: build with koffset=0 so the compiler emits s_prefetch_inst_pc_rel
         with placeholder operands.
Round 2: assemble the GPU .s via llvm-mc, disassemble with llvm-objdump to
         get exact hex addresses, compute correct PC-relative koffset/klength,
         then patch both the .s file and the GPU ELF inside the fat .o via
         direct binary patching (no recompilation needed, only a relink).

If the computed prefetch region has zero in-bounds cachelines, the 8-byte
s_prefetch_inst_pc_rel is replaced with 2× 4-byte s_nop 0.

Labels are discovered automatically from [ck_prefetch] / [ck_label] comments
in the generated .s assembly file — no source path needed.

Standalone usage (runs both rounds):
    python patch_prefetch_offset.py \\
        --build-dir  /path/to/build \\
        --target     <cmake-target> \\
        --objdump-mcpu gfx1201 \\
        [--dry-run]

CMake PRE_LINK usage (round 1 already done by cmake, only patch the .o):
    python patch_prefetch_offset.py \\
        --build-dir  /path/to/build \\
        --target     <cmake-target> \\
        --objdump-mcpu gfx1201 \\
        --skip-build-round1
"""

import argparse
import multiprocessing
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CACHELINE_SIZE = 128  # bytes per instruction cache line
KLENGTH_SHIFT = 6
KLENGTH_MASK = 0x7F << KLENGTH_SHIFT   # klength occupies bits [12:6] of dw0
KOFFSET_MASK = 0x00FFFFFF               # 24-bit signed PC-relative offset in dw1[23:0]
S_NOP_ENCODING = 0xBF800000             # s_nop 0 — SOPP opcode 0, simm16=0
NOP_KLENGTH_SENTINEL = -1               # klength sentinel: replace prefetch with 2× s_nop
PLACE_MODE_DEFAULT = 0
PLACE_MODE_BLOCK_ENTRY = 1
DIR_FORWARD = "forward"
DIR_BACKWARD = "backward"

# ---------------------------------------------------------------------------
# Module-level regex patterns
# ---------------------------------------------------------------------------

# Function-header label in .s files
FUNC_LABEL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_$.]*):\s*(?:;.*)?$")

# objdump function header (e.g. "0000000000001000 <funcname>:")
OBJDUMP_FUNC_RE = re.compile(r'^[0-9a-fA-F]+ <(.+?)>:\s*$')

# objdump instruction address from trailing comment (e.g. "// 00001000: F4...")
OBJDUMP_ADDR_RE = re.compile(r'//\s*([0-9a-fA-F]+):\s+[0-9a-fA-F]')

# Block label in .s (e.g. ".LBB1_3:")
BLOCK_LABEL_RE = re.compile(r'^\.[A-Za-z_]\w*:')


# ---------------------------------------------------------------------------
# Structured types for label classification
# ---------------------------------------------------------------------------

class PrefetchSite(NamedTuple):
    """A [ck_prefetch] marker in the merged .s ↔ objdump table."""
    idx: int            # index in merged list
    direction: str      # DIR_FORWARD or DIR_BACKWARD
    offset_cl: int      # cacheline offset from target

class TargetSite(NamedTuple):
    """A [ck_label] marker (INST_PREFETCH_TARGET) in the merged table."""
    idx: int            # index in merged list
    mode: int           # PLACE_MODE_DEFAULT or PLACE_MODE_BLOCK_ENTRY


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        self._file = log_path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        sys.stdout = self._stdout


# ---------------------------------------------------------------------------
# Instruction classification
# ---------------------------------------------------------------------------

def is_asm_instruction(line: str) -> bool:
    """Return True if the line is an instruction (not a comment/label/directive/blank)."""
    s = line.strip()
    if not s:
        return False
    if s[0] in (';', '/', '.', '#'):
        return False
    if s.split()[0].endswith(':'):
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")
    return result


def cmake_build(build_dir: Path, target: str, jobs: int) -> None:
    run(["cmake", "--build", str(build_dir), "--target", target, "-j", str(jobs), "--"], build_dir)


def find_asm_file(search_dir: Path, cpp_stem: str, gpu_arch: str = "") -> Path:
    """Find the GPU .s file produced by --save-temps."""
    all_candidates = sorted(
        search_dir.rglob(f"{cpp_stem}*.s"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not all_candidates:
        sys.exit(
            f"No .s file matching '{cpp_stem}*.s' found under {search_dir}.\n"
            "Make sure --save-temps is in the target's compile options."
        )

    def is_gpu(p: Path) -> bool:
        n = p.name
        if "-host-" in n:
            return False
        if "-hip-" in n:
            return True
        if gpu_arch and gpu_arch in n:
            return True
        return False

    gpu = [p for p in all_candidates if is_gpu(p)]
    chosen = gpu[0] if gpu else all_candidates[0]
    if not gpu:
        print(f"[warn] No GPU .s found; falling back to {chosen.name}")
    return chosen


def find_obj_file(build_dir: Path, target: str) -> Path:
    """Find the most recent .o for the given CMake target."""
    candidates = sorted(
        build_dir.rglob(f"{target}.dir/*.o"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        sys.exit(
            f"No .o file found under '*/{target}.dir/' in {build_dir}.\n"
            "Check that the target was built before running the patch script."
        )
    return candidates[0]


def run_objdump(objdump_path: str, mcpu: str, obj_path: Path) -> str:
    cmd = [objdump_path, f"--mcpu={mcpu}", "-d", str(obj_path)]
    print(f"[run] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        sys.exit(f"objdump failed:\n{result.stderr}")
    return result.stdout


def detect_mcpu_from_asm(asm_text: str) -> str:
    """Extract the GPU architecture from .amdgcn_target directive in the .s file.

    Looks for lines like:  .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
    Returns the gfx* portion (e.g. "gfx1201") or empty string if not found.
    """
    m = re.search(r'\.amdgcn_target\s+"[^"]*--(gfx[0-9a-zA-Z]+)', asm_text)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Label discovery from .s
# ---------------------------------------------------------------------------

def find_prefetch_labels_from_asm(asm_text: str) -> list[str]:
    """Return unique label names from [ck_prefetch] comments in the .s file."""
    label_re = re.compile(r';\s*\[ck_prefetch\].*\bname\s*=\s*(\w+)')
    seen: dict[str, None] = {}
    for line in asm_text.splitlines():
        m = label_re.search(line)
        if m:
            seen.setdefault(m.group(1), None)
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Assembly / objdump helpers
# ---------------------------------------------------------------------------

def assemble_gpu_asm(asm_file: Path, mcpu: str, objdump_path: str) -> Path:
    """Assemble GPU .s → temp .o via llvm-mc.  Returns path (caller deletes)."""
    llvm_mc = str(Path(objdump_path).parent / "llvm-mc")
    out_obj = asm_file.with_suffix(".ck_tmp_patching.o")
    run([llvm_mc, f"--mcpu={mcpu}", "--triple=amdgcn-amd-amdhsa",
         "--filetype=obj", "-o", str(out_obj), str(asm_file)], asm_file.parent)
    return out_obj


def parse_objdump_functions(objdump_text: str) -> dict[str, list[tuple[int, str]]]:
    """Parse llvm-objdump -d output into per-function (addr, instr_text) lists."""
    instr_re = re.compile(r'^\t(.+?)//\s*([0-9a-fA-F]+):\s+[0-9a-fA-F]')

    result: dict[str, list[tuple[int, str]]] = {}
    cur_name: str | None = None
    cur_entries: list[tuple[int, str]] = []

    for line in objdump_text.splitlines():
        m = OBJDUMP_FUNC_RE.match(line)
        if m:
            if cur_name is not None:
                result[cur_name] = cur_entries
            cur_name = m.group(1)
            cur_entries = []
        elif cur_name is not None:
            m2 = instr_re.match(line)
            if m2:
                cur_entries.append((int(m2.group(2), 16), m2.group(1).strip()))

    if cur_name is not None:
        result[cur_name] = cur_entries
    return result


def split_functions(asm_text: str) -> list[tuple[str, list[str]]]:
    """Split the .s file into per-function blocks."""
    blocks: list[tuple[str, list[str]]] = []
    current_name = "<top>"
    current_lines: list[str] = []
    for line in asm_text.splitlines():
        m = FUNC_LABEL_RE.match(line)
        if m:
            if current_lines:
                blocks.append((current_name, current_lines))
            current_name = m.group(1)
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_name, current_lines))
    return blocks


# ---------------------------------------------------------------------------
# Merge .s ↔ objdump and compute koffsets
# ---------------------------------------------------------------------------

def _merge_s_and_objdump(s_lines: list[str],
                          obj_entries: list[tuple[int, str]]) -> list[tuple[int | None, str]]:
    """Pair each .s instruction with its objdump entry by mnemonic matching.

    For each .s instruction we scan forward in objdump entries (up to
    MAX_LOOKAHEAD) to find a matching mnemonic.  This self-corrects drift
    from assembler-inserted NOPs or classifier mismatches.

    .p2align directives advance obj_idx to the next aligned entry.
    Comment/directive/label lines get addr=None.
    """
    MAX_LOOKAHEAD = 32
    p2align_re = re.compile(r'\.p2align\s+(\d+)')
    merged: list[tuple[int | None, str]] = []
    obj_idx = 0
    for line in s_lines:
        m = p2align_re.search(line)
        if m:
            if obj_idx < len(obj_entries):
                align = 1 << int(m.group(1))
                while obj_idx < len(obj_entries) and (obj_entries[obj_idx][0] % align) != 0:
                    obj_idx += 1
            merged.append((None, line))
            continue
        if is_asm_instruction(line):
            if obj_idx < len(obj_entries):
                s_mnem = line.strip().split()[0].lower()
                for scan in range(obj_idx, min(obj_idx + MAX_LOOKAHEAD, len(obj_entries))):
                    if obj_entries[scan][1].split()[0].lower() == s_mnem:
                        obj_idx = scan
                        break
                addr = obj_entries[obj_idx][0]
                obj_idx += 1
            else:
                addr = None
            merged.append((addr, line))
        else:
            merged.append((None, line))
    return merged


def _merge_all_functions(asm_text: str, objdump_text: str,
                         dump_dir: Path | None = None
                         ) -> dict[str, list[tuple[int | None, str]]]:
    """Merge .s ↔ objdump once per function.  Returns {funcname: merged_list}.
    Optionally dumps one file per function (not per label)."""
    s_blocks = split_functions(asm_text)
    obj_funcs = parse_objdump_functions(objdump_text)
    merged_funcs: dict[str, list[tuple[int | None, str]]] = {}

    for name, s_lines in s_blocks:
        if name not in obj_funcs:
            continue
        merged = _merge_s_and_objdump(s_lines, obj_funcs[name])
        merged_funcs[name] = merged

        if dump_dir is not None:
            safe = re.sub(r'[^A-Za-z0-9_]', '_', name)[:80]
            dump_path = dump_dir / f"merged_{safe}.txt"
            with dump_path.open('w', encoding='utf-8') as fh:
                for idx, (addr, line) in enumerate(merged):
                    addr_str = f'0x{addr:08x}' if addr is not None else '          '
                    fh.write(f'[{idx:5d}] {addr_str}  {line.rstrip()}\n')
            print(f"[dump] Merged table written to {dump_path}")

    return merged_funcs


def _resolve_target_address(
    merged: list[tuple[int | None, str]],
    tgt_idx: int,
    tgt_mode: int,
    name: str,
) -> int | None:
    """Resolve the target address for a prefetch's INST_PREFETCH_TARGET marker.

    BLOCK_ENTRY mode (1): scan backward for the nearest block label, then
    use the first instruction after it.
    DEFAULT mode (0): use the first instruction after the [ck_label] comment.
    """
    if tgt_mode == PLACE_MODE_BLOCK_ENTRY:
        block_idx: int | None = None
        for k in range(tgt_idx - 1, -1, -1):
            if BLOCK_LABEL_RE.match(merged[k][1].strip()):
                block_idx = k
                break

        scan_from = block_idx if block_idx is not None else tgt_idx
        target: int | None = None
        for k in range(scan_from + 1, len(merged)):
            if merged[k][0] is not None:
                target = merged[k][0]
                break

        if block_idx is not None and target is not None:
            orig_target: int | None = None
            for k in range(tgt_idx + 1, len(merged)):
                if merged[k][0] is not None:
                    orig_target = merged[k][0]
                    break
            if orig_target is not None and target != orig_target:
                print(f"[adjust] {name[:60]!r}: BLOCK_ENTRY mode — "
                      f"block label at merged[{block_idx}] "
                      f"→ target 0x{target:x} (was 0x{orig_target:x}, "
                      f"saved {orig_target - target}B)")
        return target

    # DEFAULT mode (mode=0): first instruction after [ck_label].
    for k in range(tgt_idx + 1, len(merged)):
        if merged[k][0] is not None:
            return merged[k][0]
    return None


def _clamp_prefetch_region(
    name: str,
    pair_idx: int,
    pc_next: int,
    target: int,
    orig_klength: int,
    direction: str,
    offset_cl: int,
    func_end: int,
) -> tuple[int, int] | None:
    """Compute (koffset, klength) for one prefetch pair with OOB clamping.

    *klength* may be ``NOP_KLENGTH_SENTINEL`` if the prefetch is entirely
    out of bounds.  Returns ``None`` if the pair should be skipped entirely
    (e.g. negative forward koffset).
    """
    target_aligned = target & ~(CACHELINE_SIZE - 1)
    offset_bytes = offset_cl * CACHELINE_SIZE
    klength = orig_klength

    if direction == DIR_BACKWARD:
        region_end = target_aligned + CACHELINE_SIZE + offset_bytes
        region_start = region_end - (klength + 1) * CACHELINE_SIZE

        min_base = (pc_next & ~(CACHELINE_SIZE - 1)) + CACHELINE_SIZE
        if region_start < min_base:
            region_start = min_base
            usable = (region_end - region_start) // CACHELINE_SIZE
            if usable <= 0:
                klength = NOP_KLENGTH_SENTINEL
                print(f"[nop] {name[:60]!r}: backward prefetch fully OOB "
                      f"(min_base 0x{min_base:x} >= region_end 0x{region_end:x}), "
                      f"replacing with 2× s_nop")
            else:
                klength = usable - 1
                print(f"[clamp] {name[:60]!r}: backward start clamped "
                      f"(first cacheline after pc_next: 0x{min_base:x}), "
                      f"klength {orig_klength} → {klength}")

        if klength == NOP_KLENGTH_SENTINEL:
            print(f"[debug]   func={name[:60]!r}  pair {pair_idx}:  "
                  f"pc_next=0x{pc_next:x}  dir=backward  → NOP (0 cachelines in bounds)")
            return (0, NOP_KLENGTH_SENTINEL)

        prefetch_base = region_start
        koffset = prefetch_base - pc_next
        print(f"[debug]   func={name[:60]!r}  pair {pair_idx}:  "
              f"pc_next=0x{pc_next:x}  target=0x{target:x}  dir=backward  "
              f"offset={offset_cl}cl  prefetch_base=0x{prefetch_base:x}  "
              f"koffset=0x{koffset:x} ({koffset}B)  klength={klength}  "
              f"region=[0x{region_start:x}, 0x{region_end:x}) "
              f"({(region_end - region_start)}B = {klength + 1} cachelines)")
        return (koffset, klength)

    # ── Forward direction ────────────────────────────────────────────────
    koffset = target - pc_next + offset_bytes
    if koffset < 0:
        print(f"[warn] {name[:60]!r}: negative koffset — target before prefetch, skipping")
        return None

    prefetch_base = target_aligned + offset_bytes
    region_end = prefetch_base + (klength + 1) * CACHELINE_SIZE
    if region_end > func_end:
        needed = max(0, (func_end - prefetch_base + CACHELINE_SIZE - 1) // CACHELINE_SIZE)
        if needed == 0:
            klength = NOP_KLENGTH_SENTINEL
            print(f"[nop] {name[:60]!r}: forward prefetch fully OOB "
                  f"(prefetch_base 0x{prefetch_base:x} >= func_end 0x{func_end:x}), "
                  f"replacing with 2× s_nop")
        else:
            klength = needed - 1
            region_end = prefetch_base + (klength + 1) * CACHELINE_SIZE
            print(f"[clamp] {name[:60]!r}: forward end clamped "
                  f"(func_end 0x{func_end:x}), klength {orig_klength} → {klength}")

    if klength == NOP_KLENGTH_SENTINEL:
        print(f"[debug]   func={name[:60]!r}  pair {pair_idx}:  "
              f"pc_next=0x{pc_next:x}  dir=forward  → NOP (0 cachelines in bounds)")
    else:
        region_start = prefetch_base
        print(f"[debug]   func={name[:60]!r}  pair {pair_idx}:  "
              f"pc_next=0x{pc_next:x}  target=0x{target:x}  dir=forward  "
              f"offset={offset_cl}cl  "
              f"koffset=0x{koffset:x} ({koffset}B)  klength={klength}  "
              f"region=[0x{region_start:x}, 0x{region_end:x}) "
              f"({(region_end - region_start)}B = {klength + 1} cachelines)")
    return (koffset, klength)


def find_best_koffset_hybrid(merged_funcs: dict[str, list[tuple[int | None, str]]],
                             label: str) -> dict[str, list[tuple[int, int]]]:
    """Compute per-function (koffset, klength) for INST_PREFETCH/INST_PREFETCH_TARGET label pairs.

    Returns {funcname: [(koffset, klength), ...]} for each function containing
    the given label.  klength is clamped so the prefetch does not extend past
    the end of the function; if no cachelines are in bounds, klength is set to
    NOP_KLENGTH_SENTINEL and the prefetch will be replaced with 2× s_nop.
    """
    # [ck_prefetch] marks INST_PREFETCH sites, [ck_label] marks INST_PREFETCH_TARGET targets.
    prefetch_re = re.compile(rf";\s*\[ck_prefetch\].*\bname\s*=\s*{re.escape(label)}\b")
    target_re   = re.compile(rf";\s*\[ck_label\].*\bname\s*=\s*{re.escape(label)}\b")
    either_re   = re.compile(rf";\s*(?:\[ck_label\]|\[ck_prefetch\]).*\bname\s*=\s*{re.escape(label)}\b")
    klength_re  = re.compile(r's_prefetch_inst_pc_rel\s+\S+\s*,\s*\S+\s*,\s*(\d+)')
    mode_re     = re.compile(r'\bmode\s*=\s*(\d+)')
    dir_re      = re.compile(r'\bdir\s*=\s*(\w+)')
    offset_re   = re.compile(r'\boffset\s*=\s*(-?\d+)')

    results: dict[str, list[tuple[int, int]]] = {}

    for name, merged in merged_funcs.items():
        if not any(either_re.search(line) for _, line in merged):
            continue

        # Determine function end address (for OOB clamping).
        func_end: int = 0
        for addr, _ in reversed(merged):
            if addr is not None:
                func_end = addr + 8  # conservative: largest instruction is 8 bytes
                break

        # Classify markers from .s comments.
        prefetch_sites: list[PrefetchSite] = []
        target_sites: list[TargetSite] = []
        for idx, (_addr, line) in enumerate(merged):
            if prefetch_re.search(line):
                m_dir = dir_re.search(line)
                m_off = offset_re.search(line)
                prefetch_sites.append(PrefetchSite(
                    idx=idx,
                    direction=m_dir.group(1) if m_dir else DIR_FORWARD,
                    offset_cl=int(m_off.group(1)) if m_off else 0,
                ))
            elif target_re.search(line):
                m_mode = mode_re.search(line)
                target_sites.append(TargetSite(
                    idx=idx,
                    mode=int(m_mode.group(1)) if m_mode else PLACE_MODE_DEFAULT,
                ))

        pairs: list[tuple[int, int]] = []

        for pf in prefetch_sites:
            # Find the s_prefetch_inst_pc_rel instruction and parse its klength.
            pf_instr_idx: int | None = None
            orig_klength = 3  # default
            j = pf.idx + 1
            while j < len(merged) and 's_prefetch_inst_pc_rel' not in merged[j][1]:
                j += 1
            if j < len(merged):
                pf_instr_idx = j
                m_kl = klength_re.search(merged[j][1])
                if m_kl:
                    orig_klength = int(m_kl.group(1))

            # pc_next = address of the instruction after s_prefetch_inst_pc_rel.
            pc_next: int | None = None
            if pf_instr_idx is not None:
                for k in range(pf_instr_idx + 1, len(merged)):
                    if merged[k][0] is not None:
                        pc_next = merged[k][0]
                        break

            if pc_next is None:
                print(f"[warn] {name[:60]!r}: no pc_next for prefetch at merged[{pf.idx}], skipping")
                continue

            # Find the nearest INST_PREFETCH_TARGET after this INST_PREFETCH.
            tgt: TargetSite | None = None
            for t in target_sites:
                if t.idx > pf.idx:
                    tgt = t
                    break

            if tgt is None:
                # Unpaired prefetch — treat as forward with target=pc_next.
                print(f"[warn] {name[:60]!r}: unpaired prefetch label at merged[{pf.idx}] — "
                      f"using koffset=0, clamping klength")
                result = _clamp_prefetch_region(
                    name, len(pairs), pc_next, pc_next, orig_klength,
                    DIR_FORWARD, 0, func_end)
                if result is not None:
                    pairs.append(result)
                continue

            target = _resolve_target_address(merged, tgt.idx, tgt.mode, name)
            if target is None:
                print(f"[warn] {name[:60]!r}: no target address for label at merged[{tgt.idx}]")
                continue

            result = _clamp_prefetch_region(
                name, len(pairs), pc_next, target, orig_klength,
                pf.direction, pf.offset_cl, func_end)
            if result is not None:
                pairs.append(result)

        if pairs:
            results[name] = pairs

    if not results:
        print(f"[skip] Label '{label}' not found in any matching function block.")
    else:
        total = sum(len(v) for v in results.values())
        print(f"[offsets] {len(results)} function(s), {total} pair(s) with koffset for '{label}'.")
    return results


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_asm_s(asm_file: Path, func_koffsets: dict[str, list[tuple[int, int]]]) -> bool:
    """Patch s_prefetch_inst_pc_rel koffset and klength operands in the .s file.
    Returns True if any change was made."""
    prefetch_re = re.compile(
        r'(s_prefetch_inst_pc_rel\s+)(?:0x[0-9a-fA-F]+|0|-?\d+)'
        r'(\s*,\s*null\s*,\s*)(?:\d+)')

    # Full-line regex to capture indentation and the entire prefetch instruction
    # (used for NOP replacement).
    prefetch_full_re = re.compile(
        r'^(\s*)s_prefetch_inst_pc_rel\s+\S+\s*,\s*\S+\s*,\s*\d+(.*)$')

    text = asm_file.read_text(encoding="utf-8", errors="replace")
    out_lines: list[str] = []
    current_func = "<top>"
    func_pf_idx: dict[str, int] = {}
    n_patched = 0
    n_nopped = 0

    for line in text.splitlines(keepends=True):
        m = FUNC_LABEL_RE.match(line.rstrip())
        if m:
            current_func = m.group(1)
        if current_func in func_koffsets:
            pair_list = func_koffsets[current_func]
            idx = func_pf_idx.get(current_func, 0)
            if idx < len(pair_list):
                koffset, klength = pair_list[idx]
                if klength == NOP_KLENGTH_SENTINEL:
                    # Replace 8-byte prefetch with 2× 4-byte s_nop 0
                    m_full = prefetch_full_re.match(line.rstrip('\n\r'))
                    if m_full:
                        indent = m_full.group(1)
                        trailing = m_full.group(2)  # e.g. comment
                        eol = line[len(line.rstrip('\n\r')):]  # preserve \n
                        nop_lines = (f"{indent}s_nop 0{trailing}{eol}"
                                     f"{indent}s_nop 0{eol}")
                        print(f"[patch-s] {current_func[:60]}: prefetch[{idx}] → "
                              f"2× s_nop 0 (OOB)")
                        func_pf_idx[current_func] = idx + 1
                        n_nopped += 1
                        n_patched += 1
                        out_lines.append(nop_lines)
                        continue
                else:
                    koffset_str = hex(koffset)
                    new_line, n = prefetch_re.subn(
                        rf'\g<1>{koffset_str}\g<2>{klength}', line)
                    if n:
                        print(f"[patch-s] {current_func[:60]}: prefetch[{idx}] → "
                              f"koffset={koffset_str} klength={klength}")
                        func_pf_idx[current_func] = idx + 1
                        n_patched += n
                        out_lines.append(new_line)
                        continue
        out_lines.append(line)

    new_text = ''.join(out_lines)
    print(f"[patch-s] {n_patched} operand(s) patched ({n_nopped} replaced with NOPs).")
    if new_text == text:
        print("[patch-s] No change.")
        return False
    asm_file.write_text(new_text, encoding="utf-8")
    print(f"[patch-s] Written: {asm_file}")
    return True


# ---------------------------------------------------------------------------
# ELF / fatbin helpers
# ---------------------------------------------------------------------------

def _find_elf_text_section(data: bytes | bytearray, base: int = 0) -> tuple[int, int, int] | None:
    """Find .text section in an ELF image starting at data[base:].
    Returns (file_offset_from_base, size, vaddr) or None."""
    import struct as _s
    d = data[base:]
    if len(d) < 64 or d[:4] != b'\x7fELF':
        return None
    ei_class, ei_data = d[4], d[5]
    endian = '<' if ei_data == 1 else '>'
    try:
        if ei_class == 2:
            e_shoff,     = _s.unpack_from(f'{endian}Q', d, 40)
            e_shentsize, = _s.unpack_from(f'{endian}H', d, 58)
            e_shnum,     = _s.unpack_from(f'{endian}H', d, 60)
            e_shstrndx,  = _s.unpack_from(f'{endian}H', d, 62)
            addr_in_shdr, off_in_shdr, sz_in_shdr = 16, 24, 32
            addr_fmt, off_fmt, sz_fmt = f'{endian}Q', f'{endian}Q', f'{endian}Q'
        else:
            e_shoff,     = _s.unpack_from(f'{endian}I', d, 32)
            e_shentsize, = _s.unpack_from(f'{endian}H', d, 46)
            e_shnum,     = _s.unpack_from(f'{endian}H', d, 48)
            e_shstrndx,  = _s.unpack_from(f'{endian}H', d, 50)
            addr_in_shdr, off_in_shdr, sz_in_shdr = 12, 16, 20
            addr_fmt, off_fmt, sz_fmt = f'{endian}I', f'{endian}I', f'{endian}I'

        shstr_sh = e_shoff + e_shstrndx * e_shentsize
        shstr_off, = _s.unpack_from(off_fmt, d, shstr_sh + off_in_shdr)

        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            name_idx, = _s.unpack_from(f'{endian}I', d, sh)
            ns = shstr_off + name_idx
            ne = d.index(b'\x00', ns)
            if d[ns:ne] == b'.text':
                sec_addr, = _s.unpack_from(addr_fmt, d, sh + addr_in_shdr)
                sec_off,  = _s.unpack_from(off_fmt,  d, sh + off_in_shdr)
                sec_sz,   = _s.unpack_from(sz_fmt,   d, sh + sz_in_shdr)
                return (sec_off, sec_sz, sec_addr)
    except (_s.error, ValueError):
        pass
    return None


def _find_gpu_bundle(data: bytes | bytearray, tag: str = "fatbin"
                     ) -> tuple[int, int, int, str] | None:
    """Locate the GPU bundle in a fat .o / fatbin.

    Returns (magic_idx, gpu_off, gpu_sz, gpu_triple) or None.
    *gpu_off* is relative to *magic_idx* (the absolute start of the GPU ELF
    in *data* is ``magic_idx + gpu_off``).
    """
    import struct as _s
    MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
    magic_idx = data.find(MAGIC)
    if magic_idx < 0:
        print(f"[{tag}] __CLANG_OFFLOAD_BUNDLE__ magic not found")
        return None

    hdr = magic_idx + len(MAGIC)
    if hdr + 8 > len(data):
        print(f"[{tag}] Truncated fatbin header")
        return None

    num_bundles, = _s.unpack_from('<Q', data, hdr)
    cur = hdr + 8
    gpu_off = gpu_sz = 0
    gpu_triple = ""

    for _ in range(num_bundles):
        if cur + 24 > len(data):
            break
        off, sz, triple_sz = _s.unpack_from('<QQQ', data, cur)
        cur += 24
        if triple_sz == 0 or triple_sz > 512 or cur + triple_sz > len(data):
            break
        triple = data[cur:cur + triple_sz].decode('utf-8', errors='replace')
        cur += triple_sz
        if 'amdgcn' in triple or (triple.startswith('hip') and 'host' not in triple):
            gpu_off, gpu_sz, gpu_triple = off, sz, triple

    if not gpu_triple:
        print(f"[{tag}] No GPU entry found in fatbin header")
        return None
    return (magic_idx, gpu_off, gpu_sz, gpu_triple)


def _objdump_gpu_elf(data: bytes | bytearray, abs_gpu_start: int, gpu_sz: int,
                     mcpu: str, objdump_path: str, tmp_path: Path,
                     tag: str = "fatbin") -> str | None:
    """Extract the GPU ELF from *data*, run objdump -d, return the text or None."""
    try:
        tmp_path.write_bytes(bytes(data[abs_gpu_start:abs_gpu_start + gpu_sz]))
        result = subprocess.run(
            [objdump_path, f"--mcpu={mcpu}", "-d", str(tmp_path)],
            text=True, capture_output=True,
        )
        if result.returncode != 0:
            print(f"[{tag}] objdump on GPU ELF failed: {result.stderr[:200]}")
            return None
        return result.stdout
    finally:
        tmp_path.unlink(missing_ok=True)


def _patch_one_prefetch(fat_data: bytearray, instr_pos: int, instr_va: int,
                        idx: int, new_koffset: int, new_klength: int) -> None:
    """Patch a single s_prefetch_inst_pc_rel at *instr_pos* in *fat_data*.

    If *new_klength* is NOP_KLENGTH_SENTINEL, replaces the 8-byte instruction
    with 2× s_nop 0.  Otherwise patches koffset (dw1[23:0]) and klength
    (dw0[12:6]) in place.
    """
    import struct as _struct

    old_dw0 = _struct.unpack_from('<I', fat_data, instr_pos)[0]
    old_dw1 = _struct.unpack_from('<I', fat_data, instr_pos + 4)[0]
    raw_before = fat_data[instr_pos:instr_pos + 8]
    print(f"[patch-obj] VA 0x{instr_va:x}: BEFORE  "
          f"dw0=0x{old_dw0:08x} dw1=0x{old_dw1:08x}  "
          f"raw={raw_before.hex()}  "
          f"klength_bits12_6={(old_dw0 >> KLENGTH_SHIFT) & 0x7F}")

    if new_klength == NOP_KLENGTH_SENTINEL:
        _struct.pack_into('<I', fat_data, instr_pos, S_NOP_ENCODING)
        _struct.pack_into('<I', fat_data, instr_pos + 4, S_NOP_ENCODING)
        raw_after = fat_data[instr_pos:instr_pos + 8]
        print(f"[patch-obj] VA 0x{instr_va:x}: AFTER   "
              f"2× s_nop 0 (0x{S_NOP_ENCODING:08x})  "
              f"raw={raw_after.hex()}")
        print(f"[patch-obj] VA 0x{instr_va:x}: prefetch[{idx}] → "
              f"2× s_nop 0 (OOB)")
    else:
        first_dword = _struct.unpack_from('<I', fat_data, instr_pos)[0]
        first_dword = (first_dword & ~KLENGTH_MASK) | ((new_klength & 0x7F) << KLENGTH_SHIFT)
        _struct.pack_into('<I', fat_data, instr_pos, first_dword)
        second_dword = _struct.unpack_from('<I', fat_data, instr_pos + 4)[0]
        second_dword = (second_dword & ~KOFFSET_MASK) | (new_koffset & KOFFSET_MASK)
        _struct.pack_into('<I', fat_data, instr_pos + 4, second_dword)
        raw_after = fat_data[instr_pos:instr_pos + 8]
        new_dw0 = _struct.unpack_from('<I', fat_data, instr_pos)[0]
        new_dw1 = _struct.unpack_from('<I', fat_data, instr_pos + 4)[0]
        print(f"[patch-obj] VA 0x{instr_va:x}: AFTER   "
              f"dw0=0x{new_dw0:08x} dw1=0x{new_dw1:08x}  "
              f"raw={raw_after.hex()}  "
              f"klength_bits12_6={(new_dw0 >> KLENGTH_SHIFT) & 0x7F}")
        print(f"[patch-obj] VA 0x{instr_va:x}: prefetch[{idx}] → "
              f"koffset={hex(new_koffset)} klength={new_klength}")


def replace_gpu_in_fatobj(fat_obj: Path, mcpu: str, objdump_path: str,
                          func_koffsets: dict[str, list[tuple[int, int]]]) -> bool:
    """Patch s_prefetch_inst_pc_rel koffsets and klengths directly in the GPU ELF
    embedded in the fat .o via direct binary patching.  Returns True on success."""

    fat_data = bytearray(fat_obj.read_bytes())
    bundle = _find_gpu_bundle(fat_data, tag="patch-obj")
    if bundle is None:
        return False
    magic_idx, gpu_off, gpu_sz, gpu_triple = bundle
    abs_gpu_start = magic_idx + gpu_off
    print(f"[patch-obj] GPU bundle: '{gpu_triple}'  abs=0x{abs_gpu_start:x}  size={gpu_sz}")

    if fat_data[abs_gpu_start:abs_gpu_start + 4] != b'\x7fELF':
        print("[patch-obj] GPU data does not start with ELF magic")
        return False

    text_info = _find_elf_text_section(fat_data, abs_gpu_start)
    if text_info is None:
        print("[patch-obj] Cannot find .text in GPU ELF")
        return False
    text_foff, text_sz, text_va = text_info
    print(f"[patch-obj] .text: foff=0x{text_foff:x}  size={text_sz}  va=0x{text_va:x}")

    objdump_text = _objdump_gpu_elf(fat_data, abs_gpu_start, gpu_sz, mcpu, objdump_path,
                                     fat_obj.with_suffix(".ck_gpu_elf_tmp"), tag="patch-obj")
    if objdump_text is None:
        return False

    n_patched = 0
    current_func: str | None = None
    func_pf_idx: dict[str, int] = {}

    for line in objdump_text.splitlines():
        m = OBJDUMP_FUNC_RE.match(line)
        if m:
            current_func = m.group(1)
            continue
        if not current_func or current_func not in func_koffsets:
            continue
        if 's_prefetch_inst_pc_rel' not in line:
            continue
        m2 = OBJDUMP_ADDR_RE.search(line)
        if not m2:
            continue
        idx = func_pf_idx.get(current_func, 0)
        pair_list = func_koffsets[current_func]
        if idx >= len(pair_list):
            continue
        instr_va = int(m2.group(1), 16)
        new_koffset, new_klength = pair_list[idx]
        instr_pos = abs_gpu_start + text_foff + (instr_va - text_va)
        _patch_one_prefetch(fat_data, instr_pos, instr_va, idx,
                            new_koffset, new_klength)
        func_pf_idx[current_func] = idx + 1
        n_patched += 1

    if n_patched == 0:
        print("[patch-obj] No s_prefetch_inst_pc_rel found to patch")
        return False

    fat_obj.write_bytes(bytes(fat_data))
    # Sanity check: re-read and verify the write persisted
    import hashlib
    written_hash = hashlib.md5(fat_obj.read_bytes()).hexdigest()
    expected_hash = hashlib.md5(bytes(fat_data)).hexdigest()
    if written_hash != expected_hash:
        print(f"[patch-obj] WARNING: write verification failed! "
              f"expected={expected_hash} written={written_hash}")
    print(f"[patch-obj] Patched {n_patched} instruction(s) in {fat_obj.name}  "
          f"md5={written_hash}")
    return True


def verify_patched_obj(fat_obj: Path, mcpu: str, objdump_path: str,
                       func_koffsets: dict[str, list[tuple[int, int]]]) -> bool:
    """Verify patched s_prefetch_inst_pc_rel koffsets and klengths.  Returns True if all match.

    For NOP-replaced entries (klength == NOP_KLENGTH_SENTINEL), verification
    checks that the raw bytes at the original position are 2× s_nop encoding.
    """
    import struct as _struct

    data = fat_obj.read_bytes()
    bundle = _find_gpu_bundle(data, tag="verify")
    if bundle is None:
        return False
    magic_idx, gpu_off, gpu_sz, _ = bundle
    abs_start = magic_idx + gpu_off

    # Locate .text for raw byte diagnostics
    text_info = _find_elf_text_section(data, abs_start)
    text_foff = text_va = 0
    if text_info:
        text_foff, text_sz, text_va = text_info

    objdump_text = _objdump_gpu_elf(data, abs_start, gpu_sz, mcpu, objdump_path,
                                     fat_obj.with_suffix(".ck_verify_tmp"), tag="verify")
    if objdump_text is None:
        return False

    prefetch_re = re.compile(
        r's_prefetch_inst_pc_rel\s+(0x[0-9a-fA-F]+|\d+)\s*,\s*\S+\s*,\s*(\d+)')
    current_func: str | None = None
    func_pf_idx: dict[str, int] = {}
    # Track VAs already consumed as part of a NOP pair so the second s_nop
    # of a pair (or compiler-emitted s_nops) are not misidentified.
    consumed_nop_vas: set[int] = set()
    ok = True
    checked = 0

    for line in objdump_text.splitlines():
        m = OBJDUMP_FUNC_RE.match(line)
        if m:
            current_func = m.group(1)
            continue
        if current_func and current_func in func_koffsets:
            idx = func_pf_idx.get(current_func, 0)
            pair_list = func_koffsets[current_func]
            if idx >= len(pair_list):
                continue
            exp_koff, exp_klen = pair_list[idx]

            if exp_klen == NOP_KLENGTH_SENTINEL:
                # Expect 2× s_nop at this position.  Match by checking whether
                # the raw bytes at this VA form a NOP pair (both dwords are
                # S_NOP_ENCODING).  This avoids confusion with compiler-emitted
                # s_nop instructions that are not part of our patching.
                if 's_nop' in line:
                    m_addr = OBJDUMP_ADDR_RE.search(line)
                    if m_addr and text_info:
                        va = int(m_addr.group(1), 16)
                        if va in consumed_nop_vas:
                            continue  # second nop of an already-verified pair
                        pos = abs_start + text_foff + (va - text_va)
                        if 0 <= pos and pos + 8 <= len(data):
                            dw0 = _struct.unpack_from('<I', data, pos)[0]
                            dw1 = _struct.unpack_from('<I', data, pos + 4)[0]
                            if dw0 == S_NOP_ENCODING and dw1 == S_NOP_ENCODING:
                                consumed_nop_vas.add(va)
                                consumed_nop_vas.add(va + 4)
                                status = "OK"
                                print(f"[verify] {status}: {current_func[:60]}  prefetch[{idx}]  "
                                      f"→ 2× s_nop  dw0=0x{dw0:08x} dw1=0x{dw1:08x}")
                                func_pf_idx[current_func] = idx + 1
                                checked += 1
            else:
                m2 = prefetch_re.search(line)
                if m2:
                    actual_koff = int(m2.group(1), 0)
                    actual_klen = int(m2.group(2))
                    koff_ok = actual_koff == exp_koff
                    klen_ok = actual_klen == exp_klen
                    status = "OK" if koff_ok else "MISMATCH"
                    if not koff_ok:
                        ok = False
                    extra = ""
                    if not klen_ok:
                        extra = "  (klength mismatch — may need rebuild)"
                    print(f"[verify] {status}: {current_func[:60]}  prefetch[{idx}]  "
                          f"koffset exp={hex(exp_koff)} act={hex(actual_koff)}  "
                          f"klength exp={exp_klen} act={actual_klen}{extra}")
                    # Diagnostic: dump raw bytes from the .o at the instruction position
                    m_addr = OBJDUMP_ADDR_RE.search(line)
                    if m_addr and text_info:
                        instr_va = int(m_addr.group(1), 16)
                        instr_pos = abs_start + text_foff + (instr_va - text_va)
                        if 0 <= instr_pos and instr_pos + 8 <= len(data):
                            dw0 = _struct.unpack_from('<I', data, instr_pos)[0]
                            dw1 = _struct.unpack_from('<I', data, instr_pos + 4)[0]
                            raw8 = data[instr_pos:instr_pos + 8]
                            print(f"[verify]   raw VA 0x{instr_va:x}: "
                                  f"dw0=0x{dw0:08x} dw1=0x{dw1:08x}  "
                                  f"raw={raw8.hex()}  "
                                  f"klength_bits12_6={(dw0 >> KLENGTH_SHIFT) & 0x7F}")
                    func_pf_idx[current_func] = idx + 1
                    checked += 1

    if checked == 0:
        print("[verify] WARNING: no instructions found to verify")
        return False

    print(f"[verify] Checked {checked} instruction(s): {'ALL OK' if ok else 'FAILURES FOUND'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir",  required=True, type=Path, help="CMake build directory")
    ap.add_argument("--target",     required=True,            help="CMake target to build")
    ap.add_argument("--objdump-path", default="/opt/rocm/llvm/bin/llvm-objdump",
                    help="Path to llvm-objdump (default: /opt/rocm/llvm/bin/llvm-objdump)")
    ap.add_argument("--objdump-mcpu", default="",
                    help="--mcpu value for llvm-objdump/llvm-mc (auto-detected from .s if omitted)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Parse and print the koffset but do not write the file or rebuild")
    ap.add_argument("--skip-build-round1", action="store_true",
                    help="Skip the round-1 cmake build (use when called from a CMake POST_BUILD "
                         "command where round 1 was already performed by the normal build)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="Parallel jobs for cmake builds (default: all logical CPUs)")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="Tee all script output to this file (default: "
                         "<build-dir>/prefetch_patch_<target>.log; pass empty string to disable)")
    ap.add_argument("--label", default=None,
                    help="Process only this label name (default: all labels discovered from .s)")
    ap.add_argument("--dump-intermediates", action="store_true",
                    help="Write intermediate files (merged tables, objdump text) to the build dir")
    # Legacy: --source and --bundler-path are accepted but ignored.
    ap.add_argument("--source", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--bundler-path", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Log setup.
    log_path: Path | None
    if args.log_file is None:
        log_path = args.build_dir.resolve() / f"prefetch_patch_{args.target}.log"
    elif str(args.log_file) == "":
        log_path = None
    else:
        log_path = args.log_file

    tee: _Tee | None = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tee = _Tee(log_path)
        sys.stdout = tee  # type: ignore[assignment]
        print(f"[log] Output mirrored to {log_path}")

    build_dir = args.build_dir.resolve()
    jobs = args.jobs if args.jobs is not None else multiprocessing.cpu_count()

    # ── Round 1 ───────────────────────────────────────────────────────────────
    if args.skip_build_round1:
        print("=== Round 1: skipped (--skip-build-round1) ===")
    else:
        print("=== Round 1: building with koffset=0 ===")
        cmake_build(build_dir, args.target, jobs)

    # ── Locate .s and .o ─────────────────────────────────────────────────────
    obj_file = find_obj_file(build_dir, args.target)

    cpp_stem = Path(obj_file.stem).stem
    asm_file = find_asm_file(obj_file.parent.parent.parent, cpp_stem)
    print(f"[asm] Using {asm_file}")
    asm_text = asm_file.read_text(encoding="utf-8", errors="replace")

    # Auto-detect mcpu from .amdgcn_target directive if not provided.
    if not args.objdump_mcpu:
        args.objdump_mcpu = detect_mcpu_from_asm(asm_text)
        if args.objdump_mcpu:
            print(f"[mcpu] Auto-detected from .s: {args.objdump_mcpu}")
        else:
            sys.exit("Could not auto-detect GPU arch from .s file. "
                     "Please pass --objdump-mcpu explicitly (e.g. --objdump-mcpu gfx1201).")

    # ── Discover labels from .s ──────────────────────────────────────────────
    labels = find_prefetch_labels_from_asm(asm_text)
    if not labels:
        print("[skip] No [ck_label]+s_prefetch_inst_pc_rel found in .s — nothing to patch.")
        if tee is not None:
            tee.close()
        return

    if args.label:
        labels = [l for l in labels if l == args.label]
        if not labels:
            print(f"[skip] Label '{args.label}' not found in .s.")
            if tee is not None:
                tee.close()
            return
    print(f"[labels] Discovered from .s: {labels}")

    # ── Assemble + objdump ───────────────────────────────────────────────────
    gpu_obj = assemble_gpu_asm(asm_file, args.objdump_mcpu, args.objdump_path)
    print(f"[hybrid] Assembled GPU .s → {gpu_obj}")
    objdump_text = run_objdump(args.objdump_path, args.objdump_mcpu, gpu_obj)
    gpu_obj.unlink(missing_ok=True)

    if args.dump_intermediates:
        objdump_dump = build_dir / f"prefetch_patch_{args.target}_objdump.txt"
        objdump_dump.write_text(objdump_text, encoding="utf-8")
        print(f"[dump] Raw objdump written to {objdump_dump}")

    # ── Merge .s ↔ objdump (once per function) ──────────────────────────────
    dump_dir = build_dir if args.dump_intermediates else None
    merged_funcs = _merge_all_functions(asm_text, objdump_text, dump_dir=dump_dir)

    # ── Per-label: compute koffsets ──────────────────────────────────────────
    per_label: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for label in labels:
        print(f"\n[label] Processing '{label}'")
        fk = find_best_koffset_hybrid(merged_funcs, label)
        if not fk:
            print(f"[label] '{label}': markers not matched — skipping.")
            continue
        for fname, pairs in fk.items():
            for i, (k, kl) in enumerate(pairs):
                if kl == NOP_KLENGTH_SENTINEL:
                    print(f"[offsets]   {fname[:70]}: prefetch[{i}] → 2× s_nop (OOB)")
                else:
                    print(f"[offsets]   {fname[:70]}: prefetch[{i}] koffset={hex(k)} klength={kl}")
        per_label[label] = fk

    if args.dry_run:
        print("[dry-run] Stopping before Round 2.")
        if tee is not None:
            tee.close()
        return

    if not per_label:
        print("=== Round 2: skipped (no koffsets computed) ===")
        print("=== Done ===")
        if tee is not None:
            tee.close()
        return

    # ── Round 2: patch + rebuild ─────────────────────────────────────────────
    all_func_koffsets: dict[str, list[tuple[int, int]]] = {}
    for fk in per_label.values():
        for fname, pairs in fk.items():
            if fname in all_func_koffsets:
                all_func_koffsets[fname].extend(pairs)
            else:
                all_func_koffsets[fname] = list(pairs)

    print("=== Round 2: patching .s, replacing GPU in fat .o, relinking ===")

    # Save original mtimes so we can restore them after patching.
    # This prevents the patched files from appearing "newer" than the source,
    # which would cause CMake to skip recompilation on the next build.
    # Also save the .o.d dependency file mtime — if it drifts relative to
    # compiler_depend.internal, CMake re-evaluates dependencies unnecessarily.
    import os as _os
    obj_stat = _os.stat(obj_file)
    asm_stat = _os.stat(asm_file)
    obj_dep_file = obj_file.with_suffix(obj_file.suffix + ".d")
    obj_dep_stat = _os.stat(obj_dep_file) if obj_dep_file.exists() else None

    patch_asm_s(asm_file, all_func_koffsets)

    if not replace_gpu_in_fatobj(obj_file, args.objdump_mcpu, args.objdump_path,
                                  all_func_koffsets):
        sys.exit("[error] Direct binary patching of fat .o failed.")

    # Restore original mtimes so build-system dependency tracking is not disrupted.
    _os.utime(obj_file, (obj_stat.st_atime, obj_stat.st_mtime))
    _os.utime(asm_file, (asm_stat.st_atime, asm_stat.st_mtime))
    if obj_dep_stat is not None and obj_dep_file.exists():
        _os.utime(obj_dep_file, (obj_dep_stat.st_atime, obj_dep_stat.st_mtime))

    print("[patch-obj] GPU code replaced via direct binary patching.")

    # When invoked from PRE_LINK (--skip-build-round1), do NOT call cmake_build
    # again — the linker is about to run and will consume the patched .o.
    # PRE_LINK fires after compilation but before linking, which is exactly
    # what we need: the .o is patched in-place, then the linker embeds the
    # patched device ISA into the final executable.
    if args.skip_build_round1:
        print("=== Verifying patched object ===")
        if not verify_patched_obj(obj_file, args.objdump_mcpu, args.objdump_path,
                                   all_func_koffsets):
            print("[verify] WARNING: verification failed — koffsets may be stale")
        print("=== Done (PRE_LINK mode) ===")
    else:
        # Standalone mode: we need to relink so the final executable picks up
        # the patched .o.
        import hashlib as _hashlib
        obj_hash_before = _hashlib.md5(obj_file.read_bytes()).hexdigest()
        print(f"[diag] .o md5 BEFORE cmake_build: {obj_hash_before}  ({obj_file.name})")
        cmake_build(build_dir, args.target, jobs)
        obj_hash_after = _hashlib.md5(obj_file.read_bytes()).hexdigest()
        print(f"[diag] .o md5 AFTER  cmake_build: {obj_hash_after}  ({obj_file.name})")
        if obj_hash_before != obj_hash_after:
            print("[diag] WARNING: .o was RECOMPILED by cmake_build — binary patches lost!")
        else:
            print("[diag] .o unchanged — binary patches preserved.")
        print("=== Done ===")

    if tee is not None:
        tee.close()


if __name__ == "__main__":
    main()
