#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the batched GEMM bridge.

Batched GEMM reuses the plain-GEMM config/codegen machinery but pins the
variant to ``"batched"`` so ``BatchedGemmKernelConfig.name`` gains a trailing
``_batched`` token. That token is the byte-parity invariant tying config ->
codegen -> the compiled kernel name the runtime reports, so distinct batched
kernels never collapse onto the plain-GEMM name.

Everything under test is pure host-side logic (name generation, the codegen
JSON projection, the batched problem flop count, and the shipped
configs/*.json). No GPU, hipcc, or dispatcher build is required.

Run: python3 -m pytest tests/test_batched_bridge.py -v
"""

import ctypes
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import batched_gemm_utils  # noqa: E402
from batched_gemm_utils import (  # noqa: E402
    BatchedGemmDispatcherLib,
    BatchedGemmKernelConfig,
    BatchedGemmProblem,
    _C_SIZEOF,
    _get_arch,
    _repeat_ok,
    _resolve_arch,
    expand_sweep,
)
from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    _output_dtype,
    _dtype_from_kernel_name,
)

_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "batched_gemm"
    / "configs"
)


def _make_config(**overrides):
    kw = dict(
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_c="fp16",
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
    )
    kw.update(overrides)
    return BatchedGemmKernelConfig(**kw)


class TestBatchedName(unittest.TestCase):
    """The batched kernel-name contract (the byte-parity invariant)."""

    def test_variant_is_batched(self):
        self.assertEqual(_make_config().variant, "batched")

    def test_name_gains_batched_suffix(self):
        cfg = _make_config()
        self.assertTrue(cfg.name.endswith("_batched"), cfg.name)

    def test_name_is_plain_gemm_plus_suffix(self):
        # The batched name must be exactly the plain-GEMM name + "_batched",
        # so the two bridges share codegen but never collide.
        common = dict(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32",
            layout_a="row", layout_b="row", layout_c="row",
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        plain = GemmKernelConfig(**common)
        batched = BatchedGemmKernelConfig(**common)
        self.assertEqual(batched.name, plain.name + "_batched")

    def test_suffix_not_doubled(self):
        # Re-deriving the name must not append a second "_batched".
        cfg = _make_config()
        self.assertEqual(cfg.name.count("_batched"), 1)

    def test_dtype_recovers_from_name(self):
        for dtype in ("fp16", "bf16"):
            cfg = _make_config(
                dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
            )
            self.assertEqual(_dtype_from_kernel_name(cfg.name), dtype)


class TestBatchedCodegenJson(unittest.TestCase):
    """The codegen JSON projection is inherited from the plain-GEMM config."""

    def test_codegen_json_core_blocks(self):
        j = _make_config().to_codegen_json()
        self.assertIn("tile_config", j)
        self.assertIn("trait_config", j)
        # dispatcher wave_* is projected onto codegen warp_* (warps per block).
        self.assertEqual(j["tile_config"]["warp_m"], [2])


class TestBatchedProblem(unittest.TestCase):
    """The batched problem carries the batch dimension into the flop count."""

    def test_flops_scale_with_batch(self):
        one = BatchedGemmProblem(batch_count=1, M=128, N=128, K=64)
        eight = BatchedGemmProblem(batch_count=8, M=128, N=128, K=64)
        self.assertEqual(eight.flops, 8 * one.flops)

    def test_flops_formula(self):
        p = BatchedGemmProblem(batch_count=4, M=32, N=16, K=8)
        self.assertEqual(p.flops, 2.0 * 4 * 32 * 16 * 8)

    def test_dict_roundtrip(self):
        p = BatchedGemmProblem(batch_count=3, M=64, N=48, K=16)
        back = BatchedGemmProblem.from_dict(p.to_dict())
        self.assertEqual(back.batch_count, 3)
        self.assertEqual((back.M, back.N, back.K), (64, 48, 16))


class TestBatchedShippedConfigs(unittest.TestCase):
    """The configs/*.json the bridge codegens from must be well-formed sweeps."""

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), _CONFIG_DIR)

    def test_configs_are_valid_sweeps(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no batched configs shipped")
        for path in configs:
            with self.subTest(config=path.name):
                with open(path) as f:
                    data = json.load(f)
                self.assertIn("tile_config", data)
                self.assertIn("trait_config", data)
                tc = data["tile_config"]
                for key in (
                    "tile_m", "tile_n", "tile_k",
                    "warp_m", "warp_n", "warp_k",
                    "warp_tile_m", "warp_tile_n", "warp_tile_k",
                ):
                    self.assertIn(key, tc, f"{path.name} missing {key}")
                tr = data["trait_config"]
                for key in ("pipeline", "scheduler", "epilogue"):
                    self.assertIn(key, tr, f"{path.name} missing {key}")


class TestBatchedArchResolution(unittest.TestCase):
    """`_get_arch` / `_resolve_arch`: detect + validate, never a silent default.

    `_get_arch` shells out to `rocminfo`; these tests mock
    `subprocess.check_output` so they run GPU-free, covering the three outcomes
    the arch feedback demanded: valid detect, undetectable -> RuntimeError,
    unsupported -> ValueError.
    """

    _ROCMINFO = "Agent 1\n  Name:      AMD EPYC\nAgent 2\n  Name:      gfx942\n"

    def test_get_arch_detects_supported(self):
        with mock.patch("subprocess.check_output", return_value=self._ROCMINFO):
            self.assertEqual(_get_arch(), "gfx942")

    def test_get_arch_undetectable_raises_runtimeerror(self):
        # rocminfo missing / no gfx line -> refuse to guess.
        with mock.patch("subprocess.check_output", side_effect=FileNotFoundError()):
            with self.assertRaises(RuntimeError):
                _get_arch()

    def test_get_arch_unsupported_raises_valueerror(self):
        out = "Agent 1\n  Name:      gfx1030\n"
        with mock.patch("subprocess.check_output", return_value=out):
            with self.assertRaises(ValueError):
                _get_arch()

    def test_resolve_arch_explicit_valid_passthrough(self):
        # An explicit, supported arch must not touch rocminfo at all.
        with mock.patch("subprocess.check_output", side_effect=AssertionError("called")):
            self.assertEqual(_resolve_arch("gfx950"), "gfx950")

    def test_resolve_arch_explicit_invalid_raises(self):
        with self.assertRaises(ValueError):
            _resolve_arch("gfx942x")

    def test_resolve_arch_none_autodetects(self):
        with mock.patch("subprocess.check_output", return_value=self._ROCMINFO):
            self.assertEqual(_resolve_arch(None), "gfx942")


class TestBatchedAbiMarshalling(unittest.TestCase):
    """GPU-free tests of the C-ABI surface in ``BatchedGemmDispatcherLib``.

    The ``.so`` is mocked (``ctypes.CDLL`` patched), so no generated kernel or
    GPU is needed. Locks the declared ABI (19 args + time_ms) and the positional
    order/values ``run()`` forwards -- the surface most likely to drift silently
    against the C entry point ``dispatcher_run_batched``.
    """

    def _make_lib(self, status=0):
        fake = mock.MagicMock()
        fake.dispatcher_run_batched.return_value = status
        with mock.patch("ctypes.CDLL", return_value=fake):
            lib = BatchedGemmDispatcherLib(Path("/nonexistent/batched.so"))
        return lib, fake

    def test_run_abi_argtypes(self):
        _, fake = self._make_lib()
        argt = fake.dispatcher_run_batched.argtypes
        # 18 scalar/pointer args + the time_ms out-pointer.
        self.assertEqual(len(argt), 19)
        for i in range(3):  # A/B/C host pointers
            self.assertEqual(argt[i], ctypes.c_void_p)
        for i in range(3, 18):  # M..rotating_count are all int64
            self.assertEqual(argt[i], ctypes.c_int64)
        self.assertEqual(argt[-1], ctypes.POINTER(ctypes.c_float))
        self.assertEqual(fake.dispatcher_run_batched.restype, ctypes.c_int)

    def test_run_marshals_positional_order(self):
        lib, fake = self._make_lib(status=0)
        A = np.zeros((2, 2, 2), np.float16)
        B = np.zeros((2, 2, 2), np.float16)
        C = np.zeros((2, 2, 2), np.float16)
        status, _ = lib.run(
            A, B, C, M=2, N=2, K=2, batch_count=2, k_batch=1,
            stride_A=2, stride_B=2, stride_C=2,
            batch_stride_A=4, batch_stride_B=4, batch_stride_C=4,
            warmup=50, repeat=100, flush_cache=True, rotating_count=1000,
        )
        self.assertEqual(status, 0)
        args = fake.dispatcher_run_batched.call_args[0]
        self.assertEqual(len(args), 19)
        # M/N/K/batch_count/k_batch occupy positions 3..7.
        self.assertEqual((args[3], args[4], args[5]), (2, 2, 2))
        self.assertEqual((args[6], args[7]), (2, 1))
        # flush_cache marshals to the int 1 (position 16), not the bool True.
        self.assertEqual(args[16], 1)

    def test_run_forwards_nonzero_status(self):
        # A thin shim must surface the C error code verbatim (e.g. -2 launch).
        lib, fake = self._make_lib(status=-2)
        A = np.zeros((1, 1, 1), np.float16)
        status, _ = lib.run(A, A, A, M=1, N=1, K=1, batch_count=1)
        self.assertEqual(status, -2)


class TestBatchedDtypeLayoutGate(unittest.TestCase):
    """`expand_sweep` must reject anything outside Old-TE's fp16/rcr set.

    Old-TE ``batched_gemm_instance_builder`` declares ``--datatype
    choices=['fp16']`` / ``--layout choices=['rcr']``; the bridge must match that
    EXACTLY rather than silently building a signature Old-TE never validated. An
    explicit (supported) arch is passed so the gate fires before rocminfo/config
    I/O is touched.
    """

    def test_rejects_non_fp16_dtype(self):
        with self.assertRaises(ValueError):
            expand_sweep("/nonexistent/config.json", arch="gfx942", dtype="bf16")

    def test_rejects_non_rcr_layout(self):
        with self.assertRaises(ValueError):
            expand_sweep("/nonexistent/config.json", arch="gfx942", layout="rrr")


class TestBatchedSplitKContract(unittest.TestCase):
    """F2: lock the generated split-K reduction contract.

    For ``k_batch > 1`` the ck_tile batched kernel switches its epilogue to
    ``memory_operation_enum::atomic_add``, so C MUST be zeroed before the launch
    (a stale C would be accumulated into and silently corrupt the result). For
    ``k_batch == 1`` the epilogue is ``set`` and C must NOT be pre-zeroed (that
    is the Old-TE byte-identical parity path). This test renders the batched
    launch via the codegen and asserts both halves of the contract, so a future
    codegen edit that drops the memset (or the return check) fails loudly here
    instead of only on a GPU under split-K.
    """

    @classmethod
    def setUpClass(cls):
        from codegen_common import TileConfig
        from unified_gemm_codegen import (
            CKTileKernelGenerator,
            GemmVariant,
            KernelConfig,
            TraitConfig,
        )

        tile = TileConfig(
            tile_m=128, tile_n=128, tile_k=32,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        )
        trait = TraitConfig(
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            pad_m=False, pad_n=False, pad_k=False, persistent=False,
        )
        cfg = KernelConfig(tile=tile, trait=trait, variant=GemmVariant.BATCHED)
        cls.src = CKTileKernelGenerator("fp16", "rcr").generate(cfg)

    def test_split_k_branch_zeroes_c(self):
        # The k_batch>1 branch exists and zeroes C via hipMemsetAsync.
        self.assertIn("if(args.k_batch > 1)", self.src)
        self.assertIn("hipMemsetAsync(args.e_ptr", self.src)
        # The reason is the atomic-add reduction; the contract is documented.
        self.assertIn("atomic_add", self.src)

    def test_split_k_memset_return_is_checked(self):
        # The Copilot follow-up: a failed reset must throw, not run silently.
        self.assertIn("!= hipSuccess", self.src)
        self.assertIn("failed to reset C", self.src)

    def test_k_batch_one_uses_set_no_memset(self):
        # The k_batch==1 parity path uses memory_operation_enum::set and must not
        # pre-zero C (byte-identical to the Old-TE batched launch).
        self.assertIn("memory_operation_enum::set", self.src)
        # Exactly one memset in the batched launch: the split-K path only.
        self.assertEqual(self.src.count("hipMemsetAsync(args.e_ptr"), 1)


class TestBatchedCDtypeSizeMap(unittest.TestCase):
    """F6: the host numpy C dtype size must equal sizeof(CDataType)."""

    def test_c_sizeof_matches_numpy_itemsize(self):
        _C_NP = {"fp16": np.float16, "bf16": np.uint16, "int32": np.int32}
        for out_dtype, nbytes in _C_SIZEOF.items():
            self.assertIn(out_dtype, _C_NP)
            self.assertEqual(
                np.dtype(_C_NP[out_dtype]).itemsize, nbytes,
                f"{out_dtype}: numpy itemsize != declared sizeof(CDataType)",
            )


class TestBatchedRepeatGate(unittest.TestCase):
    """Old-TE IsSupportedArgument parity: reject the odd-per-wave-repeat /
    32-wide-warp-tile signature (issue #9684).

    The batched default_config sweeps tile dims 64/128/192/256 across BOTH the
    cshuffle and default epilogues. A 192 tile with wave=2 / warp_tile=32 gives
    an odd MRepeat = 192/(2*32) = 3, which the ck_tile batched kernel mis-stores
    and returns garbage. Old-TE's batched kernel refuses these at launch
    ("Arguments not supported"); gemm_utils only gates them for the cshuffle
    epilogue, so the batched bridge re-applies the identical rule here regardless
    of epilogue. The gate must drop the 192/wt32 case while leaving 128/256 tiles
    -- and the valid 192 geometries (even repeat, or a 16-wide warp tile) --
    untouched, so it must not over-prune.
    """

    def test_rejects_192_odd_repeat_wt32_in_m(self):
        # tile_m=192 / wave_m=2 / warp_tile_m=32 => MRepeat=3 (odd, >1) + wt32.
        self.assertFalse(
            _repeat_ok(192, 128, 2, 2, 32, 32),
            "192/wave2/wt32 (MRepeat=3) must be rejected (Old-TE reject set)",
        )

    def test_rejects_192_odd_repeat_wt32_in_n(self):
        self.assertFalse(
            _repeat_ok(128, 192, 2, 2, 32, 32),
            "N-dim 192/wave2/wt32 (NRepeat=3) must be rejected",
        )

    def test_accepts_128_tile(self):
        # tile=128 / wave=2 / wt=32 => repeat=2 (even) -- valid on both dims.
        self.assertTrue(_repeat_ok(128, 128, 2, 2, 32, 32))

    def test_accepts_256_tile(self):
        # tile=256 / wave=2 / wt=32 => repeat=4 (even) -- valid.
        self.assertTrue(_repeat_ok(256, 256, 2, 2, 32, 32))

    def test_accepts_192_with_16_wide_warp_tile(self):
        # 192 / wave=4 / wt=16 => repeat=3 (odd) but wt!=32 -> GPU-verified OK.
        self.assertTrue(
            _repeat_ok(192, 192, 4, 4, 16, 16),
            "odd repeat with a 16-wide warp tile is valid; must not be pruned",
        )

    def test_accepts_192_even_repeat(self):
        # 192 / wave=1 / wt=32 => repeat=6 (even) -- valid.
        self.assertTrue(_repeat_ok(192, 192, 1, 1, 32, 32))

    def test_uneven_split_is_not_flagged_here(self):
        # tile % (wave*wt) != 0 is dropped by the upstream tile/CShuffle gate;
        # this predicate treats it as "not this bug" (returns True).
        self.assertTrue(_repeat_ok(192, 192, 4, 4, 32, 32))


# --- gfx1250 (MI400 / RDNA4-WMMA) enablement -------------------------------
# The batched-GEMM bridge historically allow-listed only CDNA (gfx90a/942/950,
# MFMA). gfx1250 uses WMMA, so it needs an arch-tuple entry and a WMMA CI config
# (warp_tile 16x16x32 -- the CDNA MFMA 32x32x16 tile does not run on gfx1250).
# These CPU-only tests lock that surface in.
from batched_gemm_utils import _SUPPORTED_ARCHES  # noqa: E402


class TestGfx1250Enablement(unittest.TestCase):
    def test_gfx1250_in_supported_arches(self):
        self.assertIn("gfx1250", _SUPPORTED_ARCHES)

    def test_resolve_arch_accepts_gfx1250(self):
        # An explicit, supported arch must not touch rocminfo at all.
        with mock.patch("subprocess.check_output", side_effect=AssertionError("called")):
            self.assertEqual(_resolve_arch("gfx1250"), "gfx1250")

    def test_get_arch_detects_gfx1250(self):
        with mock.patch("subprocess.check_output", return_value="  Name:  gfx1250\n"):
            self.assertEqual(_get_arch(), "gfx1250")

    def test_gfx1250_ci_config_present_and_wmma(self):
        cfg_path = _CONFIG_DIR / "default_ci_config_gfx1250.json"
        self.assertTrue(cfg_path.is_file(), cfg_path)
        tc = json.loads(cfg_path.read_text())["tile_config"]
        # WMMA warp tile for fp16/bf16 on gfx1250 is 16x16x32.
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32])


if __name__ == "__main__":
    unittest.main()
