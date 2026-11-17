#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Pin that the committed arch_specs_generated.py / arch_specs_generated.hpp are
exactly what generate_arch_specs.py emits from arch_specs.json.

A hand edit to either generated file is silently dropped the next time anyone
regenerates, so every rule must live in arch_specs.json or in the generator.
Only the "Generated at:" timestamp line is allowed to differ.

Run: python3 -m pytest tests/test_arch_specs_regeneration.py -v
"""

import shutil
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CODEGEN_DIR = DISPATCHER_DIR / "codegen"
COMMITTED_PY = CODEGEN_DIR / "arch_specs_generated.py"
COMMITTED_HPP = (
    DISPATCHER_DIR / "include" / "ck_tile" / "dispatcher" / "arch_specs_generated.hpp"
)


def _strip_timestamp(text: str) -> list:
    return [line for line in text.splitlines() if "Generated at:" not in line]


class TestArchSpecsRegeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="arch_specs_regen_")
        out = Path(cls._tmp.name)
        subprocess.run(
            [
                sys.executable,
                str(CODEGEN_DIR / "generate_arch_specs.py"),
                "--json",
                str(CODEGEN_DIR / "arch_specs.json"),
                "--output-dir",
                str(out),
                "--cpp-output-dir",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        cls.regen_py = out / "arch_specs_generated.py"
        cls.regen_hpp = out / "arch_specs_generated.hpp"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _assert_same(self, regenerated: Path, committed: Path):
        got = _strip_timestamp(regenerated.read_text())
        want = _strip_timestamp(committed.read_text())
        self.assertEqual(
            got,
            want,
            f"{committed.name} does not match what generate_arch_specs.py emits "
            f"from arch_specs.json. Edit arch_specs.json or the generator and "
            f"regenerate instead of editing the generated file.",
        )

    def test_python_module_matches_committed(self):
        self._assert_same(self.regen_py, COMMITTED_PY)

    def test_cpp_header_matches_committed(self):
        if not (shutil.which("clang-format-18") or shutil.which("clang-format")):
            self.skipTest("clang-format not available; header is left unformatted")
        self._assert_same(self.regen_hpp, COMMITTED_HPP)

    def test_cpp_trait_and_key_parity(self):
        sys.path.insert(0, str(CODEGEN_DIR))
        from codegen_common import CommonTypeMappings
        from generate_arch_specs import generate_cpp_header
        from unified_gemm_codegen import DOUBLE_SMEM_BUFFER_PIPELINES

        specs = json.loads((CODEGEN_DIR / "arch_specs.json").read_text())
        # Exercise the generator's input, not just today's hardcoded combinations.
        combos = specs["unsupported_trait_combos"]["combinations"]
        combos.remove(["compv3", "cshuffle", "interwave"])
        combos.append(["mem", "default", "intrawave"])
        pipelines = sorted(
            set(CommonTypeMappings.PIPELINE_TO_DISPATCHER)
            | {pipeline for pipeline, _, _ in combos}
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            header_dir = out / "ck_tile" / "dispatcher"
            header_dir.mkdir(parents=True)
            generate_cpp_header(specs, header_dir / COMMITTED_HPP.name)
            checks = []
            for pipeline, epilogue, scheduler in itertools.product(
                pipelines,
                ("cshuffle", "default", "tdm"),
                ("intrawave", "interwave"),
            ):
                expected = str([pipeline, epilogue, scheduler] in combos).lower()
                checks.append(
                    f'CHECK(arch_specs::is_trait_unsupported(string_to_pipeline("{pipeline}"),'
                    f' string_to_epilogue("{epilogue}"), string_to_scheduler("{scheduler}"))'
                    f" == {expected});"
                )
            for pipeline in CommonTypeMappings.PIPELINE_TO_DISPATCHER:
                expected = str(pipeline in DOUBLE_SMEM_BUFFER_PIPELINES).lower()
                checks.append(
                    f'CHECK(KernelConfig().pipeline(string_to_pipeline("{pipeline}"))'
                    f".build_key().algorithm.double_buffer == {expected});"
                )
            checks.extend(
                [
                    "const auto shared_1250 = arch_specs::get_supported_warp_configs(GpuArch::GFX_1250);",
                    "CHECK(std::find(shared_1250.begin(), shared_1250.end(), WarpConfig{4, 4, 1}) == shared_1250.end());",
                    "const auto v3_1250 = arch_specs::get_supported_warp_configs(GpuArch::GFX_1250, Pipeline::CompTDMV3);",
                    "CHECK(std::find(v3_1250.begin(), v3_1250.end(), WarpConfig{4, 4, 1}) != v3_1250.end());",
                    "const auto v1_1250 = arch_specs::get_supported_warp_configs(GpuArch::GFX_1250, Pipeline::CompTDMV1);",
                    "CHECK(std::find(v1_1250.begin(), v1_1250.end(), WarpConfig{4, 4, 1}) == v1_1250.end());",
                    "const auto v3_942 = arch_specs::get_supported_warp_configs(GpuArch::GFX_942, Pipeline::CompTDMV3);",
                    "CHECK(std::find(v3_942.begin(), v3_942.end(), WarpConfig{4, 4, 1}) == v3_942.end());",
                    'const ArchFilter filt_1250("gfx1250");',
                    'auto v3_cfg = KernelConfig::bf16_rcr().arch("gfx1250").tile(256, 256, 64).wave(4, 4, 1).warp_tile(16, 16, 32).pipeline(Pipeline::CompTDMV3).epilogue(Epilogue::CShuffle).scheduler(Scheduler::Intrawave).block(512);',
                    "CHECK(filt_1250.is_valid(v3_cfg.build_key()));",
                    "CHECK(filt_1250.is_valid(v3_cfg.tile(256, 256, 128).build_key()));",
                    "CHECK(!filt_1250.is_valid(v3_cfg.pipeline(Pipeline::CompTDMV1).epilogue(Epilogue::Tdm).build_key()));",
                ]
            )
            source = out / "probe.cpp"
            source.write_text(
                '#include "ck_tile/dispatcher/arch_specs_generated.hpp"\n'
                '#include "ck_tile/dispatcher/arch_filter.hpp"\n'
                '#include "ck_tile/dispatcher/kernel_config.hpp"\n'
                "#include <algorithm>\n"
                "#include <cstdio>\n"
                "using namespace ck_tile::dispatcher;\n"
                "#define CHECK(c) if(!(c)) { std::puts(#c); return 1; }\n"
                "int main() {\n" + "\n".join(checks) + "\n}\n"
            )
            compiler = shutil.which("g++") or shutil.which("clang++")
            if not compiler:
                self.skipTest("no host C++ compiler available")
            executable = out / "probe"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-I",
                    str(out),
                    "-I",
                    str(DISPATCHER_DIR / "include"),
                    str(source),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
