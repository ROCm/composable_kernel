# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only regression checks: python3 -m unittest discover -s . -p test_generate.py."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "layernorm_generate", Path(__file__).with_name("generate.py")
)
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)


class InstanceFilterTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(disable_16b_welford=False)
        self.all_blobs = generate.layernorm_fwd_codegen(".", None).get_blobs(self.args)

    def test_empty_filter_preserves_all_instances(self):
        self.assertTrue(self.all_blobs)
        for pattern in ("", "*"):
            with self.subTest(pattern=pattern):
                codegen = generate.layernorm_fwd_codegen(".", pattern)
                self.assertEqual(codegen.get_blobs(self.args), self.all_blobs)

    def test_exact_filter_keeps_original_instance_content(self):
        name = "layernorm2d_fwd_fp16_n256"
        selected = generate.layernorm_fwd_codegen(".", name).get_blobs(self.args)
        self.assertEqual([blob.name for blob in selected], [name])
        original = next(blob for blob in self.all_blobs if blob.name == name)
        self.assertEqual(selected[0].content, original.content)

    def test_wildcard_selects_multiple_instances(self):
        blobs = generate.layernorm_fwd_codegen(
            ".", "layernorm2d_fwd_fp16_n*"
        ).get_blobs(self.args)
        self.assertGreater(len(blobs), 1)
        self.assertLess(len(blobs), len(self.all_blobs))
        self.assertTrue(
            all(blob.name.startswith("layernorm2d_fwd_fp16_n") for blob in blobs)
        )

    def test_unmatched_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matched no kernels"):
            generate.layernorm_fwd_codegen(".", "nonexistent").get_blobs(self.args)

    def test_big_only_filter_keeps_unbounded_dispatch(self):
        for pattern in (
            "layernorm2d_fwd_fp16_nbig",
            "layernorm2d_fwd_bf16_nbig",
            "*_nbig*",
        ):
            with self.subTest(pattern=pattern):
                codegen = generate.layernorm_fwd_codegen(".", pattern)
                api = codegen.content_api(self.args)
                self.assertNotRegex(api, r"\bif\s*\{")
                self.assertNotIn("a.n <=", api)
                for blob in codegen.get_blobs(self.args):
                    for instance in blob.instance_list:
                        self.assertIn(instance.call_name, api)

    def test_list_generation_and_dispatch_select_same_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            codegen = generate.layernorm_fwd_codegen(
                directory, "layernorm2d_fwd_fp16_n256"
            )
            codegen.list_blobs(self.args)
            codegen.gen_blobs(self.args)
            listed = (
                Path(directory, "layernorm2d_fwd_blobs.txt").read_text().splitlines()
            )
            self.assertEqual(len(listed), 3)
            self.assertEqual(
                set(listed),
                {
                    str(p)
                    for p in Path(directory).iterdir()
                    if p.suffix in (".cpp", ".hpp")
                },
            )
            api = Path(directory, "layernorm2d_fwd_api.cpp").read_text()
            selected = codegen.get_blobs(self.args)[0]
            for instance in selected.instance_list:
                self.assertIn(instance.call_name, api)
            excluded = generate.layernorm_fwd_codegen(
                ".", "layernorm2d_fwd_bf16_n256"
            ).get_blobs(self.args)[0]
            for instance in excluded.instance_list:
                self.assertNotIn(instance.call_name, api)


class CMakeInstanceFilterTests(unittest.TestCase):
    def setUp(self):
        if not shutil.which("cmake"):
            self.skipTest("cmake is required")
        directory = tempfile.TemporaryDirectory(prefix="ck-instance-filters-")
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)

    def run_command(self, command, succeeds=True):
        result = subprocess.run(command, text=True, capture_output=True)
        output = result.stdout + result.stderr
        if succeeds:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        return output

    def cmake_fixture(self, example, target):
        source = self.work / "fixture"
        source.mkdir()
        for path in example.glob("*.cpp"):
            shutil.copyfile(path, source / path.name)
        # Include the real example CMake in a small project. Only generation and
        # source selection are exercised, so no HIP compiler or GPU is needed.
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(instance_filters LANGUAGES CXX)\n"
            f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
            "set(GPU_TARGETS gfx942)\n"
            f'include("{example.as_posix()}/CMakeLists.txt")\n'
            f"get_target_property(sources {target} SOURCES)\n"
            'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sources.txt" '
            'CONTENT "$<JOIN:${sources},\\n>\\n")\n'
            "if(LAYERNORM2D_FWD_GEN_BLOBS)\n"
            "  add_custom_target(generate_instances DEPENDS ${LAYERNORM2D_FWD_GEN_BLOBS})\n"
            "endif()\n"
        )
        return source, self.work / "build"

    def configure(self, source, build, variable, pattern, succeeds=True):
        return self.run_command(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-D{variable}={pattern}",
            ],
            succeeds,
        )

    def test_gemm_cmake_source_selection_and_reset(self):
        example = Path(__file__).resolve().parent.parent / "38_block_scale_gemm"
        source, build = self.cmake_fixture(example, "tile_example_gemm_quant")
        variable = "GEMM_QUANT_INSTANCE_FILTER"
        self.configure(source, build, variable, "")
        default_sources = (build / "sources.txt").read_text().splitlines()
        self.assertGreater(len(default_sources), 3)
        self.configure(source, build, variable, "^gemm_bquant_quantgrouped_fp8$")
        self.assertEqual(
            set((build / "sources.txt").read_text().splitlines()),
            {"gemm_quant.cpp", "gemm_bquant_quantgrouped_fp8.cpp"},
        )
        self.configure(source, build, variable, "^gemm_bquant_quantgrouped_(fp8|bf8)$")
        self.assertEqual(
            set((build / "sources.txt").read_text().splitlines()),
            {
                "gemm_quant.cpp",
                "gemm_bquant_quantgrouped_fp8.cpp",
                "gemm_bquant_quantgrouped_bf8.cpp",
            },
        )
        self.configure(source, build, variable, "")
        self.assertEqual(
            (build / "sources.txt").read_text().splitlines(), default_sources
        )
        output = self.configure(
            source, build, variable, "no_such_source", succeeds=False
        )
        self.assertIn("matched no instance sources", output)

    def test_layernorm_cmake_regenerates_dispatcher_when_reusing_filter(self):
        example = Path(__file__).resolve().parent
        source, build = self.cmake_fixture(example, "tile_example_layernorm2d_fwd")
        for size in (256, 512, 256):
            self.configure(
                source,
                build,
                "LAYERNORM2D_FWD_INSTANCE_FILTER",
                f"layernorm2d_fwd_fp16_n{size}",
            )
            self.run_command(
                ["cmake", "--build", str(build), "--target", "generate_instances"]
            )
            dispatcher = (build / "layernorm2d_fwd_api.cpp").read_text()
            self.assertIn(f"a.n <= {size}", dispatcher)
            self.assertNotIn(f"a.n <= {768 - size}", dispatcher)


if __name__ == "__main__":
    unittest.main()
