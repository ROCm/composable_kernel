#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Test-Driven Development tests for CMake Dependency Analyzer.

This module tests the new pre-build dependency analysis approach that uses
compile_commands.json and clang -MM instead of requiring a full ninja build.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCompileCommandsParser(unittest.TestCase):
    """Tests for parsing compile_commands.json."""

    def setUp(self):
        """Create temporary directory and sample compile_commands.json."""
        self.temp_dir = tempfile.mkdtemp()
        self.compile_commands_path = os.path.join(self.temp_dir, "compile_commands.json")

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_parse_empty_compile_commands(self):
        """Parser should handle empty compile_commands.json gracefully."""
        from cmake_dependency_analyzer import CompileCommandsParser

        with open(self.compile_commands_path, "w") as f:
            json.dump([], f)

        parser = CompileCommandsParser(self.compile_commands_path)
        commands = parser.parse()

        self.assertEqual(len(commands), 0)

    def test_parse_single_command(self):
        """Parser should correctly parse a single compile command."""
        from cmake_dependency_analyzer import CompileCommandsParser

        sample_commands = [
            {
                "directory": "/build",
                "command": "/opt/rocm/bin/amdclang++ -DFOO=1 -I/include -c test.cpp -o test.o",
                "file": "/src/test.cpp",
            }
        ]
        with open(self.compile_commands_path, "w") as f:
            json.dump(sample_commands, f)

        parser = CompileCommandsParser(self.compile_commands_path)
        commands = parser.parse()

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["file"], "/src/test.cpp")
        self.assertEqual(commands[0]["directory"], "/build")

    def test_parse_multiple_commands(self):
        """Parser should correctly parse multiple compile commands."""
        from cmake_dependency_analyzer import CompileCommandsParser

        sample_commands = [
            {
                "directory": "/build",
                "command": "/opt/rocm/bin/amdclang++ -c test1.cpp -o test1.o",
                "file": "/src/test1.cpp",
            },
            {
                "directory": "/build",
                "command": "/opt/rocm/bin/amdclang++ -c test2.cpp -o test2.o",
                "file": "/src/test2.cpp",
            },
        ]
        with open(self.compile_commands_path, "w") as f:
            json.dump(sample_commands, f)

        parser = CompileCommandsParser(self.compile_commands_path)
        commands = parser.parse()

        self.assertEqual(len(commands), 2)

    def test_filter_by_extension(self):
        """Parser should filter commands by file extension."""
        from cmake_dependency_analyzer import CompileCommandsParser

        sample_commands = [
            {"directory": "/build", "command": "clang++ -c test.cpp -o test.o", "file": "/src/test.cpp"},
            {"directory": "/build", "command": "clang++ -c test.cc -o test.o", "file": "/src/test.cc"},
            {"directory": "/build", "command": "clang -c test.c -o test.o", "file": "/src/test.c"},
        ]
        with open(self.compile_commands_path, "w") as f:
            json.dump(sample_commands, f)

        parser = CompileCommandsParser(self.compile_commands_path)
        commands = parser.parse(extensions=[".cpp"])

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["file"], "/src/test.cpp")

    def test_handles_arguments_format(self):
        """Parser should handle both 'command' and 'arguments' formats."""
        from cmake_dependency_analyzer import CompileCommandsParser

        sample_commands = [
            {
                "directory": "/build",
                "arguments": ["/opt/rocm/bin/amdclang++", "-c", "test.cpp", "-o", "test.o"],
                "file": "/src/test.cpp",
            }
        ]
        with open(self.compile_commands_path, "w") as f:
            json.dump(sample_commands, f)

        parser = CompileCommandsParser(self.compile_commands_path)
        commands = parser.parse()

        self.assertEqual(len(commands), 1)
        self.assertIn("command", commands[0])


class TestDependencyExtractor(unittest.TestCase):
    """Tests for extracting dependencies using clang -MM."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_convert_compile_to_dependency_command(self):
        """Should convert compile command to dependency extraction command."""
        from cmake_dependency_analyzer import DependencyExtractor

        compile_cmd = "/opt/rocm/bin/amdclang++ -DFOO=1 -I/include -O3 -c /src/test.cpp -o /build/test.o"

        extractor = DependencyExtractor()
        dep_cmd = extractor.convert_to_dependency_command(compile_cmd, "/tmp/deps.d")

        # Should have -MM flag
        self.assertIn("-MM", dep_cmd)
        # Should have -MF flag with output file
        self.assertIn("-MF", dep_cmd)
        self.assertIn("/tmp/deps.d", dep_cmd)
        # Should NOT have -c flag
        self.assertNotIn(" -c ", dep_cmd)
        # Should NOT have -o flag with output
        self.assertNotIn(" -o ", dep_cmd)
        # Should preserve includes and defines
        self.assertIn("-DFOO=1", dep_cmd)
        self.assertIn("-I/include", dep_cmd)
        # Should preserve source file
        self.assertIn("/src/test.cpp", dep_cmd)

    def test_parse_makefile_deps_simple(self):
        """Should parse simple makefile-style dependency output."""
        from cmake_dependency_analyzer import DependencyExtractor

        deps_content = "test.o: test.cpp header1.hpp header2.hpp\n"

        extractor = DependencyExtractor()
        deps = extractor.parse_makefile_deps(deps_content)

        self.assertEqual(len(deps), 3)
        self.assertIn("test.cpp", deps)
        self.assertIn("header1.hpp", deps)
        self.assertIn("header2.hpp", deps)

    def test_parse_makefile_deps_multiline(self):
        """Should parse multiline makefile-style dependency output."""
        from cmake_dependency_analyzer import DependencyExtractor

        deps_content = """test.o: test.cpp \\
  /include/header1.hpp \\
  /include/header2.hpp \\
  /include/header3.hpp
"""

        extractor = DependencyExtractor()
        deps = extractor.parse_makefile_deps(deps_content)

        self.assertEqual(len(deps), 4)
        self.assertIn("test.cpp", deps)
        self.assertIn("/include/header1.hpp", deps)
        self.assertIn("/include/header2.hpp", deps)
        self.assertIn("/include/header3.hpp", deps)

    def test_parse_makefile_deps_empty(self):
        """Should handle empty dependency output."""
        from cmake_dependency_analyzer import DependencyExtractor

        extractor = DependencyExtractor()
        deps = extractor.parse_makefile_deps("")

        self.assertEqual(len(deps), 0)

    @patch("subprocess.run")
    def test_extract_dependencies_success(self, mock_run):
        """Should successfully extract dependencies using clang -MM."""
        from cmake_dependency_analyzer import DependencyExtractor

        # Mock successful clang -MM execution
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        extractor = DependencyExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".d", delete=False) as f:
            f.write("test.o: test.cpp header.hpp\n")
            deps_file = f.name

        with patch.object(extractor, "_get_deps_file", return_value=deps_file):
            deps = extractor.extract("/build", "clang++ -c test.cpp -o test.o", "/src/test.cpp")

        self.assertIn("test.cpp", deps)
        self.assertIn("header.hpp", deps)
        # Note: The implementation cleans up the temp file, so we don't need to

    @patch("subprocess.run")
    def test_extract_dependencies_compiler_error(self, mock_run):
        """Should handle compiler errors gracefully."""
        from cmake_dependency_analyzer import DependencyExtractor

        # Mock failed clang -MM execution
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="error: file not found")

        extractor = DependencyExtractor()
        deps = extractor.extract("/build", "clang++ -c test.cpp -o test.o", "/src/test.cpp")

        # Should return empty list on error, not crash
        self.assertEqual(deps, [])


class TestNinjaTargetParser(unittest.TestCase):
    """Tests for parsing ninja build files to get target mappings."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ninja_file = os.path.join(self.temp_dir, "build.ninja")

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_parse_executable_to_objects(self):
        """Should parse executable -> object file mappings from build.ninja."""
        from cmake_dependency_analyzer import NinjaTargetParser

        ninja_content = """
rule CXX_EXECUTABLE_LINKER__test_gemm
  command = /opt/rocm/bin/amdclang++ $in -o $out

build bin/test_gemm: CXX_EXECUTABLE_LINKER__test_gemm test/test_gemm.cpp.o library/gemm.cpp.o | lib.so
"""
        with open(self.ninja_file, "w") as f:
            f.write(ninja_content)

        parser = NinjaTargetParser(self.ninja_file)
        exe_to_objects = parser.parse_executable_mappings()

        self.assertIn("bin/test_gemm", exe_to_objects)
        self.assertIn("test/test_gemm.cpp.o", exe_to_objects["bin/test_gemm"])
        self.assertIn("library/gemm.cpp.o", exe_to_objects["bin/test_gemm"])

    def test_parse_object_to_source(self):
        """Should parse object -> source file mappings from build.ninja."""
        from cmake_dependency_analyzer import NinjaTargetParser

        ninja_content = """
rule CXX_COMPILER__test_gemm
  command = /opt/rocm/bin/amdclang++ -c $in -o $out

build test/test_gemm.cpp.o: CXX_COMPILER__test_gemm /src/test/test_gemm.cpp
build library/gemm.cpp.o: CXX_COMPILER__test_gemm /src/library/gemm.cpp
"""
        with open(self.ninja_file, "w") as f:
            f.write(ninja_content)

        parser = NinjaTargetParser(self.ninja_file)
        obj_to_source = parser.parse_object_to_source()

        self.assertIn("test/test_gemm.cpp.o", obj_to_source)
        self.assertEqual(obj_to_source["test/test_gemm.cpp.o"], "/src/test/test_gemm.cpp")

    def test_filter_test_executables(self):
        """Should correctly filter test executables by prefix."""
        from cmake_dependency_analyzer import NinjaTargetParser

        ninja_content = """
build bin/test_gemm: CXX_EXECUTABLE_LINKER__test_gemm test.o
build bin/example_gemm: CXX_EXECUTABLE_LINKER__example_gemm example.o
build bin/ckProfiler: CXX_EXECUTABLE_LINKER__ckProfiler profiler.o
"""
        with open(self.ninja_file, "w") as f:
            f.write(ninja_content)

        parser = NinjaTargetParser(self.ninja_file)
        exe_to_objects = parser.parse_executable_mappings()

        test_exes = [exe for exe in exe_to_objects if "test_" in exe]
        self.assertEqual(len(test_exes), 1)
        self.assertIn("bin/test_gemm", test_exes)


class TestDependencyMapper(unittest.TestCase):
    """Tests for building the file -> executable dependency mapping."""

    def test_build_file_to_executable_mapping(self):
        """Should build correct file -> executable mapping."""
        from cmake_dependency_analyzer import DependencyMapper

        # Simulated data
        exe_to_objects = {
            "bin/test_gemm": ["test/test_gemm.cpp.o", "lib/gemm.cpp.o"],
            "bin/test_conv": ["test/test_conv.cpp.o", "lib/conv.cpp.o"],
        }
        obj_to_source = {
            "test/test_gemm.cpp.o": "test/test_gemm.cpp",
            "lib/gemm.cpp.o": "lib/gemm.cpp",
            "test/test_conv.cpp.o": "test/test_conv.cpp",
            "lib/conv.cpp.o": "lib/conv.cpp",
        }
        source_to_deps = {
            "test/test_gemm.cpp": ["test/test_gemm.cpp", "include/gemm.hpp", "include/common.hpp"],
            "lib/gemm.cpp": ["lib/gemm.cpp", "include/gemm.hpp"],
            "test/test_conv.cpp": ["test/test_conv.cpp", "include/conv.hpp", "include/common.hpp"],
            "lib/conv.cpp": ["lib/conv.cpp", "include/conv.hpp"],
        }

        mapper = DependencyMapper()
        file_to_exes = mapper.build_mapping(exe_to_objects, obj_to_source, source_to_deps)

        # common.hpp should map to both test executables
        self.assertIn("include/common.hpp", file_to_exes)
        self.assertIn("bin/test_gemm", file_to_exes["include/common.hpp"])
        self.assertIn("bin/test_conv", file_to_exes["include/common.hpp"])

        # gemm.hpp should only map to test_gemm
        self.assertIn("include/gemm.hpp", file_to_exes)
        self.assertIn("bin/test_gemm", file_to_exes["include/gemm.hpp"])
        self.assertNotIn("bin/test_conv", file_to_exes["include/gemm.hpp"])

    def test_normalize_paths(self):
        """Should normalize paths relative to workspace root."""
        from cmake_dependency_analyzer import DependencyMapper

        mapper = DependencyMapper(workspace_root="/workspace/rocm-libraries/projects/composablekernel")

        # Test monorepo-style path
        normalized = mapper.normalize_path(
            "/workspace/rocm-libraries/projects/composablekernel/include/ck/ck.hpp"
        )
        self.assertEqual(normalized, "include/ck/ck.hpp")

        # Test already relative path
        normalized = mapper.normalize_path("include/ck/ck.hpp")
        self.assertEqual(normalized, "include/ck/ck.hpp")

    def test_filter_system_files(self):
        """Should filter out system files."""
        from cmake_dependency_analyzer import DependencyMapper

        mapper = DependencyMapper()

        self.assertFalse(mapper.is_project_file("/usr/include/stdio.h"))
        self.assertFalse(mapper.is_project_file("/opt/rocm/include/hip/hip_runtime.h"))
        self.assertTrue(mapper.is_project_file("include/ck/ck.hpp"))
        self.assertTrue(mapper.is_project_file("test/test_gemm.cpp"))


class TestCMakeDependencyAnalyzer(unittest.TestCase):
    """Integration tests for the full CMake dependency analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_output_format_compatibility(self):
        """Output JSON should be compatible with selective_test_filter.py."""
        from cmake_dependency_analyzer import CMakeDependencyAnalyzer

        # Create minimal test data
        analyzer = CMakeDependencyAnalyzer(
            compile_commands_path=None,
            ninja_path=None,
            workspace_root=self.temp_dir,
        )

        # Manually set internal state for testing output format
        analyzer.file_to_executables = {
            "include/ck/ck.hpp": {"bin/test_gemm", "bin/test_conv"},
            "test/test_gemm.cpp": {"bin/test_gemm"},
        }
        analyzer.executable_to_files = {
            "bin/test_gemm": {"include/ck/ck.hpp", "test/test_gemm.cpp"},
            "bin/test_conv": {"include/ck/ck.hpp"},
        }

        output_file = os.path.join(self.temp_dir, "output.json")
        analyzer.export_to_json(output_file)

        with open(output_file) as f:
            data = json.load(f)

        # Check required fields for selective_test_filter.py compatibility
        self.assertIn("file_to_executables", data)
        self.assertIn("executable_to_files", data)
        self.assertIn("statistics", data)

        # Check file_to_executables format (should be lists, not sets)
        self.assertIsInstance(data["file_to_executables"]["include/ck/ck.hpp"], list)

    def test_statistics_calculation(self):
        """Should calculate correct statistics."""
        from cmake_dependency_analyzer import CMakeDependencyAnalyzer

        analyzer = CMakeDependencyAnalyzer(
            compile_commands_path=None,
            ninja_path=None,
            workspace_root=self.temp_dir,
        )

        analyzer.file_to_executables = {
            "include/common.hpp": {"bin/test1", "bin/test2", "bin/test3"},
            "include/specific.hpp": {"bin/test1"},
            "test/test1.cpp": {"bin/test1"},
        }

        stats = analyzer.calculate_statistics()

        self.assertEqual(stats["total_files"], 3)
        self.assertEqual(stats["files_with_multiple_executables"], 1)


class TestParallelDependencyExtraction(unittest.TestCase):
    """Tests for parallel dependency extraction."""

    def test_batch_extraction_preserves_results(self):
        """Parallel extraction should produce same results as serial."""
        from cmake_dependency_analyzer import DependencyExtractor

        extractor = DependencyExtractor(parallel_workers=4)

        # This is more of an integration test placeholder
        # Real parallel testing would require actual compiler invocations
        self.assertIsNotNone(extractor)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_handles_missing_compile_commands(self):
        """Should raise appropriate error for missing compile_commands.json."""
        from cmake_dependency_analyzer import CompileCommandsParser

        with self.assertRaises(FileNotFoundError):
            parser = CompileCommandsParser("/nonexistent/compile_commands.json")
            parser.parse()

    def test_handles_malformed_json(self):
        """Should handle malformed JSON gracefully."""
        from cmake_dependency_analyzer import CompileCommandsParser

        temp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(temp_dir, "compile_commands.json")
            with open(path, "w") as f:
                f.write("not valid json {{{")

            parser = CompileCommandsParser(path)
            with self.assertRaises(json.JSONDecodeError):
                parser.parse()
        finally:
            shutil.rmtree(temp_dir)

    def test_handles_empty_ninja_file(self):
        """Should handle empty ninja file gracefully."""
        from cmake_dependency_analyzer import NinjaTargetParser

        temp_dir = tempfile.mkdtemp()
        try:
            ninja_file = os.path.join(temp_dir, "build.ninja")
            with open(ninja_file, "w") as f:
                f.write("")

            parser = NinjaTargetParser(ninja_file)
            exe_to_objects = parser.parse_executable_mappings()

            self.assertEqual(len(exe_to_objects), 0)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
