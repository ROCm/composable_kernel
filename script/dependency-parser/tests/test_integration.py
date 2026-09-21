#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Integration tests for CMake Dependency Analyzer.

These tests use real compile_commands.json and actual AMD clang compiler
to verify the analyzer works correctly in production environment.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Skip all tests if compile_commands.json doesn't exist
CK_ROOT = Path(__file__).parent.parent.parent.parent
BUILD_DIR = CK_ROOT / "build"
COMPILE_COMMANDS = BUILD_DIR / "compile_commands.json"
BUILD_NINJA = BUILD_DIR / "build.ninja"

SKIP_INTEGRATION = not COMPILE_COMMANDS.exists()
SKIP_REASON = f"compile_commands.json not found at {COMPILE_COMMANDS}"


@unittest.skipIf(SKIP_INTEGRATION, SKIP_REASON)
class TestRealCompileCommands(unittest.TestCase):
    """Tests using real compile_commands.json."""

    def test_parse_real_compile_commands(self):
        """Should parse real CK compile_commands.json."""
        from cmake_dependency_analyzer import CompileCommandsParser

        parser = CompileCommandsParser(str(COMPILE_COMMANDS))
        commands = parser.parse()

        # CK has thousands of source files
        self.assertGreater(len(commands), 100)

        # Verify structure
        for cmd in commands[:5]:
            self.assertIn("file", cmd)
            self.assertIn("directory", cmd)
            self.assertIn("command", cmd)

    def test_filter_cpp_files_only(self):
        """Should correctly filter to only .cpp files."""
        from cmake_dependency_analyzer import CompileCommandsParser

        parser = CompileCommandsParser(str(COMPILE_COMMANDS))
        commands = parser.parse(extensions=[".cpp"])

        for cmd in commands:
            self.assertTrue(
                cmd["file"].endswith(".cpp"),
                f"Expected .cpp file, got {cmd['file']}",
            )


@unittest.skipIf(SKIP_INTEGRATION, SKIP_REASON)
class TestRealDependencyExtraction(unittest.TestCase):
    """Tests using real AMD clang for dependency extraction."""

    def test_extract_real_dependencies(self):
        """Should extract dependencies using real AMD clang."""
        from cmake_dependency_analyzer import CompileCommandsParser, DependencyExtractor

        parser = CompileCommandsParser(str(COMPILE_COMMANDS))
        commands = parser.parse(extensions=[".cpp"])

        # Test with first command
        if not commands:
            self.skipTest("No compile commands found")

        cmd = commands[0]
        extractor = DependencyExtractor()
        deps = extractor.extract(cmd["directory"], cmd["command"], cmd["file"])

        # Should have at least the source file itself
        self.assertGreater(len(deps), 0, f"No deps found for {cmd['file']}")

        # Should include the source file
        source_basename = os.path.basename(cmd["file"])
        found_source = any(source_basename in d for d in deps)
        self.assertTrue(found_source, f"Source file not in deps: {deps[:5]}")

    def test_extract_header_dependencies(self):
        """Should find CK header dependencies."""
        from cmake_dependency_analyzer import CompileCommandsParser, DependencyExtractor

        parser = CompileCommandsParser(str(COMPILE_COMMANDS))
        commands = parser.parse(extensions=[".cpp"])

        # Find a test file that includes CK headers
        test_cmd = None
        for cmd in commands:
            if "test_" in cmd["file"] or "example_" in cmd["file"]:
                test_cmd = cmd
                break

        if not test_cmd:
            self.skipTest("No test file found")

        extractor = DependencyExtractor()
        deps = extractor.extract(test_cmd["directory"], test_cmd["command"], test_cmd["file"])

        # Should include CK headers
        ck_headers = [d for d in deps if "include/ck" in d or "include/ck_tile" in d]
        self.assertGreater(
            len(ck_headers), 0,
            f"No CK headers found in deps for {test_cmd['file']}"
        )


@unittest.skipIf(SKIP_INTEGRATION, SKIP_REASON)
@unittest.skipIf(not BUILD_NINJA.exists(), "build.ninja not found")
class TestRealNinjaParsing(unittest.TestCase):
    """Tests using real build.ninja."""

    def test_parse_real_executables(self):
        """Should parse real executable mappings from build.ninja."""
        from cmake_dependency_analyzer import NinjaTargetParser

        parser = NinjaTargetParser(str(BUILD_NINJA))
        exe_to_objects = parser.parse_executable_mappings()

        # CK has many test executables
        test_exes = [e for e in exe_to_objects if "test_" in e]
        self.assertGreater(len(test_exes), 10, "Expected many test executables")

        # Each executable should have at least one object file
        for exe, objs in list(exe_to_objects.items())[:5]:
            self.assertGreater(len(objs), 0, f"No objects for {exe}")

    def test_parse_real_object_sources(self):
        """Should parse real object -> source mappings."""
        from cmake_dependency_analyzer import NinjaTargetParser

        parser = NinjaTargetParser(str(BUILD_NINJA))
        obj_to_source = parser.parse_object_to_source()

        # Should have many object files
        self.assertGreater(len(obj_to_source), 100)

        # Each mapping should have valid source file
        for obj, src in list(obj_to_source.items())[:5]:
            self.assertTrue(
                src.endswith((".cpp", ".cc", ".cu", ".hip")),
                f"Invalid source for {obj}: {src}",
            )


@unittest.skipIf(SKIP_INTEGRATION, SKIP_REASON)
@unittest.skipIf(not BUILD_NINJA.exists(), "build.ninja not found")
class TestFullIntegration(unittest.TestCase):
    """Full integration test of the analyzer."""

    def test_small_batch_analysis(self):
        """Should analyze a small batch of files correctly."""
        from cmake_dependency_analyzer import (
            CompileCommandsParser,
            DependencyExtractor,
            NinjaTargetParser,
            DependencyMapper,
        )

        # Parse compile commands (limit to 10 for speed)
        parser = CompileCommandsParser(str(COMPILE_COMMANDS))
        all_commands = parser.parse(extensions=[".cpp"])
        commands = all_commands[:10]

        # Extract dependencies
        extractor = DependencyExtractor()
        source_to_deps = extractor.extract_batch(commands)

        self.assertEqual(len(source_to_deps), len(commands))

        # Parse ninja
        ninja_parser = NinjaTargetParser(str(BUILD_NINJA))
        exe_to_objects = ninja_parser.parse_executable_mappings()
        obj_to_source = ninja_parser.parse_object_to_source()

        # Build mapping
        mapper = DependencyMapper(workspace_root=str(CK_ROOT))
        file_to_exes = mapper.build_mapping(exe_to_objects, obj_to_source, source_to_deps)

        # Should have some mappings (depends on which files were analyzed)
        # This test mainly verifies no crashes occur
        self.assertIsInstance(file_to_exes, dict)

    def test_output_json_format(self):
        """Should produce JSON compatible with selective_test_filter.py."""
        from cmake_dependency_analyzer import CMakeDependencyAnalyzer

        # Create analyzer with limited scope
        analyzer = CMakeDependencyAnalyzer(
            compile_commands_path=str(COMPILE_COMMANDS),
            ninja_path=str(BUILD_NINJA),
            workspace_root=str(CK_ROOT),
            parallel_workers=1,
        )

        # Manually set minimal data for output test
        analyzer.file_to_executables = {
            "include/ck/ck.hpp": {"bin/test_gemm"},
        }
        analyzer.executable_to_files = {
            "bin/test_gemm": {"include/ck/ck.hpp"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            analyzer.export_to_json(output_path)

            with open(output_path) as f:
                data = json.load(f)

            # Verify format matches selective_test_filter.py expectations
            self.assertIn("file_to_executables", data)
            self.assertIn("statistics", data)

            # Values should be lists, not sets
            for key, value in data["file_to_executables"].items():
                self.assertIsInstance(value, list)
        finally:
            os.unlink(output_path)


@unittest.skipIf(SKIP_INTEGRATION, SKIP_REASON)
class TestPerformance(unittest.TestCase):
    """Performance tests."""

    def test_extraction_speed(self):
        """Single file extraction should be fast (<1s)."""
        import time
        from cmake_dependency_analyzer import CompileCommandsParser, DependencyExtractor

        parser = CompileCommandsParser(str(COMPILE_COMMANDS))
        commands = parser.parse(extensions=[".cpp"])

        if not commands:
            self.skipTest("No compile commands")

        cmd = commands[0]
        extractor = DependencyExtractor()

        start = time.time()
        deps = extractor.extract(cmd["directory"], cmd["command"], cmd["file"])
        elapsed = time.time() - start

        self.assertLess(elapsed, 1.0, f"Extraction took {elapsed:.2f}s, expected <1s")
        self.assertGreater(len(deps), 0, "No dependencies extracted")


if __name__ == "__main__":
    unittest.main()
