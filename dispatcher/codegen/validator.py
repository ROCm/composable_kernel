#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Validator - Verify Generated Kernels

Validates generated kernel code and dispatcher wrappers to ensure:
- Syntactic correctness
- Semantic consistency
- Naming conventions
- Type safety
- Integration compatibility
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


# ============================================================================
# Validation Results
# ============================================================================

class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"      # Must fix
    WARNING = "warning"  # Should fix
    INFO = "info"        # Nice to have


@dataclass
class ValidationIssue:
    """Single validation issue"""
    level: ValidationLevel
    file_path: Path
    line_number: Optional[int]
    message: str
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        loc = f"{self.file_path}"
        if self.line_number:
            loc += f":{self.line_number}"
        
        msg = f"[{self.level.value.upper()}] {loc}: {self.message}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Validation results for a file or set of files"""
    file_path: Path
    passed: bool
    issues: List[ValidationIssue]
    
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)
    
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)
    
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.INFO)
    
    def summary(self) -> str:
        return (f"Validation: {'PASSED' if self.passed else 'FAILED'} - "
                f"Errors: {self.error_count()}, "
                f"Warnings: {self.warning_count()}, "
                f"Info: {self.info_count()}")


# ============================================================================
# Base Validator
# ============================================================================

class BaseValidator:
    """Base class for validators"""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def add_error(self, file_path: Path, message: str, 
                  line_number: Optional[int] = None,
                  suggestion: Optional[str] = None):
        """Add error issue"""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            file_path=file_path,
            line_number=line_number,
            message=message,
            suggestion=suggestion
        ))
    
    def add_warning(self, file_path: Path, message: str,
                    line_number: Optional[int] = None,
                    suggestion: Optional[str] = None):
        """Add warning issue"""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.WARNING,
            file_path=file_path,
            line_number=line_number,
            message=message,
            suggestion=suggestion
        ))
    
    def add_info(self, file_path: Path, message: str,
                 line_number: Optional[int] = None,
                 suggestion: Optional[str] = None):
        """Add info issue"""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.INFO,
            file_path=file_path,
            line_number=line_number,
            message=message,
            suggestion=suggestion
        ))
    
    def validate(self, file_path: Path) -> ValidationResult:
        """Validate file (to be implemented by subclasses)"""
        raise NotImplementedError


# ============================================================================
# Kernel Header Validator
# ============================================================================

class KernelHeaderValidator(BaseValidator):
    """Validate generated CK Tile kernel headers"""
    
    def validate(self, file_path: Path) -> ValidationResult:
        """Validate kernel header file"""
        self.issues = []
        
        if not file_path.exists():
            self.add_error(file_path, "File does not exist")
            return ValidationResult(file_path, False, self.issues)
        
        try:
            content = file_path.read_text()
        except Exception as e:
            self.add_error(file_path, f"Failed to read file: {e}")
            return ValidationResult(file_path, False, self.issues)
        
        # Run validation checks
        self._check_header_guard(file_path, content)
        self._check_includes(file_path, content)
        self._check_namespace(file_path, content)
        self._check_kernel_struct(file_path, content)
        self._check_types(file_path, content)
        self._check_launch_function(file_path, content)
        self._check_naming_convention(file_path, content)
        
        # Passed if no errors
        passed = all(i.level != ValidationLevel.ERROR for i in self.issues)
        
        return ValidationResult(file_path, passed, self.issues)
    
    def _check_header_guard(self, file_path: Path, content: str):
        """Check for proper header guard"""
        if '#pragma once' not in content:
            if '#ifndef' not in content or '#define' not in content:
                self.add_warning(
                    file_path,
                    "Missing header guard",
                    suggestion="Add '#pragma once' at the top"
                )
    
    def _check_includes(self, file_path: Path, content: str):
        """Check for required includes"""
        required_includes = [
            'ck_tile/core.hpp',
            'ck_tile/ops/gemm.hpp',
        ]
        
        for inc in required_includes:
            if inc not in content:
                self.add_warning(
                    file_path,
                    f"Missing include: {inc}",
                    suggestion=f'Add: #include "{inc}"'
                )
    
    def _check_namespace(self, file_path: Path, content: str):
        """Check namespace usage"""
        # Should not have 'using namespace' in headers
        if re.search(r'using\s+namespace\s+\w+', content):
            self.add_warning(
                file_path,
                "Avoid 'using namespace' in headers",
                suggestion="Use explicit namespace qualifications"
            )
    
    def _check_kernel_struct(self, file_path: Path, content: str):
        """Check for SelectedKernel struct"""
        if 'struct SelectedKernel' not in content:
            self.add_error(
                file_path,
                "Missing 'struct SelectedKernel'",
                suggestion="Kernel must define SelectedKernel struct"
            )
    
    def _check_types(self, file_path: Path, content: str):
        """Check type definitions"""
        required_types = [
            'ADataType', 'BDataType', 'CDataType', 'AccDataType',
            'ALayout', 'BLayout', 'CLayout',
        ]
        
        for dtype in required_types:
            if f'using {dtype}' not in content:
                self.add_warning(
                    file_path,
                    f"Missing type definition: {dtype}",
                    suggestion=f"Add: using {dtype} = ...;"
                )
    
    def _check_launch_function(self, file_path: Path, content: str):
        """Check for launch function"""
        if 'static float launch(' not in content:
            self.add_error(
                file_path,
                "Missing launch function",
                suggestion="Add: static float launch(const ck_tile::GemmHostArgs&, ...)"
            )
    
    def _check_naming_convention(self, file_path: Path, content: str):
        """Check naming conventions"""
        # Check KERNEL_NAME constant
        if 'constexpr const char* KERNEL_NAME' not in content:
            self.add_info(
                file_path,
                "Missing KERNEL_NAME constant",
                suggestion="Add: constexpr const char* KERNEL_NAME = \"...\";"
            )


# ============================================================================
# Dispatcher Wrapper Validator
# ============================================================================

class DispatcherWrapperValidator(BaseValidator):
    """Validate generated dispatcher wrapper headers"""
    
    def validate(self, file_path: Path) -> ValidationResult:
        """Validate dispatcher wrapper file"""
        self.issues = []
        
        if not file_path.exists():
            self.add_error(file_path, "File does not exist")
            return ValidationResult(file_path, False, self.issues)
        
        try:
            content = file_path.read_text()
        except Exception as e:
            self.add_error(file_path, f"Failed to read file: {e}")
            return ValidationResult(file_path, False, self.issues)
        
        # Run validation checks
        self._check_header_guard(file_path, content)
        self._check_dispatcher_include(file_path, content)
        self._check_namespace(file_path, content)
        self._check_make_function(file_path, content)
        self._check_kernel_key(file_path, content)
        
        # Passed if no errors
        passed = all(i.level != ValidationLevel.ERROR for i in self.issues)
        
        return ValidationResult(file_path, passed, self.issues)
    
    def _check_header_guard(self, file_path: Path, content: str):
        """Check for proper header guard"""
        if '#pragma once' not in content:
            self.add_warning(
                file_path,
                "Missing header guard",
                suggestion="Add '#pragma once'"
            )
    
    def _check_dispatcher_include(self, file_path: Path, content: str):
        """Check for dispatcher include"""
        if '#include "ck_tile/dispatcher.hpp"' not in content:
            self.add_error(
                file_path,
                "Missing dispatcher include",
                suggestion='Add: #include "ck_tile/dispatcher.hpp"'
            )
    
    def _check_namespace(self, file_path: Path, content: str):
        """Check namespace structure"""
        required_namespaces = [
            'namespace ck_tile',
            'namespace dispatcher',
            'namespace generated',
        ]
        
        for ns in required_namespaces:
            if ns not in content:
                self.add_error(
                    file_path,
                    f"Missing namespace: {ns}",
                    suggestion=f"Add: {ns} {{ ... }}"
                )
    
    def _check_make_function(self, file_path: Path, content: str):
        """Check for make_* function"""
        if not re.search(r'inline\s+KernelInstancePtr\s+make_\w+', content):
            self.add_error(
                file_path,
                "Missing make_* function",
                suggestion="Add: inline KernelInstancePtr make_kernel_name(...)"
            )
    
    def _check_kernel_key(self, file_path: Path, content: str):
        """Check KernelKey setup"""
        key_fields = [
            'key.signature.dtype_a',
            'key.signature.dtype_b',
            'key.signature.dtype_c',
            'key.algorithm.tile_shape',
            'key.algorithm.pipeline',
            'key.gfx_arch',
        ]
        
        for field in key_fields:
            if field not in content:
                self.add_warning(
                    file_path,
                    f"Missing KernelKey field: {field}",
                    suggestion=f"Set: {field} = ...;"
                )


# ============================================================================
# Registration Header Validator
# ============================================================================

class RegistrationHeaderValidator(BaseValidator):
    """Validate registration header"""
    
    def validate(self, file_path: Path) -> ValidationResult:
        """Validate registration header"""
        self.issues = []
        
        if not file_path.exists():
            self.add_error(file_path, "File does not exist")
            return ValidationResult(file_path, False, self.issues)
        
        try:
            content = file_path.read_text()
        except Exception as e:
            self.add_error(file_path, f"Failed to read file: {e}")
            return ValidationResult(file_path, False, self.issues)
        
        # Check registration function
        if 'inline void register_all_tile_gemm_kernels' not in content:
            self.add_error(
                file_path,
                "Missing registration function",
                suggestion="Add: inline void register_all_tile_gemm_kernels(...)"
            )
        
        # Check count function
        if 'inline std::size_t get_tile_gemm_kernel_count' not in content:
            self.add_warning(
                file_path,
                "Missing count function",
                suggestion="Add: inline std::size_t get_tile_gemm_kernel_count()"
            )
        
        passed = all(i.level != ValidationLevel.ERROR for i in self.issues)
        return ValidationResult(file_path, passed, self.issues)


# ============================================================================
# Batch Validator
# ============================================================================

class BatchValidator:
    """Validate multiple files"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def validate_directory(self, directory: Path) -> List[ValidationResult]:
        """Validate all files in directory"""
        log.info(f"Validating directory: {directory}")
        
        # Validate kernel headers
        for kernel_file in directory.glob("gemm_*.hpp"):
            validator = KernelHeaderValidator()
            result = validator.validate(kernel_file)
            self.results.append(result)
            
            if not result.passed:
                log.warning(f"Validation failed: {kernel_file.name}")
        
        # Validate dispatcher wrappers
        wrapper_dir = directory / "dispatcher_wrappers"
        if wrapper_dir.exists():
            for wrapper_file in wrapper_dir.glob("dispatcher_wrapper_*.hpp"):
                validator = DispatcherWrapperValidator()
                result = validator.validate(wrapper_file)
                self.results.append(result)
                
                if not result.passed:
                    log.warning(f"Validation failed: {wrapper_file.name}")
            
            # Validate registration header
            reg_file = wrapper_dir / "register_all_kernels.hpp"
            if reg_file.exists():
                validator = RegistrationHeaderValidator()
                result = validator.validate(reg_file)
                self.results.append(result)
        
        return self.results
    
    def print_summary(self):
        """Print validation summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        total_errors = sum(r.error_count() for r in self.results)
        total_warnings = sum(r.warning_count() for r in self.results)
        total_info = sum(r.info_count() for r in self.results)
        
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total files: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"\nIssues:")
        print(f"  Errors: {total_errors}")
        print(f"  Warnings: {total_warnings}")
        print(f"  Info: {total_info}")
        print("=" * 70)
        
        # Print failed files
        if failed > 0:
            print("\nFailed files:")
            for result in self.results:
                if not result.passed:
                    print(f"  {result.file_path.name}")
                    for issue in result.issues:
                        if issue.level == ValidationLevel.ERROR:
                            print(f"    - {issue.message}")
    
    def get_all_issues(self) -> List[ValidationIssue]:
        """Get all issues from all results"""
        issues = []
        for result in self.results:
            issues.extend(result.issues)
        return issues


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    from utils import setup_logging
    
    parser = argparse.ArgumentParser(description='Validate generated kernels')
    parser.add_argument('directory', type=Path,
                       help='Directory containing generated kernels')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all issues (including warnings and info)')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Validate directory
    validator = BatchValidator()
    validator.validate_directory(args.directory)
    
    # Print summary
    validator.print_summary()
    
    # Print detailed issues if requested
    if args.show_all:
        print("\nDetailed Issues:")
        print("=" * 70)
        for issue in validator.get_all_issues():
            print(issue)
            print()
    
    # Exit with error if any validation failed
    failed = sum(1 for r in validator.results if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    exit(main())

