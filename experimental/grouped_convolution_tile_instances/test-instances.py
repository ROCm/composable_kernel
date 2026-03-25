#!/usr/bin/env python3
# filepath: /home/AMD/vpietila/git/rocm-libraries/projects/composablekernel/experimental/grouped_convolution_tile_instances/check_instances.py

"""
Script to check which backward weight convolution instances compile successfully.
Compiles each .cpp file independently and reports failures grouped by layout/datatype.
"""

import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configuration
CXX_COMPILER = "/opt/rocm/bin/hipcc"
CXX_STANDARD = "20"

def get_include_dirs(project_root: Path) -> list[str]:
    """Get the include directories needed for compilation."""
    return [
        str(project_root / "build" / "include"),
        str(project_root / "include"),
        str(project_root / "library" / "include"),
        str(project_root / "experimental" / "builder" / "include"),
        str(project_root / "experimental" / "builder" / "test" / "utils"),
        str(project_root / "experimental" / "grouped_convolution_tile_instances" / "instances"),
    ]

def compile_single_file(cpp_file: Path, project_root: Path, gpu_target: str, verbose: bool) -> tuple[bool, str]:
    """
    Attempt to compile a single .cpp file.
    Returns (success, error_message).
    """
    include_dirs = get_include_dirs(project_root)
    include_flags = [f"-I{d}" for d in include_dirs]
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.o"
        
        cmd = [
            CXX_COMPILER,
            "-c",  # Compile only, don't link
            f"-std=c++{CXX_STANDARD}",
            f"--offload-arch={gpu_target}",
            "-D__HIP_PLATFORM_AMD__",
            "-D CK_EXPERIMENTAL_BUILDER=ON",
            "-O3",
            "-Wno-unknown-warning-option",
            *include_flags,
            str(cpp_file),
            "-o", str(output_file)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            
            print(f"\n\n    Command: {' '.join(cmd)}\n") if verbose else None

            if result.returncode == 0:
                return True, ""
            else:
                # Extract the key error message
                if verbose and result.stderr:
                    print(f"    {result.stderr}")
                    print()
                error_output = result.stderr
                return False, error_output
                
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT: Compilation took too long"
        except Exception as e:
            return False, f"EXCEPTION: {str(e)}"

def extract_key_error(error_output: str) -> str:
    """Extract the most relevant error message from compiler output."""
    lines = error_output.split('\n')
    for line in lines:
        if 'error:' in line:
            return line.strip()
    # Return first non-empty line if no explicit error found
    for line in lines:
        if line.strip():
            return line.strip()[:200]  # Limit length
    return "Unknown error"

def find_instance_files(instances_dir: Path, direction: str = "backward_weight") -> dict[str, list[Path]]:
    """
    Find all instance .cpp files grouped by subdirectory (layout/datatype).
    Returns dict: subdirectory_name -> list of cpp files
    """
    target_dir = instances_dir / direction
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist")
        sys.exit(1)
    
    files_by_subdir = defaultdict(list)
    
    for subdir in sorted(target_dir.iterdir()):
        if subdir.is_dir():
            cpp_files = sorted(subdir.glob("*.cpp"))
            if cpp_files:
                files_by_subdir[subdir.name] = cpp_files
    
    return files_by_subdir

def parse_args():
    parser = argparse.ArgumentParser(description="Check which convolution instances compile")
    parser.add_argument("--direction", default="backward_weight", 
                        choices=["forward", "backward_weight", "backward_data"],
                        help="Convolution direction to check")
    parser.add_argument("--subdir", default=None,
                        help="Only check specific subdirectory (e.g., 'nhwgc_bf16')")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to check per subdirectory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose output including compile commands")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file for the blacklist")
    parser.add_argument("--project-root", default=None,
                        help="Project root directory (auto-detected if not specified)")
    parser.add_argument("--instance", type=int, default=None,
                        help="Only check a single instance by its index in the config file.")
    parser.add_argument(
        "--parallel-jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel compilation jobs (default: 1)",
    )
    parser.add_argument("--gpu-target", type=str, default="gfx950", help="GPU target architecture (default: gfx950)")
    
    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    
    # Find project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Assume script is in experimental/grouped_convolution_tile_instances/
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # Go up to composablekernel/
    
    instances_dir = project_root / "experimental" / "grouped_convolution_tile_instances" / "instances"
    
    print(f"Project root: {project_root}")
    print(f"Instances directory: {instances_dir}")
    print(f"Compiler: {CXX_COMPILER}")
    print(f"GPU Target: {args.gpu_target}")
    print(f"Direction: {args.direction}")
    if args.instance is not None:
        print(f"Checking only instance index: {args.instance}")
    print()
    
    # Find all instance files
    files_by_subdir = find_instance_files(instances_dir, args.direction)
    
    # If sub directory is defined, check only that sub directory
    if args.subdir:
        if args.subdir not in files_by_subdir:
            print(f"Error: Subdirectory '{args.subdir}' not found")
            print(f"Available: {list(files_by_subdir.keys())}")
            sys.exit(1)
        files_by_subdir = {args.subdir: files_by_subdir[args.subdir]}

    if args.instance is not None:
        # If instance index is specified, find the corresponding file for each subdir
        instance_files = {}
        for subdir, files in files_by_subdir.items():
            if args.instance >= 0:
                target_suffix = f"_{args.instance}.cpp"
                matched_files = [f for f in files if f.name.endswith(target_suffix)]
                if matched_files:
                    assert len(matched_files) == 1, f"Expected exactly one file ending with {target_suffix} in {subdir}, found {len(matched_files)}"
                    instance_files[subdir] = matched_files
            else:
                if args.subdir is None:
                    print(f"Warning: Subdirectory '{subdir}' does not have instance index {args.instance}")
        files_by_subdir = instance_files

    if args.subdir:
        if args.instance is not None and args.subdir not in files_by_subdir:
            print(f"Instance index {args.instance} was not found in subdirectory '{args.subdir}'")
            sys.exit(1)
        elif args.subdir not in files_by_subdir:
            print(f"Error: Subdirectory '{args.subdir}' not found")
            print(f"Available: {list(files_by_subdir.keys())}")
            sys.exit(1)
        files_by_subdir = {args.subdir: files_by_subdir[args.subdir]}
    
    # Track results
    all_failures = defaultdict(list)  # subdir -> list of (filename, error)
    all_successes = defaultdict(list)  # subdir -> list of filenames
    error_types = defaultdict(set)  # error_key -> set of files
    
    total_files = sum(len(files) for files in files_by_subdir.values())
    if args.max_files:
        total_files = min(total_files, args.max_files * len(files_by_subdir))
    
    print(f"Found {total_files} instance files to check")
    print("=" * 60)
    
    checked = 0
    for subdir_name, cpp_files in sorted(files_by_subdir.items()):
        print(f"\nChecking {subdir_name}...", flush=True)
        
        files_to_check = cpp_files
        if args.max_files:
            files_to_check = cpp_files[:args.max_files]
        
        if args.parallel_jobs > 1:
            # Parallel compilation
            with ThreadPoolExecutor(max_workers=args.parallel_jobs) as executor:
                # Submit all compilation jobs
                futures = {
                    executor.submit(compile_single_file, cpp_file, project_root, args.gpu_target, args.verbose): cpp_file
                    for cpp_file in files_to_check
                }
                
                # Process results as they complete
                for future in as_completed(futures):
                    cpp_file = futures[future]
                    filename = cpp_file.name
                    checked += 1
                    success, error = future.result()
                    
                    if success:
                        print(f"  [{checked}/{total_files}] {filename}... OK", flush=True)
                        all_successes[subdir_name].append(filename)
                    else:
                        key_error = extract_key_error(error)
                        print(f"  [{checked}/{total_files}] {filename}... FAILED", flush=True)
                        if args.verbose:
                            print(f"    Error: {key_error}")
                        all_failures[subdir_name].append((filename, key_error))
                        error_types[key_error].add(f"{subdir_name}/{filename}")
        else:
            # Sequential compilation
            print(f"Compiling {len(files_to_check)} files sequentially...")
            for cpp_file in files_to_check:
                checked += 1
                filename = cpp_file.name
                
                if args.verbose:
                    print(f"  [{checked}/{total_files}] {filename}...", end=" ", flush=True)
                else:
                    print(f"  [{checked}/{total_files}] {filename}...", end=" ", flush=True)
                
                success, error = compile_single_file(cpp_file, project_root, args.gpu_target, args.verbose)
                
                if success:
                    print("OK")
                    all_successes[subdir_name].append(filename)
                else:
                    key_error = extract_key_error(error)
                    print(f"FAILED")
                    if args.verbose:
                        print(f"    Error: {key_error}")
                    all_failures[subdir_name].append((filename, key_error))
                    error_types[key_error].add(f"{subdir_name}/{filename}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for subdir_name in sorted(files_by_subdir.keys()):
        successes = len(all_successes.get(subdir_name, []))
        failures = len(all_failures.get(subdir_name, []))
        total = successes + failures
        print(f"\n{subdir_name}: {successes}/{total} passed, {failures} failed")
        
        if failures > 0:
            print(f"  Failed files:")
            # Order the failures by the filename for consistency
            # Each filename ends with _{instance_index}.cpp, so we can sort by instance index
            sorted_failures = sorted(
                all_failures[subdir_name],
                key=lambda x: int(re.search(r'_(\d+)\.cpp$', x[0]).group(1)) 
                              if re.search(r'_(\d+)\.cpp$', x[0]) else 0
            )
            for filename, error in sorted_failures:
                print(f"    - {filename}")
    
    # Return exit code based on failures
    total_failures = sum(len(f) for f in all_failures.values())
    if total_failures > 0:
        print(f"\n{total_failures} total failures found")
        return 1
    else:
        print("\nAll instances compiled successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())