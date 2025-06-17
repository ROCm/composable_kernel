#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path
from datetime import datetime

def get_copyright_header():
    """
    Get the copyright header for C/C++ files.
    Returns the header as a string with proper comment style.
    """
    current_year = datetime.now().year
    
    return f"""// SPDX-License-Identifier: MIT
// Copyright (c) {current_year}, Advanced Micro Devices, Inc. All rights reserved.

"""

def insert_copyright_header(file_path):
    """
    Insert copyright header at the top of a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        header = get_copyright_header()
        
        # Insert header at the beginning of the file
        new_content = header + content
        
        # Write the modified content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
        
    except Exception as e:
        print(f"Error inserting header into {file_path}: {e}")
        return False

def has_copyright_header(file_path):
    """
    Check if a file has the proper copyright header.
    Returns True if copyright header is found, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read first few lines to check for copyright header
            lines = []
            for i, line in enumerate(f):
                if i >= 10:  # Only check first 10 lines
                    break
                lines.append(line.strip()) # remove whitespace
            
        content = '\n'.join(lines) # join the lines into a single string
        
        # Check for SPDX-License-Identifier: MIT (the standard format used in this repo)
        has_spdx = re.search(r'SPDX-License-Identifier:\s*MIT', content, re.IGNORECASE)
        
        # Check for specific AMD copyright format used in this repo
        has_amd_copyright = re.search(r'Copyright.*Advanced Micro Devices, Inc\..*All rights reserved', content, re.IGNORECASE)
        
        return bool(has_spdx and has_amd_copyright)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def find_files_without_copyright(root_dir):
    """
    Scan directory recursively for files with specified extensions
    that don't have copyright headers.
    """
    extensions = {'.cpp', '.hpp'}
    files_without_copyright = []
    
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob('*'): # for all files in the repo
        if file_path.is_file() and file_path.suffix in extensions: # if the file is a file and has a valid extension
            if not has_copyright_header(file_path): # if the file does not have a copyright header
                files_without_copyright.append(str(file_path)) # add the file to the list of files without copyright
    
    return files_without_copyright

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Check and optionally fix copyright headers in C/C++ source files')
    parser.add_argument('directory', nargs='?', default=os.getcwd(), 
                       help='Directory to scan (default: current directory)')
    parser.add_argument('--fix', action='store_true', 
                       help='Automatically insert copyright headers into files that are missing them')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be fixed without making changes (use with --fix)')
    
    args = parser.parse_args()
    
    repo_dir = args.directory
    
    if not os.path.exists(repo_dir):
        print(f"Error: Directory {repo_dir} does not exist")
        sys.exit(1)
    
    print(f"Scanning repository: {repo_dir}")
    print("Looking for files with extensions: .cpp, .hpp")
    print("Checking for copyright header containing:")
    print("  - SPDX-License-Identifier: MIT")
    print("  - Copyright...Advanced Micro Devices, Inc...All rights reserved")
    print()
    
    files_without_copyright = find_files_without_copyright(repo_dir)
    
    if files_without_copyright:
        print(f"Found {len(files_without_copyright)} files without proper copyright header:")
        print()
        
        if args.fix:
            if args.dry_run:
                print("DRY RUN - Would insert copyright headers into these files:")
                for file_path in sorted(files_without_copyright):
                    print(f"  {file_path}")
            else:
                print("Inserting copyright headers...")
                success_count = 0
                for file_path in sorted(files_without_copyright):
                    if insert_copyright_header(file_path):
                        print(f"  ✓ {file_path}")
                        success_count += 1
                    else:
                        print(f"  ✗ {file_path}")
                
                print(f"\nSuccessfully updated {success_count} out of {len(files_without_copyright)} files")
                
                # Return the number of files that still need fixing
                return len(files_without_copyright) - success_count
        else:
            for file_path in sorted(files_without_copyright):
                print(file_path)
            print(f"\nUse --fix to automatically insert copyright headers")
    else:
        print("All files have proper copyright headers!")
    
    return len(files_without_copyright)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(min(exit_code, 255))  # Limit exit code to valid range 