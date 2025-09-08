#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import re
import sys
from pathlib import Path


def get_copyright_header():
    return """Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT"""


def has_shebang(content):
    """Check if file starts with a shebang."""
    return content.startswith("#!")


def extract_shebang(content):
    """Extract shebang line if present."""
    lines = content.split("\n")
    if lines and lines[0].startswith("#!"):
        return lines[0] + "\n"
    return ""


def has_existing_copyright(content):
    """Check if file already has a copyright header."""
    copyright_patterns = [
        r"Copyright.*Advanced Micro Devices",
        r"SPDX-License-Identifier",
        r"Copyright.*AMD",
        r"Copyright.*Advanced Micro Devices, Inc\.",
    ]

    for pattern in copyright_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def find_copyright_section(content):
    """Find the copyright section boundaries - be very conservative."""
    lines = content.split("\n")
    start_idx = -1
    end_idx = -1

    # Look for copyright patterns - only match lines that clearly contain copyright info
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Only match lines that are clearly copyright/SPDX related AND not other important headers
        if re.search(
            r"Copyright.*Advanced Micro Devices|SPDX-License-Identifier",
            stripped,
            re.IGNORECASE,
        ) and not re.search(
            r"#include|#pragma|#ifndef|#define|#endif|# -\*-|coding:",
            stripped,
            re.IGNORECASE,
        ):
            if start_idx == -1:
                start_idx = i
            end_idx = i

    # Be extremely conservative - only remove consecutive copyright lines
    if start_idx != -1:
        # Look backwards only for empty lines or comment lines that are clearly copyright
        for i in range(start_idx - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped == "":
                start_idx = i
            elif (stripped.startswith("//") or stripped.startswith("#")) and re.search(
                r"Copyright|SPDX", stripped, re.IGNORECASE
            ):
                start_idx = i
            else:
                break

        # Look forwards only for empty lines or comment lines that are clearly copyright
        for i in range(end_idx + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped == "":
                end_idx = i
            elif (stripped.startswith("//") or stripped.startswith("#")) and re.search(
                r"Copyright|SPDX", stripped, re.IGNORECASE
            ):
                end_idx = i
            else:
                break

    return start_idx, end_idx


def add_copyright_to_file(file_path, dry_run=False):
    """Add or replace copyright header in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping {file_path} (encoding issue)")
        return False

    original_content = content
    shebang = extract_shebang(content)
    has_shebang_line = bool(shebang)

    # Remove shebang from content for processing
    if has_shebang_line:
        content = content[len(shebang) :]

    # Check if copyright already exists
    if has_existing_copyright(content):
        start_idx, end_idx = find_copyright_section(content)
        if start_idx != -1 and end_idx != -1:
            lines = content.split("\n")
            # Safety check: don't remove more than 20 lines (likely not just copyright)
            if end_idx - start_idx + 1 <= 20:
                # Remove existing copyright section
                new_lines = lines[:start_idx] + lines[end_idx + 1 :]
                content = "\n".join(new_lines)
            else:
                print(
                    f"Warning: Skipping {file_path} - copyright section too large ({end_idx - start_idx + 1} lines)"
                )
                return False

    # Determine comment style
    if file_path.suffix in [".py", ".sh"]:
        comment_prefix = "# "
    elif file_path.suffix in [".cpp", ".hpp", ".c", ".h", ".inc"]:
        comment_prefix = "// "
    else:
        comment_prefix = "# "

    # Create new copyright header
    copyright_lines = get_copyright_header().split("\n")
    copyright_header = "\n".join([comment_prefix + line for line in copyright_lines])

    # Add copyright header
    if content.strip():
        # Add blank line after copyright if content exists
        new_content = copyright_header + "\n\n" + content
    else:
        new_content = copyright_header + "\n"

    # Reconstruct with shebang if present
    if has_shebang_line:
        new_content = shebang + new_content

    # Only write if content changed
    if new_content != original_content:
        if dry_run:
            print(f"Would update: {file_path}")
            return True
        else:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated: {file_path}")
                return True
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False
    else:
        if not dry_run:
            print(f"No changes needed: {file_path}")
        return False


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        sys.argv.remove("--dry-run")

    # Parse max files limit
    max_files = 200  # default
    if "--max-files" in sys.argv:
        try:
            max_idx = sys.argv.index("--max-files")
            if max_idx + 1 < len(sys.argv):
                max_files = int(sys.argv[max_idx + 1])
                sys.argv.remove("--max-files")
                sys.argv.remove(str(max_files))
        except (ValueError, IndexError):
            print("Error: --max-files requires a number")
            return 1

    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path(".")

    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist")
        return 1

    if dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("=" * 50)

    print(f"Maximum files to modify: {max_files}")
    if not dry_run:
        print("Use --dry-run to preview changes first")

    # File extensions to process
    extensions = {".py", ".cpp", ".hpp", ".c", ".h", ".inc", ".sh"}

    files_processed = 0
    files_updated = 0

    # Walk through directory recursively
    for file_path in target_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in extensions:
            files_processed += 1
            if add_copyright_to_file(file_path, dry_run):
                files_updated += 1
                if files_updated >= max_files:
                    print(f"\nWARNING: Reached maximum file limit ({max_files})")
                    print("Use --max-files N to increase the limit")
                    break

    action = "Would update" if dry_run else "Updated"
    print(
        f"\nProcessed {files_processed} files, {action.lower()} {files_updated} files"
    )
    if files_updated >= max_files:
        print(f"Stopped at limit of {max_files} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
