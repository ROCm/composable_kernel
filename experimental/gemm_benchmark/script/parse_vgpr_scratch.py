import re
import sys


def extract_and_calculate_max(file_path):
    """
    Extract NumVgprs and ScratchSize values from a .s file, and calculate their maximum values
    Parameters:
        file_path (str): Path to the .s file
    Returns:
        tuple: (max_vgprs, max_scratch, vgpr_count, scratch_count)
               Returns (None, None, 0, 0) if no values are found or error occurs
    """
    # Initialize lists to store values
    num_vgprs_list = []
    scratch_size_list = []

    # Define regular expressions for matching
    # Pattern explanation: Match lines starting with optional whitespace, followed by keyword, colon, and numbers
    vgpr_pattern = re.compile(r"\s*NumVgprs:\s*(\d+)")
    scratch_pattern = re.compile(r"\s*ScratchSize:\s*(\d+)")

    try:
        # Open and read the file with encoding handling to avoid errors
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Match NumVgprs
                vgpr_match = vgpr_pattern.search(line)
                if vgpr_match:
                    num = int(vgpr_match.group(1))
                    num_vgprs_list.append(num)

                # Match ScratchSize
                scratch_match = scratch_pattern.search(line)
                if scratch_match:
                    num = int(scratch_match.group(1))
                    scratch_size_list.append(num)

        # Calculate maximum values
        max_vgprs = max(num_vgprs_list) if num_vgprs_list else None
        max_scratch = max(scratch_size_list) if scratch_size_list else None
        return max_vgprs, max_scratch, len(num_vgprs_list), len(scratch_size_list)

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}", file=sys.stderr)
        return None, None, 0, 0


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python parse_vgpr_scratch.py <file_pattern>")
        print("Example: python parse_vgpr_scratch.py *.s")
        sys.exit(1)

    # Get file pattern from command line argument
    # Windows
    # file_pattern = sys.argv[1]
    # glob.glob(file_pattern)
    # Get all files matching the pattern in current directory
    # Linux
    file_list = sys.argv[2:]

    if not file_list:
        print("No files found in the list", file=sys.stderr)
        sys.exit(1)

    # Process each file and output results in single line
    for file_path in file_list:
        max_vgprs, max_scratch, vgpr_count, scratch_count = extract_and_calculate_max(
            file_path
        )
        # Format output values (display 'N/A' if no value found)
        vgpr_output = max_vgprs if max_vgprs is not None else "N/A"
        scratch_output = max_scratch if max_scratch is not None else "N/A"
        # Single line output with file name
        print(
            f"{file_path}: NumVgprs_Max={vgpr_output}, ScratchSize_Max={scratch_output}"
        )
