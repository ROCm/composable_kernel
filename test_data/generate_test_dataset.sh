#!/bin/bash
# Generate Comprehensive Convolution Test Dataset for CK
# This script captures MIOpen commands from PyTorch models and generates test cases

set -e  # Exit on error

# Check if target files already exist
# if [ -f "conv_test_set_2d_dataset.csv" ] && [ -f "conv_test_set_3d_dataset.csv" ]; then
#     echo "Target files already exist:"
#     [ -f "conv_test_set_2d_dataset.csv" ] && echo "  - conv_test_set_2d_dataset.csv ($(wc -l < conv_test_set_2d_dataset.csv) lines)"
#     [ -f "conv_test_set_3d_dataset.csv" ] && echo "  - conv_test_set_3d_dataset.csv ($(wc -l < conv_test_set_3d_dataset.csv) lines)"
#     echo ""
#     echo "To regenerate, please remove these files first:"
#     echo "  rm conv_test_set_2d_dataset.csv conv_test_set_3d_dataset.csv"
#     exit 0
# fi

echo "=========================================="
echo "CK Convolution Test Dataset Generator"
echo "=========================================="

# Configuration
OUTPUT_DIR="generated_datasets"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAX_ITERATIONS=0  # Maximum number of iterations per model type (set to 0 for unlimited)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

echo ""
echo "Step 1: Generating model configurations"
echo "-----------------------------------------"

# Generate model configuration files (with limit for testing)
echo "Generating model configuration files..."
python3 generate_model_configs.py \
    --output-2d $OUTPUT_DIR/model_configs_2d.csv \
    --output-3d $OUTPUT_DIR/model_configs_3d.csv 

if [ ! -f "$OUTPUT_DIR/model_configs_2d.csv" ] || [ ! -f "$OUTPUT_DIR/model_configs_3d.csv" ]; then
    echo "ERROR: Failed to generate configuration files"
    exit 1
fi


# Check if running on GPU
if ! command -v rocm-smi &> /dev/null; then
    echo "WARNING: ROCm not detected. Models will run on CPU (no MIOpen commands)."
    echo "For actual MIOpen commands, run this on a system with AMD GPU."
fi


echo ""
echo "Step 2: Running 2D/3D models and capturing MIOpen commands"
echo "-----------------------------------------"


# Process 2D models from CSV configuration file
echo "Processing 2D models from $OUTPUT_DIR/model_configs_2d.csv..."

# Count total configurations (excluding comments and header)
TOTAL_CONFIGS=$(grep -v "^#" $OUTPUT_DIR/model_configs_2d.csv | tail -n +2 | wc -l)
CURRENT_CONFIG=0

echo "Total configurations to process: $TOTAL_CONFIGS"
echo ""

# Read 2D configurations from CSV (skip comments and header)
while IFS=',' read -r config_name model batch_size channels height width precision; do
    # Skip comments and empty lines
    [[ "$config_name" =~ ^#.*$ ]] && continue
    [[ "$config_name" == "config_name" ]] && continue  # Skip header
    [[ -z "$config_name" ]] && continue
    
    # Increment counter
    CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
    
    # Stop after MAX_ITERATIONS if set
    if [ $MAX_ITERATIONS -gt 0 ] && [ $CURRENT_CONFIG -gt $MAX_ITERATIONS ]; then
        echo -e "${RED}Stopping after $MAX_ITERATIONS iterations (testing mode)${NC}"
        break
    fi
    
    # Build configuration command
    CONFIG="--model $model --batch-size $batch_size --channels $channels --height $height --width $width --precision $precision"
    CONFIG_NAME="$config_name"
    
    echo -e "${GREEN}[${CURRENT_CONFIG}/${TOTAL_CONFIGS}]${NC} ${PURPLE}Running MIOpenDriver${NC} ${CYAN}2D${NC} ${YELLOW}$CONFIG_NAME${NC}: ${BLUE}$CONFIG${NC}"
    
    # Actual run with logging
    MIOPEN_ENABLE_LOGGING_CMD=1 python3 run_model_with_miopen.py \
        --model $model --batch-size $batch_size --channels $channels --height $height --width $width --precision $precision \
        2>> $OUTPUT_DIR/${model}_miopen_log_2d.txt || true 


done < $OUTPUT_DIR/model_configs_2d.csv

# Process 3D models from CSV configuration file
echo "Processing 3D models from $OUTPUT_DIR/model_configs_3d.csv..."

# Count total 3D configurations (excluding comments and header)
TOTAL_3D_CONFIGS=$(grep -v "^#" $OUTPUT_DIR/model_configs_3d.csv | tail -n +2 | wc -l)
CURRENT_3D_CONFIG=0

echo "Total 3D configurations to process: $TOTAL_3D_CONFIGS"
echo ""

# Read 3D configurations from CSV (skip comments and header)
while IFS=',' read -r config_name model batch_size channels temporal_size height width precision; do
    # Skip comments and empty lines  
    [[ "$config_name" =~ ^#.*$ ]] && continue
    [[ "$config_name" == "config_name" ]] && continue  # Skip header
    [[ -z "$config_name" ]] && continue
    
    # Increment counter
    CURRENT_3D_CONFIG=$((CURRENT_3D_CONFIG + 1))
    
    # Stop after MAX_ITERATIONS if set
    if [ $MAX_ITERATIONS -gt 0 ] && [ $CURRENT_3D_CONFIG -gt $MAX_ITERATIONS ]; then
        echo -e "${RED}Stopping after $MAX_ITERATIONS iterations (testing mode)${NC}"
        break
    fi

    # Build configuration command for 3D models
    CONFIG="--model $model --batch-size $batch_size --channels $channels --temporal-size $temporal_size --height $height --width $width --precision $precision"
    CONFIG_NAME="$config_name"
    
    echo -e "${GREEN}[${CURRENT_3D_CONFIG}/${TOTAL_3D_CONFIGS}]${NC} ${PURPLE}Running MIOpenDriver${NC} ${CYAN}3D${NC} ${YELLOW}$CONFIG_NAME${NC}: ${BLUE}$CONFIG${NC}"
    
    
    # Actual run with logging
    MIOPEN_ENABLE_LOGGING_CMD=1 python3 run_model_with_miopen.py \
        --model $model --batch-size $batch_size --channels $channels --temporal-size $temporal_size --height $height --width $width --precision $precision \
        2>> $OUTPUT_DIR/${model}_miopen_log_3d.txt || true

done < $OUTPUT_DIR/model_configs_3d.csv


echo ""
echo "Step 3: Converting MIOpen commands to CSV test cases"
echo "-----------------------------------------"

# Convert 2D MIOpen logs to CSV
echo "Converting 2D MIOpen logs to CSV..."
for log_file in $OUTPUT_DIR/*_miopen_log_2d.txt; do
    if [ -f "$log_file" ]; then
        # Extract model name from filename (e.g., resnet_miopen_log_2d.txt -> resnet)
        base_name=$(basename "$log_file" _miopen_log_2d.txt)
        output_csv="$OUTPUT_DIR/${base_name}_cases_2d.csv"
        
        echo "  Converting $log_file -> $output_csv"
        python3 miopen_to_csv.py \
            --input "$log_file" \
            --output-2d "$output_csv" \
            --model-name "$base_name" \
            --filter-duplicates || true
    fi
done

# Convert 3D MIOpen logs to CSV
echo "Converting 3D MIOpen logs to CSV..."
for log_file in $OUTPUT_DIR/*_miopen_log_3d.txt; do
    if [ -f "$log_file" ]; then
        # Extract model name from filename (e.g., resnet3d_18_miopen_log_3d.txt -> resnet3d_18)
        base_name=$(basename "$log_file" _miopen_log_3d.txt)
        output_csv="$OUTPUT_DIR/${base_name}_cases_3d.csv"
        
        echo "  Converting $log_file -> $output_csv"
        python3 miopen_to_csv.py \
            --input "$log_file" \
            --output-3d "$output_csv" \
            --model-name "$base_name" \
            --filter-duplicates || true
    fi
done

echo ""
echo "Step 4: Combining CSV files into final datasets"
echo "-----------------------------------------"

# Combine all 2D CSV files into one
echo "Combining all 2D test cases..."
# First create empty file with comment headers
echo "# 2D Convolution Test Cases" > conv_test_set_2d_dataset.csv
echo "# Combined from multiple models" >> conv_test_set_2d_dataset.csv
# Add header from first file as a comment
first_2d_file=$(ls $OUTPUT_DIR/*_cases_2d.csv 2>/dev/null | head -1)
if [ -f "$first_2d_file" ]; then
    # Get the CSV header line and prefix with #
    header_line=$(grep "^NDim," "$first_2d_file" | head -1)
    if [ ! -z "$header_line" ]; then
        echo "# $header_line" >> conv_test_set_2d_dataset.csv
    fi
fi
# Append all data rows (skip comment lines and CSV header) from all files
for csv_file in $OUTPUT_DIR/*_cases_2d.csv; do
    if [ -f "$csv_file" ]; then
        # Skip lines starting with # and the NDim header line
        grep -v "^#" "$csv_file" | grep -v "^NDim," >> conv_test_set_2d_dataset.csv 2>/dev/null || true
    fi
done

# Combine all 3D CSV files into one
echo "Combining all 3D test cases..."
# First create empty file with comment headers
echo "# 3D Convolution Test Cases" > conv_test_set_3d_dataset.csv
echo "# Combined from multiple models" >> conv_test_set_3d_dataset.csv
# Add header from first file as a comment
first_3d_file=$(ls $OUTPUT_DIR/*_cases_3d.csv 2>/dev/null | head -1)
if [ -f "$first_3d_file" ]; then
    # Get the CSV header line and prefix with #
    header_line=$(grep "^NDim," "$first_3d_file" | head -1)
    if [ ! -z "$header_line" ]; then
        echo "# $header_line" >> conv_test_set_3d_dataset.csv
    fi
fi
# Append all data rows (skip comment lines and CSV header) from all files
for csv_file in $OUTPUT_DIR/*_cases_3d.csv; do
    if [ -f "$csv_file" ]; then
        # Skip lines starting with # and the NDim header line
        grep -v "^#" "$csv_file" | grep -v "^NDim," >> conv_test_set_3d_dataset.csv 2>/dev/null || true
    fi
done

# Count test cases
COUNT_2D=0
COUNT_3D=0
if [ -f "conv_test_set_2d_dataset.csv" ]; then
    COUNT_2D=$(grep -v "^#" conv_test_set_2d_dataset.csv | tail -n +2 | wc -l)
fi
if [ -f "conv_test_set_3d_dataset.csv" ]; then
    COUNT_3D=$(grep -v "^#" conv_test_set_3d_dataset.csv | tail -n +2 | wc -l)
fi

echo ""
echo "=========================================="
echo "Dataset Generation Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
if [ $COUNT_2D -gt 0 ]; then
    echo "  - conv_test_set_2d_dataset.csv: $COUNT_2D test cases"
fi
if [ $COUNT_3D -gt 0 ]; then
    echo "  - conv_test_set_3d_dataset.csv: $COUNT_3D test cases"
fi
echo "  - Intermediate files in: $OUTPUT_DIR/"
echo ""
echo "To use these datasets:"
echo "  1. Build the test: cd ../script && make -j64 test_grouped_convnd_fwd_dataset_xdl"
echo "  2. Run the test: ./bin/test_grouped_convnd_fwd_dataset_xdl"
echo ""