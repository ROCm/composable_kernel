#!/bin/bash
# Generate Comprehensive Convolution Test Dataset for CK
# This script captures MIOpen commands from PyTorch models and generates test cases

set -e  # Exit on error

echo "=========================================="
echo "CK Convolution Test Dataset Generator"
echo "=========================================="

# Configuration
OUTPUT_DIR="generated_datasets"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Step 1: Capturing MIOpen commands from PyTorch models..."
echo "-----------------------------------------"

# Check if running on GPU
if ! command -v rocm-smi &> /dev/null; then
    echo "WARNING: ROCm not detected. Models will run on CPU (no MIOpen commands)."
    echo "For actual MIOpen commands, run this on a system with AMD GPU."
fi

# Run ResNet model
echo "Running ResNet model..."
MIOPEN_ENABLE_LOGGING_CMD=1 python3 run_model_with_miopen.py \
    --model resnet \
    --batch-size 32 \
    --input-size 224 \
    2> $OUTPUT_DIR/resnet_miopen_log.txt || true

# Run MobileNet model  
echo "Running MobileNet model..."
MIOPEN_ENABLE_LOGGING_CMD=1 python3 run_model_with_miopen.py \
    --model mobilenet \
    --batch-size 16 \
    --input-size 224 \
    2> $OUTPUT_DIR/mobilenet_miopen_log.txt || true

# Run simple conv model with various sizes
echo "Running simple conv model (multiple sizes)..."
for size in 32 64 128 256; do
    echo "  - Input size: ${size}x${size}"
    MIOPEN_ENABLE_LOGGING_CMD=1 python3 run_model_with_miopen.py \
        --model simple \
        --batch-size 8 \
        --input-size $size \
        2>> $OUTPUT_DIR/simple_miopen_log.txt || true
done

echo ""
echo "Step 2: Converting MIOpen commands to CSV..."
echo "-----------------------------------------"

# Convert each log file to CSV
if [ -s "$OUTPUT_DIR/resnet_miopen_log.txt" ]; then
    echo "Converting ResNet commands..."
    python3 miopen_to_csv.py \
        --input $OUTPUT_DIR/resnet_miopen_log.txt \
        --output-2d $OUTPUT_DIR/resnet_cases_2d.csv \
        --output-3d $OUTPUT_DIR/resnet_cases_3d.csv \
        --filter-duplicates || true
fi

if [ -s "$OUTPUT_DIR/mobilenet_miopen_log.txt" ]; then
    echo "Converting MobileNet commands..."
    python3 miopen_to_csv.py \
        --input $OUTPUT_DIR/mobilenet_miopen_log.txt \
        --output-2d $OUTPUT_DIR/mobilenet_cases_2d.csv \
        --output-3d $OUTPUT_DIR/mobilenet_cases_3d.csv \
        --filter-duplicates || true
fi

if [ -s "$OUTPUT_DIR/simple_miopen_log.txt" ]; then
    echo "Converting simple model commands..."
    python3 miopen_to_csv.py \
        --input $OUTPUT_DIR/simple_miopen_log.txt \
        --output-2d $OUTPUT_DIR/simple_cases_2d.csv \
        --output-3d $OUTPUT_DIR/simple_cases_3d.csv \
        --filter-duplicates || true
fi

echo ""
echo "Step 3: Generating systematic test cases..."
echo "-----------------------------------------"

# Generate systematic 3D cases (smaller count for testing)
echo "Generating 100 systematic 3D cases..."
python3 generate_systematic_cases.py \
    --count 100 \
    --ndim 3 \
    --output $OUTPUT_DIR/systematic_cases_3d.csv

# Generate preset architectures (these are mostly 2D)
echo "Generating preset architectures..."
python3 generate_systematic_cases.py \
    --preset resnet \
    --output $OUTPUT_DIR/preset_resnet.csv

python3 generate_systematic_cases.py \
    --preset mobilenet \
    --output $OUTPUT_DIR/preset_mobilenet.csv

# Optional: Generate systematic 2D cases (uncomment for many dataset)
# echo "Generating 5000 systematic 2D cases..."
# python3 generate_systematic_cases.py \
#     --count 5000 \
#     --ndim 2 \
#     --output $OUTPUT_DIR/systematic_cases_2d.csv

echo ""
echo "Step 4: Combining all test cases..."
echo "-----------------------------------------"

# Combine all 2D cases
echo "Combining 2D test cases..."
{
    echo "# Combined 2D Convolution Test Cases"
    echo "# Generated on $TIMESTAMP"
    
    # Add header from first available 2D file (skip comment lines)
    HEADER_WRITTEN=false
    for file in $OUTPUT_DIR/*_2d.csv $OUTPUT_DIR/preset_*.csv; do
        if [ -f "$file" ] && [ "$HEADER_WRITTEN" = false ]; then
            grep -v "^#" "$file" | head -n 1
            HEADER_WRITTEN=true
            break
        fi
    done
    
    # Combine all 2D CSV files from output directory (skip headers and comments)
    for file in $OUTPUT_DIR/*_2d.csv $OUTPUT_DIR/preset_*.csv; do
        if [ -f "$file" ]; then
            echo "  Adding data from: $(basename $file)" >&2
            grep -v "^#" "$file" | tail -n +2 || true
        fi
    done
} > $OUTPUT_DIR/combined_test_cases_2d.csv

# Count final 2D cases
COUNT_2D=$(tail -n +2 $OUTPUT_DIR/combined_test_cases_2d.csv | grep -v "^#" | wc -l)
echo "Combined 2D dataset: $COUNT_2D test cases"

# Combine all 3D cases
echo "Combining 3D test cases..."

# Check if any 3D files exist in output directory
THREE_D_FILES=$(find $OUTPUT_DIR -name "*_3d.csv" 2>/dev/null | wc -l)

if [ $THREE_D_FILES -gt 0 ]; then
    echo "Found 3D data sources to combine..."
    {
        echo "# Combined 3D Convolution Test Cases"
        echo "# Generated on $TIMESTAMP"
        
        # Add header from first available 3D file (skip comment lines)
        HEADER_WRITTEN=false
        for file in $OUTPUT_DIR/*_3d.csv; do
            if [ -f "$file" ] && [ "$HEADER_WRITTEN" = false ]; then
                grep -v "^#" "$file" | head -n 1
                HEADER_WRITTEN=true
                break
            fi
        done
        
        # Combine generated 3D files from output directory
        for file in $OUTPUT_DIR/*_3d.csv; do
            if [ -f "$file" ]; then
                echo "  Adding data from: $(basename $file)" >&2
                grep -v "^#" "$file" | tail -n +2 || true
            fi
        done
        
    } > $OUTPUT_DIR/combined_test_cases_3d.csv
    
    # Count final 3D cases
    COUNT_3D=$(tail -n +2 $OUTPUT_DIR/combined_test_cases_3d.csv | grep -v "^#" | wc -l)
    echo "Combined 3D dataset: $COUNT_3D test cases"
else
    echo "No 3D test data found - skipping 3D combination"
    echo "To generate 3D data, uncomment Step 3 systematic generation in this script"
fi

echo ""
echo "Step 5: Creating final production datasets..."
echo "-----------------------------------------"

# Copy combined files to final names
cp $OUTPUT_DIR/combined_test_cases_2d.csv conv_test_set_2d_dataset.csv
if [ -f "$OUTPUT_DIR/combined_test_cases_3d.csv" ]; then
    cp $OUTPUT_DIR/combined_test_cases_3d.csv conv_test_set_3d_dataset.csv
fi

# Count test cases
COUNT_2D=$(tail -n +4 conv_test_set_2d_dataset.csv | wc -l)
if [ -f "conv_test_set_3d_dataset.csv" ]; then
    COUNT_3D=$(tail -n +4 conv_test_set_3d_dataset.csv | wc -l)
else
    COUNT_3D=0
fi

echo ""
echo "=========================================="
echo "Dataset Generation Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - conv_test_set_2d_dataset.csv: $COUNT_2D test cases"
if [ $COUNT_3D -gt 0 ]; then
    echo "  - conv_test_set_3d_dataset.csv: $COUNT_3D test cases"
fi
echo "  - Intermediate files in: $OUTPUT_DIR/"
echo ""
echo "To use these datasets:"
echo "  1. Build the test: cd ../script && make -j64 test_grouped_convnd_fwd_dataset_xdl"
echo "  2. Run the test: ./bin/test_grouped_convnd_fwd_dataset_xdl"
echo ""
echo "For production (10,000+ cases), edit this script to increase --count parameter"