#!/bin/bash

OUTPUT_DIR="generated_datasets"

echo "Testing CSV parsing..."
echo "====================="

# Read first 5 lines of 2D configurations from CSV (skip comments and header)
head -8 $OUTPUT_DIR/model_configs_2d.csv | while IFS=',' read -r config_name model batch_size channels height width device precision output_mode; do
    # Skip comments and empty lines
    [[ "$config_name" =~ ^#.*$ ]] && continue
    [[ "$config_name" == "config_name" ]] && continue  # Skip header
    [[ -z "$config_name" ]] && continue
    
    # Build configuration command
    CONFIG="--model $model --batch-size $batch_size --channels $channels --height $height --width $width --device $device --precision $precision --$output_mode"
    CONFIG_NAME="$config_name"
    
    echo "Config: $CONFIG_NAME"
    echo "Command: $CONFIG"
    echo "---"
done