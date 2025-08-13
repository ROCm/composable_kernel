#!/usr/bin/env python3
"""
Generate Model Configuration Combinations for MIOpen Testing

This script generates all possible combinations of model parameters
and saves them as CSV files that can be read by the shell script.
"""

import csv
import itertools
import argparse

def generate_2d_configs():
    """Generate all 2D model configuration combinations"""
    
    # Define parameter ranges
    models_2d = [
        'resnet18', 'resnet34', 'resnet50', 
        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
        'vgg11', 'vgg16', 'vgg19',
        'alexnet', 'googlenet',
        'densenet121', 'densenet161',
        'squeezenet1_0', 'squeezenet1_1',
        'shufflenet_v2_x1_0'
    ]
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    # Input dimensions: (height, width)
    input_dims = [
        (64, 64), (128, 128), (224, 224), (256, 256), (512, 512),  # Square
        (224, 320), (224, 448), (320, 224), (448, 224),            # Rectangular
        (227, 227),  # AlexNet preferred
        (299, 299)   # Inception preferred
    ]
    
    precisions = ['fp32'] #, 'fp16', 'bf16']
    channels = [3]  # Most models expect RGB
    
    configs = []
    config_id = 1
    
    # Generate all combinations (but limit to reasonable subset)
    for model in models_2d:
        for batch_size in batch_sizes:
            for height, width in input_dims:
                for precision in precisions:
                    # Skip some combinations to keep dataset manageable
                    if batch_size > 16 and height > 256:
                        continue  # Skip large batch + large image combinations
                    if precision != 'fp32' and batch_size < 8:
                        continue  # Skip mixed precision with tiny batches
                    
                    config_name = f"{model}_b{batch_size}_{height}x{width}_{precision}"
                    
                    config = {
                        'config_name': config_name,
                        'model': model,
                        'batch_size': batch_size,
                        'channels': channels[0],
                        'height': height,
                        'width': width,
                        'precision': precision
                    }
                    
                    configs.append(config)
                    config_id += 1
    
    return configs

def generate_3d_configs():
    """Generate all 3D model configuration combinations"""
    
    models_3d = ['r3d_18', 'mc3_18', 'r2plus1d_18']
    
    batch_sizes = [1, 2, 4, 8]  # 3D models are more memory intensive
    temporal_sizes = [8, 16, 32]
    
    # 3D input dimensions: (height, width) 
    input_dims = [
        (112, 112), (224, 224), (256, 256),  # Standard sizes
        (224, 320), (320, 224)               # Rectangular
    ]
    
    precisions = ['fp32'] #, 'fp16']  # Skip bf16 for 3D to reduce combinations
    channels = [3]
    
    configs = []
    
    for model in models_3d:
        for batch_size in batch_sizes:
            for temporal_size in temporal_sizes:
                for height, width in input_dims:
                    for precision in precisions:
                        # Skip very large combinations
                        if batch_size > 4 and temporal_size > 16:
                            continue
                        if batch_size > 2 and height > 224:
                            continue
                            
                        config_name = f"{model}_b{batch_size}_t{temporal_size}_{height}x{width}_{precision}"
                        
                        config = {
                            'config_name': config_name,
                            'model': model,
                            'batch_size': batch_size,
                            'channels': channels[0],
                            'temporal_size': temporal_size,
                            'height': height,
                            'width': width,
                                'precision': precision
                            }
                        
                        configs.append(config)
    
    return configs

def save_configs_to_csv(configs, filename, config_type):
    """Save configurations to CSV file"""
    
    if not configs:
        print(f"No {config_type} configurations generated")
        return
    
    fieldnames = list(configs[0].keys())
    
    with open(filename, 'w', newline='\n', encoding='utf-8') as csvfile:
        csvfile.write(f"# {config_type} Model Configurations\n")
        csvfile.write(f"# Generated {len(configs)} configurations\n")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        
        for config in configs:
            writer.writerow(config)
    
    print(f"Generated {len(configs)} {config_type} configurations → {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate model configuration combinations')
    parser.add_argument('--output-2d', type=str, default='model_configs_2d.csv',
                       help='Output file for 2D configurations')
    parser.add_argument('--output-3d', type=str, default='model_configs_3d.csv', 
                       help='Output file for 3D configurations')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of configurations per type (for testing)')
    
    args = parser.parse_args()
    
    print("Generating 2D model configurations...")
    configs_2d = generate_2d_configs()
    if args.limit:
        configs_2d = configs_2d[:args.limit]
    save_configs_to_csv(configs_2d, args.output_2d, "2D")
    
    print("Generating 3D model configurations...")
    configs_3d = generate_3d_configs()
    if args.limit:
        configs_3d = configs_3d[:args.limit]
    save_configs_to_csv(configs_3d, args.output_3d, "3D")
    
    print(f"\nTotal configurations: {len(configs_2d)} 2D + {len(configs_3d)} 3D = {len(configs_2d) + len(configs_3d)}")
    print("\nTo use these configurations:")
    print("  Update generate_test_dataset.sh to read from these CSV files")

if __name__ == "__main__":
    main()