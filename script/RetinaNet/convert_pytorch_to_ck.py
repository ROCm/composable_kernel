#!/usr/bin/env python3
"""
Convert PyTorch convolution operations JSON to CK Profiler configuration JSON.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Configuration constants
DATA_TYPE = 1           # FP16
LAYOUT = 3              # NGCHW_GKCYX_NGKHW
INDEX_TYPE = 1          # 64-bit (only for forward)
VERIFY = 0              # No verification
INIT = 1                # Integer initialization
LOG = 0                 # No log printing
TIME = 1                # Time kernels
SPLIT_K = "all"         # For backward ops


class PyTorchToCKConverter:
    """Convert PyTorch convolution operations to CK Profiler format."""
    
    def __init__(self):
        self.converted_ops = []
        self.skipped_ops = []
        self.stats = {
            'total': 0,
            'forward': 0,
            'backward_data': 0,
            'backward_weight': 0,
            'skipped': 0
        }
    
    def compute_output_spatial_dims(self, input_spatial: List[int], 
                                    filter_spatial: List[int],
                                    strides: List[int], 
                                    paddings: List[int],
                                    dilations: List[int]) -> List[int]:
        """Compute output spatial dimensions."""
        output_spatial = []
        for i in range(len(input_spatial)):
            # Formula: output = floor((input + 2*pad - dilation*(kernel-1) - 1) / stride) + 1
            kernel_eff = (filter_spatial[i] - 1) * dilations[i] + 1
            output = (input_spatial[i] + 2 * paddings[i] - kernel_eff) // strides[i] + 1
            output_spatial.append(output)
        return output_spatial
    
    def extract_forward_conv_params(self, op: Dict) -> Optional[Dict]:
        """Extract parameters for forward convolution (aten::miopen_convolution)."""
        try:
            args = op['replay_ir']['list_pos_args']
            
            # Find arguments by name
            arg_map = {arg['arg_name']: arg for arg in args}
            
            # Extract tensor shapes
            input_shape = arg_map['self']['value']['shape']  # [N, C_total, H, W]
            weight_shape = arg_map['weight']['value']['shape']  # [K_total, C_per_group, Y, X]
            
            # Extract convolution parameters
            groups = arg_map['groups']['value']
            stride = arg_map['stride']['value']
            padding = arg_map['padding']['value']
            dilation = arg_map['dilation']['value']
            
            # Compute per-group channels
            N = input_shape[0]
            C_total = input_shape[1]
            K_total = weight_shape[0]
            
            G = groups
            C = C_total // G  # Per-group input channels
            K = K_total // G  # Per-group output channels
            
            # Spatial dimensions
            num_dim_spatial = len(stride)
            input_spatial = input_shape[2:]
            filter_spatial = weight_shape[2:]
            
            # Convert single values to lists if needed
            if isinstance(padding, int):
                padding = [padding] * num_dim_spatial
            if isinstance(stride, int):
                stride = [stride] * num_dim_spatial
            if isinstance(dilation, int):
                dilation = [dilation] * num_dim_spatial
            
            return {
                'operation_type': 'grouped_conv_fwd',
                'profiler_args': {
                    'data_type': DATA_TYPE,
                    'layout': LAYOUT,
                    'index_type': INDEX_TYPE,
                    'verify': VERIFY,
                    'init': INIT,
                    'log': LOG,
                    'time': TIME,
                    'num_dim_spatial': num_dim_spatial,
                    'G': G,
                    'N': N,
                    'K': K,
                    'C': C,
                    'filter_spatial': filter_spatial,
                    'input_spatial': input_spatial,
                    'strides': stride,
                    'dilations': dilation,
                    'left_pads': padding,
                    'right_pads': padding
                },
                'metadata': {
                    'priority_rank': op['metadata']['priority_rank'],
                    'pytorch_op': op['op_name'],
                    'description': f"Forward conv: G={G}, N={N}, K={K}, C={C}, {filter_spatial} kernel, {input_spatial} input"
                }
            }
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Warning: Failed to extract forward conv params: {e}{Colors.RESET}")
            return None
    
    def extract_backward_conv_params(self, op: Dict) -> Optional[List[Dict]]:
        """Extract parameters for backward convolution (aten::convolution_backward)."""
        try:
            args = op['replay_ir']['list_pos_args']
            
            # Find arguments by name
            arg_map = {arg['arg_name']: arg for arg in args}
            
            # Extract tensor shapes
            grad_output_shape = arg_map['grad_output']['value']['shape']  # [N, K_total, Ho, Wo]
            input_shape = arg_map['input']['value']['shape']  # [N, C_total, Hi, Wi]
            weight_shape = arg_map['weight']['value']['shape']  # [K_total, C_per_group, Y, X]
            
            # Extract convolution parameters
            groups = arg_map['groups']['value']
            stride = arg_map['stride']['value']
            padding = arg_map['padding']['value']
            dilation = arg_map['dilation']['value']
            output_mask = arg_map['output_mask']['value']
            
            # Compute per-group channels
            N = input_shape[0]
            C_total = input_shape[1]
            K_total = weight_shape[0]
            
            G = groups
            C = C_total // G
            K = K_total // G
            
            # Spatial dimensions
            num_dim_spatial = len(stride)
            input_spatial = input_shape[2:]
            filter_spatial = weight_shape[2:]
            output_spatial = grad_output_shape[2:]
            
            # Convert single values to lists if needed
            if isinstance(padding, int):
                padding = [padding] * num_dim_spatial
            if isinstance(stride, int):
                stride = [stride] * num_dim_spatial
            if isinstance(dilation, int):
                dilation = [dilation] * num_dim_spatial
            
            results = []
            
            # Backward data (computing gradient w.r.t. input)
            if output_mask[0]:  # grad_input
                results.append({
                    'operation_type': 'grouped_conv_bwd_data',
                    'profiler_args': {
                        'data_type': DATA_TYPE,
                        'layout': LAYOUT,
                        'verify': VERIFY,
                        'init': INIT,
                        'log': LOG,
                        'time': TIME,
                        'num_dim_spatial': num_dim_spatial,
                        'G': G,
                        'N': N,
                        'K': K,
                        'C': C,
                        'filter_spatial': filter_spatial,
                        'input_spatial': input_spatial,
                        'strides': stride,
                        'dilations': dilation,
                        'left_pads': padding,
                        'right_pads': padding,
                        'split_k': SPLIT_K
                    },
                    'metadata': {
                        'priority_rank': op['metadata']['priority_rank'],
                        'pytorch_op': op['op_name'],
                        'description': f"Backward data: G={G}, N={N}, K={K}, C={C}, {filter_spatial} kernel, {input_spatial} input"
                    }
                })
            
            # Backward weight (computing gradient w.r.t. weight)
            if output_mask[1]:  # grad_weight
                results.append({
                    'operation_type': 'grouped_conv_bwd_weight',
                    'profiler_args': {
                        'data_type': DATA_TYPE,
                        'layout': LAYOUT,
                        'verify': VERIFY,
                        'init': INIT,
                        'log': LOG,
                        'time': TIME,
                        'num_dim_spatial': num_dim_spatial,
                        'G': G,
                        'N': N,
                        'K': K,
                        'C': C,
                        'filter_spatial': filter_spatial,
                        'input_spatial': input_spatial,
                        'strides': stride,
                        'dilations': dilation,
                        'left_pads': padding,
                        'right_pads': padding,
                        'split_k': SPLIT_K
                    },
                    'metadata': {
                        'priority_rank': op['metadata']['priority_rank'],
                        'pytorch_op': op['op_name'],
                        'description': f"Backward weight: G={G}, N={N}, K={K}, C={C}, {filter_spatial} kernel, {input_spatial} input"
                    }
                })
            
            return results if results else None
            
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Warning: Failed to extract backward conv params: {e}{Colors.RESET}")
            return None
    
    def convert_operation(self, op: Dict) -> Optional[List[Dict]]:
        """Convert a single PyTorch operation to CK profiler config(s)."""
        op_name = op['op_name']
        
        if op_name == 'aten::miopen_convolution':
            result = self.extract_forward_conv_params(op)
            if result:
                self.stats['forward'] += 1
                return [result]
        
        elif op_name == 'aten::convolution_backward':
            results = self.extract_backward_conv_params(op)
            if results:
                for result in results:
                    if result['operation_type'] == 'grouped_conv_bwd_data':
                        self.stats['backward_data'] += 1
                    elif result['operation_type'] == 'grouped_conv_bwd_weight':
                        self.stats['backward_weight'] += 1
                return results
        else:
            print(f"{Colors.YELLOW}⚠ Warning: Skipping unsupported operation: {op_name}{Colors.RESET}")
            self.skipped_ops.append({
                'op_name': op_name,
                'priority_rank': op['metadata']['priority_rank']
            })
            self.stats['skipped'] += 1
            return None
        
        return None
    
    def convert_file(self, input_path: str, output_path: str):
        """Convert PyTorch JSON file to CK Profiler JSON file."""
        print(f"{Colors.BOLD}Converting PyTorch convolutions to CK Profiler format...{Colors.RESET}\n")
        
        # Load input JSON
        with open(input_path, 'r') as f:
            pytorch_ops = json.load(f)
        
        self.stats['total'] = len(pytorch_ops)
        
        # Convert each operation
        for i, op in enumerate(pytorch_ops, 1):
            op_name = op['op_name']
            priority = op['metadata']['priority_rank']
            
            results = self.convert_operation(op)
            
            if results:
                self.converted_ops.extend(results)
                for result in results:
                    op_type = result['operation_type'].replace('grouped_conv_', '')
                    print(f"{Colors.GREEN}✓{Colors.RESET} [{i}/{len(pytorch_ops)}] Converted: {op_name} (rank {priority}) → {op_type}")
            else:
                print(f"{Colors.YELLOW}⚠{Colors.RESET} [{i}/{len(pytorch_ops)}] Skipped: {op_name} (rank {priority})")
        
        # Write output JSON
        with open(output_path, 'w') as f:
            json.dump(self.converted_ops, f, indent=2)
        
        # Print summary
        self.print_summary(output_path)
    
    def print_summary(self, output_path: str):
        """Print conversion summary."""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Conversion Summary{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"Total PyTorch operations:     {self.stats['total']}")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Forward convolutions:       {self.stats['forward']}")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Backward data convolutions: {self.stats['backward_data']}")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Backward weight convolutions: {self.stats['backward_weight']}")
        if self.stats['skipped'] > 0:
            print(f"  {Colors.YELLOW}⚠{Colors.RESET} Skipped operations:         {self.stats['skipped']}")
        print(f"\nTotal CK profiler configs:    {len(self.converted_ops)}")
        print(f"Output file:                  {output_path}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert PyTorch convolution JSON to CK Profiler configuration JSON'
    )
    parser.add_argument(
        '--input',
        nargs='?',
        default='build/test-data/conv_repros_ir.json',
        help='Input PyTorch JSON file (default: build/test-data/conv_repros_ir.json)'
    )
    parser.add_argument(
        '--output',
        nargs='?',
        default='ck_profiler_configs.json',
        help='Output CK Profiler JSON file (default: ck_profiler_configs.json)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"{Colors.RED}Error: Input file not found: {args.input}{Colors.RESET}")
        sys.exit(1)
    
    # Run conversion
    converter = PyTorchToCKConverter()
    try:
        converter.convert_file(args.input, args.output)
        print(f"{Colors.GREEN}Conversion completed successfully!{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Error during conversion: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
