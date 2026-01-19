#!/usr/bin/env python3
"""
Execute CK Profiler for each configuration in the JSON file.
"""

import json
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class CKProfilerExecutor:
    """Execute CK Profiler for each configuration."""
    
    def __init__(self, profiler_path: str = './build/bin', dry_run: bool = False):
        self.profiler_path = Path(profiler_path)
        self.dry_run = dry_run
        self.results = []
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def build_command(self, config: Dict) -> List[str]:
        """Build command line from JSON config."""
        op_type = config['operation_type']
        args = config['profiler_args']
        
        # Build argument list
        cmd = [str(self.profiler_path), op_type]
        
        # Add arguments based on operation type
        if op_type == 'grouped_conv_fwd':
            # Forward convolution arguments
            cmd.extend([
                str(args['data_type']),
                str(args['layout']),
                str(args['index_type']),
                str(args['verify']),
                str(args['init']),
                str(args['log']),
                str(args['time']),
                str(args['num_dim_spatial']),
                str(args['G']),
                str(args['N']),
                str(args['K']),
                str(args['C'])
            ])
        else:
            # Backward convolution arguments (no index_type)
            cmd.extend([
                str(args['data_type']),
                str(args['layout']),
                str(args['verify']),
                str(args['init']),
                str(args['log']),
                str(args['time']),
                str(args['num_dim_spatial']),
                str(args['G']),
                str(args['N']),
                str(args['K']),
                str(args['C'])
            ])
        
        # Add spatial parameters (same order for all operation types)
        # filter_spatial, input_spatial, strides, dilations, left_pads, right_pads
        cmd.extend([str(x) for x in args['filter_spatial']])
        cmd.extend([str(x) for x in args['input_spatial']])
        cmd.extend([str(x) for x in args['strides']])
        cmd.extend([str(x) for x in args['dilations']])
        cmd.extend([str(x) for x in args['left_pads']])
        cmd.extend([str(x) for x in args['right_pads']])
        
        # Add split_k for backward ops
        if 'split_k' in args:
            cmd.append(str(args['split_k']))
        
        return cmd
    
    def run_profiler(self, config: Dict, index: int, total: int, verbose: bool = True) -> Dict:
        """Run profiler for a single config."""
        cmd = self.build_command(config)
        metadata = config['metadata']
        
        if verbose:
            print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
            print(f"{Colors.BOLD}[{index}/{total}] {metadata['description']}{Colors.RESET}")
            print(f"{Colors.CYAN}Priority Rank: {metadata['priority_rank']}{Colors.RESET}")
            print(f"{Colors.CYAN}PyTorch Op:    {metadata['pytorch_op']}{Colors.RESET}")
            print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
        
        # Print command
        cmd_str = ' '.join(cmd)
        if self.dry_run:
            print(f"{Colors.YELLOW}[DRY RUN]{Colors.RESET} Would execute:")
            print(f"  {cmd_str}")
            return {
                'success': True,
                'dry_run': True,
                'command': cmd_str
            }
        
        if verbose:
            print(f"{Colors.BLUE}Command:{Colors.RESET} {cmd_str}\n")
        
        # Execute command
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            elapsed_time = time.time() - start_time
            
            success = result.returncode == 0
            
            if verbose:
                if success:
                    print(f"{Colors.GREEN}✓ SUCCESS{Colors.RESET} (completed in {elapsed_time:.2f}s)")
                else:
                    print(f"{Colors.RED}✗ FAILED{Colors.RESET} (return code: {result.returncode})")
                
                # Print stdout if present
                if result.stdout:
                    print(f"\n{Colors.BOLD}Output:{Colors.RESET}")
                    print(result.stdout)
                
                # Print stderr if present and failed
                if result.stderr and not success:
                    print(f"\n{Colors.RED}Error Output:{Colors.RESET}")
                    print(result.stderr)
            
            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed_time': elapsed_time,
                'command': cmd_str
            }
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"{Colors.RED}✗ TIMEOUT{Colors.RESET} after {elapsed_time:.2f}s")
            return {
                'success': False,
                'error': f'Timeout after {elapsed_time:.2f}s',
                'command': cmd_str
            }
        except FileNotFoundError:
            if verbose:
                print(f"{Colors.RED}✗ ERROR{Colors.RESET}: Profiler executable not found")
                print(f"  Looking for: {cmd[0]}")
                print(f"  Please check that CK profilers are built in: {self.profiler_path}")
            return {
                'success': False,
                'error': f'Profiler executable not found: {cmd[0]}',
                'command': cmd_str
            }
        except Exception as e:
            if verbose:
                print(f"{Colors.RED}✗ ERROR{Colors.RESET}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'command': cmd_str
            }
    
    def run_all(self, config_file: str, max_ops: Optional[int] = None, 
                verbose: bool = True, save_results: bool = True) -> List[Dict]:
        """Execute all profiler configurations."""
        # Load configurations
        with open(config_file) as f:
            configs = json.load(f)
        
        total = len(configs)
        if max_ops:
            total = min(max_ops, total)
            configs = configs[:max_ops]
        
        self.stats['total'] = total
        
        print(f"{Colors.BOLD}Executing CK Profiler for {total} configurations...{Colors.RESET}\n")
        
        # Execute each configuration
        for i, config in enumerate(configs, 1):
            result = self.run_profiler(config, i, total, verbose)
            
            # Track result
            self.results.append({
                'config': config,
                'result': result
            })
            
            # Update stats
            if result.get('success'):
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
        
        # Print summary
        self.print_summary()
        
        # Save results if requested
        if save_results and not self.dry_run:
            self.save_results()
        
        return self.results
    
    def print_summary(self):
        """Print execution summary."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}Execution Summary{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"Total configurations: {self.stats['total']}")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Successful:  {self.stats['success']}")
        if self.stats['failed'] > 0:
            print(f"  {Colors.RED}✗{Colors.RESET} Failed:      {self.stats['failed']}")
        
        if self.stats['success'] > 0:
            success_rate = (self.stats['success'] / self.stats['total']) * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")
        
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
    
    def save_results(self, output_file: str = 'profiler_results.json'):
        """Save execution results to JSON file."""
        timestamp = datetime.now().isoformat()
        
        output_data = {
            'timestamp': timestamp,
            'stats': self.stats,
            'results': []
        }
        
        for item in self.results:
            config = item['config']
            result = item['result']
            
            output_data['results'].append({
                'operation': config['operation_type'],
                'description': config['metadata']['description'],
                'priority_rank': config['metadata']['priority_rank'],
                'success': result.get('success', False),
                'returncode': result.get('returncode'),
                'elapsed_time': result.get('elapsed_time'),
                'command': result.get('command'),
                'error': result.get('error'),
                'stdout': result.get('stdout', '')[:500] if result.get('stdout') else None  # Truncate long output
            })
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"{Colors.GREEN}Results saved to: {output_file}{Colors.RESET}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Execute CK Profiler for configurations in JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configurations
  python run_ck_profiler.py ck_profiler_configs.json
  
  # Run first 10 only
  python run_ck_profiler.py ck_profiler_configs.json --max-ops 10
  
  # Dry run (show commands without executing)
  python run_ck_profiler.py ck_profiler_configs.json --dry-run
  
  # Specify profiler path
  python run_ck_profiler.py ck_profiler_configs.json --profiler-path ./build/bin
        """
    )
    
    parser.add_argument(
        'config_file',
        help='CK Profiler configuration JSON file'
    )
    parser.add_argument(
        '--profiler-path',
        default='./build/bin',
        help='Path to CK profiler binaries (default: ./build/bin)'
    )
    parser.add_argument(
        '--max-ops',
        type=int,
        help='Maximum number of operations to execute'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show commands without executing them'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to JSON file'
    )
    parser.add_argument(
        '--output',
        default='profiler_results.json',
        help='Output file for results (default: profiler_results.json)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config_file).exists():
        print(f"{Colors.RED}Error: Config file not found: {args.config_file}{Colors.RESET}")
        sys.exit(1)
    
    # Create executor
    executor = CKProfilerExecutor(
        profiler_path=args.profiler_path,
        dry_run=args.dry_run
    )
    
    # Run profilers
    try:
        executor.run_all(
            args.config_file,
            max_ops=args.max_ops,
            verbose=not args.quiet,
            save_results=not args.no_save
        )
        
        # Exit with error code if any failed
        if executor.stats['failed'] > 0 and not args.dry_run:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
