#!/usr/bin/env python3

import subprocess
import sys
from bs4 import BeautifulSoup
import re
import json

def get_mangled_kernel_names_from_html(html_file):
    with open(html_file, 'r') as f:
      html_content = f.read()

      # Parse with BeautifulSoup to find the right script tag
      soup = BeautifulSoup(html_content, 'html.parser')
      script_tags = soup.find_all('script')
      
      for i, script in enumerate(script_tags):
          if script.string:
              # Look for scripts containing our kernel signature strings
              if 'tensor_operation' in script.string and 'device' in script.string:
                  print(f"Found kernel data in script tag {i}")
                  
                  # Now extract the data array from this specific script
                  match = re.search(r'var\s+data\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
                  if match:
                      try:
                          data_str = match.group(1)
                          data = json.loads(data_str)
                          
                          # Extract kernel names
                          kernel_names = [item['name'] for item in data]
                          
                          print(f"Found {len(kernel_names)} kernel names:")
                          for name in kernel_names:
                              print(f"  {name}")
                          
                          return kernel_names
                      except json.JSONDecodeError as e:
                          print(f"Error parsing JSON: {e}")
                          # Continue to next script tag
      
      print("Could not find kernel data in any script tag")
      return []

def de_mangle_name(mangled):
  demangler = '/opt/rocm/llvm/bin/llvm-cxxfilt'

  try:
    result = subprocess.run(
        [demangler],
        input=mangled,
        text=True,
        capture_output=True,
        check=True
    )
    demangled = result.stdout.strip()
    if demangled and demangled != mangled:
        return demangled
  except (FileNotFoundError, subprocess.CalledProcessError) as e:
      print(f"Error using {demangler}: {e}")
      return None

def extract_instance_name(demangled, prefix, prefix_instance):
    if not demangled.startswith(prefix + prefix_instance):
        return None
    
    # Start after the prefix
    start = len(prefix + prefix_instance)
    
    # Track angle bracket depth to find matching closing bracket
    depth = 1  # We already counted the opening '<' from prefix
    i = start
    
    while i < len(demangled) and depth > 0:
        if demangled[i] == '<':
            depth += 1
        elif demangled[i] == '>':
            depth -= 1
        i += 1
    
    if depth == 0:
        # Extract from start to just before the matching '>'
        return prefix_instance + demangled[start:i]
    
    return None

def extract_GridwiseGemmMultiD_xdl_cshuffle_v3(demangled):
    prefix = "void ck::tensor_operation::device::(anonymous namespace)::kernel_grouped_conv_fwd_xdl_cshuffle_v3<"
    prefix_instance = "ck::GridwiseGemmMultiD_xdl_cshuffle_v3<"
    return extract_instance_name(demangled, prefix, prefix_instance)

def extract_GridwiseGemmMultiD_xdl_cshuffle(demangled):
    prefix = "void ck::tensor_operation::device::(anonymous namespace)::kernel_grouped_conv_fwd_xdl_cshuffle<"
    prefix_instance = "ck::GridwiseGemmMultiD_xdl_cshuffle<"
    return extract_instance_name(demangled, prefix, prefix_instance)

if __name__ == "__main__":
    # get_mangled_kernel_names_from_html(sys.argv[1])
    
    # Mangled name is the first argument    
    if len(sys.argv) > 1:
      mangled = sys.argv[1]
      demangled = de_mangle_name(mangled)
      print()
      print("Demangled name:")
      print(demangled)
      v3_instance_name = extract_GridwiseGemmMultiD_xdl_cshuffle_v3(demangled)
      if v3_instance_name:
          print()
          print("Extracted GridwiseGemmMultiD_xdl_cshuffle_v3 instance name:")
          print(v3_instance_name)
      v1_instance_name = extract_GridwiseGemmMultiD_xdl_cshuffle(demangled)
      if v1_instance_name:
          print()
          print("Extracted GridwiseGemmMultiD_xdl_cshuffle instance name:")
          print(v1_instance_name)
    else:
      print("Please provide a mangled name as an argument.")