#!/usr/bin/env python3
import os, io
import argparse

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def parse_args():
    parser = argparse.ArgumentParser(description='Parse results from tf benchmark runs')
    parser.add_argument('filename', type=str, help='Log file to prase or directory containing log files')
    args = parser.parse_args()
    files = []
    if os.path.isdir(args.filename):
        all_files = os.listdir(args.filename)
        for name in all_files:
            if not 'log' in name:
                continue
            files.append(os.path.join(args.filename, name))
    else:
        files = [args.filename]
    args.files = files
    return args

def main():
    args = parse_args()
    results = []
    #parse results
    for filename in args.files:
        for line in open(filename):
            if 'Best Perf' in line:
                lst=line.split()
                results.append(print_to_string(lst[8:],lst[2],lst[4]))
    #sort results        
    print(results)
    return 0

if __name__ == '__main__':
    main()