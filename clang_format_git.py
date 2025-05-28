#!/usr/bin/env python3

import subprocess
import sys
import re
import os
from pathlib import Path
from typing import List

# 定义颜色代码
class Color:
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # 无颜色

def print_color(color: str, message: str):
    """打印带颜色的消息"""
    print(f"{color}{message}{Color.NC}")

def check_command_exists(command: str) -> bool:
    """检查命令是否存在"""
    try:
        subprocess.run(['which', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def is_git_repo() -> bool:
    """检查当前目录是否是Git仓库"""
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], 
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_git_modified_files() -> List[str]:
    """获取Git中已修改但未提交的文件及未跟踪的新文件"""
    try:
        # 获取已修改的文件
        modified_result = subprocess.run(['git', 'status', '--porcelain'], 
                                      stdout=subprocess.PIPE, text=True, check=True)
        
        files = []
        for line in modified_result.stdout.splitlines():
            if not line.strip():
                continue
            
            # 检查文件状态（M=修改, A=添加, ??=未跟踪）
            if re.match(r'^\s*[AM]', line) or line.startswith('??'):
                # 提取文件名
                parts = line.strip().split(maxsplit=1)
                if len(parts) > 1:
                    files.append(parts[1])
                else:
                    # 处理未跟踪文件的情况
                    files.append(parts[0][2:].strip())
                    
        return files
    except subprocess.CalledProcessError as e:
        print_color(Color.RED, f"获取Git修改文件失败: {e}")
        return []

def filter_cpp_files(files: List[str]) -> List[str]:
    """筛选C++相关文件"""
    cpp_extensions = ['.cpp', '.hpp', '.h', '.cc', '.c', '.cxx']
    return [file for file in files if Path(file).suffix.lower() in cpp_extensions]

def format_files(files: List[str]) -> tuple:
    """使用clang-format格式化文件"""
    success_count = 0
    fail_count = 0
    
    for file in files:
        print(f"格式化: {file} ... ", end="")
        try:
            # 先运行dos2unix确保文件使用Unix换行符
            subprocess.run(['dos2unix', file], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # 使用clang-format格式化文件
            subprocess.run(['clang-format-12', '-style=file', '-i', file], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            print_color(Color.GREEN, "成功")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print_color(Color.RED, f"失败: {e}")
            fail_count += 1
    
    return success_count, fail_count

def main():
    # 检查clang-format-12是否安装
    if not check_command_exists('clang-format-12'):
        print_color(Color.RED, "错误: clang-format-12 未安装或不在PATH中")
        sys.exit(1)
        
    # 检查dos2unix是否安装
    if not check_command_exists('dos2unix'):
        print_color(Color.YELLOW, "警告: dos2unix 未安装，将跳过行尾符转换")
        
    # 检查是否在Git仓库中
    if not is_git_repo():
        print_color(Color.RED, "错误: 当前目录不是Git仓库")
        sys.exit(1)
    
    print_color(Color.YELLOW, "获取Git修改的文件列表...")
    
    # 获取修改的文件
    all_files = get_git_modified_files()
    
    # 筛选C++文件
    cpp_files = filter_cpp_files(all_files)
    
    if not cpp_files:
        print_color(Color.YELLOW, "没有找到需要格式化的C++文件")
        sys.exit(0)
    
    print_color(Color.GREEN, "找到以下文件需要格式化:")
    for file in cpp_files:
        print(f"  - {file}")
    
    print_color(Color.YELLOW, "开始格式化文件...")
    
    # 格式化文件
    success_count, fail_count = format_files(cpp_files)
    
    print()
    print_color(Color.GREEN, "格式化完成!")
    print_color(Color.GREEN, f"  - 成功: {success_count} 个文件")
    if fail_count > 0:
        print_color(Color.RED, f"  - 失败: {fail_count} 个文件")
    
    print()
    print_color(Color.YELLOW, "提示: 您可以使用 'git diff' 查看格式化后的变更")

if __name__ == "__main__":
    main()
