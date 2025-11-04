"""
Setup script for CK Tile Dispatcher Python package
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension built with CMake"""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command that runs CMake"""
    
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake configuration
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_PYTHON=ON',
        ]
        
        # Build configuration
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        # Platform-specific settings
        if sys.platform.startswith('win'):
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            build_args += ['--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            build_args += ['--', '-j4']
        
        # Build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Run CMake
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )


# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = ''
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

# Read version
version = '1.0.0'

setup(
    name='ck-tile-dispatcher',
    version=version,
    author='AMD CK Tile Team',
    author_email='',
    description='Python bindings for CK Tile GEMM dispatcher',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ROCm/composable_kernel',
    packages=find_packages(),
    ext_modules=[CMakeExtension('ck_tile_dispatcher._ck_dispatcher_cpp', sourcedir='..')],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=[
        'numpy>=1.19',
    ],
    extras_require={
        'torch': ['torch>=2.0'],
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'viz': [
            'matplotlib>=3.3',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='gpu gemm matrix-multiplication rocm amd composable-kernel',
    project_urls={
        'Documentation': 'https://github.com/ROCm/composable_kernel/tree/main/dispatcher/python',
        'Source': 'https://github.com/ROCm/composable_kernel',
        'Bug Reports': 'https://github.com/ROCm/composable_kernel/issues',
    },
)

