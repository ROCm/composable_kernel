# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build the bindings with PyTorch and the kernels with CK's FMHA CMake targets."""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, find_packages, setup
from setuptools.command.sdist import sdist

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME
except ImportError as error:
    raise RuntimeError(
        "Install ROCm PyTorch first, then use pip install --no-build-isolation . "
        "or python setup.py install in that environment."
    ) from error


PACKAGE_ROOT = Path(__file__).resolve().parent


def ck_root():
    if os.environ.get("CK_ROOT"):
        root = Path(os.environ["CK_ROOT"]).resolve()
    elif (PACKAGE_ROOT / "ck_sources").is_dir():
        root = PACKAGE_ROOT / "ck_sources"
    else:
        root = PACKAGE_ROOT.parents[3]
    if not (root / "include/ck_tile/core.hpp").is_file():
        raise RuntimeError("CK_ROOT must point to a Composable Kernel source checkout")
    return root


def gpu_targets():
    value = os.environ.get("GPU_ARCHS", "native")
    if value == "native":
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU detected; set GPU_ARCHS for cross-compilation")
        value = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(
            ":"
        )[0]
    targets = list(dict.fromkeys(re.split(r"[;,]", value)))
    if not all(re.fullmatch(r"gfx[0-9a-f]+", target) for target in targets):
        raise RuntimeError("GPU_ARCHS must contain semicolon-separated GPU names, e.g. gfx1250")
    return targets


def update_file(path, content):
    if not path.exists() or path.read_bytes() != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


class FmhaBuildExtension(BuildExtension):
    def build_extensions(self):
        if torch.version.hip is None or ROCM_HOME is None:
            raise RuntimeError("Building ck_tile_fmha requires ROCm PyTorch and the ROCm compiler")
        root = ck_root()
        targets = gpu_targets()
        # PEP 660 gives build_ext a disposable temporary directory. Use the same
        # persistent object cache for pip, setup.py install and explicit rebuilds.
        cache_root = Path(os.environ.get("CK_TILE_FMHA_BUILD_DIR", PACKAGE_ROOT / "build"))
        self.build_temp = str(
            cache_root.resolve() / f"temp.{sysconfig.get_platform()}-{sys.implementation.cache_tag}"
        )
        build_dir = Path(self.build_temp).resolve()
        kernel_dir = build_dir / "kernels"
        compiler = Path(ROCM_HOME) / "llvm/bin/clang++"
        cmake_args = [
            "cmake",
            "-S",
            str(PACKAGE_ROOT / "cmake"),
            "-B",
            str(kernel_dir),
            "-G",
            "Ninja",
            # Discard receipts and kernel filters cached by earlier Python builds.
            "-UFMHA_*_CODEGEN_RECEIPT",
            "-UFMHA_*_INSTANCE_FILTER",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DCMAKE_HIP_COMPILER={compiler}",
            f"-DCMAKE_HIP_ARCHITECTURES={';'.join(targets)}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCK_ROOT={root}",
            f"-DTORCH_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        ] + shlex.split(os.environ.get("CK_TILE_FMHA_CMAKE_ARGS", ""))
        subprocess.run(cmake_args, check=True)
        jobs = str(self.parallel or os.environ.get("MAX_JOBS", min(8, os.cpu_count() or 1)))
        subprocess.run(
            ["cmake", "--build", str(kernel_dir), "--target", "ck_tile_fmha_kernels", "-j", jobs],
            check=True,
        )

        # PyTorch hipifies private build copies. The source checkout stays untouched,
        # and unchanged copies retain their timestamps for Ninja's dependency checks.
        binding_dir = build_dir / "bindings"
        sources = []
        for source in sorted((PACKAGE_ROOT / "csrc").iterdir()):
            destination = (
                binding_dir
                / source.with_suffix(".cu" if source.suffix == ".cpp" else source.suffix).name
            )
            update_file(destination, source.read_bytes())
            if source.suffix == ".cpp":
                sources.append(str(destination))

        definitions = ["-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3"]
        if "gfx1250" in targets:
            definitions.append("-DUSE_NEW_UNIFIED_FRAMEWORK=0")
        definitions.append(
            "-DCK_CMAKE_GPU_TARGET_IDS=" + ",".join("0x" + target[3:] for target in targets)
        )
        try:
            from torch.utils.hipify import __version__ as hipify_version

            if int(hipify_version.split(".")[0]) >= 2:
                definitions.append("-DHIPIFY_V2")
        except ImportError:
            pass
        torch_include = Path(torch.__file__).parent / "include"
        if (torch_include / "ATen/CUDAGeneratorImpl.h").is_file():
            definitions.append("-DOLD_GENERATOR_PATH")
        if any(target.startswith(("gfx11", "gfx12")) for target in targets):
            definitions.append("-DCK_TILE_USE_WMMA=1")
        archive = kernel_dir / "libck_tile_fmha_kernels.a"
        extension = CUDAExtension(
            "ck_tile_fmha._C",
            sources=sources,
            include_dirs=[
                str(binding_dir),
                str(root / "include"),
                str(root / "library/include"),
                str(root / "example/ck_tile/01_fmha"),
                str(kernel_dir / "include"),
            ],
            extra_objects=[str(archive)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"] + definitions,
                "nvcc": ["-O3", "-std=c++20", "-fbracket-depth=1024"]
                + [f"--offload-arch={target}" for target in targets]
                + definitions,
            },
        )
        # Keep the extension object finalized by setuptools (including its
        # editable-install bookkeeping), and replace its build inputs.
        self.extensions[0].__dict__.update(extension.__dict__)
        extension = self.extensions[0]
        # PyTorch's generated Ninja link rule does not track extra_objects as
        # dependencies. A changed CK archive must invalidate the extension too.
        output = Path(self.get_ext_fullpath(extension.name))
        if output.exists() and archive.stat().st_mtime_ns > output.stat().st_mtime_ns:
            output.unlink()
        super().build_extensions()


class FmhaSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        root = ck_root()
        destination = Path(base_dir) / "ck_sources"
        for relative in ["include", "library/include", "example/ck_tile/01_fmha"]:
            shutil.copytree(
                root / relative,
                destination / relative,
                ignore=shutil.ignore_patterns(
                    "python", "build", "__pycache__", ".pytest_cache", "*.pyc"
                ),
                dirs_exist_ok=True,
            )
        shutil.copy2(root / "LICENSE", destination / "LICENSE")


setup(
    name="ck-tile-fmha",
    version="0.1.0",
    description="CK Tile FMHA PyTorch bindings",
    long_description=(PACKAGE_ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["ck_tile_fmha", "ck_tile_fmha.*"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.4"],
    extras_require={"test": ["pytest>=7", "einops"]},
    ext_modules=[Extension("ck_tile_fmha._C", sources=[])],
    cmdclass={"build_ext": FmhaBuildExtension, "sdist": FmhaSdist},
    zip_safe=False,
)
