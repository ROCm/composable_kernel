# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

from rocm_docs import ROCmDocs

with open("../CMakeLists.txt", encoding="utf-8") as f:
    match = re.search(r".*set\(version ([0-9.]+)[^0-9.]+", f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"Composable Kernel {version_number} Documentation"

# for PDF output on Read the Docs
project = "Composable Kernel Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.setup()

external_projects_current_project = "composable_kernel"

mathjax3_config = {
    "tex": {
        "macros": {
            "diag": "\\operatorname{diag}",
        }
    }
}

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Theme-related settings
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "repository_url": "https://github.com/ROCm/rocm-libraries",
    "path_to_docs": "projects/composablekernel/docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}

extensions += [
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
]

mermaid_output_format = "raw"
bibtex_bibfiles = ["refs.bib"]

cpp_id_attributes = ["__global__", "__device__", "__host__"]
extensions = globals().get("extensions", []) + ["sphinxcontrib.datatemplates"]
