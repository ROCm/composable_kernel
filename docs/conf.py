# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

from rocm_docs import ROCmDocs


name = "Composable Kernel"
get_version = r'sed -n -e "s/^rocm_setup_version(.* \([0-9\.]\{1,\}\).*/\1/p" ../CMakeLists.txt'
version = subprocess.getoutput(get_version)
if len(version) > 0:
    name = f"{name} {version}"

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(f"{name} Documentation")
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/docBin/xml")
docs_core.setup()

mathjax3_config = {
'tex': {
    'macros': {
        'diag': '\\operatorname{diag}',
        }
    }
}

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

extensions += ['sphinxcontrib.bibtex']
bibtex_bibfiles = ['refs.bib']
