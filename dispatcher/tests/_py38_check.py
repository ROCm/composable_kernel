# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python 3.8 compatibility check for shipped modules.

Imported (never collected: its name has no ``test_`` prefix) by the adaptive
search tests. CK supports Python 3.8 while CI may run a newer one, so a module
must be checked statically: parsing with ``feature_version=(3, 8)`` alone
accepts runtime PEP 585 aliases such as ``X = dict[str, int]`` and PEP 604
unions such as ``isinstance(x, int | float)``, which fail on 3.8.
"""

import ast
import builtins

_PEP585 = {"tuple", "list", "dict", "set", "frozenset", "type"}
# ``A | B`` with a builtin class on either side is a PEP 604 union (3.10+).
_BUILTIN_TYPES = {n for n, v in vars(builtins).items() if isinstance(v, type)}
_NEW_APIS = {"removeprefix", "removesuffix", "bit_count", "randbytes", "pairwise"}


def py38_problems(source, filename="<src>"):
    """Constructs that fail on Python 3.8: 3.9+ syntax, PEP 585 / 604 type
    expressions evaluated at runtime (annotations are lazy only under
    ``from __future__ import annotations``), and a few 3.9+ APIs."""
    tree = ast.parse(source, filename, feature_version=(3, 8))
    lazy = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in tree.body
    )
    abc_names = {
        a.asname or a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "collections.abc"
        for a in n.names
    }
    skip = set()
    for n in (ast.walk(tree) if lazy else ()):
        ann = getattr(n, "annotation", None) or getattr(n, "returns", None)
        if ann is not None:
            skip.update(id(x) for x in ast.walk(ann))
    out = []
    for n in ast.walk(tree):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Subscript):
            base = n.value
            name = getattr(base, "id", None) or getattr(base, "attr", "")
            if name in _PEP585 or name in abc_names:
                out.append(f"{filename}:{n.lineno} runtime {name}[...]")
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
            sides = (n.left, n.right)
            if any(
                (isinstance(s, ast.Constant) and s.value is None)
                or getattr(s, "id", None) in _BUILTIN_TYPES
                for s in sides
            ):
                out.append(f"{filename}:{n.lineno} runtime X | Y type union")
        elif isinstance(n, ast.Attribute) and n.attr in _NEW_APIS:
            out.append(f"{filename}:{n.lineno} 3.9+ API .{n.attr}")
    return out
