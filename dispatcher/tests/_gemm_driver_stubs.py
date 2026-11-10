# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The tile_engine GEMM sweep driver (gemm_full_benchmark.main) under stubs.

Imported (never collected: its name has no ``test_`` prefix) by the driver CLI
tests. It owns the fixed command line (gfx950, fp16, rcr), the problems file
and the patches of device and arch resolution; each test passes its own build
and batch-runner stubs. No GPU or compiler needed.
"""

import json
import sys

ARCH, DTYPE, LAYOUT = "gfx950", "fp16", "rcr"


def run_stub_driver(drv, monkeypatch, capsys, tmp_path, problems, *argv, build,
                    run_batch, devices=("0",)):
    """``drv.main()`` on ``problems`` with ``argv`` after the fixed options.

    ``build`` replaces ``setup_multiple_gemm_dispatchers`` and ``run_batch``
    replaces ``_run_batch_on_device`` (called as ``run_batch(device_id, unit,
    args, worker_path, base_env, extra=None, **kw)``); ``--devices`` resolves
    to ``devices``. The CSV goes to ``tmp_path/out.csv``. Returns (exit code,
    captured output); a parser error raises SystemExit as usual.
    """
    probs = tmp_path / "problems.json"
    probs.write_text(json.dumps({"problems": list(problems)}))
    monkeypatch.setattr(sys, "argv", [
        "gemm_full_benchmark.py", "--arch", ARCH, "--dtype", DTYPE,
        "--layout", LAYOUT, "--problems", str(probs),
        "--csv", str(tmp_path / "out.csv"), *argv,
    ])
    monkeypatch.setattr(drv, "resolve_devices", lambda spec: list(devices))
    monkeypatch.setattr(drv, "_resolve_arch", lambda a: ARCH)
    monkeypatch.setattr(drv, "setup_multiple_gemm_dispatchers", build)
    monkeypatch.setattr(drv, "_run_batch_on_device", run_batch)
    rc = drv.main()
    return rc, capsys.readouterr()
