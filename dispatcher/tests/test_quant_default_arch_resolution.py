# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline defaults must resolve for the build target without losing overrides."""

from dataclasses import asdict
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
OPS = ("aquant", "abquant", "bquant", "rowcolquant", "tensor_quant")
MODULES = {op: importlib.import_module(f"gemm_{op}_utils") for op in OPS}


def capture_build(monkeypatch, module):
    calls = []

    def build(configs, **kwargs):
        calls.append((configs, kwargs["arch"]))
        return [Path(f"{cfg.name}.so") for cfg in configs]

    monkeypatch.setattr(module, "build_dispatchers", build)
    return calls


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("variant", ("fp8", "bf8"))
@pytest.mark.parametrize("arch,k", (("gfx942", 32), ("gfx950", 128), ("gfx1250:xnack-", 128)))
@pytest.mark.parametrize("explicit_build_arch", (False, True))
def test_untargeted_defaults_resolve_at_build(op, variant, arch, k, explicit_build_arch, monkeypatch):
    module = MODULES[op]
    detection = []
    monkeypatch.setattr(module, "_detect_gpu_arch", lambda: detection.append(arch) or arch)
    calls = capture_build(monkeypatch, module)
    config = getattr(module, f"default_{variant}_config")()
    original = asdict(config)
    assert not detection, "Offline naming must not query hardware"

    setup = getattr(module, f"setup_multiple_{op}_dispatchers")
    output = setup([config], gfx_arch=arch if explicit_build_arch else None)
    built, target = calls[0]
    assert len(output) == 1
    assert target == arch
    assert built[0].gfx_arch == arch
    assert built[0].warp_tile_k == k
    assert asdict(config) == original, "Resolving a build must not mutate its naming preview"
    assert detection == ([] if explicit_build_arch else [arch])


# Exercise every public factory, including the preshuffle families and fp4/MX.
FACTORIES = [(op, name) for op, module in MODULES.items()
             for name in vars(module) if name.startswith("default_") and name.endswith("_config")]


@pytest.mark.parametrize("op,name", FACTORIES)
def test_all_untargeted_factories_resolve_or_reject_on_gfx1250(op, name, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    monkeypatch.setattr(module, "_detect_gpu_arch", lambda: "gfx1250")
    config = getattr(module, name)()
    setup = getattr(module, f"setup_multiple_{op}_dispatchers")
    if config.variant_key in ("fp8i4", "bf8i4"):
        with pytest.raises(ValueError, match="packed-int4"):
            setup([config])
        assert calls == []
    elif config.variant_key.startswith("mx_"):
        with pytest.raises(ValueError, match="requires gfx950"):
            setup([config])
        assert calls == []
    else:
        setup([config])
        built = calls[0][0][0]
        assert built.gfx_arch == "gfx1250"
        assert built.warp_tile_k == 128
        assert not getattr(built, "eight_waves", False)


def test_abquant_reselects_pipeline_and_keeps_factory_arguments(monkeypatch):
    module = MODULES["abquant"]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config(bquant_group_n=128)
    assert config.eight_waves  # The offline preview uses gfx950's instance.
    module.setup_multiple_abquant_dispatchers([config], gfx_arch="gfx1250")
    built = calls[0][0][0]
    assert built.pipeline == "compv3"
    assert not built.eight_waves
    assert built.bquant_group_n == 128
    assert built.warp_tile_k == 128


def test_abquant_all_defaults_can_use_detected_arch(monkeypatch):
    module = MODULES["abquant"]
    calls = capture_build(monkeypatch, module)
    monkeypatch.setattr(module, "_detect_gpu_arch", lambda: "gfx1250")
    configs = module.all_default_configs()
    module.setup_multiple_abquant_dispatchers(configs)
    built = calls[0][0]
    assert len(built) == len(configs) == 14
    assert all(config.gfx_arch == "gfx1250" for config in built)
    assert all(config.warp_tile_k == 128 for config in built)
    assert all(not config.eight_waves for config in built)


@pytest.mark.parametrize("op", OPS)
def test_explicit_config_arch_still_rejects_mismatch(op, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    monkeypatch.setattr(module, "_detect_gpu_arch", lambda: "gfx1250")
    config = module.default_fp8_config(gfx_arch="gfx950")
    with pytest.raises(ValueError, match="different architecture"):
        getattr(module, f"setup_multiple_{op}_dispatchers")([config])
    assert calls == []


@pytest.mark.parametrize("op", OPS)
def test_modified_untargeted_config_cannot_lose_changes(op, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config()
    config.tile_m *= 2
    modified = asdict(config)
    with pytest.raises(ValueError, match="modified.*construction"):
        getattr(module, f"setup_multiple_{op}_dispatchers")([config], gfx_arch="gfx1250")
    assert calls == []
    assert asdict(config) == modified


@pytest.mark.parametrize("op", OPS)
def test_explicitly_targeted_custom_fields_are_preserved(op, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config(gfx_arch="gfx1250")
    config.tile_m *= 2
    getattr(module, f"setup_multiple_{op}_dispatchers")([config], gfx_arch="gfx1250")
    assert calls[0][0][0] is config


@pytest.mark.parametrize("op", OPS)
def test_missing_detected_arch_never_uses_preview_target(op, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)

    def no_gpu():
        raise RuntimeError("No supported GPU architecture detected")

    monkeypatch.setattr(module, "_detect_gpu_arch", no_gpu)
    config = module.default_fp8_config()
    with pytest.raises(RuntimeError, match="No supported GPU"):
        getattr(module, f"setup_multiple_{op}_dispatchers")([config])
    assert calls == []


@pytest.mark.parametrize("variant", ("fp8", "bf8"))
@pytest.mark.parametrize("suffix", ("", "_preshufflequant"))
@pytest.mark.parametrize("arch", ("gfx1250", "gfx1250:xnack-"))
def test_aquant_rejects_unsafe_explicit_warp_override(variant, suffix, arch):
    factory = getattr(MODULES["aquant"], f"default_{variant}{suffix}_config")
    with pytest.raises(ValueError, match=r"warp_tile_k in \(64, 128\)"):
        factory(gfx_arch=arch, warp_tile_k=32)
    assert factory(gfx_arch=arch, warp_tile_k=64).warp_tile_k == 64
    assert factory(gfx_arch=arch, warp_tile_k=128).warp_tile_k == 128


def test_aquant_deferred_override_is_preserved_and_validated(monkeypatch):
    module = MODULES["aquant"]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config(warp_tile_k=32, quant_group_n=2)
    with pytest.raises(ValueError, match=r"warp_tile_k in \(64, 128\)"):
        module.setup_multiple_aquant_dispatchers([config], gfx_arch="gfx1250")
    assert calls == []
    module.setup_multiple_aquant_dispatchers([config], gfx_arch="gfx942")
    built = calls[0][0][0]
    assert built.warp_tile_k == 32
    assert built.quant_group_n == 2


def test_aquant_direct_config_and_mutations_are_validated(monkeypatch):
    module = MODULES["aquant"]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config(gfx_arch="gfx1250")
    arguments = {**asdict(config), "warp_tile_k": 32}
    with pytest.raises(ValueError, match=r"warp_tile_k in \(64, 128\)"):
        module.AQuantKernelConfig(**arguments)
    config.warp_tile_k = 32
    with pytest.raises(ValueError, match=r"warp_tile_k in \(64, 128\)"):
        config.to_codegen_config()
    with pytest.raises(ValueError, match=r"warp_tile_k in \(64, 128\)"):
        module.setup_multiple_aquant_dispatchers([config], gfx_arch="gfx1250")
    config.gfx_arch = None
    with pytest.raises(ValueError, match=r"warp_tile_k in \(64, 128\)"):
        module.setup_multiple_aquant_dispatchers([config], gfx_arch="gfx1250")
    assert calls == []


@pytest.mark.parametrize("arch", ("gfx1200", "gfx1201:xnack-", "gfx12500", "gfx1250x"))
def test_aquant_direct_config_rejects_unsupported_arch(arch):
    module = MODULES["aquant"]
    config = module.default_fp8_config(gfx_arch="gfx1250")
    arguments = {**asdict(config), "gfx_arch": arch}
    with pytest.raises(ValueError, match="Unsupported GPU architecture"):
        module.AQuantKernelConfig(**arguments)
    config = module.default_fp8_config(gfx_arch="gfx1250")
    config.gfx_arch = arch
    with pytest.raises(ValueError, match="Unsupported GPU architecture"):
        config.to_codegen_config()


def test_aquant_sweep_rejects_unsafe_explicit_tile(tmp_path):
    import json

    module = MODULES["aquant"]
    sweep = module.default_fp8_config(gfx_arch="gfx950", warp_tile_k=32).to_codegen_config()
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(sweep))
    with pytest.raises(ValueError, match=r"warp_tile_k in \(64, 128\)"):
        module.expand_aquant_sweep(str(path), gfx_arch="gfx1250")


@pytest.mark.parametrize("arch", ("gfx12500", "gfx1250x", "gfx9500", "gfx1200", "gfx90a", ""))
def test_tensor_quant_setup_rejects_invalid_explicit_target(arch, monkeypatch):
    module = MODULES["tensor_quant"]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config()
    with pytest.raises(ValueError, match="Unsupported GPU architecture"):
        module.setup_multiple_tensor_quant_dispatchers([config], gfx_arch=arch)
    assert calls == []


@pytest.mark.parametrize("op", ("abquant", "bquant", "rowcolquant", "tensor_quant"))
@pytest.mark.parametrize("arch", ("gfx1250", "gfx1250:xnack-"))
@pytest.mark.parametrize("mutation", ({"warp_tile_k": 16}, {"warp_tile_m": 32}))
def test_quant_final_config_rejects_unsupported_wmma_fragment(op, arch, mutation, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config(gfx_arch=arch)
    args = {**asdict(config), **mutation}
    with pytest.raises(ValueError, match="16x16 WMMA"):
        type(config)(**args)
    for key, value in mutation.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match="16x16 WMMA"):
        config.to_codegen_config()
    with pytest.raises(ValueError, match="16x16 WMMA"):
        getattr(module, f"setup_multiple_{op}_dispatchers")([config], gfx_arch=arch)
    assert calls == []


@pytest.mark.parametrize("op,variant", [
    ("abquant", "fp8"), ("abquant", "bf8"), ("abquant", "fp4"),
    ("bquant", "fp8"), ("bquant", "bf8"),
    ("rowcolquant", "fp8"), ("rowcolquant", "bf8"),
])
@pytest.mark.parametrize("arch", ("gfx1250", "gfx1250:xnack-"))
def test_other_quant_bridges_cannot_bypass_k32_boundary(op, variant, arch, monkeypatch, tmp_path):
    import json

    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    config = getattr(module, f"default_{variant}_config")(gfx_arch=arch)
    sweep = config.to_codegen_config()
    sweep["tile_configs"][0]["warp_tile_k"] = 32
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(sweep))
    with pytest.raises(ValueError, match="16x16 WMMA"):
        getattr(module, f"expand_{op}_sweep")(str(path), gfx_arch=arch)
    config.warp_tile_k = 32
    with pytest.raises(ValueError, match="16x16 WMMA"):
        config.to_codegen_config()
    setup = getattr(module, f"setup_multiple_{op}_dispatchers")
    with pytest.raises(ValueError, match="16x16 WMMA"):
        setup([config], gfx_arch=arch)
    config.gfx_arch = None
    with pytest.raises(ValueError, match="16x16 WMMA"):
        setup([config], gfx_arch=arch)
    assert calls == []


@pytest.mark.parametrize("variant", ("fp8i4", "bf8i4", "mx_bf16bf16", "mx_bf16bf8", "mx_bf16fp4"))
def test_bquant_mutated_variant_is_rejected_at_both_entry_points(variant, monkeypatch):
    module = MODULES["bquant"]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config(gfx_arch="gfx1250:xnack-")
    config.variant_key = variant
    message = "requires gfx950" if variant.startswith("mx_") else "packed-int4"
    with pytest.raises(ValueError, match=message):
        config.to_codegen_config()
    with pytest.raises(ValueError, match=message):
        module.setup_multiple_bquant_dispatchers([config], gfx_arch="gfx1250:xnack-")
    assert calls == []


@pytest.mark.parametrize("op", ("abquant", "bquant", "rowcolquant", "tensor_quant"))
@pytest.mark.parametrize("arch", ("gfx1200", "gfx1201:xnack-", "gfx12500", "gfx1250x", ""))
def test_final_quant_setup_rejects_unsupported_arch(op, arch, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    config = module.default_fp8_config(gfx_arch="gfx1250")
    config.gfx_arch = arch
    with pytest.raises(ValueError, match="Unsupported GPU architecture"):
        getattr(module, f"setup_multiple_{op}_dispatchers")([config], gfx_arch=arch)
    if arch:
        with pytest.raises(ValueError, match="Unsupported GPU architecture"):
            config.to_codegen_config()
    assert calls == []


@pytest.mark.parametrize("op", ("abquant", "bquant", "rowcolquant", "tensor_quant"))
@pytest.mark.parametrize("k", (64, 128))
def test_final_quant_validation_preserves_supported_explicit_tiles(op, k, monkeypatch):
    module = MODULES[op]
    calls = capture_build(monkeypatch, module)
    config = module.default_bf8_config(gfx_arch="gfx1250:xnack-")
    config.warp_tile_k = k
    assert config.to_codegen_config()["tile_configs"][0]["warp_tile_k"] == k
    getattr(module, f"setup_multiple_{op}_dispatchers")([config], gfx_arch="gfx1250:xnack-")
    assert calls[0][0][0] is config
    assert calls[0][1] == "gfx1250:xnack-"


def test_bquant_mx_accepts_supported_target_with_features(monkeypatch):
    module = MODULES["bquant"]
    calls = capture_build(monkeypatch, module)
    config = module.default_mx_bf16bf16_config(gfx_arch="gfx950:xnack-")
    config.to_codegen_config()
    module.setup_multiple_bquant_dispatchers([config], gfx_arch="gfx950:xnack-")
    assert calls[0][1] == "gfx950:xnack-"
