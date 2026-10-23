# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Keep the gfx1250 sweep, bridge metadata and legacy TE enumeration in agreement."""

import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

CK_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CK_ROOT / 'dispatcher/python'), str(CK_ROOT / 'dispatcher/codegen'),
               str(CK_ROOT / 'tile_engine/ops/gemm')]
from gemm_utils import expand_sweep
from ctypes_utils import validate_kernel_config, _parse_gemm_header_metadata
from unified_gemm_codegen import CKTileKernelGenerator, GemmVariant, KernelNaming, UnifiedGemmCodegen
from gemm_instance_builder import GemmKernelBuilder
from gemm_full_benchmark import _resolve_arch, resolve_configs
from gemm_validation_utils import validate_gemm_preshuffle_warp_tile_combination

CONFIG = CK_ROOT / 'tile_engine/ops/gemm/gemm_preshuffle/configs/default_config_gfx1250.json'
# Only the preshuffle pipelines read shuffle_b_v0-packed B; the compute ones read ordinary B.
PACKED_B = ('preshufflev2', 'preshuffle_tdm')
PIPELINES = {
    'preshufflev2': 'WeightPreshufflePipelineAGmemBGmemCRegV2',
    'comp_tdm': 'GemmPipelineAgBgCrCompTDMV1',
    'comp_tdm_v2': 'GemmPipelineAgBgCrCompTDMV2',
    'preshuffle_tdm': 'WeightPreshufflePipelineAGmemBGmemCRegTDM',
    'comp_async': 'GemmPipelineAgBgCrCompAsync',
}
# One kernel per tile_k; preshufflev2 adds a pad_m variant; comp_async needs 8-bit wtk>=128.
EXPECTED = {d: {p: 4 if p == 'preshufflev2' else 2 for p in PIPELINES
                if d in ('fp16', 'bf16') or p != 'comp_async'}
            for d in ('fp16', 'bf16', 'fp8', 'bf8')}


def configs(dtype='fp16', arch='gfx1250'):
    return expand_sweep(str(CONFIG), arch=arch, dtype=dtype, variant='preshuffle')


@pytest.mark.parametrize('dtype', list(EXPECTED))
def test_complete_sweep_reaches_both_generators(tmp_path, dtype):
    sweep = configs(dtype)
    assert Counter(c.pipeline for c in sweep) == EXPECTED[dtype]
    assert len({c.name for c in sweep}) == sum(EXPECTED[dtype].values())
    legacy = GemmKernelBuilder('gemm_preshuffle', tmp_path / 'te', 'gfx1250', dtype, 'rcr', CONFIG)
    assert len(legacy._get_sampled_kernel_list()) == len(sweep)
    for cfg in sweep:
        path = tmp_path / 'config.json'
        path.write_text(json.dumps(cfg.to_codegen_json()))
        gen = UnifiedGemmCodegen(tmp_path / 'bridge', dtype, 'rcr', 'gfx1250', path)
        kernels = gen._get_configs_for_variant(GemmVariant.PRESHUFFLE)
        assert len(kernels) == 1, cfg.name
        kernel = kernels[0]
        assert KernelNaming.generate(kernel, dtype, 'rcr') == cfg.name
        impl = PIPELINES[cfg.pipeline]
        packed = cfg.pipeline in PACKED_B
        assert kernel.preshuffle == packed
        assert kernel.block_size == 128
        source = CKTileKernelGenerator(dtype, 'rcr').generate(kernel)
        assert f'using GemmPipeline = {impl}<UniversalGemmProblem>;' in source
        assert f'#define GEMM_KEY_PIPELINE "{cfg.pipeline}"' in source
        assert f'#define GEMM_KEY_PRESHUFFLE {int(packed)}' in source
        assert '#define GEMM_KEY_DOUBLE_BUFFER 1' in source
        assert ('TdmEpilogue<EpilogueProblem>' in source) == (cfg.epilogue == 'tdm')
        metadata = _parse_gemm_header_metadata(Path(cfg.name + '.hpp'))
        assert metadata['pipeline'] == cfg.pipeline
        assert metadata['epilogue'] == cfg.epilogue
        assert metadata['tile'] == (cfg.tile_m, cfg.tile_n, cfg.tile_k)
        te_tile = {k: v[0] for k, v in cfg.to_codegen_json()['tile_config'].items()}
        _, te_source = legacy._generate_kernel_instance(te_tile, (
            cfg.pipeline, cfg.epilogue, cfg.scheduler, cfg.pad_m, cfg.pad_n, cfg.pad_k, cfg.persistent))
        assert f'using GemmPipeline = ck_tile::{impl}<UniversalGemmProblem>;' in te_source
        assert f'Preshuffle = {str(packed).lower()};' in te_source
        assert ('TdmEpilogue<EpilogueProblem>' in te_source) == (cfg.epilogue == 'tdm')


@pytest.mark.parametrize('pipeline', PIPELINES)
def test_invalid_scheduler_and_epilogue_are_rejected(pipeline):
    cfg = next(c for c in configs() if c.pipeline == pipeline).to_ctypes_config()
    assert not validate_kernel_config(replace(cfg, scheduler='interwave')).is_valid
    invalid_epilogue = 'cshuffle' if cfg.epilogue == 'tdm' else 'tdm'
    assert not validate_kernel_config(replace(cfg, epilogue=invalid_epilogue)).is_valid


@pytest.mark.parametrize('pipeline', list(PIPELINES)[1:])
def test_new_pipelines_reject_other_architectures_and_tiles(pipeline):
    cfg = next(c for c in configs() if c.pipeline == pipeline).to_ctypes_config()
    for updates in ({'gfx_arch': 'gfx950'}, {'warp_k': 16}, {'dtype_a': 'fp8', 'dtype_b': 'fp8'},
                    {'wave_k': 2}, {'layout_b': 'row'}):
        assert not validate_kernel_config(replace(cfg, **updates)).is_valid
    assert validate_kernel_config(replace(cfg, gfx_arch='gfx1250:xnack-')).is_valid


@pytest.mark.parametrize('pipeline', PIPELINES)
def test_ctypes_validation_applies_codegen_pad_rules(pipeline):
    sweep = [c.to_ctypes_config() for c in configs() if c.pipeline == pipeline]
    assert all(validate_kernel_config(c).is_valid for c in sweep)
    # comp_async needs every pad; the others reject pad_n/pad_k on gfx1250.
    bad = dict.fromkeys(('pad_m', 'pad_n', 'pad_k'), pipeline != 'comp_async')
    assert not validate_kernel_config(replace(sweep[0], **bad)).is_valid


def test_tdm_v2_requires_four_waves():
    cfg = next(c for c in configs() if c.pipeline == 'comp_tdm_v2').to_ctypes_config()
    assert not validate_kernel_config(replace(cfg, wave_n=2)).is_valid


def test_gfx1250_feature_suffix_does_not_disable_warp_validation():
    for target in ('gfx1250', 'gfx1250:xnack-'):
        assert validate_gemm_preshuffle_warp_tile_combination(16, 16, 32, 'fp16', 'fp16', 'fp16', target)[0]
        assert not validate_gemm_preshuffle_warp_tile_combination(32, 32, 16, 'fp16', 'fp16', 'fp16', target)[0]


def test_arch_default_and_explicit_config_selection():
    args = SimpleNamespace(configs=[], variant='gemm_preshuffle', dtype='fp16')
    assert resolve_configs(args, _resolve_arch('gfx1250:xnack-')) == [str(CONFIG)]
    args.configs = ['custom.json']
    assert resolve_configs(args, 'gfx1250') == ['custom.json']
    args.configs = []
    assert Path(resolve_configs(args, 'gfx950')[0]).name == 'default_ci_config.json'


@pytest.mark.parametrize('pipeline', ['comp_tdm', 'comp_tdm_v2', 'preshuffle_tdm', 'comp_async'])
def test_legacy_single_instance_cli_keeps_full_pipeline_name(tmp_path, pipeline):
    cfg = next(c for c in configs() if c.pipeline == pipeline)
    trait = '_'.join(map(str, [cfg.pipeline, cfg.epilogue, cfg.scheduler, cfg.pad_m, cfg.pad_n, cfg.pad_k,
                               cfg.persistent]))
    tile = (f'{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}_{cfg.wave_m}x{cfg.wave_n}x{cfg.wave_k}_'
            f'{cfg.warp_tile_m}x{cfg.warp_tile_n}x{cfg.warp_tile_k}')
    script = CK_ROOT / 'tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_instance_builder.py'
    subprocess.run([sys.executable, str(script), '--working_path', str(tmp_path), '--datatype', 'fp16',
                    '--layout', 'rcr', '--config_json', str(CONFIG), '--gen_single', '--kernel_name', 'probe',
                    '--tile_config', tile, '--trait_combo', trait,
                    '--gpu_target', 'gfx1250'], check=True, capture_output=True, text=True)
    headers = list(tmp_path.glob('*.hpp'))
    assert len(headers) == 1
    assert f'ck_tile::{PIPELINES[pipeline]}<UniversalGemmProblem>' in headers[0].read_text()


def test_builtin_codegen_still_has_a_preshuffle_default(tmp_path):
    gen = UnifiedGemmCodegen(tmp_path, 'fp16', 'rcr', 'gfx1250')
    gen.config['tile_config'] = configs()[0].to_codegen_json()['tile_config']
    kernels = gen._get_configs_for_variant(GemmVariant.PRESHUFFLE)
    assert len(kernels) == 4
    assert {c.trait.pipeline for c in kernels} == {'preshufflev2'}


@pytest.mark.parametrize('pipeline', ['comp_tdm', 'comp_tdm_v2', 'comp_async'])
def test_compute_pipelines_reject_persistent_launch(pipeline):
    cfg = next(c for c in configs() if c.pipeline == pipeline)
    assert not validate_kernel_config(replace(cfg, persistent=True).to_ctypes_config()).is_valid


@pytest.mark.skipif(not (shutil.which('cmake') and shutil.which('c++')),
                    reason='requires CMake and a host C++ compiler')
@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1250:xnack-'])
def test_legacy_cmake_default_creates_all_pipeline_targets(tmp_path, arch):
    op = CK_ROOT / 'tile_engine/ops/gemm/gemm_preshuffle'
    build = tmp_path / 'build'
    cmake = (
        'cmake_minimum_required(VERSION 3.16)\n'
        'project(preshuffle_cmake_probe LANGUAGES CXX)\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'set(SUPPORTED_GPU_TARGETS "{arch}")\n'
        'set(GEMM_PRESHUFFLE_DATATYPE "fp16;bf16" CACHE STRING "")\n'
        f'add_subdirectory("{op}" preshuffle)\n'
    )
    for pipeline in PIPELINES:
        cmake += (
            f'get_target_property(deps benchmark_gemm_preshuffle_{pipeline}_pipeline '
            'MANUALLY_ADDED_DEPENDENCIES)\n'
            f'file(WRITE "${{CMAKE_BINARY_DIR}}/{pipeline}.txt" "${{deps}}")\n'
        )
    (tmp_path / 'CMakeLists.txt').write_text(cmake)
    result = subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    for pipeline in PIPELINES:
        targets = (build / f'{pipeline}.txt').read_text().split(';')
        assert len(targets) == 2 * EXPECTED['fp16'][pipeline]
        assert all(f'_{pipeline}_' in target for target in targets)
