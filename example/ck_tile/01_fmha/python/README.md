# CK Tile FMHA Python

`ck_tile_fmha` provides PyTorch bindings for CK Tile FMHA kernels.
It supports FP16/BF16 fixed-length and variable-length attention, packed inputs,
forward/backward, MHA/GQA/MQA, causal/local masks, ALiBi, dropout, deterministic
backward, and KV-cache operations, subject to CK backend restrictions.

Validated on **gfx942, gfx950 and gfx1250**. Nonzero `softcap`, attention sinks, and unequal Q/V
head dimensions are unsupported. The package can coexist with `flash_attn` and
does not fall back to other backends.

## Build and install

Requires Linux, ROCm PyTorch >= 2.4, a ROCm compiler supporting the target GPU,
Python >= 3.9, CMake >= 3.21, Ninja, setuptools >= 64, and wheel. Install PyTorch
first; `--no-build-isolation` builds against that installation.

From this directory:

```bash
MAX_JOBS=64 python -m pip install --no-build-isolation '.[test]'
```

`GPU_ARCHS` defaults to the detected GPU; set `gfx942`, `gfx950` or `gfx1250` for cross-compilation.
`CK_ROOT` selects an alternate CK checkout. The build uses the C++ example's
architecture-specific kernels with receipt 5, which generates only the variants
needed by the current Python API, including bias and deterministic backward.

## Development and incremental builds

```bash
MAX_JOBS=64 python -m pip install --no-build-isolation -e '.[test]'
# After editing a kernel, code generator, or binding:
MAX_JOBS=64 python setup.py build_ext --inplace
```

Keep `python/build/` to reuse compiled objects, or set `CK_TILE_FMHA_BUILD_DIR`
to relocate the cache. Editable installs pick up Python edits immediately;
restart Python after rebuilding a loaded native extension.

An inference-only build compiles the forward kernels without dropout and no backward ones (`fmha_infer`,
GDPA and the forward of `fmha_func`/`fmha_varlen_func` without dropout; the rest raise
`NotImplementedError`):

```bash
GPU_ARCHS=gfx1250 CK_TILE_FMHA_INFERENCE_ONLY=1 python setup.py build_ext --inplace
```

## Use

```python
import torch
from ck_tile_fmha import fmha_func

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = fmha_func(q, k, v, causal=True)
out.sum().backward()
```

Import FMHA functions from `ck_tile_fmha`. ROCm PyTorch uses the `cuda` device name.

Fixed-length `fmha_func`, `fmha_kvpacked_func`, and
`fmha_qkvpacked_func` additionally accept an optional trailing `bias` argument:

```python
bias = torch.randn(128, 128, device="cuda", dtype=q.dtype, requires_grad=True)
out = fmha_func(q, k, v, bias=bias, causal=True)
out.sum().backward()  # Computes gradients for q, k, v, and bias.
```

`bias` has shape `(seqlen_q, seqlen_k)`, matches Q/K/V's device and FP16/BF16 dtype,
and requires a contiguous last dimension. It is shared across batches and heads
and added to `Q @ K.T * softmax_scale` before softmax. It cannot be combined with
`alibi_slopes` and is unavailable in variable-length and KV-cache APIs.

Bias gradients require a `(batch_size, nheads, seqlen_q, seqlen_k)` temporary,
reduced across batches and heads.

## Inference

`fmha_infer` is the forward pass with the least host work per call, for decode and other
calls whose kernel runs in microseconds. It skips the autograd Function, dropout and its RNG
state, allocates only the output (and the LSE when asked), and launches one kernel:

```python
from ck_tile_fmha import fmha_infer

q = torch.randn(64, 1, 8, 128, device="cuda", dtype=torch.bfloat16)   # [b, s_q, h, d]
k = torch.randn(64, 4096, 2, 128, device="cuda", dtype=torch.bfloat16)  # [b, s_k, h_k, d]
v = torch.randn_like(k)
out = fmha_infer(q, k, v, causal=True)
out, lse = fmha_infer(q, k, v, return_lse=True)
```

It takes `causal`, `window_size`, `alibi_slopes` and `bias` as `fmha_func` does, and also the
head-less layout `[b, s, d]` (one head; `k` and `v` may be the same tensor), `out=` for a
preallocated output, and `lse_log2=True` for the LSE in base 2. It is forward only: inputs
that need a gradient raise. FP16/BF16; it works under `torch.compile`, and in plain eager mode
it calls the binding directly.

Timing a call (device time per call, and the host time to issue it):

```python
import time, torch
from ck_tile_fmha import fmha_infer

q = torch.randn(1024, 16, 256, device="cuda", dtype=torch.bfloat16)   # [b, s_q, d], one head
kv = torch.randn(1024, 200, 256, device="cuda", dtype=torch.bfloat16)  # K and V one tensor
f = lambda: fmha_infer(q, kv, kv, return_lse=True, lse_log2=True)
for _ in range(20):
    f()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(200):
    f()
end.record()
torch.cuda.synchronize()
print(f"device {start.elapsed_time(end) / 200 * 1e3:.1f} us per call")
torch.cuda._sleep(100_000_000)  # keep the GPU busy so the loop below measures issuing only
t = time.perf_counter()
for _ in range(200):
    f()
print(f"host {(time.perf_counter() - t) / 200 * 1e6:.1f} us per call")
torch.cuda.synchronize()
```

## Forward-only options

`fmha_func` and `fmha_varlen_func` take two options for inference callers:

- `lse_log2=True` returns the LSE in base 2, log2 of the softmax normalizer (the natural-log
  LSE times log2(e)). The gfx1250 `qr_tdm` kernels store it directly; elsewhere the natural log
  is rescaled after the kernel.
- `broadcast_q=True` (`fmha_varlen_func`, FP16/BF16, no paged KV): `q` is one block of rows that
  every batch attends, `max_seqlen_q` is `q.shape[0]`, and batch `b`'s output is rows
  `b * q.shape[0]` onward of `batch_size * q.shape[0]`. `cu_seqlens_q` only gives the batch
  size; its values are not read.

## Generalized dot-product attention

`fmha_varlen_gdpa_func` computes `out = act(scale * q @ k.T) @ v` per batch and head, with no
softmax, no normalization and no LSE (gfx1250, forward only):

```python
from ck_tile_fmha import fmha_varlen_gdpa_func

out = fmha_varlen_gdpa_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
```

`act` is `fast_gelu`, GELU with the tanh approximation; `P` is rounded to `q`'s dtype before the
second GEMM. `scale` defaults to 1.0, not `1 / sqrt(d)`. Head dims 32, 64, 96, 128, 160 and 256,
FP16/BF16; no mask, ALiBi, dropout, paged KV or FP8. Few keys per batch run on `qr_tdm_pingpong`,
many on `qr_tdm_sched` (d64, d128) or `qr_tdm`.

## Unsupported arguments

Valid arguments that no kernel of the build serves on the GPU at hand (a dtype, head dim, mask,
bias or option without instances there) raise `NotImplementedError`, a `RuntimeError`, so a
caller can fall back to another implementation; malformed arguments raise `RuntimeError`.

## FP8 forward

`fmha_fp8_func` takes prequantized E4M3 Q/K/V and returns BF16 output.
Use `torch.float8_e4m3fn` on gfx1250/gfx950 and `torch.float8_e4m3fnuz` on gfx94x.
Descales are FP32 tensors on the same GPU; dequantization is `input.float() * descale`.

| `qscale` | Q descale | K/V descales |
| --- | --- | --- |
| `no_scale` | None | None |
| `pertensor` (default) | One element | One element each |
| `perhead` | `[batch, nheads]` | `[batch, nheads_kv]` |
| `blockscale` | `[batch, nheads, ceil(seqlen_q / block_size_q)]` | `[batch, nheads_kv, ceil(seqlen_k / block_size_kv)]` |

```python
from ck_tile_fmha import fmha_fp8_func

def quantize_pertensor(x):
    arch = torch.cuda.get_device_properties(x.device).gcnArchName
    dtype = torch.float8_e4m3fnuz if arch.startswith("gfx94") else torch.float8_e4m3fn
    descale = x.float().abs().amax().clamp_min(1e-12) / torch.finfo(dtype).max
    quantized = (x.float() / descale).to(dtype)
    return quantized, descale

q8, qs = quantize_pertensor(q.detach())
k8, ks = quantize_pertensor(k.detach())
v8, vs = quantize_pertensor(v.detach())
out = fmha_fp8_func(q8, k8, v8, qs, ks, vs)
```

Supports MHA/GQA/MQA and causal/local masks. FP8 backward, dropout, bias and ALiBi
are not exposed. `return_lse=True` returns `(out, lse)` with FP32 LSE
`[batch, nheads, seqlen_q]`.

`fmha_varlen_fp8_func` supports `no_scale`, `pertensor` and `perhead` with
packed Q `[total_q, nheads, dim]` and K/V `[total_k, nheads_kv, dim]`:

```python
from ck_tile_fmha import fmha_varlen_fp8_func

out = fmha_varlen_fp8_func(q8, k8, v8, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, qs, ks, vs, qscale="perhead")
```

The cumulative lengths are GPU int32 tensors of shape `[batch + 1]`.
Per-tensor scales cover each entire packed tensor; per-head scales retain the
`[batch, heads]` layout above. Optional LSE is `[nheads, total_q]`.
Varlen blockscale and paged KV FP8 are not exposed.

- **gfx1250:** all four modes and LSE, head dimensions <= 128.
- **gfx950:** no-scale, per-tensor and block-scale, head dimensions <= 256.
  LSE requires no-scale/per-tensor, head dimension 128 and no/causal mask.
  Per-head is unsupported.
- **gfx942:** no-scale, per-tensor and block-scale, head dimensions <= 256.
  Per-head and LSE are unsupported.

Block sizes default to 128 and must satisfy the kernel's tile alignment. Block
scales require a contiguous last dimension. On gfx1250, **blockscale V descales
must be positive powers of two** (E8M0); Q/K descales may be general FP32 values.
`kv_blockscale` is reserved for a future paged-KV API.

## Tests

| File | Coverage |
| --- | --- |
| `tests/test_fixedlen.py` | Fixed-length/packed forward and backward, bias |
| `tests/test_fp8.py` | FP8 fixed/variable-length quantization, validation and `torch.compile` |
| `tests/test_varlen.py` | Variable-length/packed forward and backward, paged attention |
| `tests/test_kvcache.py` | Cache updates, decode, rotary embeddings, cache indexing |
| `tests/test_interface.py` | Validation, native compatibility, fake tensors, `torch.compile` |
| `tests/test_infer.py` | `fmha_infer`: layouts, masks, bias, LSE, shared KV, validation, `torch.compile` |
| `tests/test_gdpa.py` | GDPA (`fmha_varlen_gdpa_func`, gfx1250) |
| `tests/test_fwd_options.py` | `broadcast_q` and `lse_log2` |

FP8 tests compare against `fmha_func` using dequantized inputs cast to BF16;
other numerical tests use GPU PyTorch references in `tests/reference.py`.
FlashAttention and Triton are not required. Shared helpers live in `tests/utils.py`.

```bash
# Full suite (add --junitxml=test-results/fmha.xml to save a report)
HIP_VISIBLE_DEVICES=0 python -m pytest tests

# Fixed-length bias tests
HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_fixedlen.py -k bias
```
