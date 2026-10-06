# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU regressions for BQuant packing/guards and ABQuant WMMA distributions.

The C++ probes execute production guard code and instantiate the production
WMMA distribution traits. They stop before device allocation or kernel launch.
"""

import ctypes
import os
import re
from dataclasses import replace
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

DISPATCHER = Path(__file__).resolve().parents[1]
CK = DISPATCHER.parent
sys.path[:0] = [str(DISPATCHER / "python"), str(DISPATCHER / "codegen")]
import gemm_bquant_utils as bq
from header_scrape import between, find, read_header

LAYOUTS = ("rcr", "ccr", "rrr", "crr")


def compile_cpp(tmp_path, source, shared=False):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        pytest.skip("requires a host C++ compiler")
    path = tmp_path / "probe.cpp"
    path.write_text(source)
    output = tmp_path / ("probe.so" if shared else "probe.o")
    flags = ["-shared", "-fPIC"] if shared else ["-c"]
    result = subprocess.run(compiler + ["-std=c++17", *flags, str(path), "-o", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return output


@pytest.fixture(params=LAYOUTS)
def packed_guard(request, tmp_path):
    layout = request.param
    bridge = read_header(DISPATCHER / "bindings/ctypes/gemm_bquant_ctypes_lib.cpp")
    prefix = between(bridge, "int dispatcher_run_bquant_gemm(", "    const BDataType* B_host",
                     where="gemm_bquant_ctypes_lib.cpp")
    common = read_header(DISPATCHER / "bindings/ctypes/quant_bridge_common.hpp")
    groups = between(common, "inline bool check_quant_group_count(", "// The identical tail",
                     where="quant_bridge_common.hpp")
    source = """#include <cstdint>
#include <iostream>
#include <type_traits>
#include <initializer_list>
namespace ck_tile::tensor_layout::gemm { struct RowMajor {}; struct ColumnMajor {}; }
namespace quant_bridge {
constexpr bool g_initialized = true;
// Entry/lifecycle/GPU architecture checks are outside this host stride probe.
inline bool check_entry_args(const char*, bool, std::initializer_list<const void*>,
                             std::initializer_list<int64_t>, bool) { return true; }
""" + groups + "}\n"
    for operand, char in zip(("A", "B"), layout):
        source += f"using {operand}Layout = ck_tile::tensor_layout::gemm::{'Column' if char == 'c' else 'Row'}Major;\n"
    source += "struct QuantGroupSize { static constexpr int kK=128, kN=1; };\n"
    # 17 means only that the host prefix accepted the packed arguments.
    source += 'extern "C" {\n' + prefix + "return 17;\n}\n}\n"
    lib = ctypes.CDLL(str(compile_cpp(tmp_path, source, shared=True)))
    run = lib.dispatcher_run_bquant_gemm
    run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 9 + [ctypes.c_int, ctypes.c_void_p]
    run.restype = ctypes.c_int
    return layout, run


@pytest.mark.parametrize("shape", [(512, 64, 640), (256, 128, 1024), (128, 128, 128)])
def test_production_bquant_guard_accepts_only_packed_layout(packed_guard, shape):
    layout, run = packed_guard
    M, N, K = shape
    QK = (K + 127) // 128
    strides = [M if layout[0] == "c" else K, K if layout[1] == "c" else N, QK, N]
    args = [1] * 4 + [M, N, K, *strides, QK, N, 1, None]
    assert run(*args) == 17
    for index in range(4):
        invalid = args.copy()
        invalid[7 + index] += 1
        assert run(*invalid) == -1
    for index in (11, 12):
        invalid = args.copy()
        invalid[index] += 1
        assert run(*invalid) == -1


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("variant", ["bf8", "mx_bf16bf8"])
def test_bquant_low_level_packs_logical_matrices(layout, variant):
    # Non-square matrices and distinctive byte payloads expose transposition.
    M, N, K = 7, 11, 13
    A = np.arange(M * K, dtype=np.uint8).reshape(M, K)
    B = np.arange(K * N, dtype=np.uint8).reshape(K, N)
    BQ = np.arange(3 * N, dtype=np.float32).reshape(3, N)
    C = np.zeros((M, N), dtype=np.float16)
    name = f"gemm_bquant_{variant}_{layout}_compv3_cshuffle_intrawave_test"
    observed = {}
    def native(*args):
        for key, pointer, array in zip(("A", "B", "BQ", "C"), args[:4], (A, B, BQ, C)):
            observed[key] = ctypes.string_at(pointer, array.nbytes)
        return 0
    lib = bq.BQuantDispatcherLib.__new__(bq.BQuantDispatcherLib)
    lib._lib = SimpleNamespace(dispatcher_run_bquant_gemm=native)
    lib.get_kernel_name = lambda: name
    rc, _ = lib.run(A, B, BQ, C, M, N, K, M if layout[0] == "c" else K,
                   K if layout[1] == "c" else N, 3, N, 3, N)
    assert rc == 0
    assert observed["A"] == A.tobytes(order="F" if layout[0] == "c" else "C")
    assert observed["B"] == B.tobytes(order="F" if layout[1] == "c" else "C")
    assert observed["BQ"] == BQ.tobytes(order="F")
    # Already-packed one-dimensional byte payloads must remain byte exact.
    observed.clear()
    lib.run(A.ravel(), B.ravel(), BQ.ravel(), C, M, N, K,
            M if layout[0] == "c" else K, K if layout[1] == "c" else N, 3, N, 3, N)
    assert observed["A"] == A.tobytes()
    assert observed["B"] == B.tobytes()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bquant_high_level_passes_generated_layout_strides(layout):
    M, N, K = 512, 64, 640  # Original 00a2669d103fc2234a75_0 shape.
    name = f"gemm_bquant_bf8_{layout}_compv3_cshuffle_intrawave_test"
    observed = {}
    def run(**kwargs):
        observed.update(kwargs)
        return 0, 0.0
    runner = bq.BQuantGpuGemmRunner.__new__(bq.BQuantGpuGemmRunner)
    runner._lib = SimpleNamespace(get_kernel_name=lambda: name, run=run)
    runner.run(np.zeros((M, K), np.uint8), np.zeros((K, N), np.uint8),
               np.ones((5, N), np.float32), bq.BQuantGemmProblem(M=M, N=N, K=K))
    assert observed["stride_A"] == (M if layout[0] == "c" else K)
    assert observed["stride_B"] == (K if layout[1] == "c" else N)
    assert observed["stride_BQ"] == 5
    assert observed["stride_C"] == N


def test_bquant_layout_metadata_fails_closed():
    with pytest.raises(ValueError, match="Cannot determine BQuant matrix layout"):
        bq._layout_from_kernel_name("unrecognized_kernel")


@pytest.mark.parametrize("arch,warp_size", [("gfx125", 32), ("gfx950", 64), ("gfx942", 64)])
def test_abquant_production_selector_and_wmma_distribution(arch, warp_size, tmp_path):
    policy_name = "gemm_abquant_pipeline_ag_bg_cr_policy.hpp"
    policy = read_header(CK / "include/ck_tile/ops/gemm_quant/pipeline" / policy_name)
    begin = policy.find("#if defined(__gfx125__)")
    if begin < 0:
        begin = find(policy, "        constexpr index_t vector_size =", where=policy_name)
    selector = policy[begin:find(policy, "        using WarpGemm =", begin, where=policy_name)]
    enum_header = read_header(CK / "include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp")
    enum = between(enum_header, "enum class WGAttrNumAccessEnum", "template <WGAttrNumAccessEnum",
                   where="warp_gemm_attribute_mfma.hpp")
    wmma_header = read_header(CK / "include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp")
    traits = between(wmma_header, "template <typename Impl", "template <typename Impl>\nstruct CWarp",
                     where="warp_gemm_attribute_wmma.hpp")
    source = f"#define __{arch}__\n#include <type_traits>\n#include <tuple>\nusing index_t=int;\n{enum}\n"
    source += """
template<int... X> struct sequence { static constexpr bool valid=((X>0)&&...); };
template<class... T> using tuple=std::tuple<T...>;
template<class R, class H, class P, class Q, class Y, class Z>
struct tile_distribution_encoding { using lengths=H; };
""" + traits
    source += f"constexpr int get_warp_size() {{ return {warp_size}; }}\nconstexpr int DS_READ_TR_SIZE() {{ return 8; }}\n"
    source += """
template<int K> struct Tile { static constexpr int at(int i) { return i==1 ? 16 : K; } };
template<int K, bool Tr> struct ProblemT { using AComputeDataType=unsigned char; static constexpr bool tr=Tr; };
struct Base { template<class P> static constexpr bool is_a_load_tr=P::tr; template<class P> static constexpr bool is_b_load_tr=false; };
template<int K, bool Tr> constexpr auto access() {
using Problem=ProblemT<K,Tr>; using WarpTile=Tile<K>; constexpr int I1=1,I2=2;
""" + selector + "return wg_attr_num_access;\n}\n"
    layout_header = read_header(CK / "include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp")
    native_layout = between(layout_header, "template <typename DataType, index_t K>\n", "template <typename Arch,",
                            where="warp_gemm_attribute_wmma_impl_base_traits.hpp")
    source += "using fp32_t=float; using fp64_t=double; template<class,int,bool> struct LayoutFromDataType;\n" + native_layout
    source += """
template<int K> struct Impl {
using ADataType=unsigned char; using BDataType=unsigned char;
static constexpr int kAK0PerLane=LayoutFromDataType<unsigned char,K,false>::kK0PerLane;
static constexpr int kAK1PerLane=LayoutFromDataType<unsigned char,K,false>::kK1PerLane;
static constexpr int kBK0PerLane=kAK0PerLane,kBK1PerLane=kAK1PerLane;
static constexpr int kRepeat=1,kAMBlock=1,kBNBlock=1,kAMLane=16,kBNLane=16,kABKLane=2;
using kABPs2RHssMajor=sequence<2,1>; using kABPs2RHssMinor=sequence<1,1>;
using kABYs2RHsMajor=sequence<1,2,2>; using kABYs2RHsMinor=sequence<0,0,2>;
};
"""
    for K in (64, 128):
        for transpose in (False, True):
            tr = str(transpose).lower()
            expected = "Default" if arch == "gfx125" else ("Single" if not transpose or arch == "gfx942" else "Double" if K == 64 else "Quad")
            # gfx942 has no transpose loads, regardless of requested A layout.
            effective_tr = "false" if arch == "gfx942" else tr
            source += f"static_assert(access<{K},{effective_tr}>()==WGAttrNumAccessEnum::{expected});\n"
            if arch == "gfx125":
                for operand in ("A", "B"):
                    alias = f"{operand}{K}{tr}"
                    source += f"using {alias}=typename {operand}WarpDstrEncodingTrait<Impl<{K}>,access<{K},{tr}>()>::type;\n"
                    source += f"static_assert(std::tuple_element_t<1,typename {alias}::lengths>::valid);\n"
                    source += f"static_assert(std::is_same_v<std::tuple_element_t<1,typename {alias}::lengths>,sequence<{K//16},2,8>>);\n"
    # K64 Quad already has native ordering. Do not attribute separate scale failures to this selector.
    source += "static_assert(std::is_same_v<AWarpDstrEncodingTrait<Impl<64>,WGAttrNumAccessEnum::Quad>::type,AWarpDstrEncodingTrait<Impl<64>,WGAttrNumAccessEnum::Default>::type>);\n"
    compile_cpp(tmp_path, source)


@pytest.fixture(scope="module")
def aq_coordinates(tmp_path_factory):
    header = read_header(CK / "include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_base.hpp")
    method = between(header, "template <typename AQDramBlockWindowTmp>", "    template <typename BQDramBlockWindowTmp>",
                     where="gemm_abquant_pipeline_ag_bg_cr_base.hpp")
    source = """#include <array>
#include <tuple>
#include <type_traits>
#define CK_TILE_DEVICE
using index_t=int;
template<int I> struct number { static constexpr int value=I; constexpr operator int() const { return I; } };
template<int I> using sequence=number<I>;
namespace tensor_layout::gemm { struct RowMajor {}; struct ColumnMajor {}; }
template<class... T> auto make_tuple(T... x) { return std::tuple<T...>{x...}; }
template<class... T> auto make_array(T... x) { return std::array<int,sizeof...(T)>{x...}; }
struct View {
 std::array<int,2> lengths,strides;
 const View& get_tensor_descriptor() const { return *this; }
 int get_length(int i) const { return lengths[i]; }
};
struct Window {
 View view; std::array<int,2> lengths,origin;
 const View& get_bottom_tensor_view() const { return view; }
 const auto& get_window_lengths() const { return lengths; }
 const auto& get_window_origin() const { return origin; }
};
int make_pass_through_transform(int x) { return x; }
// Generic descriptor permutation stub: applies the production lower/upper IDs.
template<class T,class L,class U> View transform_tensor_view(View view,T,L,U) {
 View result{};
 constexpr int l0=std::tuple_element_t<0,L>::value,l1=std::tuple_element_t<1,L>::value;
 constexpr int u0=std::tuple_element_t<0,U>::value,u1=std::tuple_element_t<1,U>::value;
 result.lengths[u0]=view.lengths[l0]; result.lengths[u1]=view.lengths[l1];
 result.strides[u0]=view.strides[l0]; result.strides[u1]=view.strides[l1];
 return result;
}
template<class L> Window make_tile_window(View view,L lengths,std::array<int,2> origin,int) {
 return {view,{std::get<0>(lengths),std::get<1>(lengths)},origin};
}
struct Policy { template<class P> static int MakeAQDramTileDistribution() { return 0; } };
struct AQuantBase { Window GetAQDramLoadWindow(Window w) const { return w; } };
template<bool Column,bool Preshuffle> struct ProblemT {
 using AQLayout=std::conditional_t<Column,tensor_layout::gemm::ColumnMajor,tensor_layout::gemm::RowMajor>;
 struct Traits { static constexpr bool APreshuffleQuant=Preshuffle; };
};
template<class Problem> struct Consumer {
""" + method + """};
template<bool C,bool P> Window run(Window input) {
 return Consumer<ProblemT<C,P>>{}.GetAQDramLoadWindow(input);
}
extern "C" void coordinates(int column,int preshuffle,int M,int QK,int tileM,int tileQK,
                            int originM,int originQK,int* output) {
 Window input{{{M,QK},column?std::array<int,2>{1,M}:std::array<int,2>{QK,1}},
              {tileM,tileQK},{originM,originQK}};
 auto out=column?(preshuffle?run<true,true>(input):run<true,false>(input)):
                 (preshuffle?run<false,true>(input):run<false,false>(input));
 int j=0; for(auto a:{out.view.lengths,out.view.strides,out.lengths,out.origin})
 for(int x:a) output[j++]=x;
}
"""
    lib = ctypes.CDLL(str(compile_cpp(tmp_path_factory.mktemp("aq_coordinates"), source, shared=True)))
    run = lib.coordinates
    run.argtypes = [ctypes.c_int] * 8 + [ctypes.POINTER(ctypes.c_int)]
    return run


@pytest.mark.parametrize("column,preshuffle", [(False, False), (True, False), (False, True), (True, True)])
def test_abquant_production_aq_coordinates_and_k_steps(aq_coordinates, column, preshuffle):
    # Original K64 failures: 8a435 (M=512,QK=2,tileM=128), 5e330 (M=192,QK=4,tileM=64).
    for M, QK, tileM, tileQK in [(512, 2, 128, 1), (192, 4, 64, 1), (256, 8, 64, 2)]:
        for originM in (0, tileM, M - tileM):
            for originQK in (0, QK - tileQK):
                result = (ctypes.c_int * 8)()
                aq_coordinates(column, preshuffle, M, QK, tileM, tileQK, originM, originQK, result)
                lengths, strides, tiles, origin = np.asarray(result).reshape(4, 2)
                transpose = column and not preshuffle
                assert tuple(lengths) == ((QK, M) if transpose else (M, QK))
                assert tuple(tiles) == ((tileQK, tileM) if transpose else (tileM, tileQK))
                assert tuple(origin) == ((originQK, originM) if transpose else (originM, originQK))
                # Compare every element of each tile against direct matrix addressing,
                # including successive K-group advances and nonzero split-K bases.
                for m in range(tileM):
                    for q in range(tileQK):
                        local = np.array((q, m) if transpose else (m, q))
                        actual = int(np.dot(origin + local, strides))
                        expected = ((originQK + q) * M + originM + m if column
                                    else (originM + m) * QK + originQK + q)
                        assert actual == expected
                if not preshuffle:
                    k_step = np.array((tileQK, 0) if column else (0, tileQK))
                    assert int(np.dot(k_step, strides)) == (M * tileQK if column else tileQK)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bquant_generated_layout_aliases_reach_cpp_bridge(layout, tmp_path):
    cfg = replace(bq.default_bf8_config(gfx_arch="gfx1250"), layout=layout)
    header = bq._generate_bquant_kernel(cfg, tmp_path / "headers").read_text()
    ns = re.search(r"namespace (\w+) \{", header).group(1)
    definitions = re.findall(r"^using [AB]Layout\s*=.*;", header, re.MULTILINE)
    # Compile the actual namespace declarations and global re-exports emitted
    # by codegen. Missing aliases make the real bridge's type checks ill formed.
    local = [line for line in definitions if "ck_tile::" in line]
    exported = [line for line in definitions if ns + "::" in line]
    source = "#include <type_traits>\nnamespace ck_tile::tensor_layout::gemm { struct RowMajor {}; struct ColumnMajor {}; }\n"
    source += "namespace " + ns + " {\n" + "\n".join(local) + "\n}\n" + "\n".join(exported)
    for operand, char in zip(("A", "B"), layout):
        expected = "ColumnMajor" if char == "c" else "RowMajor"
        source += f"\nstatic_assert(std::is_same_v<{operand}Layout, ck_tile::tensor_layout::gemm::{expected}>);\n"
    compile_cpp(tmp_path, source)
