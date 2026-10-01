# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU coordinate regressions for the CompV3 preshuffled BQ path.

Compile the production policy/encoding, consumer register/lane expressions,
K-window step, and kernel descriptor recipe with generic host-only metadata
and descriptor stubs. Compare their composition with direct logical BQ indexing.
No HIP runtime, GPU discovery, or device execution is used.
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

from header_scrape import between, find, read_header, rfind


CK = Path(__file__).resolve().parents[2]
QUANT = CK / "include/ck_tile/ops/gemm_quant"


@dataclass(frozen=True)
class Case:
    name: str
    tile_n: int
    nwarps: int
    mwarps: int
    kq: int
    group_n: int
    n: int
    k: int


CASES = (
    Case("117fbcf4c44d8998bc51", 128, 1, 2, 2, 128, 128, 1024),
    Case("8a43564d079558cf88a8", 256, 2, 4, 1, 128, 512, 256),
    Case("5e330bb96993d02b80ad", 64, 4, 2, 1, 128, 384, 512),
    Case("a32b6f5afcc38eeb50c3_k_step", 256, 1, 2, 1, 128, 512, 640),
    Case("coarse_decode_kq2", 64, 1, 2, 2, 128, 128, 512),
    Case("medium_group32", 256, 4, 1, 2, 32, 512, 512),
    Case("medium_group64", 256, 8, 1, 1, 64, 512, 256),
    Case("coarse_equal_warp_span", 128, 8, 1, 2, 128, 256, 512),
    Case("06b001012b6e3c4e4e91_fine_control", 128, 1, 2, 1, 1, 512, 256),
    Case("a6307979d253a88763c9_fine_control", 256, 2, 4, 1, 1, 512, 256),
    Case("fine_group1_kq2", 128, 4, 2, 2, 1, 384, 1792),
    Case("fine_group8_kq2", 128, 4, 2, 2, 8, 256, 512),
    Case("group_equals_warp_n", 128, 4, 2, 2, 16, 256, 512),
    Case("coarse_padded_n_descriptor", 256, 1, 2, 1, 128, 384, 512),
)


PRELUDE = r"""
#include <algorithm>
#include <array>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#define CK_TILE_HOST_DEVICE
#define CK_TILE_DEVICE
using index_t=int;
template<int I> using number=std::integral_constant<int,I>;
template<int... I> struct sequence {
    inline static constexpr std::array<int,sizeof...(I)> values{I...};
    static constexpr int at(int n) { return values[n]; }
    static std::vector<int> vec() { return {I...}; }
};
template<class... T> using tuple=std::tuple<T...>;
using std::make_tuple;
template<class T> using remove_cvref_t=std::remove_cv_t<std::remove_reference_t<T>>;
constexpr auto I0=number<0>{};
constexpr auto I1=number<1>{};
constexpr auto I2=number<2>{};
constexpr int get_warp_size() { return 32; }
namespace tensor_layout::gemm { struct ColumnMajor {}; struct RowMajor {}; }
namespace core {
struct amdgcn_compiler_target_state {
    static constexpr bool CK_TILE_ARCH_GFX1250=TEST_GFX1250;
};
}
namespace ck_tile {
constexpr int integer_divide_ceil(int x,int y) { return (x+y-1)/y; }
constexpr int integer_least_multiple(int x,int y) { return integer_divide_ceil(x,y)*y; }
}
template<class... T> auto make_array(T... v) { return std::array<int,sizeof...(T)>{int(v)...}; }
struct tile_distribution_encoding_pattern {};
template<class R_,class H_,class P_,class Q_,class Y_,class Z_>
struct tile_distribution_encoding {
    using R=R_; using H=H_; using P=P_; using Q=Q_; using Y=Y_; using Z=Z_;
};
template<class T> constexpr auto make_static_tile_distribution(T) { return T{}; }
template<class A,class B,class C,int M,int N,int K,bool T>
struct WarpGemmDispatcher { static constexpr int kM=M,kN=N,kK=K; };
template<int N,int NW,int MW,int Q,int G> struct ProblemT {
    using AComputeDataType=unsigned char; using BComputeDataType=unsigned char;
    using CDataType=float; using BQLayout=tensor_layout::gemm::ColumnMajor;
    static constexpr bool TransposeC=false;
    static constexpr int kBlockSize=NW*MW*32;
    struct BlockGemmShape {
        using BlockWarps=sequence<MW,NW,1>; using WarpTile=sequence<16,16,64>;
        static constexpr int kN=N,kK=Q*128;
    };
    struct BQuantGroupSize { static constexpr int kN=G,kK=128; };
    struct Traits { static constexpr bool BPreshuffleQuant=true; };
};
template<class T> void emit_seq(int*& out) {
    auto a=T::vec(); *out++=int(a.size()); for(int x:a) *out++=x;
}
template<class T,std::size_t... I> void emit_tuple(int*& out,std::index_sequence<I...>) {
    *out++=sizeof...(I); (emit_seq<std::tuple_element_t<I,T>>(out),...);
}
template<class T> void emit_tuple(int*& out) {
    emit_tuple<T>(out,std::make_index_sequence<std::tuple_size_v<T>>{});
}
template<class T> void emit_encoding(int* out) {
    emit_seq<typename T::R>(out); emit_tuple<typename T::H>(out);
    emit_tuple<typename T::P>(out); emit_tuple<typename T::Q>(out);
    emit_seq<typename T::Y>(out); emit_seq<typename T::Z>(out);
}

// Generic index-transform composition: no quantization-specific formulas.
struct Address { int offset; bool valid; };
struct Descriptor {
    std::vector<int> lengths;
    std::function<Address(std::vector<int>)> resolve;
    const auto& get_lengths() const { return lengths; }
};
struct Transform { int kind; std::vector<int> lower,upper; };
template<class T,std::size_t... I> auto ints(T t,std::index_sequence<I...>) {
    return std::vector<int>{int(std::get<I>(t))...};
}
template<class... T> auto ints(std::tuple<T...> t) {
    return ints(t,std::index_sequence_for<T...>{});
}
int product(const std::vector<int>& a) { int p=1; for(int x:a)p*=x; return p; }
auto make_pass_through_transform(int n) { return Transform{0,{n},{n}}; }
auto make_right_pad_transform(int n,int pad) { return Transform{0,{n},{n+pad}}; }
template<class T> auto make_unmerge_transform(T t) {
    auto a=ints(t); return Transform{1,{product(a)},a};
}
template<class T> auto make_merge_transform(T t) {
    auto a=ints(t); return Transform{2,a,{product(a)}};
}
// Both production merge implementations have the same coordinate semantics.
// This host model tests descriptor composition, not GPU division lowering.
template<class T> auto make_merge_transform_v3_division_mod(T t) {
    return make_merge_transform(t);
}
template<class L,class S,class... V> auto make_naive_tensor_descriptor(L l,S s,V...) {
    auto lengths=ints(l),strides=ints(s);
    return Descriptor{lengths,[=](std::vector<int> idx) {
        Address a{0,true}; for(unsigned i=0;i<idx.size();++i) {
            a.offset+=idx[i]*strides[i]; a.valid &= 0<=idx[i] && idx[i]<lengths[i];
        } return a;
    }};
}
template<class TS,class LS,class US,std::size_t... I>
auto transform_impl(Descriptor old,TS ts,LS,US,std::index_sequence<I...>) {
    std::vector<Transform> transforms{std::get<I>(ts)...};
    std::vector<std::vector<int>> lows{std::tuple_element_t<I,LS>::vec()...};
    std::vector<std::vector<int>> ups{std::tuple_element_t<I,US>::vec()...};
    std::vector<int> lengths;
    for(unsigned j=0;j<ups.size();++j) for(unsigned k=0;k<ups[j].size();++k) {
        lengths.resize(std::max(lengths.size(),std::size_t(ups[j][k]+1)));
        lengths[ups[j][k]]=transforms[j].upper[k];
    }
    return Descriptor{lengths,[=](std::vector<int> top) {
        std::vector<int> bottom(old.lengths.size()); bool valid=true;
        for(unsigned j=0;j<transforms.size();++j) {
            auto tr=transforms[j]; std::vector<int> x;
            for(int i:ups[j])x.push_back(top[i]);
            for(unsigned i=0;i<x.size();++i) valid &= 0<=x[i] && x[i]<tr.upper[i];
            std::vector<int> y(tr.lower.size());
            if(tr.kind==0) y[0]=x[0];
            else if(tr.kind==1) { for(unsigned i=0;i<x.size();++i)y[0]=y[0]*tr.upper[i]+x[i]; }
            else { int flat=x[0]; for(int i=int(y.size())-1;i>=0;--i) { y[i]=flat%tr.lower[i]; flat/=tr.lower[i]; } valid &= flat==0; }
            for(unsigned i=0;i<y.size();++i) { valid &= 0<=y[i] && y[i]<tr.lower[i]; bottom[lows[j][i]]=y[i]; }
        }
        auto a=old.resolve(bottom); a.valid &= valid; return a;
    }};
}
template<class... T,class L,class U> auto transform_tensor_descriptor(Descriptor d,std::tuple<T...> t,L l,U u) {
    return transform_impl(d,t,l,u,std::index_sequence_for<T...>{});
}
enum class address_space_enum { global };
struct View { Descriptor desc; };
template<address_space_enum A,class T> auto make_tensor_view(const T*,Descriptor d) { return View{d}; }
struct InputWindow {
    int get_bottom_tensor_view() const { return 0; }
    auto get_window_origin() const { return std::array<int,2>{13,17}; }
};
struct LoadWindow { std::vector<int> lengths; std::array<int,2> origin; };
template<class L,class D> auto make_tile_window(int,L lengths,std::array<int,2> origin,D) {
    return LoadWindow{ints(lengths),origin};
}
int thread_lane=0;
int __lane_id() { return thread_lane; }
"""


def policy_method(path, name):
    """The member template of a policy struct that declares ``name``."""
    text = read_header(path)
    where = path.name
    decl = find(text, name, where=where)
    start = rfind(text, "    template <", decl, where=where)
    return text[start:find(text, "    template <", decl, where=where)]


def probe_source():
    utility = read_header(QUANT / "pipeline/gemm_group_quant_utils.hpp")
    encoding = between(utility, "// TODO:: might need to update", "template <typename GroupSizes>",
                       where="gemm_group_quant_utils.hpp")
    policy = policy_method(QUANT / "pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp",
                           "static constexpr auto MakeBQDramTileDistribution")
    ab_policy = policy_method(QUANT / "pipeline/gemm_abquant_pipeline_ag_bg_cr_policy.hpp",
                              "static constexpr auto MakeBQDramTileDistribution")
    base = read_header(QUANT / "pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp")
    call = between(base, "Policy::template MakeBQDramTileDistribution", ");",
                   where="gemm_bquant_pipeline_ag_bg_cr_base.hpp")
    kernel = read_header(QUANT / "kernel/gemm_quant_kernel.hpp")
    descriptor = between(kernel, "    template <index_t KPerBlockBQ,", "    public:\n    struct SplitKBatchOffset",
                         where="gemm_quant_kernel.hpp")
    source = PRELUDE + encoding + "\nstruct BPolicy {\n" + policy + "};\n"
    source += "using GemmBQuantPipelineAgBgCrDefaultPolicy=BPolicy;\nstruct ABPolicy {\n" + ab_policy + "};\n"
    source += "struct Kernel { static int get_padding_size(int n,int a) { return ck_tile::integer_least_multiple(n,a)-n; }\n" + descriptor + "};\n"
    loader = between(base, "    template <typename BQDramBlockWindowTmp>", "\n};",
                     where="gemm_bquant_pipeline_ag_bg_cr_base.hpp")
    source += "template<class Problem> struct Loader { using Policy=BPolicy; using BQLayout=typename Problem::BQLayout;\n"
    source += "static constexpr int NPerBlock=Problem::BlockGemmShape::kN,NPerBlockBQ=(NPerBlock>=Problem::BQuantGroupSize::kN?NPerBlock/Problem::BQuantGroupSize::kN:1),KPerBlockBQ=Problem::BlockGemmShape::kK/128;\n"
    source += loader + "};\n"
    for op, filename in enumerate(("block_universal_gemm_as_bs_bquant_cr.hpp", "block_universal_gemm_as_aquant_bs_bquant_cr.hpp")):
        block = read_header(QUANT / "block" / filename)
        # The preshuffled-BQ branch is the reg_offset block that gathers via pull_from_lane.
        pull = find(block, "auto pull_from_lane", where=filename)
        start = rfind(block, "constexpr index_t reg_offset =", pull, where=filename)
        reg = block[start:find(block, "auto& scale_reg", pull, where=filename)]
        source += f"template<class P,int NI,int Q> int reg{op}(int lane,int* pull) {{\n"
        source += "using GemmTraits=P; using WarpGemm=WarpGemmDispatcher<int,int,int,16,16,64,false>;\n"
        source += "constexpr int NWarp=P::BlockGemmShape::BlockWarps::at(1),nIter=NI,kQScale=Q;\n"
        source += "struct Traits { enum { NPerBlock=P::BlockGemmShape::kN,KQPerBlock=P::BlockGemmShape::kK/128 }; };\n"
        source += "thread_lane=lane;\n" + reg + "*pull=pull_from_lane; return reg_offset; }\n"
        pipe_name = "gemm_" + ("bquant" if op == 0 else "abquant") + "_pipeline_ag_bg_cr_v3.hpp"
        pipe = read_header(QUANT / "pipeline" / pipe_name)
        step = between(pipe, "const BQDramTileWindowStep bq_dram_tile_window_step =", ";", where=pipe_name) + ";\n"
        source += f"template<class P> int step{op}(int n) {{\n"
        source += "using BlockGemmShape=typename P::BlockGemmShape; using BQuantGroupSize=typename P::BQuantGroupSize; using BQDramTileWindowStep=std::array<int,2>;\n"
        source += "constexpr bool BPreshuffleQuant=true,is_bq_row_major=false; constexpr int NPerBlock=BlockGemmShape::kN,NPerBlockBQ=(NPerBlock>=BQuantGroupSize::kN?NPerBlock/BQuantGroupSize::kN:1),KPerBlockBQ=BlockGemmShape::kK/128;\n"
        source += step + "return bq_dram_tile_window_step[0]; }\n"
    source += "template<class P,int... I> int regs(int op,int ni,int q,int lane,int* pull,std::integer_sequence<int,I...>) {\n"
    source += "constexpr int Q=P::BlockGemmShape::kK/128; using F=int(*)(int,int*); F b[]={&reg0<P,I/Q,I%Q>...}; F ab[]={&reg1<P,I/Q,I%Q>...}; return (op?ab:b)[ni*Q+q](lane,pull); }\n"
    for i, c in enumerate(CASES):
        source += f"using P{i}=ProblemT<{c.tile_n},{c.nwarps},{c.mwarps},{c.kq},{c.group_n}>;\n"
    source += 'extern "C" void metadata(int ci,int op,int compact,int* out) { switch(ci) {\n'
    for i in range(len(CASES)):
        source += f"case {i}: {{ using Problem=P{i}; if(!compact) emit_encoding<decltype(BPolicy::MakeBQDramTileDistribution<Problem>())>(out); else if(op) {{ using Policy=ABPolicy; emit_encoding<decltype({call})>(out); }} else {{ using Policy=BPolicy; emit_encoding<decltype({call})>(out); }} return; }}\n"
    source += "} }\n"
    source += 'extern "C" int consumer(int ci,int op,int ni,int q,int lane,int* pull) { switch(ci) {\n'
    for i, c in enumerate(CASES):
        source += f"case {i}:return regs<P{i}>(op,ni,q,lane,pull,std::make_integer_sequence<int,{c.tile_n//(c.nwarps*16)*c.kq}>{{}});\n"
    source += "} return -1; }\n"
    source += 'extern "C" int step(int ci,int op,int n) { switch(ci) {\n'
    for i in range(len(CASES)):
        source += f"case {i}:return op?step1<P{i}>(n):step0<P{i}>(n);\n"
    source += "} return -1; }\n"
    source += 'extern "C" void load_lengths(int ci,int* out) { LoadWindow w; switch(ci) {\n'
    for i in range(len(CASES)):
        source += f"case {i}:w=Loader<P{i}>{{}}.GetBQDramLoadWindow(InputWindow{{}});break;\n"
    source += "} out[0]=w.lengths[0];out[1]=w.lengths[1];out[2]=w.origin[0];out[3]=w.origin[1]; }\n"
    source += 'extern "C" int address(int ci,int n,int qk,int row,int col,int* info) { View v; switch(ci) {\n'
    for i, c in enumerate(CASES):
        source += f"case {i}:v=Kernel::MakePreshuffledQuantTensorView<{c.kq},{max(1,c.tile_n//c.group_n)},{c.tile_n},16,1>(static_cast<float*>(nullptr),(n+{c.group_n}-1)/{c.group_n},{c.group_n},qk);break;\n"
    source += "} auto a=v.desc.resolve({row,col}); info[0]=a.valid; info[1]=v.desc.lengths[0]; info[2]=v.desc.lengths[1]; return a.offset; }\n"
    return source


@pytest.fixture(scope="module", params=[False, True], ids=["other_arch", "gfx1250"])
def probe(tmp_path_factory, request):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        pytest.skip("requires a host C++ compiler")
    tmp = tmp_path_factory.mktemp("bq_coordinates")
    path = tmp / "probe.cpp"
    path.write_text(probe_source())
    result = subprocess.run(compiler + ["-std=c++17", "-O0", "-shared", "-fPIC",
                            f"-DTEST_GFX1250={int(request.param)}", str(path),
                            "-o", str(tmp / "probe.so")], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    lib = ctypes.CDLL(str(tmp / "probe.so"))
    for name, argc in (("metadata", 4), ("consumer", 6), ("address", 6), ("load_lengths", 2)):
        getattr(lib, name).argtypes = [ctypes.c_int] * (argc - 1) + [ctypes.POINTER(ctypes.c_int)]
    lib.step.argtypes = [ctypes.c_int] * 3
    return lib


def metadata(probe, ci, op=0, compact=True):
    output = (ctypes.c_int * 128)()
    probe.metadata(ci, op, int(compact), output)
    data = iter(output)
    def sequence():
        return [next(data) for _ in range(next(data))]
    def tuples():
        return [sequence() for _ in range(next(data))]
    return sequence(), tuples(), tuples(), tuples(), sequence(), sequence()


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def coordinate(encoding, warp, lane, register):
    """Generic CK encoding evaluator: unmerge P/Y coordinates, then merge H."""
    replicas, hidden, pmajor, pminor, ymajor, yminor = encoding
    lengths = [replicas, *hidden]
    values = [[0] * len(x) for x in lengths]
    ylengths = [lengths[m][i] for m, i in zip(ymajor, yminor)]
    yvalues = [0] * len(ylengths)
    for i in reversed(range(len(ylengths))):
        yvalues[i] = register % ylengths[i]
        register //= ylengths[i]
    assert register == 0
    for m, i, value in zip(ymajor, yminor, yvalues):
        values[m][i] = value
    for value, majors, minors in zip((warp, lane), pmajor, pminor):
        for m, i in reversed(list(zip(majors, minors))):
            values[m][i] = value % lengths[m][i]
            value //= lengths[m][i]
        assert value == 0
    result = []
    for dims, indices in zip(hidden, values[1:]):
        flat = 0
        for length, index in zip(dims, indices):
            flat = flat * length + index
        result.append(flat)
    return result


def register_count(encoding):
    replicas, hidden, _, _, ym, yi = encoding
    lengths = [replicas, *hidden]
    return product(lengths[m][i] for m, i in zip(ym, yi))


@pytest.mark.parametrize("ci", [i for i, c in enumerate(CASES) if c.group_n > 16],
                         ids=[c.name for c in CASES if c.group_n > 16])
def test_compact_loader_lengths_match_sampled_descriptor_region(probe, ci):
    output = (ctypes.c_int * 4)()
    probe.load_lengths(ci, output)
    encoding = metadata(probe, ci)
    assert list(output[:2]) == [product(h) for h in encoding[1]]
    assert list(output[2:]) == [13, 17]


@pytest.mark.parametrize("ci", range(len(CASES)), ids=[c.name for c in CASES])
def test_compact_load_rows_and_legacy_default(probe, ci):
    c = CASES[ci]
    current = metadata(probe, ci)
    legacy = metadata(probe, ci, compact=False)
    assert metadata(probe, ci, op=1) == current
    expected_rows = c.tile_n // 16 if c.group_n <= 16 else max(1, c.tile_n // c.group_n)
    assert product(current[1][0]) == expected_rows
    assert product(legacy[1][0]) == c.tile_n // 16
    if c.group_n <= 16:
        assert current == legacy
    for warp in range(c.mwarps * c.nwarps):
        for lane in range(32):
            for reg in range(register_count(current)):
                row, col = coordinate(current, warp, lane, reg)
                assert 0 <= row < expected_rows
                assert 0 <= col < max(1, 16 // c.group_n) * c.kq


@pytest.mark.parametrize("op", (0, 1), ids=("bquant", "abquant"))
@pytest.mark.parametrize("ci", range(len(CASES)), ids=[c.name for c in CASES])
def test_k_step_matches_production_descriptor(probe, ci, op):
    c = CASES[ci]
    info = (ctypes.c_int * 3)()
    probe.address(ci, c.n, c.k // 128, 0, 0, info)
    k_blocks = c.k // (c.kq * 128)
    assert probe.step(ci, op, c.n) == info[1] // k_blocks


@pytest.mark.parametrize("op", (0, 1), ids=("bquant", "abquant"))
@pytest.mark.parametrize("ci", range(len(CASES)), ids=[c.name for c in CASES])
def test_consumed_scale_matches_logical_n_and_k_group(probe, ci, op):
    c = CASES[ci]
    encoding = metadata(probe, ci, op)
    pull = ctypes.c_int()
    info = (ctypes.c_int * 3)()
    groups_n = (c.n + c.group_n - 1) // c.group_n
    for block_n in range(0, c.n, c.tile_n):
        # Actual kernel origin for observed blocks: G<=tileN, or G/tileN==2.
        origin = block_n // 16 if c.group_n <= 16 else block_n // c.group_n
        for kb in range(c.k // (128 * c.kq)):
            for warp in range(c.mwarps * c.nwarps):
                for ni in range(c.tile_n // (c.nwarps * 16)):
                    for lane in range(32):
                        logical_n = block_n + (ni * c.nwarps + warp % c.nwarps) * 16 + lane % 16
                        if logical_n >= c.n:
                            continue
                        for q in range(c.kq):
                            reg = probe.consumer(ci, op, ni, q, lane, ctypes.byref(pull))
                            assert 0 <= reg < register_count(encoding)
                            assert 0 <= pull.value < 32
                            row, col = coordinate(encoding, warp, pull.value, reg)
                            row += origin + kb * probe.step(ci, op, c.n)
                            actual = probe.address(ci, c.n, c.k // 128, row, col, info)
                            expected = kb * groups_n * c.kq + (logical_n // c.group_n) * c.kq + q
                            assert info[0], (c.name, kb, logical_n, q, row, col)
                            assert actual == expected, (c.name, kb, logical_n, q, actual, expected)
