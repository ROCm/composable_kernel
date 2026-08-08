#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补 jagged x single 覆盖缺口的 no-group jagged=1 case(卡 JAG-B2 / K2)。

背景
----
现状(基线 166,见 docs/JAG-TESTFACE.md / _jag-q3-coverage.md):
`-jagged=1` 有 84 条,**全部 route=fallback(走 base 路),走 single 的 0 条**。
即「single 路的 jagged 分支」当前零覆盖。B1(pane/K1)已把 run_v6.py 的
classify_route 改为「no-group: single <=> !has_dropout」(run_v6.py:263-268,
标 JAG-B1),jagged 不再触发回落。本脚本补的就是「no-group + jagged=1 + p_drop=0
=> 走 single」的 case,并覆盖 single 已有的正交轴。

风格与命令行接口
----------------
完全照 ck_qf 现有 test_hstu_attention_bwd.py / _hdim96_hdim64_bwd.py /
_cross_attention_bwd.py 的写法:每个 case 一次 subprocess.run,打印 "+ cmd"
(等价 set -x),按 EXE=build/bin/tile_example_hstu_attention_bwd 直调。
run_v6.py 通过 monkeypatch subprocess.run 拦截命令行,把首个 token 换成 flag=ON
的 exe;故本脚本的 EXE 与 ck_qf 脚本保持完全一致(硬编码 "build/...")。

★★ 最要命的坑(逐 batch 写全 N 个)
--------------------------------
example_helper.hpp:110-120 supplement_array_by_last_element:当 -seqlens 元素数
< -b 时,用最后一个元素**静默补齐**、不报错。
=> -b=12 -jagged=1 -seqlens=100 会静默变成 12 个全 100(退化定长),得到「名字叫
   jagged、其实定长」的假覆盖。
本脚本铁律:**每个 case 的 -b 严格等于 -seqlens 的元素数,且元素两两不等**。
`--list` 子命令做静态自验(不跑 harness),证明「长度真的不等」+ 每个 case 的
静态 route(与 run_v6.py:classify_route 同规则)。

已确证的默认值(要什么显式写什么,不吃默认)
------------------------------------------
example_hstu_attention_bwd.cpp: g=1(:57) jagged=0(:59) b=12(:60)
seqlens=400(:64) seqlens_kv=""(:65) p_drop=0(:73) causal=1(:74)
local_len=5(:75) context_len=6(:77)。
-seqlens_kv 非空 = 悄悄切 cross-attention(:187-190)。-g>1 时 jagged 被忽略
(:1547-1560)。本脚本一律 g=1(不给 -g),p_drop=0(不给 -p_drop)。
"""

import argparse
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_bwd"


# ---------------------------------------------------------------------------
# case 定义:每个 dict 是一个 no-group jagged=1 的 case。
# 'seqlens' 必是逐 batch 全写、两两不等的列表;'b' 由 len(seqlens) 推出,不单独写。
# 'seqlens_kv' 非 None => cross-attention(必须也逐 batch 全写、与 b 等长)。
# ---------------------------------------------------------------------------

def _b(case):
    """batch 数 = seqlens 元素数(铁律:严格相等,杜绝静默补齐)。"""
    return len(case["seqlens"])


def build_cmd(exe_prefix, case):
    """把一个 case dict 展开成完整命令行(list)。"""
    b = _b(case)
    seqlens = ",".join(str(x) for x in case["seqlens"])
    cmd = (
        list(exe_prefix)
        + ["-v=1", f"-prec={case['prec']}", f"-b={b}", "-jagged=1",
           f"-nhead={case.get('nhead', 4)}",
           f"-hdim_qk={case['hdim']}", f"-hdim_v={case['hdim']}",
           f"-seqlens={seqlens}"]
    )
    if case.get("seqlens_kv") is not None:
        kv = ",".join(str(x) for x in case["seqlens_kv"])
        cmd += [f"-seqlens_kv={kv}"]
    cmd += [
        f"-causal={case['causal']}",
        f"-local_len={case['local_len']}",
        f"-context_len={case['context_len']}",
        f"-minfull_len={case['minfull_len']}",
        f"-targets={case['targets']}",
        f"-attn_scale={case.get('attn_scale', 0)}",
        f"-norm_dist={case.get('norm_dist', 0)}",
    ]
    return cmd


# 变长长度池:两两不等,覆盖 <block 与 >block、奇偶混合。
# self-attention 用 SEQ_Q;cross 用 SEQ_Q + 不同的 SEQ_KV(kv>q 与 kv<q 混合)。
SEQ_Q6 = [300, 291, 277, 256, 312, 264]          # b=6,6 个两两不等
SEQ_Q5 = [301, 288, 260, 315, 274]               # b=5
SEQ_KV6 = [330, 271, 300, 240, 288, 305]         # 与 SEQ_Q6 逐位不等,kv 有大有小
TGT6 = "12,7,15,4,9,11"                           # targets 逐 batch(6 个)
TGT5 = "10,6,14,8,5"                              # targets 逐 batch(5 个)


def make_cases():
    """返回 (core, cross, group_consistency) 三组 case。

    正交轴覆盖(克制,不穷尽笛卡尔积):
      softmax {0,1} / causal {0,1} / hdim {128, 96, 64} /
      self vs cross(seqlens_kv) / prec {fp16, bf16} / targets & context 非零。
    """
    # --- A. 核心缺口:no-group jagged=1 走 single,self-attention,轴组合 ---
    core = [
        # A1 最小 jagged single:softmax=0, no-causal, hd128, fp16, 无 mask
        dict(tag="A1_soft0_nocausal_hd128_fp16", softmax=0, prec="fp16",
             hdim=128, seqlens=SEQ_Q6, causal=0, local_len=0, context_len=0,
             minfull_len=0, targets=0),
        # A2 softmax=0, causal, hd128, bf16
        dict(tag="A2_soft0_causal_hd128_bf16", softmax=0, prec="bf16",
             hdim=128, seqlens=SEQ_Q6, causal=1, local_len=0, context_len=0,
             minfull_len=0, targets=0),
        # A3 softmax=1(softmax 轴), causal, hd128, fp16
        dict(tag="A3_soft1_causal_hd128_fp16", softmax=1, prec="fp16",
             hdim=128, seqlens=SEQ_Q6, causal=1, local_len=0, context_len=0,
             minfull_len=0, targets=0),
        # A4 causal + local + context + target(全 mask 轴打开), hd128, bf16
        dict(tag="A4_soft0_full_mask_hd128_bf16", softmax=0, prec="bf16",
             hdim=128, seqlens=SEQ_Q6, causal=1, local_len=5, context_len=8,
             minfull_len=7, targets=TGT6),
        # A5 softmax=1 + 全 mask + target, hd128, fp16
        dict(tag="A5_soft1_full_mask_hd128_fp16", softmax=1, prec="fp16",
             hdim=128, seqlens=SEQ_Q6, causal=1, local_len=5, context_len=8,
             minfull_len=7, targets=TGT6),
        # A6 hdim96(WarpGemm 32x32 阳性对照,单 kernel 下预期仍受限 => 见交付说明)
        dict(tag="A6_soft0_causal_hd96_fp16", softmax=0, prec="fp16",
             hdim=96, seqlens=SEQ_Q6, causal=1, local_len=5, context_len=0,
             minfull_len=0, targets=0),
        # A7 hdim64 + softmax=1(触发 32x32x16 路径), fp16
        dict(tag="A7_soft1_causal_hd64_fp16", softmax=1, prec="fp16",
             hdim=64, seqlens=SEQ_Q6, causal=1, local_len=5, context_len=0,
             minfull_len=0, targets=0),
        # A8 no-causal + context + target(context 加物理长度的边界), hd128, bf16
        dict(tag="A8_soft0_nocausal_ctx_tgt_hd128_bf16", softmax=0, prec="bf16",
             hdim=128, seqlens=SEQ_Q5, causal=0, local_len=5, context_len=6,
             minfull_len=4, targets=TGT5),
        # A9 minfull_len > max_uih(退化全满行) 边界, hd128, fp16
        dict(tag="A9_soft0_minfull_gt_uih_hd128_fp16", softmax=0, prec="fp16",
             hdim=128, seqlens=SEQ_Q5, causal=1, local_len=5, context_len=0,
             minfull_len=400, targets=TGT5),
    ]

    # --- B. cross-attention(seqlens_kv 显式,逐 batch 全写、与 b 等长) ---
    cross = [
        # B1 cross, kv 长度各异, softmax=0, causal, hd128, fp16
        dict(tag="B1_cross_soft0_causal_hd128_fp16", softmax=0, prec="fp16",
             hdim=128, seqlens=SEQ_Q6, seqlens_kv=SEQ_KV6, causal=1,
             local_len=0, context_len=0, minfull_len=0, targets=0),
        # B2 cross, softmax=1, no-causal, hd128, bf16
        dict(tag="B2_cross_soft1_nocausal_hd128_bf16", softmax=1, prec="bf16",
             hdim=128, seqlens=SEQ_Q6, seqlens_kv=SEQ_KV6, causal=0,
             local_len=0, context_len=0, minfull_len=0, targets=0),
        # B3 cross + full mask + target, hd128, fp16
        dict(tag="B3_cross_soft0_full_mask_hd128_fp16", softmax=0, prec="fp16",
             hdim=128, seqlens=SEQ_Q6, seqlens_kv=SEQ_KV6, causal=1,
             local_len=5, context_len=8, minfull_len=7, targets=TGT6),
    ]

    # --- C. 与 group 变长交叉一致性的「no-group 侧」case ---
    # 这里只给 no-group jagged 侧;对应的 group 侧命令行由 run_v6.py 已有的
    # test_group_* 覆盖(g=3),两侧用相同的变长分布(SEQ_Q6)以便人工/脚本核对
    # 「同一变长语义两条路径结果一致」。group 侧不在本脚本(g>1 会忽略 jagged)。
    group_consistency = [
        # C1 与 group 主力同轴:softmax=0, causal, hd128, bf16,变长分布同 SEQ_Q6
        dict(tag="C1_xcheck_group_soft0_causal_hd128_bf16", softmax=0,
             prec="bf16", hdim=128, seqlens=SEQ_Q6, causal=1, local_len=5,
             context_len=0, minfull_len=0, targets=0),
        # C2 与 group softmax 主力同轴:softmax=1, causal, hd128, fp16
        dict(tag="C2_xcheck_group_soft1_causal_hd128_fp16", softmax=1,
             prec="fp16", hdim=128, seqlens=SEQ_Q6, causal=1, local_len=5,
             context_len=0, minfull_len=0, targets=0),
    ]
    return core, cross, group_consistency


def all_cases():
    core, cross, gc = make_cases()
    return core + cross + gc


def exe_prefix_for(case):
    if case.get("softmax", 0) == 1:
        return [EXE, "-softmax=1"]
    return [EXE, "-softmax=0"]


# ---------------------------------------------------------------------------
# 静态 route 判定(与 run_v6.py:classify_route 同规则,独立实现,供自验)
# no-group: single <=> p_drop==0  (本脚本所有 case 均不给 -p_drop => p_drop=0)
# jagged=1 在 JAG-B1 后不再触发回落。g=1(不给 -g)=> no-group 路。
# ---------------------------------------------------------------------------

def static_route(case):
    # 本脚本从不设 p_drop、从不设 g>1 => 全部 no-group、p_drop=0 => single。
    return "single"


def run_one(case):
    cmd = build_cmd(exe_prefix_for(case), case)
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    print("")
    return result.returncode


def cmd_list():
    """静态自验:不跑 harness。打印每个 case 的 route + 「长度真的不等」检查。"""
    cases = all_cases()
    print("# jagged x single 新增 case 静态清单(未跑 harness)")
    print("# route 规则 = run_v6.py:classify_route(no-group: single<=>p_drop==0)")
    all_ok = True
    for i, c in enumerate(cases):
        b = _b(c)
        q = c["seqlens"]
        q_distinct = (len(set(q)) == len(q))          # q 两两不等
        q_match_b = (len(q) == b)                       # 元素数 == b(不触发补齐)
        line = (f"[{i:02d}] {c['tag']:44s} b={b} prec={c['prec']} "
                f"softmax={c.get('softmax',0)} causal={c['causal']} "
                f"hdim={c['hdim']} route={static_route(c)} "
                f"seqlens={q}")
        ok = q_distinct and q_match_b
        # cross:seqlens_kv 也要逐 batch 全写、与 b 等长、两两不等
        if c.get("seqlens_kv") is not None:
            kv = c["seqlens_kv"]
            kv_ok = (len(kv) == b) and (len(set(kv)) == len(kv))
            line += f" | kv={kv} kv_ok={kv_ok}"
            ok = ok and kv_ok
        flag = "OK " if ok else "BAD"
        if not ok:
            all_ok = False
        print(f"{flag} {line}")
    print(f"\n# 长度真的不等 & b 严格匹配 & 无静默补齐:{'全部通过' if all_ok else '存在 BAD'}")
    print(f"# 新增 case 总数 = {len(cases)}  "
          f"(core={len(make_cases()[0])} cross={len(make_cases()[1])} "
          f"group_xcheck={len(make_cases()[2])})")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="no-group jagged=1 走 single 的补充 case(卡 JAG-B2)。")
    parser.add_argument("--list", action="store_true",
                        help="静态自验:打印 case 清单 + route + 长度不等检查,不跑 harness")
    args, _ = parser.parse_known_args()

    if args.list:
        return cmd_list()

    rc = 0
    for case in all_cases():
        ret = run_one(case)
        if ret != 0:
            rc = ret
    return rc


if __name__ == "__main__":
    sys.exit(main())
