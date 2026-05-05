#!/usr/bin/env python3
"""Plot sparge perf charts from full_grid.csv.

Re-run with different fixed (b, h, s, dtype, topk) by editing the constants below.
No GPU / no srun / no rebuild — pure matplotlib from CSV.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Tunable constants — edit these to regenerate for a different point.
# ----------------------------------------------------------------------
CSV_PATH        = "/home/AMD/ginolu12/gino_tmp/full_grid.csv"
OUT_DIR         = os.path.dirname(os.path.abspath(__file__))

# Chart 1 — speedup vs topk for one fixed (b, h, s, dtype)
CHART1_B        = 2
CHART1_H        = 32
CHART1_S        = 16384
CHART1_DTYPE    = "fp16"
CHART1_HEAD_DIM = 128  # for title only

# Chart 2 — kernel breakdown across s for fixed (b, h, dtype, topk)
CHART2_B        = 2
CHART2_H        = 32
CHART2_DTYPE    = "fp16"
CHART2_TOPK     = 0.4
CHART2_S_LIST   = [2048, 4096, 8192, 16384]
CHART2_HEAD_DIM = 128  # for title only

DPI = 140

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def is_fail(note):
    if not isinstance(note, str):
        return False
    return "FAIL" in note

def is_high_spread(note):
    if not isinstance(note, str):
        return False
    return "HIGH_SPREAD" in note

def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

# ----------------------------------------------------------------------
# Chart 1
# ----------------------------------------------------------------------
def plot_chart1(df, out_path):
    sel = df[
        (df["b"] == CHART1_B)
        & (df["h"] == CHART1_H)
        & (df["s"] == CHART1_S)
        & (df["dtype"] == CHART1_DTYPE)
    ].copy()
    sel = sel.sort_values("topk").reset_index(drop=True)

    if sel.empty:
        print(f"[chart1] WARNING: no rows for b={CHART1_B} h={CHART1_H} s={CHART1_S} dtype={CHART1_DTYPE}")
        return [], 0

    # Drop fully failed rows but keep partial-fail rows; we'll mask per-series.
    # Convert numeric columns
    for col in ["sparge_jenga", "sparge_vsa", "sparse_jenga", "sparse_vsa", "fmha_us"]:
        sel[col] = pd.to_numeric(sel[col], errors="coerce")

    fmha = sel["fmha_us"]

    # Compute speedups; rows with FAIL on a given column will have NaN already.
    series = {
        "sparge_vsa":   fmha / sel["sparge_vsa"],
        "sparge_jenga": fmha / sel["sparge_jenga"],
        "sparse_vsa":   fmha / sel["sparse_vsa"],
        "sparse_jenga": fmha / sel["sparse_jenga"],
    }

    style = {
        "sparge_vsa":   {"color": "#1f77b4", "marker": "o", "lw": 2.0},
        "sparge_jenga": {"color": "#ff7f0e", "marker": "s", "lw": 2.0},
        "sparse_vsa":   {"color": "#2ca02c", "marker": "^", "lw": 1.5, "ls": "--"},
        "sparse_jenga": {"color": "#d62728", "marker": "v", "lw": 1.5, "ls": "--"},
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=DPI)

    x = sel["topk"].to_numpy()

    # HIGH_SPREAD overlay first (under main markers)
    hs_mask = sel["note"].apply(is_high_spread)
    high_spread_cells = []
    if hs_mask.any():
        for _, row in sel[hs_mask].iterrows():
            high_spread_cells.append((row["topk"], row["max_spread_pct"]))
        # gray ring underneath every series's data point at that x
        for label, sp in series.items():
            xs_hs = x[hs_mask.to_numpy()]
            ys_hs = sp[hs_mask.to_numpy()].to_numpy()
            ax.scatter(xs_hs, ys_hs, s=180, facecolors="none",
                       edgecolors="gray", linewidths=1.5, zorder=2)

    for label, sp in series.items():
        st = style[label]
        ax.plot(x, sp.to_numpy(), label=label,
                color=st["color"], marker=st["marker"],
                linewidth=st["lw"], linestyle=st.get("ls", "-"),
                markersize=7, zorder=3)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="fmha (baseline)", zorder=1)

    ax.set_xlabel("topk (kept fraction)")
    ax.set_ylabel("speedup vs FMHA dense (×)")
    ax.set_title(
        f"Speedup vs FMHA "
        f"(b={CHART1_B} h={CHART1_H} s={CHART1_S} d={CHART1_HEAD_DIM} {CHART1_DTYPE})"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_xticks(np.arange(0.1, 0.71, 0.1))
    ax.legend(loc="best", framealpha=0.9)

    # Footnote about HIGH_SPREAD overlay
    if high_spread_cells:
        ax.text(0.01, -0.16,
                "Gray rings: HIGH_SPREAD cells (high run-to-run variance)",
                transform=ax.transAxes, fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return high_spread_cells, os.path.getsize(out_path)


# ----------------------------------------------------------------------
# Chart 2
# ----------------------------------------------------------------------
def plot_chart2(df, out_path):
    sel = df[
        (df["b"] == CHART2_B)
        & (df["h"] == CHART2_H)
        & (df["dtype"] == CHART2_DTYPE)
        & (np.isclose(df["topk"], CHART2_TOPK))
        & (df["s"].isin(CHART2_S_LIST))
    ].copy()
    sel = sel.sort_values("s").reset_index(drop=True)

    if sel.empty:
        print(f"[chart2] WARNING: no rows for b={CHART2_B} h={CHART2_H} dtype={CHART2_DTYPE} topk={CHART2_TOPK}")
        return 0

    for col in ["sparge_jenga_pre", "sparge_jenga_attn",
                "sparge_vsa_pre", "sparge_vsa_attn", "fmha_us"]:
        sel[col] = pd.to_numeric(sel[col], errors="coerce")

    s_vals = sel["s"].to_numpy()
    n = len(s_vals)
    idx = np.arange(n, dtype=float)

    width = 0.35
    offset = width / 2 + 0.02

    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=DPI)

    # Jenga bars (left of group)
    jenga_pre  = sel["sparge_jenga_pre"].to_numpy()
    jenga_attn = sel["sparge_jenga_attn"].to_numpy()
    vsa_pre    = sel["sparge_vsa_pre"].to_numpy()
    vsa_attn   = sel["sparge_vsa_attn"].to_numpy()
    fmha_vals  = sel["fmha_us"].to_numpy()

    color_jenga_pre  = "#fdbf6f"  # light orange
    color_jenga_attn = "#ff7f0e"  # orange
    color_vsa_pre    = "#a6cee3"  # light blue
    color_vsa_attn   = "#1f77b4"  # blue

    bj_pre = ax.bar(idx - offset, jenga_pre, width,
                    color=color_jenga_pre, edgecolor="black", linewidth=0.6,
                    label="sparge_jenga _pre (BlockMap)")
    bj_at  = ax.bar(idx - offset, jenga_attn, width, bottom=jenga_pre,
                    color=color_jenga_attn, edgecolor="black", linewidth=0.6,
                    label="sparge_jenga _attn")
    bv_pre = ax.bar(idx + offset, vsa_pre, width,
                    color=color_vsa_pre, edgecolor="black", linewidth=0.6,
                    label="sparge_vsa _pre (BlockMap)")
    bv_at  = ax.bar(idx + offset, vsa_attn, width, bottom=vsa_pre,
                    color=color_vsa_attn, edgecolor="black", linewidth=0.6,
                    label="sparge_vsa _attn")

    # Add total labels on top of each stack
    totals_jenga = jenga_pre + jenga_attn
    totals_vsa   = vsa_pre + vsa_attn
    for i in range(n):
        ax.text(idx[i] - offset, totals_jenga[i], f"{totals_jenga[i]:.0f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(idx[i] + offset, totals_vsa[i], f"{totals_vsa[i]:.0f}",
                ha="center", va="bottom", fontsize=8)

    # FMHA reference: short horizontal dashed segment per group
    seg_half = 0.40
    fmha_label_done = False
    for i in range(n):
        ax.hlines(fmha_vals[i], idx[i] - seg_half, idx[i] + seg_half,
                  colors="black", linestyles="dashed", linewidth=1.2,
                  label="fmha dense (reference)" if not fmha_label_done else None,
                  zorder=5)
        ax.text(idx[i] + seg_half + 0.02, fmha_vals[i],
                f"fmha {fmha_vals[i]:.0f}", fontsize=7, va="center", color="black")
        fmha_label_done = True

    ax.set_xticks(idx)
    ax.set_xticklabels([f"s={s}" for s in s_vals.astype(int)])
    ax.set_xlabel("sequence length (s)")
    ax.set_ylabel("kernel time (µs)")
    ax.set_title(
        f"Sparge kernel time breakdown "
        f"(b={CHART2_B} h={CHART2_H} d={CHART2_HEAD_DIM} {CHART2_DTYPE}, topk={CHART2_TOPK})"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # log-y is too aggressive — leave linear; bars will just be tall.
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return os.path.getsize(out_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    chart1_path = os.path.join(OUT_DIR, "speedup_vs_sparsity.png")
    chart2_path = os.path.join(OUT_DIR, "kernel_breakdown.png")

    hs_cells, size1 = plot_chart1(df, chart1_path)
    size2 = plot_chart2(df, chart2_path)

    print(f"Wrote {chart1_path} ({size1} bytes)")
    print(f"Wrote {chart2_path} ({size2} bytes)")

    if hs_cells:
        print("HIGH_SPREAD cells in chart-1 selection:")
        for topk, pct in hs_cells:
            print(f"  topk={topk} max_spread_pct={pct}")
    else:
        print("No HIGH_SPREAD cells in chart-1 selection.")


if __name__ == "__main__":
    main()
