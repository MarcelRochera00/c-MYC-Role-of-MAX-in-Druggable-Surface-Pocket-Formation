#!/usr/bin/env python3
"""
MDpocket Descriptors Analysis (Step 3 in Workflow)
==================================================
Analyses: Cavity volume timeseries, distribution, and residue composition.
Requires: _descriptors.txt files from MDpocket.
Outputs: Results/Pockets/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os

# ─────────────────────────────────────────────
# EDIT THESE PATHS WITH YOUR ACTUAL FILES
# ─────────────────────────────────────────────
REPLICA_FILES = {
    "Replica 1 (500 ns eq)":  "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/mdpocket_pocket1_stats_descriptors.txt",
    "Replica 2 (1000 ns eq)": "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/mdpocket_pocket1_stats_descriptors.txt",
    "Replica 3 (1500 ns eq)": "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/mdpocket_pocket1_stats_descriptors.txt",
}

TIMESTEP_NS = 0.1   # ns per snapshot (change if your dt is different)
OUTPUT_DIR  = "Results/Pockets"
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting params
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "legend.frameon":     False,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

COLORS  = ["#C0392B", "#2980B9", "#27AE60"]

AA_COLS = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
           "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
           "THR","TRP","TYR","VAL"]

HYDROPHOBIC = {"ALA","ILE","LEU","MET","PHE","TRP","TYR","VAL","PRO"}
POLAR       = {"ASN","CYS","GLN","SER","THR"}
CHARGED_POS = {"ARG","HIS","LYS"}
CHARGED_NEG = {"ASP","GLU"}

AA_COLORS = {
    "Hydrophobic":  "#E74C3C",
    "Polar":        "#3498DB",
    "Charged (+)":  "#F39C12",
    "Charged (-)":  "#27AE60",
    "Other":        "#95A5A6",
}

def classify_aa(aa):
    if aa in HYDROPHOBIC:  return "Hydrophobic"
    if aa in POLAR:        return "Polar"
    if aa in CHARGED_POS:  return "Charged (+)"
    if aa in CHARGED_NEG:  return "Charged (-)"
    return "Other"

METRICS = {
    "pock_volume":          "Pocket Volume (Å³)",
    "hydrophobicity_score": "Hydrophobicity Score",
    "polarity_score":       "Polarity Score",
    "charge_score":         "Charge Score",
}

def load_replica(path):
    df = pd.read_csv(path, sep=r"\s+")
    df["time_ns"] = df["snapshot"] * TIMESTEP_NS
    return df

def summary_table(dfs):
    rows = []
    for name, df in dfs.items():
        rows.append({
            "Replica": name,
            "Frames": len(df),
            "Mean Volume (Å³)": f"{df['pock_volume'].mean():.1f} ± {df['pock_volume'].std():.1f}",
            "Mean Hydrophobicity": f"{df['hydrophobicity_score'].mean():.2f} ± {df['hydrophobicity_score'].std():.2f}",
            "Mean Polarity": f"{df['polarity_score'].mean():.1f}",
            "Mean Charge": f"{df['charge_score'].mean():.1f}",
            "Volume > 400 Å³ (%)": f"{(df['pock_volume'] > 400).mean()*100:.1f}%",
        })
    return pd.DataFrame(rows)


def plot_timeseries(dfs):
    """4-panel timeseries: volume, hydrophobicity, polarity, charge."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    axes = axes.flatten()

    for ax_idx, (col, ylabel) in enumerate(METRICS.items()):
        ax = axes[ax_idx]
        ax.set_facecolor("white")

        for i, (name, df) in enumerate(dfs.items()):
            color   = COLORS[i % len(COLORS)]
            rolling = df[col].rolling(window=50, center=True).mean()
            ax.plot(df["time_ns"], df[col], alpha=0.12, color=color, linewidth=0.4)
            ax.plot(df["time_ns"], rolling, color=color, linewidth=1.8, label=name)
            ax.axhline(df[col].mean(), color=color, linestyle="--",
                       linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.legend()

    fig.suptitle("Myc-Max Pocket Dynamics — MDpocket Analysis",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "4.1_pocket_timeseries.png")
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def plot_volume_distribution(dfs):
    """Volume distribution histogram across replicas."""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    ax.set_facecolor("white")

    for i, (name, df) in enumerate(dfs.items()):
        color = COLORS[i % len(COLORS)]
        ax.hist(df["pock_volume"], bins=60, alpha=0.45, color=color,
                label=name, density=True, edgecolor="none")
        ax.axvline(df["pock_volume"].mean(), color=color,
                   linestyle="--", linewidth=1.5,
                   label=f"Mean {df['pock_volume'].mean():.0f} Å³")

    ax.set_xlabel("Pocket Volume (Å³)")
    ax.set_ylabel("Density")
    ax.set_title("Pocket Volume Distribution", fontweight="bold")
    ax.legend(ncol=2)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "4.2_volume_distribution.png")
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def plot_residue_composition(dfs):
    """Bar chart: mean residue count per frame, one panel per replica."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(5.5 * len(dfs), 5),
                             facecolor="white", sharey=True)
    if len(dfs) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, dfs.items()):
        ax.set_facecolor("white")
        aa_means   = df[AA_COLS].mean().sort_values(ascending=False)
        bar_colors = [AA_COLORS[classify_aa(aa)] for aa in aa_means.index]

        ax.bar(aa_means.index, aa_means.values, color=bar_colors,
               edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Residue type")
        ax.set_ylabel("Mean count per frame")
        ax.tick_params(axis="x", rotation=45)

    legend_elements = [
        mpatches.Patch(color=AA_COLORS["Hydrophobic"], label="Hydrophobic"),
        mpatches.Patch(color=AA_COLORS["Polar"],       label="Polar"),
        mpatches.Patch(color=AA_COLORS["Charged (+)"], label="Charged (+)"),
        mpatches.Patch(color=AA_COLORS["Charged (-)"], label="Charged (-)"),
    ]
    axes[-1].legend(handles=legend_elements, loc="upper right")
    fig.suptitle("Pocket Residue Composition — Mean per Frame",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "4.3_residue_composition.png")
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def plot_residue_heatmap(dfs):
    """Heatmap: residue occupancy over time per replica."""
    n   = len(dfs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n), facecolor="white")
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, dfs.items()):
        smoothed = df[AA_COLS].rolling(window=50, center=True).mean().T
        smoothed.columns = df["time_ns"]

        im = ax.imshow(smoothed.values, aspect="auto", cmap="YlOrRd",
                       extent=[df["time_ns"].min(), df["time_ns"].max(),
                                len(AA_COLS) - 0.5, -0.5])
        ax.set_yticks(range(len(AA_COLS)))
        ax.set_yticklabels(AA_COLS, fontsize=8)
        ax.set_xlabel("Time (ns)")
        ax.set_title(name, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
        cbar.set_label("Mean residue count", fontsize=8)

    fig.suptitle("Pocket Residue Occupancy Over Time",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "4.4_residue_heatmap.png")
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def residue_summary_table(dfs):
    """CSV: top residues averaged across replicas."""
    all_means = pd.DataFrame({
        name: df[AA_COLS].mean()
        for name, df in dfs.items()
    })
    replica_cols = list(dfs.keys())

    all_means["Mean (all replicas)"] = all_means[replica_cols].mean(axis=1)
    all_means["Std (all replicas)"] = all_means[replica_cols].std(axis=1)
    all_means["Property"] = [classify_aa(aa) for aa in all_means.index]
    all_means = all_means.sort_values("Mean (all replicas)", ascending=False)

    out = os.path.join(OUTPUT_DIR, "residue_summary.csv")
    all_means.to_csv(out)
    print(f"Saved: {out}")

    print("\n── Top 10 Pocket Residues (mean across replicas) ──")
    print(all_means[["Mean (all replicas)", "Std (all replicas)", "Property"]]
          .head(10).to_string())
    return all_means


def main():
    dfs = {}
    for name, path in REPLICA_FILES.items():
        if os.path.exists(path):
            dfs[name] = load_replica(path)
            print(f"Loaded {name}: {len(dfs[name])} frames")
        else:
            print(f"WARNING: File not found — {path} (skipping)")

    if not dfs:
        print("No files found! Check REPLICA_FILES paths.")
        return

    table = summary_table(dfs)
    print("\n── Summary Table ──────────────────────────────")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"), index=False)

    plot_timeseries(dfs)
    plot_volume_distribution(dfs)
    plot_residue_composition(dfs)
    plot_residue_heatmap(dfs)
    residue_summary_table(dfs)

    print(f"\nDone! All outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()