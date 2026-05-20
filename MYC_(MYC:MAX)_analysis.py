#!/usr/bin/env python3
"""
Comparative MD analysis: MYC alone vs MYC:MAX complex
Metrics: RMSD, RMSF, Radius of Gyration — 3 replicas each condition
Goal: Show MYC is stabilized upon MAX binding
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

# ============================================================
#                     CONFIGURATION
# ============================================================

CONDITIONS = {
    "MYC alone": {
        "replicas": [
            {"psf": "MYC/MYC-500/charmm-gui/gromacs/protein_only.psf",
             "xtc": "MYC/MYC-500/charmm-gui/gromacs/run1/step6.0_protein_only_omit_100.xtc"},
            {"psf": "MYC/MYC-500/charmm-gui/gromacs/protein_only.psf",
             "xtc": "MYC/MYC-1000/charmm-gui/gromacs/run1/step6.0_protein_only_omit_100.xtc"},
            {"psf": "MYC/MYC-500/charmm-gui/gromacs/protein_only.psf",
             "xtc": "MYC/MYC-1500/charmm-gui/gromacs/run1/step6.0_protein_only_omit_100.xtc"},
        ],
        "crystal_psf":   "MYC/MYC-500/charmm-gui/gromacs/protein_only.psf",
        "crystal_pdb":   "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.pdb",
        "color":         "#3498db",
        "mean_color":    "#1a5276",
        "label":         "MYC alone",
        # whole system is MYC only, so both selections are identical
        "ca_selection":  "backbone and name CA",
        "rg_selection":  "backbone and name CA",
    },
    "MYC:MAX isolated MYC": {
        "replicas": [
            {"psf": "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.psf",
             "xtc": "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/run1/step6.0_myc_only_omit_100.xtc"},
            {"psf": "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.psf",
             "xtc": "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/run1/step6.0_myc_only_omit_100.xtc"},
            {"psf": "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.psf",
             "xtc": "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/run1/step6.0_myc_only_omit_100.xtc"},
        ],
        "crystal_psf":   "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.psf",
        "crystal_pdb":   "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.pdb",
        "color":         "#e67e22",
        "mean_color":    "#784212",
        "label":         "MYC:MAX complex",
        # align & RMSD/RMSF on all CA in the (already MYC-only) trajectory
        "ca_selection":  "backbone and name CA",
        # trajectory is already MYC-only (myc_only xtc), so no segid filter needed
        "rg_selection":  "backbone and name CA",
    },
}

STRIDE      = 1    # use every frame (increase to 2/5/10 to speed up)
FRAME_DT_PS = 100  # ps between saved frames — adjust to your xtc frequency

OUT = "Results/Comparison"
os.makedirs(OUT, exist_ok=True)
OUTPUT_PLOT = os.path.join(OUT, "myc_vs_mycmax_comparison.png")
OUTPUT_DATA = os.path.join(OUT, "myc_vs_mycmax_statistics.txt")

REPLICA_ALPHAS = [0.55, 0.45, 0.35]


# ============================================================
#                     SECTION 1 — RMSD
# ============================================================

def compute_rmsd(u, ca_sel, ref_positions, stride):
    """Frame-by-frame RMSD of Ca atoms vs crystal reference positions."""
    ca = u.select_atoms(ca_sel)
    rmsd_vals = []
    for ts in u.trajectory[::stride]:
        val = np.sqrt(np.mean((ca.positions - ref_positions) ** 2))
        rmsd_vals.append(val)
    return np.array(rmsd_vals)


def plot_rmsd(ax, results_dict):
    """Plot RMSD for all conditions on a single axis."""
    for cname, r in results_dict.items():
        cfg  = CONDITIONS[cname]
        mean = np.mean(r["rmsd"], axis=0)
        std  = np.std(r["rmsd"],  axis=0)

        for i, rep_data in enumerate(r["rmsd"]):
            ax.plot(r["time_ns"], rep_data,
                    color=cfg["color"], linewidth=0.8,
                    alpha=REPLICA_ALPHAS[i])

        ax.plot(r["time_ns"], mean,
                color=cfg["mean_color"], linewidth=2.2,
                label=f"{cfg['label']}  mean = {np.mean(mean):.2f} Å")
        ax.fill_between(r["time_ns"], mean - std, mean + std,
                        color=cfg["mean_color"], alpha=0.12)

    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("RMSD (Å)",  fontsize=12, fontweight="bold")
    ax.set_title("Backbone RMSD vs Crystal State", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.25)


# ============================================================
#                     SECTION 2 — RMSF
# ============================================================

def compute_rmsf(u, ca_sel):
    """Per-residue RMSF over all frames, aligned to the first frame of this trajectory."""
    # Re-align to the trajectory's own first frame so RMSF reflects
    # internal fluctuations, not deviation from the crystal state.
    align.AlignTraj(u, u, select=ca_sel, in_memory=True).run()
    ca = u.select_atoms(ca_sel)
    rmsf_obj = rms.RMSF(ca).run()
    return rmsf_obj.results.rmsf, ca.resids


def plot_rmsf(ax, results_dict):
    """Plot RMSF for all conditions on a single axis."""
    for cname, r in results_dict.items():
        cfg  = CONDITIONS[cname]
        mean = np.mean(r["rmsf"], axis=0)
        std  = np.std(r["rmsf"],  axis=0)

        for i, rep_data in enumerate(r["rmsf"]):
            ax.plot(r["residues"], rep_data,
                    color=cfg["color"], linewidth=0.8,
                    alpha=REPLICA_ALPHAS[i])

        ax.plot(r["residues"], mean,
                color=cfg["mean_color"], linewidth=2.2,
                label=f"{cfg['label']}  mean = {np.mean(mean):.2f} Å")
        ax.fill_between(r["residues"], mean - std, mean + std,
                        color=cfg["mean_color"], alpha=0.12)

    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.0,
               alpha=0.6, label="High flexibility (3 Å)")
    ax.set_xlabel("Residue number", fontsize=11)
    ax.set_ylabel("RMSF (Å)",       fontsize=12, fontweight="bold")
    ax.set_title("Per-Residue Flexibility (RMSF, ref = first frame)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.25)


# ============================================================
#                  SECTION 3 — RADIUS OF GYRATION
# ============================================================

def compute_rg(u, rg_sel, stride):
    """Frame-by-frame radius of gyration for a given selection."""
    sel = u.select_atoms(rg_sel)
    rg_vals = []
    for ts in u.trajectory[::stride]:
        rg_vals.append(sel.radius_of_gyration())
    return np.array(rg_vals)


def plot_rg(ax, results_dict):
    """Plot Rg for all conditions on a single axis."""
    for cname, r in results_dict.items():
        cfg  = CONDITIONS[cname]
        mean = np.mean(r["rg"], axis=0)
        std  = np.std(r["rg"],  axis=0)

        for i, rep_data in enumerate(r["rg"]):
            ax.plot(r["time_ns"], rep_data,
                    color=cfg["color"], linewidth=0.8,
                    alpha=REPLICA_ALPHAS[i])

        ax.plot(r["time_ns"], mean,
                color=cfg["mean_color"], linewidth=2.2,
                label=f"{cfg['label']}  mean = {np.mean(mean):.2f} Å")
        ax.fill_between(r["time_ns"], mean - std, mean + std,
                        color=cfg["mean_color"], alpha=0.12)

    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Rg (Å)",    fontsize=12, fontweight="bold")
    ax.set_title("Radius of Gyration (MYC chain only)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.25)


# ============================================================
#               SECTION 4 — LOAD & PROCESS TRAJECTORIES
# ============================================================

def process_condition(cond_name, cond_cfg):
    """Run RMSD / RMSF / Rg for all replicas of one condition."""
    print(f"\n{'='*55}")
    print(f"  Processing: {cond_name}")
    print(f"{'='*55}")

    # Load crystal reference
    print("  Loading crystal state reference...")
    ref = mda.Universe(cond_cfg["crystal_psf"], cond_cfg["crystal_pdb"])
    ref_positions = ref.select_atoms(cond_cfg["ca_selection"]).positions.copy()

    all_rmsd, all_rmsf, all_rg = [], [], []
    residues = None
    time_ns  = None

    for idx, rep in enumerate(cond_cfg["replicas"]):
        print(f"  -> Replica {idx+1}: {rep['xtc']}")
        u = mda.Universe(rep["psf"], rep["xtc"])

        # Align trajectory to crystal state before computing any metric
        align.AlignTraj(u, ref, select=cond_cfg["ca_selection"],
                        in_memory=True).run()

        # RMSD — requires crystal-state alignment (done above)
        rmsd_vals = compute_rmsd(u, cond_cfg["ca_selection"],
                                 ref_positions, STRIDE)
        all_rmsd.append(rmsd_vals)

        # Rg — no alignment dependency, compute before RMSF re-aligns
        rg_vals = compute_rg(u, cond_cfg["rg_selection"], STRIDE)
        all_rg.append(rg_vals)

        # RMSF — must come last: internally re-aligns to first frame of
        # this trajectory, overwriting the in-memory crystal alignment.
        rmsf_vals, res_ids = compute_rmsf(u, cond_cfg["ca_selection"])
        all_rmsf.append(rmsf_vals)
        if residues is None:
            residues = res_ids

        if time_ns is None:
            n_frames = len(rmsd_vals)
            time_ns = np.array([ts.time for ts in u.trajectory[::STRIDE]]) / 1000.0

    return {
        "rmsd":     np.array(all_rmsd),
        "rmsf":     np.array(all_rmsf),
        "rg":       np.array(all_rg),
        "residues": residues,
        "time_ns":  time_ns,
    }


# ============================================================
#                  SECTION 5 — FIGURE ASSEMBLY
# ============================================================

def build_figure(results_dict):
    """Assemble 3-row figure and save."""
    fig = plt.figure(figsize=(15, 13))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)
    ax_rmsd, ax_rmsf, ax_rg = [fig.add_subplot(gs[i]) for i in range(3)]

    plot_rmsd(ax_rmsd, results_dict)
    plot_rmsf(ax_rmsf, results_dict)
    plot_rg  (ax_rg,   results_dict)

    fig.suptitle(
        "MYC alone vs MYC:MAX — MD Trajectory Comparison\n"
        "(Ca atoms, 3 replicas each, vs crystal state)",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"\n  Plot saved -> {OUTPUT_PLOT}")


# ============================================================
#                  SECTION 6 — STATISTICS OUTPUT
# ============================================================

def write_stats(results_dict):
    """Print and save comparative statistics table."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MYC ALONE vs MYC:MAX — COMPARATIVE STATISTICS")
    lines.append("  Reference: individual crystal states")
    lines.append("=" * 70)

    summary_rows = []

    for cname, r in results_dict.items():
        lines.append(f"\n{'-'*70}")
        lines.append(f"  Condition: {cname}")
        lines.append(f"{'-'*70}")

        rep_rmsd_means, rep_rg_means = [], []

        for i in range(len(r["rmsd"])):
            rmsd_m = np.mean(r["rmsd"][i]); rmsd_s = np.std(r["rmsd"][i])
            rmsf_m = np.mean(r["rmsf"][i]); rmsf_s = np.std(r["rmsf"][i])
            rg_m   = np.mean(r["rg"][i]);   rg_s   = np.std(r["rg"][i])
            lines.append(f"  Replica {i+1}:")
            lines.append(f"    RMSD : {rmsd_m:.2f} +/- {rmsd_s:.2f} Å")
            lines.append(f"    RMSF : {rmsf_m:.2f} +/- {rmsf_s:.2f} Å")
            lines.append(f"    Rg   : {rg_m:.3f} +/- {rg_s:.3f} Å")
            rep_rmsd_means.append(rmsd_m)
            rep_rg_means.append(rg_m)

        g_rmsd = np.mean(rep_rmsd_means); g_rmsd_sd = np.std(rep_rmsd_means)
        g_rmsf = np.mean(r["rmsf"]);      g_rmsf_sd = np.std(r["rmsf"])
        g_rg   = np.mean(rep_rg_means);   g_rg_sd   = np.std(rep_rg_means)

        lines.append(f"\n  Global (across replicas):")
        lines.append(f"    RMSD : {g_rmsd:.2f} +/- {g_rmsd_sd:.2f} Å")
        lines.append(f"    RMSF : {g_rmsf:.2f} +/- {g_rmsf_sd:.2f} Å")
        lines.append(f"    Rg   : {g_rg:.3f} +/- {g_rg_sd:.3f} Å")

        summary_rows.append((cname, g_rmsd, g_rmsd_sd,
                              g_rmsf, g_rmsf_sd,
                              g_rg,   g_rg_sd))

    # Delta row
    if len(summary_rows) == 2:
        a, b = summary_rows[0], summary_rows[1]
        lines.append(f"\n{'='*70}")
        lines.append("  Delta (MYC:MAX - MYC alone)  —  negative = stabilized by MAX")
        lines.append(f"{'-'*70}")
        lines.append(f"  DELTA RMSD : {b[1]-a[1]:+.2f} Å")
        lines.append(f"  DELTA RMSF : {b[3]-a[3]:+.2f} Å")
        lines.append(f"  DELTA Rg   : {b[5]-a[5]:+.3f} Å")

    lines.append(f"\n{'='*70}\n")
    text = "\n".join(lines)

    with open(OUTPUT_DATA, "w") as fh:
        fh.write(text)
    print(text)
    print(f"  Statistics saved -> {OUTPUT_DATA}")


# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  MYC alone vs MYC:MAX — Comparative MD Analysis")
    print("=" * 55)

    # Section 4: load and process all trajectories
    results = {}
    for cond_name, cond_cfg in CONDITIONS.items():
        results[cond_name] = process_condition(cond_name, cond_cfg)

    # Section 5: build and save figure
    build_figure(results)

    # Section 6: print and save statistics
    write_stats(results)

    plt.show()
    print("\nDone.")