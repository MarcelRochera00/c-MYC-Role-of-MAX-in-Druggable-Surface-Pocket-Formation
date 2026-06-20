#!/usr/bin/env python3
"""
Individual Protein Analysis (Step 1 in Workflow)
================================================
Analyses: RMSD, RMSF, Radius of Gyration (Rg).
Requires: PSF and XTC files for single-protein replicas.
Outputs: Results/Individual/
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

# =============================================================================
# SELECT ANALYSES TO RUN
# =============================================================================
print("\nSelect analyses to run (leave blank to run all):")
print("  1. RMSD")
print("  2. RMSF")
print("  3. Radius of Gyration (Rg)")
print("Enter numbers separated by commas (e.g. 1,2) or press Enter for all: ")
_selection = input().strip()

try:
    RUN = {1, 2, 3} if _selection == "" else set(int(x.strip()) for x in _selection.split(","))
except ValueError:
    print("  Invalid input — running all analyses.")
    RUN = {1, 2, 3}

print(f"  → Running: {', '.join(str(x) for x in sorted(RUN))}\n")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Default paths for MYC alone replicas (adjust if needed)
PSF_FILE = "MYC/MYC-500/charmm-gui/gromacs/protein_only.psf"

TRAJS = [
    "MYC/MYC-500/charmm-gui/gromacs/run1/step6.0_protein_only_omit_100.xtc",
    "MYC/MYC-1000/charmm-gui/gromacs/run1/step6.0_protein_only_omit_100.xtc",
    "MYC/MYC-1500/charmm-gui/gromacs/run1/step6.0_protein_only_omit_100.xtc",
]

# Optional: Reference structure for RMSD (crystal or first frame)
CRYSTAL_PDB = "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.pdb"
#CRYSTAL_PDB = None 

OUT       = "Results/Individual"
COLORS    = ["#e74c3c", "#2ecc71", "#3498db"]
STRIDE    = 1
DT_PS     = 100 # ps between frames

os.makedirs(OUT, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def trajectory_time_ns(u):
    return np.array(
        [ts.time for ts in u.trajectory]
    ) / 1000.0

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_rmsd(u, ref_u):
    """
    Backbone RMSD against crystal reference.
    """

    rmsd_obj = rms.RMSD(
        u,
        ref_u,
        select="backbone"
    ).run()

    return rmsd_obj.results.rmsd[:, 2]

def analyze_rmsf(u):
    """
    CA RMSF after backbone alignment.
    """

    align.AlignTraj(
        u,
        u,
        select="backbone",
        in_memory=True
    ).run()

    ca = u.select_atoms("name CA")

    rmsf_obj = rms.RMSF(ca).run()

    return rmsf_obj.results.rmsf, ca.resids

def analyze_rg(u):
    """
    Whole-protein radius of gyration.
    """

    protein = u.select_atoms("protein")

    rg_vals = []

    for ts in u.trajectory[::STRIDE]:
        rg_vals.append(
            protein.radius_of_gyration()
        )

    return np.array(rg_vals)
# =============================================================================
# MAIN EXECUTION
# =============================================================================

results = {"rmsd": [], "rmsf": [], "rg": [], "resids": None, "time": None}

for i, xtc in enumerate(TRAJS):
    print(f"Processing Replica {i+1}: {xtc}")
    u_rmsd = mda.Universe(PSF_FILE, xtc)
    u_rmsf = mda.Universe(PSF_FILE, xtc)
    u_rg   = mda.Universe(PSF_FILE, xtc)
    
    if 1 in RUN:
        ref_u = mda.Universe(
            PSF_FILE,
            CRYSTAL_PDB
        )

        results["rmsd"].append(
            analyze_rmsd(u_rmsd, ref_u)
        )
            
    if 2 in RUN:
        rmsf, resids = analyze_rmsf(u_rmsf)
        results["rmsf"].append(rmsf)
        results["resids"] = resids
        
    if 3 in RUN:
        results["rg"].append(
            analyze_rg(u_rg)
        )
        
    if results["time"] is None:
        results["time"] = trajectory_time_ns(u_rmsd)

# =============================================================================
# PLOTTING
# =============================================================================
sns.set_style("whitegrid")

if 1 in RUN:
    plt.figure(figsize=(8, 5))
    for i, data in enumerate(results["rmsd"]):
        plt.plot(results["time"], data, color=COLORS[i], alpha=0.6, label=f"Rep {i+1}")
    plt.plot(results["time"], np.mean(results["rmsd"], axis=0), color="black", linewidth=2, label="Mean")
    plt.xlabel("Time (ns)")
    plt.ylabel("RMSD (Å)")
    plt.title("Individual Protein RMSD")
    plt.legend()
    plt.savefig(f"{OUT}/1.1_rmsd.png", dpi=300)
    plt.close()

if 2 in RUN:
    plt.figure(figsize=(8, 5))
    for i, data in enumerate(results["rmsf"]):
        plt.plot(results["resids"], data, color=COLORS[i], alpha=0.6, label=f"Rep {i+1}")
    plt.plot(results["resids"], np.mean(results["rmsf"], axis=0), color="black", linewidth=2, label="Mean")
    plt.xlabel("Residue Number")
    plt.ylabel("RMSF (Å)")
    plt.title("Individual Protein RMSF")
    plt.legend()
    plt.savefig(f"{OUT}/1.2_rmsf.png", dpi=300)
    plt.close()

if 3 in RUN:
    plt.figure(figsize=(8, 5))
    for i, data in enumerate(results["rg"]):
        plt.plot(results["time"], data, color=COLORS[i], alpha=0.6, label=f"Rep {i+1}")
    plt.plot(results["time"], np.mean(results["rg"], axis=0), color="black", linewidth=2, label="Mean")
    plt.xlabel("Time (ns)")
    plt.ylabel("Rg (Å)")
    plt.title("Individual Protein Radius of Gyration")
    plt.legend()
    plt.savefig(f"{OUT}/1.3_rg.png", dpi=300)
    plt.close()



# =============================================================================
# COMBINED PLOT
# =============================================================================

sns.set_style("whitegrid")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 14),
    constrained_layout=True
)

rep_colors = [
    "#8ecae6",   # light blue
    "#f4a340",   # orange
    "#c8a2d9"    # lavender
]

mean_color = "#2b2d42"

rep_labels = [
    "Replica 1 (500ps eq)",
    "Replica 2 (1000ps eq)",
    "Replica 3 (1500ps eq)"
]

ax = axes[0]

rmsd_arr = np.array(results["rmsd"])

mean_rmsd = rmsd_arr.mean(axis=0)
std_rmsd = rmsd_arr.std(axis=0)

for i, y in enumerate(rmsd_arr):
    ax.plot(
        results["time"],
        y,
        color=rep_colors[i],
        alpha=0.75,
        linewidth=1,
        label=rep_labels[i]
    )

ax.fill_between(
    results["time"],
    mean_rmsd - std_rmsd,
    mean_rmsd + std_rmsd,
    color="lightgray",
    alpha=0.5,
    label="± SD"
)

ax.plot(
    results["time"],
    mean_rmsd,
    color=mean_color,
    linewidth=4,
    label=f"Mean: {mean_rmsd.mean():.2f} Å"
)

ax.set_title(
    "RMSD vs Crystal State - Three Replicas",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Time (ns)")
ax.set_ylabel("RMSD (Å)")
ax.legend(loc="lower right")

ax = axes[1]

rmsf_arr = np.array(results["rmsf"])

mean_rmsf = rmsf_arr.mean(axis=0)
std_rmsf = rmsf_arr.std(axis=0)

for i, y in enumerate(rmsf_arr):
    ax.plot(
        results["resids"],
        y,
        color=rep_colors[i],
        alpha=0.75,
        linewidth=1,
        label=rep_labels[i]
    )

ax.fill_between(
    results["resids"],
    mean_rmsf - std_rmsf,
    mean_rmsf + std_rmsf,
    color="lightgray",
    alpha=0.5,
    label="± SD"
)

ax.plot(
    results["resids"],
    mean_rmsf,
    color=mean_color,
    linewidth=2.5,
    label=f"Mean: {mean_rmsf.mean():.2f} Å"
)

ax.axhline(
    3.0,
    linestyle="--",
    color="#d9534f",
    label="High flexibility (3 Å)"
)

ax.set_title(
    "Per-Residue Flexibility - Three Replicas",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Residue Number")
ax.set_ylabel("RMSF (Å)")
ax.legend(loc="upper right")

ax = axes[2]

rg_arr = np.array(results["rg"])

mean_rg = rg_arr.mean(axis=0)
std_rg = rg_arr.std(axis=0)

for i, y in enumerate(rg_arr):
    ax.plot(
        results["time"],
        y,
        color=rep_colors[i],
        alpha=0.75,
        linewidth=1,
        label=rep_labels[i]
    )

ax.fill_between(
    results["time"],
    mean_rg - std_rg,
    mean_rg + std_rg,
    color="lightgray",
    alpha=0.5,
    label="± SD"
)

ax.plot(
    results["time"],
    mean_rg,
    color=mean_color,
    linewidth=2.5,
    label=f"Mean: {mean_rg.mean():.2f} Å"
)

ax.set_title(
    "Radius of Gyration - Three Replicas",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Rg (Å)")
ax.legend(loc="upper right")

plt.savefig(
    f"{OUT}/1.4_combined_three_replica_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

with open(f"{OUT}/analysis_statistics.txt", "w") as f:

    f.write("MYC THREE-REPLICA ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    for i in range(3):

        f.write(f"Replica {i+1}\n")
        f.write("-" * 20 + "\n")

        if results["rmsd"]:
            f.write(
                f"RMSD mean = {np.mean(results['rmsd'][i]):.3f} Å\n"
            )
            f.write(
                f"RMSD std  = {np.std(results['rmsd'][i]):.3f} Å\n"
            )

        if results["rg"]:
            f.write(
                f"Rg mean   = {np.mean(results['rg'][i]):.3f} Å\n"
            )
            f.write(
                f"Rg std    = {np.std(results['rg'][i]):.3f} Å\n"
            )

        if results["rmsf"]:
            f.write(
                f"RMSF mean = {np.mean(results['rmsf'][i]):.3f} Å\n"
            )
            f.write(
                f"RMSF std  = {np.std(results['rmsf'][i]):.3f} Å\n"
            )

        f.write("\n")

    f.write("=" * 60 + "\n")

    f.write(
        f"Global RMSD mean ± SD = "
        f"{mean_rmsd.mean():.3f} ± {std_rmsd.mean():.3f} Å\n"
    )

    f.write(
        f"Global RMSF mean ± SD = "
        f"{mean_rmsf.mean():.3f} ± {std_rmsf.mean():.3f} Å\n"
    )

    f.write(
        f"Global Rg mean ± SD = "
        f"{mean_rg.mean():.3f} ± {std_rg.mean():.3f} Å\n"
    )
    
    
print(f"\nAnalysis complete. Results saved in '{OUT}/'")