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
# CRYSTAL_PDB = "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/myc_crystal.pdb"
CRYSTAL_PDB = None 

OUT       = "Results/Individual"
SEL_CA    = "backbone and name CA"
COLORS    = ["#e74c3c", "#2ecc71", "#3498db"]
STRIDE    = 1
DT_PS     = 100 # ps between frames

os.makedirs(OUT, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def frames_to_ns(n_frames, ps_per_frame=DT_PS):
    return np.arange(n_frames) * ps_per_frame / 1000

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_rmsd(u, sel_ca, ref_positions=None):
    """Compute RMSD vs reference (crystal or first frame)."""
    ca = u.select_atoms(sel_ca)
    if ref_positions is None:
        ref_positions = ca.positions.copy()
    
    rmsd_vals = []
    for ts in u.trajectory[::STRIDE]:
        # Simple RMSD without alignment (assuming trajectory is pre-aligned or we align per frame)
        # For standard RMSD, we usually align to reference first
        val = np.sqrt(np.mean((ca.positions - ref_positions) ** 2))
        rmsd_vals.append(val)
    return np.array(rmsd_vals)

def analyze_rmsf(u, sel_ca):
    """Compute per-residue RMSF after aligning to the first frame."""
    align.AlignTraj(u, u, select=sel_ca, in_memory=True).run()
    ca = u.select_atoms(sel_ca)
    rmsf_obj = rms.RMSF(ca).run()
    return rmsf_obj.results.rmsf, ca.resids

def analyze_rg(u, sel_ca):
    """Compute Radius of Gyration over time."""
    ca = u.select_atoms(sel_ca)
    rg_vals = []
    for ts in u.trajectory[::STRIDE]:
        rg_vals.append(ca.radius_of_gyration())
    return np.array(rg_vals)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

results = {"rmsd": [], "rmsf": [], "rg": [], "resids": None, "time": None}

for i, xtc in enumerate(TRAJS):
    print(f"Processing Replica {i+1}: {xtc}")
    u = mda.Universe(PSF_FILE, xtc)
    
    if 1 in RUN:
        # If CRYSTAL_PDB provided, align and compute vs crystal
        if CRYSTAL_PDB and os.path.exists(CRYSTAL_PDB):
            u_ref = mda.Universe(CRYSTAL_PDB)
            ref_pos = u_ref.select_atoms(SEL_CA).positions
            # Alignment to crystal
            align.AlignTraj(u, u_ref, select=SEL_CA, in_memory=True).run()
            results["rmsd"].append(analyze_rmsd(u, SEL_CA, ref_pos))
        else:
            # Align to first frame
            u.trajectory[0]
            ref_pos = u.select_atoms(SEL_CA).positions.copy()
            align.AlignTraj(u, u, select=SEL_CA, in_memory=True).run()
            results["rmsd"].append(analyze_rmsd(u, SEL_CA, ref_pos))
            
    if 2 in RUN:
        rmsf, resids = analyze_rmsf(u, SEL_CA)
        results["rmsf"].append(rmsf)
        results["resids"] = resids
        
    if 3 in RUN:
        results["rg"].append(analyze_rg(u, SEL_CA))
        
    if results["time"] is None:
        results["time"] = frames_to_ns(len(u.trajectory))

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
    plt.savefig(f"{OUT}/01_rmsd.png", dpi=300)
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
    plt.savefig(f"{OUT}/02_rmsf.png", dpi=300)
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
    plt.savefig(f"{OUT}/03_rg.png", dpi=300)
    plt.close()

print(f"\nAnalysis complete. Results saved in '{OUT}/'")
