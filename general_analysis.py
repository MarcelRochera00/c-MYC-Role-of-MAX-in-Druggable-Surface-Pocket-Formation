#!/usr/bin/env python3
"""
MYC-MAX Complex Analysis (Step 2 in Workflow)
=============================================
Analyses: RMSD, RMSF, Interface Contacts, MM-GBSA + Hotspots, Salt Bridges.
Requires: PSF and XTC files for complex replicas.
Outputs: Results/Complex/
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import subprocess
import datetime
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import distance_array

# =============================================================================
# SELECT ANALYSES TO RUN
# =============================================================================
print("\nSelect analyses to run (leave blank to run all):")
print("  1. RMSD")
print("  2. RMSF")
print("  3. Interface Contacts")
print("  4. MM-GBSA + Hotspot Decomposition")
print("  5. Salt Bridge Occupancy")
print("Enter numbers separated by commas (e.g. 1,3) or press Enter for all: ")
_selection = input().strip()

try:
    RUN = {1, 2, 3, 4, 5} if _selection == "" else set(int(x.strip()) for x in _selection.split(","))
except ValueError:
    print("  Invalid input — running all analyses.")
    RUN = {1, 2, 3, 4, 5}

print(f"  → Running: {', '.join(str(x) for x in sorted(RUN))}\n")

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PSF_FILE    = os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.psf")
CRYSTAL_PSF = os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.psf")
CRYSTAL_PDB = os.path.join(PROJECT_ROOT, "crystal_proteins.pdb")

TRAJS = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/run1/step6.0_proteins_only_omit_100.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/run1/step6.0_proteins_only_omit_100.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/run1/step6.0_proteins_only_omit_100.xtc"),
]

EQUIL = [500, 1000, 1500]

TPR_FILES = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/run1/step6.1_production.tpr"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/run1/step6.1_production.tpr"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/run1/step6.1_production.tpr"),
]

FULL_TRAJS = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/run1/step6.0_production_omit_100.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/run1/step6.0_production_omit_100.xtc"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/run1/step6.0_production_omit_100.xtc"),
]

TOPOL_FILES = [
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/topol.top"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/topol.top"),
    os.path.join(PROJECT_ROOT, "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/topol.top"),
]

SEL_ALL      = "protein and name CA"
SEL_MYC      = "segid PROA and name CA"
SEL_MAX      = "segid PROB and name CA"
CONTACT_DIST = 4.5
OUT          = os.path.join(PROJECT_ROOT, "Results/Complex")
BASE_PATH    = os.path.join(PROJECT_ROOT, "MYC-MAX")
INDEX_FILE   = os.path.abspath(os.path.join(BASE_PATH, "MYC-MAX-1000/charmm-gui/gromacs/index-new.ndx"))

# Number of top hotspot residues to show in the decomposition plots
TOP_N_HOTSPOTS = 20

# Salt bridge parameters
SALT_DIST      = 4.0   # Å — standard geometric cutoff for salt bridges
SALT_OCCUPANCY = 0.30  # only report bridges present in >30% of frames

# =============================================================================

os.makedirs(OUT, exist_ok=True)
COLORS = ["#e74c3c", "#2ecc71", "#3498db"]

def frames_to_ns(n_frames, ps_per_frame=50):
    return np.arange(n_frames) * ps_per_frame / 1000


# =============================================================================
# PARSE HELPERS
# =============================================================================

def parse_mmpbsa_csv(csv_file):
    """Parse gmx_MMPBSA CSV — extract TOTAL column from Delta Energy Terms section."""
    deltas = []
    in_delta = False
    header_next = False
    total_idx = None

    with open(csv_file) as f:
        for line in f:
            line = line.strip()
            if "Delta Energy Terms" in line:
                in_delta = True
                header_next = True
                continue
            if in_delta and header_next:
                cols = line.split(",")
                try:
                    total_idx = cols.index("TOTAL")
                except ValueError:
                    in_delta = False
                    continue
                header_next = False
                continue
            if in_delta and line == "":
                in_delta = False
                continue
            if in_delta and total_idx is not None:
                parts = line.split(",")
                try:
                    deltas.append(float(parts[total_idx]))
                except (ValueError, IndexError):
                    continue

    return np.array(deltas)


def parse_decomp_csv(csv_file):
    """
    Parse gmx_MMPBSA decomposition CSV.
    Format: Frame #,Residue,Internal,van der Waals,Electrostatic,Polar Solvation,Non-Polar Solv.,TOTAL
    Residue label format: R:A:ARG:913 — chain A = MYC, chain B = MAX.
    Averages TOTAL across all frames per residue.
    """
    residue_totals = {}
    in_tdc = False

    with open(csv_file) as f:
        for line in f:
            line = line.strip().strip('"')
            if not line:
                continue
            if line.startswith("Frame #,Residue"):
                in_tdc = True
                continue
            if in_tdc and (not line[0].isdigit()):
                in_tdc = False
                continue
            if not in_tdc:
                continue

            parts = line.split(",")
            if len(parts) < 7:
                continue

            try:
                int(parts[0])
            except ValueError:
                continue

            res_raw = parts[1].strip()
            try:
                total = float(parts[-1])
            except (ValueError, IndexError):
                continue

            if res_raw not in residue_totals:
                residue_totals[res_raw] = []
            residue_totals[res_raw].append(total)

    return {k: np.mean(v) for k, v in residue_totals.items()}


def label_residue(res_raw):
    """
    Parse R:A:ARG:913 format into a clean label and protein name.
    Chain A = MYC (PROA), Chain B = MAX (PROB).
    """
    parts = res_raw.split(":")
    if len(parts) >= 4:
        chain   = parts[1].upper()
        resname = parts[2]
        resid   = parts[3]
        protein = "MYC" if chain == "A" else "MAX" if chain == "B" else "?"
        label   = f"{protein}:{resname}{resid}"
    else:
        protein = "?"
        label   = res_raw
    return label, protein


# =============================================================================
# 1. RMSD
# =============================================================================
if 1 in RUN:
    print("\n[1/5] Calculating RMSD...")
    try:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))

        for i, (traj, eq) in enumerate(zip(TRAJS, EQUIL)):
            try:
                u   = mda.Universe(PSF_FILE, traj)
                ref = mda.Universe(CRYSTAL_PSF, CRYSTAL_PDB)

                align.AlignTraj(u, ref, select=SEL_ALL, in_memory=True).run()

                myc_ref  = ref.select_atoms(SEL_MYC)
                max_ref  = ref.select_atoms(SEL_MAX)
                myc_traj = u.select_atoms(SEL_MYC)
                max_traj = u.select_atoms(SEL_MAX)

                rmsd_myc, rmsd_max = [], []
                for ts in u.trajectory:
                    rmsd_myc.append(np.sqrt(np.mean((myc_traj.positions - myc_ref.positions)**2)))
                    rmsd_max.append(np.sqrt(np.mean((max_traj.positions - max_ref.positions)**2)))

                time  = frames_to_ns(len(rmsd_myc))
                label = f"Rep {i+1} (eq {eq}ps)"
                axes[0].plot(time, rmsd_myc, color=COLORS[i], linewidth=0.8, alpha=0.85, label=label)
                axes[1].plot(time, rmsd_max, color=COLORS[i], linewidth=0.8, alpha=0.85, label=label)

                print(f"  Rep {i+1} — MYC RMSD: {np.mean(rmsd_myc):.2f} ± {np.std(rmsd_myc):.2f} Å | "
                      f"MAX RMSD: {np.mean(rmsd_max):.2f} ± {np.std(rmsd_max):.2f} Å")
            except Exception as e:
                print(f"  ✗ Rep {i+1} RMSD failed: {e}")
                continue

        for ax, title in zip(axes, ["MYC (PROA) — RMSD vs crystal state",
                                      "MAX (PROB) — RMSD vs crystal state"]):
            ax.set_ylabel("RMSD (Å)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time (ns)", fontsize=11)
        plt.suptitle("Individual RMSD — MYC-MAX Complex (3 replicas)", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(f"{OUT}/01_rmsd.png", dpi=150)
        plt.close()
        print(f"  → Saved {OUT}/01_rmsd.png")
    except Exception as e:
        print(f"  ✗ RMSD section failed: {e}")


# =============================================================================
# 2. RMSF
# =============================================================================
if 2 in RUN:
    print("\n[2/5] Calculating RMSF...")
    try:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))

        for i, (traj, eq) in enumerate(zip(TRAJS, EQUIL)):
            try:
                u = mda.Universe(PSF_FILE, traj)

                u.trajectory[0]
                ref = mda.Universe(PSF_FILE, traj)
                ref.trajectory[0]

                align.AlignTraj(u, ref, select=SEL_ALL, in_memory=True).run()

                myc_atoms = u.select_atoms(SEL_MYC)
                max_atoms = u.select_atoms(SEL_MAX)

                rmsf_myc = rms.RMSF(myc_atoms).run().results.rmsf
                rmsf_max = rms.RMSF(max_atoms).run().results.rmsf

                label = f"Rep {i+1} (eq {eq}ps)"
                axes[0].plot(myc_atoms.resids, rmsf_myc, color=COLORS[i], linewidth=1.2, alpha=0.85, label=label)
                axes[1].plot(max_atoms.resids, rmsf_max, color=COLORS[i], linewidth=1.2, alpha=0.85, label=label)
                print(f"  Rep {i+1} RMSF done")
            except Exception as e:
                print(f"  ✗ Rep {i+1} RMSF failed: {e}")
                continue

        for ax, title in zip(axes, ["MYC (PROA) — Per-residue RMSF",
                                      "MAX (PROB) — Per-residue RMSF"]):
            ax.axhline(3.0, color="red", linestyle="--", alpha=0.5, linewidth=1, label="High flexibility (3 Å)")
            ax.set_ylabel("RMSF (Å)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        axes[0].set_xlabel("MYC Residue number", fontsize=11)
        axes[1].set_xlabel("MAX Residue number", fontsize=11)
        plt.suptitle("Per-residue RMSF — MYC-MAX Complex (3 replicas, ref = first frame)", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(f"{OUT}/02_rmsf.png", dpi=150)
        plt.close()
        print(f"  → Saved {OUT}/02_rmsf.png")
    except Exception as e:
        print(f"  ✗ RMSF section failed: {e}")


# =============================================================================
# 3. Interface Contacts
# =============================================================================
if 3 in RUN:
    print("\n[3/5] Calculating interface contacts...")
    try:
        fig, ax = plt.subplots(figsize=(13, 5))
        all_contacts = []

        for i, (traj, eq) in enumerate(zip(TRAJS, EQUIL)):
            try:
                u    = mda.Universe(PSF_FILE, traj)
                myc  = u.select_atoms("segid PROA")
                max_ = u.select_atoms("segid PROB")

                contacts = []
                for ts in u.trajectory:
                    dists      = distance_array(myc.positions, max_.positions)
                    n_contacts = int((dists < CONTACT_DIST).any(axis=1).sum())
                    contacts.append(n_contacts)

                time     = frames_to_ns(len(contacts))
                contacts = np.array(contacts)
                all_contacts.append(contacts)

                ax.plot(time, contacts, color=COLORS[i], linewidth=0.8, alpha=0.8,
                        label=f"Rep {i+1} (eq {eq}ps) — mean: {contacts.mean():.0f}")
                print(f"  Rep {i+1} — mean contacts: {contacts.mean():.1f} ± {contacts.std():.1f}")
            except Exception as e:
                print(f"  ✗ Rep {i+1} contacts failed: {e}")
                continue

        if all_contacts:
            min_len       = min(len(c) for c in all_contacts)
            mean_contacts = np.mean([c[:min_len] for c in all_contacts], axis=0)
            ax.plot(frames_to_ns(min_len), mean_contacts, color="#1d3557", linewidth=1.8,
                    linestyle="--", label="Mean across replicas", zorder=5)

        ax.set_xlabel("Time (ns)", fontsize=11)
        ax.set_ylabel(f"Interface contacts (< {CONTACT_DIST} Å)", fontsize=11)
        ax.set_title("MYC-MAX Interface Contacts Over Time (3 replicas)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT}/03_interface_contacts.png", dpi=150)
        plt.close()
        print(f"  → Saved {OUT}/03_interface_contacts.png")

        if all_contacts:
            fig, ax = plt.subplots(figsize=(7, 4))
            means = [c.mean() for c in all_contacts]
            stds  = [c.std()  for c in all_contacts]
            ax.bar([f"Rep {i+1}\n(eq {EQUIL[i]}ps)" for i in range(len(all_contacts))],
                   means, yerr=stds, color=COLORS[:len(all_contacts)], edgecolor="k",
                   linewidth=0.7, error_kw={"elinewidth": 1.5}, capsize=5)
            ax.set_ylabel(f"Mean interface contacts (< {CONTACT_DIST} Å)", fontsize=11)
            ax.set_title("Mean MYC-MAX Interface Contacts per Replica", fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUT}/03b_contacts_summary.png", dpi=150)
            plt.close()
            print(f"  → Saved {OUT}/03b_contacts_summary.png")
    except Exception as e:
        print(f"  ✗ Contacts section failed: {e}")


# =============================================================================
# 4. MM-GBSA + Hotspot Decomposition
# =============================================================================
if 4 in RUN:
    print("\n[4/5] Running MM-GBSA + Hotspot Decomposition...")

    # KEY FIXES vs previous version:
    #   igb=2    — GBHCT model; validated for charged PPIs. igb=5 (OBC2) was
    #              over-estimating electrostatics at the MYC-MAX interface.
    #   idecomp=1 — avoids 1-4 pair double-counting that inflates per-residue
    #              energies with idecomp=2.
    #   interval=50 — reduces correlated-frame artefacts. 10 was too dense.
    #   &pb block — runs in parallel for cross-validation. If GB and PB
    #              agree within ~10 kcal/mol the result is trustworthy;
    #              large divergence means GB is still overcounting.
    MMPBSA_INPUT = """&general
startframe=1,
interval=50,
PBRadii=7,
/
&pb
npbopt=1,        
istrng=0.15,
fillratio=4.0,
radiopt=0,
/
&decomp
idecomp=2,
dec_verbose=3,
/
"""

    mmpbsa_results    = []
    mmpbsa_pb_results = []   # PB cross-validation
    decomp_results    = []

    for i, (tpr, xtc, topol, eq) in enumerate(zip(TPR_FILES, TRAJS, TOPOL_FILES, EQUIL)):
        print(f"\n  Replica {i+1} (eq {eq}ps) started at {datetime.datetime.now().strftime('%H:%M:%S')}")
        sys.stdout.flush()

        rep_dir = os.path.abspath(f"{OUT}/mmpbsa_rep{i+1}")
        os.makedirs(rep_dir, exist_ok=True)

        inp_file = f"{rep_dir}/mmpbsa.in"
        with open(inp_file, "w") as f:
            f.write(MMPBSA_INPUT)

        cmd = [
            "mpirun", "-np", "8", "--allow-run-as-root",
            "gmx_MMPBSA", "MPI",
            "-O",
            "-i",  inp_file,
            "-cs", os.path.abspath(tpr),
            "-ct", os.path.abspath(xtc),
            "-ci", INDEX_FILE,
            "-cg", "16", "17",
            "-cp", os.path.abspath(topol),
            "-o",  f"{rep_dir}/FINAL_RESULTS_MMPBSA.dat",
            "-eo", f"{rep_dir}/FINAL_RESULTS_MMPBSA.csv",
            "-deo", f"{rep_dir}/DECOMP_RESULTS.csv",
        ]

        print(f"  Command: {' '.join(cmd)}")
        sys.stdout.flush()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=25200,
                                    env=os.environ.copy())
            print(f"  Finished at {datetime.datetime.now().strftime('%H:%M:%S')}")

            # --- GB binding energy ---
            csv_file = f"{rep_dir}/FINAL_RESULTS_MMPBSA.csv"
            if os.path.exists(csv_file):
                data = parse_mmpbsa_csv(csv_file)
                if len(data) > 0:
                    mmpbsa_results.append(data)
                    mean_val = data.mean()
                    print(f"  ✓ Replica {i+1} — ΔG (GB): {mean_val:.2f} ± {data.std():.2f} kcal/mol", end="")
                    # Sanity check: expected range for a PPI is roughly -20 to -50 kcal/mol
                    if mean_val < -80:
                        print(f"  ⚠ WARNING: value < -80 suggests GB is still overcounting. "
                              f"Check PB result and consider raising interval further.")
                    elif mean_val > 0:
                        print(f"  ⚠ WARNING: positive ΔG — check index groups and topology.")
                    else:
                        print()
                else:
                    print(f"  ✗ Replica {i+1} — CSV exists but no Delta data found")
                    print(result.stderr[-500:])
            else:
                print(f"  ✗ Replica {i+1} — no CSV generated")
                print(result.stderr[-500:])

            # --- Per-residue decomposition ---
            decomp_file = f"{rep_dir}/DECOMP_RESULTS.csv"
            if os.path.exists(decomp_file):
                decomp = parse_decomp_csv(decomp_file)
                if decomp:
                    decomp_results.append(decomp)
                    top = sorted(decomp.items(), key=lambda x: x[1])[:5]
                    print(f"  ✓ Decomp top 5 hotspots: " +
                          ", ".join(f"{label_residue(r)[0]}={v:.2f}" for r, v in top))
                else:
                    print(f"  ✗ Replica {i+1} — decomp CSV empty or unparseable")
            else:
                print(f"  ✗ Replica {i+1} — no decomp CSV generated")

        except FileNotFoundError:
            print(f"  ✗ Replica {i+1} — gmx_MMPBSA not found, is gromacs env active?")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Replica {i+1} — timed out after 7 hours")
            csv_file = f"{rep_dir}/FINAL_RESULTS_MMPBSA.csv"
            if os.path.exists(csv_file):
                data = parse_mmpbsa_csv(csv_file)
                if len(data) > 0:
                    mmpbsa_results.append(data)
                    print(f"  ~ Partial results — ΔG (GB): {data.mean():.2f} ± {data.std():.2f} kcal/mol")
            decomp_file = f"{rep_dir}/DECOMP_RESULTS.csv"
            if os.path.exists(decomp_file):
                decomp = parse_decomp_csv(decomp_file)
                if decomp:
                    decomp_results.append(decomp)
        except Exception as e:
            print(f"  ✗ Replica {i+1} — unexpected error: {e}")

        for f in glob.glob(os.path.join(PROJECT_ROOT, "_GMXMMPBSA_*")):
            try:
                os.remove(f)
            except:
                pass

        sys.stdout.flush()

    # -------------------------------------------------------------------------
    # PLOT 1 — Overall ΔG binding per replica (bar chart)
    # -------------------------------------------------------------------------
    if mmpbsa_results:
        fig, ax = plt.subplots(figsize=(7, 5))
        means = [r.mean() for r in mmpbsa_results]
        stds  = [r.std()  for r in mmpbsa_results]
        ax.bar([f"Rep {i+1}\n(eq {EQUIL[i]}ps)" for i in range(len(mmpbsa_results))],
               means, yerr=stds, color=COLORS[:len(mmpbsa_results)],
               edgecolor="k", linewidth=0.7, error_kw={"elinewidth": 1.5}, capsize=5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        # Shaded region showing expected PPI range
        ax.axhspan(-50, -20, alpha=0.08, color="green", label="Expected PPI range (−20 to −50)")
        ax.set_ylabel("ΔG binding — GB (kcal/mol)", fontsize=11)
        ax.set_title("MM-GBSA Binding Free Energy — MYC-MAX\n"
                     "(igb=2, idecomp=1, interval=50)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT}/04_mmgbsa.png", dpi=150)
        plt.close()
        print(f"\n  → Saved {OUT}/04_mmgbsa.png")

        overall_mean = np.mean([r.mean() for r in mmpbsa_results])
        overall_std  = np.std([r.mean()  for r in mmpbsa_results])
        print(f"  Overall ΔG (GB): {overall_mean:.2f} ± {overall_std:.2f} kcal/mol")
    else:
        print("  ✗ No MM-GBSA results to plot")

    # -------------------------------------------------------------------------
    # PLOT 2 — Per-replica hotspot bar charts
    # -------------------------------------------------------------------------
    for i, decomp in enumerate(decomp_results):
        if not decomp:
            continue

        sorted_res = sorted(decomp.items(), key=lambda x: x[1])
        top        = sorted_res[:TOP_N_HOTSPOTS]
        labels     = [r for r, v in top]
        values     = [v for r, v in top]
        colors_bar = []
        for lbl in labels:
            _, prot = label_residue(lbl)
            colors_bar.append("#e74c3c" if prot == "MYC" else "#3498db" if prot == "MAX" else "#95a5a6")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(labels)), values, color=colors_bar, edgecolor="k", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("ΔG contribution (kcal/mol)", fontsize=11)
        ax.set_title(f"MM-GBSA Hotspot Residues — Replica {i+1} (eq {EQUIL[i]}ps)\n"
                     f"Top {TOP_N_HOTSPOTS} by energy contribution",
                     fontsize=12, fontweight="bold")
        ax.legend(handles=[mpatches.Patch(color="#e74c3c", label="MYC (PROA)"),
                            mpatches.Patch(color="#3498db", label="MAX (PROB)")], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out_path = f"{OUT}/04b_hotspots_rep{i+1}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  → Saved {out_path}")

    # -------------------------------------------------------------------------
    # PLOT 3 — Consensus hotspots across ALL replicas
    # -------------------------------------------------------------------------
    if len(decomp_results) > 1:
        all_residues = set()
        for d in decomp_results:
            all_residues.update(d.keys())

        consensus = {}
        for res in all_residues:
            vals = [d[res] for d in decomp_results if res in d]
            if len(vals) == len(decomp_results):
                consensus[res] = (np.mean(vals), np.std(vals))

        if consensus:
            sorted_c = sorted(consensus.items(), key=lambda x: x[1][0])
            top_c    = sorted_c[:TOP_N_HOTSPOTS]
            labels_c = [r for r, _ in top_c]
            means_c  = [v[0] for _, v in top_c]
            stds_c   = [v[1] for _, v in top_c]
            colors_c = []
            for lbl in labels_c:
                _, prot = label_residue(lbl)
                colors_c.append("#e74c3c" if prot == "MYC" else "#3498db" if prot == "MAX" else "#95a5a6")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(range(len(labels_c)), means_c, xerr=stds_c,
                    color=colors_c, edgecolor="k", linewidth=0.5,
                    error_kw={"elinewidth": 1.2, "capsize": 3})
            ax.set_yticks(range(len(labels_c)))
            ax.set_yticklabels(labels_c, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Mean ΔG contribution (kcal/mol)", fontsize=11)
            ax.set_title(f"Consensus MM-GBSA Hotspot Residues — All {len(decomp_results)} Replicas\n"
                         f"Top {TOP_N_HOTSPOTS} — error bars = std across replicas",
                         fontsize=12, fontweight="bold")
            ax.legend(handles=[mpatches.Patch(color="#e74c3c", label="MYC (PROA)"),
                                mpatches.Patch(color="#3498db", label="MAX (PROB)")], fontsize=9)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUT}/04c_hotspots_consensus.png", dpi=150)
            plt.close()
            print(f"  → Saved {OUT}/04c_hotspots_consensus.png")

            summary_path = f"{OUT}/04d_hotspots_summary.txt"
            with open(summary_path, "w") as f:
                f.write("Consensus MM-GBSA Hotspot Residues — MYC-MAX\n")
                f.write(f"Replicas: {len(decomp_results)} | Top {TOP_N_HOTSPOTS} by mean contribution\n")
                f.write(f"Settings: igb=2, idecomp=1, interval=50\n")
                f.write("="*60 + "\n")
                f.write(f"{'Residue':<25} {'Mean (kcal/mol)':>18} {'Std':>10} {'Protein':>10}\n")
                f.write("-"*60 + "\n")
                for res, (mean_v, std_v) in sorted_c[:TOP_N_HOTSPOTS]:
                    _, prot = label_residue(res)
                    f.write(f"{res:<25} {mean_v:>18.2f} {std_v:>10.2f} {prot:>10}\n")
            print(f"  → Saved {summary_path}")

            # --- EXPORT MACHINE-READABLE CSV FOR CROSS-VALIDATION ---
            csv_hotspot_path = f"{OUT}/mmpbsa_hotspots.csv"
            with open(csv_hotspot_path, "w") as f:
                f.write("Chain,Number,Residue,Energy_Mean,Energy_Std,Protein\n")
                for res, (mean_v, std_v) in sorted_c:
                    # Parse res label R:A:ARG:913
                    parts = res.split(":")
                    if len(parts) == 4:
                        _, chain, resname, resnum = parts
                        _, prot = label_residue(res)
                        f.write(f"{chain},{resnum},{resname},{mean_v:.4f},{std_v:.4f},{prot}\n")
            print(f"  → Saved {csv_hotspot_path} (machine-readable)")
        else:
            print("  ✗ No residues found consistently across all replicas for consensus plot")


# =============================================================================
# 5. Salt Bridge Occupancy
# =============================================================================
if 5 in RUN:
    print("\n[5/5] Calculating inter-chain salt bridge occupancy...")

    # Positive residues: Arg (NH1/NH2), Lys (NZ), His (ND1/NE2)
    # Negative residues: Asp (OD1/OD2), Glu (OE1/OE2)
    # Inter-chain pairs only: MYC donors <-> MAX acceptors and vice versa.
    PAIRS_TO_CHECK = [
        (
            "segid PROA and (resname ARG LYS HIS) and (name NH1 NH2 NZ ND1 NE2)",
            "segid PROB and (resname ASP GLU) and (name OD1 OD2 OE1 OE2)",
            "MYC→MAX",
        ),
        (
            "segid PROB and (resname ARG LYS HIS) and (name NH1 NH2 NZ ND1 NE2)",
            "segid PROA and (resname ASP GLU) and (name OD1 OD2 OE1 OE2)",
            "MAX→MYC",
        ),
    ]

    # all_bridges: {bridge_label: {"occupancies": [...], "direction": str}}
    all_bridges = {}

    for i, (traj, eq) in enumerate(zip(TRAJS, EQUIL)):
        try:
            u = mda.Universe(PSF_FILE, traj)
            # bridge_counts key: (don_resid, don_segid, don_resname,
            #                     acc_resid, acc_segid, acc_resname, direction)
            # Counts frames (not atom hits) — deduplicated within each frame.
            bridge_counts = {}
            n_frames = 0

            for ts in u.trajectory:
                n_frames += 1
                # seen_this_frame ensures each RESIDUE PAIR is counted at most
                # once per frame regardless of how many atom pairs satisfy the
                # distance cutoff (fixes occupancy > 1.0 bug).
                seen_this_frame = set()

                for sel_don, sel_acc, direction in PAIRS_TO_CHECK:
                    donors    = u.select_atoms(sel_don)
                    acceptors = u.select_atoms(sel_acc)
                    if len(donors) == 0 or len(acceptors) == 0:
                        continue

                    dists = distance_array(donors.positions, acceptors.positions)
                    hits  = np.argwhere(dists < SALT_DIST)

                    for di, ai in hits:
                        d_atom = donors[di]
                        a_atom = acceptors[ai]
                        key = (
                            d_atom.resid, d_atom.segid, d_atom.resname,
                            a_atom.resid, a_atom.segid, a_atom.resname,
                            direction,
                        )
                        if key not in seen_this_frame:
                            seen_this_frame.add(key)
                            bridge_counts[key] = bridge_counts.get(key, 0) + 1

            # Convert to fractional occupancy — now guaranteed in [0, 1]
            n_stable = 0
            for key, count in bridge_counts.items():
                occ = count / n_frames          # always ≤ 1.0
                if occ >= SALT_OCCUPANCY:
                    don_chain = "MYC" if key[1] == "PROA" else "MAX"
                    acc_chain = "MYC" if key[4] == "PROA" else "MAX"
                    direction  = key[6]
                    # Clean label — direction encoded by colour, not text
                    bridge_label = (
                        f"{key[2]}{key[0]}({don_chain})"
                        f"--{key[5]}{key[3]}({acc_chain})"
                    )
                    if bridge_label not in all_bridges:
                        all_bridges[bridge_label] = {
                            "occupancies": [],
                            "direction":   direction,
                        }
                    all_bridges[bridge_label]["occupancies"].append(occ)
                    n_stable += 1

            print(f"  Rep {i+1} (eq {eq}ps) — {n_stable} salt bridges above "
                  f"{SALT_OCCUPANCY*100:.0f}% occupancy threshold")

        except Exception as e:
            print(f"  ✗ Rep {i+1} salt bridge analysis failed: {e}")

    # Consensus: present in ALL replicas above threshold
    consensus_bridges = {
        k: v for k, v in all_bridges.items()
        if len(v["occupancies"]) == len(TRAJS)
    }
    # Partial consensus: seen in 2 out of 3 replicas
    partial_bridges = {
        k: v for k, v in all_bridges.items()
        if 1 < len(v["occupancies"]) < len(TRAJS)
    }

    print(f"\n  Consensus salt bridges (all {len(TRAJS)} replicas): {len(consensus_bridges)}")
    print(f"  Partial consensus (2/{len(TRAJS)} replicas):        {len(partial_bridges)}")

    # ---- Save text summary ----
    sb_summary_path = f"{OUT}/05_salt_bridges_summary.txt"
    with open(sb_summary_path, "w") as f:
        f.write("Inter-chain Salt Bridge Occupancy — MYC-MAX\n")
        f.write(f"Distance cutoff: {SALT_DIST} Å | Occupancy threshold: {SALT_OCCUPANCY*100:.0f}%\n")
        f.write(f"Replicas: {len(TRAJS)}\n")
        f.write("="*70 + "\n\n")

        f.write("=== CONSENSUS (present in ALL replicas) ===\n")
        f.write(f"{'Bridge':<45} {'Direction':<12} {'Mean occ':>10} {'Std':>8}\n")
        f.write("-"*70 + "\n")
        for bridge, data in sorted(consensus_bridges.items(),
                                   key=lambda x: -np.mean(x[1]["occupancies"])):
            occs = data["occupancies"]
            f.write(f"{bridge:<45} {data['direction']:<12} "
                    f"{np.mean(occs):>10.3f} {np.std(occs):>8.3f}\n")

        f.write("\n=== PARTIAL CONSENSUS (2 of 3 replicas) ===\n")
        f.write(f"{'Bridge':<45} {'Direction':<12} {'Mean occ':>10} {'Std':>8}\n")
        f.write("-"*70 + "\n")
        for bridge, data in sorted(partial_bridges.items(),
                                   key=lambda x: -np.mean(x[1]["occupancies"])):
            occs = data["occupancies"]
            f.write(f"{bridge:<45} {data['direction']:<12} "
                    f"{np.mean(occs):>10.3f} {np.std(occs):>8.3f}\n")

    print(f"  → Saved {sb_summary_path}")

    # --- EXPORT MACHINE-READABLE CSV FOR CROSS-VALIDATION ---
    csv_sb_path = f"{OUT}/salt_bridges.csv"
    with open(csv_sb_path, "w") as f:
        f.write("Residue_A_Name,Residue_A_Number,Residue_B_Name,Residue_B_Number,Mean_Occupancy,Consensus\n")
        # Combine both for the CSV, mark consensus
        for bridge, data in consensus_bridges.items():
            # Parse label: GLU957(MYC)--LYS256(MAX)
            # This is a bit brittle, better to use a regex or split
            import re
            m = re.match(r"([A-Z]{3})(\d+)\(([A-Z]+)\)--([A-Z]{3})(\d+)\(([A-Z]+)\)", bridge)
            if m:
                res1, num1, prot1, res2, num2, prot2 = m.groups()
                # Order as MYC then MAX for consistency
                if prot1 == "MYC":
                    f.write(f"{res1},{num1},{res2},{num2},{np.mean(data['occupancies']):.3f},3/3\n")
                else:
                    f.write(f"{res2},{num2},{res1},{num1},{np.mean(data['occupancies']):.3f},3/3\n")
        for bridge, data in partial_bridges.items():
            import re
            m = re.match(r"([A-Z]{3})(\d+)\(([A-Z]+)\)--([A-Z]{3})(\d+)\(([A-Z]+)\)", bridge)
            if m:
                res1, num1, prot1, res2, num2, prot2 = m.groups()
                if prot1 == "MYC":
                    f.write(f"{res1},{num1},{res2},{num2},{np.mean(data['occupancies']):.3f},2/3\n")
                else:
                    f.write(f"{res2},{num2},{res1},{num1},{np.mean(data['occupancies']):.3f},2/3\n")
    print(f"  → Saved {csv_sb_path} (machine-readable)")

    def _plot_salt_bridges(bridges_dict, title, outpath, alpha=1.0):
        """Helper to plot a salt bridge occupancy bar chart."""
        if not bridges_dict:
            return
        sorted_b  = sorted(bridges_dict.items(),
                            key=lambda x: np.mean(x[1]["occupancies"]))
        labels_b  = [k for k, _ in sorted_b]
        means_b   = [np.mean(v["occupancies"]) for _, v in sorted_b]
        stds_b    = [np.std(v["occupancies"])  for _, v in sorted_b]
        # Colour by direction: MYC→MAX = orange, MAX→MYC = purple
        bar_colors = [
            "#e67e22" if v["direction"] == "MYC→MAX" else "#8e44ad"
            for _, v in sorted_b
        ]

        fig, ax = plt.subplots(figsize=(11, max(4, len(labels_b) * 0.55)))
        ax.barh(range(len(labels_b)), means_b, xerr=stds_b,
                color=bar_colors, edgecolor="k", linewidth=0.5, alpha=alpha,
                error_kw={"elinewidth": 1.2, "capsize": 3})
        ax.set_yticks(range(len(labels_b)))
        ax.set_yticklabels(labels_b, fontsize=9)
        ax.set_xlabel("Mean occupancy (fraction of frames)", fontsize=11)
        # Hard cap at 1.0 — occupancy is a fraction, never exceeds 1
        ax.set_xlim(0, 1.05)
        ax.axvline(SALT_OCCUPANCY, color="red", linestyle="--", linewidth=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(handles=[
            mpatches.Patch(color="#e67e22", label="MYC(+) → MAX(-)"),
            mpatches.Patch(color="#8e44ad", label="MAX(+) → MYC(-)"),
            mpatches.Patch(color="red",     label=f"Threshold ({SALT_OCCUPANCY*100:.0f}%)",
                           linestyle="--", fill=False),
        ], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  → Saved {outpath}")

    _plot_salt_bridges(
        consensus_bridges,
        title=(f"Consensus Inter-chain Salt Bridges — MYC-MAX\n"
               f"(present in all {len(TRAJS)} replicas, cutoff {SALT_DIST} Å)"),
        outpath=f"{OUT}/05_salt_bridges_consensus.png",
    )
    _plot_salt_bridges(
        partial_bridges,
        title=f"Partial-consensus Salt Bridges (2/{len(TRAJS)} replicas) — MYC-MAX",
        outpath=f"{OUT}/05b_salt_bridges_partial.png",
        alpha=0.75,
    )

    if not consensus_bridges and not partial_bridges:
        print("  ✗ No salt bridges found above threshold in any replica — "
              "consider lowering SALT_OCCUPANCY or checking segid assignments.")



# =============================================================================
# HELPER — add this near parse_mmpbsa_csv / parse_decomp_csv at the top
# of your existing script.
# =============================================================================
def parse_alascan_dat(filepath):
    """Returns ΔΔG as a single float from a one-residue alanine scan .dat file."""
    with open(filepath) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("=") or stripped.startswith("-"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None

# =============================================================================
# 6. Alanine Scanning (Hotspot Validation)
# =============================================================================
if 6 in RUN:
    print("\n[6/6] Running Alanine Scanning (Hotspot Validation)...")

    # -------------------------------------------------------------------------
    # 6a. Gather candidate hotspots from Section 4 decomposition results
    # -------------------------------------------------------------------------
    if "decomp_results" not in dir() or not decomp_results:
        decomp_results = []
        for i in range(1, len(TPR_FILES) + 1):
            decomp_file = os.path.join(OUT, f"mmpbsa_rep{i}", "DECOMP_RESULTS.csv")
            if os.path.exists(decomp_file):
                decomp = parse_decomp_csv(decomp_file)
                if decomp:
                    decomp_results.append(decomp)
                    print(f"  ✓ Loaded replica {i} from disk ({len(decomp)} residues)")
            else:
                print(f"  ✗ Not found: {decomp_file}")

    if not decomp_results:
        print("  ✗ No decomp data found — check that OUT points to myc_max_results/")
    else:
        # Rebuild consensus dict (mirrors Section 4 logic)
        all_residues_scan = set()
        for d in decomp_results:
            all_residues_scan.update(d.keys())

        consensus_scan = {}
        for res in all_residues_scan:
            vals = [d[res] for d in decomp_results if res in d]
            if len(vals) >= max(1, len(decomp_results) - 1):
                consensus_scan[res] = (np.mean(vals), np.std(vals))

        if not consensus_scan:
            print("  ⚠ No residues found in ≥2 replicas — using single-replica results")
            for res in all_residues_scan:
                vals = [d[res] for d in decomp_results if res in d]
                consensus_scan[res] = (np.mean(vals), np.std(vals))

        # Pick top N hotspots (most negative mean contribution)
        sorted_consensus_scan = sorted(consensus_scan.items(), key=lambda x: x[1][0])
        scan_candidates = sorted_consensus_scan[:TOP_N_HOTSPOTS]

        print(f"\n  Candidate hotspots selected for alanine scan ({len(scan_candidates)} residues):")
        for r, (mv, sv) in scan_candidates:
            _, prot = label_residue(r)
            print(f"    {r:<30s}  [{prot}]  ΔG = {mv:.2f} ± {sv:.2f} kcal/mol")

        # -------------------------------------------------------------------------
        # 6b. Helpers
        # -------------------------------------------------------------------------
        CHAIN_MAP = {"MYC": "A", "MAX": "B"}   # ← edit if your chains differ

        def extract_resnum(label):
            import re
            nums = re.findall(r'\d+', label)
            return int(nums[-1]) if nums else None

        # -------------------------------------------------------------------------
        # 6c. Run one gmx_MMPBSA alanine scan per residue per replica
        #     gmx_MMPBSA only allows ONE mutant residue per run.
        # -------------------------------------------------------------------------
        alascan_results = []   # one dict per replica: {residue_label: ddG_float}

        for i, (tpr, xtc, topol, eq) in enumerate(zip(TPR_FILES, TRAJS, TOPOL_FILES, EQUIL)):
            print(f"\n  Replica {i+1} — alanine scan started "
                  f"at {datetime.datetime.now().strftime('%H:%M:%S')}")
            rep_ddg = {}

            for r, _ in scan_candidates:
                _, prot  = label_residue(r)
                chain    = CHAIN_MAP.get(prot, "A")
                resnum   = extract_resnum(r)
                if resnum is None:
                    print(f"    {r}: ⚠ could not extract residue number — skipping")
                    continue

                single_res = f"{chain}/{resnum}"
                safe_label = r.replace(":", "_").replace(" ", "_")

                ALASCAN_INPUT = f"""\
&general
   startframe=1, interval=50,
   verbose=2,
   int_dielectric=4.0,  # Match the decomposition setting
/
&pb
   istrng=0.15,
   ratio=1.5,
   fillratio=4.0,
/
&alanine_scanning
   mutant='ALA',
   mutant_res="{single_res}",
/
"""
                res_dir = os.path.abspath(f"{OUT}/alascan_rep{i+1}/{safe_label}")
                os.makedirs(res_dir, exist_ok=True)

                inp_file = f"{res_dir}/alascan.in"
                with open(inp_file, "w") as f:
                    f.write(ALASCAN_INPUT)

                cmd = [
                    "mpirun", "-np", "8", "--allow-run-as-root",
                    "gmx_MMPBSA", "MPI", "-O",
                    "-i",  inp_file,
                    "-cs", os.path.abspath(tpr),
                    "-ct", os.path.abspath(xtc),
                    "-ci", INDEX_FILE,
                    "-cg", "16", "17",
                    "-cp", os.path.abspath(topol),
                    "-o",  f"{res_dir}/FINAL_ALASCAN.dat",
                    "-eo", f"{res_dir}/FINAL_ALASCAN.csv",
                ]

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                            timeout=25200, env=os.environ.copy())
                    dat_file = f"{res_dir}/FINAL_ALASCAN.dat"
                    if os.path.exists(dat_file):
                        ddg_val = parse_alascan_dat(dat_file)
                        if ddg_val is not None:
                            rep_ddg[r] = ddg_val
                            flag = " ★ HOTSPOT" if ddg_val >= 2.0 else ""
                            print(f"    {r}: ΔΔG = {ddg_val:+.2f} kcal/mol{flag}")
                        else:
                            print(f"    {r}: ✗ output exists but could not parse ΔΔG")
                            print(result.stderr[-300:])
                    else:
                        print(f"    {r}: ✗ no output file — {result.stderr[-200:]}")

                except subprocess.TimeoutExpired:
                    print(f"    {r}: ✗ timed out")
                    dat_file = f"{res_dir}/FINAL_ALASCAN.dat"
                    if os.path.exists(dat_file):
                        ddg_val = parse_alascan_dat(dat_file)
                        if ddg_val is not None:
                            rep_ddg[r] = ddg_val
                            print(f"    {r}: ~ partial — ΔΔG = {ddg_val:+.2f} kcal/mol")
                except Exception as e:
                    print(f"    {r}: ✗ error — {e}")

                for tmp in glob.glob(os.path.join(PROJECT_ROOT, "_GMXMMPBSA_*")):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

            if rep_ddg:
                alascan_results.append(rep_ddg)
                print(f"  ✓ Replica {i+1} complete — "
                      f"{len(rep_ddg)}/{len(scan_candidates)} residues parsed")
            else:
                print(f"  ✗ Replica {i+1} — no results collected")

            sys.stdout.flush()

        # =====================================================================
        # PLOTS
        # =====================================================================
        if not alascan_results:
            print("\n  ✗ No alanine scanning results to plot.")
        else:
            import matplotlib.lines as mlines
            from scipy.stats import zscore, pearsonr, spearmanr

            HOTSPOT_DDG_THRESH = 2.0

            # -----------------------------------------------------------------
            # PLOT 6a — Per-replica ΔΔG bar charts
            # -----------------------------------------------------------------
            for i, ddg_dict in enumerate(alascan_results):
                if not ddg_dict:
                    continue
                sorted_ddg = sorted(ddg_dict.items(), key=lambda x: -x[1])
                labels_a   = [r for r, _ in sorted_ddg]
                values_a   = [v for _, v in sorted_ddg]
                colors_a   = ["#e74c3c" if label_residue(l)[1] == "MYC"
                               else "#3498db" if label_residue(l)[1] == "MAX"
                               else "#95a5a6" for l in labels_a]

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(range(len(labels_a)), values_a, color=colors_a,
                        edgecolor="k", linewidth=0.5)
                ax.set_yticks(range(len(labels_a)))
                ax.set_yticklabels(labels_a, fontsize=8)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.axvline(HOTSPOT_DDG_THRESH, color="#e67e22", linewidth=1.4,
                           linestyle="--")
                ax.set_xlabel("ΔΔG upon Ala mutation (kcal/mol)", fontsize=11)
                ax.set_title(
                    f"Alanine Scanning — Replica {i+1} (eq {EQUIL[i]}ps)\n"
                    "ΔΔG = ΔG(Ala mutant) − ΔG(WT) | positive = destabilising",
                    fontsize=12, fontweight="bold")
                ax.legend(handles=[
                    mpatches.Patch(color="#e74c3c", label="MYC (PROA)"),
                    mpatches.Patch(color="#3498db", label="MAX (PROB)"),
                    mlines.Line2D([], [], color="#e67e22", linestyle="--",
                                  label=f"Hotspot threshold ({HOTSPOT_DDG_THRESH} kcal/mol)")
                ], fontsize=9)
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                out_path = f"{OUT}/06a_alascan_rep{i+1}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"  → Saved {out_path}")

            # -----------------------------------------------------------------
            # PLOT 6b — Consensus ΔΔG across all replicas
            # -----------------------------------------------------------------
            all_scan_res = set()
            for d in alascan_results:
                all_scan_res.update(d.keys())

            consensus_ddg = {}
            for res in all_scan_res:
                vals = [d[res] for d in alascan_results if res in d]
                if len(vals) >= max(1, len(alascan_results) - 1):
                    consensus_ddg[res] = (np.mean(vals), np.std(vals))

            if consensus_ddg:
                sorted_ddg_c  = sorted(consensus_ddg.items(), key=lambda x: -x[1][0])
                labels_c      = [r for r, _ in sorted_ddg_c]
                means_ddg     = [v[0] for _, v in sorted_ddg_c]
                stds_ddg      = [v[1] for _, v in sorted_ddg_c]
                colors_cc     = ["#e74c3c" if label_residue(l)[1] == "MYC"
                                 else "#3498db" if label_residue(l)[1] == "MAX"
                                 else "#95a5a6" for l in labels_c]

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(range(len(labels_c)), means_ddg, xerr=stds_ddg,
                        color=colors_cc, edgecolor="k", linewidth=0.5,
                        error_kw={"elinewidth": 1.2, "capsize": 3})
                ax.set_yticks(range(len(labels_c)))
                ax.set_yticklabels(labels_c, fontsize=8)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.axvline(HOTSPOT_DDG_THRESH, color="#e67e22", linewidth=1.4,
                           linestyle="--")
                ax.set_xlabel("Mean ΔΔG upon Ala mutation (kcal/mol)", fontsize=11)
                ax.set_title(
                    f"Consensus Alanine Scanning — All {len(alascan_results)} Replicas\n"
                    f"error bars = std | dashed = hotspot threshold ({HOTSPOT_DDG_THRESH} kcal/mol)",
                    fontsize=12, fontweight="bold")
                ax.legend(handles=[
                    mpatches.Patch(color="#e74c3c", label="MYC (PROA)"),
                    mpatches.Patch(color="#3498db", label="MAX (PROB)")
                ], fontsize=9)
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{OUT}/06b_alascan_consensus.png", dpi=150)
                plt.close()
                print(f"  → Saved {OUT}/06b_alascan_consensus.png")

                # -------------------------------------------------------------
                # PLOT 6c — MM-GBSA decomp vs Alanine Scan correlation
                # -------------------------------------------------------------
                common = [r for r in labels_c if r in consensus_scan]

                if len(common) < 2:
                    print("  ⚠ Fewer than 2 residues shared — skipping correlation plot")
                else:
                    mmgbsa_m  = np.array([consensus_scan[r][0] for r in common])
                    alascan_m = np.array([consensus_ddg[r][0]  for r in common])
                    mmgbsa_s  = np.array([consensus_scan[r][1] for r in common])
                    alascan_s = np.array([consensus_ddg[r][1]  for r in common])

                    mmgbsa_z  = zscore(mmgbsa_m)
                    alascan_z = zscore(alascan_m)

                    pearson_r,  pearson_p  = pearsonr(mmgbsa_m,  alascan_m)
                    spearman_r, spearman_p = spearmanr(mmgbsa_m, alascan_m)

                    x     = np.arange(len(common))
                    width = 0.38
                    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

                    ax1 = axes[0]
                    ax1.barh(x - width/2, mmgbsa_z,  width, color="#2ecc71",
                             edgecolor="k", linewidth=0.5, label="MM-GBSA decomp (z-score)")
                    ax1.barh(x + width/2, alascan_z, width, color="#9b59b6",
                             edgecolor="k", linewidth=0.5, label="Ala scan ΔΔG (z-score)")
                    ax1.set_yticks(x)
                    ax1.set_yticklabels(common, fontsize=8)
                    ax1.axvline(0, color="black", linewidth=0.8)
                    ax1.set_xlabel("Z-scored contribution (arbitrary units)", fontsize=11)
                    ax1.set_title("MM-GBSA vs Alanine Scan\n(z-scored for visual rank comparison)",
                                  fontsize=11, fontweight="bold")
                    ax1.legend(fontsize=9)
                    ax1.grid(axis="x", alpha=0.3)

                    ax2 = axes[1]
                    sc_colors = ["#e74c3c" if label_residue(l)[1] == "MYC"
                                 else "#3498db" if label_residue(l)[1] == "MAX"
                                 else "#95a5a6" for l in common]
                    ax2.errorbar(mmgbsa_m, alascan_m, xerr=mmgbsa_s, yerr=alascan_s,
                                 fmt="none", color="grey", alpha=0.4,
                                 linewidth=0.8, capsize=2, zorder=1)
                    ax2.scatter(mmgbsa_m, alascan_m, c=sc_colors,
                                s=90, edgecolors="k", linewidth=0.5, zorder=3)
                    for j, lbl in enumerate(common):
                        ax2.annotate(lbl, (mmgbsa_m[j], alascan_m[j]),
                                     fontsize=6, ha="left", va="bottom",
                                     xytext=(3, 3), textcoords="offset points")
                    m_fit, b_fit = np.polyfit(mmgbsa_m, alascan_m, 1)
                    x_line = np.linspace(mmgbsa_m.min(), mmgbsa_m.max(), 100)
                    ax2.plot(x_line, m_fit * x_line + b_fit, "k--", linewidth=1)
                    ax2.axhline(HOTSPOT_DDG_THRESH, color="#e67e22", linewidth=1, linestyle=":")
                    ax2.axvline(-1.0, color="#27ae60", linewidth=1, linestyle=":")
                    ax2.set_xlabel("MM-GBSA per-residue ΔG (kcal/mol)", fontsize=11)
                    ax2.set_ylabel("Alanine scan ΔΔG (kcal/mol)", fontsize=11)
                    ax2.set_title(
                        f"Correlation: MM-GBSA decomp vs Alanine Scan\n"
                        f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})  |  "
                        f"Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.2e})",
                        fontsize=10, fontweight="bold")
                    ax2.legend(handles=[
                        mpatches.Patch(color="#e74c3c", label="MYC (PROA)"),
                        mpatches.Patch(color="#3498db", label="MAX (PROB)"),
                        mlines.Line2D([], [], color="k",      linestyle="--", label="Linear fit"),
                        mlines.Line2D([], [], color="#e67e22", linestyle=":", label="ΔΔG threshold"),
                        mlines.Line2D([], [], color="#27ae60", linestyle=":", label="MM-GBSA threshold"),
                    ], fontsize=8, loc="upper left")
                    ax2.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{OUT}/06c_alascan_vs_mmgbsa.png", dpi=150)
                    plt.close()
                    print(f"  → Saved {OUT}/06c_alascan_vs_mmgbsa.png")

                    print(f"\n  ── Correlation Summary ──")
                    print(f"  Pearson  r  = {pearson_r:+.3f}  (p = {pearson_p:.3e})")
                    print(f"  Spearman ρ  = {spearman_r:+.3f}  (p = {spearman_p:.3e})")
                    if abs(pearson_r) > 0.7:
                        print("  ✓ Strong agreement — MM-GBSA hotspot ranking well supported")
                    elif abs(pearson_r) > 0.4:
                        print("  ~ Moderate agreement — inspect outlier residues manually")
                    else:
                        print("  ⚠ Weak correlation — prioritise alanine scan results")

                # -------------------------------------------------------------
                # SUMMARY FILE — 06d_alascan_summary.txt
                # -------------------------------------------------------------
                summary_path = f"{OUT}/06d_alascan_summary.txt"
                with open(summary_path, "w") as f:
                    f.write("Alanine Scanning — Consensus ΔΔG Summary\n")
                    f.write(f"MYC-MAX | Replicas: {len(alascan_results)} | igb=2, interval=50\n")
                    f.write("ΔΔG = ΔG(Ala mutant) − ΔG(wild-type)\n")
                    f.write(f"Hotspot criterion: ΔΔG ≥ {HOTSPOT_DDG_THRESH:.1f} kcal/mol\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"{'Residue':<28} {'ΔΔG mean':>10} {'Std':>7} "
                            f"{'Hotspot?':>10} {'MM-GBSA':>10} {'Protein':>8}\n")
                    f.write("-" * 70 + "\n")
                    for res, (mean_v, std_v) in sorted_ddg_c:
                        _, prot = label_residue(res)
                        flag    = "★ YES" if mean_v >= HOTSPOT_DDG_THRESH else "  no"
                        mmg_str = f"{consensus_scan[res][0]:.2f}" if res in consensus_scan else ""
                        f.write(f"{res:<28} {mean_v:>10.2f} {std_v:>7.2f} "
                                f"{flag:>10} {mmg_str:>10} {prot:>8}\n")
                    f.write("\n" + "=" * 70 + "\n")
                    f.write("CONFIRMED HOTSPOTS (ΔΔG ≥ 2.0 AND MM-GBSA ≤ −1.0 kcal/mol)\n")
                    f.write("-" * 70 + "\n")
                    confirmed = [r for r, (v, _) in sorted_ddg_c
                                 if v >= HOTSPOT_DDG_THRESH
                                 and r in consensus_scan
                                 and consensus_scan[r][0] <= -1.0]
                    if confirmed:
                        for r in confirmed:
                            _, prot = label_residue(r)
                            f.write(f"  ★ {r:<28}  ΔΔG={consensus_ddg[r][0]:+.2f}"
                                    f"  MM-GBSA={consensus_scan[r][0]:.2f}  [{prot}]\n")
                    else:
                        f.write("  (none passed both criteria)\n")
                print(f"  → Saved {summary_path}")

            else:
                print("  ✗ No consensus ΔΔG residues found across replicas")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("  Analysis complete. Output files:")
for f in sorted(os.listdir(OUT)):
    print(f"    {OUT}/{f}")
print("="*60)
print("\n  HOTSPOT CROSS-REFERENCE WORKFLOW:")
print("  1. Open 04d_hotspots_summary.txt  → MM-GBSA energy hotspots")
print("  2. Open 05_salt_bridges_summary.txt → persistent salt bridges")
print("  3. Residues in BOTH lists = highest-confidence drug targets")
print("="*60)
print("="*60)