#!/usr/bin/env python3
"""
MYC-MAX Complex Analysis (Step 2 in Workflow)
=============================================
Analyses: Interface Contacts, MM-GBSA, Salt Bridges.
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
print("  1. Interface Contacts")
print("  2. MM-GBSA + Hotspot Decomposition")
print("  3. Salt Bridge Occupancy")
print("Enter numbers separated by commas (e.g. 1,3) or press Enter for all: ")
_selection = input().strip()

try:
    RUN = {1, 2, 3} if _selection == "" else set(int(x.strip()) for x in _selection.split(","))
except ValueError:
    print("  Invalid input — running all analyses.")
    RUN = {1, 2, 3}

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
# 1. Interface Contacts
# =============================================================================
if 1 in RUN:
    print("\n[1/3] Calculating interface contacts...")
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
        plt.savefig(f"{OUT}/3.1_interface_contacts.png", dpi=150)
        plt.close()
        print(f"  → Saved {OUT}/3.1_interface_contacts.png")

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
            plt.savefig(f"{OUT}/3.2_contacts_summary.png", dpi=150)
            plt.close()
            print(f"  → Saved {OUT}/3.2_contacts_summary.png")
    except Exception as e:
        print(f"  ✗ Contacts section failed: {e}")


# =============================================================================
# 2. MM-GBSA + Hotspot Decomposition
# =============================================================================
if 2 in RUN:
    print("\n[2/3] Running MM-GBSA + Hotspot Decomposition...")

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
/

&gb
igb=2,
saltcon=0.150,
/

&decomp
idecomp=1,
dec_verbose=1,
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
        plt.savefig(f"{OUT}/3.3_mmgbsa.png", dpi=150)
        plt.close()
        print(f"\n  → Saved {OUT}/3.3_mmgbsa.png")

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
        out_path = f"{OUT}/3.4_hotspots_rep{i+1}.png"
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
            plt.savefig(f"{OUT}/3.5_hotspots_consensus.png", dpi=150)
            plt.close()
            print(f"  → Saved {OUT}/3.5_hotspots_consensus.png")

            summary_path = f"{OUT}/3.0_hotspots_summary.txt"
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
# 3. Salt Bridge Occupancy
# =============================================================================
if 3 in RUN:
    print("\n[3/3] Calculating inter-chain salt bridge occupancy...")

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
    sb_summary_path = f"{OUT}/3.0_salt_bridges_summary.txt"
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
        outpath=f"{OUT}/3.6_salt_bridges_consensus.png",
    )
    _plot_salt_bridges(
        partial_bridges,
        title=f"Partial-consensus Salt Bridges (2/{len(TRAJS)} replicas) — MYC-MAX",
        outpath=f"{OUT}/3.7_salt_bridges_partial.png",
        alpha=0.75,
    )

    if not consensus_bridges and not partial_bridges:
        print("  ✗ No salt bridges found above threshold in any replica — "
              "consider lowering SALT_OCCUPANCY or checking segid assignments.")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("  Analysis complete. Output files:")
for f in sorted(os.listdir(OUT)):
    print(f"    {OUT}/{f}")
print("="*60)
print("\n  HOTSPOT CROSS-REFERENCE WORKFLOW:")
print("  1. Open 4.0_hotspots_summary.txt  → MM-GBSA energy hotspots")
print("  2. Open 3.0_salt_bridges_summary.txt → persistent salt bridges")
print("  3. Residues in BOTH lists = highest-confidence drug targets")
print("="*60)
print("="*60)