#!/usr/bin/env python3
"""
Master Cross-Validation & Consensus (Step 4 in Workflow)
========================================================
Analyses: Hotspot consensus (MM-GBSA, salt bridges) cross-referenced against
your real MDpocket sub-pockets (SP1..SPn), Consensus Table & PyMOL.

CHANGE FROM PREVIOUS VERSION: sub-pockets are no longer re-derived here from
per-replica .dx files (that was the old, abandoned 3-separate-replica-runs
method). They're loaded directly from extract_subpockets.py's real output
(Results/Pockets/summary/subpockets_summary.csv and subpockets_residues.csv),
which was built the correct way: single concatenated, common-reference-
aligned trajectory -> one mdpocket run -> connected-component blob
detection -> residues within RESIDUE_CUTOFF of each blob ("core" residues).

Requires: Results/Pockets/summary/subpockets_summary.csv and
          Results/Pockets/summary/subpockets_residues.csv (from
          extract_subpockets.py), plus your MM-GBSA / salt-bridge CSVs.
Outputs: Results/Pockets/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import MDAnalysis as mda

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = "Results/Pockets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. SUB-POCKET DATA (from extract_subpockets.py - the real, validated pipeline)
SP_SUMMARY_CSV  = "Results/Pockets/summary/subpockets_core_summary.csv"   # sp_name, centroid_x/y/z, volume, freq, ...
SP_RESIDUES_CSV = "Results/Pockets/summary/subpockets_core_residues.csv"  # sp_name, chain, resnum, resname, dist_to_pocket_A

# 2. PARAMETERS
# RESIDUE_CUTOFF is NOT re-applied here - the "core" residues in
# SP_RESIDUES_CSV were already selected using RESIDUE_CUTOFF back in
# extract_subpockets.py. EXTENDED_CUTOFF below is a SEPARATE, wider net
# applied only to hotspot/salt-bridge residues that fall outside every
# pocket's core, to flag them as "plausibly associated with a nearby
# pocket" (marked with a '+' suffix, e.g. "SP2+") rather than silently
# dropping them - see get_nearest_sp().
EXTENDED_CUTOFF = 8.0
PROTEIN_PDB     = "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.pdb"

# 3. EXTERNAL DATA PATHS
CSV_MMPBSA = "Results/Complex/mmpbsa_hotspots.csv"
CSV_SALT   = "Results/Complex/salt_bridges.csv"

# Fallback values if CSVs are missing
SALT_BRIDGES_FALLBACK = {
    ("A", 957, "GLU"), ("B", 256, "LYS"),
    ("A", 970, "ARG"), ("B", 275, "GLU"),
}
MMGBSA_FALLBACK = {
    ("A", 972, "GLU"), ("A", 916, "GLU"), ("A", 926, "ASP"), ("A", 965, "ASP"),
    ("A", 925, "ARG"), ("A", 971, "ARG"), ("A", 970, "ARG"), ("A", 913, "ARG"),
    ("A", 968, "ARG"), ("B", 248, "GLU"), ("B", 263, "ASP"), ("B", 265, "ASP"),
    ("B", 244, "ASP"), ("B", 220, "ASP"), ("B", 239, "ARG"), ("B", 214, "ARG"),
    ("B", 269, "ARG"), ("B", 279, "ARG"), ("B", 254, "ARG"), ("B", 226, "ARG"),
}

N_HOTSPOTS = 20

TABLE_COLORS = {
    "header":     "#2C3E50",
    "2/3_MYC":    "#FADBD8",
    "2/3_MAX":    "#D6EAF8",
    "1/3":        "#F9F9F9",
    "check_bg":   "#A9DFBF",
}

# =============================================================================
# DATA PARSING
# =============================================================================

def load_external_data():
    hotspots, salt_bridges = set(), set()
    print("\n[Loading External Data]")
    if os.path.exists(CSV_MMPBSA):
        df_gb = pd.read_csv(CSV_MMPBSA)
        top_gb = df_gb.sort_values("Energy_Mean", ascending=True).head(N_HOTSPOTS)
        for _, row in top_gb.iterrows():
            hotspots.add((row["Chain"], int(row["Number"]), row["Residue"]))
        print(f"  \u2192 Loaded {len(hotspots)} MM-GBSA hotspots from CSV.")
    else:
        print("  \u2192 CSV_MMPBSA not found. Using Fallback MM-GBSA data.")
        hotspots = MMGBSA_FALLBACK

    if os.path.exists(CSV_SALT):
        df_sb = pd.read_csv(CSV_SALT)
        for _, row in df_sb.iterrows():
            salt_bridges.add(("A", int(row["Residue_A_Number"]), row["Residue_A_Name"]))
            salt_bridges.add(("B", int(row["Residue_B_Number"]), row["Residue_B_Name"]))
        print(f"  \u2192 Loaded {len(salt_bridges)} Salt Bridge residues from CSV.")
    else:
        print("  \u2192 CSV_SALT not found. Using Fallback Salt Bridge data.")
        salt_bridges = SALT_BRIDGES_FALLBACK
    return hotspots, salt_bridges


def load_subpockets():
    """Load sub-pockets from extract_subpockets.py's real output, instead of
    re-deriving them from per-replica .dx files (the old, abandoned method).

    sp_residues[sp_name] = set of (chain, resnum, resname) "core" residues -
        i.e. within RESIDUE_CUTOFF of that pocket's voxel cloud, exactly as
        computed by extract_subpockets.py. This script does not re-apply or
        second-guess that cutoff.
    sp_centroids[sp_name] = np.array([x, y, z]) - used only by
        get_nearest_sp() to catch hotspot/salt-bridge residues that fall
        just outside every pocket's core (see EXTENDED_CUTOFF above).
    """
    print("\n[Loading Sub-pockets from extract_subpockets.py output]")
    missing = [p for p in (SP_SUMMARY_CSV, SP_RESIDUES_CSV) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} - run extract_subpockets.py first (it writes "
            f"these into Results/Pockets/summary/)."
        )

    res_df = pd.read_csv(SP_RESIDUES_CSV)
    summary_df = pd.read_csv(SP_SUMMARY_CSV)

    sp_residues = {}
    for sp_name, group in res_df.groupby("sp_name"):
        sp_residues[sp_name] = set(
            (str(row["chain"]), int(row["resnum"]), str(row["resname"]))
            for _, row in group.iterrows()
        )

    sp_centroids = {
        row["sp_name"]: np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]])
        for _, row in summary_df.iterrows()
    }

    for name in sorted(sp_residues, key=lambda n: int(n.replace("SP", ""))):
        n_res = len(sp_residues[name])
        interface_flag = ""
        row = summary_df[summary_df["sp_name"] == name]
        if not row.empty and bool(row.iloc[0].get("is_interface_pocket", False)):
            interface_flag = "  [INTERFACE]"
        print(f"  \u2192 {name}: {n_res} core residues{interface_flag}")

    return sp_residues, sp_centroids


def get_nearest_sp(res, centroids_sp):
    if not centroids_sp:
        return None
    u = mda.Universe(PROTEIN_PDB)
    seg = "PROA" if res[0] == "A" else "PROB"
    sel = u.select_atoms(f"segid {seg} and resid {res[1]}")
    if not sel:
        # fall back to plain chain ID selection, in case this structure
        # doesn't use CHARMM-GUI's PROA/PROB segid convention
        sel = u.select_atoms(f"chainID {res[0]} and resid {res[1]}")
    if not sel:
        return None
    pos = sel.center_of_mass()
    best_sp, min_dist = None, 999.9
    for name, center in centroids_sp.items():
        dist = np.linalg.norm(pos - center)
        if dist < min_dist:
            best_sp, min_dist = name, dist

    if min_dist < EXTENDED_CUTOFF:
        return f"{best_sp}+"
    return None


def build_sp_color_map(sp_names):
    """Assign each SP a distinct color from a soft pastel palette (matches
    seaborn's 'pastel' set), consistent with the light pink/blue already
    used for MYC/MAX backgrounds. Cycles if there are more SPs than
    palette entries. Used for the flat 2D matplotlib TABLE only."""
    palette = [
        "#A1C9F4",  # pastel blue
        "#FFB482",  # pastel orange
        "#8DE5A1",  # pastel green
        "#D0BBFF",  # pastel purple
        "#FF9F9B",  # pastel red/pink
        "#DEBB9B",  # pastel brown
        "#FAB0E4",  # pastel pink
        "#B9F2F0",  # pastel cyan
    ]
    ordered = sorted(sp_names, key=lambda n: int(n.replace("SP", "")))
    return {name.lower(): palette[i % len(palette)] for i, name in enumerate(ordered)}


def main():
    print("\n" + "=" * 60 + "\n      MYC-MAX UNIFIED CROSS-VALIDATION\n" + "=" * 60)

    hotspots, salt_bridges = load_external_data()
    sp_residues, sp_centroids = load_subpockets()
    all_sp_residues = set().union(*sp_residues.values()) if sp_residues else set()
    print("SP core residues (union across all pockets):", len(all_sp_residues))

    sp_color_map = build_sp_color_map(sp_residues.keys())

    all_res = salt_bridges | hotspots | all_sp_residues
    rows = []
    print("\n[Assigning Pockets & Generating Table]")
    for res in sorted(all_res, key=lambda x: (x[0], x[1])):
        in_sb, in_gb = res in salt_bridges, res in hotspots
        sps = [name for name, s in sp_residues.items() if res in s]

        # EXTENDED-NEIGHBORHOOD FALLBACK: if this residue isn't a core
        # member of any pocket, but sits within EXTENDED_CUTOFF of one,
        # flag it as e.g. "SP2+" rather than dropping it - see the note on
        # core vs. extended residues at the top of this file.
        if not sps and sp_centroids:
            nearest = get_nearest_sp(res, sp_centroids)
            if nearest:
                sps = [nearest]

        if not in_sb and not in_gb and not sps:
            continue

        score = sum([in_sb, in_gb, len(sps) > 0])

        rows.append({
            "Chain": res[0], "Protein": "MYC" if res[0] == "A" else "MAX",
            "Residue": res[2], "Number": res[1],
            "Salt Bridge": "\u2713" if in_sb else "-", "MM-GBSA": "\u2713" if in_gb else "-",
            "MDpocket": "+".join(sps) if sps else "-",
            "Score": f"{score}/3", "_score_val": score,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[!] No residues met any criteria - check your input CSVs and paths.")
        return
    df_filtered = df[df["_score_val"] >= 2].sort_values(
        by=["_score_val", "Chain", "Number"], ascending=[False, True, True])

    out_csv = os.path.join(OUTPUT_DIR, "cross_validation_unified.csv")
    df_filtered.to_csv(out_csv, index=False)
    plot_consensus_table(df_filtered, sp_color_map, os.path.join(OUTPUT_DIR, "5.1_cross_validation_table.png"))
    write_pymol_script(df_filtered, os.path.join(OUTPUT_DIR, "visualize_cross_validation.pml"))

    print(f"\n[\u2713] Finished. {len(df_filtered)} high-confidence residues identified.")
    print(f"    Results saved in {OUTPUT_DIR}/")


def plot_consensus_table(df, sp_color_map, output_path):
    display_df = df.drop(columns=["_score_val", "Chain"])
    cols = display_df.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.4)))
    ax.axis("off")
    table = ax.table(cellText=display_df.values.tolist(), colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    for j in range(len(cols)):
        table[0, j].set_facecolor(TABLE_COLORS["header"])
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=12)
    for i, (_, row) in enumerate(df.iterrows()):
        base = TABLE_COLORS["2/3_MYC"] if row["Protein"] == "MYC" else TABLE_COLORS["2/3_MAX"]
        for j, col in enumerate(cols):
            cell = table[i + 1, j]
            cell.set_facecolor(base)
            if str(row[col]) == "\u2713":
                cell.set_facecolor(TABLE_COLORS["check_bg"])
                cell.set_text_props(fontweight="bold")
            if col == "MDpocket" and "SP" in str(row[col]):
                sp_key = str(row[col]).replace("+", "").split("+")[0].lower()
                cell.set_facecolor(sp_color_map.get(sp_key, base))
                cell.set_text_props(fontweight="bold", color=TABLE_COLORS["header"])

    fig.suptitle("Cross-validation: Salt Bridges | MM-GBSA | MDpocket", fontsize=16, fontweight="bold", y=0.88)

    legend_elements = [
        mpatches.Patch(facecolor=TABLE_COLORS["2/3_MYC"], label="Score >= 2 (MYC)"),
        mpatches.Patch(facecolor=TABLE_COLORS["2/3_MAX"], label="Score >= 2 (MAX)"),
        mpatches.Patch(facecolor=TABLE_COLORS["check_bg"], label="Method Detected (\u2713)"),
    ]
    for sp_key, color in sorted(sp_color_map.items()):
        legend_elements.append(mpatches.Patch(facecolor=color, label=f"MDpocket {sp_key.upper()}"))

    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, fontsize=10, frameon=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


SHOW_LABELS = False     # set False to skip labels entirely - add them yourself
                        # later in an image editor, on the rendered PNG
LABEL_OFFSET = (2.0, 2.0, 2.0)   # Angstrom - nudges labels away from the atom
                                   # they're attached to, so they don't sit
                                   # directly on top of overlapping sticks


def write_pymol_script(df, pml_path):
    def get_sel(sub_df):
        return " or ".join(f"(chain {r['Chain']} and resi {r['Number']})" for _, r in sub_df.iterrows()) or "none"

    with open(pml_path, "w") as f:
        f.write(f"load {os.path.abspath(PROTEIN_PDB)}, prot\n")
        f.write("hide everything, prot\nshow cartoon, prot\n")
        f.write("color salmon, prot and chain A\n")
        f.write("color palecyan, prot and chain B\n")
        f.write("set cartoon_transparency, 0\n")

        f.write("\n# --- Cross-validated (score >= 2) residues, in red ---\n")
        f.write(f"select flagged, {get_sel(df)}\n")
        f.write("show sticks, flagged and not name C+N+O\n")
        f.write("color red, flagged\n")

        if SHOW_LABELS:
            ox, oy, oz = LABEL_OFFSET
            f.write(f"set label_position, ({ox}, {oy}, {oz})\n")
            f.write("set label_size, 16\nset label_color, black\n")
            f.write("set label_outline_color, white\nset label_font_id, 7\n")
            f.write("label flagged and name CA, '%s%s' % (resn, resi)\n")
            f.write("# still overlapping in the rendered view? in PyMOL you can drag\n"
                    "# individual labels by hand: wizard label_wizard, then click-drag\n"
                    "# any label to reposition it manually before exporting the image.\n")
        else:
            f.write("# labels skipped (SHOW_LABELS = False) - add them manually in an\n"
                    "# image editor afterward, using cross_validation_unified.csv as\n"
                    "# the reference for which residue is which.\n")

        f.write("\nzoom flagged\n")
        f.write("bg_color white\nset ray_shadows, 0\n")


if __name__ == "__main__":
    main()