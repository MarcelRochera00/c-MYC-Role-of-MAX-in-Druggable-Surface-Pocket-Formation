#!/usr/bin/env python3
"""
Master Cross-Validation & Consensus (Step 4 in Workflow)
========================================================
Analyses: Sub-pocket clustering (.dx), Hotspot consensus, Consensus Table & PyMOL.
Requires: MDpocket grid files (.dx), residue PDBs, and Step 1 results.
Outputs: Results/Pockets/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import MDAnalysis as mda
from scipy import ndimage
try:
    from gridData import Grid
except ImportError:
    print("Warning: gridData-python not installed. Sub-pocket clustering will be skipped.")
    Grid = None

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = "Results/Pockets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. MDPOCKET REPLICAS (PDBs for residue identification)
REPLICA_ATOMS = {
    "Rep1": "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/mdpocket_pocket1_stats_mdpocket_atoms.pdb",
    "Rep2": "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/mdpocket_pocket1_stats_mdpocket_atoms.pdb",
    "Rep3": "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/mdpocket_pocket1_stats_mdpocket_atoms.pdb",
}

# 2. GRID DATA (DX files for sub-pocket clustering)
REPLICA_DX = {
    "Rep1": "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/mdpocket_correct_freq.dx",
    "Rep2": "MYC-MAX/MYC-MAX-1000/charmm-gui/gromacs/mdpocket_correct_freq.dx",
    "Rep3": "MYC-MAX/MYC-MAX-1500/charmm-gui/gromacs/mdpocket_correct_freq.dx",
}

# 3. PARAMETERS
FREQ_THRESHOLD  = 0.3
MIN_VOXELS      = 200
RESIDUE_CUTOFF  = 5.0    # Strict core radius
EXTENDED_CUTOFF = 8.0    # Extended "neighborhood" radius
OVERLAP_CUTOFF  = 10.0
PROTEIN_PDB     = "crystal_proteins.pdb"

# 4. EXTERNAL DATA PATHS
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
FALLBACK_SP = {
    "SP1": {("B", 248, "GLU"), ("B", 231, "SER"), ("B", 249, "TYR"), ("B", 252, "TYR")},
    "SP2": {("B", 214, "ARG"), ("B", 215, "ARG"), ("B", 218, "ILE"), ("A", 939, "LYS"), ("A", 940, "VAL")},
    "SP3": {("B", 239, "ARG"), ("B", 240, "ALA"), ("B", 238, "SER"), ("A", 913, "ARG"), ("A", 916, "GLU"), ("A", 917, "LEU")},
}

MMPBSA_ENERGY_CUTOFF = -1.0 

TABLE_COLORS = {
    "header":     "#2C3E50",
    "2/3_MYC":    "#FADBD8",
    "2/3_MAX":    "#D6EAF8",
    "1/3":        "#F9F9F9",
    "check_bg":   "#A9DFBF",
    "sp1":        "#A7F1E9",
    "sp2":        "#FAD7A0",
    "sp3":        "#CEA1DA",
}

# =============================================================================
# DATA PARSING
# =============================================================================

def load_external_data():
    hotspots, salt_bridges = set(), set()
    print("\n[Loading External Data]")
    if os.path.exists(CSV_MMPBSA):
        df_gb = pd.read_csv(CSV_MMPBSA)
        top_gb = df_gb[df_gb["Energy_Mean"] <= MMPBSA_ENERGY_CUTOFF]
        for _, row in top_gb.iterrows(): hotspots.add((row["Chain"], int(row["Number"]), row["Residue"]))
        print(f"  → Loaded {len(hotspots)} MM-GBSA hotspots from CSV.")
    else:
        print("  → CSV_MMPBSA not found. Using Fallback MM-GBSA data.")
        hotspots = MMGBSA_FALLBACK

    if os.path.exists(CSV_SALT):
        df_sb = pd.read_csv(CSV_SALT)
        for _, row in df_sb.iterrows():
            salt_bridges.add(("A", int(row["Residue_A_Number"]), row["Residue_A_Name"]))
            salt_bridges.add(("B", int(row["Residue_B_Number"]), row["Residue_B_Name"]))
        print(f"  → Loaded {len(salt_bridges)} Salt Bridge residues from CSV.")
    else:
        print("  → CSV_SALT not found. Using Fallback Salt Bridge data.")
        salt_bridges = SALT_BRIDGES_FALLBACK
    return hotspots, salt_bridges

def detect_subpockets():
    if Grid is None: 
        print("  → gridData not available. Using Fallback Sub-pockets.")
        return FALLBACK_SP, {}
    
    print("\n[Sub-pocket Clustering]")
    replica_pockets = {}
    for name, path in REPLICA_DX.items():
        if not os.path.exists(path):
            print(f"  Warning: {path} not found.")
            continue
        grid = Grid(path)
        data = grid.grid
        labeled, n_raw = ndimage.label(data > FREQ_THRESHOLD)
        pockets = []
        for i in range(1, n_raw + 1):
            mask = labeled == i
            if np.sum(mask) < MIN_VOXELS: continue
            centroid = np.array(grid.origin) + np.array(ndimage.center_of_mass(mask)) * np.array(grid.delta)
            pockets.append({"centroid": centroid, "freq": float(data[mask].mean())})
        replica_pockets[name] = pockets
        print(f"  → {name}: {len(pockets)} raw sub-pockets detected.")

    if len(replica_pockets) < 3: 
        print("  → Not enough replicas for consensus clustering. Using Fallback.")
        return FALLBACK_SP, {}

    rep_names = list(replica_pockets.keys())
    consensus_sp, centroids_sp, sp_idx = {}, {}, 1
    u = mda.Universe(PROTEIN_PDB)
    protein = u.select_atoms("protein")

    for p1 in replica_pockets[rep_names[0]]:
        matches = [p1]
        for other in rep_names[1:]:
            best = min(replica_pockets[other], key=lambda p: np.linalg.norm(p1["centroid"] - p["centroid"]), default=None)
            if best and np.linalg.norm(p1["centroid"] - best["centroid"]) < OVERLAP_CUTOFF: matches.append(best)
        
        if len(matches) == 3:
            avg_centroid = np.mean([m["centroid"] for m in matches], axis=0)
            sel = protein.select_atoms(f"point {avg_centroid[0]} {avg_centroid[1]} {avg_centroid[2]} {RESIDUE_CUTOFF}")
            res_set = set((r.segid.replace("PROA","A").replace("PROB","B"), r.resid, r.resname) for r in sel.residues)
            name = f"SP{sp_idx}"
            consensus_sp[name] = res_set
            centroids_sp[name] = avg_centroid
            print(f"  → {name} confirmed at {avg_centroid.round(1)} with {len(res_set)} core residues.")
            sp_idx += 1
    
    return (consensus_sp if consensus_sp else FALLBACK_SP), centroids_sp

def get_nearest_sp(res, centroids_sp):
    if not centroids_sp: return None
    u = mda.Universe(PROTEIN_PDB)
    seg = "PROA" if res[0] == "A" else "PROB"
    sel = u.select_atoms(f"segid {seg} and resid {res[1]}")
    if not sel: return None
    pos = sel.center_of_mass()
    best_sp, min_dist = None, 999.9
    for name, center in centroids_sp.items():
        dist = np.linalg.norm(pos - center)
        if dist < min_dist: best_sp, min_dist = name, dist
    
    if min_dist < EXTENDED_CUTOFF:
        return f"{best_sp}+"
    return None

def parse_pdb_residues(pdb_path):
    residues = set()
    if not os.path.exists(pdb_path): return residues
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname, chain, resnum = line[17:20].strip(), line[21].strip(), int(line[22:26].strip())
                residues.add((chain, resnum, resname))
    return residues

def main():
    print("\n" + "="*60 + "\n      MYC-MAX UNIFIED CROSS-VALIDATION\n" + "="*60)
    
    hotspots, salt_bridges = load_external_data()
    sp_residues, sp_centroids = detect_subpockets()
    all_sp_residues = set().union(*sp_residues.values())
    print("SP residues:", len(all_sp_residues))
    '''
    print("\n[Parsing MDpocket PDBs]")
    replica_sets = []
    for name, path in REPLICA_ATOMS.items():
        if os.path.exists(path):
            res = parse_pdb_residues(path)
            replica_sets.append(res)
            print(f"  → {name}: {len(res)} residues.")
    
    mdpocket_any2 = set()
    if len(replica_sets) >= 2:
        for i in range(len(replica_sets)):
            for j in range(i+1, len(replica_sets)):
                mdpocket_any2 |= (replica_sets[i] & replica_sets[j])
    print(f"  → MDpocket Any-2 Consensus: {len(mdpocket_any2)} residues.")
    '''
    # Build Table
    #all_res = salt_bridges | hotspots | mdpocket_any2 | all_sp_residues 
    all_res = salt_bridges | hotspots | all_sp_residues
    rows = []
    print("\n[Assigning Pockets & Generating Table]")
    for res in sorted(all_res, key=lambda x: (x[0], x[1])):
        #in_sb, in_gb, in_mp = res in salt_bridges, res in hotspots, res in mdpocket_any2
        in_sb, in_gb = res in salt_bridges, res in hotspots
        sps = [name for name, s in sp_residues.items() if res in s]
        
        # NEAREST SP FALLBACK
        '''
        if in_mp and not sps and sp_centroids:
            nearest = get_nearest_sp(res, sp_centroids)
            if nearest: sps = [nearest]
        '''
        if not sps and sp_centroids:
            nearest = get_nearest_sp(res, sp_centroids)
            if nearest: sps = [nearest]
            
        if not in_sb and not in_gb and not sps: continue 

        score = sum([in_sb, in_gb, len(sps) > 0])
        
        rows.append({
            "Chain": res[0], "Protein": "MYC" if res[0] == "A" else "MAX",
            "Residue": res[2], "Number": res[1],
            "Salt Bridge": "✓" if in_sb else "-", "MM-GBSA": "✓" if in_gb else "-",
            "MDpocket": "+".join(sps) if sps else "-",
            "Score": f"{score}/3", "_score_val": score
        })

    df = pd.DataFrame(rows)
    df_filtered = df[df["_score_val"] >= 2].sort_values(by=["_score_val", "Chain", "Number"], ascending=[False, True, True])
    
    out_csv = os.path.join(OUTPUT_DIR, "cross_validation_unified.csv")
    df_filtered.to_csv(out_csv, index=False)
    plot_consensus_table(df_filtered, os.path.join(OUTPUT_DIR, "5.1_cross_validation_table.png"))
    write_pymol_script(df_filtered, os.path.join(OUTPUT_DIR, "visualize_cross_validation.pml"))
    
    print(f"\n[✓] Finished. {len(df_filtered)} high-confidence residues identified.")
    print(f"    Results saved in {OUTPUT_DIR}/")

def plot_consensus_table(df, output_path):
    display_df = df.drop(columns=["_score_val", "Chain"])
    cols = display_df.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, max(8, len(df)*0.4)))
    ax.axis("off")
    table = ax.table(cellText=display_df.values.tolist(), colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.0)
    for j in range(len(cols)):
        table[0, j].set_facecolor(TABLE_COLORS["header"]); table[0, j].set_text_props(color="white", fontweight="bold", fontsize=12)
    for i, (_, row) in enumerate(df.iterrows()):
        base = TABLE_COLORS["2/3_MYC"] if row["Protein"] == "MYC" else TABLE_COLORS["2/3_MAX"]
        for j, col in enumerate(cols):
            cell = table[i+1, j]; cell.set_facecolor(base)
            if str(row[col]) == "✓": cell.set_facecolor(TABLE_COLORS["check_bg"]); cell.set_text_props(fontweight="bold")
            if col == "MDpocket" and "SP" in str(row[col]):
                sp_key = str(row[col]).replace("+","").split("+")[0].lower()
                cell.set_facecolor(TABLE_COLORS.get(sp_key, base)); cell.set_text_props(fontweight="bold")
    
    fig.suptitle("Cross-validation: Salt Bridges | MM-GBSA | MDpocket", fontsize=16, fontweight="bold", y=0.88)

    # --- LEGEND ---
    legend_elements = [
        mpatches.Patch(facecolor=TABLE_COLORS["2/3_MYC"], label="Score >= 2 (MYC)"),
        mpatches.Patch(facecolor=TABLE_COLORS["2/3_MAX"], label="Score >= 2 (MAX)"),
        mpatches.Patch(facecolor=TABLE_COLORS["check_bg"], label="Method Detected (✓)"),
        mpatches.Patch(facecolor=TABLE_COLORS["sp1"], label="MDpocket SP1 (Cyan)"),
        mpatches.Patch(facecolor=TABLE_COLORS["sp2"], label="MDpocket SP2 (Yellow)"),
        mpatches.Patch(facecolor=TABLE_COLORS["sp3"], label="MDpocket SP3 (Magenta)"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=11, frameon=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()

def write_pymol_script(df, pml_path):
    def get_sel(sub_df): return " or ".join([f"(chain {r['Chain']} and resi {r['Number']})" for _, r in sub_df.iterrows()]) or "none"
    with open(pml_path, "w") as f:
        f.write(f"load {os.path.abspath(PROTEIN_PDB)}, prot\nhide everything, prot\nshow cartoon, prot\ncolor gray80, prot\nset cartoon_transparency, 0.5\n")
        f.write(f"select score3, {get_sel(df[df['_score_val'] == 3])}\nshow sticks, score3\ncolor red, score3\nlabel score3 and name CA, '%s%s' % (resn, resi)\n")
        f.write(f"select score2, {get_sel(df[df['_score_val'] == 2])}\nshow sticks, score2\ncolor orange, score2\nzoom score3 or score2\nbg_color white\nset ray_shadows, 0\n")

if __name__ == "__main__": main()
