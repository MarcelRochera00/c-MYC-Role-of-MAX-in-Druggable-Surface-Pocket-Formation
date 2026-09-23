#!/usr/bin/env python3
"""

Pocket Identification (Step 4.1 in Workflow) - ChimeraX version
=================================================================
Processes the frequency grid produced by MDpocket on the concatenated trajectory:
  1. Thresholds grid at ISOVALUE.
  2. Identifies distinct spatial sub-pockets (connected voxel blobs).
  3. Identifies lining residues for each sub-pocket.
  4. Outputs summary tables, residue lists, and a ChimeraX visualization script
     (.cxc) that also renders a high-resolution image.

Requires: numpy, scipy, gridData, MDAnalysis, matplotlib
"""

import csv
import os
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from gridData import Grid
import MDAnalysis as mda
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================== CONFIG ===================================
FREQ_DX = "MYC-MAX/mdpout_freq_grid.dx"
DENS_DX = "MYC-MAX/mdpout_dens_grid.dx"
# Must be the SAME reference structure you fit everything onto and passed
# to mdpocket with -f (e.g. MYC-MAX-500's crystal_proteins.pdb).
STRUCTURE_PDB = "MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.pdb"

ISOVALUE = 0.5        # pick from what you saw in ChimeraX / your earlier isovalue scan
MIN_VOXELS = 5        # ignore tiny/spurious sub-pockets below this voxel count
RESIDUE_CUTOFF = 5.0  # Angstrom: max residue-CA-to-nearest-pocket-voxel distance to call it "lining" the pocket

OUTPUT_DIR = "Results/Pockets"
SUMMARY_SUBDIR = "summary"                  # subpockets_summary.csv, residues.csv, report.txt, subpockets.cxc go here
DEFINITIONS_SUBDIR = "pocket_definitions"   # SP{n}_pocket_grid.pdb (round-2 input) go here
# Each SP receives its own subdirectory to store MDpocket outputs.

OUT_SUMMARY_CSV = "subpockets_core_summary.csv"
OUT_RESIDUES_CSV = "subpockets_core_residues.csv"
OUT_CXC = "subpockets.cxc"
OUT_LEGEND_PNG = "subpockets_legend.png"
OUT_RENDER_PNG = "subpockets_render.png"   # high-res image saved by the .cxc script itself
OUT_REPORT_TXT = "subpockets_core_report.txt"
SP_NAME_PREFIX = "SP"   # SP1, SP2, ... named by rank (largest volume first)
TOP_N_FOR_PML = 6
WANTED_POCKET_SUFFIX = "_pocket_grid.pdb"  # SP1_pocket_grid.pdb ... for mdpocket round-2 --selected_pocket input

# --- ChimeraX Scene Color Scheme ---
# Distinct colors to visually differentiate the main chains.
# (used only to color the cartoon in the .cxc script - chains are no longer
#  put in the legend image; identify them on-structure via the chain labels
#  the script itself places, or your own labels in ChimeraX.)
CHAIN_PALETTE = ["salmon", "paleturquoise", "wheat", "palegreen"]
# Pockets: plain, standard, high-saturation colors - kept distinct from the
# pale chain colors above so pockets pop out clearly.
POCKET_PALETTE = ["red", "green", "blue", "yellow", "magenta", "cyan"]

# --- Rendering / image quality ---
IMAGE_WIDTH = 3000
IMAGE_HEIGHT = 2400
IMAGE_SUPERSAMPLE = 3          # anti-aliasing factor for the saved PNG
# ===========================================================================


def find_subpockets(grid_obj, isovalue, min_voxels):
    """Threshold the grid and label connected voxel blobs (26-connectivity)."""
    data = np.asarray(grid_obj.grid)
    origin = np.asarray(grid_obj.origin)
    delta = np.asarray(grid_obj.delta)
    spacing = np.diag(delta) if delta.ndim == 2 else delta

    mask = data >= isovalue
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))

    pockets = []
    for label_id in range(1, n + 1):
        idx = np.argwhere(labeled == label_id)
        if len(idx) < min_voxels:
            continue
        coords = origin + idx * spacing
        freqs = data[labeled == label_id]
        pockets.append({
            "id": label_id,
            "n_voxels": len(idx),
            "volume_A3": float(len(idx) * np.prod(spacing)),
            "centroid": coords.mean(axis=0),
            "mean_freq": float(freqs.mean()),
            "max_freq": float(freqs.max()),
            "voxel_coords": coords,
            "voxel_idx": idx,  # integer grid indices, kept to re-sample the density grid at the exact same voxels
        })
    pockets.sort(key=lambda p: -p["volume_A3"])
    return pockets


def add_density_info(pockets, dens_dx_path):
    """
    Sample MDpocket's alpha-sphere density grid (mdpout_dens_grid.dx) at
    the voxel indices corresponding to each sub-pocket's frequency blob.
    Density provides complementary information to frequency, describing cavity
    depth and packing.

    Skips this step if the density grid is not found or dimensions mismatch.
    """
    try:
        dens_obj = Grid(dens_dx_path)
    except (IOError, OSError):
        print(f"  (note: {dens_dx_path} not found - skipping density descriptors)")
        for p in pockets:
            p["mean_density"] = None
            p["max_density"] = None
        return pockets

    dens_data = np.asarray(dens_obj.grid)
    freq_shape = None
    for p in pockets:
        freq_shape = p["voxel_idx"].max(axis=0) + 1 if freq_shape is None else freq_shape

    for p in pockets:
        idx = p["voxel_idx"]
        try:
            vals = dens_data[idx[:, 0], idx[:, 1], idx[:, 2]]
            p["mean_density"] = float(vals.mean())
            p["max_density"] = float(vals.max())
        except IndexError:
            print(f"  (note: density grid shape {dens_data.shape} doesn't match freq grid - "
                  f"skipping density for pocket {p['id']}. Were both grids from the same mdpocket run?)")
            p["mean_density"] = None
            p["max_density"] = None
    return pockets


def get_structure_chains(structure_pdb):
    """Unique chain identifiers in the reference structure, in a stable
    order, used to color each chain distinctly in the ChimeraX scene, plus
    each chain's CA centroid (used to place an on-structure chain label)."""
    u = mda.Universe(structure_pdb)
    ca = u.select_atoms("name CA")
    try:
        chain_ids = ca.chainIDs
    except (AttributeError, mda.exceptions.NoDataError):
        chain_ids = ca.segids  # fallback if no chain ID field in this PDB

    seen = []
    for c in chain_ids:
        if c not in seen:
            seen.append(c)

    centroids = {}
    for c in seen:
        mask = chain_ids == c
        centroids[c] = ca.positions[mask].mean(axis=0)

    return seen, centroids


def assign_residues(pockets, structure_pdb, cutoff):
    """For each sub-pocket, find residues whose CA is within `cutoff` of any
    voxel in that pocket (nearest-voxel distance, not centroid distance -
    respects the pocket's actual shape)."""
    u = mda.Universe(structure_pdb)
    ca = u.select_atoms("name CA")
    res_coords = ca.positions

    try:
        chains = ca.chainIDs
    except (AttributeError, mda.exceptions.NoDataError):
        chains = ca.segids  # fallback if no chain ID field in this PDB

    for pocket in pockets:
        tree = cKDTree(pocket["voxel_coords"])
        dists, _ = tree.query(res_coords, k=1)
        hits = np.where(dists <= cutoff)[0]
        pocket["residues"] = sorted(
            [{"chain": chains[i], "resnum": int(ca.resnums[i]),
              "resname": ca.resnames[i], "dist_to_pocket": float(dists[i])}
             for i in hits],
            key=lambda r: r["dist_to_pocket"])
    return pockets


def _chimerax_residue_spec(model_id, residues):
    """Build a ChimeraX atom-spec selecting the given residues, grouped by
    chain: '#1/A:12,34,56 #1/B:78,90'."""
    by_chain = {}
    for r in residues:
        by_chain.setdefault(r["chain"], []).append(r["resnum"])
    parts = []
    for chain, resnums in by_chain.items():
        resnum_list = ",".join(str(n) for n in sorted(set(resnums)))
        parts.append(f"#{model_id}/{chain}:{resnum_list}")
    return " ".join(parts)


def write_outputs(pockets, chains, chain_centroids):
    summary_dir = os.path.join(OUTPUT_DIR, SUMMARY_SUBDIR)
    os.makedirs(summary_dir, exist_ok=True)

    def out(fname):
        return os.path.join(summary_dir, fname)

    # ---- summary CSV ----
    with open(out(OUT_SUMMARY_CSV), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sp_name", "pocket_id", "rank", "n_voxels", "volume_A3", "mean_freq",
                    "max_freq", "mean_density", "max_density", "centroid_x", "centroid_y",
                    "centroid_z", "n_residues", "chains_spanned", "is_interface_pocket"])
        for rank, p in enumerate(pockets, 1):
            cx, cy, cz = p["centroid"]
            p_chains = sorted(set(r["chain"] for r in p["residues"]))
            w.writerow([f"{SP_NAME_PREFIX}{rank}", p["id"], rank, p["n_voxels"],
                        round(p["volume_A3"], 1), round(p["mean_freq"], 3), round(p["max_freq"], 3),
                        round(p["mean_density"], 2) if p["mean_density"] is not None else "NA",
                        round(p["max_density"], 2) if p["max_density"] is not None else "NA",
                        round(cx, 2), round(cy, 2), round(cz, 2), len(p["residues"]),
                        "|".join(p_chains), len(p_chains) > 1])

    # ---- residues CSV ----
    with open(out(OUT_RESIDUES_CSV), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sp_name", "pocket_id", "rank", "chain", "resnum", "resname", "dist_to_pocket_A"])
        for rank, p in enumerate(pockets, 1):
            for r in p["residues"]:
                w.writerow([f"{SP_NAME_PREFIX}{rank}", p["id"], rank, r["chain"], r["resnum"],
                            r["resname"], round(r["dist_to_pocket"], 2)])

    # ---- ChimeraX script ----
    chain_color_map = {ch: CHAIN_PALETTE[i % len(CHAIN_PALETTE)] for i, ch in enumerate(chains)}
    pocket_color_map = {}
    STRUCT_MODEL = 1          # model id assigned to the structure by "open"
    marker_model_id = 2       # markers (labels) each get their own model id, starting after the structure

    with open(out(OUT_CXC), "w") as f:
        f.write(f"# Auto-generated ChimeraX script - open with:\n")
        f.write(f"#   chimerax --script {OUT_CXC}\n")
        f.write(f"# or drag-and-drop this file onto a running ChimeraX window.\n\n")
        f.write(f"open {STRUCTURE_PDB}\n")
        f.write(f"style #{STRUCT_MODEL} stick\n")
        f.write("set bgColor white\n")

        f.write("\n# --- Chains: warm vs. cool colors so the two main chains are easy to tell apart ---\n")
        for ch, color in chain_color_map.items():
            f.write(f"color #{STRUCT_MODEL}/{ch} {color}\n")

        f.write("\n# --- Chain labels placed directly on the structure (no legend entry needed) ---\n")
        for ch, color in chain_color_map.items():
            cx, cy, cz = chain_centroids[ch]
            f.write(f"marker #{marker_model_id} position {cx:.2f},{cy:.2f},{cz:.2f} "
                    f"radius 0.01 color {color}\n")
            f.write(f"label #{marker_model_id} text \"Chain {ch}\" height 2.0 color black\n")
            marker_model_id += 1

        f.write(f"\n# --- Sub-pockets (top {TOP_N_FOR_PML} by volume), shown as colored sticks. "
                f"See {OUT_LEGEND_PNG} for the pocket color key. ---\n")
        for rank, p in enumerate(pockets[:TOP_N_FOR_PML], 1):
            if not p["residues"]:
                continue
            sp_name = f"{SP_NAME_PREFIX}{rank}"
            color = POCKET_PALETTE[(rank - 1) % len(POCKET_PALETTE)]
            pocket_color_map[sp_name] = color
            sel_spec = _chimerax_residue_spec(STRUCT_MODEL, p["residues"])
            cx, cy, cz = p["centroid"]

            f.write(f"\n# {sp_name} (pocket {p['id']}, {p['volume_A3']:.0f} A^3, "
                    f"mean freq {p['mean_freq']:.2f}, {len(p['residues'])} residues)\n")
            f.write(f"select {sel_spec}\n")
            f.write(f"color sel {color}\n")
            f.write(f"show sel atoms\n")
            f.write(f"style sel stick\n")
            f.write("~select\n")
            # a free-floating marker at the pocket centroid, labeled with its
            # SP name, so the identity is readable straight off the structure
            f.write(f"marker #{marker_model_id} position {cx:.2f},{cy:.2f},{cz:.2f} "
                    f"radius 0.01 color {color}\n")
            f.write(f"label #{marker_model_id} text \"{sp_name}\" height 2.2 color {color}\n")
            marker_model_id += 1

        f.write("\nview\n")
        f.write(f"\n# High-resolution render (adjust width/height/supersample as needed)\n")
        f.write(f"save {OUT_RENDER_PNG} width {IMAGE_WIDTH} height {IMAGE_HEIGHT} "
                f"supersample {IMAGE_SUPERSAMPLE} transparentBackground false\n")
        f.write(f"\n# See {OUT_LEGEND_PNG} alongside this render for the pocket color key\n"
                "# (chain identity is already labeled directly on the structure above).\n")

    write_legend(pocket_color_map, out(OUT_LEGEND_PNG))

    # ---- human-readable report: SP name, residues by chain, all descriptors ----
    with open(out(OUT_REPORT_TXT), "w") as f:
        f.write("Sub-pocket report\n")
        f.write(f"Source frequency grid : {FREQ_DX}\n")
        f.write(f"Source density grid   : {DENS_DX}\n")
        f.write(f"Reference structure   : {STRUCTURE_PDB}\n")
        f.write(f"Isovalue (frequency)  : {ISOVALUE}\n")
        f.write(f"Min voxels per pocket : {MIN_VOXELS}\n")
        f.write(f"Residue cutoff        : {RESIDUE_CUTOFF} A (CA to nearest pocket voxel)\n")
        f.write(f"Total sub-pockets     : {len(pockets)}\n")
        f.write("=" * 70 + "\n\n")

        for rank, p in enumerate(pockets, 1):
            name = f"{SP_NAME_PREFIX}{rank}"
            p_chains = sorted(set(r["chain"] for r in p["residues"]))
            interface_tag = "INTERFACE POCKET (spans multiple chains)" if len(p_chains) > 1 \
                else "single-chain pocket"

            f.write(f"{name}  (mdpocket internal id {p['id']}, rank {rank} by volume)\n")
            f.write("-" * 50 + "\n")
            f.write(f"  {interface_tag}\n")
            f.write(f"  Volume            : {p['volume_A3']:.1f} A^3  ({p['n_voxels']} voxels)\n")
            f.write(f"  Frequency (mean/max) : {p['mean_freq']:.3f} / {p['max_freq']:.3f}\n")
            if p["mean_density"] is not None:
                f.write(f"  Density (mean/max)   : {p['mean_density']:.2f} / {p['max_density']:.2f}\n")
            else:
                f.write(f"  Density (mean/max)   : not available "
                        f"({DENS_DX} missing or grid mismatch)\n")
            cx, cy, cz = p["centroid"]
            f.write(f"  Centroid (x,y,z)  : ({cx:.2f}, {cy:.2f}, {cz:.2f})\n")
            f.write(f"  Chains spanned    : {', '.join(p_chains) if p_chains else '(none within cutoff)'}\n")
            f.write(f"  Residues ({len(p['residues'])} total, within {RESIDUE_CUTOFF} A):\n")

            for chain in p_chains:
                chain_res = [r for r in p["residues"] if r["chain"] == chain]
                res_str = ", ".join(f"{r['resname']}{r['resnum']} ({r['dist_to_pocket']:.2f} A)"
                                     for r in chain_res)
                f.write(f"    Chain {chain}: {res_str}\n")
            f.write("\n")


def write_legend(pocket_color_map, output_path):
    """Standalone color-key image for the sub-pockets only. Chains are no
    longer included here - they're identified with labels directly on the
    structure by the .cxc script instead."""
    n_entries = len(pocket_color_map)
    fig, ax = plt.subplots(figsize=(3.5, 1.0 + 0.35 * (n_entries + 2)))
    ax.axis("off")

    elements = [mpatches.Patch(facecolor=color, edgecolor="black", label=sp_name)
                for sp_name, color in pocket_color_map.items()]

    ax.legend(handles=elements, loc="center", frameon=True, fontsize=10,
              title="Sub-pocket color key", title_fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_wanted_pocket_pdbs(pockets):
    """
    Write a dummy-atom PDB for each sub-pocket representing its voxel grid.
    This generates the 'selected zone' files required for MDpocket's round-2
    characterization (--selected_pocket), enabling tracking of specific pocket
    descriptors (volume, hydrophobicity, polarity) across the trajectory.

    Pre-creates subdirectories (e.g., Results/Pockets/SP1/) for organized
    output storage of the round-2 descriptors.
    """
    definitions_dir = os.path.join(OUTPUT_DIR, DEFINITIONS_SUBDIR)
    os.makedirs(definitions_dir, exist_ok=True)

    written = []
    for rank, p in enumerate(pockets, 1):
        name = f"{SP_NAME_PREFIX}{rank}"

        def_path = os.path.join(definitions_dir, f"{name}{WANTED_POCKET_SUFFIX}")
        with open(def_path, "w") as f:
            for i, (x, y, z) in enumerate(p["voxel_coords"], 1):
                f.write(f"ATOM  {i:5d}  C   STP X   1    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("END\n")
        written.append(def_path)

        # pre-create this pocket's own output subfolder, ready for its round-2 run
        os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

    return written


def main():
    print(f"Reading {FREQ_DX} ...")
    grid_obj = Grid(FREQ_DX)
    print(f"  grid shape {grid_obj.grid.shape}")

    pockets = find_subpockets(grid_obj, ISOVALUE, MIN_VOXELS)
    print(f"Found {len(pockets)} sub-pockets at isovalue {ISOVALUE} (min {MIN_VOXELS} voxels)")

    pockets = add_density_info(pockets, DENS_DX)
    pockets = assign_residues(pockets, STRUCTURE_PDB, RESIDUE_CUTOFF)
    chains, chain_centroids = get_structure_chains(STRUCTURE_PDB)

    print(f"\n{'name':>6s} {'volume(A3)':>11s} {'mean_freq':>10s} {'mean_dens':>10s} {'n_res':>6s} {'chains':>8s}")
    for rank, p in enumerate(pockets, 1):
        p_chains = "|".join(sorted(set(r["chain"] for r in p["residues"])))
        dens_str = f"{p['mean_density']:.2f}" if p["mean_density"] is not None else "NA"
        print(f"{SP_NAME_PREFIX}{rank:<5d} {p['volume_A3']:11.1f} {p['mean_freq']:10.2f} "
              f"{dens_str:>10s} {len(p['residues']):6d} {p_chains:>8s}")

    write_outputs(pockets, chains, chain_centroids)
    wanted_paths = write_wanted_pocket_pdbs(pockets)
    print(f"\nWrote {OUT_SUMMARY_CSV}, {OUT_RESIDUES_CSV}, {OUT_CXC}, {OUT_LEGEND_PNG}, {OUT_REPORT_TXT} "
          f"into {OUTPUT_DIR}/{SUMMARY_SUBDIR}/")
    print(f"Wrote {len(wanted_paths)} wanted-pocket PDBs into {OUTPUT_DIR}/{DEFINITIONS_SUBDIR}/:")
    for p in wanted_paths:
        print(f"  {p}")
    print(f"Pre-created {OUTPUT_DIR}/SP1/ ... {OUTPUT_DIR}/SP{len(pockets)}/ for round-2 outputs")
    print(f"\nTo render: open ChimeraX and run 'open {OUTPUT_DIR}/{SUMMARY_SUBDIR}/{OUT_CXC}' "
          f"(or 'chimerax --script ...' from the command line). This will also save a "
          f"{IMAGE_WIDTH}x{IMAGE_HEIGHT} (supersample {IMAGE_SUPERSAMPLE}) render as {OUT_RENDER_PNG}.")
    print("\nRun mdpocket round 2 for each pocket, e.g. for SP1:")
    print(f"  mdpocket --trajectory_file traj_all_fit_v2.xtc --trajectory_format xtc \\\n"
          f"           -f {STRUCTURE_PDB} \\\n"
          f"           --selected_pocket {wanted_paths[0]} \\\n"
          f"           -o {OUTPUT_DIR}/SP1/SP1")


if __name__ == "__main__":
    main()