#!/usr/bin/env python3
"""
extract_subpockets.py

Takes the SINGLE frequency grid produced by running mdpocket once on your
concatenated, common-reference-aligned trajectory (mdpout_freq_grid.dx) and:
  1. thresholds it at ISOVALUE
  2. splits it into distinct spatial sub-pockets (connected voxel blobs)
  3. finds which residues line each sub-pocket
  4. writes a summary CSV, a residues CSV, and a PyMOL script to view them

This is deliberately much simpler than the old per-replica consensus script:
because all 3 replicas were combined into ONE trajectory before mdpocket
ran, there's only one grid here. No cross-replica blob matching, no
alignment step (that already happened upstream, before mdpocket ran) - just
threshold -> label -> assign residues.

Requires: numpy, scipy, gridData (pip install gridData), MDAnalysis, matplotlib
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

ISOVALUE = 0.5        # pick from what you saw in PyMOL / your earlier isovalue scan
MIN_VOXELS = 5        # ignore tiny/spurious sub-pockets below this voxel count
RESIDUE_CUTOFF = 5.0  # Angstrom: max residue-CA-to-nearest-pocket-voxel distance to call it "lining" the pocket

OUTPUT_DIR = "Results/Pockets"
SUMMARY_SUBDIR = "summary"                  # subpockets_summary.csv, residues.csv, report.txt, subpockets.pml go here
DEFINITIONS_SUBDIR = "pocket_definitions"   # SP{n}_pocket_grid.pdb (round-2 input) go here
# each SP gets its own subfolder directly under OUTPUT_DIR (e.g. Results/Pockets/SP1/)
# for its round-2 mdpocket output files (descriptors.txt, freq_grid.dx, etc.) - keeps
# 6 pockets' worth of mdpocket output files from piling up in one flat folder

OUT_SUMMARY_CSV = "subpockets_core_summary.csv"
OUT_RESIDUES_CSV = "subpockets_core_residues.csv"
OUT_PML = "subpockets.pml"
OUT_LEGEND_PNG = "subpockets_legend.png"
OUT_REPORT_TXT = "subpockets_core_report.txt"
SP_NAME_PREFIX = "SP"   # SP1, SP2, ... named by rank (largest volume first)
TOP_N_FOR_PML = 6
WANTED_POCKET_SUFFIX = "_pocket_grid.pdb"  # SP1_pocket_grid.pdb ... for mdpocket round-2 --selected_pocket input

# --- Color scheme for the PyMOL scene ---
# Chains: warm vs. cool so the two main chains are easy to tell apart at a
# glance, not just two shades of gray.
CHAIN_PALETTE = ["salmon", "palecyan", "wheat", "palegreen"]
# Pockets: plain, standard, high-saturation colors.
POCKET_PALETTE = ["red", "green", "blue", "yellow", "magenta", "cyan"]
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
    """Sample mdpocket's alpha-sphere density grid (mdpout_dens_grid.dx) at
    the exact same voxel indices as each sub-pocket's frequency blob.
    Density is a separate descriptor from frequency: frequency says "how
    often is this point part of a pocket", density says "how many alpha
    spheres (packing/cavity depth) are found there" - a shallow, rarely-open
    pocket and a deep, often-open one can have similar frequency but very
    different density, so this is worth reporting alongside frequency
    rather than assuming they track each other.

    Silently skips this enrichment (with a warning) if the density grid
    isn't found or its shape doesn't match the frequency grid - it's a
    bonus descriptor, not something that should crash the whole extraction.
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
    order, used to color each chain distinctly in the PyMOL scene."""
    u = mda.Universe(structure_pdb)
    ca = u.select_atoms("name CA")
    try:
        chains = ca.chainIDs
    except (AttributeError, mda.exceptions.NoDataError):
        chains = ca.segids  # fallback if no chain ID field in this PDB
    seen = []
    for c in chains:
        if c not in seen:
            seen.append(c)
    return seen


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


def write_outputs(pockets, chains):
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

    # ---- PyMOL script ----
    chain_color_map = {ch: CHAIN_PALETTE[i % len(CHAIN_PALETTE)] for i, ch in enumerate(chains)}
    pocket_color_map = {}
    with open(out(OUT_PML), "w") as f:
        f.write(f"load {STRUCTURE_PDB}, struct\n")
        f.write("hide everything, struct\nshow cartoon, struct\n")
        f.write("bg_color white\nset ray_shadows, 0\n")
        f.write("\n# --- Chains: warm vs. cool colors so the two main chains are easy to tell apart ---\n")
        for ch, color in chain_color_map.items():
            f.write(f"color {color}, struct and chain {ch}\n")
        f.write("set cartoon_transparency, 0.5, struct\n")

        f.write(f"\n# --- Sub-pockets (top {TOP_N_FOR_PML} by volume), shown as colored sticks. "
                f"See {OUT_LEGEND_PNG} for the key, or just read the on-screen labels below. ---\n")
        for rank, p in enumerate(pockets[:TOP_N_FOR_PML], 1):
            if not p["residues"]:
                continue
            sp_name = f"{SP_NAME_PREFIX}{rank}"
            sel = " or ".join(f"(chain {r['chain']} and resi {r['resnum']})" for r in p["residues"])
            color = POCKET_PALETTE[(rank - 1) % len(POCKET_PALETTE)]
            pocket_color_map[sp_name] = color
            cx, cy, cz = p["centroid"]
            f.write(f"\n# {sp_name} (pocket {p['id']}, {p['volume_A3']:.0f} A^3, "
                    f"mean freq {p['mean_freq']:.2f}, {len(p['residues'])} residues)\n")
            f.write(f"select {sp_name.lower()}_residues, {sel}\n")
            f.write(f"color {color}, {sp_name.lower()}_residues\n")
            f.write(f"show sticks, {sp_name.lower()}_residues and not name C+N+O\n")
            # a free-floating pseudoatom at the pocket centroid, labeled with
            # its SP name, so the identity is readable straight off the
            # structure without cross-referencing the legend image
            f.write(f"pseudoatom {sp_name.lower()}_label_anchor, pos=[{cx:.2f}, {cy:.2f}, {cz:.2f}]\n")
            f.write(f"label {sp_name.lower()}_label_anchor, \"{sp_name}\"\n")
            f.write(f"color {color}, {sp_name.lower()}_label_anchor\n")

        f.write("\nset label_size, 20\nset label_color, black\n"
                "set label_outline_color, white\nset label_font_id, 7\n")
        f.write("hide everything, *_label_anchor\nshow labels, *_label_anchor\n")
        f.write("zoom struct\n")
        f.write(f"\n# See {OUT_LEGEND_PNG} alongside this render for the chain + pocket color key\n"
                "# (PyMOL doesn't render a legend into the 3D scene itself).\n")

    write_legend(chain_color_map, pocket_color_map, out(OUT_LEGEND_PNG))

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


_PYMOL_ONLY_COLORS = {
    "palecyan": "#AAFFFF",  # PyMOL's palecyan isn't a matplotlib/CSS4 name
}


def _mpl_color(pymol_color_name):
    """Most PyMOL color names (e.g. 'wheat', 'palegreen', 'salmon') are also
    valid matplotlib names, but a few (like PyMOL's grayNN/greyNN shades, or
    palecyan) aren't - convert those to a matplotlib-compatible equivalent."""
    name = pymol_color_name.lower()
    if name in _PYMOL_ONLY_COLORS:
        return _PYMOL_ONLY_COLORS[name]
    if name.startswith("gray") or name.startswith("grey"):
        digits = name[4:]
        if digits.isdigit():
            return str(int(digits) / 100)
    return pymol_color_name


def write_legend(chain_color_map, pocket_color_map, output_path):
    """Standalone color-key image pairing chains and pockets to the colors
    used in subpockets.pml, since PyMOL itself has no built-in legend panel
    for a 3D scene."""
    n_entries = len(chain_color_map) + len(pocket_color_map)
    fig, ax = plt.subplots(figsize=(3.5, 1.0 + 0.35 * (n_entries + 2)))
    ax.axis("off")

    elements = []
    for ch, color in chain_color_map.items():
        elements.append(mpatches.Patch(facecolor=_mpl_color(color), edgecolor="black", label=f"Chain {ch}"))
    for sp_name, color in pocket_color_map.items():
        elements.append(mpatches.Patch(facecolor=_mpl_color(color), edgecolor="black", label=sp_name))

    ax.legend(handles=elements, loc="center", frameon=True, fontsize=10,
              title="Structure color key", title_fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_wanted_pocket_pdbs(pockets):
    """Write one dummy-atom PDB per sub-pocket, at its voxel grid positions,
    into OUTPUT_DIR/pocket_definitions/. This is the 'selected zone' file
    format mdpocket's round-2 mode needs (fed via --selected_pocket) to
    track that specific pocket's descriptors (volume, hydrophobicity,
    polarity, charge, ...) per snapshot across the whole trajectory -
    MDpocket doesn't detect/track pocket identity across frames on its own,
    so this file is what tells it which zone to measure.

    Also pre-creates OUTPUT_DIR/SP{n}/ for each pocket - that's where you
    should point mdpocket's -o for that pocket's round-2 run, so each
    pocket's own set of output files (descriptors.txt, freq_grid.dx, etc.)
    lands in its own folder instead of all 6 pockets' files mixing together
    in one flat directory.
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
    chains = get_structure_chains(STRUCTURE_PDB)

    print(f"\n{'name':>6s} {'volume(A3)':>11s} {'mean_freq':>10s} {'mean_dens':>10s} {'n_res':>6s} {'chains':>8s}")
    for rank, p in enumerate(pockets, 1):
        p_chains = "|".join(sorted(set(r["chain"] for r in p["residues"])))
        dens_str = f"{p['mean_density']:.2f}" if p["mean_density"] is not None else "NA"
        print(f"{SP_NAME_PREFIX}{rank:<5d} {p['volume_A3']:11.1f} {p['mean_freq']:10.2f} "
              f"{dens_str:>10s} {len(p['residues']):6d} {p_chains:>8s}")

    write_outputs(pockets, chains)
    wanted_paths = write_wanted_pocket_pdbs(pockets)
    print(f"\nWrote {OUT_SUMMARY_CSV}, {OUT_RESIDUES_CSV}, {OUT_PML}, {OUT_LEGEND_PNG}, {OUT_REPORT_TXT} "
          f"into {OUTPUT_DIR}/{SUMMARY_SUBDIR}/")
    print(f"Wrote {len(wanted_paths)} wanted-pocket PDBs into {OUTPUT_DIR}/{DEFINITIONS_SUBDIR}/:")
    for p in wanted_paths:
        print(f"  {p}")
    print(f"Pre-created {OUTPUT_DIR}/SP1/ ... {OUTPUT_DIR}/SP{len(pockets)}/ for round-2 outputs")
    print("\nRun mdpocket round 2 for each pocket, e.g. for SP1:")
    print(f"  mdpocket --trajectory_file traj_all_fit_v2.xtc --trajectory_format xtc \\\n"
          f"           -f {STRUCTURE_PDB} \\\n"
          f"           --selected_pocket {wanted_paths[0]} \\\n"
          f"           -o {OUTPUT_DIR}/SP1/SP1")


if __name__ == "__main__":
    main()