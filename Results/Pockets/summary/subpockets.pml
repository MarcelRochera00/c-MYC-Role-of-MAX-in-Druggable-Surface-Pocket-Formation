load MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.pdb, struct
hide everything, struct
show cartoon, struct
bg_color white
set ray_shadows, 0

# --- Chains: warm vs. cool colors so the two main chains are easy to tell apart ---
color salmon, struct and chain A
color palecyan, struct and chain B
set cartoon_transparency, 0.5, struct

# --- Sub-pockets (top 6 by volume), shown as colored sticks. See subpockets_legend.png for the key, or just read the on-screen labels below. ---

# SP1 (pocket 5, 93 A^3, mean freq 0.61, 4 residues)
select sp1_residues, (chain B and resi 249) or (chain B and resi 230) or (chain B and resi 248) or (chain B and resi 245)
color red, sp1_residues
show sticks, sp1_residues and not name C+N+O
pseudoatom sp1_label_anchor, pos=[55.31, 66.57, 61.84]
label sp1_label_anchor, "SP1"
color red, sp1_label_anchor

# SP2 (pocket 6, 89 A^3, mean freq 0.62, 6 residues)
select sp2_residues, (chain B and resi 239) or (chain A and resi 914) or (chain B and resi 238) or (chain A and resi 913) or (chain B and resi 240) or (chain A and resi 917)
color green, sp2_residues
show sticks, sp2_residues and not name C+N+O
pseudoatom sp2_label_anchor, pos=[58.89, 51.20, 47.91]
label sp2_label_anchor, "SP2"
color green, sp2_label_anchor

# SP3 (pocket 3, 45 A^3, mean freq 0.54, 5 residues)
select sp3_residues, (chain B and resi 214) or (chain A and resi 940) or (chain A and resi 939) or (chain A and resi 938) or (chain B and resi 218)
color blue, sp3_residues
show sticks, sp3_residues and not name C+N+O
pseudoatom sp3_label_anchor, pos=[42.88, 49.13, 43.56]
label sp3_label_anchor, "SP3"
color blue, sp3_label_anchor

# SP4 (pocket 1, 26 A^3, mean freq 0.54, 3 residues)
select sp4_residues, (chain A and resi 949) or (chain A and resi 948) or (chain A and resi 945)
color yellow, sp4_residues
show sticks, sp4_residues and not name C+N+O
pseudoatom sp4_label_anchor, pos=[36.77, 61.01, 57.97]
label sp4_label_anchor, "SP4"
color yellow, sp4_label_anchor

# SP5 (pocket 4, 21 A^3, mean freq 0.52, 2 residues)
select sp5_residues, (chain A and resi 918) or (chain A and resi 914)
color magenta, sp5_residues
show sticks, sp5_residues and not name C+N+O
pseudoatom sp5_label_anchor, pos=[50.86, 49.09, 46.87]
label sp5_label_anchor, "SP5"
color magenta, sp5_label_anchor

# SP6 (pocket 2, 18 A^3, mean freq 0.53, 3 residues)
select sp6_residues, (chain B and resi 254) or (chain B and resi 258) or (chain B and resi 257)
color cyan, sp6_residues
show sticks, sp6_residues and not name C+N+O
pseudoatom sp6_label_anchor, pos=[40.43, 64.70, 68.60]
label sp6_label_anchor, "SP6"
color cyan, sp6_label_anchor

set label_size, 20
set label_color, black
set label_outline_color, white
set label_font_id, 7
hide everything, *_label_anchor
show labels, *_label_anchor
zoom struct

# See subpockets_legend.png alongside this render for the chain + pocket color key
# (PyMOL doesn't render a legend into the 3D scene itself).
