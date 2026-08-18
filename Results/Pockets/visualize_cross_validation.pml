load /home/marcel/Desktop/Project1/MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/crystal_proteins.pdb, prot
hide everything, prot
show cartoon, prot
color salmon, prot and chain A
color palecyan, prot and chain B
set cartoon_transparency, 0

# --- Cross-validated (score >= 2) residues, in red ---
select flagged, (chain A and resi 913) or (chain A and resi 914) or (chain A and resi 916) or (chain A and resi 970) or (chain B and resi 239) or (chain B and resi 248) or (chain B and resi 254) or (chain B and resi 275)
show sticks, flagged and not name C+N+O
color red, flagged
# labels skipped (SHOW_LABELS = False) - add them manually in an
# image editor afterward, using cross_validation_unified.csv as
# the reference for which residue is which.

zoom flagged
bg_color white
set ray_shadows, 0
