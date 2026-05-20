# MYC-MAX Sub-pocket Visualization
# Run: pymol mdpocket_analysis/visualize_subpockets.pml

# ── Estructura ──────────────────────────────
load /home/marcel/Project1/MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/rep1_frame250_protein.pdb, protein
hide everything
show cartoon, protein
color grey80, protein
set cartoon_transparency, 0.3
bg_color white

# ── Pocket surface (frecuencia >25%) ────────
load /home/marcel/Project1/MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/mdpocket_correct_freq.dx, freq_map
isosurface pocket_surface, freq_map, 0.25
color tv_blue, pocket_surface
set transparency, 0.5, pocket_surface

# ── SP1: Sub-pocket 1 (MAX interface) ──
select SP1, (chain B and resi 248) or (chain B and resi 231) or (chain B and resi 249) or (chain B and resi 252)
show sticks, SP1
color cyan, SP1
set stick_radius, 0.25, SP1

# ── SP2: Sub-pocket 2 (mixed, cationic) ──
select SP2, (chain B and resi 214) or (chain B and resi 215) or (chain B and resi 218) or (chain A and resi 939) or (chain A and resi 940)
show sticks, SP2
color yellow, SP2
set stick_radius, 0.25, SP2

# ── SP3: Sub-pocket 3 (mixed, central) ──
select SP3, (chain B and resi 239) or (chain B and resi 240) or (chain B and resi 238) or (chain A and resi 913) or (chain A and resi 916) or (chain A and resi 917)
show sticks, SP3
color magenta, SP3
set stick_radius, 0.25, SP3

# ── Labels ─────────────────────────────────
label (SP1 or SP2 or SP3) and name CA, "%s%s" % (resn, resi)
set label_size, 12
set label_color, black

# ── Vista final ─────────────────────────────
zoom SP1 or SP2 or SP3
set ray_shadows, 0
set antialias, 2
ray 1200, 1200

print "CYAN    = SP1: GLU248, SER231, TYR249, TYR252 (MAX)"
print "YELLOW  = SP2: ARG214, ARG215, ILE218 (MAX) + LYS939, VAL940 (MYC)"
print "MAGENTA = SP3: ARG239, ALA240, SER238 (MAX) + ARG913, GLU916, LEU917 (MYC)"
