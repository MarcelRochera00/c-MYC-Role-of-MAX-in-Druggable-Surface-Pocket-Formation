
# Carga la estructura
load MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/step3_input.pdb, protein
hide everything
show cartoon, protein
color grey80, protein

# Sub-pocket 1 — azul
select sp1_residues, (chain B and resi 248+231+249+252)
show sticks, sp1_residues
color blue, sp1_residues

# Sub-pocket 2 — rojo
select sp2_residues, (chain B and resi 214+215+218) or (chain A and resi 939+940)
show sticks, sp2_residues
color red, sp2_residues

# Sub-pocket 3 — naranja
select sp3_residues, (chain B and resi 239+240+238) or (chain A and resi 913+916+917)
show sticks, sp3_residues
color orange, sp3_residues

# Esferas en centroides
pseudoatom centroid_sp1, pos=[60.7, 70.5, 56.7]
pseudoatom centroid_sp2, pos=[80.0, 55.4, 63.0]
pseudoatom centroid_sp3, pos=[79.6, 69.0, 60.1]

show spheres, centroid_sp1 or centroid_sp2 or centroid_sp3
color blue,   centroid_sp1
color red,    centroid_sp2
color orange, centroid_sp3
set sphere_scale, 3.0

zoom sp1_residues or sp2_residues or sp3_residues
