# MYC-MAX Cross-validation Visualization
# Score 3/3 = gold  (detected by all 3 methods)
# Score 2/3 = silver (detected by 2 methods)

load /home/marcel/Project1/MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/rep1_frame250_protein.pdb, protein
hide everything
show cartoon, protein
color grey80, protein
set cartoon_transparency, 0.3
bg_color white

load /home/marcel/Project1/MYC-MAX/MYC-MAX-500/charmm-gui/gromacs/mdpocket_correct_freq.dx, freq_map
isosurface pocket_surface, freq_map, 0.25
color tv_blue, pocket_surface
set transparency, 0.5, pocket_surface

select score_3, none
show sticks, score_3
color gold, score_3
set stick_radius, 0.35, score_3

select score_2, (chain A and resi 913) or (chain A and resi 916) or (chain A and resi 970) or (chain B and resi 214) or (chain B and resi 239) or (chain B and resi 248)
show sticks, score_2
color silver, score_2
set stick_radius, 0.25, score_2

zoom score_3 or score_2
set ray_shadows, 0
set antialias, 2
bg_color white

print "GOLD   = detected by all 3 methods (salt bridges + MM-GBSA + MDpocket)"
print "SILVER = detected by 2/3 methods"
