load /home/marcel/Project1/crystal_proteins.pdb, prot
hide everything, prot
show cartoon, prot
color gray80, prot
set cartoon_transparency, 0.5
select score3, none
show sticks, score3
color red, score3
label score3 and name CA, '%s%s' % (resn, resi)
select score2, (chain A and resi 913) or (chain A and resi 916) or (chain A and resi 970) or (chain B and resi 214) or (chain B and resi 239) or (chain B and resi 248)
show sticks, score2
color orange, score2
zoom score3 or score2
bg_color white
set ray_shadows, 0
