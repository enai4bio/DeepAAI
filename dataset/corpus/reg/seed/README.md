### Note   
We remove instances in the seen/unseen test set and unseen test set that are similar to any instance in the training and validation sets in each seed. See 
Two Ab-Ag pairwise instances are considered to be similar when they have similar Abs (BlastP ≥ 90%), similar Ags (BlastP ≥ 90%), and the same neutralization/non-neutralization effects.

`antibody_index_in_pair_seed=0.npy`: antibody index after removing instances in the seen/unseen test set that are similar to any instance in the training and validation sets in seed 0.

`virus_index_in_pair_seed=0.npy`: virus index after removing instances in the seen/unseen test set that are similar to any instance in the training and validation sets in seed 0.

`all_label_mat_seed=0.npy`: label after removing instances in the seen/unseen test set that are similar to any instance in the training and validation sets in seed 0.