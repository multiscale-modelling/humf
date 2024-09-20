import torch

# TODO: This implementation compares each pair of interaction sites to
# determine if they should interact. In an MD application, we initially obtain
# a neighbor list from the MD software that represents a graph. We should
# determine pairs of interacting sites from the information in that graph, i.e.
# two sites interact if they belong to molecules that are interconnected
# through a long distance edge.


def get_pairs(sites_pos, sites_params, sites_batch, sites_mol):
    same_frame = sites_batch.unsqueeze(0) == sites_batch.unsqueeze(1)
    different_mol = sites_mol.unsqueeze(0) != sites_mol.unsqueeze(1)
    interacting_pairs = torch.nonzero(
        same_frame & different_mol
    )  # [num_interactions, 2]
    pair_sites = sites_pos[interacting_pairs]  # [num_interactions, 2, 3]
    pair_distances = torch.norm(
        pair_sites[:, 0] - pair_sites[:, 1], dim=1
    )  # [num_interactions]
    pair_params = sites_params[
        interacting_pairs
    ]  # [num_interactions, 2, num_params_per_site]
    pair_frame = sites_batch[interacting_pairs[:, 0]]  # [num_interactions]
    return pair_distances, pair_params, pair_frame
