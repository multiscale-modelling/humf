import torch


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
