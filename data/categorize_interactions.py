
from collections import defaultdict
import inspect

import numpy as np

from ped_interactions import INTERACTION_CAT_ABBRS, INTERACTION_CAT_TO_FN, INTERACTION_HPARAMS


def get_interaction_matrix_for_scene(scene, interaction_hparams=INTERACTION_HPARAMS):
    """scene: shape (ts=16/20, num_peds, 2)
    returns: int_cat_vec, shape (num_peds, num_int_cats) which maps each ped in the scene to its many-hot int_cat vector"""
    traj_len, num_peds, obs_dim = scene.shape
    assert obs_dim == 2

    datas = {int_cat: [] for int_cat in INTERACTION_CAT_ABBRS}
    infos = {int_cat: [] for int_cat in INTERACTION_CAT_ABBRS}

    for ped_i in range(num_peds):
        path = scene[:, ped_i]
        # All other agents
        neigh_path = np.concatenate(
            [scene[:, :ped_i], scene[:, ped_i + 1:]], axis=1)

        # iterate through the int_cats and add the interaction data and infos to a list
        for int_cat, int_cat_fn in INTERACTION_CAT_TO_FN.items():
            hparams = {
                key: value
                for key, value in {
                    **interaction_hparams, 'neigh_path': neigh_path
                }.items() if key in inspect.getfullargspec(int_cat_fn).args
            }
            in_int_cat, info = int_cat_fn(path, **hparams)
            datas[int_cat].append(in_int_cat)
            assert isinstance(info, dict)
            infos[int_cat].append(info)

    int_cats_vec = []
    new_infos = {
        int_cat: defaultdict(list)
        for int_cat in INTERACTION_CAT_ABBRS
    }
    for int_cat in INTERACTION_CAT_ABBRS:
        # for each ped, if they have at least one neighbor that causes them to fall into a certain category,
        # then classify that ped in this scene as falling into that category
        score = np.array(datas[int_cat]).flatten()
        int_cats_vec.append(score)

        # move the int_cat axis to the front, peds axis to the back
        for info in infos[int_cat]:  # iterating over peds
            for key, value in info.items():  # iterating over int_cats
                new_infos[int_cat][key].append(value)

        # make an array
        for int_cat in new_infos:
            for key, value in new_infos[int_cat].items():
                new_infos[int_cat][key] = np.array(value)

    vec = np.array(int_cats_vec).T
    return vec, new_infos
