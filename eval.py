import numpy as np

from metrics import stats_func


def eval_one_seq(agent_traj, gt_traj, collision_rad, return_sample_vals=False):
    """new function, for returning necessary vals for plotting"""
    assert isinstance(gt_traj, np.ndarray) and len(gt_traj.shape) == 3, f"len(gt_traj.shape) should be 3 but is {len(gt_traj.shape)}"
    assert isinstance(agent_traj, np.ndarray) and len(agent_traj.shape) == 4, f"len(agent_traj.shape) should be 4 but is {len(agent_traj.shape)}"

    """compute stats"""
    values = []
    all_sample_vals = {}
    argmins = None
    collision_mats = None
    for stats_name in stats_func:
        func = stats_func[stats_name]
        return_sample_vals_this_stat = return_sample_vals if stats_name in ['ADE_joint', 'FDE_joint', 'CR_mean'] else False
        return_argmins_this_stat = return_sample_vals if stats_name == 'ADE_marginal' else False
        return_collision_mats_this_stat = return_sample_vals if stats_name in ['CR_max', 'CR_mADE'] else False
        stats_func_args = {'pred_arr': agent_traj, 'gt_arr': gt_traj, 'collision_rad': collision_rad,
                           'return_sample_vals': return_sample_vals_this_stat,
                           'return_argmin': return_argmins_this_stat,
                           'return_collision_mat': return_collision_mats_this_stat}
        value = func(**stats_func_args)
        if return_sample_vals_this_stat:
            value, sample_vals = value
            all_sample_vals[stats_name.split('_')[0]] = sample_vals
        if return_argmins_this_stat:
            value, argmins = value
        if return_collision_mats_this_stat:
            if collision_mats is None:
                value, collision_mats = value
            else:
                value, minADE_collision_mats = value
                collision_mats.extend(minADE_collision_mats)
                collision_mats = np.array(collision_mats)
        values.append(value)

    return values, all_sample_vals, argmins, collision_mats
