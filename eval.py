import numpy as np
from metrics import stats_func


def eval_one_seq(agent_traj, gt_traj, collision_rad, return_sample_vals=False):
    """new function, for returning necessary vals for plotting"""
    assert isinstance(gt_traj, np.ndarray) and len(
            gt_traj.shape) == 3, f"len(gt_traj.shape) should be 3 but is {len(gt_traj.shape)}"
    assert isinstance(agent_traj, np.ndarray) and len(
            agent_traj.shape) == 4, f"len(agent_traj.shape) should be 4 but is {len(agent_traj.shape)}"

    """compute stats"""
    values = {}
    ped_values = {}
    all_sample_vals = {}

    # 'ADE_joint'
    value, sample_vals, ped_vals, _ = stats_func['ADE_joint'](pred_arr=agent_traj, gt_arr=gt_traj,
                                                           collision_rad=collision_rad,
                                                           return_ped_vals=True,
                                                           return_sample_vals=return_sample_vals)
    values['ADE_joint'] = value
    ped_values['ADE_joint'] = ped_vals
    all_sample_vals['ADE'] = sample_vals
    # 'FDE_joint'
    value, sample_vals, ped_vals, _ = stats_func['FDE_joint'](pred_arr=agent_traj, gt_arr=gt_traj,
                                                           collision_rad=collision_rad,
                                                           return_ped_vals=True,
                                                           return_sample_vals=return_sample_vals)
    values['FDE_joint'] = value
    ped_values['FDE_joint'] = ped_vals
    all_sample_vals['FDE'] = sample_vals
    # 'ADE_marginal'
    value, _, ped_vals, argmins = stats_func['ADE_marginal'](pred_arr=agent_traj, gt_arr=gt_traj,
                                                          collision_rad=collision_rad,
                                                          return_ped_vals=True,
                                                          return_argmin=True)
    values['ADE_marginal'] = value
    ped_values['ADE_marginal'] = ped_vals

    # 'FDE_marginal'
    value, _, ped_vals, _ = stats_func['FDE_marginal'](pred_arr=agent_traj, gt_arr=gt_traj,
                                                 collision_rad=collision_rad,
                                                 return_ped_vals=True)
    values['FDE_marginal'] = value
    ped_values['FDE_marginal'] = ped_vals

    # 'CR_mean'
    value, sample_vals, ped_vals, collision_mats = stats_func['CR_mean'](pred_arr=agent_traj, gt_arr=gt_traj,
                                                                         collision_rad=collision_rad,
                                                                         return_sample_vals=return_sample_vals,
                                                                         return_ped_vals=True,
                                                                         return_collision_mat=True)
    values['CR_mean'] = value
    all_sample_vals['CR'] = sample_vals
    ped_values['CR_mean'] = ped_vals

    # # 'CR_mADE'
    # value, _, ped_vals, collision_mats_mADE = stats_func['CR_mADE'](pred_arr=agent_traj, gt_arr=gt_traj,
    #                                                              collision_rad=collision_rad,
    #                                                              return_ped_vals=True,
    #                                                              return_collision_mat=True)
    # values['CR_mADE'] = value
    # ped_values['CR_mADE'] = ped_vals
    # collision_mats.extend(collision_mats_mADE)
    #
    # # 'CR_mADEjoint'
    # value, _, ped_vals, _ = stats_func['CR_mADEjoint'](pred_arr=agent_traj, gt_arr=gt_traj,
    #                                              collision_rad=collision_rad,
    #                                              return_ped_vals=True)
    # values['CR_mADEjoint'] = value
    # ped_values['CR_mADEjoint'] = ped_vals

    return values, ped_values, all_sample_vals, argmins, collision_mats


def eval_one_seq_new(agent_traj, gt_traj, collision_rad, return_sample_vals=False):
    """new function, for returning necessary vals for plotting
    doesn't work right now"""
    assert isinstance(gt_traj, np.ndarray) and len(
        gt_traj.shape) == 3, f"len(gt_traj.shape) should be 3 but is {len(gt_traj.shape)}"
    assert isinstance(agent_traj, np.ndarray) and len(
        agent_traj.shape) == 4, f"len(agent_traj.shape) should be 4 but is {len(agent_traj.shape)}"

    """compute stats"""
    values = []
    ped_values = []
    all_sample_vals = {}
    argmins = None
    collision_mats = None

    for stats_name in stats_func:
        func = stats_func[stats_name]
        return_sample_vals_this_stat = return_sample_vals if stats_name in ['ADE_joint', 'FDE_joint',
                                                                            'CR_mean'] else False
        return_argmins_this_stat = return_sample_vals if stats_name == 'ADE_marginal' else False
        return_collision_mats_this_stat = return_sample_vals if stats_name in ['CR_mean',
                                                                               'CR_mADE'] else False  # ['CR_max', 'CR_mADE'] else False
        return_ped_vals_this_stat = True if return_sample_vals else False
        stats_func_args = {'pred_arr': agent_traj, 'gt_arr': gt_traj, 'collision_rad': collision_rad,
                           'return_sample_vals': return_sample_vals_this_stat,
                           'return_argmin': return_argmins_this_stat,
                           'return_ped_vals': return_ped_vals_this_stat,
                           'return_collision_mat': return_collision_mats_this_stat}
        value = func(**stats_func_args)

        if return_sample_vals_this_stat and return_ped_vals_this_stat:
            value, sample_vals, ped_vals = value
            all_sample_vals[stats_name.split('_')[0]] = sample_vals
        elif return_sample_vals_this_stat and not return_ped_vals_this_stat:
            value, sample_vals = value
            all_sample_vals[stats_name.split('_')[0]] = sample_vals
        if return_argmins_this_stat:
            value, sample_vals, ped_vals, argmins = value
        if return_collision_mats_this_stat:
            if collision_mats is None:
                value, ped_vals, collision_mats = value
            else:
                value, sample_vals, ped_vals, minADE_collision_mats = value
                collision_mats.extend(minADE_collision_mats)
                collision_mats = np.array(collision_mats)
        values.append(value)
        ped_values.append(ped_values)

    return values, ped_values, all_sample_vals, argmins, collision_mats


def eval_one_seq2(agent_traj, gt_traj, collision_rad, return_sample_vals=False):
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


if __name__ == '__main__':
    __spec__ = None
    pass
