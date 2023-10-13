import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from functools import partial


def compute_ADE_joint(pred_arr, gt_arr, return_sample_vals=False, return_argmin=False, **kwargs):
    pred_arr = np.array(pred_arr)
    gt_arr = np.array(gt_arr)
    diff = pred_arr - np.expand_dims(gt_arr, axis=1)  # num_peds x samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # num_peds x samples x frames
    ade_per_sample = dist.mean(axis=-1).mean(axis=0)  # samples
    ade = ade_per_sample.min(axis=0)  # (1, )
    return_vals = [ade]
    if return_sample_vals:  # for each sample: the avg ped ADE
        return_vals.append(ade_per_sample)
    if return_argmin:  # index of sample that has min sequence ADE (1,)
        return_vals.append(ade_per_sample[:, np.newaxis].argmin(axis=0))
    return return_vals[0] if len(return_vals) == 1 else return_vals


def compute_FDE_joint(pred_arr, gt_arr, return_sample_vals=False, return_argmin=False, **kwargs):
    pred_arr = np.array(pred_arr)
    gt_arr = np.array(gt_arr)
    diff = pred_arr - np.expand_dims(gt_arr, axis=1)  # num_peds x samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # num_peds x samples x frames
    fde_per_sample = dist[..., -1].mean(axis=0)  # samples
    fde = fde_per_sample.min(axis=0)  # (1, )
    return_vals = [fde]
    if return_sample_vals:  # for each sample: the avg ped ADE (n_samples,)
        return_vals.append(fde_per_sample)
    if return_argmin:  # index of sample that has min sequence FDE (1,)
        return_vals.append(fde_per_sample[:, np.newaxis].argmin(axis=0))
    return return_vals[0] if len(return_vals) == 1 else return_vals


def compute_ADE_marginal(pred_arr, gt_arr, return_sample_vals=False, return_ped_vals=False,
                         return_argmin=False, **kwargs):
    """about 4 times faster due to numpy vectorization"""
    # assert pred_arr.shape[1] == 20, pred_arr.shape
    pred_arr = np.array(pred_arr)
    gt_arr = np.array(gt_arr)
    diff = pred_arr - np.expand_dims(gt_arr, axis=1)  # num_peds x samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # num_peds x samples x frames
    ades_per_sample = dist.mean(axis=-1)  # num_peds x samples
    made_per_ped = ades_per_sample.min(axis=-1)  # num_peds
    avg_made = made_per_ped.mean(axis=-1)  # (1,)
    return_vals = [avg_made]
    if return_sample_vals:  # for each sample: the avg ped ADE
        return_vals.append(ades_per_sample.mean(axis=0))
    if return_ped_vals:  # the ADE of each ped-sample (n_ped, samples)
        return_vals.append(ades_per_sample)
    if return_argmin:  # for each ped: index of sample that is argmin
        return_vals.append(ades_per_sample.argmin(axis=-1))
    return return_vals[0] if len(return_vals) == 1 else return_vals


def compute_FDE_marginal(pred_arr, gt_arr, return_sample_vals=False, return_ped_vals=False,
                         return_argmin=False, **kwargs):
    """about 4 times faster due to numpy vectorization"""
    pred_arr = np.array(pred_arr)
    gt_arr = np.array(gt_arr)
    diff = pred_arr - np.expand_dims(gt_arr, axis=1)  # num_peds x samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # num_peds x samples x frames
    fdes_per_sample = dist[..., -1]  # num_peds x samples
    mfde_per_ped = fdes_per_sample.min(axis=-1)  # num_peds
    avg_mfde = mfde_per_ped.mean(axis=-1)  # (1,)
    return_vals = [avg_mfde]
    if return_sample_vals:  # for each sample: the avg ped ADE (n_samples,)
        return_vals.append(fdes_per_sample.mean(axis=0))
    if return_ped_vals: # the FDE of each ped-sample (n_ped, samples)
        return_vals.append(fdes_per_sample)
    if return_argmin:  # for each ped: index of sample that is argmin (n_ped,)
        return_vals.append(fdes_per_sample.argmin(axis=-1))
    return return_vals[0] if len(return_vals) == 1 else return_vals


def _lineseg_dist(a, b):
    """
    https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
    """
    # reduce computation
    if np.all(a == b):
        return np.linalg.norm(-a, axis=1)

    # normalized tangent vector
    d = np.zeros_like(a)
    # assert np.all(np.all(a == b, axis=-1) == np.isnan(ans))
    a_eq_b = np.all(a == b, axis=-1)
    d[~a_eq_b] = (b - a)[~a_eq_b] / np.linalg.norm(b[~a_eq_b] - a[~a_eq_b], axis=-1, keepdims=True)

    # signed parallel distance components
    s = (a * d).sum(axis=-1)
    t = (-b * d).sum(axis=-1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros_like(t)], axis=0)

    # perpendicular distance component
    c = np.cross(-a, d, axis=-1)

    ans = np.hypot(h, np.abs(c))

    # edge case where agent stays still
    ans[a_eq_b] = np.linalg.norm(-a, axis=1)[a_eq_b]

    return ans


def _get_diffs_pred(traj):
    """Same order of ped pairs as pdist.
    Input:
        - traj: (ts, n_ped, 2)"""
    num_peds = traj.shape[1]
    return np.concatenate([np.tile(traj[:, ped_i:ped_i + 1], (1, num_peds - ped_i - 1, 1)) - traj[:, ped_i + 1:]
                           for ped_i in range(num_peds)], axis=1)


def _get_diffs_gt(traj, gt_traj):
    """same order of ped pairs as pdist"""
    num_peds = traj.shape[1]
    return np.stack([
            np.tile(traj[:, ped_i:ped_i + 1], (1, num_peds, 1)) - gt_traj
            for ped_i in range(num_peds)
    ],
            axis=1)


def check_collision_per_sample_no_gt(sample, ped_radius=0.1):
    """sample: (num_peds, ts, 2)"""

    sample = sample.transpose(1, 0, 2)  # (ts, n_ped, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    collision_0_pred = pdist(sample[0]) < ped_radius * 2
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred_t_bool = np.stack([squareform(cm) for cm in np.concatenate([collision_0_pred[np.newaxis,...], collision_t_pred])])
    collision_mat_pred = squareform(np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)

    return n_ped_with_col_pred_per_sample, collision_mat_pred_t_bool


def compute_CR(pred_arr,
               gt_arr,
               aggregation='max',
               return_sample_vals=False,
               return_collision_mat=False,
               collision_rad=None,
               k2=False,
               **kwargs):
    """Compute collision rate and collision-free likelihood.
    Input:
        - pred_arr: (np.array) (n_pedestrian, n_samples, timesteps, 4)
        - gt_arr: (np.array) (n_pedestrian, timesteps, 4)
        - k2: if True, computes the proportion of K^2 agent-pairs that collide.
    Return:
        Collision rates
    """
    n_ped, n_sample, _, _ = pred_arr.shape

    # if evaluating different indices
    if callable(aggregation):  # aggregation can be marginal ADE or FDE
        indices = aggregation(pred_arr, gt_arr, return_argmin=True)[-1]
        n_sample = 1
        if indices.shape[0] == 1:
            pred_arr = pred_arr[:, indices]
        elif indices.shape[0] == n_ped:
            assert len(indices.shape) == 1
            pred_arr = pred_arr[np.arange(n_ped), indices][:, np.newaxis]  # only 1 sample per ped
        else:
            raise RuntimeError(f'indices is wrong shape: is {indices.shape} but should be (1,) or ({n_ped},)')
        assert len(pred_arr.shape) == 4

    pred_arr = np.array(pred_arr).swapaxes(1, 0)
    # (n_agents, n_samples, timesteps, 4) > (n_samples, n_agents, timesteps 4)
    col_pred = np.zeros((n_sample))
    col_mats = []
    for sample_idx, pa in enumerate(pred_arr):
        if k2:
            n_ped_with_col_pred, col_mat = check_collision_per_sample_k2(pa, collision_rad)
            col_pred[sample_idx] += n_ped_with_col_pred.sum()
        else:
            n_ped_with_col_pred, col_mat = check_collision_per_sample_no_gt(pa, collision_rad)
            col_pred[sample_idx] += n_ped_with_col_pred.sum()
        col_mats.append(col_mat)

    if aggregation == 'mean':
        cr_pred = col_pred.mean(axis=0)
    elif aggregation == 'min':
        cr_pred = col_pred.min(axis=0)
    elif aggregation == 'max':
        cr_pred = col_pred.max(axis=0)
    elif callable(aggregation):
        cr_pred = col_pred[0]
    else:
        raise NotImplementedError()

    if k2:
        crs = [cr_pred / n_ped/(n_ped-1)]
        if return_sample_vals:
            crs.append(col_pred / n_ped/(n_ped-1))
    else:
        crs = [cr_pred / n_ped]
        if return_sample_vals:
            crs.append(col_pred / n_ped)
    if return_collision_mat:
        crs.append(col_mats if len(col_mats) > 0 else None)
    return tuple(crs) if len(crs) > 1 else crs[0]


stats_func = {
        'ADE_marginal': compute_ADE_marginal,
        'FDE_marginal': compute_FDE_marginal,
        'CR_mean': partial(compute_CR, aggregation='mean'),
        'ADE_joint': compute_ADE_joint,
        'FDE_joint': compute_FDE_joint,
        'CR_mADEjoint': partial(compute_CR, aggregation=compute_ADE_joint),
}
