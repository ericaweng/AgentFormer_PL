"""eval gt stats (collision metrics"""

import os
import argparse
from filelock import FileLock
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from data.stanford_drone_split import get_stanford_drone_split
from data.dataloader import data_generator
from utils.config import Config
from utils.utils import AverageMeter, find_unique_common_from_lists, load_list_from_folder
from viz_utils import plot_traj_anim

""" Metrics """
def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist.mean(axis=-1)  # samples
        ade += dist.min(axis=0)  # (1, )
    ade /= len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist[..., -1]  # samples
        fde += dist.min(axis=0)  # (1, )
    fde /= len(pred_arr)
    return fde


def _lineseg_dist(a, b):
    """
    https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    """
    # normalized tangent vector
    d = (b - a) / (np.linalg.norm(b - a, axis=-1, keepdims=True) + 1e-8)

    # signed parallel distance components
    s = (a * d).sum(axis=-1)
    t = (-b * d).sum(axis=-1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros_like(t)], axis=0)

    # perpendicular distance component
    c = np.cross(-a, d, axis=-1)

    return np.hypot(h, np.abs(c))


def _get_diffs_pred(traj):
    """Same order of ped pairs as pdist.
    Input:
        - traj: (ts, n_ped, 2)"""
    num_peds = traj.shape[1]
    return np.concatenate([
            np.tile(traj[:, ped_i:ped_i + 1],
                    (1, num_peds - ped_i - 1, 1)) - traj[:, ped_i + 1:]
            for ped_i in range(num_peds)
    ],
            axis=1)


def _get_diffs_gt(traj, gt_traj):
    """same order of ped pairs as pdist"""
    num_peds = traj.shape[1]
    return np.stack([
            np.tile(traj[:, ped_i:ped_i + 1], (1, num_peds, 1)) - gt_traj
            for ped_i in range(num_peds)
    ],
            axis=1)


def point_to_segment_dist_old(x1, y1, x2, y2, p1, p2):
    """
    Calculate the closest distance between start(p1, p2) and a line segment with two endpoints (x1, y1), (x2, y2)
    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((p1 - x1, p2 - y1), axis=-1)

    u = ((p1 - x1) * px + (p2 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest start to (p1, p2) on the line segment
    x = x1 + u * px
    y = y1 + u * py
    return np.linalg.norm((x - p1, y - p2), axis=-1)


def get_collisions_mat_old(sample_idx, pred_traj_fake, threshold=0.2):
    """threshold: radius + discomfort distance of agents"""
    pred_traj_fake.transpose(1, 0, 2)
    ts, num_peds, _ = pred_traj_fake.shape
    collision_mat = np.full((ts, num_peds, num_peds), False)
    collision_mat_vals = np.full((ts, num_peds, num_peds), np.inf)
    # dist_mat = np.full((ts, num_peds, num_peds), False)
    # test initial timesteps
    for ped_i, x_i in enumerate(pred_traj_fake[0]):
        for ped_j, x_j in enumerate(pred_traj_fake[0]):
            if ped_i == ped_j:
                continue
            closest_dist = np.linalg.norm(x_i - x_j) - threshold * 2
            if closest_dist < 0:
                collision_mat[0, ped_i, ped_j] = True
            collision_mat_vals[0, ped_i, ped_j] = closest_dist

    # test t-1 later timesteps
    for t in range(ts - 1):
        for ped_i, ((ped_ix, ped_iy), (ped_ix1, ped_iy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
            for ped_j, ((ped_jx, ped_jy), (ped_jx1, ped_jy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
                if ped_i == ped_j:
                    continue
                px = ped_ix - ped_jx
                py = ped_iy - ped_jy
                ex = ped_ix1 - ped_jx1
                ey = ped_iy1 - ped_jy1
                # closest distance between boundaries of two agents
                # closest_dist = point_to_segment_dist((px, py), (ex, ey), (0, 0)) - threshold * 2
                closest_dist = point_to_segment_dist_old(px, py, ex, ey, 0, 0) - threshold * 2
                # closest_dist_old = point_to_segment_dist_old(px, py, ex, ey, 0, 0) - threshold * 2
                # assert np.all(np.abs(closest_dist - closest_dist_old) < 1e-7)
                if closest_dist < 0:
                    collision_mat[t+1, ped_i, ped_j] = True
                collision_mat_vals[t + 1, ped_i, ped_j] = closest_dist
                # elif closest_dist < dmin:
                #     dmin = closest_dist

    # print("collision_mat.shape:", collision_mat.shape)
    # print("collision_mat:", collision_mat)
    # import ipdb; ipdb.set_trace()
    return sample_idx, np.any(np.any(collision_mat, axis=0), axis=0), collision_mat  # collision_mat_pred_t_bool
    # return sample_idx, n_ped_with_col_pred_per_sample, collision_mat#collision_mat_pred_t_bool
    # return collision_mat, collision_mat_vals


def check_collision_per_sample(sample_idx, sample, gt_arr, ped_radius=0.1):
    """sample: (num_peds, ts, 2) and same for gt_arr"""

    sample = sample.transpose(1, 0, 2)  # (ts, n_ped, 2)
    gt_arr = gt_arr.transpose(1, 0, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    # pred
    # Get collision for timestep=0
    collision_0_pred = pdist(sample[0]) < ped_radius
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(
            ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred = squareform(
            np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)
    # gt
    collision_0_gt = cdist(sample[0], gt_arr[0]) < ped_radius
    np.fill_diagonal(collision_0_gt, False)
    ped_pair_diffs_gt = _get_diffs_gt(sample, gt_arr)
    pxy_gt = ped_pair_diffs_gt[:-1].reshape(-1, 2)
    exy_gt = ped_pair_diffs_gt[1:].reshape(-1, 2)
    collision_t_gt = _lineseg_dist(pxy_gt, exy_gt).reshape(
            ts - 1, num_peds, num_peds) < ped_radius * 2
    for ped_mat in collision_t_gt:
        np.fill_diagonal(ped_mat, False)
    collision_mat_gt = np.any(collision_t_gt, axis=0) | collision_0_gt
    n_ped_with_col_gt_per_sample = np.any(collision_mat_gt, axis=0)

    return sample_idx, n_ped_with_col_pred_per_sample, n_ped_with_col_gt_per_sample


def compute_CR(pred_arr,
               gt_arr,
               pred=False,
               gt=False,
               aggregation='max',
               **kwargs):
    """Compute collision rate and collision-free likelihood.
    Input:
        - pred_arr: (np.array) (n_pedestrian, n_samples, timesteps, 4)
        - gt_arr: (np.array) (n_pedestrian, timesteps, 4)
    Return:
        Collision rates
    """
    # (n_agents, n_samples, timesteps, 4) > (n_samples, n_agents, timesteps 4)
    pred_arr = np.array(pred_arr).transpose(1, 0, 2, 3)

    n_sample, n_ped, _, _ = pred_arr.shape

    col_pred = np.zeros((n_sample))  # cr_pred

    if n_ped > 1:
        # with nool(processes=multiprocessing.cpu_count() - 1) as pool:
        #     r = pool.starmap(
        #             partial(check_collision_per_sample, gt_arr=gt_arr),
        #             enumerate(pred_arr))
        r = []
        for i, pa in enumerate(pred_arr):
            # import time
            # start = time.time()
            # tup = check_collision_per_sample_no_gt(i, pa)
            # print(time.time() - start)
            # start = time.time()
            # tup = get_collisions_mat_old(i, pa)
            # print(time.time() - start)
            # import ipdb; ipdb.set_trace()
            tup = get_collisions_mat_old(i, pa)
            r.append(tup)

        for sample_idx, n_ped_with_col_pred, _ in r:
            col_pred[sample_idx] += (n_ped_with_col_pred.sum())

    if aggregation == 'mean':
        cr_pred = col_pred.mean(axis=0)
    elif aggregation == 'min':
        cr_pred = col_pred.min(axis=0)
    elif aggregation == 'max':
        cr_pred = col_pred.max(axis=0)
    else:
        raise NotImplementedError()

    # Multiply by 100 to make it percentage
    cr_pred *= 100
    return cr_pred / n_ped, r[-1][-1] if n_ped > 1 else None


def check_collision_per_sample_no_gt(sample_idx, sample, ped_radius=0.1):
    """sample: (num_peds, ts, 2)"""

    sample = sample.transpose(1, 0, 2)  # (ts, n_ped, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    # pred
    # Get collision for timestep=0
    collision_0_pred = pdist(sample[0]) < ped_radius
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)

    collision_t_pred1 = _lineseg_dist(pxy, exy).reshape(
            ts - 1, num_ped_pairs) #< ped_radius * 2
    collision_t_pred = collision_t_pred1 < ped_radius * 2
    if np.any(collision_t_pred):
        collision_mat_pred_t = np.stack(
                [squareform(cm - ped_radius * 2) for cm in np.concatenate([pdist(sample[0])[np.newaxis, ...], collision_t_pred1])])
    collision_mat_pred = squareform(np.any(collision_t_pred, axis=0) | collision_0_pred)
    if True:
        collision_mat_pred_t_bool = np.stack([squareform(cm) for cm in np.concatenate([collision_0_pred[np.newaxis,...], collision_t_pred])])
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)
    # print(collision_mat_pred)
    if np.any(n_ped_with_col_pred_per_sample):
        print(list(zip(*np.where(collision_mat_pred))))

    return sample_idx, n_ped_with_col_pred_per_sample, collision_mat_pred_t_bool


def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new


def write_metrics_to_csv(stats_meter, csv_file, label, results_dir, epoch, data):
    lock = FileLock(f'{csv_file}.lock')
    with lock:
        df = pd.read_csv(csv_file)
        if not ((df['label'] == label) & (df['epoch'] == epoch)).any():
            df = df.append({'label': label, 'epoch': epoch}, ignore_index=True)
        index = (df['label'] == label) & (df['epoch'] == epoch)
        df.loc[index, 'results_dir'] = results_dir
        for metric, meter in stats_meter.items():
            mname = ('' if data != 'train' else 'train_') + metric
            if mname not in df.columns:
                if data != 'train':
                    ind = 0
                    for i, col in enumerate(df.columns):
                        if 'train' in col:
                            ind = i
                            break
                else:
                    ind = len(df.columns) - 1
                df.insert(ind, mname, 0)
            df.loc[index, mname] = meter.avg
        df.to_csv(csv_file, index=False, float_format='%f')


def eval_one_seq(gt_eval, stats_meter, stats_func, obs_traj=None, i=None):
    """compute stats"""
    gt_eval = [g.cpu().numpy() for g in gt_eval]
    gt_eval_unsqueeze = [g[np.newaxis, ...] for g in gt_eval]

    values = []
    agent_traj_nums = []
    for stats_name in stats_meter:
        func = stats_func[stats_name]
        stats_func_args = {'pred_arr': gt_eval_unsqueeze, 'gt_arr': gt_eval}
        if stats_name == 'CR_pred':
            stats_func_args['pred'] = True
        elif stats_name == 'CR_pred_mean':
            stats_func_args['pred'] = True
            stats_func_args['aggregation'] = 'mean'

        value = func(**stats_func_args)
        if isinstance(value, tuple):
            value, col_mat = value
        if 'CR_pred_mean' in stats_name and value > 0 and obs_traj is not None and i is not None:
            gt = np.array(gt_eval).transpose(1, 0, 2)
            # plot_traj_anim(obs_traj, f'viz/{i}.mp4', pred_traj_gt=gt, collision_mats=col_mat)
        values.append(value)
        agent_traj_nums.append(len(gt_eval))

    return values, agent_traj_nums


def main(args):
    cfg = Config(args.cfg)
    dataset = cfg.dataset
    results_dir = args.results_dir

    resize = 1.0
    if dataset == 'nuscenes_pred':  # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{args.data}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = locals()[f'seq_{args.data}']
    elif dataset == 'trajnet_sdd':
        data_root = 'datasets/trajnet_split'
        # data_root = 'datasets/stanford_drone_all'
        gt_dir = f'{data_root}/{args.data}'
        seq_train, seq_val, seq_test = get_stanford_drone_split()
        seq_eval = locals()[f'seq_{args.data}']
        # resize = 0.25
        indices = [0, 1, 2, 3]
    elif dataset == 'sdd':
        data_root = 'datasets/stanford_drone_all'
        gt_dir = f'{data_root}/{args.data}'
        seq_train, seq_val, seq_test = get_stanford_drone_split()
        seq_eval = locals()[f'seq_{args.data}']
        indices = [0, 1, 2, 3]
    else:  # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(dataset)
        seq_eval = locals()[f'seq_{args.data}']
        indices = [0, 1, 13, 15]

    stats_func = {
            'ADE': compute_ADE,
            'FDE': compute_FDE,
            'CR_pred': compute_CR,
            'CR_pred_mean': compute_CR,
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    _, num_seq = load_list_from_folder(gt_dir)
    print('\n\nnumber of sequences to evaluate is %d' % len(seq_eval))
    print('number of sequences to evaluate is %d' % num_seq)

    log = open(os.path.join(cfg.log_dir, '../temp.txt'), 'a+')
    generator = data_generator(cfg, log, split=args.data, phase='testing')
    args_list = []
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        pre_motion = np.array([g.cpu().numpy() for g in data['pre_motion_3D']]).transpose(1, 0, 2)
        args_list.append((data['fut_motion_3D'], stats_meter, stats_func, pre_motion, f"{data['seq']}-{data['frame']}"))
    print("num sequences:", len(args_list))
    all_meters_values, all_meters_agent_traj_nums = [], []

    if args.mp:
        with Pool() as pool:
            all_meters_values, all_meters_agent_traj_nums = zip(*pool.starmap(partial(eval_one_seq,
                                                                                      collision_rad=0.1,
                                                                                      return_agent_traj_nums=True), args_list))
            for meter, values, agent_traj_num in zip(stats_meter.values(), zip(*all_meters_values), zip(*all_meters_agent_traj_nums)):
                meter.update((np.sum(np.array(values) * np.array(agent_traj_num)) / np.sum(agent_traj_num)).item(),
                             n=np.sum(agent_traj_num).item())
    else:
        for arg in args_list:  # each example e.g., seq_0001 - frame_000009
            meters, agent_traj_nums = eval_one_seq(*arg)
            # all_meters_values.append(meters)
            # all_meters_agent_traj_nums.append(agent_traj_nums)
            for meter, value, agent_traj_num in zip(stats_meter.values(), meters, agent_traj_nums):
                meter.update(value, n=agent_traj_num)
            # meter.update((np.sum(np.array(values) * np.array(agent_traj_num)) / np.sum(agent_traj_num)).item(),
            #              n=np.sum(agent_traj_num).item())

    print('-' * 30 + ' STATS ' + '-' * 30)
    for name, meter in stats_meter.items():
        print(f'{meter.count} {name}: {meter.avg:.4f}')
    print('-' * 67)
    for name, meter in stats_meter.items():
        if 'gt' not in name:
            print(f"{meter.avg:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data', default='test')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--mp', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
