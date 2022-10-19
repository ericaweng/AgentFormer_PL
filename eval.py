import os
import numpy as np
import argparse
import pandas as pd
from filelock import FileLock
from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from data.stanford_drone_split import get_stanford_drone_split
from utils.utils import print_log, AverageMeter, isfile, print_log, AverageMeter, isfile, isfolder, \
    find_unique_common_from_lists, load_list_from_folder, load_txt_file
import multiprocessing
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import pdist, squareform, cdist

""" Metrics """


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


def get_collisions_mat_old(sample_idx, pred_traj_fake, threshold=0.1):
    """threshold: radius + discomfort distance of agents"""
    pred_traj_fake = pred_traj_fake.transpose(1, 0, 2)
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
               gt_arr=None,
               pred=False,
               gt=False,
               aggregation='max',
               return_sample_crs=False,
               return_collision_mat=False,
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
    col_mats = []
    if n_ped > 1:
        # with nool(processes=multiprocessing.cpu_count() - 1) as pool:
        #     r = pool.starmap(
        #             partial(check_collision_per_sample, gt_arr=gt_arr),
        #             enumerate(pred_arr))
        # r = []
        for i, pa in enumerate(pred_arr):
            # r.append(check_collision_per_sample(i, pa, gt_arr))
            sample_idx, n_ped_with_col_pred, col_mat = get_collisions_mat_old(i, pa)
            col_mats.append(col_mat)
            # tup = get_collisions_mat_old(i, pa)
            # r.append(tup)

        # for sample_idx, n_ped_with_col_pred, _ in r:
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
    crs = [cr_pred / n_ped]
    if return_sample_crs:
        crs.append(col_pred / n_ped)
    if return_collision_mat:
        crs.append(col_mats)
    return tuple(crs) if len(crs) > 1 else crs[0]


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
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(
            ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred = squareform(
            np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)

    return sample_idx, n_ped_with_col_pred_per_sample


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


def eval_one_seq(gt_raw, data_file, stats_meter, stats_func):
    # for reconsutrction or deterministic
    if isfile(data_file):
        all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')  # (frames x agents) x 4
        all_traj = np.expand_dims(all_traj, axis=0)  # 1 x (frames x agents) x 4
    # for stochastic with multiple samples
    elif isfolder(data_file):
        sample_list, _ = load_list_from_folder(data_file)
        sample_list = sample_list[:20]
        sample_all = []
        for sample in sample_list:
            sample = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            sample_all.append(sample)
        all_traj = np.stack(sample_all, axis=0)  # samples x (framex x agents) x 4
        # assert len(sample_all) == 20
    else:
        assert False, 'error'

    # convert raw data to our format for evaluation
    id_list = np.unique(all_traj[:, :, 1])
    frame_list = np.unique(all_traj[:, :, 0])
    agent_traj = []
    gt_traj = []
    for idx in id_list:
        # GT traj
        gt_idx = gt_raw[gt_raw[:, 1] == idx]  # frames x 4
        # predicted traj
        ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
        pred_idx = all_traj[:, ind, :]  # sample x frames x 4
        # filter data
        pred_idx, gt_idx = align_gt(pred_idx, gt_idx)
        # append
        if not args.eval_gt:
            agent_traj.append(pred_idx)
        else:
            agent_traj.append(gt_idx[np.newaxis, ...])
        gt_traj.append(gt_idx)

    """compute stats"""
    values = []
    agent_traj_nums = []
    for stats_name in stats_meter:
        func = stats_func[stats_name]
        stats_func_args = {'pred_arr': agent_traj, 'gt_arr': gt_traj}
        if stats_name == 'CR_pred':
            stats_func_args['pred'] = True
        elif stats_name == 'CR_gt':
            stats_func_args['gt'] = True
        elif stats_name == 'CR_pred_mean':
            stats_func_args['pred'] = True
            stats_func_args['aggregation'] = 'mean'
        elif stats_name == 'CR_gt_mean':
            stats_func_args['gt'] = True
            stats_func_args['aggregation'] = 'mean'

        value = func(**stats_func_args)
        values.append(value)
        agent_traj_nums.append(len(agent_traj))
        # if value > 0 and stats_name == 'CR_pred':
        #     import ipdb; ipdb.set_trace()

        # stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
        # print_log(
        #         f'evaluating seq {seq_name:s}, forecasting frame {int(frame_list[0]):06d} to {int(frame_list[-1]):06d}',# {stats_str}',
        #         log_file)

    return values, agent_traj_nums


if __name__ == '__main__':
    __spec__ = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuscenes_pred')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--label', default='')
    parser.add_argument('--eval_gt', action='store_true', default=False)
    parser.add_argument('--multiprocess', '-mp' , action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--data', default='test')
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    results_dir = args.results_dir

    resize = 1.0
    if dataset == 'nuscenes_pred':  # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{args.data}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = globals()[f'seq_{args.data}']
        csv_file = f'metrics/metrics_nup{args.sample_num}.csv'
    elif dataset == 'trajnet_sdd':
        data_root = 'datasets/trajnet_split'
        # data_root = 'datasets/stanford_drone_all'
        gt_dir = f'{data_root}/{args.data}'
        seq_train, seq_val, seq_test = get_stanford_drone_split()
        seq_eval = globals()[f'seq_{args.data}']
        # resize = 0.25
        indices = [0, 1, 2, 3]
        csv_file = f'metrics/metrics_trajnet_sdd.csv'
    elif dataset == 'sdd':
        data_root = 'datasets/stanford_drone_all'
        gt_dir = f'{data_root}/{args.data}'
        seq_train, seq_val, seq_test = get_stanford_drone_split()
        seq_eval = globals()[f'seq_{args.data}']
        indices = [0, 1, 2, 3]
        csv_file = f'metrics/metrics_sdd.csv'
    else:  # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{args.dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(args.dataset)
        seq_eval = globals()[f'seq_{args.data}']
        indices = [0, 1, 13, 15]
        csv_file = f'metrics/metrics_{args.dataset}12.csv'

    if args.log_file is None:
        log_file = os.path.join(results_dir, 'log_eval.txt')
    else:
        log_file = args.log_file
    log_file = open(log_file, 'a+')
    print_log('loading results from %s' % results_dir, log_file)
    print_log('loading GT from %s' % gt_dir, log_file)

    stats_func = {
            'ADE': compute_ADE,
            'FDE': compute_FDE,
            'CR_pred': compute_CR,
            # 'CR_gt': compute_CR,
            'CR_pred_mean': compute_CR,
            # 'CR_gt_mean': compute_CR,
            # 'ACFL': compute_ACFL
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    _, num_seq = load_list_from_folder(gt_dir)
    print_log('\n\nnumber of sequences to evaluate is %d' % len(seq_eval), log_file)
    print_log('number of sequences to evaluate is %d' % num_seq, log_file)
    for seq_name in seq_eval:
        # load GT raw data
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name + '.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, indices][0].astype('float32')
            line_data[2:4] = line_data[2:4] * resize
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)
        if dataset == 'trajnet_sdd':
            gt_raw[:, 0] = np.round(gt_raw.astype(np.float)[:, 0] / 12.0)

        data_filelist, _ = load_list_from_folder(os.path.join(results_dir, seq_name))
        if args.multiprocess:
            args_list = [(gt_raw, data_file, stats_meter, stats_func) for data_file in data_filelist]
            with Pool() as pool:
                all_meters_values, all_meters_agent_traj_nums = zip(*pool.starmap(eval_one_seq, args_list))
        else:
            all_meters_values, all_meters_agent_traj_nums = [],[]
            for data_file in data_filelist:  # each example e.g., seq_0001 - frame_000009
                meters, agent_traj_nums = eval_one_seq(gt_raw, data_file, stats_meter, stats_func)
                all_meters_values.append(meters)
                all_meters_agent_traj_nums.append(agent_traj_nums)
        for meter, values, agent_traj_num in zip(stats_meter.values(), zip(*all_meters_values), zip(*all_meters_agent_traj_nums)):
            meter.update((np.sum(np.array(values) * np.array(agent_traj_num)) / np.sum(agent_traj_num)).item(),
                         n=np.sum(agent_traj_num).item())

    print_log('-' * 30 + ' STATS ' + '-' * 30, log_file)
    for name, meter in stats_meter.items():
        print_log(f'{meter.count} {name}: {meter.avg:.4f}', log_file)
    print_log('-' * 67, log_file)
    for name, meter in stats_meter.items():
        if 'gt' not in name:
            print_log(f"{meter.avg:.4f}", log_file)
    print_log(f'epoch: {args.epoch}', log_file)
    log_file.close()

    write_metrics_to_csv(stats_meter, csv_file, args.label, results_dir, args.epoch, args.data)