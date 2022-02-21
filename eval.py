import os
import numpy as np
import argparse
import pandas as pd
from filelock import FileLock
from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from data.stanford_drone_split import get_stanford_drone_split
from utils.utils import print_log, AverageMeter, isfile, print_log, AverageMeter, isfile, isfolder, find_unique_common_from_lists, load_list_from_folder, load_txt_file


""" Metrics """

def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.mean(axis=-1)                       # samples
        ade += dist.min(axis=0)                         # (1, )
    ade /= len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist[..., -1]                            # samples 
        fde += dist.min(axis=0)                         # (1, )
    fde /= len(pred_arr)
    return fde


def _collision(path1,
               path2,
               n_predictions=12,
               person_radius=0.1,
               inter_parts=2):
    """Check if there is collision or not.
    Source: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/evaluator/eval_utils.py#L22
    """

    def getinsidepoints(p1, p2, parts=2):
        """return: equally distanced points between starting and ending "control" points"""

        return np.array(
            (np.linspace(p1[0], p2[0],
                         parts + 1), np.linspace(p1[1], p2[1], parts + 1)))

    for i in range(len(path1) - 1):
        p1, p2 = [path1[i][0], path1[i][1]], [path1[i + 1][0], path1[i + 1][1]]
        p3, p4 = [path2[i][0], path2[i][1]], [path2[i + 1][0], path2[i + 1][1]]
        # Check current point
        if np.min(np.linalg.norm(np.array(p1) - np.array(p3)),
                  axis=0) <= 2 * person_radius:
            return True

        # Compute inbetween points
        if np.min(np.linalg.norm(getinsidepoints(p1, p2, inter_parts) - getinsidepoints(p3, p4, inter_parts), axis=0)) \
           <= 2 * person_radius:
            return True

    return False


def compute_CR(pred_arr,
               gt_arr,
               pred=False,
               gt=False,
               aggregation='max',
               **kwargs):
    """Compute collision rate.
    Input:
        - pred_arr: (np.array) (n_pedestrian, n_samples, timesteps, 4)
        - gt_arr: (np.array) (n_pedestrian, timesteps, 4)
    Return:
        Collision rates
    """
    # Collision rate is calculated per sample
    # (n_agents, n_samples, timesteps, 4) > (n_samples, n_agents, timesteps 4)
    pred_arr = np.array(pred_arr).transpose(1, 0, 2, 3)
    gt_arr = np.array(gt_arr)

    n_samples, n_ped, _, _ = pred_arr.shape

    # For each pedestrian, store 1 if they had collision. Else 0.
    n_ped_with_col_pred = np.zeros((n_samples, 1))
    n_ped_with_col_gt = np.zeros((n_samples, 1))

    # If only 1 ped, there will be no collision. Else, check if each pedestrian
    # has a collision with other agents.
    if n_ped > 1:
        for sample_idx, sample in enumerate(pred_arr):
            n_ped_with_col_pred_per_sample = np.zeros((sample.shape[0], 1))
            n_ped_with_col_gt_per_sample = np.zeros((sample.shape[0], 1))
            for pred_idx, pred_sample in enumerate(sample):
                others = np.concatenate(
                    (sample[:pred_idx], sample[pred_idx + 1:]))
                for other_idx, other_sample in enumerate(others):
                    if pred and _collision(pred_sample, other_sample):
                        n_ped_with_col_pred_per_sample[pred_idx] = 1
                    if gt and _collision(pred_sample, gt_arr[pred_idx]):
                        n_ped_with_col_gt_per_sample[pred_idx] = 1

            if pred:
                n_ped_with_col_pred[
                    sample_idx] += n_ped_with_col_pred_per_sample.sum()
            elif gt:
                n_ped_with_col_gt[
                    sample_idx] += n_ped_with_col_gt_per_sample.sum()

    if aggregation == 'mean':
        cr_pred = n_ped_with_col_pred.mean()
        cr_gt = n_ped_with_col_gt.mean()
    elif aggregation == 'min':
        cr_pred = n_ped_with_col_pred.min()
        cr_gt = n_ped_with_col_gt.min()
    elif aggregation == 'max':
        cr_pred = n_ped_with_col_pred.max()
        cr_gt = n_ped_with_col_gt.max()
    else:
        raise NotImplementedError()

    # Multiply by 100 to make it percentage
    cr_pred *= 100
    cr_gt *= 100

    if pred:
        return cr_pred / n_ped
    elif gt:
        return cr_gt / n_ped


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
                    ind = len(df.columns)-1
                df.insert(ind, mname, 0)
            df.loc[index, mname] = meter.avg
        df.to_csv(csv_file, index=False, float_format='%f')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuscenes_pred')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--label', default='')
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--data', default='test')
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    results_dir = args.results_dir
    
    resize = 1.0
    if dataset == 'nuscenes_pred':   # nuscenes
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
        'CR_gt': compute_CR,
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    print_log('\n\nnumber of sequences to evaluate is %d' % len(seq_eval), log_file)
    for seq_name in seq_eval:
        # load GT raw data
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name+'.txt'))
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
            
        for data_file in data_filelist:      # each example e.g., seq_0001 - frame_000009
            # for reconsutrction or deterministic
            if isfile(data_file):
                all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                all_traj = np.expand_dims(all_traj, axis=0)                             # 1 x (frames x agents) x 4
            # for stochastic with multiple samples
            elif isfolder(data_file):
                sample_list, _ = load_list_from_folder(data_file)
                sample_all = []
                for sample in sample_list:
                    sample = np.loadtxt(sample, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                    sample_all.append(sample)
                all_traj = np.stack(sample_all, axis=0)                                # samples x (framex x agents) x 4
            else:
                assert False, 'error'

            # convert raw data to our format for evaluation
            id_list = np.unique(all_traj[:, :, 1])
            frame_list = np.unique(all_traj[:, :, 0])
            agent_traj = []
            gt_traj = []
            for idx in id_list:
                # GT traj
                gt_idx = gt_raw[gt_raw[:, 1] == idx]                          # frames x 4
                # predicted traj
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]                                # sample x frames x 4
                # filter data
                pred_idx, gt_idx = align_gt(pred_idx, gt_idx)
                # append
                agent_traj.append(pred_idx)
                gt_traj.append(gt_idx)

            """compute stats"""
            for stats_name, meter in stats_meter.items():
                func = stats_func[stats_name]
                stats_func_args = {'pred_arr': agent_traj, 'gt_arr': gt_traj}
                if stats_name == 'CR_pred':
                    stats_func_args['pred'] = True
                elif stats_name == 'CR_gt':
                    stats_func_args['gt'] = True
                
                value = func(**stats_func_args)
                meter.update(value, n=len(agent_traj))

            stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
            print_log(f'evaluating seq {seq_name:s}, forecasting frame {int(frame_list[0]):06d} to {int(frame_list[-1]):06d} {stats_str}', log_file)

    print_log('-' * 30 + ' STATS ' + '-' * 30, log_file)
    for name, meter in stats_meter.items():
        print_log(f'{meter.count} {name}: {meter.avg:.4f}', log_file)
    print_log('-' * 67, log_file)
    log_file.close()

    write_metrics_to_csv(stats_meter, csv_file, args.label, results_dir, args.epoch, args.data)