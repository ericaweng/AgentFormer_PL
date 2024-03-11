import os
import numpy as np
import argparse
from functools import partial
import pandas as pd
from filelock import FileLock

from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from data.stanford_drone_split import get_stanford_drone_split
from utils.utils import print_log, AverageMeter, isfile, print_log, AverageMeter, isfile, isfolder, \
    find_unique_common_from_lists, load_list_from_folder, load_txt_file
from multiprocessing import Pool
from metrics import stats_func


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


def get_gt_from_raw_and_preds_from_file(gt_raw, data_file):
    # for reconsutrction or deterministic
    if isfile(data_file):
        all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')  # (frames x agents) x 4
        all_traj = np.expand_dims(all_traj, axis=0)  # 1 x (frames x agents) x 4
    # for stochastic with multiple samples
    elif isfolder(data_file):
        sample_list, _ = load_list_from_folder(data_file)
        # sample_list = sample_list[:20]
        sample_all = []
        if len(sample_list) == 0:
            print(f'No samples in {data_file}')
            return [0] * len(stats_func), [0] * len(stats_func)
        for sample in sample_list:
            # if 'sample' not in sample:
            #     continue
            sample = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            sample_all.append(sample)
        all_traj = np.stack(sample_all, axis=0)  # samples x (framex x agents) x 4
        # assert len(sample_all) == 20
    else:
        assert False, 'error'

    # convert raw data to our format for evaluation
    id_list = np.unique(all_traj[:, :, 1])
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
    return np.array(gt_traj), np.array(agent_traj)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuscenes_pred')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--label', default='')
    parser.add_argument('--eval_gt', action='store_true', default=False)
    parser.add_argument('--multiprocess', '-mp' , action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--collision_rad', type=int, default=0.1)
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--data', default='test')
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    collision_rad = args.collision_rad
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
            with Pool() as pool:
                all_meters_values, all_meters_agent_traj_nums = zip(*pool.map(partial(eval_one_seq,
                                                                                      gt_raw=gt_raw,
                                                                                      collision_rad=collision_rad,
                                                                                      return_agent_traj_nums=True),
                                                                                  data_filelist))
        else:
            all_meters_values, all_meters_agent_traj_nums = [],[]
            for data_file in data_filelist:  # each example e.g., seq_0001 - frame_000009
                meters, agent_traj_nums = eval_one_seq(data_file, gt_raw, collision_rad, return_agent_traj_nums=True)
                all_meters_values.append(meters)
                all_meters_agent_traj_nums.append(agent_traj_nums)

        for meter, values, agent_traj_num in zip(stats_meter.values(), zip(*all_meters_values), zip(*all_meters_agent_traj_nums)):
            meter.update((np.sum(np.array(values) * np.array(agent_traj_num)) / np.sum(agent_traj_num)).item(),
                         n=np.sum(agent_traj_num).item())

    # print_log('-' * 30 + ' STATS ' + '-' * 30, log_file)
    # for name, meter in stats_meter.items():
    #     print_log(f'{meter.count} {name}: {meter.avg:.4f}', log_file)
    # print('-' * 67)
    print(args.epoch)
    for name, meter in stats_meter.items():
        if 'gt' not in name:
            print(f"{meter.avg:.4f}")
    log_file.close()

    write_metrics_to_csv(stats_meter, csv_file, args.label, results_dir, args.epoch, args.data)