import numpy as np
import argparse
import os
import sys
import torch
import shutil
import itertools
import subprocess
import multiprocessing
from functools import partial

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing, AverageMeter
from eval import check_collision_per_sample_no_gt, get_collisions_mat_old, eval_one_seq, stats_func, write_metrics_to_csv


def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D


def save_prediction(pred, data, suffix, save_dir, indices, num_future_frames, scale=1.0):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(num_future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[indices[-2:]] = pred[i, j] / scale  # [13, 15] or [2, 3] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num


def run_model_w_col_rej(data, traj_scale, sample_k, collisions_ok, collision_rad):
    """run model with collision rejection"""
    seq_name, frame = data['seq'], data['frame']
    frame = int(frame)
    sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))
    sys.stdout.flush()

    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * traj_scale
    samples_to_return = torch.empty(0).to(device)
    num_tries = 0
    num_zeros = 0
    MAX_NUM_ZEROS = 3
    MAX_NUM_TRIES = 10
    while samples_to_return.shape[0] < sample_k:
        with torch.no_grad():
            num_samples = 40 if num_tries == 0 and not collisions_ok else 20
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, num_samples)
            num_tries += 1
        recon_motion_3D, sample_motion_3D = recon_motion_3D * traj_scale, sample_motion_3D * traj_scale

        # compute number of colliding samples
        pred_arr = sample_motion_3D.cpu().numpy()
        if collisions_ok or pred_arr.shape[0] == 1:
            break

        # with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        #     mask = pool.starmap(check_collision_per_sample_no_gt, enumerate(pred_arr))
        args_list = list(zip(np.arange(len(pred_arr)), pred_arr, [collision_rad for _ in range(len(pred_arr))]))
        mask = itertools.starmap(get_collisions_mat_old, args_list)
        # mask = itertools.starmap(check_collision_per_sample_no_gt, args_list)
        maskk = np.where(~np.any(np.array(list(zip(*mask))[1]).astype(np.bool), axis=-1))[0]  # get indices of samples that have 0 collisions
        if maskk.shape[0] == 0:
            num_zeros += 1
            if num_zeros > MAX_NUM_ZEROS or num_tries > MAX_NUM_TRIES:
                print_log(f"frame {data['frame']} with {len(data['pre_motion_3D'])} peds: "
                          f"collected {40+ (num_tries-1)*20} samples, only "
                          f"{samples_to_return.shape[0]} non-colliding. \n", log)
                sample_motion_3D = samples_to_return[:sample_k]  # select only 20 non-colliding samples
                break
            continue
        # append new non-colliding samples to list
        non_collide_idx = torch.LongTensor(maskk)
        assert torch.max(non_collide_idx) < sample_motion_3D.shape[0]
        assert 0 <= torch.max(non_collide_idx)
        sample_motion_3D = torch.index_select(sample_motion_3D, 0, non_collide_idx.to(device))  # select only those in current sample who don't collide
        samples_to_return = torch.cat([samples_to_return, sample_motion_3D])
        sample_motion_3D = samples_to_return[:sample_k]  # select only 20 non-colliding samples

    if sample_motion_3D.shape[0] == 0:
        return
    recon_motion_3D = sample_motion_3D[0].cpu().numpy()
    sample_motion_3D = sample_motion_3D.cpu().numpy()
    gt_motion_3D = gt_motion_3D.cpu().numpy()
    return gt_motion_3D, recon_motion_3D, sample_motion_3D


def run_model(data, traj_scale, sample_k):
    seq_name, frame = data['seq'], data['frame']
    frame = int(frame)
    sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))
    sys.stdout.flush()

    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * traj_scale
    with torch.no_grad():
        recon_motion_3D, sample_motion_3D = get_model_prediction(data, sample_k)
    recon_motion_3D, sample_motion_3D = recon_motion_3D * traj_scale, sample_motion_3D * traj_scale
    return gt_motion_3D, recon_motion_3D, sample_motion_3D


def save_results(data, save_dir, indices, num_future_frames, gt_motion_3D, recon_motion_3D, sample_motion_3D):
    """save samples"""
    recon_dir = os.path.join(save_dir, 'recon')
    mkdir_if_missing(recon_dir)
    sample_dir = os.path.join(save_dir, 'samples')
    mkdir_if_missing(sample_dir)
    gt_dir = os.path.join(save_dir, 'gt')
    mkdir_if_missing(gt_dir)
    for i in range(sample_motion_3D.shape[0]):
        save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir, indices, num_future_frames)
    save_prediction(recon_motion_3D, data, '', recon_dir, indices, num_future_frames)  # save recon
    num_pred = save_prediction(gt_motion_3D, data, '', gt_dir, indices, num_future_frames)  # save gt
    print(f"saved frame {data['frame']} to {save_dir}")
    return num_pred


def test_one_dont_save(data, save_dir, indices, future_frames, traj_scale, sample_k, collisions_ok, collision_rad):
    tup = run_model_w_col_rej(data, traj_scale, sample_k, collisions_ok, collision_rad)
    if tup is None:
        return 0
    return tup


def test_one_sequence(data, save_dir, indices, num_future_frames, traj_scale, sample_k, collisions_ok, collision_rad):
    tup = run_model_w_col_rej(data, traj_scale, sample_k, collisions_ok, collision_rad)
    if tup is None:
        return 0
    gt_motion_3D, recon_motion_3D, sample_motion_3D = tup
    num_pred = save_results(data, save_dir, indices, num_future_frames, gt_motion_3D, recon_motion_3D, sample_motion_3D)
    return num_pred


def test_model(generator, save_dir, cfg, start_frame):
    if 'sdd' in cfg.dataset:
        indices = [0, 1, 2, 3]
    else:
        # frame, ID, x, z (remove y which is the height)
        indices = [0, 1, 13, 15]
    num_future_frames = cfg.future_frames

    num_samples_needed = []  # number of samples needed before we reached 20
    args_list = []
    collisions_ok = cfg.get('collisions_ok', True)
    collision_rad = cfg.get('collision_rad', 0.1)
    # collision_rad = cfg.get('collision_rad')
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']
        if frame < start_frame:
            continue
        if 'pred_mask' in data and np.all(data['pred_mask'] == -1):
            continue
        if not args.multiprocess2:
            args_list.append((data, save_dir, indices, cfg.future_frames, cfg.traj_scale, cfg.sample_k, collisions_ok, collision_rad))
        else:
            args_list.append((data, cfg.traj_scale, cfg.sample_k, collisions_ok, collision_rad))

    total_num_preds = []
    # both run and save in subprocess
    if args.dont_save:  # multiprocess and dont save
        if not args.multiprocess and not args.multiprocess2:
            for data in args_list:
                total_num_preds.append(test_one_dont_save(*data))
        else:
            stats_meter = {x: AverageMeter() for x in stats_func.keys()}
            # with multiprocessing.Pool() as pool:
            #     test_results = pool.starmap(test_one_dont_save, args_list)
            test_results = []
            for arg_list in args_list:
                test_results.append(test_one_dont_save(*arg_list))
            args_list = [(gt_motion_3D, sample_motion_3D, stats_func, collision_rad) for gt_motion_3D, _, sample_motion_3D in test_results]
            with multiprocessing.Pool() as pool:
                all_meters_values, all_meters_agent_traj_nums = zip(*pool.starmap(eval_one_seq, args_list))
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

            # write_metrics_to_csv(stats_meter, csv_file, args.label, results_dir, args.epoch, args.data)
            exit(0)

    elif not args.multiprocess2:  # sequentially
        for data in args_list:
            total_num_preds.append(test_one_sequence(*data))

    else:  # multiprocess2
        # run first sequentially, then mp save
        args_list_w_results = []
        for args_l in args_list:
            res = run_model_w_col_rej(*args_l)
            if res is None:
                continue
            args_list_w_results.append((args_l[0], save_dir, indices, num_future_frames, *res))
        with multiprocessing.Pool() as pool:
            total_num_preds = pool.starmap(save_results, args_list_w_results)

    total_num_pred = sum(total_num_preds)

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)
    print_log(f'\n\n avg_num_pred: {total_num_pred / (len(total_num_preds) + 1e-7)}', log)
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {
            'train': 32186,
            'val': 8560,
            'test': 9041
        }
        assert total_num_pred == scene_num[generator.split]

    # print_log(f"avg num_samples: {np.mean(num_samples_needed)}", log)
    # print_log(f"std num_samples: {np.std(num_samples_needed)}", log)


if __name__ == '__main__':
    __spec__ = None
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default='last')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--multiprocess', '-mp', action='store_true', default=False)
    parser.add_argument('--multiprocess2', '-mp2', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--eval_gt', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--dont_save', action='store_true', default=False)
    parser.add_argument('--all_epochs', action='store_true', default=False)
    parser.add_argument('--weight', type=float, default=None)
    parser.add_argument('--sigma_d', type=float, default=None)
    args = parser.parse_args()

    if args.eval_gt:
        args.cached = True
    """ setup """
    cfg = Config(args.cfg, additional_cfg_vars={'weight': args.weight, 'sigma_d': args.sigma_d})
    args.collisions_ok = cfg.get('collisions_ok', True)
    print("collisions_ok:", args.collisions_ok)

    if args.all_epochs:
        epochs = range(cfg.model_save_freq, cfg.num_epochs + 1, cfg.model_save_freq)
    elif args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    elif args.epochs == 'last':
        epochs = ['last']
    else:
        epochs = [int(x) for x in args.epochs.split(',')]
    print("epochs:", epochs)

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch == 'last':
                cp_path = cfg.model_path_last
            elif epoch > 0:
                cp_path = cfg.model_path % epoch
            else:
                raise NotImplementedError
            print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
            model_cp = torch.load(cp_path, map_location='cpu')
            epoch = model_cp['epoch']
            model.load_state_dict(model_cp['model_dict'], strict=False)
        else:
            if epoch == 'last':
                import glob
                epoch = int(str(sorted(glob.glob(os.path.join(cfg.result_dir, 'epoch_*')))[-1][-4:]))

        print_log(f"doing epoch: {epoch}", log)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'

            start_frame = 0
            if args.resume:  # resume in case previous test run got interrupted
                import glob
                result_files = sorted(glob.glob(os.path.join(eval_dir, '*/frame*/**.txt'), recursive=True))
                if len(result_files) > 0:
                    start_frame = int(result_files[-1].split('frame_')[-1].split('/')[0]) + 1
                    print("start testing at frame:", start_frame)

            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            if not args.cached:
                test_model(generator, save_dir, cfg, start_frame)

            # import ipdb; ipdb.set_trace()
            eval_gt = " --eval_gt" if args.eval_gt else ""
            mp = " -mp" if args.multiprocess2 or args.multiprocess else ""
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --label {cfg.id} --epoch {epoch} --sample_num {cfg.sample_k} --data {split} --log {log_file}{eval_gt}{mp}"
            subprocess.run(cmd.split(' '))

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)


