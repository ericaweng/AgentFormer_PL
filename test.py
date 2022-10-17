import numpy as np
import argparse
import os
import sys
import torch
import shutil
import itertools
import subprocess
import multiprocessing
# multiprocessing.set_start_method('spawn')
from functools import partial

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from eval import check_collision_per_sample_no_gt


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


def run_model_w_col_rej(data, traj_scale, sample_k, collisions_ok):
    """run model with collision rejection"""
    seq_name, frame = data['seq'], data['frame']
    frame = int(frame)
    sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))
    sys.stdout.flush()

    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * traj_scale
    non_colliding_samples = torch.empty(0).to(device)
    num_tries = 0
    MAX_NUM_TRIES = 10
    while non_colliding_samples.shape[0] < sample_k:
        with torch.no_grad():
            num_samples = 40 if num_tries == 0 and not collisions_ok else 20
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, num_samples)
            num_tries += 1
        recon_motion_3D, sample_motion_3D = recon_motion_3D * traj_scale, sample_motion_3D * traj_scale

        # compute number of colliding samples
        if collisions_ok:
            break

        pred_arr = sample_motion_3D.cpu().numpy()
        if pred_arr.shape[0] > 1:
            # with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            #     mask = pool.starmap(check_collision_per_sample_no_gt, enumerate(pred_arr))
            mask = itertools.starmap(check_collision_per_sample_no_gt, enumerate(pred_arr))
            maskk = np.where(~np.any(np.array(list(zip(*mask))[1]).astype(np.bool), axis=-1))[0]
            if maskk.shape[0] == 0:
                if num_tries > MAX_NUM_TRIES:
                    print_log(f"collected {40+ (num_tries-1)*20} and only "
                              f"{non_colliding_samples.shape[0]} non-colliding samples \n"
                              f"num_peds: {pred_arr.shape[0]}", log)
                    return None
                continue
            non_collide_idx = torch.LongTensor(maskk)
            assert torch.max(non_collide_idx) < sample_motion_3D.shape[0]
            assert 0 <= torch.max(non_collide_idx)
            sample_motion_3D = torch.index_select(sample_motion_3D, 0, non_collide_idx.to(device))
        non_colliding_samples = torch.cat([non_colliding_samples, sample_motion_3D])
        sample_motion_3D = non_colliding_samples[:sample_k]  # select only 20 non-colliding samples

    # num_samples_needed.append(20 + 10 * (num_tries - 1))

    return gt_motion_3D, sample_motion_3D[0], sample_motion_3D


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
    print(f"saved to {save_dir}")
    return num_pred


def test_one_sequence(data, save_dir, indices, num_future_frames, traj_scale, sample_k):
    seq_name, frame = data['seq'], data['frame']
    frame = int(frame)
    sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))
    sys.stdout.flush()

    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * traj_scale
    with torch.no_grad():
        recon_motion_3D, sample_motion_3D = get_model_prediction(data, sample_k)
    recon_motion_3D, sample_motion_3D = recon_motion_3D * traj_scale, sample_motion_3D * traj_scale

    """save samples"""
    recon_dir = os.path.join(save_dir, 'recon')
    mkdir_if_missing(recon_dir)
    sample_dir = os.path.join(save_dir, 'samples')
    mkdir_if_missing(sample_dir)
    gt_dir = os.path.join(save_dir, 'gt')
    mkdir_if_missing(gt_dir)
    for i in range(sample_motion_3D.shape[0]):
        save_prediction(sample_motion_3D[i].cpu().numpy(), data, f'/sample_{i:03d}', sample_dir, indices, num_future_frames)
    save_prediction(recon_motion_3D.cpu().numpy(), data, '', recon_dir, indices, num_future_frames)  # save recon
    num_pred = save_prediction(gt_motion_3D.cpu().numpy(), data, '', gt_dir, indices, num_future_frames)  # save gt
    # print(f"saved to {save_dir}")
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
            args_list.append((data, save_dir, indices, cfg.future_frames, cfg.traj_scale, cfg.sample_k))
        else:
            args_list.append((data, cfg.traj_scale, cfg.sample_k, collisions_ok))

    total_num_preds = []
    # both run and save in subprocess
    if args.multiprocess:
        with multiprocessing.Pool() as pool:
            total_num_preds = pool.starmap(test_one_sequence, args_list)
    elif not args.multiprocess2:
        for data in args_list:
            total_num_preds.append(test_one_sequence(*data))
    else:  # args.multiprocess2 and not args.multiprocess:
        # run first sequentially, then mp save
        args_list_w_results = []
        for args_l in args_list:
            # if collisions_ok:
            #     res = run_model(*args_l)
            # else:
            res = run_model_w_col_rej(*args_l)
            if res is None:
                continue
            res = list(map(lambda x: x.cpu().numpy(), res))
            args_list_w_results.append((args_l[0], save_dir, indices, num_future_frames, *res))
        # cpus_per_gpu = int(multiprocessing.cpu_count() / torch.cuda.device_count()) - 1
        # print("cpus_per_gpu:", cpus_per_gpu)
        # with multiprocessing.Pool(cpus_per_gpu) as pool:
        #     total_num_preds = pool.starmap(save_results, args_list_w_results)
        # print("saved preds")
        total_num_preds = []
        for args_tup in args_list_w_results:
            total_num_preds.append(save_results(*args_tup))

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
    # __spec__ = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--multiprocess', '-mp', action='store_true', default=False)
    parser.add_argument('--multiprocess2', '-mp2', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--eval_gt', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
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
                result_files = sorted(glob.glob(os.path.join(eval_dir, '*/frame*'), recursive=True))
                if len(result_files) > 0:
                    start_frame = int(result_files[-1][-6:]) + 1
                    print("start testing at frame:", start_frame)
            if not args.cached:
                test_model(generator, save_dir, cfg, start_frame)

            # import ipdb; ipdb.set_trace()
            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            eval_gt = " --eval_gt" if args.eval_gt else ""
            mp = " -mp" if args.multiprocess2 or args.multiprocess else ""
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --label {cfg.id} --epoch {epoch} --sample_num {cfg.sample_k} --data {split} --log {log_file}{eval_gt}{mp}"
            subprocess.run(cmd.split(' '))

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)


