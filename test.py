import numpy as np
import argparse
import os
import sys
import shutil
import subprocess
import multiprocessing
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

def save_prediction(pred, data, suffix, save_dir, scale=1.0):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']
    if 'sdd' in cfg.dataset:
        indices = [0, 1, 2, 3]
    else:
        # frame, ID, x, z (remove y which is the height)
        indices = [0, 1, 13, 15]

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            
            if 'sdd' in cfg.dataset:
                data[[2, 3]] = pred[i, j].cpu().numpy(
                ) / scale  # [13, 15] corresponds to 2D pos
            else:
                data[[
                    13, 15
                ]] = pred[i, j].cpu().numpy()  # [13, 15] corresponds to 2D pos
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

def test_model(generator, save_dir, cfg):
    total_num_pred = 0
    failures = 0
    num_samples_needed = []  # number of samples needed before we reached 20
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        if 'pred_mask' in data and np.all(data['pred_mask'] == -1):
            continue
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()

        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        not_colliding_samples = torch.empty(0).to(device)
        num_tries = 0
        while not_colliding_samples.shape[0] < cfg.sample_k:
            with torch.no_grad():
                num_samples = 40 if num_tries == 0 else 20
                recon_motion_3D, sample_motion_3D = get_model_prediction(data, num_samples)
                num_tries += 1
            recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale
            # comput number of colliding samples
            if args.collisions_ok:
                break

            pred_arr = sample_motion_3D.cpu().numpy()
            if pred_arr.shape[0] > 1:
                # multiprocessing.set_start_method('spawn')
                with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1, ) as pool:
                    mask = pool.starmap(check_collision_per_sample_no_gt, enumerate(pred_arr))
                # mask = pool.starmap(partial(check_collision_per_sample_no_gt), enumerate(pred_arr))
                maskk = np.where(~np.any(np.array(list(zip(*mask))[1]).astype(np.bool), axis=-1))[0]
                if maskk.shape[0] == 0:
                    MAX_NUM_TRIES = 10
                    if num_tries > MAX_NUM_TRIES:
                        print_log(f"num_tries greater than {MAX_NUM_TRIES}", log)
                        failures += 1
                        break
                    continue
                non_collide_idx = torch.LongTensor(maskk)
                assert torch.max(non_collide_idx) < sample_motion_3D.shape[0]
                assert 0 <= torch.max(non_collide_idx)
                sample_motion_3D = torch.index_select(sample_motion_3D, 0, non_collide_idx.to(device))
            not_colliding_samples = torch.cat([not_colliding_samples, sample_motion_3D])
            sample_motion_3D = not_colliding_samples[:cfg.sample_k]  # select only 20 non-colliding samples

        num_samples_needed.append(20 + 10 * (num_tries - 1))

        """save samples"""
        recon_dir = os.path.join(save_dir, 'recon'); mkdir_if_missing(recon_dir)
        sample_dir = os.path.join(save_dir, 'samples'); mkdir_if_missing(sample_dir)
        gt_dir = os.path.join(save_dir, 'gt'); mkdir_if_missing(gt_dir)
        for i in range(sample_motion_3D.shape[0]):
            save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
        save_prediction(recon_motion_3D, data, '', recon_dir)        # save recon
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir)              # save gt
        total_num_pred += num_pred

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {
            'train': 32186,
            'val': 8560,
            'test': 9041
        }
        assert total_num_pred == scene_num[generator.split]

    print_log(f"avg num_samples: {np.mean(num_samples_needed)}", log)
    print_log(f"std num_samples: {np.std(num_samples_needed)}", log)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--all_epochs', action='store_true', default=False)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
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
            print_log(f"doing epoch: {epoch}", log)
            model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                test_model(generator, save_dir, cfg)

            # import ipdb; ipdb.set_trace()
            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --label {args.cfg} --epoch {epoch} --sample_num {cfg.sample_k} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)


