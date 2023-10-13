"""find outstanding examples between AF and Our method and plot image"""

import os
import argparse
import multiprocessing
from pathlib import Path
import numpy as np

from scripts.evaluate_all import peds_pandas_way
from viz_utils2 import plot_anim_grid#, get_metrics_str, get_max_bounds
from metrics import compute_CR


# OURS = 'af_mg1_jr1_w10'
# OURS = 'af_mg-1,5_jr-1,7.5'
# OURS = 'af_mg-1,5_jr-1,5'
OURS = 'vv_ml_a-0.5,0.5_b-0.4,0.4,0.2'

METHOD_DISP_NAMES = {
        'sgan': 'S-GAN',
        'trajectron': 'Trajectron++',
        'pecnet': 'PECNet',
        'ynet': 'Y-Net',
        'memonet': 'MemoNet',
        'vv': 'View Vertically',
        'vv_ml_a-0.5,0.5_b-0.4,0.4,0.2': 'Joint View Vertically (Ours)',
        'agentformer': 'AgentFormer',
        'af_mg1_jr1_w10': 'Joint AgentFormer (Ours) (old)',
        'af_mg-1,5_jr-1,7.5': 'Joint AgentFormer (Ours)',
        'af_mg-1,5_jr-1,5': 'Joint AgentFormer (Ours)',
}

def get_trajs(frame_path, method):
    pred_gt_traj = obs_traj = None
    samples = []
    for filename in frame_path.glob('*.txt'):
        if 'gt' in str(filename.name):
            pred_gt_traj = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            pred_gt_traj = peds_pandas_way(pred_gt_traj, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
        elif 'obs' in str(filename.name):
            obs_traj_raw = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            obs_traj_ = peds_pandas_way(obs_traj_raw, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])  # todo
            if method == 'agentformer' or 'af' in method or method == 'memonet':
                obs_traj = obs_traj_[:,::-1]
            else:
                obs_traj = obs_traj_
        elif 'sample' in str(filename.name):
            sample = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            sample = peds_pandas_way(sample, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
            samples.append(sample)
        else:
            continue
            raise RuntimeError(f"Unknown file {filename}")
    assert pred_gt_traj is not None, f"gt and obs should be loaded from {frame_path}"
    assert len(samples) == 20, f"20 samples should be loaded from {frame_path}"
    if obs_traj is None or obs_traj.shape[0] != 8:
        obs_traj = None
    else:
        assert obs_traj.shape[0] == 8
    assert pred_gt_traj.shape[0] == 12
    pred_fake_traj = np.stack(samples, axis=0)  # (num_samples, frames, agents, 2)
    return pred_fake_traj, pred_gt_traj, obs_traj


def get_col_mats(pred_fake_traj, pred_gt_traj):
    _, sample_collisions, collision_mats = compute_CR(pred_fake_traj, pred_gt_traj, return_sample_vals=True, return_collision_mat=True, collision_rad=0.1)
    return collision_mats

def main(args):

    SEQUENCE_NAMES = {
        'eth': ['biwi_eth'],
        'hotel': ['biwi_hotel'],
        'zara1': ['crowds_zara01'],
        'zara2': ['crowds_zara02'],
        'univ': ['students001', 'students003'],
        'trajnet_sdd': [ 'coupa_0', 'hyang_3', 'quad_3', 'little_2', 'nexus_5', 'quad_2',
                         'gates_2', 'coupa_1', 'quad_1', 'hyang_1', 'hyang_8', 'little_1',
                         'nexus_6', 'hyang_0', 'quad_0', 'little_0', 'little_3']
    }

    # gather all frames for all methods to plot
    all_frames = []
    placeholder_method = 'agentformer'
    for dset in args.dset:
        frames_this_dset = []
        if dset not in SEQUENCE_NAMES:
            if dset in SEQUENCE_NAMES['trajnet_sdd']:
                trajs_dir = os.path.join(args.trajs_dir, placeholder_method, 'trajnet_sdd', dset)
                frames_this_dset.extend(list(Path(trajs_dir).glob('frame_*')))
        else:
            for seq in SEQUENCE_NAMES[dset]:
                if dset == 'trajnet_sdd':
                    trajs_dir = os.path.join(args.trajs_dir, placeholder_method, 'trajnet_sdd', seq)
                else:
                    trajs_dir = os.path.join(args.trajs_dir, placeholder_method, seq)
                if args.frames_to_plot is None:
                    frames_this_dset.extend(list(Path(trajs_dir).glob('frame_*')))
                else:
                    for frame in args.frames_to_plot:
                        print(frame)
                        frames_this_dset.extend(list(Path(trajs_dir).glob(f'frame_*{frame}*')))

        if args.save_num is None:
            skip = 1
        else:
            skip = max(1, int(len(frames_this_dset) / args.save_num))
        all_frames.extend(frames_this_dset[::skip])

    print(f"Saving {len(all_frames)} frames per method across all dsets except frames with only 1 ped")

    skip = 1
    for frame_path_ in all_frames[::skip]:
        seq = frame_path_.parent.name
        trajs_list_for_bounds_calculation = []
        for method_i, method in enumerate(args.method):
            frame = int(frame_path_.name.split('_')[-1])
            frame_path = Path(str(frame_path_).replace(placeholder_method, method))
            res = get_trajs(frame_path, method)
            if method == 'trajectron':
                diff = obs_traj - res[-1]
                assert np.allclose(diff, diff[0,0], atol=1e-3), f"diff: {diff[0,0]}"
                res = tuple(r + diff[0,0] for r in res)
            trajs_list_for_bounds_calculation.extend([r for r in res if r is not None])
            if method == 'ynet':
                pred_fake_traj, pred_gt_traj, _ = res
            else:
                pred_fake_traj, pred_gt_traj, obs_traj = res
            NUM_SAMPLES, _, n_ped, _ = pred_fake_traj.shape

            collision_mats = get_col_mats(pred_fake_traj.transpose(2,0,1,3), pred_gt_traj.swapaxes(0,1))

            NUM_SAMPLES = 10
            args_list = []
            method_to_is={'vv':(1,4,7,8), 'agentformer':(0,1,2,3), 'af_mg-1,5_jr-1,5': (2,5,7,8), 'vv_ml_a-0.5,0.5_b-0.4,0.4,0.2':(2,3,4,5)}
            selected_samples = list(method_to_is[method])#range(NUM_SAMPLES))
            subplot_i = 0
            to_plot_pred_fake_traj = pred_fake_traj[selected_samples]
            for sample_i, sample in enumerate(to_plot_pred_fake_traj):
                args_dict = {
                             'obs_traj': obs_traj,
                             'gt_traj': pred_gt_traj,
                             'pred_traj': sample,
                             'collision_mats': collision_mats[selected_samples[sample_i]]
                }
                args_list.append(args_dict)
                subplot_i += 1

            anim_save_fn = os.path.join(args.save_dir, f'{method}_{seq}_frame_{frame:06d}.mp4')

            plot_anim_grid(anim_save_fn, "", (4,1), *args_list)

    print(f"done plotting plots")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--trajs_dir', type=str, default='../trajectory_reward/results/trajectories')
    ap.add_argument('--frames_to_plot', '-f', nargs='+', type=int, default=None)
    ap.add_argument('--method', '-m', type=str, nargs='+', default=['agentformer', 'af_mg-1,5_jr-1,5', 'vv', 'vv_ml_a-0.5,0.5_b-0.4,0.4,0.2'])  # ['vv', 'vv_ml_a-0.5,0.5_b-0.4,0.4,0.2'])
    ap.add_argument('--dset', '-d', type=str, nargs='+', default=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'trajnet_sdd'])
    ap.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    ap.add_argument('--save_num', '-s', type=int, default=None, help='number of frames to save per dset')
    ap.add_argument('--metrics_path', '-mp', default='../trajectory_reward/results/evaluations_rad-0.1_samples-20')
    ap.add_argument('--no_mp', dest='mp', action='store_false')
    ap.add_argument('--save_dir', '-sd', type=str, default='viz_outstanding')
    ap.add_argument('--refine', '-r', action='store_true')
    ap.add_argument('--verbose', '-v', action='store_true')
    ap.add_argument('--png', action='store_true')
    args = ap.parse_args()

    main(args)