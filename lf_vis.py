"""find outstanding examples between AF and Our method and plot image"""

import os
import argparse
import multiprocessing
from pathlib import Path
from itertools import starmap
import numpy as np

from utils.utils import mkdir_if_missing
from scripts.evaluate_all import peds_pandas_way
from viz_utils import plot_anim_grid, get_metrics_str, get_max_bounds
from viz_utils_img import plot_scene, plot_img_grid, plot_img_grid_old
from metrics import compute_ADE_marginal, compute_FDE_marginal, compute_ADE_joint, \
    compute_FDE_joint, compute_CR


OURS = 'af_mg1_jr1_w10'
OURS = 'af_mg-1,5_jr-1,7.5'
OURS = 'af_mg-1,5_jr-1,5'

METHOD_DISP_NAMES = {
        # 'sgan': 'S-GAN []',
        # 'trajectron': 'Trajectron++ []',
        # 'pecnet': 'PECNet []',
        # 'ynet': 'Y-Net []',
        # 'memonet': 'MemoNet []',
        # 'vv': 'View Vertically [52]',
        # 'agentformer': 'AgentFormer [60]',
        # 'af_mg1_jr1_w10': 'Joint AgentFormer (Ours) (old)',
        # 'af_mg-1,5_jr-1,7.5': 'Joint AgentFormer (Ours)',
        'sgan': 'S-GAN',
        'trajectron': 'Trajectron++',
        'pecnet': 'PECNet',
        'ynet': 'Y-Net',
        'memonet': 'MemoNet',
        'vv': 'View Vertically',
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
        # load obs from other method folder
        # import ipdb;
        # ipdb.set_trace()
        # obs_path = os.path.join(str(frame_path).replace(method, 'agentformer'), 'obs.txt')
        # obs_traj_raw = np.loadtxt(obs_path, delimiter=' ', dtype='float32')  # (frames x agents) x 4
        # obs_traj_ = peds_pandas_way(obs_traj_raw, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
        # print("method:", method)
        # print(f"obs_traj_.shape: {obs_traj_.shape}")
        # obs_traj = obs_traj_[:,::-1]
        # print(f"obs_traj.shape: {obs_traj.shape}")
        # import ipdb; ipdb.set_trace()
    else:
        assert obs_traj.shape[0] == 8
    assert pred_gt_traj.shape[0] == 12
    pred_fake_traj = np.stack(samples, axis=0)  # (num_samples, frames, agents, 2)
    return pred_fake_traj, pred_gt_traj, obs_traj


def get_metrics_dict(pred_fake_traj, pred_gt_traj):
    _, sample_collisions, collision_mats = compute_CR(pred_fake_traj, pred_gt_traj, return_sample_vals=True, return_collision_mat=True, collision_rad=0.1)
    ade, ade_ped_val, ade_argmins = compute_ADE_marginal(pred_fake_traj, pred_gt_traj, return_ped_vals=True, return_argmin=True)
    fde, fde_ped_val, fde_argmins = compute_FDE_marginal(pred_fake_traj, pred_gt_traj, return_ped_vals=True, return_argmin=True)
    sade, sade_samples, sade_argmin = compute_ADE_joint(pred_fake_traj, pred_gt_traj, return_argmin=True, return_sample_vals=True)
    sfde, sfde_samples, sfde_argmin = compute_FDE_joint(pred_fake_traj, pred_gt_traj, return_argmin=True, return_sample_vals=True)
    metrics_dict = {'collision_mats': collision_mats,
                    'ADE': ade,
                    'FDE': fde,
                    'SADE': sade,
                    'SFDE': sfde,
                    'ade_ped_val': ade_ped_val,
                    'fde_ped_val': fde_ped_val,
                    'ade_argmins': ade_argmins,
                    'fde_argmins': fde_argmins,
                    'sade_argmin': sade_argmin,
                    'sfde_argmin': sfde_argmin, }
    samples_dict = {'SADE': sade_samples,
                    'SFDE': sfde_samples,
                    'CR': sample_collisions,}
    return samples_dict, metrics_dict

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

        all_frames.extend(frames_this_dset)

    print(f"Saving {len(all_frames)} frames per method across all dsets except frames with only 1 ped")

    # gather list of args for plotting
    seq_to_plot_args = []
    for frame_path_ in all_frames:
        seq = frame_path_.parent.name
        other_mSFDE, other_mSADE, other_mFDE, other_mADE = [], [], [], []
        non_ours_args_list = []
        non_ours_args_list_nl = []
        trajs_list_for_bounds_calculation = []
        at_least_one_method_has_cols = False
        for method_i, method in enumerate(args.method):
            frame = int(frame_path_.name.split('_')[-1])
            frame_path = Path(str(frame_path_).replace(placeholder_method, method))
            res = get_trajs(frame_path, method)
            if method == 'trajectron':
                diff = obs_traj - res[-1]
                assert np.allclose(diff, diff[0,0], atol=1e-3), f"diff: {diff[0,0]}"
                res = tuple(r + diff[0,0] for r in res)
            # borrow pecnet's obs_traj and gt_traj (mostly need just the obs_traj...
            # but gt_traj just in case? bc the gt_trajs differ slightly)
            trajs_list_for_bounds_calculation.extend([r for r in res if r is not None])
            if method == 'ynet':
                pred_fake_traj, pred_gt_traj, _ = res
            else:
                pred_fake_traj, pred_gt_traj, obs_traj = res
            n_samples, _, n_ped, _ = pred_fake_traj.shape

            sample_metrics, all_metrics = get_metrics_dict(pred_fake_traj.transpose(2,0,1,3), pred_gt_traj.swapaxes(0,1))
            collision_mats = all_metrics['collision_mats']
            mADE, ade_argmins = all_metrics['ADE'], all_metrics['ade_argmins']
            mFDE, fde_argmins = all_metrics['FDE'], all_metrics['fde_argmins']
            mSADE, sade_argmin = all_metrics['SADE'], all_metrics['sade_argmin']
            mSFDE, sfde_argmin = all_metrics['SFDE'], all_metrics['sfde_argmin']
            sample_crs = sample_metrics['CR']

            # filter
            if args.refine and (n_ped <= 1 or n_ped > 5):
                break
            # our method has to have better SXDE and worse XDE than other methods
            if args.refine and method == OURS and not (
                                       np.all(other_mSADE > mSADE)
                                       and np.any(other_mADE < mADE)
                                       ):
                print(f"ours not better than other methods: "
                      f"other_mSADE ({np.array2string(np.array(other_mSADE), precision=2)}) !> mSADE ({mSADE:0.2f}) ")
                break

            NUM_SAMPLES = 20
            args_list = []
            args_list_nl = []

            if method != OURS:
                # pick out best XDE samples from other methods
                selected_samples = np.argsort(sample_crs)[-NUM_SAMPLES:]
                if method == 'memonet':
                    selected_samples = np.arange(NUM_SAMPLES)
                    # selected_samples = np.array([0, 2, 9, 10, 18])[-NUM_SAMPLES:]
                elif method == 'pecnet':
                    selected_samples = np.arange(NUM_SAMPLES)
                    # selected_samples = np.array([0, 3, 7, 9, 15])[-NUM_SAMPLES:]
                elif method == 'trajectron':
                    selected_samples = np.arange(NUM_SAMPLES)
                    # selected_samples = np.array([0, 2, 6, 11, 19])[-NUM_SAMPLES:]
                elif method == 'ynet':
                    selected_samples = np.arange(NUM_SAMPLES)
                    # selected_samples = np.array([0, 1, 5, 7, 12])[-NUM_SAMPLES:]
                elif method == 'agentformer':
                    selected_samples = np.array([2, 3, 4, 18, 19])[-NUM_SAMPLES:]
                elif method == 'sgan':
                    selected_samples = np.arange(NUM_SAMPLES)
            else:  # pick out best SXDE samples from OURS
                if args.refine and not at_least_one_method_has_cols:
                    print("no other method has cols, so we can't compare ours to them")
                    break
                selected_samples = np.argsort(sample_crs)[:NUM_SAMPLES]
                pass
            np.random.shuffle(selected_samples)

            subplot_i = 0
            to_plot_pred_fake_traj = pred_fake_traj[selected_samples]
            MIDDLE_SUBPLOT_I = 0#len(selected_samples) // 2
            for sample_i, sample in enumerate(to_plot_pred_fake_traj):
                args_dict = {'plot_title': f"{METHOD_DISP_NAMES[method]}" if subplot_i == MIDDLE_SUBPLOT_I else "",#f'{mADE:0.2f}',#f"" if sample_i == 2 and method == 'agentformer' else f"{method} {sample_i}",
                             'obs_traj': obs_traj,
                             'gt_traj': pred_gt_traj,
                             'y_label': f'Sample {subplot_i + 1}' if method_i == 0 else None,
                             'pred_traj': sample,
                             'plot_ped_texts': False,
                             'plot_velocity_arrows': True,
                             # 'collision_mats': collision_mats[selected_samples[sample_i]],
                             }
                args_list.append(args_dict)
                subplot_i += 1

            if len(args_list) == 0:  # if not plots for this frame from af or ours
                continue

            if method != OURS:
                other_mADE.append(mADE)
                other_mFDE.append(mFDE)
                other_mSFDE.append(mSFDE)
                other_mSADE.append(mSADE)
                non_ours_args_list.append(args_list)
                non_ours_args_list_nl.append(args_list_nl)
                continue

            else:  # method is OURS: plot
                png_or_pdf = 'pdf'
                anim_save_fn = os.path.join(args.save_dir, f'collisions_{seq}_frame_{frame:06d}.{png_or_pdf}')
                bounds = get_max_bounds(trajs_list_for_bounds_calculation, padding=-0.2)
                x_offset = 0.0
                y_offset_high = 2.8
                y_offset_low = 2.8
                bounds = [bounds[0] + x_offset, bounds[1] + y_offset_low, bounds[2] - x_offset, bounds[3] - y_offset_high]
                plot_args = [anim_save_fn, "", bounds, (len(args.method), NUM_SAMPLES), *non_ours_args_list, args_list]
                seq_to_plot_args.append(plot_args)
            if args.plot_online and len(seq_to_plot_args) > 0:
                OURS_plot_args_list = seq_to_plot_args.pop(0)
                mkdir_if_missing(anim_save_fn)
                plot_img_grid(*OURS_plot_args_list)

    # print(f"done plotting {len(sps)} plots")
    for args_list in seq_to_plot_args:
        plot_img_grid(*args_list)
    print(f"done plotting plots")

    # plot in parallel
    if not args.plot_online:
        print(f"plotting {len(seq_to_plot_args)} plots")
        if args.mp:
            with multiprocessing.Pool(args.num_workers) as pool:
                pool.starmap(plot_anim_grid, seq_to_plot_args)
        else:
            list(starmap(plot_anim_grid, seq_to_plot_args))
        print(f"done plotting {len(seq_to_plot_args)} plots")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--trajs_dir', type=str, default='../trajectory_reward/results/trajectories')
    ap.add_argument('--frames_to_plot', '-f', nargs='+', type=int, default=None)
    ap.add_argument('--method', '-m', type=str, nargs='+', default=['agentformer', 'af_mg1_jr1_w10'])
    ap.add_argument('--dset', '-d', type=str, nargs='+', default=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'trajnet_sdd'])
    ap.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    ap.add_argument('--save_num', '-s', type=int, default=None, help='number of frames to save per dset')
    ap.add_argument('--metrics_path', '-mp', default='../trajectory_reward/results/evaluations_rad-0.1_samples-20')
    ap.add_argument('--no_mp', dest='mp', action='store_false')
    # ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--save_dir', '-sd', type=str, default='viz2')
    ap.add_argument('--dont_plot_online', '-dpo', dest='plot_online', action='store_false')
    ap.add_argument('--refine', '-r', action='store_true')
    ap.add_argument('--verbose', '-v', action='store_true')
    ap.add_argument('--png', action='store_true')
    args = ap.parse_args()

    main(args)