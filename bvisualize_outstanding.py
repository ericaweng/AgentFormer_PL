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
from viz_utils_img import plot_scene, plot_img_grid
from metrics import compute_ADE_marginal, compute_FDE_marginal, compute_ADE_joint, \
    compute_FDE_joint, compute_CR

def get_trajs(frame_path, method):
    pred_gt_traj = obs_traj = None
    samples = []
    for filename in frame_path.glob('*.txt'):
        if 'gt' in str(filename.name):
            pred_gt_traj = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            pred_gt_traj = peds_pandas_way(pred_gt_traj, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
        elif 'obs' in str(filename.name):
            obs_traj = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            obs_traj = peds_pandas_way(obs_traj, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])[:,
                       ::-1]  # todo
        elif 'sample' in str(filename.name):
            sample = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            sample = peds_pandas_way(sample, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
            samples.append(sample)
        else:
            continue
            raise RuntimeError(f"Unknown file {filename}")
    assert pred_gt_traj is not None, f"gt and obs should be loaded from {frame_path}"
    assert len(samples) == 20, f"20 samples should be loaded from {frame_path}"
    if obs_traj is None:
        # load obs from other method folder
        import ipdb;
        ipdb.set_trace()
        obs_path = os.path.join(str(frame_path).replace(method, 'agentformer'), 'obs.txt')
        obs_traj = np.loadtxt(obs_path, delimiter=' ', dtype='float32')  # (frames x agents) x 4
        obs_traj = peds_pandas_way(obs_traj, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
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
                frames_this_dset.extend(list(Path(trajs_dir).glob('frame_*')))
                # all_frames.extend(list(Path(trajs_dir).glob('frame_*')))

        if args.save_num is None:
            skip = 1
        else:
            skip = max(1, int(len(frames_this_dset) / args.save_num))
        all_frames.extend(frames_this_dset[::skip])

    print(f"Saving {len(all_frames)} frames per method across all dsets except frames with only 1 ped")

    skip = 1
    # gather list of args for plotting
    seq_to_plot_args = []
    sps = []
    for frame_path in all_frames[::skip]:
        seq = frame_path.parent.name
        AF_best_SFDE, AF_best_SADE, AF_best_FDE, AF_best_ADE = None, None, None, None
        AF_SADEs = []
        AF_SFDEs = []
        AF_args_list = []
        pred_fake_traj = None
        for method in args.method:
            frame = int(frame_path.name.split('_')[-1])
            frame_path = Path(str(frame_path).replace(placeholder_method, method))
            res = get_trajs(frame_path, method)
            if pred_fake_traj is not None:
                bounds = get_max_bounds(pred_fake_traj, pred_gt_traj, obs_traj, *res)
            pred_fake_traj, pred_gt_traj, obs_traj = res
            num_samples, _, n_ped, _ = pred_fake_traj.shape
            if args.refine and n_ped < 1 or n_ped > 5:
                continue

            sample_metrics, all_metrics = get_metrics_dict(pred_fake_traj.transpose(2,0,1,3), pred_gt_traj.swapaxes(0,1))
            collision_mats = all_metrics['collision_mats']
            min_ADE, ade_argmins = all_metrics['ADE'], all_metrics['ade_argmins']
            sample_crs = sample_metrics['CR']
            sample_sades = sample_metrics['SADE']
            sample_sfdes = sample_metrics['SFDE']

            if seq in SEQUENCE_NAMES['trajnet_sdd']:
                # if dset == 'trajnet_sdd':
                bkg_img_path = os.path.join(f'datasets/trajnet_sdd/reference_img/{seq[:-2]}/video{seq[-1]}/reference.jpg')
                anim_save_fn = os.path.join(args.save_dir, 'trajnet_sdd', seq, f'frame_{frame:06d}', f'{method}.png')
                # anim_save_fn = os.path.join(args.save_dir, 'trajnet_sdd', seq, f'frame_{frame:06d}', f'{method}_{sample_i:.02d}.png')
            else:
                bkg_img_path = None
                anim_save_fn = os.path.join(args.save_dir, seq, f'frame_{frame:06d}', f'{method}.png')
                # anim_save_fn = os.path.join(args.save_dir, seq, f'frame_{frame:06d}', f'{method}_{sample_i:.02d}.png')

            args_list = []
            for sample_i in range(num_samples):
                ade_ped_vals = all_metrics['ade_ped_val'][:,sample_i]
                if len(args_list) == 10:  # only plot max 10 from each method
                    break

                if args.refine:
                    if method == 'agentformer':
                        if sample_crs[sample_i] == 0:
                            if args.verbose:
                                print(f"skipping {method} {sample_i} because of no collisions")
                            continue
                        # if np.sum(ade_ped_vals < 0.1) <= 0:
                        #     if args.verbose:
                        #         print(f"skipping {method} {sample_i} because only {np.sum(ade_ped_vals < 0.1)} ade_ped_vals < 0.1")
                        #     continue
                        # if np.sum(ade_ped_vals > 0.9) <= 0:
                        #     if args.verbose:
                        #         print(f"skipping {method} {sample_i} because only {np.sum(ade_ped_vals > 0.9)} ade_ped_vals > 0.9")
                        #     continue
                        AF_SADEs.append(sample_sades[sample_i])
                        AF_SFDEs.append(sample_sfdes[sample_i])
                    elif method != 'agentformer' and len(AF_args_list) > 0:
                        if sample_crs[sample_i] > 0:
                            if args.verbose:
                                print(f"skipping {method} {sample_i} because of collisions")
                            continue
                        if np.min(ade_ped_vals) < 0.1:
                            if args.verbose:
                                print(f"skipping {method} {sample_i} because of ade_ped_vals {np.min(ade_ped_vals)} < 0.1")
                            continue
                        if np.max(ade_ped_vals) > 0.7:
                            if args.verbose:
                                print(f"skipping {method} {sample_i} because of ade_ped_vals {np.max(ade_ped_vals)} > 0.4")
                            continue
                        # if np.all(sample_sades[sample_i] > np.array(AF_SADEs)):
                        #     if args.verbose:
                        #         print(f"skipping {method} {sample_i} because sample SADE {sample_sades[sample_i]} > all AF_SADEs {AF_SADEs}")
                        #     continue
                        print(f"PLOTTING {method} {sample_i}")
                    else:
                        continue

                stats = get_metrics_str(sample_metrics, sample_i)
                args_dict = {'plot_title': f"{method[:2]}" if sample_i == 2 and method == 'agentformer' else f"{method} {sample_i}",
                             # 'save_fn': save_fn,
                             'obs_traj': obs_traj,
                             'gt_traj': pred_gt_traj,
                             'pred_traj': pred_fake_traj[sample_i],
                             'text_fixed': stats,
                             'bkg_img_path': bkg_img_path,
                             'plot_velocity_arrows': True,
                             # 'highlight_peds': argmins[frame_i],
                             'collision_mats': collision_mats[sample_i]}
                args_list.append(args_dict)

            if len(args_list) == 0:  # if not plots for this frame from af or ours
                continue

            if method == 'agentformer':
                AF_args_list = args_list
            else:
                # seq_to_plot_args.append(AF_plot_args_list)
                plot_args = [anim_save_fn, f"Seq: {seq} frame: {frame} Method: {method}", bounds, (5, 4), *AF_args_list, *args_list]
                seq_to_plot_args.append(plot_args)
            if args.plot_online and len(seq_to_plot_args) > 0:
                OURS_plot_args_list = seq_to_plot_args.pop(0)
                sps.append(None)
                mkdir_if_missing(anim_save_fn)
                plot_img_grid(*OURS_plot_args_list)
                continue
                if len(sps) >= args.num_workers:
                    # this should work if all subprocesses take the same amount of time;
                    # otherwise we might be waiting longer than necessary
                    sps[0].join()
                    sps = sps[1:]
                seq_to_plot_online = seq_to_plot_args.pop(0)
                mkdir_if_missing(anim_save_fn)
                process = multiprocessing.Process(target=plot_img_grid, args=seq_to_plot_online)
                process.start()
                sps.append(process)
                # print(f"plotting {seq} frame {frame} method {method} online")

    print(f"done plotting {len(sps)} plots")

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
    ap.add_argument('--method', '-m', type=str, nargs='+', default=['agentformer', 'af_mg1_jr1_w10'])
    ap.add_argument('--dset', '-d', type=str, nargs='+', default=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'trajnet_sdd'])
    ap.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    ap.add_argument('--save_num', '-s', type=int, default=None, help='number of frames to save per dset')
    ap.add_argument('--metrics_path', '-mp', default='../trajectory_reward/results/evaluations_rad-0.1_samples-20')
    ap.add_argument('--no_mp', dest='mp', action='store_false')
    # ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--save_dir', '-sd', type=str, default='viz2')
    ap.add_argument('--plot_online', '-po', action='store_true')
    ap.add_argument('--refine', '-r', action='store_true')
    ap.add_argument('--verbose', '-v', action='store_true')
    args = ap.parse_args()

    main(args)