"""find outstanding examples between AF and Our method and plot animation"""

import os
import argparse
import multiprocessing
from pathlib import Path
from itertools import starmap
import numpy as np

from utils.utils import mkdir_if_missing
from scripts.evaluate_all import peds_pandas_way
from viz_utils2 import plot_anim_grid, get_metrics_str
from metrics import compute_ADE_marginal, compute_FDE_marginal, compute_ADE_joint, \
    compute_FDE_joint, compute_CR


def get_metrics_dict(pred_fake_traj, pred_gt_traj):
    _, sample_collisions, collision_mats = compute_CR(pred_fake_traj, pred_gt_traj, return_sample_vals=True, return_collision_mat=True, collision_rad=0.1)
    ade, ade_argmins = compute_ADE_marginal(pred_fake_traj, pred_gt_traj, return_argmin=True)
    fde, fde_argmins = compute_FDE_marginal(pred_fake_traj, pred_gt_traj, return_argmin=True)
    sade, sade_samples, sade_argmin = compute_ADE_joint(pred_fake_traj, pred_gt_traj, return_argmin=True, return_sample_vals=True)
    sfde, sfde_samples, sfde_argmin = compute_FDE_joint(pred_fake_traj, pred_gt_traj, return_argmin=True, return_sample_vals=True)
    metrics_dict = {'collision_mats': collision_mats,
                    'ADE': ade,
                    'FDE': fde,
                    'SADE': sade,
                    'SFDE': sfde,
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

    # determine which frames to save
    # if args.save_num is None:
    #     skip = 50
    # else:
    #     skip = max(1, int(len(all_frames) / args.save_num))
    # print(f"Saving {len(all_frames[::skip])} frames")

    print(f"Saving {len(all_frames)} frames per method across all dsets except frames with only 1 ped")

    skip = 1
    # gather list of args for plotting
    seq_to_plot_args = []
    sps = []
    for frame_path in all_frames[::skip]:
        seq = frame_path.parent.name
        AF_best_SFDE, AF_best_SADE, AF_best_FDE, AF_best_ADE = None, None, None, None
        for method in args.method:
            pred_gt_traj = obs_traj = None
            samples = []
            frame = int(frame_path.name.split('_')[-1])
            frame_path = Path(str(frame_path).replace(placeholder_method, method))
            for filename in frame_path.glob('*.txt'):
                if 'gt' in str(filename.name):
                    pred_gt_traj = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
                    pred_gt_traj = peds_pandas_way(pred_gt_traj, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
                elif 'obs' in str(filename.name):
                    obs_traj = np.loadtxt(filename, delimiter=' ', dtype='float32')  # (frames x agents) x 4
                    obs_traj = peds_pandas_way(obs_traj, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])[:,::-1]  # todo
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
                import ipdb; ipdb.set_trace()
                obs_path = os.path.join(str(frame_path).replace(method, 'agentformer'), 'obs.txt')
                obs_traj = np.loadtxt(obs_path, delimiter=' ', dtype='float32')  # (frames x agents) x 4
                obs_traj = peds_pandas_way(obs_traj, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id'])
            assert obs_traj.shape[0] == 8
            assert pred_gt_traj.shape[0] == 12
            pred_fake_traj = np.stack(samples, axis=0)  # (num_samples, frames, agents, 2)

            num_samples, _, n_ped, _ = pred_fake_traj.shape
            if n_ped == 1:
                continue

            if seq in SEQUENCE_NAMES['trajnet_sdd']:
                # if dset == 'trajnet_sdd':
                bkg_img_path = os.path.join(f'datasets/trajnet_sdd/reference_img/{seq[:-2]}/video{seq[-1]}/reference.jpg')
                anim_save_fn = os.path.join(args.save_dir, 'trajnet_sdd', seq, f'frame_{frame:06d}', f'{method}.mp4')
            else:
                bkg_img_path = None
                anim_save_fn = os.path.join(args.save_dir, seq, f'frame_{frame:06d}', f'{method}.mp4')

            plot_args_list = [anim_save_fn, f"Seq: {seq} frame: {frame} Method: {method}", (5, 4)]

            sample_metrics, all_metrics = get_metrics_dict(pred_fake_traj.transpose(2,0,1,3), pred_gt_traj.swapaxes(0,1))
            collision_mats = all_metrics['collision_mats']
            min_SADE, min_SADE_i = all_metrics['SADE'], int(all_metrics['sade_argmin'])
            min_SFDE, min_SFDE_i = all_metrics['SFDE'], int(all_metrics['sfde_argmin'])
            min_ADE, ade_argmins = all_metrics['ADE'], all_metrics['ade_argmins']
            min_FDE, fde_argmins = all_metrics['FDE'], all_metrics['fde_argmins']
            # plot only the plots that are beneficial to our claim
            if AF_best_SADE is not None and method != 'agentformer':
                if min_SADE >= AF_best_SADE:
                    # print(f"skipping {seq} frame {frame} bc mSADE ({min_SADE:.2f}) not better than AF's ({AF_best_SADE:.2f})")
                    continue
                if min_SFDE >= AF_best_SFDE:
                    # print(f"skipping {seq} frame {frame} bc {method}'s mSFDE ({min_SFDE:.2f}) not better than AF's ({AF_best_SFDE:.2f})")
                    continue
                if min_ADE <= AF_best_ADE:
                    # print(f"skipping {seq} frame {frame} bc ADE ({min_ADE:.2f}) not worse than AF's ({AF_best_ADE:.2f})")
                    continue
                if min_FDE <= AF_best_FDE:
                    # print(f"skipping {seq} frame {frame} bc FDE ({min_FDE:.2f}) not worse than AF's ({AF_best_FDE:.2f})")
                    continue
                if n_ped < 2:
                    # print(f"skipping {seq} frame {frame} bc n_ped ({n_ped}) < 2")
                    continue
                if collision_mats[min_SADE_i].sum() > 0:
                    # print(f"skipping {seq} frame {frame} bc mSADE ({min_SADE:.2f}) has collisions")
                    continue
                # if AF_cr_SADE < sample_metrics['CR'][min_SADE_i]:  # AF's CR for min ade example is better than our cr_mADE
                #     print(f"skipping {seq} frame {frame} bc mSADE ({min_SADE:.2f}) has CR ({sample_metrics['CR'][min_SADE_i]:.2f}) < {cr_mADE:.2f}")
                #     continue
                print(f"PLOTTING frame {frame}")
            elif AF_best_SADE is None and method == 'agentformer':
                # save AF data
                if min_SADE_i not in ade_argmins:  # min SADE scene should have some best ADE peds
                    continue
                if sample_metrics['CR'][min_SADE_i] == 0:
                    continue
                AF_best_SADE = min_SADE
                AF_best_SFDE = min_SFDE
                AF_best_FDE = min_FDE
                AF_best_ADE = min_ADE
            else:
                continue

            # okay now we can plot
            pred_fake_traj_min = pred_fake_traj[min_SADE_i]
            min_SADE_stats = get_metrics_str(sample_metrics, min_SADE_i)
            # if method != "agentformer":  # is af-mg-jr not AF
            args_dict = {'plot_title': f"best mSADE (pred {min_SADE_i}) best SFDE (pred {min_SFDE_i})"
                                       f"\nADE: {min_ADE:.2f} (peds {ade_argmins}) "
                                       f"\nFDE: {min_FDE:.2f} (peds {fde_argmins})",
                         'obs_traj': obs_traj,
                         'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_fake_traj_min,
                         'collision_mats': collision_mats[min_SADE_i],
                         'bkg_img_path': bkg_img_path,
                         'text_fixed': min_SADE_stats
                         }
            # else:  # is plain AF: show min ADE example mix-and-match
            #     cr_mADE, collision_mats_mSADE = compute_CR(pred_fake_traj_min, pred_gt_traj, aggregation=compute_ADE_marginal)
            #     sample_metrics_min_ade = {
            #             'cr': cr_mADE,
            #             'ade': min_ADE,
            #             'fde': min_FDE,
            #     }
            #     min_SADE_stats = get_metrics_str(sample_metrics_min_ade, ade_argmins)
            #     args_dict = {'plot_title': f"best ADE: {min_ADE:.2f} ({ade_argmins}) "
            #                                f"\nFDE: {min_FDE:.2f} ({fde_argmins})",
            #                  'obs_traj': obs_traj,
            #                  'pred_traj_gt': pred_gt_traj,
            #                  'pred_traj_fake': pred_fake_traj_min,
            #                  'collision_mats': collision_mats_mSADE,
            #                  'bkg_img_path': bkg_img_path,
            #                  'text_fixed': min_SADE_stats
            #                  }
            #     pred_fake_traj_min = pred_fake_traj[ade_argmins]
            #     print("pred_fake_traj.shape:", pred_fake_traj.shape)
            #     print("pred_fake_traj_min.shape:", pred_fake_traj_min.shape)

            plot_args_list.append(args_dict)

            for sample_i in range(num_samples - 1):
                stats = get_metrics_str(sample_metrics, sample_i)
                args_dict = {'plot_title': "" if sample_i == 1 else f"Pred {sample_i}",
                             'obs_traj': obs_traj,
                             'pred_traj_gt': pred_gt_traj,
                             'pred_traj_fake': pred_fake_traj[sample_i],
                             'text_fixed': stats,
                             'bkg_img_path': bkg_img_path,
                             # 'highlight_peds': argmins[frame_i],
                             'collision_mats': collision_mats[sample_i]}
                plot_args_list.append(args_dict)
            if method == 'agentformer':
                AF_plot_args_list = plot_args_list
            else:
                seq_to_plot_args.append(AF_plot_args_list)
                seq_to_plot_args.append(plot_args_list)
            if args.plot_online:
                if len(sps) >= args.num_workers:
                    # this should work if all subprocesses take the same amount of time;
                    # otherwise we might be waiting longer than necessary
                    sps[0].join()
                    sps = sps[1:]
                if len(seq_to_plot_args) == 0:
                    continue
                seq_to_plot_online = seq_to_plot_args.pop(0)
                mkdir_if_missing(anim_save_fn)
                process = multiprocessing.Process(target=plot_anim_grid, args=seq_to_plot_online)
                process.start()
                sps.append(process)
                # print(f"plotting {seq} frame {frame} method {method} online")

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
    ap.add_argument('--plot_online', '-po', action='store_true')
    args = ap.parse_args()

    main(args)