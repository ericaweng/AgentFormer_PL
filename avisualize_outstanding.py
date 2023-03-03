import os
import argparse
import multiprocessing
from pathlib import Path
from itertools import starmap

from utils.utils import mkdir_if_missing
from scripts.evaluate_all import peds_pandas_way
from viz_utils import plot_fig
from metrics import compute_ADE_marginal, compute_FDE_marginal, compute_ADE_joint, \
    compute_FDE_joint, compute_CR

def get_metrics_dict(pred_fake_traj, pred_gt_traj, metrics_list=None):
    # if metrics_list is None:
    #     metrics_list = ['cr', 'ADE', 'FDE', 'SADE', 'SFDE']
    # metric_name_to_fn = {'cr': partial(compute_CR, return_sample_vals=True, return_collision_mat=True,
    #                                    collision_rad=0.1),
    #                      'ADE': partial(compute_ADE_marginal, return_argmin=True),
    #                      'FDE': partial(compute_FDE_marginal, return_argmin=True),
    #                      'SADE': partial(compute_ADE_joint, return_sample_vals=True),
    #                      'SFDE': partial(compute_FDE_joint, return_sample_vals=True)}
    # metrics_dict = {}
    # collision_mats = None
    # for metric in metrics_list:
    #     res = metric_name_to_fn[metric](pred_fake_traj, pred_gt_traj)
    #     if metric == 'cr':
    #         _, metrics_dict[metric], collision_mats = res
    #     else:
    #         , metrics_dict[metric] = res
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
            mkdir_if_missing(anim_save_fn)
            plot_args_list = [anim_save_fn, f"Seq: {seq} frame: {frame} Method: {method}", (5, 4)]

            sample_metrics, all_metrics = get_metrics_dict(pred_fake_traj.transpose(2,0,1,3), pred_gt_traj.swapaxes(0,1))
            # min_SADE_i = np.argmin(sample_metrics['SADE'])
            collision_mats = sample_metrics['collision_mats']
            min_SADE, min_SADE_i = all_metrics['SADE'], all_metrics['sade_argmin']
            min_SFDE, min_SFDE_i = all_metrics['SFDE'], all_metrics['sfde_argmin']
            min_ADE, ade_argmins = all_metrics['ADE'], all_metrics['ade_argmins']
            min_FDE, fde_argmins = all_metrics['FDE'], all_metrics['fde_argmins']
            # plot only the plots that are beneficial to our claim
            if AF_best_SADE is not None:
                if min_SADE >= AF_best_SADE:
                    print(f"skipping {seq} frame {frame} bc mSADE ({min_SADE:.2f}) not better than AF's ({AF_best_SADE:.2f})")
                    continue
                if min_SFDE >= AF_best_SFDE:
                    print(f"skipping {seq} frame {frame} bc {method}'s mSFDE ({min_SFDE:.2f}) not better than AF's ({AF_best_SFDE:.2f})")
                    continue
                if min_ADE <= AF_best_ADE:
                    print(f"skipping {seq} frame {frame} bc ADE ({min_ADE:.2f}) not worse than AF's ({AF_best_ADE:.2f})")
                    continue
                if min_FDE <= AF_best_FDE:
                    print(f"skipping {seq} frame {frame} bc FDE ({min_FDE:.2f}) not worse than AF's ({AF_best_FDE:.2f})")
                    continue
                if n_ped < 2:
                    print(f"skipping {seq} frame {frame} bc n_ped ({n_ped}) < 2")
                    continue
                print(f"FINALLY PLOTTING frame {frame}")
            else:
                # save AF data
                AF_best_SADE = min_SADE
                AF_best_SFDE = min_SFDE
                AF_best_FDE = min_FDE
                AF_best_ADE = min_ADE

            # okay now we can plot
            pred_fake_traj_min = pred_fake_traj[min_SADE_i]
            min_SADE_stats = get_metrics_str(sample_metrics, min_SADE_i)
            args_dict = {'plot_title': f"best mSADE (pred {min_SADE_i})"
                                       f"\nbest SFDE: {min_SFDE:.2f} ({min_SFDE_i})"
                                       f"\nbest ADE: {min_ADE:.2f} ({ade_argmins}) "
                                       f"\nFDE: {min_FDE:.2f} ({fde_argmins})",
                         'obs_traj': obs_traj,
                         'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_fake_traj_min,
                         'collision_mats': collision_mats[min_SADE_i],
                         'bkg_img_path': bkg_img_path,
                         'text_fixed': min_SADE_stats
                         }
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
                if len(sps) >= args.max_cmds_at_a_time:
                    # this should work if all subprocesses take the same amount of time;
                    # otherwise we might be waiting longer than necessary
                    sps[0].wait()
                    sps = sps[1:]
                sps.append(multiprocessing.Process(target=plot_fig, args=plot_args_list))

    # plot in parallel
    print(f"plotting {len(seq_to_plot_args)} plots")
    if not args.plot_online:
        if args.mp:
            with multiprocessing.Pool(args.num_workers) as pool:
                pool.starmap(plot_fig, seq_to_plot_args)
        else:
            list(starmap(plot_fig, seq_to_plot_args))
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
    ap.add_argument('--save_dir', type=str, default='viz2')
    ap.add_argument('--plot_online', '-po', action='store_true')
    args = ap.parse_args()

    main(args)