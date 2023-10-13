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


OURS = 'af_mg1_jr1_w10'
OURS = 'af_mg-1,5_jr-1,7.5'
OURS = 'af_mg-1,5_jr-1,5'
OURS = 'vv_ml_a-0.5,0.5_b-0.4,0.4,0.2'

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
    for frame_path_ in all_frames[::skip]:
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
            NUM_SAMPLES, _, n_ped, _ = pred_fake_traj.shape

            sample_metrics, all_metrics = get_metrics_dict(pred_fake_traj.transpose(2,0,1,3), pred_gt_traj.swapaxes(0,1))
            collision_mats = all_metrics['collision_mats']
            mADE, ade_argmins = all_metrics['ADE'], all_metrics['ade_argmins']
            mFDE, fde_argmins = all_metrics['FDE'], all_metrics['fde_argmins']
            mSADE, sade_argmin = all_metrics['SADE'], all_metrics['sade_argmin']
            mSFDE, sfde_argmin = all_metrics['SFDE'], all_metrics['sfde_argmin']
            sample_crs = sample_metrics['CR']
            sample_sades = sample_metrics['SADE']
            sample_sfdes = sample_metrics['SFDE']

            # filter
            if args.refine and (n_ped <= 1 or n_ped > 5):
                break
            # our method has to have better SXDE and worse XDE than other methods
            if args.refine and method == OURS and not (
                    # np.all(other_mSFDE < mSFDE)
                                       np.all(other_mSADE > mSADE)
                                       # and np.all(other_mFDE > mFDE)
                                       and np.any(other_mADE < mADE)
                                       ):
                print(f"ours not better than other methods: "
                      f"other_mSADE ({np.array2string(np.array(other_mSADE), precision=2)}) !> mSADE ({mSADE:0.2f}) ")
                      # f"or other_mADE ({np.array2string(np.array(other_mADE), precision=2)}) !< mADE ({mADE:0.2f})")
                break

            NUM_SAMPLES = 20
            args_list = []
            args_list_nl = []
            selected_samples = list(range(NUM_SAMPLES))

            if method != OURS:
                # pick out best XDE samples from other methods
                # selected_samples = ade_argmins[:NUM_SAMPLES]
                # selected_samples = set(selected_samples)
                # sades_reverse = np.argsort(sample_sades)[::-1]
                # sade_i = 0
                # while len(selected_samples) < 3:
                #     selected_samples.add(sades_reverse[sade_i])
                # selected_samples.add(np.random.choice(sample_sades))
                # selected_samples = list(selected_samples)
                if args.refine and sample_crs[selected_samples].sum() > 0:  # other methods must have collisions
                    at_least_one_method_has_cols = True
                # print(f"{method} selected_samples: {selected_samples}")
            else:  # pick out best SXDE samples from OURS
                if args.refine and not at_least_one_method_has_cols:
                    print("no other method has cols, so we can't compare ours to them")
                    break
                # selected_samples = np.argpartition(-sample_sades, -num_samples)[-num_samples:]
                # argsorted_sample_is = np.argsort(sample_sades)
                # np.random.seed(0)
                # last_sample = np.random.choice(argsorted_sample_is[2:8], 2)
                # selected_samples = [*argsorted_sample_is[:1], *last_sample]
                # selected_samples = np.random.choice(np.argpartition(-sample_sades, -10)[-10:], 3)
                pass

            subplot_i = 0
            selected_samples_SADE = sample_metrics['SADE'][selected_samples]
            best_JADE_selected = np.argmin(selected_samples_SADE)
            selected_samples_ADE_ped_vals = all_metrics['ade_ped_val'][:, selected_samples]
            selected_samples_mADE = np.mean(np.min(selected_samples_ADE_ped_vals, axis=-1))
            selected_samples_mADE_is = np.argmin(selected_samples_ADE_ped_vals, axis=-1)
            to_plot_pred_fake_traj = pred_fake_traj[selected_samples]
            MIDDLE_SUBPLOT_I = 0#len(selected_samples) // 2
            for sample_i, sample in enumerate(to_plot_pred_fake_traj):
                # for sample_i, sample in enumerate(pred_fake_traj):
                ade_ped_vals = all_metrics['ade_ped_val'][:,sample_i]
                other_text = dict(zip([f'A{i}:' for i in range(n_ped)], ade_ped_vals))

                # if sample_i not in selected_samples:
                #     continue
                if seq in SEQUENCE_NAMES['trajnet_sdd']:
                    bkg_img_path = os.path.join(f'datasets/trajnet_sdd/reference_img/{seq[:-2]}/video{seq[-1]}/reference.jpg')
                else:
                    bkg_img_path = None

                stats = get_metrics_str({'JADE:': sample_metrics['SADE']}, sample_i)
                other_text = 'ADE\n'+ get_metrics_str(other_text)
                # print("method:", method, "ADE", mADE)
                args_dict = {'plot_title': f"{METHOD_DISP_NAMES[method]}" if subplot_i == MIDDLE_SUBPLOT_I else "",#f'{mADE:0.2f}',#f"" if sample_i == 2 and method == 'agentformer' else f"{method} {sample_i}",
                             'obs_traj': obs_traj,
                             'gt_traj': pred_gt_traj,
                             'pred_traj': sample,
                             'text_fixed_tl': other_text,
                             'text_fixed_tr': stats,
                             'sample_i': subplot_i,
                             'bkg_img_path': bkg_img_path,
                             'ade_is': selected_samples_mADE_is,
                             'is_best_JADE': subplot_i == best_JADE_selected,#sade_argmin,
                             'y_label': f'Sample {subplot_i + 1}' if method_i == 0 else None,
                             'subtitle': f"min JADE: {selected_samples_SADE.min():0.2f}   min ADE: {selected_samples_mADE:0.2f}" if subplot_i == MIDDLE_SUBPLOT_I else None,  # sample_sades[sade_argmin]
                             'plot_velocity_arrows': True,
                             'is_best_JADE_of_all': True if method == OURS else False,
                             'is_best_ADE_of_all': False,
                             'collision_mats': collision_mats[selected_samples[sample_i]]}
                args_dict_nl = {'plot_title': "",
                                'obs_traj': obs_traj,
                                'gt_traj': pred_gt_traj,
                                'pred_traj': sample,
                                'plot_velocity_arrows': True,
                                'collision_mats': collision_mats[sample_i]}
                args_list.append(args_dict)
                args_list_nl.append(args_dict_nl)
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
                if args.png:
                    png_or_pdf = 'png'
                    anim_save_fn = os.path.join(args.save_dir, seq, f'frame_{frame:06d}', f'{method}.{png_or_pdf}')
                    anim_save_fn_nl = os.path.join(args.save_dir, seq, f'frame_{frame:06d}',
                                                   f'{method}-nl.{png_or_pdf}')
                else:
                    png_or_pdf = 'pdf'
                    anim_save_fn = os.path.join(args.save_dir, f'{seq}_frame_{frame:06d}.{png_or_pdf}')
                    anim_save_fn_nl = os.path.join(args.save_dir, f'{seq}_frame_{frame:06d}-nl.{png_or_pdf}')
                bounds = get_max_bounds(trajs_list_for_bounds_calculation, padding=0.2)
                best_ADE_method_i = np.argmin(other_mADE)
                best_JADE_method_i = len(args.method) - 1
                non_ours_args_list[best_ADE_method_i][MIDDLE_SUBPLOT_I]['is_best_ADE_of_all'] = True
                non_ours_args_list[best_ADE_method_i][MIDDLE_SUBPLOT_I]['plot_title'] = f"{METHOD_DISP_NAMES[args.method[best_ADE_method_i]]} (best ADE)"
                args_list[MIDDLE_SUBPLOT_I]['plot_title'] = f"{METHOD_DISP_NAMES[args.method[best_JADE_method_i]]}\n(best JADE)"
                plot_args = [anim_save_fn, "", bounds, (len(args.method), NUM_SAMPLES), *non_ours_args_list, args_list]
                plot_args_nl = [anim_save_fn_nl, "", bounds, (len(args.method), NUM_SAMPLES), *non_ours_args_list_nl, args_list_nl]
                seq_to_plot_args.append(plot_args)
                # seq_to_plot_args.append(plot_args_nl)
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
    ap.add_argument('--method', '-m', type=str, nargs='+', default=['vv', 'vv_ml_a-0.5,0.5_b-0.4,0.4,0.2'])
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