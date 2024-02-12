import os
from itertools import starmap
import multiprocessing
import numpy as np

from utils.utils import mkdir_if_missing
from viz_utils import get_metrics_str
from visualization_scripts.viz_utils2 import plot_anim_grid
from visualization_scripts.viz_utils_3d import plot_anim_grid_3d
from visualization_scripts.viz_utils_2d import plot_anim_grid_2d


def _save_viz_w_heading(self, outputs, all_sample_vals, collision_mats, tag=''):
    seq_to_plot_args = []
    for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
        frame = output['frame']
        seq = output['seq']
        obs_traj = output['obs_motion'].numpy()
        heading = output['data']['heading_vec'].detach().cpu().numpy()  # (1,2)
        heading_avg = output['data']['heading_avg'].detach().cpu().numpy()  # (1,2)
        # swap (num_peds, ts, 2) --> (ts, num_peds, 2) for visualization
        pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
        # (samples, ts, n_peds, 2) --> (samples, ts, n_peds, 2)
        pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)

        num_samples, _, n_ped, _ = pred_fake_traj.shape

        anim_save_fn = None  # f'../viz/{seq}/frame_{frame:06d}/{self.model_name}_epoch-{self.current_epoch}_{tag}.mp4'
        # mkdir_if_missing(anim_save_fn)
        title = f"Model: {self.cfg.id} Seq: {seq} frame: {frame} Epoch: {self.current_epoch}"
        plot_args_list = [anim_save_fn, title, (3, 2)]
        list_of_arg_dicts = []

        SADE_min_i = np.argmin(seq_to_sample_metrics['ADE'])
        pred_fake_traj_min = pred_fake_traj[SADE_min_i]
        min_SADE_stats = get_metrics_str(seq_to_sample_metrics, SADE_min_i)
        args_dict = {'plot_title': f"best mSADE sample",
                     'obs_traj': obs_traj,
                     'gt_traj': pred_gt_traj,
                     'last_heading': heading,
                     'avg_heading': heading_avg,
                     'pred_traj': pred_fake_traj_min,
                     'collision_mats': collision_mats[frame_i][-1],
                     'text_fixed': min_SADE_stats}
        list_of_arg_dicts.append(args_dict)

        for sample_i in range(num_samples - 1):
            stats = get_metrics_str(seq_to_sample_metrics, sample_i)
            args_dict = {'plot_title': f"Sample {sample_i}",
                         'obs_traj': obs_traj,
                         'gt_traj': pred_gt_traj,
                         'last_heading': heading,
                         'avg_heading': heading_avg,
                         'pred_traj': pred_fake_traj[sample_i],
                         'text_fixed': stats,
                         # 'highlight_peds': argmins[frame_i],
                         'collision_mats': collision_mats[frame_i][sample_i]}
            list_of_arg_dicts.append(args_dict)

        plot_args_list.append(list_of_arg_dicts)
        seq_to_plot_args.append(plot_args_list)

    if self.args.mp:
        with multiprocessing.Pool(self.num_workers) as pool:
            all_figs = pool.starmap(plot_anim_grid, seq_to_plot_args)

    else:
        all_figs = list(starmap(plot_anim_grid, seq_to_plot_args))

    return all_figs

def _save_viz_w_pose_3d(self, outputs, all_sample_vals, collision_mats, tag=''):
    seq_to_plot_args = []
    for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
        frame = output['frame']
        seq = output['seq']
        obs_traj = output['obs_motion'].numpy()
        joints_history = output['data']['pre_joints'].detach().cpu().numpy().transpose(1, 2, 0, 3)
        joints_future = output['data']['fut_joints'].detach().cpu().numpy().transpose(1, 2, 0, 3)
        # swap (num_peds, ts, 2) --> (ts, num_peds, 2) for visualization
        pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
        # (samples, ts, n_peds, 2) --> (samples, ts, n_peds, 2)
        pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)

        num_samples, _, n_ped, _ = pred_fake_traj.shape

        anim_save_fn = None
        # mkdir_if_missing(anim_save_fn)
        title = f"Model: {self.cfg.id} Seq: {seq} frame: {frame} Epoch: {self.current_epoch}"
        plot_args_list = [anim_save_fn, title, (3, 2)]
        list_of_arg_dicts = []

        args_dict = {'gt_history': joints_history,
                     'gt_future': joints_future}
        list_of_arg_dicts.append(args_dict)

        SADE_min_i = np.argmin(seq_to_sample_metrics['ADE'])
        pred_fake_traj_min = pred_fake_traj[SADE_min_i]
        min_SADE_stats = get_metrics_str(seq_to_sample_metrics, SADE_min_i)
        args_dict = {'plot_title': f"best mSADE sample",
                     'obs_traj': obs_traj,
                     'gt_traj': pred_gt_traj,
                     'pred_traj': pred_fake_traj_min,
                     'collision_mats': collision_mats[frame_i][-1],
                     'text_fixed': min_SADE_stats}
        list_of_arg_dicts.append(args_dict)

        for sample_i in range(num_samples - 1):
            stats = get_metrics_str(seq_to_sample_metrics, sample_i)
            args_dict = {'plot_title': f"Sample {sample_i}",
                         'obs_traj': obs_traj,
                         'gt_traj': pred_gt_traj,
                         'pred_traj': pred_fake_traj[sample_i],
                         'text_fixed': stats,
                         # 'highlight_peds': argmins[frame_i],
                         'collision_mats': collision_mats[frame_i][sample_i]}
            list_of_arg_dicts.append(args_dict)

        plot_args_list.append(list_of_arg_dicts)
        seq_to_plot_args.append(plot_args_list)

    if self.args.mp:
        with multiprocessing.Pool(self.num_workers) as pool:
            all_figs = pool.starmap(plot_anim_grid_3d, seq_to_plot_args)

    else:
        all_figs = list(starmap(plot_anim_grid_3d, seq_to_plot_args))

    return all_figs


def _save_viz_w_pose_2d(self, outputs, all_sample_vals, collision_mats, tag=''):
    seq_to_plot_args = []
    for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
        frame = output['frame']
        seq = output['seq']
        obs_traj = output['obs_motion'].numpy()
        joints_history = output['data']['pre_joints'].detach().cpu().numpy().transpose(1, 2, 0, 3)
        joints_future = output['data']['fut_joints'].detach().cpu().numpy().transpose(1, 2, 0, 3)
        # swap (num_peds, ts, 2) --> (ts, num_peds, 2) for visualization
        pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
        # (samples, ts, n_peds, 2) --> (samples, ts, n_peds, 2)
        pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)

        num_samples, _, n_ped, _ = pred_fake_traj.shape

        anim_save_fn = None
        # mkdir_if_missing(anim_save_fn)
        title = f"Model: {self.cfg.id} Seq: {seq} frame: {frame} Epoch: {self.current_epoch}"
        plot_args_list = [anim_save_fn, title, (3, 2)]
        list_of_arg_dicts = []

        args_dict = {'gt_history': joints_history,
                     'gt_future': joints_future,
                     'positions': np.concatenate([obs_traj, pred_gt_traj])}
        list_of_arg_dicts.append(args_dict)

        SADE_min_i = np.argmin(seq_to_sample_metrics['ADE'])
        pred_fake_traj_min = pred_fake_traj[SADE_min_i]
        min_SADE_stats = get_metrics_str(seq_to_sample_metrics, SADE_min_i)
        args_dict = {'plot_title': f"best mSADE sample",
                     'obs_traj': obs_traj,
                     'gt_traj': pred_gt_traj,
                     'pred_traj': pred_fake_traj_min,
                     'collision_mats': collision_mats[frame_i][-1],
                     'text_fixed': min_SADE_stats}
        list_of_arg_dicts.append(args_dict)

        for sample_i in range(num_samples - 1):
            stats = get_metrics_str(seq_to_sample_metrics, sample_i)
            args_dict = {'plot_title': f"Sample {sample_i}",
                         'obs_traj': obs_traj,
                         'gt_traj': pred_gt_traj,
                         'pred_traj': pred_fake_traj[sample_i],
                         'text_fixed': stats,
                         'collision_mats': collision_mats[frame_i][sample_i]}
            list_of_arg_dicts.append(args_dict)

        plot_args_list.append(list_of_arg_dicts)
        seq_to_plot_args.append(plot_args_list)

    if self.args.mp:
        with multiprocessing.Pool(self.num_workers) as pool:
            all_figs = pool.starmap(plot_anim_grid_2d, seq_to_plot_args)

    else:
        all_figs = list(starmap(plot_anim_grid_2d, seq_to_plot_args))

    return all_figs


def _save_viz(self, outputs, all_sample_vals, all_meters_values, argmins, collision_mats, tag=''):
    seq_to_plot_args = []
    for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
        frame = output['frame']
        seq = output['seq']
        obs_traj = output['obs_motion'].numpy()
        pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
        pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)  # (samples, ts, n_peds, 2)

        num_samples, _, n_ped, _ = pred_fake_traj.shape

        anim_save_fn = None  # f'../viz/{seq}/frame_{frame:06d}/{self.model_name}_epoch-{self.current_epoch}_{tag}.mp4'
        # mkdir_if_missing(anim_save_fn)
        title = f"Model: {self.cfg.id} Seq: {seq} frame: {frame} Epoch: {self.current_epoch}"
        plot_args_list = [anim_save_fn, title, (3, 2)]
        list_of_arg_dicts = []

        # pred_fake_traj_min = pred_fake_traj[argmins[frame_i],:,np.arange(n_ped)].swapaxes(0, 1)  # (n_ped, )
        # min_ADE_stats = get_metrics_str(dict(zip(stats_func.keys(), all_meters_values[frame_i])))
        if self.dataset_name == 'trajnet_sdd':
            bkg_img_path = os.path.join(f'datasets/trajnet_sdd/reference_img/{seq[:-2]}/video{seq[-1]}/reference.jpg')
        else:
            bkg_img_path = None
        SADE_min_i = np.argmin(seq_to_sample_metrics['ADE'])
        pred_fake_traj_min = pred_fake_traj[SADE_min_i]
        min_SADE_stats = get_metrics_str(seq_to_sample_metrics, SADE_min_i)
        args_dict = {'plot_title': f"best mSADE sample ({SADE_min_i})",
                     'obs_traj': obs_traj,
                     'gt_traj': pred_gt_traj,
                     'pred_traj': pred_fake_traj_min,
                     'collision_mats': collision_mats[frame_i][-1],
                     'bkg_img_path': bkg_img_path,
                     'text_fixed': min_SADE_stats}
        list_of_arg_dicts.append(args_dict)

        for sample_i in range(num_samples):
            stats = get_metrics_str(seq_to_sample_metrics, sample_i)
            args_dict = {'plot_title': f"Sample {sample_i}",
                         'obs_traj': obs_traj,
                         'gt_traj': pred_gt_traj,
                         'pred_traj': pred_fake_traj[sample_i],
                         'text_fixed': stats,
                         'bkg_img_path': bkg_img_path,
                         # 'highlight_peds': argmins[frame_i],
                         'collision_mats': collision_mats[frame_i][sample_i]}
            list_of_arg_dicts.append(args_dict)
        plot_args_list.append(list_of_arg_dicts)
        seq_to_plot_args.append(plot_args_list)

    if self.args.mp:
        with multiprocessing.Pool(self.num_workers) as pool:
            all_figs = pool.starmap(plot_anim_grid, seq_to_plot_args)

    else:
        all_figs = list(starmap(plot_anim_grid, seq_to_plot_args))

    return all_figs


def _save_viz_nuscenes(self, outputs, all_sample_vals, collision_mats, tag=''):
    # log
    # map_name
    # nusc_map = NuScenesMap(dataroot='dataset/nuscenes', map_name=map_name)
    # Plotting the map
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # nusc_map.render_map_patch(ax, nusc_map.extract_map_patch([-500, 1500, -1000, 1000], map_location),
    #                           layers=['drivable_area', 'lane', 'ped_crossing', 'walkway'])
    seq_to_plot_args = []
    for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
        frame = output['frame']
        seq = output['seq']
        obs_traj = output['obs_motion'].numpy()
        pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
        pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)  # (samples, ts, n_peds, 2)

        heading = output['data']['heading'].detach().cpu().numpy()  # (1,2)
        map = output['data']['scene_vis_map']
        num_samples, _, n_ped, _ = pred_fake_traj.shape

        anim_save_fn = None#f'../viz/{seq}/frame_{frame:06d}/{self.model_name}_epoch-{self.current_epoch}_{tag}.mp4'
        # mkdir_if_missing(anim_save_fn)
        title = f"Model: {self.cfg.id} Seq: {seq} frame: {frame} Epoch: {self.current_epoch}"
        plot_args_list = [anim_save_fn, title, (3, 2)]
        list_of_arg_dicts = []

        SADE_min_i = np.argmin(seq_to_sample_metrics['ADE'])
        pred_fake_traj_min = pred_fake_traj[SADE_min_i]
        min_SADE_stats = get_metrics_str(seq_to_sample_metrics, SADE_min_i)
        args_dict = {'plot_title': f"best mSADE sample ({SADE_min_i})",
                     'obs_traj': obs_traj,
                     'gt_traj': pred_gt_traj,
                     'pred_traj': pred_fake_traj_min,
                     'last_heading': heading,
                     'collision_mats': collision_mats[frame_i][-1],
                     'map': map,
                     'text_fixed': min_SADE_stats}
        list_of_arg_dicts.append(args_dict)

        for sample_i in range(num_samples):
            stats = get_metrics_str(seq_to_sample_metrics, sample_i)
            args_dict = {'plot_title': f"Sample {sample_i}",
                         'obs_traj': obs_traj,
                         'gt_traj': pred_gt_traj,
                         'pred_traj': pred_fake_traj[sample_i],
                         'text_fixed': stats,
                         'last_heading': heading,
                         'map': map,
                         'collision_mats': collision_mats[frame_i][sample_i]}
            list_of_arg_dicts.append(args_dict)
            plot_args_list.append(list_of_arg_dicts)
        seq_to_plot_args.append(plot_args_list)

    if self.args.mp:
        with multiprocessing.Pool(self.num_workers) as pool:
            all_figs = pool.starmap(plot_anim_grid, seq_to_plot_args)

    else:
        all_figs = list(starmap(plot_anim_grid, seq_to_plot_args))

    return all_figs