import os
from itertools import starmap
import multiprocessing
import numpy as np

from utils.utils import mkdir_if_missing
from viz_utils import get_metrics_str

from visualization_scripts.viz_utils_univ import AnimObjBEVTraj2d, plot_anim_grid



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


def _save_catch_all(self, outputs, all_sample_vals, collision_mats, tag='', anim_save_dir=None):
    seq_to_plot_args = []
    for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
        frame = output['frame']
        seq = output['seq']

        obs_traj = output['obs_motion'].numpy()
        # swap (num_peds, ts, 2) --> (ts, num_peds, 2) for visualization
        pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
        # (samples, ts, n_peds, 2) --> (samples, ts, n_peds, 2)
        pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)

        num_samples, _, n_ped, _ = pred_fake_traj.shape

        title = f"{tag} Model: {self.cfg.id} Seq: {seq} frame: {frame} Epoch: {self.current_epoch}"
        if anim_save_dir is not None:
            anim_save_fn = os.path.join(anim_save_dir, f'{tag}/{seq}/frame_{frame:06d}/{self.model_name}_epoch-{self.current_epoch}.mp4')
            mkdir_if_missing(anim_save_fn)
        else:
            anim_save_fn = None
        plot_args_list = {'save_fn': anim_save_fn, 'title': title}

        list_of_arg_dicts = []
        list_of_functions = []

        if "joints" in self.cfg.id:
            from visualization_scripts.viz_utils_univ import AnimObjPose2d, AnimObjPose3d
            images = output['data']['image_paths']
            kp_history = output['data']['pre_kp'].detach().cpu().numpy().transpose(1, 2, 0, 3)
            kp_future = output['data']['fut_kp'].detach().cpu().numpy().transpose(1, 2, 0, 3)
            args_dict = {'gt_history': kp_history,
                         'gt_future': kp_future,
                         'positions': np.concatenate([obs_traj, pred_gt_traj])}
            list_of_arg_dicts.append(args_dict)
            list_of_functions.append(AnimObjPose2d)

        # best JADE sample
        if seq_to_sample_metrics['ADE'] is not None:
            SADE_min_i = np.argmin(seq_to_sample_metrics['ADE'])
            pred_fake_traj_min = pred_fake_traj[SADE_min_i]
            min_SADE_stats = get_metrics_str(seq_to_sample_metrics, SADE_min_i)
            args_dict = {'plot_title': f"best mSADE sample",
                         'obs_traj': obs_traj,
                         'gt_traj': pred_gt_traj,
                         'pred_traj': pred_fake_traj_min,
                         'collision_mats': collision_mats[frame_i][-1],
                         'text_fixed': min_SADE_stats}
            if "heading" in self.cfg.id or output['data']['heading_vec'] is not None and output['data']['heading_vec'].detach().cpu().numpy() is not None:
                heading = output['data']['heading_vec'].detach().cpu().numpy()  # (1,2)
                heading_avg = output['data']['heading_avg'].detach().cpu().numpy()  # (1,2)
                args_dict['last_heading'] = heading
                args_dict['avg_heading'] = heading_avg

            list_of_arg_dicts.append(args_dict)
            list_of_functions.append(AnimObjBEVTraj2d)

        # other samples
        for sample_i in range(num_samples - 1):
            if seq_to_sample_metrics['ADE'] is not None:
                stats = get_metrics_str(seq_to_sample_metrics, sample_i)
            args_dict = {'plot_title': f"Sample {sample_i}",
                         'obs_traj': obs_traj,
                         'gt_traj': pred_gt_traj,
                         'pred_traj': pred_fake_traj[sample_i],
                         'text_fixed': stats if seq_to_sample_metrics['ADE'] is not None else None,
                         'collision_mats': collision_mats[frame_i][sample_i]}
            if "heading" in self.cfg.id or output['data']['heading_vec'] is not None and output['data'][
                'heading_vec'].detach().cpu().numpy() is not None:
                args_dict['last_heading'] = heading
                args_dict['avg_heading'] = heading_avg

            list_of_arg_dicts.append(args_dict)
            list_of_functions.append(AnimObjBEVTraj2d)

        plot_args_list['list_of_arg_dicts'] = list_of_arg_dicts
        plot_args_list['list_of_plotting_objs'] = list_of_functions
        seq_to_plot_args.append(plot_args_list)

    if self.args.mp:
        with multiprocessing.Pool(self.args.num_workers) as pool:
            async_results = []
            for one_anim in seq_to_plot_args:
                async_results.append(pool.apply_async(plot_anim_grid, kwds=one_anim))
            all_figs = [async_result.get() for async_result in async_results]
    else:
        all_figs = []
        for plot_args in seq_to_plot_args:
            all_figs.append(plot_anim_grid(**plot_args))

    return all_figs


def _save_viz_gt(outputs, args, tag):
    """save the ground truth trajectories animations"""
    seq_to_plot_args = []
    for frame_i, output in enumerate(outputs):
        frame = output['frame']
        seq = output['seq']
        obs_traj = output['obs_motion']
        pred_gt_traj = output['gt_motion']
        heading = output['data']['heading']

        anim_save_fn = f'../viz/jrdb_traj_scenes/{tag}/{seq}/frame_{frame:06d}.mp4'
        mkdir_if_missing(anim_save_fn)
        title = f"GT seq: {seq} frame: {frame}"
        args_dict = {'obs_traj': obs_traj,
                     'gt_traj': pred_gt_traj,
                     'last_heading': heading,}

        plot_args_list = {'save_fn': anim_save_fn, 'title': title, 'list_of_arg_dicts': [args_dict],
                          'list_of_plotting_objs': [AnimObjBEVTraj2d]}
        seq_to_plot_args.append(plot_args_list)

    if args.mp:
        with multiprocessing.Pool(args.num_workers) as pool:
            async_results = []
            for one_anim in seq_to_plot_args:
                async_results.append(pool.apply_async(plot_anim_grid, kwds=one_anim))
            all_figs = [async_result.get() for async_result in async_results]
    else:
        all_figs = []
        for plot_args in seq_to_plot_args:
            all_figs.append(plot_anim_grid(**plot_args))

    return all_figs
