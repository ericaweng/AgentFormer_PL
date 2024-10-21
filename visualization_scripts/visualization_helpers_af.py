import os
import multiprocessing
import numpy as np

from utils.utils import mkdir_if_missing

from traj_toolkit.visualisation.viz_utils_univ import AnimObjBEVTraj2d, plot_anim_grid
from traj_toolkit.visualisation.viz_utils_univ import AnimObjPose2d, AnimObjPose3d


def get_metrics_str(sample_vals, i=None):
    stats = []
    for k, v in sample_vals.items():
        if i is None:
            stats.append(f"{k} {v:0.2f}")
        else:
            stats.append(f"{k} {v[int(i)]:0.2f}")
    stats = "\n".join(stats)
    return stats

SCENE_FRAMES_TO_PLOT = {
        # test
        ('cubberly-auditorium-2019-04-22_1', 40),
        ('discovery-walk-2019-02-28_0', 40),
        ('discovery-walk-2019-02-28_1', 40),
        ('food-trucks-2019-02-12_0', 40),
        ('gates-ai-lab-2019-04-17_0', 40),
        ('gates-basement-elevators-2019-01-17_0', 40),
        ('gates-foyer-2019-01-17_0', 40),
        ('gates-to-clark-2019-02-28_0', 40),
        ('hewlett-class-2019-01-23_0', 40),
        ('hewlett-class-2019-01-23_1', 40),
        ('huang-2-2019-01-25_1', 40),
        ('huang-intersection-2019-01-22_0', 40),
        ('indoor-coupa-cafe-2019-02-06_0', 40),
        ('lomita-serra-intersection-2019-01-30_0', 40),
        ('meyer-green-2019-03-16_1', 40),
        ('nvidia-aud-2019-01-25_0', 40),
        ('nvidia-aud-2019-04-18_1', 40),
        ('nvidia-aud-2019-04-18_2', 40),
        ('outdoor-coupa-cafe-2019-02-06_0', 40),
        ('quarry-road-2019-02-28_0', 40),
        ('serra-street-2019-01-30_0', 40),
        ('stlc-111-2019-04-19_1', 40),
        ('stlc-111-2019-04-19_2', 40),
        ('tressider-2019-03-16_2', 40),
        ('tressider-2019-04-26_0', 40),
        ('tressider-2019-04-26_1', 40),
        ('tressider-2019-04-26_3', 40),

        # train
        ('clark-center-2019-02-28_1', 40),
}

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

        # remove 0,0 samples and replace with nan for plotting
        obs_traj = np.where(obs_traj == 0, np.nan, obs_traj)
        pred_gt_traj = np.where(pred_gt_traj == 0, np.nan, pred_gt_traj)
        pred_fake_traj = np.where(pred_fake_traj == 0, np.nan, pred_fake_traj)

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

        if "joints" in self.cfg.id or 'kp' in self.cfg.id or 'pre_kp' in output['data'] and 'fut_kp' in output['data']:
            # images = output['data']['image_paths']
            kp_history = output['data']['pre_kp'].detach().cpu().numpy().transpose(1, 2, 0, 3)
            kp_future = output['data']['fut_kp'].detach().cpu().numpy().transpose(1, 2, 0, 3)
            # shape: (n_peds, n_kp, ts, 3)
            args_dict = {'gt_history': kp_history,
                         'gt_future': kp_future,
                         'positions': np.concatenate([obs_traj, pred_gt_traj])}
            list_of_arg_dicts.append(args_dict)
            if output['data']['pre_kp'].shape[-1] == 3:
                list_of_functions.append(AnimObjPose3d)
            else:
                import ipdb; ipdb.set_trace()
                assert output['data']['pre_kp'].shape[-1] == 2
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
            if output['data'].get('heading_vec', None) is not None:
                args_dict['last_heading'] = output['data']['heading_vec'].detach().cpu().numpy()  # (1,2)
            if 'heading_avg' in output['data'] and output['data']['heading_avg'] is not None:
                args_dict['avg_heading'] = output['data']['heading_avg'].detach().cpu().numpy()  # (1,2)
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
            if output['data'].get('heading_vec', None) is not None:
                args_dict['last_heading'] = output['data']['heading_vec'].detach().cpu().numpy()  # (1,2)
            if 'heading_avg' in output['data'] and output['data']['heading_avg'] is not None:
                args_dict['avg_heading'] = output['data']['heading_avg'].detach().cpu().numpy()  # (1,2)
            list_of_arg_dicts.append(args_dict)
            list_of_functions.append(AnimObjBEVTraj2d)

        plot_args_list['list_of_arg_dicts'] = list_of_arg_dicts
        plot_args_list['list_of_plotting_objs'] = list_of_functions
        seq_to_plot_args.append(plot_args_list)

    if self.args.mp:
        # min(self.args.num_workers)
        with multiprocessing.Pool(10) as pool:
            async_results = []
            for one_anim in seq_to_plot_args:
                async_results.append(pool.apply_async(plot_anim_grid, kwds=one_anim))
            all_figs = [async_result.get() for async_result in async_results]
    else:
        all_figs = []
        for plot_args in seq_to_plot_args:
            all_figs.append(plot_anim_grid(**plot_args))

    return all_figs


def _save_viz_gt(outputs, args, tag, anim_save_dir='../viz/jrdb_traj_scenes'):
    """save the ground truth trajectories animations"""

    seq_to_plot_args = []
    for frame_i, output in enumerate(outputs):
        frame = output['frame']
        seq = output['seq']

        # set up trajectories data for plotting
        obs_traj = output['obs_motion']
        pred_gt_traj = output['gt_motion']
        heading = output['data']['heading']
        motion_args_dict = {'obs_traj': obs_traj,
                            'gt_traj': pred_gt_traj,
                            # 'obs_mask': output['data']['pre_mask'], #torch.stack(output['data']['pre_mask']),
                            # 'gt_mask': output['data']['fut_mask'], #torch.stack(output['data']['fut_mask']),
                            'last_heading': heading,}

        # set up human keypoints data for plotting
        # (n_ped, ts, n_kp=33, 2) --> (n_peds, n_kp, ts, 2)
        kp_history = output['data']['pre_kp'].cpu().numpy().transpose(1, 2, 0, 3)
        kp_future = output['data']['fut_kp'].cpu().numpy().transpose(1, 2, 0, 3)
        kp_args_dict = {'gt_history': kp_history,
                        'gt_future': kp_future,
                        'positions': np.concatenate([obs_traj, pred_gt_traj])}

        anim_save_fn = f'{anim_save_dir}/{tag}/{seq}/frame_{frame:06d}.mp4'
        mkdir_if_missing(anim_save_fn)
        title = f"GT seq: {seq} frame: {frame}"

        plot_args_list = {'save_fn': anim_save_fn, 'title': title,
                          'list_of_arg_dicts': [kp_args_dict, motion_args_dict],
                          'list_of_plotting_objs': [AnimObjPose3d, AnimObjBEVTraj2d],
                          'save_test_frame': True, }
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
