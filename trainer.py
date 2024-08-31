import os
from itertools import starmap
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from model.model_lib import model_dict
from eval import eval_one_seq
from utils.utils import mkdir_if_missing
from utils.torch import get_scheduler

from visualization_scripts.visualization_helpers_af import _save_catch_all
from data.categorize_interactions import get_interaction_matrix_for_scene, INTERACTION_CAT_ABBRS
from data.ped_interactions import INTERACTION_CAT_NAMES
from data.preprocess_w_odometry import agents_to_robot_frame


def agg_metrics(all_metrics):
    """Aggregate metrics across sequences. from list of dicts to dict of lists"""
    d = {}
    for elem in all_metrics:
        for k, v in elem.items():
            if k not in d:
                d[k] = []
            d[k].append(v)
    return d

def save_trajectories(trajectory, save_dir, seq_name, frame, suffix=''):
    """Save trajectories in a text file.
    Input:
        trajectory: (np.array/torch.Tensor) Predcited trajectories with shape
                    of (n_pedestrian, future_timesteps, 4). The last elemen is
                    [frame_id, track_id, x, y] where each element is float.
        save_dir: (str) Directory to save into.
        seq_name: (str) Sequence name (e.g., eth_biwi, coupa_0)
        frame: (num) Frame ID.
        suffix: (str) Additional suffix to put into file name.
    """
    fname = f"{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt"
    mkdir_if_missing(fname)

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    np.savetxt(fname, trajectory, fmt="%.3f")


def df_to_array(df):
    """
    Convert DataFrame with x and y columns to (num_timesteps, num_peds, 2) array, filling in blanks with n/as.
    The index of the DataFrame should be (timestep, id), and timestep does not start at 0, and there may be a frame_skip.
    """
    # Ensure the DataFrame has the correct index and columns
    assert isinstance(df.index, pd.MultiIndex), "Index should be a MultiIndex with (timestep, id)"
    assert 'x' in df.columns and 'y' in df.columns or 'p' in df.columns, "DataFrame should have 'x' and 'y' columns or 'p'"

    # Get unique timesteps and pedestrian IDs
    unique_timesteps = df.index.get_level_values('timestep').unique()
    unique_ped_ids = df.index.get_level_values('id').unique()

    # Initialize an array with shape (num_timesteps, num_peds, 2) filled with np.nan
    num_timesteps = len(unique_timesteps)
    num_ped_ids = len(unique_ped_ids)
    result_array = np.full((num_timesteps, num_ped_ids, 2), np.nan)  # Use np.nan as a placeholder for missing values

    # Create mappings from timesteps and pedestrian IDs to array indices
    timestep_to_index = {timestep: i for i, timestep in enumerate(unique_timesteps)}
    ped_id_to_index = {ped_id: i for i, ped_id in enumerate(unique_ped_ids)}

    # Populate the array with values from the DataFrame
    for (timestep, ped_id), row in df.iterrows():
        timestep_index = timestep_to_index[timestep]
        ped_id_index = ped_id_to_index[ped_id]
        if 'x' in df.columns and 'y' in df.columns:
            result_array[timestep_index, ped_id_index] = [row['x'], row['y']]
        else:
            result_array[timestep_index, ped_id_index] = row['p'][:2]

    return result_array


def format_and_save_trajs(trajectory, data, future=True, save_name=None):
    """trajectory: (num_peds, timesteps, 2)"""
    num_agents, num_timesteps, _ = trajectory.shape
    formatted_trajectories = []
    if not future:
        trajectory = torch.flip(trajectory, [0, 1])

    # Initialize the agents_df dictionary
    agents_dict = {
            'timestep': [],
            'id': [],
            'p': [],
            'yaw': []
    }

    for ped_i, track_id in enumerate(data['valid_id']):
        for ts_i in range(num_timesteps):
            if future:
                curr_data = data['fut_data'][ts_i]['pos']
            else:
                curr_data = data['pre_data'][ts_i]['pos']
            # Get data with the same track_id
            updated_data = curr_data[curr_data[:, 1] == track_id].squeeze()[...,:4]
            formatted_trajectories.append(updated_data)
            agents_dict['timestep'].append(updated_data[0])
            agents_dict['id'].append(track_id)
            agents_dict['p'].append(np.concatenate([trajectory[ped_i, ts_i], [0]]))

    if not future:
        formatted_trajectories = np.flip(formatted_trajectories, axis=0)

    agents_dict['yaw'] = np.zeros_like(agents_dict['id'])
    agents_df = pd.DataFrame(agents_dict).set_index(['timestep', 'id'])
    agents_df = agents_to_robot_frame(agents_df, data['robot_data'])
    agents_df['x'] = agents_df['p'].apply(lambda x: x[0])
    agents_df['y'] = agents_df['p'].apply(lambda x: x[1])
    agents_df2 = agents_df[['x', 'y']]

    if save_name is not None:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        agents_df2.to_csv(save_name, sep=' ', header=False)

    return agents_df[['p', 'q']], agents_df2


def format_agentformer_trajectories(trajectory, data, cfg, timesteps=12, frame_scale=10, future=True):
    formatted_trajectories = []
    if not future:
        trajectory = torch.flip(trajectory, [0, 1])
    for i, track_id in enumerate(data['valid_id']):
        # don't print trajs that are not part of this scene
        if data['pred_mask'] is not None and data['pred_mask'][i] != 1.0:
            continue
        for j in range(timesteps):
            if future:
                curr_data = data['future_data'][j]['pos']
            else:
                curr_data = data['history_data'][j]['pos']
            # Get data with the same track_id
            updated_data = curr_data[curr_data[:, 1] == track_id].squeeze()
            if cfg.dataset in [
                    'eth', 'hotel', 'univ', 'zara1', 'zara2'
            ]:
                # [13, 15] correspoinds to the 2D position
                updated_data[[13, 15]] = trajectory[i, j].cpu().numpy()
            elif 'sdd' in cfg.dataset:
                updated_data[[2, 3]] = trajectory[i, j].cpu().numpy()
            else:
                updated_data[[2, 3]] = trajectory[i, j].cpu().numpy()
                # raise NotImplementedError()
            formatted_trajectories.append(updated_data)
    if len(formatted_trajectories) == 0:
        return np.array([])

    # Convert to numpy array and get [frame_id, track_id, x, y]
    formatted_trajectories = np.vstack(formatted_trajectories)
    if cfg.dataset in [ 'eth', 'hotel', 'univ', 'zara1', 'zara2' ]:
        formatted_trajectories = formatted_trajectories[:, [0, 1, 13, 15]]
        formatted_trajectories[:, 0] *= frame_scale
    elif cfg.dataset == 'trajnet_sdd':
        formatted_trajectories[:, 0] *= frame_scale
    else:
        formatted_trajectories[:, 0] *= frame_scale

    if not future:
        formatted_trajectories = np.flip(formatted_trajectories, axis=0)

    return formatted_trajectories


class AgentFormerTrainer(pl.LightningModule):
    def __init__(self, cfg, args):
        super().__init__()
        model_id = cfg.get('model_id', 'agentformer')
        self.model = model_dict[model_id](cfg)#, self)
        self.cfg = cfg
        self.args = args
        self.current_epoch_model = args.current_epoch_model
        num_workers = int(multiprocessing.cpu_count() / (args.devices + 1e-5)) if args.devices is not None else float('inf')
        self.num_workers = max(min(args.num_workers, num_workers), 1)
        self.batch_size = args.batch_size
        self.collision_rad = cfg.get('collision_rad', 0.1)
        # self.hparams.update(vars(cfg))
        # self.hparams.update({'args': vars(args)})
        self.model_name = "_".join(self.cfg.id.split("_")[1:])
        self.dataset_name = self.cfg.id.split("_")[1].replace('-', '_')
        self.log_train_this_time = False
        self.int_matrices = None
        # if self.args.trial:
        #     self.args.save_num = max(1, self.args.save_num // 10)

    def update_args(self, args):
        self.args = args

    def on_test_start(self):
        self.model.set_device(self.device)

    def on_fit_start(self):
        self.model.set_device(self.device)

    def _step(self, batch, mode):
        # Compute predictions
        # if self.current_epoch == 0 and self.global_step <= 5 and self.model.training:
        #     print(f"node rank: {torch.cuda.current_device()} step: {self.global_step}, frame: {batch['frame']}")

        # if torch.all(torch.stack(batch['pre_kp']) == 0):
        #     return

        if batch is None:
            return
        self.model.set_data(batch)
        data = self.model()
        total_loss, loss_dict, loss_unweighted_dict = self.model.compute_loss()

        # losses
        if mode != 'test':
            self.log(f'{mode}/loss', total_loss, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)
            for loss_name, loss in loss_dict.items():
                self.log(f'{mode}/{loss_name}', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)

        # make gt_motion and pred_motion both have peds first, for the sake of evaluation
        gt_motion = self.cfg.traj_scale * data['fut_motion'].cpu().transpose(1, 0)
        pred_motion = self.cfg.traj_scale * data[f'infer_dec_motion'].detach().cpu()
        obs_motion = self.cfg.traj_scale * data[f'pre_motion'].cpu()  # .transpose(1, 0).cpu()

        return {'loss': total_loss, 'frame': batch['frame'], 'seq': batch['seq'],
                'gt_motion': gt_motion, 'pred_motion': pred_motion, 'obs_motion': obs_motion, 'data': data}

    def training_step(self, batch, batch_idx):
        if self.args.tqdm_rate == 0 and batch_idx % 5 == 0:
            print(f"epoch: {self.current_epoch} batch: {batch_idx}")
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        if self.args.test_certain_frames_only and (batch['seq'], batch['frame']) not in self.args.peds:
            print(f"{(batch['seq'], batch['frame'])=}")
            return

        with torch.no_grad():
            return_dict = self._step(batch, 'test')
            if return_dict is None:
                return

        pred_motion = return_dict['pred_motion']  # (num_peds, num_samples, pred_steps, 2)
        gt_motion = return_dict['gt_motion'].transpose(0,1).cpu().numpy()  # (pred_steps, num_peds, 2)
        obs_motion = return_dict['obs_motion'].cpu().numpy()  # (obs_steps, num_peds, 2)

        if self.args.save_traj:
            if self.dataset_name == 'trajnet_sdd':
                save_dir = f'../trajectory_reward/results/trajectories/{self.model_name}/trajnet_sdd'
                frame = batch['frame'] * batch['frame_skip']
            elif self.dataset_name == 'jrdb':
                save_dir = f'../results_traj_preds/af_traj_preds/{self.model_name}/{batch["seq"]}'
                frame = batch['frame']
            else:
                raise NotImplementedError
            for idx, sample in enumerate(pred_motion.transpose(0, 1)):
                save_name = os.path.join(save_dir, f'{frame}_pred-{idx}.txt')
                format_and_save_trajs(sample, batch, True, save_name)

            # pred_traj = pred_motion.transpose(0, 2)[:, 0].cpu().numpy()
            #
            # plot_scene(obs_motion, gt_motion, pred_traj, ped_ids=np.arange(obs_motion.shape[1]), frame_id=9,
            #            save_fn='../viz/test_global.png')
            # agents_df, agents_df2 = format_and_save_trajs(pred_motion.transpose(0,1)[0].cpu().numpy(), batch, True, save_name)
            # new_pred_traj = df_to_array(agents_df2)
            # plot_scene(obs_motion, gt_motion, new_pred_traj, ped_ids=np.arange(obs_motion.shape[1]), frame_id=9,
            #            save_fn='../viz/test_robot.png')
            # back_to_global_agents_df = agents_to_odometry_frame(agents_df, batch['robot_data'])
            # new_pred_traj = df_to_array(back_to_global_agents_df)
            # print(f"{new_pred_traj.shape=}")
            # plot_scene(obs_motion, gt_motion, new_pred_traj, ped_ids=np.arange(obs_motion.shape[1]), frame_id=9,
            #            save_fn='../viz/test_back_global.png')
            #
            # import ipdb; ipdb.set_trace()

        return return_dict


    def _compute_and_log_metrics(self, outputs, mode='test'):
        args_list = [(output['pred_motion'].numpy(), output['gt_motion'].numpy(), None) for output in outputs]  # if output is not None
        # output['data']['fut_mask'].numpy()

        # calculate metrics for each sequence
        if self.args.mp:
            with multiprocessing.Pool(self.num_workers) as pool:
                all_metrics = pool.starmap(partial(eval_one_seq,
                                                   collision_rad=self.collision_rad,
                                                   return_sample_vals=self.args.save_viz), args_list)
        else:
            all_metrics = starmap(partial(eval_one_seq,
                                          collision_rad=self.collision_rad,
                                          return_sample_vals=self.args.save_viz), args_list)
        all_metrics, all_ped_vals, all_sample_vals, argmins, collision_mats = zip(*all_metrics)

        # aggregate metrics across sequences
        num_agent_per_seq = np.array([output['gt_motion'].shape[0] for output in outputs])
        total_num_agents = np.sum(num_agent_per_seq)
        results_dict = {}
        all_metrics = agg_metrics(all_metrics)
        for metric_name, results in all_metrics.items():
            if metric_name not in ['ADE_marginal', 'FDE_marginal', 'ADE_joint', 'FDE_joint', 'FDE_marginal_2s_agent', 'CR_mean']:
                continue
            if 'marginal' in metric_name or metric_name == "MR":
                value = np.sum(results * num_agent_per_seq) / total_num_agents
                results_dict[f"{metric_name}_agent"] = value
            else:
                value = np.mean(results)
                results_dict[metric_name] = value
            
        # mask should be [num_samples, num_agents, 1]

        # get stats related to collision_rejection sampling
        is_test_mode = mode == 'test'

        if not self.cfg.get('collisions_ok', True) and is_test_mode:
            tot_samples_w_col = np.sum([0 if output['num_samples_w_col'] is None
                                        else output['num_samples_w_col'][1] for output in outputs])
            tot_frames_w_col = np.sum([0 if output['num_samples_w_col'] is None else 1 for output in outputs])
            results_dict['tot_samples_w_col'] = tot_samples_w_col
            results_dict['tot_frames_w_col'] = tot_frames_w_col

        # save results to file
        if is_test_mode and self.args.save_test_results and not self.args.trial:
            test_results_filename = f'{self.args.logs_root}/test_results/{self.cfg.id}.tsv'
            mkdir_if_missing(test_results_filename)
            with open(test_results_filename, 'w') as f:
                f.write(f"{self.cfg.id}\n")
                # f.write(f"epoch\t{self.current_epoch}\n")
                f.write(f"epoch\t{self.current_epoch_model if self.current_epoch_model is not None else self.current_epoch}\n")
                metrics_to_print = {'ADE_marginal', 'FDE_marginal', 'CR_mean', 'ADE_joint', 'FDE_joint'}
                for metric_name, value in results_dict.items():
                    if metric_name not in metrics_to_print:
                        continue
                    f.write(f"{value:.4f}\n")
                f.write(f"total_peds\t{total_num_agents}")

            # save results broken down by interaction categories
            self.save_interaction_cat_results(outputs, all_ped_vals, total_num_agents)

        # print results to console for easy copy-and-paste
        if is_test_mode:
            print(f"\n\n\n{self.current_epoch}")
            for metric_name, value in results_dict.items():
                print(f"{metric_name}\t{value:.4f}")
            print(total_num_agents)

        # log metrics to tensorboard
        if not is_test_mode:
            # for metric_name, value in results_dict.items():
            for metric_name in ['ADE_marginal_agent', 'FDE_marginal_agent', 'ADE_joint', 'FDE_joint', 'FDE_marginal_2s_agent', 'CR_mean']:
                value = results_dict[metric_name]
                self.log(f'{mode}/{metric_name}', value, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f'{mode}/total_num_agents', float(total_num_agents), sync_dist=True, logger=True)
            self.log(f'{mode}/lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, logger=True)

        # save outputs for later visualization
        # self.outputs = outputs, all_sample_vals, all_metrics, argmins, collision_mats
        # torch.save((outputs, all_sample_vals, all_metrics, argmins, collision_mats), f'../viz/af/{self.cfg.id}_{mode}_outputs.pt')
        return outputs, all_sample_vals, all_metrics, argmins, collision_mats

    def log_viz(self, args, mode):
        outputs, all_sample_vals, all_metrics, argmins, collision_mats = args

        num_test_samples = len(outputs)

        skip = max(1, int(num_test_samples / self.args.save_num))

        if self.logger is None:
            anim_save_dir = f'../viz/af_jrdb_viz/'
        else:
            anim_save_dir = None
        all_figs = _save_catch_all(self, outputs[::skip], all_sample_vals[::skip], collision_mats[::skip], mode, anim_save_dir)

        # plot videos to tensorboard
        instance_is = np.arange(0, num_test_samples, skip)
        video_tensor = np.stack(all_figs).transpose(0, 1, 4, 2, 3)

        # (num_samples (bs), num_timesteps, channels (4), height, width)
        if self.args.wandb_project_name is not None and self.logger is not None:
            import wandb
            for idx, (instance_i, figs) in enumerate(zip(instance_is, all_figs)):
                self.logger.experiment.log({f'{mode}_viz/traj_{instance_i}': wandb.Video(video_tensor[idx], fps=2.5, format='gif')})
        elif self.logger is not None:  # tensorboard
            for idx, (instance_i, figs) in enumerate(zip(instance_is, all_figs)):
                self.logger.experiment.add_video(f'{mode}/traj_{instance_i}', video_tensor[idx:idx+1], self.global_step, fps=6)

    def training_epoch_end(self, outputs):
        self.outputs = self._compute_and_log_metrics(outputs, 'train')
        self.model.step_annealer()

    def validation_epoch_end(self, outputs):
        if len(outputs) > 0:
            self.outputs = self._compute_and_log_metrics(outputs, 'val')

    def test_epoch_end(self, outputs):
        outputs_processed = self._compute_and_log_metrics(outputs, 'test')
        if self.args.save_viz:
            self.log_viz(outputs_processed, 'test')

    def on_load_checkpoint(self, checkpoint):
        if 'model_dict' in checkpoint and 'epoch' in checkpoint:
            checkpoint['state_dict'] = {f'model.{k}': v for k, v in checkpoint['model_dict'].items()}
            checkpoint['global_step'] = None  # checkpoint['epoch'] * jb
            checkpoint['lr_schedulers'] = [checkpoint['scheduler_dict']]
            checkpoint['optimizer_states'] = [checkpoint['opt_dict']]
        print(f"EPOCH {checkpoint['epoch']}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)#, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.scheduler_step_size,
        #                                                  gamma=0.5)
        scheduler_type = self.cfg.get('lr_scheduler', 'linear')
        if scheduler_type == 'linear_ramp_up':
            scheduler = get_scheduler(optimizer, policy='linear_ramp_up', nepoch_fix=self.cfg.lr_fix_epochs, nepoch=self.cfg.num_epochs,
                                      decay_step=self.cfg.decay_step, decay_gamma=self.cfg.decay_gamma)
        elif scheduler_type == 'linear':
            scheduler = [get_scheduler(optimizer, policy='lambda', nepoch_fix=self.cfg.lr_fix_epochs, nepoch=self.cfg.num_epochs)]
        elif scheduler_type == 'step':
            scheduler = [get_scheduler(optimizer, policy='step', decay_step=self.cfg.decay_step, decay_gamma=self.cfg.decay_gamma)]
        else:
            raise ValueError('unknown scheduler type!')

        return [optimizer], scheduler

    def save_interaction_cat_results(self, outputs, all_ped_vals, total_num_agents):
        """
        Save the results of the interaction category breakdown analysis to two results files:
        results1: contains the performance of each category for each metric
        results2: contains the fraction of pedestrians in each category relative to the total number of pedestrians
        Args:
            outputs (list): List of output data.
            all_ped_vals (list): List of pedestrian values.
            total_num_agents (int): Total number of agents.
        Returns:
            None
        """
        results1 = {}
        results2 = {}
        save_dir = '../viz/af_tbd_int_cat_results'
        os.makedirs(save_dir, exist_ok=True)
        base = f'{self.cfg.dataset}_dataskip-{self.cfg.data_skip_train}-{self.cfg.data_skip_val}_frames-{self.cfg.past_frames}-{self.cfg.future_frames}_splittype-{self.cfg.split_type}'
        if self.args.interaction_category:
            if self.int_matrices is None:  # need to compute or load in
                int_matrices_path = f'{save_dir}/{base}.npy'
                if os.path.exists(int_matrices_path):  # load
                    self.int_matrices = int_matrices = np.load(int_matrices_path, allow_pickle=True).tolist()
                    # self.int_matrices = int_matrices = np.load(int_matrices_path, allow_pickle=True)['arr_0'].item()
                    # self.int_matrices = int_matrices = np.load(int_matrices_path, allow_pickle=True)['arr_0'].item()
                else: # compute
                    if self.args.mp: # compute in parallel
                        with multiprocessing.Pool(self.num_workers) as pool:
                            int_matrices = pool.map(get_interaction_matrix_for_scene,
                                                    [np.concatenate([output['obs_motion'],
                                                                     output['gt_motion'].swapaxes(0,1)], axis=0)[0]
                                                     for output in outputs])
                            self.int_matrices = int_matrices = {(output['seq'], output['frame']): mat
                                                                for output, mat in zip(outputs, int_matrices)}
                            np.save(int_matrices_path, int_matrices)
                    else:  # compute in serial
                        self.int_matrices = int_matrices = {(output['seq'], output['frame']):
                                                                get_interaction_matrix_for_scene(
                                                                        np.concatenate([output['obs_motion'],
                                                output['gt_motion'].swapaxes(0,1)], axis=0))[0] for output in outputs}
                        os.makedirs(os.path.dirname(int_matrices_path), exist_ok=True)
                        np.save(int_matrices_path, int_matrices)
            else: # already computed
                int_matrices = self.int_matrices
            # print('int_matrices', len(int_matrices), 'first 10 frames of int matrices', list(int_matrices.keys())[:10])
            assert len(outputs) == len(int_matrices), f"{len(outputs)} != {len(int_matrices)}"
            assert len(all_ped_vals) == len(int_matrices), f"{len(all_ped_vals)} != {len(int_matrices)}"

            # get performance for each category
            total_num_peds = 0
            for int_idx, int_cat in enumerate(INTERACTION_CAT_ABBRS):
                total_peds_this_cat = 0
                for scene_frame, mat in int_matrices.items():
                    total_peds_this_cat += mat[..., int_idx].sum()
                if total_peds_this_cat == 0:
                    continue
                int_name = INTERACTION_CAT_NAMES[int_cat]
                for metric_name in ['ADE_marginal', 'ADE_joint']:
                    value = 0
                    for (scene_frame, int_matrix), sequence_results in zip(int_matrices.items(), all_ped_vals):
                        value += np.sum(sequence_results[metric_name] * int_matrix[..., int_idx])
                    results1[(metric_name, int_name, 'pose+no_pose')] = value / total_peds_this_cat
                results2[(int_name, 'fraction of peds in this cat / total peds')] = total_peds_this_cat / total_num_agents
                results2[(int_name, 'total num peds')] = total_peds_this_cat
                total_peds_this_scene = np.sum([mat.shape[0] for mat in int_matrices.values()])
                total_num_peds += total_peds_this_scene
            results2[('agg', 'total num peds')] = total_num_peds
            # for metric_name in ['ADE_marginal', 'ADE_joint']:
                # results1[(metric_name, 'agg', 'pose+no_pose')] = np.sum(np.concatenate([results[metric_name] for results in all_ped_vals])) / total_num_peds
            results1[('ADE_marginal', 'agg', 'pose+no_pose')] = np.sum(np.concatenate([results['ADE_marginal'] for results in all_ped_vals])) / total_num_peds
            results1[('ADE_joint', 'agg', 'pose+no_pose')] = np.sum(np.concatenate([results['FDE_marginal'] for results in all_ped_vals])) / total_num_peds

        # aggregate by trajectories which have poses
        # (see if trajectories with poses in the observation perform better than those without)
        if 'joints' in self.cfg.id or 'kp' in self.cfg.id:
            # num_keypoints * obs_steps
            thresh = 10 * 8
            pose_masks = [torch.cat([output['data']['pre_kp_scores']]).sum(0).sum(-1).cpu().numpy() > thresh for output in outputs]
            # num_seq, (num_timesteps, num_peds, num_kp)
            total_pose_peds = np.sum([np.sum(mask) for mask in pose_masks])
            total_no_pose_peds = total_num_agents - total_pose_peds
            assert np.sum([np.sum(~m) for m in pose_masks]) == total_no_pose_peds
            for metric_name in ['ADE_marginal', 'ADE_joint']:
                value = 0
                for mask, results in zip(pose_masks, all_ped_vals):
                    value += np.sum(results[metric_name] * mask)
                results1[(metric_name, 'agg', 'pose')] = value / total_pose_peds
            for metric_name in ['ADE_marginal', 'ADE_joint']:
                value = 0
                for mask, results in zip(pose_masks, all_ped_vals):
                    value += np.sum(results[metric_name] * ~mask)
                results1[(metric_name,'agg','no_pose')] = value / total_no_pose_peds
            assert np.sum([len(m) for m in pose_masks]) == total_num_agents
            results2[('agg','fraction of pose peds / pose + no pose peds')] = total_pose_peds / total_num_agents

            if self.args.interaction_category:
                # get performance for each category
                for int_idx, int_cat in enumerate(INTERACTION_CAT_ABBRS):
                    total_peds_this_cat_pose = np.sum([(mat[..., int_idx] * mask).sum() for mask, mat in zip(pose_masks, int_matrices)])
                    total_peds_this_cat_no_pose = np.sum([np.sum(mat[..., int_idx] * ~mask) for mask, mat in zip(pose_masks, int_matrices)])
                    total_peds_this_cat = total_peds_this_cat_pose + total_peds_this_cat_no_pose
                    assert total_peds_this_cat == np.sum([np.sum(mat[..., int_idx]) for mat in int_matrices[:len(pose_masks)]])
                    if total_peds_this_cat_pose == 0 and total_peds_this_cat_no_pose == 0:
                        continue
                    int_name = INTERACTION_CAT_NAMES[int_cat]
                    # np.sum(num_agent_per_seq[int_matrix[..., int_idx]])
                    for metric_name in ['ADE_marginal', 'ADE_joint']:
                        value = 0
                        for mask, int_matrix, sequence_results in zip(pose_masks, int_matrices, all_ped_vals):
                            value += np.sum(sequence_results[metric_name] * int_matrix[..., int_idx] * mask)
                        results1[(metric_name,int_name,'pose')] = value / total_peds_this_cat_pose
                    for metric_name in ['ADE_marginal', 'ADE_joint']:
                        value = 0
                        for mask, int_matrix, sequence_results in zip(pose_masks, int_matrices, all_ped_vals):
                            value += np.sum(sequence_results[metric_name] * int_matrix[..., int_idx] * ~mask)
                        results1[(metric_name,int_name,'no_pose')] = value / total_peds_this_cat_no_pose
                    results2[(int_name, 'fraction of pose peds / pose + no pose peds')] = total_peds_this_cat_pose / total_peds_this_cat

        if len(results1) > 0:
            df = pd.DataFrame.from_dict(results1, orient='index')
            df.index = pd.MultiIndex.from_tuples(df.index)  # Convert index to MultiIndex
            df = pd.concat({self.cfg.id: df}).rename_axis(index=['method', 'metric', 'interaction_category', 'all'])
            # swap 1st and 2nd levels of the index of a non-hierarchical DataFrame
            df = df.swaplevel(0, 1, axis=0).unstack()
            df = df.swaplevel(1, 2, axis=0).unstack()
            # remove column level
            df.columns = df.columns.droplevel([0,1])
            # save each outer index to a separate file
            for outer_index in df.index.levels[0]:
                df_new = df.loc[outer_index]
                test_results_filename = f'{save_dir}/{outer_index}_{base}.tsv'
                if os.path.exists(test_results_filename):
                    df_old = pd.read_csv(test_results_filename, sep='\t', index_col=0)
                    # if column is in old df, replace
                    df_old = df_old.loc[:, ~df_old.columns.isin(df_new.columns)]
                    df_new = pd.concat([df_old, df_new], axis=1)
                df_new.to_csv(test_results_filename, sep='\t', float_format='%.3f', header=True)

        if len(results2) > 0:
            df2 = pd.DataFrame.from_dict(results2, orient='index')
            df2.index = pd.MultiIndex.from_tuples(df2.index)  # Convert index to MultiIndex
            df2 = df2.unstack()  # Unstack the MultiIndex
            test_results_filename = f'{save_dir}/{base}_breakdowns.tsv'
            df2.columns = df2.columns.droplevel(0)
            # if os.path.exists(test_results_filename):
            #     df_old = pd.read_csv(test_results_filename, sep='\t', index_col=[0, 1, 2])
            #     if not (df_old.shape == df2.shape and np.allclose(df_old.to_numpy(), df2.to_numpy())):
            #         import ipdb; ipdb.set_trace()
            #     if (df_old.shape == df2.shape and np.allclose(df_old.to_numpy(), df2.to_numpy())):
            #         df2 = pd.concat([df_old, df2])
            # test_results_filename = f'{save_dir}/{self.cfg.id}_interaction_category_totals.tsv'
            # save in float format for the first column, but not the rest
            df2[df2.columns[0]] = df2[df2.columns[0]].apply(lambda x: '{:.3f}'.format(x) if not pd.isna(x) else "")
            for col in df2.columns[1:]:
                df2[col] = df2[col].apply(lambda x: '{:.0f}'.format(x) if not pd.isna(x) else "")

            df2.to_csv(test_results_filename, sep='\t', header=True, index=True)
            print(f"wrote test results to {test_results_filename}")
