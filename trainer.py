import os
from itertools import starmap
from functools import partial
import multiprocessing
import numpy as np
import torch
import pytorch_lightning as pl

from model.model_lib import model_dict
from eval import eval_one_seq
from metrics import stats_func
from utils.utils import mkdir_if_missing
from utils.torch import get_scheduler

from viz_utils_plot import _save_catch_all, _save_viz_w_pose_3d, _save_viz_w_pose_2d, _save_viz_w_heading, _save_viz_nuscenes, _save_viz
from collision_rejection import run_model_w_col_rej
from data.categorize_interactions import get_interaction_matrix_for_scene, INTERACTION_CAT_ABBRS
from data.ped_interactions import INTERACTION_CAT_NAMES


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
        self.num_workers = min(args.num_workers, num_workers)
        self.batch_size = args.batch_size
        self.collision_rad = cfg.get('collision_rad', 0.1)
        self.hparams.update(vars(cfg))
        self.hparams.update(vars(args))
        self.model_name = "_".join(self.cfg.id.split("_")[1:])
        self.dataset_name = self.cfg.id.split("_")[0].replace('-', '_')
        self.int_matrices = None
        if self.args.trial:
            self.args.save_num = max(1, self.args.save_num // 10)

    def update_args(self, args):
        self.args = args

    def on_test_start(self):
        self.model.set_device(self.device)

    def on_fit_start(self):
        self.model.set_device(self.device)

    def _step(self, batch, mode):
        # Compute predictions
        # data = self(batch)

        # print(f"node rank: {torch.cuda.current_device()} step: {self.global_step}, frame: {batch['frame']}")

        if batch is None:
            return
        self.model.set_data(batch)
        data = self.model()
        total_loss, loss_dict, loss_unweighted_dict = self.model.compute_loss()

        # losses
        self.log(f'{mode}/loss', total_loss, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)
        for loss_name, loss in loss_dict.items():
            self.log(f'{mode}/{loss_name}', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)

        # make gt_motion and pred_motion both have peds first, for the sake of evaluation
        gt_motion = self.cfg.traj_scale * data['fut_motion'].cpu().transpose(1, 0)
        pred_motion = self.cfg.traj_scale * data[f'infer_dec_motion'].detach().cpu()
        obs_motion = self.cfg.traj_scale * data[f'pre_motion'].cpu()  # .transpose(1, 0).cpu()

        return {'loss': total_loss, **loss_dict, 'frame': batch['frame'], 'seq': batch['seq'],
                'gt_motion': gt_motion, 'pred_motion': pred_motion, 'obs_motion': obs_motion, 'data': data}

    def training_step(self, batch, batch_idx):
        if self.args.tqdm_rate == 0 and batch_idx % 5 == 0:
            print(f"epoch: {self.current_epoch} batch: {batch_idx}")
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.cfg.get('collisions_ok', True):
                return_dict = self._step(batch, 'test')
            else:
                return_dict = run_model_w_col_rej(batch, self.model, self.cfg.traj_scale, self.cfg.sample_k,
                                                  self.cfg.collision_rad, self.model.device)
        pred_motion = return_dict['pred_motion']  # (num_peds, num_samples, pred_steps, 2)
        gt_motion = return_dict['gt_motion']  # (pred_steps, num_peds, 2)
        obs_motion = return_dict['obs_motion']  # (obs_steps, num_peds, 2)

        if self.args.save_traj:
            if self.dataset_name == 'trajnet_sdd':
                save_dir = f'../trajectory_reward/results/trajectories/{self.model_name}/trajnet_sdd'
                frame = batch['frame'] * batch['frame_skip']
            elif self.dataset_name == 'jrdb':
                save_dir = f'trajectories/{self.model_name}'
                frame = batch['frame']
            else:
                save_dir = f'../trajectory_reward/results/trajectories/{self.model_name}/trajnet_sdd'
                frame = batch['frame'] * batch['frame_skip']
            for idx, sample in enumerate(pred_motion.transpose(0, 1)):
                formatted = format_agentformer_trajectories(sample, batch, self.cfg, timesteps=12,
                                                            frame_scale=batch['frame_scale'], future=True)
                save_trajectories(formatted, save_dir, batch['seq'], frame, suffix=f"/sample_{idx:03d}")
            formatted = format_agentformer_trajectories(gt_motion, batch, self.cfg, timesteps=12,
                                                        frame_scale=batch['frame_scale'], future=True)
            save_trajectories(formatted, save_dir, batch['seq'], frame, suffix='/gt')
            formatted = format_agentformer_trajectories(obs_motion.transpose(0, 1), batch, self.cfg, timesteps=8,
                                                        frame_scale=batch['frame_scale'], future=False)
            save_trajectories(formatted, save_dir, batch['seq'], frame, suffix="/obs")

        return return_dict


    def _compute_and_log_metrics(self, outputs, mode='test'):
        # save outputs
        # torch.save(f'../viz/af/{self.cfg.id}_{mode}_outputs.npy', outputs)

        args_list = [(output['pred_motion'].numpy(), output['gt_motion'].numpy()) for output in outputs]  #  if output is not None

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
        for metric_name, results in zip(stats_func.keys(), zip(*all_metrics)):
            value = np.mean(results)
            results_dict[metric_name] = value
            if 'marginal' in metric_name or metric_name == "MR":
                value = np.sum(results * num_agent_per_seq) / np.sum(num_agent_per_seq)
                results_dict[f"{metric_name}_agent"] = value
        # mask should be [num_samples, num_agents, 1]

        # save results broken down by interaction categories
        self.save_interaction_cat_results(outputs, all_ped_vals, total_num_agents)

        # get stats related to collision_rejection sampling
        is_test_mode = mode == 'test'

        if not self.cfg.get('collisions_ok', True) and is_test_mode:
            tot_samples_w_col = np.sum([0 if output['num_samples_w_col'] is None
                                        else output['num_samples_w_col'][1] for output in outputs])
            tot_frames_w_col = np.sum([0 if output['num_samples_w_col'] is None else 1 for output in outputs])
            results_dict['tot_samples_w_col'] = tot_samples_w_col
            results_dict['tot_frames_w_col'] = tot_frames_w_col

        # save results to file
        if True or is_test_mode and self.args.save_test_results and not self.args.trial:
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

        # print results to console for easy copy-and-paste
        if is_test_mode:
            print(f"\n\n\n{self.current_epoch}")
            for metric_name, value in results_dict.items():
                print(f"{metric_name}\t{value:.4f}")
            print(total_num_agents)

        # log metrics to tensorboard
        for metric_name, value in results_dict.items():
            self.log(f'{mode}/{metric_name}', value, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{mode}/total_num_agents', float(total_num_agents), sync_dist=True, logger=True)

        # save outputs for later visualization
        self.outputs = outputs, all_sample_vals, all_metrics, argmins, collision_mats

    def log_viz(self, args, mode):
        outputs, all_sample_vals, all_metrics, argmins, collision_mats = args

        num_test_samples = len(outputs)

        skip = max(1, int(num_test_samples / self.args.save_num))
        all_figs = _save_catch_all(self, outputs[::skip], all_sample_vals[::skip], collision_mats[::skip], mode)
        # if 'nuscenes' in self.cfg.id:
        #     all_figs = _save_viz_nuscenes(self, outputs[::skip], all_sample_vals[::skip], collision_mats[::skip], mode)
        # elif (('joints' in self.cfg.id or 'kp' in self.cfg.id)
        #       and self.model.context_encoder.kp_dim == 3):
        #     all_figs = _save_viz_w_pose_3d(self, outputs[::skip], all_sample_vals[::skip], collision_mats[::skip], mode)
        # elif (('joints' in self.cfg.id or 'kp' in self.cfg.id)
        #       and hasattr(self.model.context_encoder, 'kp_dim')
        #       and self.model.context_encoder.kp_dim == 2):
        #     all_figs = _save_viz_w_pose_2d(self, outputs[::skip], all_sample_vals[::skip], collision_mats[::skip], mode)
        # elif 'heading' in self.model.input_type:
        #     all_figs = _save_viz_w_heading(self, outputs[::skip], all_sample_vals[::skip], collision_mats[::skip], mode)
        # else:
        #     all_figs = _save_viz(self, outputs[::skip], all_sample_vals[::skip], all_metrics[::skip], argmins[::skip], collision_mats[::skip], mode)

        # plot videos to tensorboard
        instance_is = np.arange(0, num_test_samples, skip)
        for idx, (instance_i, figs) in enumerate(zip(instance_is, all_figs)):
            video_tensor = np.stack(all_figs).transpose(0, 1, 4, 2, 3)
            self.logger.experiment.add_video(f'{mode}/traj_{instance_i}', video_tensor[idx:idx+1], self.global_step, fps=6)

    def training_epoch_end(self, outputs):
        self._compute_and_log_metrics(outputs, 'train')
        self.model.step_annealer()

    def validation_epoch_end(self, outputs):
        self.val_outputs = outputs
        if len(outputs) > 0:
            self._compute_and_log_metrics(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._compute_and_log_metrics(outputs)

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
        if scheduler_type == 'linear':
            scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=self.cfg.lr_fix_epochs, nepoch=self.cfg.num_epochs)
        elif scheduler_type == 'step':
            scheduler = get_scheduler(optimizer, policy='step', decay_step=self.cfg.decay_step, decay_gamma=self.cfg.decay_gamma)
        else:
            raise ValueError('unknown scheduler type!')

        return [optimizer], [scheduler]

    def save_interaction_cat_results(self, outputs, all_ped_vals, total_num_agents):
        # aggregate by interaction category
        results1 = {}
        results2 = {}
        if self.args.interaction_category:
            if self.int_matrices is None:
                # load
                if os.path.exists(f'../viz/af/{self.cfg.id}_int_matrices.npy'):
                    self.int_matrices = int_matrices = np.load(f'../viz/af/{self.cfg.id}_int_matrices.npy', allow_pickle=True).tolist()
                else:
                    if self.args.mp:
                        with multiprocessing.Pool(self.num_workers) as pool:
                            int_matrices = pool.map(get_interaction_matrix_for_scene,
                                                    [np.concatenate([output['obs_motion'],
                                                                     output['gt_motion'].swapaxes(0,1)], axis=0)
                                                     for output in outputs])
                            self.int_matrices = int_matrices = [mat[0] for mat in int_matrices]
                            np.save(f'../viz/af/{self.cfg.id}_int_matrices.npy', int_matrices)
                    else:
                        int_matrices = [get_interaction_matrix_for_scene(
                            np.concatenate([output['obs_motion'], output['gt_motion'].swapaxes(0,1)], axis=0))[0]
                            for output in outputs]
            else:
                int_matrices = self.int_matrices

            # get performance for each category
            for int_idx, int_cat in enumerate(INTERACTION_CAT_ABBRS):
                total_peds_this_cat = np.sum([mat[..., int_idx].sum() for mat in int_matrices[:len(all_ped_vals)]])
                if total_peds_this_cat == 0:
                    continue
                int_name = INTERACTION_CAT_NAMES[int_cat]
                # np.sum(num_agent_per_seq[int_matrix[..., int_idx]])
                for metric_name in ['ADE_marginal', 'ADE_joint']:
                    #['ADE_marginal', 'FDE_marginal', 'CR_mean', 'ADE_joint', 'FDE_joint', 'CR_mADE', 'CR_mADEjoint']:
                    value = 0
                    for int_matrix, sequence_results in zip(int_matrices, all_ped_vals):
                        value += np.sum(sequence_results[metric_name] * int_matrix[..., int_idx])
                    results1[(metric_name, int_name, 'pose+no_pose')] = value / total_peds_this_cat
                results2[(int_name, 'fraction of this cat peds / total peds')] = total_peds_this_cat / total_num_agents

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

        import pandas as pd
        df = pd.DataFrame.from_dict(results1, orient='index')
        df.index = pd.MultiIndex.from_tuples(df.index)  # Convert index to MultiIndex
        df = df.unstack()  # Unstack the MultiIndex
        mkdir_if_missing(f'../viz/af')
        test_results_filename = f'../viz/af/{self.cfg.id}_cats.tsv'
        df.to_csv(test_results_filename, sep='\t', float_format='%.3f', header=True)

        df2 = pd.DataFrame.from_dict(results2, orient='index')
        df2.index = pd.MultiIndex.from_tuples(df2.index)  # Convert index to MultiIndex
        df2 = df2.unstack()  # Unstack the MultiIndex
        test_results_filename = f'../viz/af/{self.cfg.id}_totals.tsv'
        df2.to_csv(test_results_filename, sep='\t', float_format='%.3f', header=True)
        print(f"wrote test results to {test_results_filename}")
