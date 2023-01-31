import os
# import tracemalloc
import torch
import pytorch_lightning as pl

from torch.autograd import Variable
import numpy as np
from itertools import starmap
from functools import partial
from eval import eval_one_seq2
from metrics import stats_func
import multiprocessing
from model.model_lib import model_dict
from utils.torch import get_scheduler

from viz_utils import plot_fig, get_metrics_str
from metrics import check_collision_per_sample_no_gt
from utils.utils import mkdir_if_missing, print_log


def run_model_w_col_rej(data, model, traj_scale, sample_k, collision_rad, device):
    """run model with collision rejection"""
    samples_to_return = torch.empty(0).to(device)
    num_tries = 0
    num_zeros = 0
    MAX_NUM_ZEROS = 3
    MAX_NUM_TRIES = 10
    MAX_NUM_SAMPLES = 300
    NUM_SAMPLES_PER_FORWARD = 30
    samples_w_cols = None
    while samples_to_return.shape[0] < sample_k:
        with torch.no_grad():
            model.set_data(data)
            sample_motion_3D = model.inference(mode='infer', sample_num=NUM_SAMPLES_PER_FORWARD,
                                               need_weights=False)[0].transpose(0, 1).contiguous()
            num_tries += 1
        sample_motion_3D *= traj_scale

        # compute number of colliding samples
        pred_arr = sample_motion_3D.cpu().numpy()
        num_peds = pred_arr.shape[1]
        if num_peds == 1:  # if there's only one ped, there are necessarily no collisions
            samples_to_return = sample_motion_3D[:sample_k]
            break
        # compute collisions in parallel
        with multiprocessing.Pool(processes=min(NUM_SAMPLES_PER_FORWARD, multiprocessing.cpu_count())) as pool:
            mask = pool.map(partial(check_collision_per_sample_no_gt, ped_radius=collision_rad), pred_arr)
            # no_mp alternative:
            # mask = itertools.starmap(partial(check_collision_per_sample_no_gt, ped_radius=collision_rad), pred_arr)
            # mask contains list of length num_samples of tuples of length 2
            # (collision_per_ped_array (num_peds), collision_matrix_per_timestep (pred_steps, num_peds, num_peds))
        # get indices of samples that have 0 collisions
        maskk = np.where(~np.any(np.array(list(zip(*mask))[0]).astype(np.bool), axis=-1))[0]
        if maskk.shape[0] == 0:  # if there are no samples with 0 collisions
            num_zeros += 1
            if num_tries * NUM_SAMPLES_PER_FORWARD >= MAX_NUM_SAMPLES:
                # if num_zeros > MAX_NUM_ZEROS or num_tries > MAX_NUM_TRIES:
                print(f"frame {data['frame']} with {len(data['pre_motion_3D'])} peds: "
                  f"collected {num_tries * NUM_SAMPLES_PER_FORWARD} samples, only {samples_to_return.shape[0]} non-colliding. \n")
                samples_w_cols = num_peds, sample_k - samples_to_return.shape[0]
                samples_to_return = torch.cat([samples_to_return, sample_motion_3D])[:sample_k]  # append some colliding samples to the end
                break
            continue
        # append new non-colliding samples to list
        # at_least_1_col = np.any([np.any([np.any(ped) for ped in sample]) for sample in mask])
        # if at_least_1_col:
        #     print(f"Seq {data['seq']} frame {data['frame']} with {len(data['pre_motion_3D'])} peds has {maskk.shape[0]} non-colliding samples in 50 samples")
        non_collide_idx = torch.LongTensor(maskk)
        assert torch.max(non_collide_idx) < sample_motion_3D.shape[0]
        assert 0 <= torch.max(non_collide_idx)
        sample_motion_3D_non_colliding = torch.index_select(sample_motion_3D, 0, non_collide_idx.to(device))  # select only those in current sample who don't collide
        samples_to_return = torch.cat([samples_to_return, sample_motion_3D_non_colliding])[:sample_k]

    if samples_to_return.shape[0] == 0:  # should not get here
        print("should not get here")
        import ipdb; ipdb.set_trace()
    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * traj_scale
    samples_to_return = samples_to_return.transpose(0, 1)
    return samples_to_return.cpu(), gt_motion_3D.cpu(), samples_w_cols


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
        if data['pred_mask'] is not None and data['pred_mask'][i] != 1.0:
            continue
        for j in range(timesteps):
            if future:
                curr_data = data['fut_data'][j]
            else:
                curr_data = data['pre_data'][j]
            # Get data with the same track_id
            updated_data = curr_data[curr_data[:, 1] == track_id].squeeze()
            if cfg.dataset in [
                    'eth', 'hotel', 'univ', 'zara1', 'zara2', 'gen',
                    'real_gen', 'adversarial'
            ]:
                # [13, 15] correspoinds to the 2D position
                updated_data[[13, 15]] = trajectory[i, j].cpu().numpy()
            elif 'sdd' in cfg.dataset:
                updated_data[[2, 3]] = trajectory[i, j].cpu().numpy()
            else:
                raise NotImplementedError()
            formatted_trajectories.append(updated_data)
    if len(formatted_trajectories) == 0:
        return np.array([])

    # Convert to numpy array and get [frame_id, track_id, x, y]
    formatted_trajectories = np.vstack(formatted_trajectories)
    if cfg.dataset in [
            'eth', 'hotel', 'univ', 'zara1', 'zara2', 'gen', 'real_gen',
            'adversarial'
    ]:
        formatted_trajectories = formatted_trajectories[:, [0, 1, 13, 15]]
        formatted_trajectories[:, 0] *= frame_scale
    elif cfg.dataset == 'trajnet_sdd':
        formatted_trajectories[:, 0] *= frame_scale

    if not future:
        formatted_trajectories = np.flip(formatted_trajectories, axis=0)

    return formatted_trajectories


class AgentFormerTrainer(pl.LightningModule):
    def __init__(self, cfg, args):
        super().__init__()
        model_id = cfg.get('model_id', 'agentformer')
        self.model = model_dict[model_id](cfg)
        self.cfg = cfg
        self.args = args
        # self.traj_loss = torch.tensor([0], requires_grad=True)  #.to(data['pre_motion'].device)
        # self.traj_loss = Variable(torch.Tensor([0]), requires_grad=True)  #.to(data['pre_motion'].device)
        num_workers = int(multiprocessing.cpu_count() / (args.devices + 1e-5)) if args.devices is not None else float('inf')
        self.num_workers = min(args.num_workers, num_workers)
        self.batch_size = args.batch_size
        self.collision_rad = cfg.get('collision_rad', 0.1)
        self.hparams.update(vars(cfg))
        self.hparams.update(vars(args))

    def set_example_input_array(self, example_input_array):
        data = self.model.get_torch_data(example_input_array)
        # print("data:", data.keys())
        # dic = {k:v for k, v in data.items() if type(v) == torch.Tensor}
        # print("dic:", dic.keys())
        # import ipdb; ipdb.set_trace()
        self.example_input_array = [data]  # [self.model.set_data(example_input_array)]

    def on_test_start(self):
        self.model.set_device(self.device)

    def on_fit_start(self):
        self.model.set_device(self.device)

    def forward(self, batch):
        self.model.set_data(batch)
        return self.model()

    def _step(self, batch, mode):
        # Compute predictions
        # data = self(batch)
        self.model.set_data(batch)
        # tracemalloc.start()
        data = self.model()
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        #
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)
        # import ipdb; ipdb.set_trace()
        total_loss, loss_dict, loss_unweighted_dict = self.model.compute_loss()

        # losses
        self.log(f'{mode}/loss', total_loss, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)
        for loss_name, loss in loss_dict.items():
            self.log(f'{mode}/{loss_name}', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)

        return data, {'loss': total_loss, **loss_dict}

    def training_step(self, batch, batch_idx):
        if self.args.tqdm_rate == 0 and batch_idx % 5 == 0:
            print(f"epoch: {self.current_epoch} batch: {batch_idx}")
        data, loss_dict = self._step(batch, 'train')

        gt_motion = self.cfg.traj_scale * data['fut_motion'].transpose(1, 0).cpu()
        pred_motion = self.cfg.traj_scale * data[f'infer_dec_motion'].detach().cpu()
        obs_motion = self.cfg.traj_scale * data[f'pre_motion'].cpu()
        return {**loss_dict, 'frame': batch['frame'], 'seq': batch['seq'],
                'gt_motion': gt_motion, 'pred_motion': pred_motion, 'obs_motion': obs_motion, 'data': data}

    def validation_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'test')
        gt_motion = self.cfg.traj_scale * data['fut_motion'].transpose(1, 0).cpu()
        pred_motion = self.cfg.traj_scale * data[f'infer_dec_motion'].detach().cpu()
        obs_motion = self.cfg.traj_scale * data[f'pre_motion'].cpu()#.transpose(1, 0).cpu()
        # self.stats.update(gt_motion, pred_motion)
        return {**loss_dict, 'frame': batch['frame'], 'seq': batch['seq'],
                'gt_motion': gt_motion, 'pred_motion': pred_motion, 'obs_motion': obs_motion, 'data': data}

    def test_step(self, batch, batch_idx):
        if self.cfg.get('collisions_ok', True):
            return_dict = self.validation_step(batch, batch_idx)
            pred_motion, gt_motion, obs_motion = return_dict['pred_motion'], return_dict['gt_motion'], return_dict['obs_motion']
        else:
            pred_motion, gt_motion, num_samples_w_col = run_model_w_col_rej(batch, self.model, self.cfg.traj_scale,
                                                                            self.cfg.sample_k, self.cfg.collision_rad,
                                                                            self.model.device)
            obs_motion = self.cfg.traj_scale * torch.stack(batch[f'pre_motion_3D'], dim=1).cpu()
            return_dict = {'frame': batch['frame'], 'seq': batch['seq'], 'gt_motion': gt_motion, 'pred_motion':
                pred_motion, 'obs_motion': obs_motion, "num_samples_w_col": num_samples_w_col}

        if self.args.save_traj:
            # save_dir = './trajectories'
            # save_dir = './trajectories_optimized'
            save_dir = f'../trajectory_reward/results/trajectories/{"_".join(self.args.cfg.split("_")[1:])}'
            frame = batch['frame'] * 10
            for idx, sample in enumerate(pred_motion.transpose(0,1)):
                formatted = format_agentformer_trajectories(sample, batch, self.cfg, timesteps=12, frame_scale=10, future=True)
                save_trajectories(formatted, save_dir, batch['seq'], frame, suffix=f"/sample_{idx:03d}")
            formatted = format_agentformer_trajectories(gt_motion, batch, self.cfg, timesteps=12, frame_scale=10, future=True)
            save_trajectories(formatted, save_dir, batch['seq'], frame, suffix='/gt')
            formatted = format_agentformer_trajectories(obs_motion.transpose(0,1), batch, self.cfg, timesteps=8, frame_scale=10, future=False)
            save_trajectories(formatted, save_dir, batch['seq'], frame, suffix="/obs")

        return return_dict

    def _epoch_end(self, outputs, mode='test'):
        # datas = []
        # for i, output in enumerate(outputs):
        #     self.traj_loss[:] = 0
        #     self.traj_loss.zero_grad()
        #     self.traj_loss = Variable(torch.Tensor([0]), requires_grad=True)  # .to(data['pre_motion'].device)
        #     data = self.model.optimize_traj(self.traj_loss, output['data'])
        #     datas.append(data)
        # [self.model.optimize_traj(output['data']) for output in outputs]
        # new_outputs = [{**output, 'pred_motion': data['infer_dec_motion']} for output, data in zip(outputs, datas)]
        # pred_motion: num_peds, num_samples, ts, 2  # gt_motion: num_peds, ts, 2
        args_list = [(output['pred_motion'].numpy(), output['gt_motion'].numpy()) for output in outputs]
        # args_list_new = [(output['pred_motion'].numpy(), output['gt_motion'].numpy()) for output in new_outputs]
        if self.args.mp:
            with multiprocessing.Pool(self.num_workers) as pool:
                all_metrics = pool.starmap(partial(eval_one_seq2,
                                                   collision_rad=self.collision_rad,
                                                   return_sample_vals=self.args.save_viz), args_list)
        #     with multiprocessing.Pool(self.num_workers) as pool:
        #         all_metrics_new = pool.starmap(partial(eval_one_seq2,
        #                                                collision_rad=self.collision_rad,
        #                                                return_sample_vals=self.args.save_viz), args_list_new)
        else:
            all_metrics = starmap(partial(eval_one_seq2,
                                          collision_rad=self.collision_rad,
                                          return_sample_vals=self.args.save_viz), args_list)
            # all_metrics_new = starmap(partial(eval_one_seq2,
            #                                   collision_rad=self.collision_rad,
            #                                   return_sample_vals=self.args.save_viz), args_list_new)

        all_metrics, all_sample_vals, argmins, collision_mats = zip(*all_metrics)
        # all_metrics_new, all_sample_vals_new, argmins_new, collision_mats_new = zip(*all_metrics_new)

        num_agent_per_seq = np.array([output['gt_motion'].shape[0] for output in outputs])
        total_num_agents = np.sum(num_agent_per_seq)
        results_dict = {}
        for key, values in zip(stats_func.keys(), zip(*all_metrics)):
            if '_joint' in key or 'CR' in key:  # sequence-based metric
                value = np.mean(values)
            else:  # agent-based metric
                value = np.sum(values * num_agent_per_seq) / np.sum(num_agent_per_seq)
            results_dict[key] = value

        # stats related to collision_rejection sampling
        if not self.cfg.get('collisions_ok', True):
            tot_samples_w_col = np.sum([0 if output['num_samples_w_col'] is None
                                        else output['num_samples_w_col'][1] for output in outputs])
            tot_frames_w_col = np.sum([0 if output['num_samples_w_col'] is None else 1 for output in outputs])
            results_dict['tot_samples_w_col'] = tot_samples_w_col
            results_dict['tot_frames_w_col'] = tot_frames_w_col

        # log and print results
        is_test_mode = mode == 'test'
        frames_w_cols_filename = f'../trajectory_reward/results/trajectories/test_results/{self.args.cfg}.tsv'
        mkdir_if_missing(frames_w_cols_filename)

        if is_test_mode and self.args.save_test_results:
            with open(frames_w_cols_filename, 'w') as f:
                with open(os.path.join(self.args.default_root_dir, f'test_results.tsv'), 'w') as g:
                    f.write(f"epoch\t{self.current_epoch}\n")
                    g.write(f"epoch\t{self.current_epoch}\n")
                    for key, value in results_dict.items():
                        f.write(f"{key}\t{value:.4f}\n")
                        g.write(f"{key}\t{value:.4f}\n")
                    f.write(f"total_peds\t{total_num_agents}")
                    g.write(f"total_peds\t{total_num_agents}")

        if is_test_mode:
            print(f"\n\n\n{self.current_epoch}")
        self.log(f'val/total_num_agents', float(total_num_agents), sync_dist=True, logger=True)
        for key, value in results_dict.items():
            if is_test_mode:
                print(f"{value:.4f}")
            self.log(f'val/{key}', value, sync_dist=True, prog_bar=True, logger=True)
        if is_test_mode:
            print(total_num_agents)

        if not self.cfg.get('collisions_ok', True):
            idxs_to_plot = [i for i, output in enumerate(outputs) if output['num_samples_w_col'] is not None]
            # save the frame numbers of the scenes with collisions, label with the number of samples with collisions
            frames = np.array([[outputs[i]['seq'], outputs[i]['frame'], *outputs[i]['num_samples_w_col']] for i in idxs_to_plot])
            frames_w_cols_filename = os.path.join(self.args.logs_root, 'test_results', f'colliding_frame_nums_{self.args.cfg}.tsv')
            np.savetxt(frames_w_cols_filename, frames, fmt='%s')

        # plot if there are collisions; or if args.save_viz and in test_mode
        if not self.cfg.get('collisions_ok', True) and self.args.save_viz and is_test_mode and len(idxs_to_plot) > 0:  # plot only certain scenes
            self._save_viz(*zip(*[(outputs[i], all_sample_vals[i], all_metrics[i], argmins[i], collision_mats[i])
                                  for i in idxs_to_plot]), mode)
        # elif self.args.save_viz and is_test_mode:
        #     self._save_viz(outputs[:self.args.save_num], all_sample_vals[:self.args.save_num],
        #                    all_metrics[:self.args.save_num], argmins[:self.args.save_num],
        #                    collision_mats[:self.args.save_num], mode)
        # elif self.args.save_viz and (self.args.test and self.current_epoch % 1 == 0 or not self.args.test):
        #     self._save_viz(outputs[:self.args.save_num], all_sample_vals[:self.args.save_num],
        #                    all_metrics[:self.args.save_num], argmins[:self.args.save_num],
        #                    collision_mats[:self.args.save_num], mode)

    def _save_viz(self, outputs, all_sample_vals, all_meters_values, argmins, collision_mats, tag=''):
        seq_to_plot_args = []
        for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
            frame = output['frame']
            seq = output['seq']
            obs_traj = output['obs_motion'].numpy()
            assert obs_traj.shape[0] == 8
            pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
            pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)  # (samples, ts, n_peds, 2)

            num_samples, _, n_ped, _ = pred_fake_traj.shape

            anim_save_fn = f'viz/{self.args.default_root_dir.replace("/", "--")}/epoch-{self.current_epoch}_{seq}' \
                           f'_frame-{frame}_{tag}.mp4'
            mkdir_if_missing(anim_save_fn)
            plot_args_list = [anim_save_fn, f"Seq: {seq} frame: {frame} Epoch: {self.current_epoch}", (5, 4)]

            pred_fake_traj_min = pred_fake_traj[argmins[frame_i],:,np.arange(n_ped)].swapaxes(0, 1)  # (n_ped, )
            # import ipdb; ipdb.set_trace()
            # assert len(pred_fake_traj_min.shape) == 3
            min_ADE_stats = get_metrics_str(dict(zip(stats_func.keys(), all_meters_values[frame_i])))
            args_dict = {'plot_title': f"best mADE sample",
                         'obs_traj': obs_traj,
                         'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_fake_traj_min,
                         'collision_mats': collision_mats[frame_i][-1],
                         'text_fixed': min_ADE_stats}
            plot_args_list.append(args_dict)

            for sample_i in range(num_samples - 1):
                stats = get_metrics_str(seq_to_sample_metrics, sample_i)
                args_dict = {'plot_title': f"Sample {sample_i}",
                             'obs_traj': obs_traj,
                             'pred_traj_gt': pred_gt_traj,
                             'pred_traj_fake': pred_fake_traj[sample_i],
                             'text_fixed': stats,
                             'highlight_peds': argmins[frame_i],
                             'collision_mats': collision_mats[frame_i][sample_i]}
                plot_args_list.append(args_dict)
            seq_to_plot_args.append(plot_args_list)

        if self.args.mp:
            with multiprocessing.Pool(self.num_workers) as pool:
                pool.starmap(plot_fig, seq_to_plot_args)
        else:
            list(starmap(plot_fig, seq_to_plot_args))

    def train_epoch_end(self, outputs):
        self._epoch_end(outputs, 'train')
        self.model.step_annealer()

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs)

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
