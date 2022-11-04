import torch
import pytorch_lightning as pl
import numpy as np
from itertools import starmap
from functools import partial
from eval import eval_one_seq2
from metrics import stats_func
import multiprocessing
from model.model_lib import model_dict
from utils.torch import get_scheduler

from viz_utils import plot_fig, get_metrics_str


class AgentFormerTrainer(pl.LightningModule):
    def __init__(self, cfg, args):
        super().__init__()
        model_id = cfg.get('model_id', 'agentformer')
        self.model = model_dict[model_id](cfg)
        self.cfg = cfg
        self.args = args
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
        data = self.model()
        total_loss, loss_dict, loss_unweighted_dict = self.model.compute_loss()

        # losses
        self.log(f'{mode}/loss', total_loss, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)
        for loss_name, loss in loss_dict.items():
            self.log(f'{mode}/{loss_name}', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)

        return data, {'loss': total_loss, **loss_dict}

    def training_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'train')
        return loss_dict

    def validation_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'test')
        gt_motion = self.cfg.traj_scale * data['fut_motion'].transpose(1, 0).cpu()
        pred_motion = self.cfg.traj_scale * data[f'infer_dec_motion'].detach().cpu()
        obs_motion = self.cfg.traj_scale * data[f'pre_motion'].transpose(1, 0).cpu()
        # self.stats.update(gt_motion, pred_motion)
        return {**loss_dict, 'frame': batch['frame'], 'seq': batch['seq'],
                'gt_motion': gt_motion, 'pred_motion': pred_motion, 'obs_motion': obs_motion,}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _epoch_end(self, outputs):
        args_list = [(output['pred_motion'].numpy(), output['gt_motion'].numpy()) for output in outputs]
        # pred_motion: num_peds, num_samples, ts, 2       # gt_motion: num_peds, ts, 2
        if self.args.mp:
            with multiprocessing.Pool(self.num_workers) as pool:
                all_metrics = pool.starmap(partial(eval_one_seq2,
                                                   collision_rad=self.collision_rad,
                                                   return_sample_vals=self.args.save_viz), args_list)
        else:
            all_metrics = starmap(partial(eval_one_seq2,
                                          collision_rad=self.collision_rad,
                                          return_sample_vals=self.args.save_viz), args_list)

        all_metrics, all_sample_vals, argmins, collision_mats = zip(*all_metrics)

        num_agent_per_seq = np.array([output['gt_motion'].shape[0] for output in outputs])
        total_num_agents = np.sum(num_agent_per_seq)
        results_dict = {}
        for key, values in zip(stats_func.keys(), zip(*all_metrics)):
            if '_seq' in key:  # sequence-based metric
                value = np.mean(values)
            else:  # agent-based metric
                value = np.sum(values * num_agent_per_seq) / np.sum(num_agent_per_seq)
            results_dict[key] = value

        # log and print results
        print(f"\n\n\n{self.current_epoch}")
        self.log(f'val/total_num_agents', float(total_num_agents), sync_dist=True, logger=True)
        for key, value in results_dict.items():
            print(f"{value:.4f}")
            self.log(f'val/{key}', value, sync_dist=True, prog_bar=True, logger=True)
        print(total_num_agents)

        if self.args.save_viz:
            self._save_viz(outputs, all_sample_vals, all_metrics, argmins, collision_mats)

    def _save_viz(self, outputs, all_sample_vals, all_meters_values, argmins, collision_mats):
        seq_to_plot_args = []
        for frame_i, (output, seq_to_sample_metrics) in enumerate(zip(outputs, all_sample_vals)):
            frame = output['frame']
            seq = output['seq']
            pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
            pred_fake_traj = output['pred_motion'].numpy().transpose(1, 2, 0, 3)  # (samples, ts, n_peds, 2)

            num_samples, _, n_ped, _ = pred_fake_traj.shape

            anim_save_fn = f'viz/{seq}_frame-{frame}.mp4'
            plot_args_list = [anim_save_fn, f"Seq: {seq} frame: {frame}", (5, 4)]

            pred_fake_traj_min = pred_fake_traj[argmins[frame_i],:,np.arange(n_ped)].swapaxes(0, 1)  # (n_ped, )
            min_ADE_stats = get_metrics_str(dict(zip(stats_func.keys(), all_meters_values[frame_i])))
            args_dict = {'plot_title': f"best mADE sample",
                         'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_fake_traj_min,
                         'text_fixed': min_ADE_stats}
            plot_args_list.append(args_dict)

            for sample_i in range(num_samples - 1):
                stats = get_metrics_str(seq_to_sample_metrics, sample_i)
                args_dict = {'plot_title': f"Sample {sample_i}",
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
        self.model.step_annealer()

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs)

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
