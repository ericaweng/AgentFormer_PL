import torch
import pytorch_lightning as pl
import numpy as np
from eval import eval_one_seq, stats_func
import multiprocessing
from model.agentformer import AgentFormer
from utils.torch import get_scheduler


class AgentFormerTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = AgentFormer(cfg)
        self.cfg = cfg
        self.collision_rad = cfg.get('collision_rad', 0.1)

    def on_fit_start(self):
        self.model.set_device(self.device)

    def _step(self, batch, mode):
        self.model.set_data(batch)
        # Compute predictions
        data = self.model()
        total_loss, loss_dict, loss_unweighted_dict = self.model.compute_loss()
        # compute metrics
        gt_motion = data['fut_motion'].cpu().numpy().swapaxes(0, 1) * self.cfg.traj_scale
        # gt_motion_3D = torch.stack(data['fut_motion'], dim=0).cpu().numpy() * self.cfg.traj_scale
        pred_motion = data[f'infer_dec_motion'].detach().cpu().numpy().swapaxes(0, 1)
        metrics = dict(zip(stats_func.keys(), eval_one_seq(gt_motion, pred_motion, self.collision_rad)[0]))

        # Log results from training step
        self.log(f'{mode}/loss', total_loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        for loss_name, loss in loss_dict.items():
            self.log(f'{mode}/{loss_name}', loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        for metric_name, metric_val in metrics.items():
            self.log(f'train/{metric_name}', metric_val, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        return {'loss': total_loss, **loss_dict, 'gt_motion': gt_motion, 'pred_motion': pred_motion}#, **metrics}

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def _epoch_end(self, outputs, mode):
        args_list = [(output['gt_motion'], output['pred_motion'], self.collision_rad) for output in outputs]
        with multiprocessing.Pool() as pool:
            all_meters_values, _ = zip(*pool.starmap(eval_one_seq, args_list))  # all_meters_agent_traj_nums
        total_num_agents = np.sum([output['gt_motion'].shape[0] for output in outputs])
        # for output in outputs:
        #     assert output['gt_motion'].shape[0] == output['pred_motion'].shape[1]
        self.log(f'{mode}/total_num_agents', float(total_num_agents), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        for key, values in zip(stats_func.keys(), all_meters_values):
            value = np.sum(values) / total_num_agents
            self.log(f'{mode}/{key}', value, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

    def train_epoch_end(self, outputs):
        self._epoch_end(outputs, 'train')
        self.model.step_annealer()

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, 'test')

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