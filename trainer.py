import torch
import pytorch_lightning as pl
import numpy as np
from eval import eval_one_seq
from metrics import stats_func, get_collisions_mat_old
import multiprocessing
from model.agentformer import AgentFormer
from utils.torch import get_scheduler

from torchmetrics import Metric


class Stats(Metric):
    full_state_update = True
    higher_is_better = False
    is_differentiable = True

    def __init__(self, collision_rad):
        super().__init__()
        self.add_state("num_peds", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ade", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fde", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cr_max", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cr_mean", default=torch.tensor(0), dist_reduce_fx="sum")
        self.collision_rad = collision_rad


    def update(self, gt: torch.Tensor, preds: torch.Tensor):
        assert gt.shape[0] == preds.shape[0]
        assert gt.shape[1:] == preds.shape[2:], f"gt.shape[1:] ({gt.shape[1:]}) != preds.shape[2:] ({preds.shape[2:]})"
        self.num_peds += gt.shape[0]
        self.ade = 0.0
        self.fde = 0.0
        for pred, gt in zip(preds.transpose(1, 0, 2, 3), gt):
            diff = pred - gt.unsqueeze(0)  # samples x frames x 2
            dist = torch.norm(diff, axis=-1)  # samples x frames
            ade_dist = dist.mean(axis=-1)  # samples
            self.ade += ade_dist.min(axis=0)  # (1, )
            fde_dist = dist[..., -1]  # samples
            self.fde += fde_dist.min(axis=0)  # (1, )

        n_ped, n_sample, _, _ = preds.shape
        col_pred = torch.zeros((n_sample))  # cr_pred
        col_mats = []
        if n_ped > 1:
            for sample_idx, pa in enumerate(preds):
                _, n_ped_with_col_pred, col_mat = get_collisions_mat_old(sample_idx, pa, self.collision_rad)
                col_mats.append(col_mat)
                col_pred[sample_idx] += (n_ped_with_col_pred.sum())

        self.cr_mean = col_pred.mean(axis=0)
        self.cr_max = col_pred.max(axis=0)
        # self.cr_min = col_pred.min(axis=0)

    def compute(self):
        return [self.ade / self.num_peds, self.fde / self.num_peds, self.cr_max / self.num_peds, self.cr_mean / self.num_peds]


class AgentFormerTrainer(pl.LightningModule):
    def __init__(self, cfg, args):
        super().__init__()
        self.model = AgentFormer(cfg)
        self.cfg = cfg
        self.args = args
        self.num_workers = min(args.num_workers, int(multiprocessing.cpu_count() / torch.cuda.device_count()))
        self.batch_size = args.batch_size
        self.collision_rad = cfg.get('collision_rad', 0.1)
        self.hparams.update(vars(cfg))
        self.hparams.update(vars(args))
        self.stats = Stats(self.collision_rad)

    def on_test_start(self):
        self.model.set_device(self.device)

    def on_fit_start(self):
        self.model.set_device(self.device)

    def _step(self, batch, mode):
        self.model.set_data(batch)
        # Compute predictions
        data = self.model()
        total_loss, loss_dict, loss_unweighted_dict = self.model.compute_loss()

        # losses
        self.log(f'{mode}/loss', total_loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        for loss_name, loss in loss_dict.items():
            self.log(f'{mode}/{loss_name}', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)

        return data, {'loss': total_loss, **loss_dict}#, **metrics}

    def training_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'train')
        return loss_dict

    def validation_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'test')
        gt_motion = self.cfg.traj_scale * data['fut_motion'].transpose(1, 0)
        pred_motion = data[f'infer_dec_motion'].detach()
        self.stats.update(gt_motion, pred_motion)
        return {**loss_dict, 'gt_motion': gt_motion, 'pred_motion': pred_motion}

    def test_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'test')
        gt_motion = data['fut_motion'].cpu().numpy().swapaxes(0, 1) * self.cfg.traj_scale
        pred_motion = data[f'infer_dec_motion'].detach().cpu().numpy()
        return {**loss_dict, 'gt_motion': gt_motion, 'pred_motion': pred_motion}

    def _epoch_end(self, outputs, mode):
        args_list = [(output['gt_motion'], output['pred_motion'], self.collision_rad) for output in outputs]
        with multiprocessing.Pool(self.num_workers) as pool:
            all_meters_values, _ = zip(*pool.starmap(eval_one_seq, args_list))  # all_meters_agent_traj_nums
        total_num_agents = np.sum([output['gt_motion'].shape[0] for output in outputs])

        self.log(f'{mode}/total_num_agents', float(total_num_agents), sync_dist=True, logger=True)

        for key, metric in zip(stats_func.keys(), self.stats):
            self.log(f'{mode}/{key}-torchmetric', metric.compute(), sync_dist=True, prog_bar=True, logger=True)
        for key, values in zip(stats_func.keys(), all_meters_values):
            value = np.sum(values) / total_num_agents
            self.log(f'{mode}/{key}', value, sync_dist=True, prog_bar=True, logger=True)

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