import torch
from torch import nn
from torch.nn import functional as F
from utils.torch import *
from utils.config import Config
from .common.mlp import MLP
from .common.dist import *
from . import model_lib
from .sfm import collision_term


""" Post-trajectory prediction collision-avoidance optimizer """
class TrajectoryOptimizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg
        self.nk = nk = cfg.sample_k
        self.nz = nz = cfg.nz
        self.share_eps = cfg.get('share_eps', True)
        self.train_w_mean = cfg.get('train_w_mean', False)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())

        pred_cfg = Config(cfg.pred_cfg, tmp=False, create_dirs=False)
        pred_model = model_lib.model_dict[pred_cfg.model_id](pred_cfg)
        self.pred_model_dim = pred_cfg.tf_model_dim
        if cfg.pred_epoch > 0:
            cp_path = pred_cfg.model_path % cfg.pred_epoch
            print('loading model from checkpoint: %s' % cp_path)
            model_cp = torch.load(cp_path, map_location='cpu')
            pred_model.load_state_dict(model_cp['model_dict'])
        pred_model.eval()
        self.pred_model = [pred_model]

        # Dlow's Q net
        self.qnet_mlp = cfg.get('qnet_mlp', [512, 256])
        self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
        self.q_A = nn.Linear(self.q_mlp.out_dim, nk * nz)
        self.q_b = nn.Linear(self.q_mlp.out_dim, nk * nz)

    def set_device(self, device):
        self.device = device
        self.to(device)
        self.pred_model[0].set_device(device)

    def set_data(self, data):
        self.pred_model[0].set_data(data)
        self.data = self.pred_model[0].data

    def main(self, mean=False, need_weights=False):
        pred_model = self.pred_model[0]
        if hasattr(pred_model, 'use_map') and pred_model.use_map:
            self.data['map_enc'] = pred_model.map_encoder(self.data['agent_maps'])
        pred_model.context_encoder(self.data)

        if not mean:
            if self.share_eps:
                eps = torch.randn([1, self.nz]).to(self.device)
                eps = eps.repeat((self.data['agent_num'] * self.nk, 1))
            else:
                eps = torch.randn([self.data['agent_num'], self.nz]).to(self.device)
                eps = eps.repeat_interleave(self.nk, dim=0)

        qnet_h = self.q_mlp(self.data['agent_context'])
        A = self.q_A(qnet_h).view(-1, self.nz)
        b = self.q_b(qnet_h).view(-1, self.nz)

        z = b if mean else A* eps + b
        logvar = (A ** 2 + 1e-8).log()
        self.data['q_z_dist_dlow'] = Normal(mu=b, logvar=logvar)

        pred_model.future_decoder(self.data, mode='infer', sample_num=self.nk, autoregress=True, z=z,
                                  need_weights=need_weights)
        return self.data

    def forward(self):
        return self.main(mean=self.train_w_mean)

    def inference(self, mode, sample_num, need_weights=False):
        self.main(mean=True, need_weights=need_weights)
        res = self.data[f'infer_dec_motion']
        if mode == 'recon':
            res = res[:, 0]
        return res, self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict

    def step_annealer(self):
        pass


def compute_sample_sfm(data, cfg):
    sfm_params = cfg.get('sfm_params', data['cfg'].sfm_params)
    pred = data['infer_dec_motion']
    sample_num = pred.shape[1]
    pre_motion_orig = data['pre_motion'].transpose(0, 1).unsqueeze(1).repeat((1, sample_num, 1, 1))
    vel_pred = pred - torch.cat([pre_motion_orig[:, :, [-1]], pred[:, :, :-1]], dim=2)
    loss_unweighted = 0
    for i in range(pred.shape[2]):
        pos = pred[:, :, i]
        vel = vel_pred[:, :, i]
        col = collision_term(pos, vel, sfm_params)
        loss_unweighted += col
    loss_unweighted /= pred.shape[2]
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


loss_func = {
    'sample_sfm': compute_sample_sfm
}
