import torch
from torch import nn
from torch.nn import functional as F
# torch.set_printoptions(precision=4, linewidth=90, sci_mode=False, threshold=8, edgeitems=5)
from collections import defaultdict

from .common.mlp import MLP
from .agentformer_loss import loss_func
from .common.dist import *
from .agentformer_lib import AgentFormerEncoderLayer, AgentFormerDecoderLayer, AgentFormerDecoder, AgentFormerEncoder
from .map_encoder import MapEncoder
from utils.torch import *
from utils.utils import initialize_weights
from model.running_norm import RunningNorm
from data.jrdb_kp3 import USE_ACTIONS


INPUT_TYPE_TO_DIMS = {'scene_norm': 2, 'vel': 2, 'heading': 2,
                      'kp_norm': 99, # 34,
                      'kp_norm_3dhst': 99,
                      'kp_vel': 99, #34,
                      # 'kp_vel_3dhst': 34,
                      'kp_scores': 17,
                      'cam_intrinsics': 9, 'cam_extrinsics': 7, 'cam_id': 1,
                      'action': len(USE_ACTIONS), 'action_score': len(USE_ACTIONS)}

def generate_ar_mask(sz, agent_num, agent_mask, pre_motion_mask, fut_motion_mask):
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask

def generate_ar_mask(sz, agent_num, agent_mask):
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask


def generate_mask(tgt_sz, src_sz, agent_num, agent_mask):
    assert tgt_sz % agent_num == 0 and src_sz % agent_num == 0
    mask = agent_mask.repeat(tgt_sz // agent_num, src_sz // agent_num)
    return mask


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False, agent_enc_learn=False):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae

    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = torch.ceil(torch.tensor(x.shape[0] / num_a)).to(torch.int)  # x.shape[0] // num_a
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)[:x.shape[0]]
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x)


""" Context (Past) Encoder """
class ContextEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.motion_dim = ctx['motion_dim']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.input_type = ctx['input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.vel_heading = ctx['vel_heading']
        self.input_norm_type = cfg.get('input_norm_type', None)
        ctx['context_dim'] = self.model_dim
        self.concat_all_inputs = ctx['concat_all_inputs']
        self.pos_embedding_dim = ctx['pos_embedding_dim']
        self.kp_embedding_dim = ctx['kp_embedding_dim']
        self.input_type_to_dims = ctx['input_type_to_dims']
        self.num_kp = ctx['num_kp']
        self.kp_dim = ctx['kp_dim']

        in_dim = 0
        in_dim_kp = 0
        for key in self.input_type:
            if key in ['scene_norm', 'vel', 'heading'] or self.concat_all_inputs:
                in_dim += self.input_type_to_dims[key]
            else:
                in_dim_kp += self.input_type_to_dims[key]
        if in_dim_kp > 0:
            if ctx['add_kp']:
                self.input_fc_kp = nn.Sequential(nn.Linear(in_dim_kp, self.model_dim),
                                                 *[nn.Linear(self.model_dim, self.model_dim)
                                                   for _ in range(ctx['n_projection_layer']-1)])
                self.input_fc = nn.Linear(in_dim, self.model_dim)
            else:
                # self.kp_embedding_dim = cfg.get('kp_dim', self.model_dim // 2)
                self.input_fc_kp = nn.Linear(in_dim_kp, self.kp_embedding_dim)
                self.input_fc = nn.Linear(in_dim, self.pos_embedding_dim)
                # self.input_fc = nn.Linear(in_dim, self.model_dim - self.kp_embedding_dim)
        else:
            self.kp_embedding_dim = 0
            self.input_fc_kp = None
            self.input_fc = nn.Linear(in_dim, self.model_dim)

        if self.input_norm_type == 'running_norm':
            self.input_norm = RunningNorm(in_dim)
        else:
            self.input_norm = None

        encoder_layers = AgentFormerEncoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)
        # self.pos_encoder = PositionalAgentEncoding(self.kp_embedding_dim+self.pos_embedding_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])

    def forward(self, data):
        traj_in_list = []
        kp_input_list = []
        num_timesteps = data['pre_motion'].shape[0]
        for key in self.input_type:
            if key == 'pos':
                traj_in_list.append(data['pre_motion'])
            elif key == 'vel':
                vel = data['pre_vel']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in_list.append(vel)
            elif key == 'norm':
                traj_in_list.append(data['pre_motion_norm'])
            elif key == 'scene_norm':
                traj_in_list.append(data['pre_motion_scene_norm'])
            elif key == 'kp':
                kp_input_list.append(data['pre_kp'].reshape(*data['pre_kp'].shape[0:2], -1))
            elif key == 'kp_norm':
                kp_input_list.append(data['pre_kp_norm'].reshape(*data['pre_kp_norm'].shape[0:2], -1))
            elif key == 'kp_vel':
                vel = data['pre_kp_vel']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                kp_input_list.append(vel.reshape((vel.shape[0], vel.shape[1], -1)))
            elif key == 'kp_scores':
                kp_input_list.append(data['pre_kp_scores'])
            elif key == 'cam_id':
                kp_input_list.append(data['pre_cam_id'].unsqueeze(-1))
            elif key == 'cam_intrinsics':
                kp_input_list.append(data['pre_cam_intrinsics'])
            elif key == 'cam_extrinsics':
                kp_input_list.append(data['pre_cam_extrinsics'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((num_timesteps, 1, 1))
                traj_in_list.append(hv)
            elif key == 'heading_avg':
                hv = data['heading_avg'].unsqueeze(0).repeat((num_timesteps, 1, 1))
                traj_in_list.append(hv)
            elif key == 'action':
                traj_in_list.append(data['pre_action_label'])
            elif key == 'action_score':
                traj_in_list.append(data['pre_action_score'])
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((num_timesteps, 1, 1))
                traj_in_list.append(map_enc)
            elif key == 'sf_feat':
                traj_in_list.append(data['pre_sf_feat'])
            else:
                raise ValueError('unknown input_type!')

        if self.concat_all_inputs:
            traj_in_list = [*traj_in_list, *kp_input_list]
            kp_input_list = []
        traj_in = torch.cat(traj_in_list, dim=-1)
        traj_in = traj_in.view(-1, traj_in.shape[-1])
        if self.input_norm is not None:
            traj_in = self.input_norm(traj_in)
        if self.ctx['add_kp']:
            tf_in = self.input_fc(traj_in).view(-1, 1, self.model_dim)
        else:
            tf_in = self.input_fc(traj_in).view(-1, 1, self.pos_embedding_dim)
        if len(kp_input_list) > 0:
            kp_input_tensor = torch.cat(kp_input_list, dim=-1)
            kp_input_tensor = kp_input_tensor.view(-1, 1, kp_input_tensor.shape[-1])
            kp_embedding = self.input_fc_kp(kp_input_tensor)
            if self.ctx['add_kp']:
                tf_in += kp_embedding
            else:
                tf_in = torch.cat([tf_in, kp_embedding], dim=-1)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=data['agent_num'], agent_enc_shuffle=agent_enc_shuffle)

        src_agent_mask = data['agent_mask'].clone()
        src_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], src_agent_mask).to(tf_in.device)

        data['context_enc'] = self.tf_encoder(tf_in_pos, mask=src_mask, num_agent=data['agent_num'])

        context_rs = data['context_enc'].view(-1, data['agent_num'], self.model_dim)
        # compute per agent context
        if self.pooling == 'mean':
            data['agent_context'] = torch.mean(context_rs, dim=0)
        else:
            data['agent_context'] = torch.max(context_rs, dim=0)[0]


""" Future Encoder """
class FutureEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.z_tau_annealer = ctx.get('z_tau_annealer', None)
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.input_type = ctx['fut_input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.vel_heading = ctx['vel_heading']
        self.kp_embedding_dim = ctx['kp_embedding_dim']
        self.pos_embedding_dim = ctx['pos_embedding_dim']
        self.input_norm_type = cfg.get('input_norm_type', None)
        self.input_type_to_dims = ctx['input_type_to_dims']
        self.concat_all_inputs = ctx['concat_all_inputs']

        in_dim = 0
        in_dim_kp = 0
        for key in self.input_type:
            if key in ['scene_norm', 'vel', 'heading'] or self.concat_all_inputs:
                in_dim += self.input_type_to_dims[key]
            else:
                in_dim_kp += self.input_type_to_dims[key]
        if in_dim_kp > 0:
            if ctx['add_kp']:
                # todo
                self.input_fc_kp = nn.Sequential(nn.Linear(in_dim_kp, self.model_dim),
                                                 *[nn.Linear(self.model_dim, self.model_dim)
                                                   for _ in range(ctx['n_projection_layer']-1)])
                self.input_fc = nn.Linear(in_dim, self.model_dim)
            else:
                self.input_fc_kp = nn.Linear(in_dim_kp, self.kp_embedding_dim)
                self.input_fc = nn.Linear(in_dim, self.pos_embedding_dim)
        else:
            self.input_fc_kp = None
            self.input_fc = nn.Linear(in_dim, self.model_dim)

        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - self.motion_dim
        if self.input_norm_type == 'running_norm':
            self.input_norm = RunningNorm(in_dim)
        else:
            self.input_norm = None

        decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, data):
        traj_in_list = []
        kp_input_list = []
        num_timesteps = data['fut_motion'].shape[0]
        for key in self.input_type:
            if key == 'pos':
                traj_in_list.append(data['fut_motion'])
            elif key == 'vel':
                vel = data['fut_vel']
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in_list.append(vel)
            elif key == 'norm':
                traj_in_list.append(data['fut_motion_norm'])
            elif key == 'scene_norm':
                traj_in_list.append(data['fut_motion_scene_norm'])
            elif key == 'kp_norm':
                kp_input_list.append(data['fut_kp_norm'].reshape(*data['fut_kp_norm'].shape[0:2], -1))  # compress 17, 3 to one dimension
            elif key == 'kp_vel':
                vel = data['fut_kp_vel']
                # vel = torch.cat([vel[[0]], vel], dim=0)  # unsure what this is for
                kp_input_list.append(vel.reshape((vel.shape[0], vel.shape[1], -1)))
            elif key == 'kp_scores':
                kp_input_list.append(data['fut_kp_scores'])
            elif key == 'cam_id':
                kp_input_list.append(data['fut_cam_id'].unsqueeze(-1))
            elif key == 'cam_intrinsics':
                kp_input_list.append(data['fut_cam_intrinsics'])
            elif key == 'cam_extrinsics':
                kp_input_list.append(data['fut_cam_extrinsics'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((num_timesteps, 1, 1))
                traj_in_list.append(hv)
            elif key == 'heading_avg':
                hv = data['heading_avg'].unsqueeze(0).repeat((num_timesteps, 1, 1))
                traj_in_list.append(hv)
            elif key == 'action':
                traj_in_list.append(data['fut_action_label'])
            elif key == 'action_score':
                traj_in_list.append(data['fut_action_score'])
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((num_timesteps, 1, 1))
                traj_in_list.append(map_enc)
            else:
                raise ValueError('unknown input_type!')
        if self.concat_all_inputs:
            traj_in_list = [*traj_in_list, *kp_input_list]
            kp_input_list = []
        traj_in = torch.cat(traj_in_list, dim=-1)
        batch_size = traj_in.shape[0]
        traj_in = traj_in.view(-1, traj_in.shape[-1])
        if self.input_norm is not None:
            traj_in = self.input_norm(traj_in)
        if self.ctx['add_kp']:
            tf_in = self.input_fc(traj_in).view(-1, 1, self.model_dim)
        else:
            tf_in = self.input_fc(traj_in).view(-1, 1, self.pos_embedding_dim)
        if len(kp_input_list)> 0:
            kp_input_tensor = torch.cat(kp_input_list, dim=-1)
            kp_input_tensor = kp_input_tensor.view(-1, 1, kp_input_tensor.shape[-1])
            kp_embedding = self.input_fc_kp(kp_input_tensor)
            if self.ctx['add_kp']:
                tf_in += kp_embedding
            else:
                tf_in = torch.cat([tf_in, kp_embedding], dim=-1)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=data['agent_num'], agent_enc_shuffle=agent_enc_shuffle)
        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()
        mem_mask = generate_mask(tf_in.shape[0], data['context_enc'].shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
        tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], tgt_agent_mask).to(tf_in.device)

        # tgt_mask = generate_mask_missing_ts(tf_in.shape[0], tf_in.shape[0], data['agent_num'], tgt_agent_mask).to(tf_in.device)
        # print(f"{mem_mask.shape=}")
        # print(f"{tgt_mask.shape=}")
        # print(f"{data['pre_mask'].shape=}")

        tf_out, _ = self.tf_decoder(tf_in_pos, data['context_enc'], memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'])
        tf_out = tf_out.view(batch_size, -1, self.model_dim)

        if self.pooling == 'mean':
            h = torch.mean(tf_out, dim=0)
        else:
            h = torch.max(tf_out, dim=0)[0]
        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)
        if self.z_type == 'gaussian':
            data['q_z_dist'] = Normal(params=q_z_params)
        else:
            data['q_z_dist'] = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        data['q_z_samp'] = data['q_z_dist'].rsample()


""" Future Decoder """
class FutureDecoder(nn.Module):
    def __init__(self, cfg, ctx, loss_cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.loss_cfg = loss_cfg
        self.ar_detach = ctx['ar_detach']
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.pred_scale = cfg.get('pred_scale', 1.0)
        self.pred_type = ctx['pred_type']
        self.sn_out_type = ctx['sn_out_type']
        self.sn_out_heading = ctx['sn_out_heading']
        self.input_type = ctx['dec_input_type']
        self.future_frames = ctx['future_frames']
        self.past_frames = ctx['past_frames']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.pos_offset = cfg.get('pos_offset', False)
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.learn_prior = ctx['learn_prior']
        self.input_norm_type = cfg.get('input_norm_type', None)
        self.tune_z = cfg.get('tune_z', False)
        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        # in_dim = forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim

        if self.input_norm_type == 'running_norm':
            self.input_norm = RunningNorm(in_dim)
        else:
            self.input_norm = None
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        if self.out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, forecast_dim)
        else:
            in_dim = self.model_dim
            self.out_mlp = MLP(in_dim, self.out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.out_mlp.out_dim, forecast_dim)
        initialize_weights(self.out_fc.modules())
        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())

    def decode_traj_ar(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
                       need_weights=False, approx_grad=False):
        agent_num = data['agent_num']
        if self.pred_type == 'vel':
            dec_in = pre_vel[[-1]]
        elif self.pred_type == 'pos':
            dec_in = pre_motion[[-1]]
        elif self.pred_type == 'scene_norm':
            dec_in = pre_motion_scene_norm[[-1]]
        else:
            dec_in = torch.zeros_like(pre_motion[[-1]])
        dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])
        z_in = z.view(-1, sample_num, z.shape[-1])
        in_arr = [dec_in, z_in]
        kp_input_list = []
        for key in self.input_type:
            if key == 'heading':
                heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(heading)
            elif key == 'heading_avg':
                hv = data['heading_avg'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                in_arr.append(hv)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(map_enc)
            elif key == 'kp_norm':
                kp_norm = data['pre_kp_norm'].unsqueeze(1).repeat((1, sample_num, 1))
                kp_input_list.append(kp_norm)
            else:
                raise ValueError('wrong decode input type!')
        dec_in_z = torch.cat(in_arr, dim=-1)

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()

        for i in range(self.future_frames):
            traj_in = dec_in_z.view(-1, dec_in_z.shape[-1])
            if self.input_norm is not None:
                traj_in = self.input_norm(traj_in)
            tf_in = self.input_fc(traj_in).view(dec_in_z.shape[0], -1, self.model_dim)
            agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
            tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle, t_offset=self.past_frames-1 if self.pos_offset else 0)
            # tf_in_pos = tf_in
            mem_mask = generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
            # tgt_mask = generate_ar_mask(tf_in.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)
            tgt_mask = generate_ar_mask(tf_in.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)

            tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'], need_weights=need_weights)
            out_tmp = tf_out.view(-1, tf_out.shape[-1])
            if self.out_mlp_dim is not None:
                out_tmp = self.out_mlp(out_tmp)
            seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
            if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
                norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
                if self.sn_out_type == 'vel':
                    norm_motion = torch.cumsum(norm_motion, dim=0)
                if self.sn_out_heading:
                    angles = data['heading'].repeat_interleave(sample_num)
                    norm_motion = rotation_2d_torch(norm_motion, angles)[0]
                seq_out = norm_motion + pre_motion_scene_norm[[-1]]
                seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
            if self.ar_detach:
                out_in = seq_out[-agent_num:].clone().detach()
            else:
                out_in = seq_out[-agent_num:]

            # create dec_in_z
            in_arr = [out_in, z_in]
            for key in self.input_type:
                if key == 'heading':
                    in_arr.append(heading)  # just append the last obs heading
                elif key == 'heading_avg':
                    in_arr.append(hv)
                elif key == 'map':
                    in_arr.append(map_enc)
                elif key == 'kp_norm':
                    in_arr.append(kp_norm)
                else:
                    raise ValueError('wrong decoder input type!')
            out_in_z = torch.cat(in_arr, dim=-1)
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)

        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        data[f'{mode}_seq_out'] = seq_out

        if self.pred_type == 'vel':
            dec_motion = torch.cumsum(seq_out, dim=0)
            dec_motion += pre_motion[[-1]]
        elif self.pred_type == 'pos':
            dec_motion = seq_out.clone()
        elif self.pred_type == 'scene_norm':
            dec_motion = seq_out + data['scene_orig']
        else:
            dec_motion = seq_out + pre_motion[[-1]]

        dec_motion = dec_motion.transpose(0, 1).contiguous()       # M x frames x 7
        if mode == 'infer':
            dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])        # M x Samples x frames x 3
        data[f'{mode}_dec_motion'] = dec_motion
        if need_weights:
            data['attn_weights'] = attn_weights

    def decode_traj_batch(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num):
        raise NotImplementedError

    def forward(self, data, mode, sample_num=1, approx_grad=False, autoregress=True, z=None, need_weights=False):
        context = data['context_enc'].repeat_interleave(sample_num, dim=1)       # 80 x 64
        pre_motion = data['pre_motion'].repeat_interleave(sample_num, dim=1)             # 10 x 80 x 2
        pre_vel = data['pre_vel'].repeat_interleave(sample_num, dim=1) if self.pred_type == 'vel' else None
        pre_motion_scene_norm = data['pre_motion_scene_norm'].repeat_interleave(sample_num, dim=1)

        # p(z)
        prior_key = 'p_z_dist' + ('_infer' if mode == 'infer' else '')
        if self.learn_prior:
            h = data['agent_context'].repeat_interleave(sample_num, dim=0)
            p_z_params = self.p_z_net(h)
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(params=p_z_params)
            else:
                data[prior_key] = Categorical(params=p_z_params)
        else:
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(mu=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device), logvar=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))
            else:
                data[prior_key] = Categorical(logits=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))

        if z is None:
            if mode in {'train', 'recon'}:
                z = data['q_z_samp'] if mode == 'train' else data['q_z_dist'].mode()
            elif mode == 'infer':
                z = data['p_z_dist_infer'].sample()
            else:
                raise ValueError('Unknown Mode!')
        # print('node rank:', torch.cuda.current_device(), 'z', z[:10])
        if autoregress:
            self.decode_traj_ar(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
                                need_weights=need_weights, approx_grad=approx_grad)
        else:
            self.decode_traj_batch(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num)


""" AgentFormer """
class AgentFormer(nn.Module):
    def __init__(self, cfg):#, pl_module=None):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg
        # self.pl_module = pl_module

        self.input_type = input_type = cfg.get('input_type', 'pos')
        pred_type = cfg.get('pred_type', input_type)
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = cfg.get('fut_input_type', input_type)
        dec_input_type = cfg.get('dec_input_type', [])
        pos_embedding_dim, kp_embedding_dim = tuple(map(int, cfg.get('embedding_dim_sizes', '128,128').split(',')))
        self.ctx = {
                # 'pl_module': pl_module,
            'tf_cfg': cfg.get('tf_cfg', {}),
            'nz': cfg.nz,
            'add_kp': cfg.get('add_kp', True),  # if not add, then concat
            'z_type': cfg.get('z_type', 'gaussian'),
            'future_frames': cfg.future_frames,
            'past_frames': cfg.past_frames,
            'motion_dim': cfg.motion_dim,
            'forecast_dim': cfg.forecast_dim,
            'input_type': input_type,
            'fut_input_type': fut_input_type,
            'dec_input_type': dec_input_type,
            'pred_type': pred_type,
            'tf_nhead': cfg.tf_nhead,
            'tf_model_dim': cfg.tf_model_dim,
            'tf_ff_dim': cfg.tf_ff_dim,
            'tf_dropout': cfg.tf_dropout,
            'pos_concat': cfg.get('pos_concat', False),
            'ar_detach': cfg.get('ar_detach', True),
            'max_agent_len': cfg.get('max_agent_len', 128),
            'use_agent_enc': cfg.get('use_agent_enc', False),
            'agent_enc_learn': cfg.get('agent_enc_learn', False),
            'agent_enc_shuffle': cfg.get('agent_enc_shuffle', False),
            'sn_out_type': cfg.get('sn_out_type', 'scene_norm'),
            'sn_out_heading': cfg.get('sn_out_heading', False),
            'vel_heading': cfg.get('vel_heading', False),
            'learn_prior': cfg.get('learn_prior', False),
            'use_map': cfg.get('use_map', False),
            'concat_all_inputs': cfg.get('concat_all_inputs', True),
            'pos_embedding_dim': pos_embedding_dim,
            'kp_embedding_dim': kp_embedding_dim,
            'input_type_to_dims': cfg.get('input_type_to_dims', INPUT_TYPE_TO_DIMS),
            'kp_dim': cfg.get('kp_dim', 3),
            'num_kp': cfg.get('num_kp', 24),
            'n_projection_layer': cfg.get('n_projection_layer', 1),
        }
        self.past_frames = self.ctx['past_frames']
        self.future_frames = self.ctx['future_frames']
        self.kp_dim = cfg.get('kp_dim', 3)
        self.num_kp = cfg.get('num_kp', 24)
        self.use_map = self.ctx['use_map']
        self.rand_rot_scene = cfg.get('rand_rot_scene', False)
        self.discrete_rot = cfg.get('discrete_rot', False)
        self.map_global_rot = cfg.get('map_global_rot', False)
        self.ar_train = cfg.get('ar_train', True)
        self.approx_grad = cfg.get('approx_grad', False)
        self.max_train_agent = cfg.get('max_train_agent', 100)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names or 'joint_sample' in self.loss_names
        self.param_annealers = nn.ModuleList()
        if self.ctx['z_type'] == 'discrete':
            self.ctx['z_tau_annealer'] = z_tau_annealer = ExpParamAnnealer(cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay)
            self.param_annealers.append(z_tau_annealer)

        # save all computed variables
        self.data = None

        # map encoder
        if self.use_map:
            self.map_encoder = MapEncoder(cfg.map_encoder)
            self.ctx['map_enc_dim'] = self.map_encoder.out_dim

        # models
        self.context_encoder = ContextEncoder(cfg.context_encoder, self.ctx)
        self.future_encoder = FutureEncoder(cfg.future_encoder, self.ctx)
        self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx, self.loss_cfg)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def get_torch_data(self, data):
        device = self.device
        for k, v in data.items():
            if 'motion' in k:
                data[k] = [torch.tensor(a).to(device) for a in data[k]]
        return data

    def set_data(self, data):
        device = self.device
        if self.training and len(data['pre_motion']) > self.max_train_agent:
            in_data = {}
            ind = np.random.choice(len(data['pre_motion']), self.max_train_agent).tolist()
            for key in [k for k in data.keys() if k.split('_')[0] in ['pre', 'fut', 'heading'] and 'data' not in k]:
                in_data[key] = [data[key][i] for i in ind if key in data and data[key] is not None]
        else:
            in_data = data

        self.data = defaultdict(lambda: None)
        self.data['cfg'] = self.cfg
        self.data['batch_size'] = len(in_data['pre_motion'])
        self.data['agent_num'] = len(in_data['pre_motion'])

        for key in [k for k in data.keys() if k.split('_')[0] in ['pre', 'fut'] and 'data' not in k]:
            if isinstance(in_data[key][0], np.ndarray):
                in_data[key] = [torch.tensor(a).to(device) for a in in_data[key]]
            self.data[key] = torch.stack(in_data[key], dim=0).to(device).transpose(0, 1).contiguous()  # swap batch and time dim

        self.data['pre_motion'] = torch.stack(in_data['pre_motion'], dim=0).to(device).transpose(0, 1).contiguous()  # swap batch and time dim
        self.data['fut_motion'] = torch.stack(in_data['fut_motion'], dim=0).to(device).transpose(0, 1).contiguous()

        # if 'pre_kp' in in_data:
        #     # flip the z, bc the people are in image coords in which the z (vertical)-axis points downward wrt world coordinate frame
        #     self.data['pre_kp'][...,1] *= -1
        #     self.data['fut_kp'][...,1] *= -1
        self.data['fut_motion_orig'] = torch.stack(in_data['fut_motion'], dim=0).to(device)   # future motion without transpose
        self.data['fut_mask'] = torch.stack(in_data['fut_motion_mask'], dim=0).to(device)
        self.data['pre_mask'] = torch.stack(in_data['pre_motion_mask'], dim=0).to(device)
        scene_orig_all_past = self.cfg.get('scene_orig_all_past', False)
        if scene_orig_all_past:
            self.data['scene_orig'] = self.data['pre_motion'].view(-1, 2).mean(dim=0)  # use the meaned history pos as the scene origin
        else:
            self.data['scene_orig'] = self.data['pre_motion'][-1].mean(dim=0)  # use the mean of the last obs step as the scene origin
        if 'heading' in in_data and in_data['heading'] is not None:
            self.data['heading'] = torch.tensor(np.array(in_data['heading'])).float().to(device)
        if 'heading_avg' in in_data and in_data['heading_avg'] is not None:
            self.data['heading_avg'] = torch.tensor(np.array(in_data['heading_avg'])).float().to(device)

        # rotate the scene randomly during training to prevent scene-specific overfitting
        if self.rand_rot_scene and self.training:
            if 'cam_id' in self.input_type:  # only rotate in increments of 2/5*np.pi because there are 5 cameras
                CAM_INCREMENT = 2  # how much in between adjacent cam ids
                theta_i = torch.randint(high=5, size=(1,)).to(device) * CAM_INCREMENT
                NUM_CAMS = 5 * CAM_INCREMENT
                self.data['pre_cam_id'] = torch.where(self.data['pre_cam_id'] != -1, (self.data['pre_cam_id'] - theta_i) % NUM_CAMS, -1)
                self.data['fut_cam_id'] = torch.where(self.data['fut_cam_id'] != -1, (self.data['fut_cam_id'] - theta_i) % NUM_CAMS, -1)
                theta = theta_i * np.pi * 2 / (NUM_CAMS * CAM_INCREMENT)
                assert 0<=theta < np.pi * 2
            elif self.discrete_rot:  # only rotate scene in increments
                theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2
            for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
                self.data[key], self.data[f'{key}_scene_norm'] = rotation_2d_torch(self.data[key], theta, self.data['scene_orig'])
            if 'heading' in in_data and in_data['heading'] is not None:
                self.data['heading'] += theta
                self.data['heading_avg'] += theta
            # TODO: add functionality for poses
        else:
            theta = torch.zeros(1).to(device)
            for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
                self.data[f'{key}_scene_norm'] = self.data[key] - self.data['scene_orig']   #  subtract last obs, meaned over agents
        self.data['train_theta'] = theta  # save for plotting

        if 'kp_norm' in self.input_type and 'kp_norm' in self.input_type:
            for key in ['pre_kp', 'fut_kp']:
                # 0 = hip joint is subtracted from each ped's joints to normalize
                self.data[f'{key}_norm'] = self.data[key]# - self.data[key][:,:,:1]  # already normalized in data processing tho

        self.data['pre_vel'] = self.data['pre_motion'][1:] - self.data['pre_motion'][:-1, :]
        self.data['fut_vel'] = self.data['fut_motion'] - torch.cat([self.data['pre_motion'][[-1]], self.data['fut_motion'][:-1, :]])
        if np.any(['kp_vel' in self.input_type]):
            self.data['pre_kp_vel'] = self.data['pre_kp'][1:] - self.data['pre_kp'][:-1]
            self.data['fut_kp_vel'] = self.data['fut_kp'] - torch.cat([self.data['pre_kp'][[-1]], self.data['fut_kp'][:-1, :]])
        self.data['cur_motion'] = self.data['pre_motion'][[-1]]
        # another norm prediction option for norming each ped independently of one another
        # (we are currently using scene_norm, which normes each ped by a single mean scene point)
        # self.data['pre_motion_norm'] = self.data['pre_motion'][:-1] - self.data['cur_motion']   # subtract last obs pos
        # self.data['fut_motion_norm'] = self.data['fut_motion'] - self.data['cur_motion']
        if 'heading' in in_data and in_data['heading'] is not None:
            assert len(self.data['heading'].shape) == 1  # if not already sin / cos'ed
            self.data['heading_vec'] = torch.stack([torch.cos(self.data['heading']), torch.sin(self.data['heading'])], dim=-1)
            if 'heading_avg' in self.data:
                self.data['heading_avg'] = torch.stack([torch.cos(self.data['heading_avg']), torch.sin(self.data['heading_avg'])], dim=-1)

        # agent maps
        if self.use_map:
            scene_map = data['scene_map']
            self.data['scene_vis_map'] = data['scene_vis_map']  # for visualization
            scene_points = np.stack([d.cpu().numpy() for d in in_data['pre_motion']])[:, -1] * data['traj_scale']
            if self.map_global_rot:
                patch_size = [50, 50, 50, 50]
                rot = theta.repeat(self.data['agent_num']).cpu().numpy() * (180 / np.pi)
            else:
                patch_size = [50, 10, 50, 90]
                rot = -np.array(in_data['heading'])  * (180 / np.pi)
            self.data['agent_maps'] = scene_map.get_cropped_maps(scene_points, patch_size, rot).to(device)

        # agent shuffling
        if self.training and self.ctx['agent_enc_shuffle']:
            self.data['agent_enc_shuffle'] = torch.randperm(self.ctx['max_agent_len'])[:self.data['agent_num']].to(device)
        else:
            self.data['agent_enc_shuffle'] = None

        # mask out ped-ped cross attention links when the peds are too far apart (at the last observation step)
        conn_dist = self.cfg.get('conn_dist', 100000.0)
        cur_motion = self.data['cur_motion'][0]
        if conn_dist < 1000.0:
            threshold = conn_dist / self.cfg.traj_scale
            pdist = F.pdist(cur_motion)
            D = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
            D[np.triu_indices(cur_motion.shape[0], 1)] = pdist
            D = D + D.T
            mask = torch.zeros_like(D)
            mask[D > threshold] = float('-inf')
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        self.data['agent_mask'] = mask

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def forward(self, data=None):
        if data is not None:
            assert isinstance(data, dict)
            self.set_data(data)
        if self.use_map:
            self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])
        self.context_encoder(self.data)
        self.future_encoder(self.data)
        self.future_decoder(self.data, mode='train', autoregress=self.ar_train)
        if self.compute_sample:
            k = self.loss_cfg.get('sample', self.loss_cfg.get('joint_sample'))['k']
            self.inference(sample_num=k)
        return self.data

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        if self.use_map and self.data['map_enc'] is None:
            self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])
        if self.data['context_enc'] is None:
            self.context_encoder(self.data)
        if mode == 'recon':
            sample_num = 1
            self.future_encoder(self.data)
        self.future_decoder(self.data, mode=mode, sample_num=sample_num, approx_grad=self.approx_grad,
                            autoregress=True, need_weights=need_weights)
        return self.data[f'{mode}_dec_motion'], self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            params = [self.data, self.loss_cfg[loss_name]]
            loss, loss_unweighted = loss_func[loss_name](*params)
            total_loss += loss.squeeze()
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict
