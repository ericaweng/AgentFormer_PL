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
from viz_utils import plot_traj_img
from model.sfm import *
from model.running_norm import RunningNorm


def generate_ar_mask_1aaat3(sz, agent_num, agent_mask):
    """predict one at a time, start off with 1x1 tgt (normal triangular autoregressive mask"""
    T = torch.ceil(torch.tensor(sz / agent_num)).to(torch.int)
    agent_mask_repeat = agent_mask.repeat(T, T)[:sz,:sz]
    indexer = torch.arange(sz).to(agent_mask_repeat.device)
    timestep_constraints = indexer[:, None] < indexer
    return_mask = agent_mask & timestep_constraints
    time_mask = torch.where(return_mask, -torch.inf, 0.)
    return time_mask


def generate_ar_mask_1aaat0(sz, agent_num, agent_mask):
    """off-diagonal mask"""
    T = torch.ceil(torch.tensor(sz / agent_num)).to(torch.int)
    mask = agent_mask.repeat(T, T)[:sz,:sz]
    indexer = torch.arange(sz).to(mask.device)
    time_mask = (indexer[:, None] > indexer - agent_num)  # | (indexer[None, :] < agent_num) & (indexer[:, None] < agent_num))
    joint_mask = ~(mask.to(bool) | time_mask)
    joint_mask = joint_mask.to(torch.float16)
    joint_mask[joint_mask == 1] = -torch.inf
    return joint_mask


def generate_ar_mask_1aaat1(sz, agent_num, agent_mask):
    """mask out current timestep cols for agents-timesteps not being predicted"""
    a_i = sz % agent_num
    low_T = sz // agent_num
    T = torch.ceil(torch.tensor(sz / agent_num)).to(torch.int)
    agent_mask_repeat = agent_mask.repeat(T, T)[:sz,:sz]
    indexer = torch.arange(sz).to(agent_mask_repeat.device)
    # get columns of agent-timesteps of current timestep - 1 // set columns representing all agents of current timestep to False
    cols_constraints = indexer[None, :] >= low_T * agent_num
    # get rows of agent-timesteps of current timestep that haven't been computed yet
    rows_constraints = indexer[:, None] > (low_T - 1) * agent_num + a_i
    curr_ts_fut_agent_mask = cols_constraints & rows_constraints
    timestep_constraints = ~(indexer[:, None] > indexer - agent_num)
    time_mask_bool = timestep_constraints | curr_ts_fut_agent_mask
    time_mask = torch.where(time_mask_bool, -torch.inf, 0.)
    joint_mask = agent_mask_repeat + time_mask
    return joint_mask


def generate_ar_mask_1aaat2(sz, agent_num, agent_mask):
    """mask out rows that are not being predicted"""
    # a_i = sz % agent_num
    # low_T = sz // agent_num
    T = torch.ceil(torch.tensor(sz / agent_num)).to(torch.int)
    agent_mask_repeat = agent_mask.repeat(T, T)[:sz,:sz]
    indexer = torch.arange(sz).to(agent_mask_repeat.device)
    future_pred_constraints = indexer[:, None] > sz - agent_num
    timestep_constraints = ~(indexer[:, None] > indexer - agent_num)
    # print("timestep_constraints:", timestep_constraints)
    time_mask_bool = timestep_constraints | future_pred_constraints  # curr_ts_fut_agent_mask
    # print("time_mask_bool:", time_mask_bool)
    time_mask = torch.where(time_mask_bool, -torch.inf, 0.)
    joint_mask = agent_mask_repeat + time_mask
    return joint_mask


def generate_ar_mask_1aaat(sz, agent_num, agent_mask):
    T = torch.ceil(torch.tensor(sz / agent_num)).to(torch.int)
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask[:sz, :sz]


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


def generate_mask_1aaat(tgt_sz, src_sz, agent_num, agent_mask):
    assert src_sz % agent_num == 0
    times_repeat_x = torch.ceil(torch.tensor(tgt_sz / agent_num)).to(torch.int)
    mask = agent_mask.repeat(times_repeat_x, src_sz // agent_num)[:tgt_sz]
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
        in_dim = self.motion_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - self.motion_dim
        if 'sf_feat' in self.input_type:
            in_dim += 4 - self.motion_dim

        if self.input_norm_type == 'running_norm':
            self.input_norm = RunningNorm(in_dim)
        else:
            self.input_norm = None
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        encoder_layers = AgentFormerEncoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])

    def forward(self, data):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['pre_motion'])
            elif key == 'vel':
                vel = data['pre_vel']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['pre_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['pre_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
            elif key == 'sf_feat':
                traj_in.append(data['pre_sf_feat'])
            else:
                raise ValueError('unknown input_type!')

        traj_in = torch.cat(traj_in, dim=-1)
        traj_in = traj_in.view(-1, traj_in.shape[-1])
        if self.input_norm is not None:
            traj_in = self.input_norm(traj_in)
        tf_in = self.input_fc(traj_in).view(-1, 1, self.model_dim)
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
        self.input_norm_type = cfg.get('input_norm_type', None)
        # networks
        in_dim = forecast_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        if 'sf_feat' in self.input_type:
            in_dim += 4 - forecast_dim

        if self.input_norm_type == 'running_norm':
            self.input_norm = RunningNorm(in_dim)
        else:
            self.input_norm = None
        self.input_fc = nn.Linear(in_dim, self.model_dim)

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

    def forward(self, data, reparam=True):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['fut_motion'])
            elif key == 'vel':
                vel = data['fut_vel']
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['fut_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['fut_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
            elif key == 'sf_feat':
                traj_in.append(data['fut_sf_feat'])
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)
        batch_size = traj_in.shape[0]
        traj_in = traj_in.view(-1, traj_in.shape[-1])
        if self.input_norm is not None:
            traj_in = self.input_norm(traj_in)
        tf_in = self.input_fc(traj_in).view(-1, 1, self.model_dim)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=data['agent_num'], agent_enc_shuffle=agent_enc_shuffle)

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()
        mem_mask = generate_mask(tf_in.shape[0], data['context_enc'].shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
        tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], tgt_agent_mask).to(tf_in.device)

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
    def __init__(self, cfg, ctx, loss_cfg, sfm_learnable_hparams=None, **kwargs):
        super().__init__()
        self.cfg = cfg
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
        self.use_sfm = ctx['use_sfm']
        self.sfm_params = ctx['sfm_params']
        self.input_norm_type = cfg.get('input_norm_type', None)
        self.tune_z = cfg.get('tune_z', False)
        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        if self.use_sfm:
            in_dim += 4

        if self.input_norm_type == 'running_norm':
            self.input_norm = RunningNorm(in_dim)
        else:
            self.input_norm = None
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        # sfm learnable hparams
        self.sfm_learnable_hparams = sfm_learnable_hparams

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

    def decode_traj_ar_1aaat(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
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
        # print("dec_in.shape (last obs step + previous decoded timesteps):", dec_in.shape)
        # print("z_in (latent, output of future decoder OR trajectory sampler OR prior):", z_in.shape)
        for key in self.input_type:
            if key == 'heading':
                heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(heading)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(map_enc)
            else:
                raise ValueError('wrong decode input type!')
        if self.use_sfm:
            sf_feat = data['pre_sf_feat'][-1].unsqueeze(1).repeat((1, sample_num, 1))
            in_arr.append(sf_feat)
        dec_in_z = torch.cat(in_arr, dim=-1)

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()

        seq_outs = []
        for pred_i in range(self.future_frames):
            for a_i in range(agent_num):
                # self.eval()
                traj_in = dec_in_z.view(-1, dec_in_z.shape[-1])#[:i*agent_num + a_i]
                # if self.input_norm is not None:
                #     traj_in = self.input_norm(traj_in)
                tf_in = self.input_fc(traj_in).view(dec_in_z.shape[0], -1, self.model_dim)
                agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
                t_offset = self.past_frames-1 if self.pos_offset else 0
                tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle, t_offset=t_offset)
                mem_mask = generate_mask_1aaat(tf_in.shape[0], context.shape[0], agent_num, mem_agent_mask).to(tf_in.device)
                tgt_mask = generate_ar_mask_1aaat(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)

                tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask,
                                                       num_agent=agent_num, need_weights=True)#need_weights)

                # tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'], need_weights=need_weights)
                # print("tf_out.shape: (after tf_decoder, the next timestep prediction for all agents)", tf_out.shape)
                out_tmp = tf_out.view(-1, tf_out.shape[-1])
                # print("out_tmp.shape:", out_tmp.shape)
                # if self.out_mlp_dim is not None:
                #     out_tmp = self.out_mlp(out_tmp)
                seq_out_all = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
                # print("seq_out_all (before cutting):", seq_out_all)
                # get only the agent being predicted.
                # past agents should be the same; but we already have them in dec_in_z.
                # future agents are useless. maybe use them as loss in the future.
                num_agent_ts = seq_out_all.shape[0]
                seq_out = seq_out_all[num_agent_ts-agent_num: num_agent_ts-agent_num + 1]
                # seq_out = seq_out_all[-agent_num: -agent_num + 1]
                # seq_out = seq_out_all[num_agent_ts - agent_num + a_i: num_agent_ts - agent_num + a_i + 1]
                # print("seq_out single (after cutting):\n", seq_out)
                # print("seq_out.shape (after taking just the current timestep predictions and relevant agents for the next prediction):", seq_out.shape)
                if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
                    norm_motion = seq_out.view(-1, sample_num * 1, seq_out.shape[-1])
                    # norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
                    # print("norm_motion.shape:", norm_motion.shape)
                    # if self.sn_out_type == 'vel':
                    #     norm_motion = torch.cumsum(norm_motion, dim=0)
                    # if self.sn_out_heading:
                    #     angles = data['heading'].repeat_interleave(sample_num)
                    #     norm_motion = rotation_2d_torch(norm_motion, angles)[0]
                    times_repeat = torch.ceil(torch.tensor(seq_out_all.shape[0] / agent_num)).to(torch.int)
                    seq_out_all_normed = seq_out_all + pre_motion_scene_norm[-1,:,None].repeat(times_repeat,1,1)[:num_agent_ts]  #[-1:, a_i:a_i+1]
                    # TODO
                    # print("seq_out_all_normed:", seq_out_all_normed)
                    # print('should equal seq_outs[-1]:', seq_outs)
                    seq_out = norm_motion + pre_motion_scene_norm[-1:, a_i:a_i+1]
                    # print("pre_motion_scene_norm[-1:, a_i:a_i+1]:", pre_motion_scene_norm[-1:, a_i:a_i+1])
                    # print("pre_motion_scene_norm:", pre_motion_scene_norm[-1])
                    # print("pre_motion_scene_norm.shape:", pre_motion_scene_norm.shape)
                    seq_out = seq_out.view(-1, sample_num, seq_out.shape[-1])
                    # print("seq_out single (after adding norm):", seq_out)
                    # seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
                    # print("seq_out.shape after adding norm motion:", seq_out.shape)
                seq_outs.append(seq_out)

                save_fn = f'viz/test_t-{pred_i}_ai-{a_i}.png'
                x_attn_w = attn_weights['cross_attn_weights'][0, 0].reshape(-1, 8, agent_num)
                # get the first attention layer (0), and the first sample (0)
                s_attn_w = attn_weights['self_attn_weights'][0, 0]  # .reshape(*attn_weights['self_attn_weights'].shape[:-1], -1, 2)[0,0]
                attn_ped_i = s_attn_w.shape[0] - agent_num  # last timestep; 0th agent
                obs_traj = pre_motion.cpu().numpy()
                pred_traj_gt = data['fut_motion'].cpu().numpy()
                if sample_num == 1 and False:
                    pred_traj = (torch.cat(seq_outs).reshape(-1, 2) + data['scene_orig']).cpu().numpy()
                    plot_traj_img(obs_traj, save_fn, pred_traj=pred_traj, pred_traj_gt=pred_traj_gt,
                                  attn_ped_i=attn_ped_i, ped_radius=0.1, pred_traj_fake=None, self_attn=s_attn_w,
                                  cross_attn=x_attn_w)
                elif sample_num == 20:
                    pred_traj = (torch.cat(seq_outs) + data['scene_orig']).cpu().numpy()
                    for sample_i in range(sample_num):
                        save_fn = f'viz/sample-{sample_i}_test_t-{pred_i}_ai-{a_i}.png'
                        plot_traj_img(obs_traj[:, sample_i::sample_num], save_fn,
                                      pred_traj=pred_traj[:, sample_i].reshape(-1, 2),
                                      pred_traj_gt=pred_traj_gt,
                                      attn_ped_i=attn_ped_i, ped_radius=0.1, pred_traj_fake=None, self_attn=s_attn_w,
                                      cross_attn=x_attn_w)
                    import ipdb; ipdb.set_trace()

                # TODO today
                # print('should equal seq_out singles:', torch.cat(seq_outs).shape)
                # print('should equal seq_out singles:', torch.cat(seq_outs)[:,0:1])
                # print("dec_in (input unnormed positions only):\n", dec_in_z[...,:2])
                # get just the next ts's predictions, because the rest is previous predictions (dec_in_z),
                # which we already have (and it's identical to seq_out)
                # out_in = seq_out[-1:].clone().detach()
                # out_in = seq_out[-agent_num:-agent_num+a_i+1].clone().detach()
                out_in = seq_out
                if self.ar_detach:
                    out_in = out_in.clone().detach()
                    # out_in = seq_out[-agent_num:].clone().detach()
                    # out_in = seq_out[-agent_num+a_i:-agent_num+a_i+1].clone().detach()
                # create dec_in_z
                # out_ins.append(out_in)

                # next_ts_dec_in_z:
                # print("z_in.shape (to be concatted with out_in to format next ts input):", z_in.shape)
                in_arr = [out_in, z_in[a_i:a_i+1]]
                # for key in self.input_type:
                #     if key == 'heading':
                #         in_arr.append(heading)
                #     elif key == 'map':
                #         in_arr.append(map_enc)
                #     else:
                #         raise ValueError('wrong decoder input type!')
                # if self.use_sfm:
                #     assert self.pred_type == 'scene_norm'
                #     pos = out_in + data['scene_orig']
                #     tmp_pos = pre_motion_scene_norm[-1].view(-1, sample_num, self.forecast_dim) if i == 0 else last_pos
                #     vel = pos - tmp_pos
                #     state = torch.cat([pos, vel], dim=-1)
                #     sf_feat = compute_grad_feature(state, self.sfm_params, self.sfm_learnable_hparams)
                #     in_arr.append(sf_feat)
                #     last_pos = pos
                out_in_z = torch.cat(in_arr, dim=-1)
                # print("out_in_z.shape (to be concatted with context + previous predicted agent-ts, for next round):", out_in_z.shape)
                dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)
                if approx_grad:
                    dec_in_z = dec_in_z.detach()#.requires_grad_()
                # print("dec_in_z (positions only):", dec_in_z[...,:2])
                # print("dec_in_z.shape:", dec_in_z.shape)
                # print()

        seq_out = torch.cat(seq_outs)
        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        # print("seq_out:", seq_out)
        # print("seq_out.shape:", seq_out.shape)
        # import ipdb; ipdb.set_trace()
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
        for key in self.input_type:
            if key == 'heading':
                heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(heading)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(map_enc)
            else:
                raise ValueError('wrong decode input type!')
        if self.use_sfm:
            sf_feat = data['pre_sf_feat'][-1].unsqueeze(1).repeat((1, sample_num, 1))
            in_arr.append(sf_feat)
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
            tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)

            # if approx_grad:
            #     tf_out, attn_weights = checkpoint.checkpoint(self.tf_decoder, (tf_in_pos, context, tgt_mask, mem_mask, None, None, data['agent_num'], need_weights))
            #     tf_out, attn_weights = checkpoint.checkpoint_sequential(self.tf_decoder, (tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'], need_weights=need_weights))
            # else:
            tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'], need_weights=True)#need_weights)
            # print("tf_out:", tf_out)
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
            # print("self attn_weights.shape:", attn_weights['self_attn_weights'][0].shape)
            # print("x attn_weights.shape:", attn_weights['cross_attn_weights'][0].shape)
            # save_fn = f'viz/pre_t-{i}.png'
            # x_attn_w = attn_weights['cross_attn_weights'][0,0].reshape(-1, 8, agent_num)
            # # get the first attention layer (0), and the first sample (0)
            # s_attn_w = attn_weights['self_attn_weights'][0,0]#.reshape(*attn_weights['self_attn_weights'].shape[:-1], -1, 2)[0,0]
            # attn_ped_i = s_attn_w.shape[0] - agent_num  # last timestep; 0th agent
            # obs_traj = pre_motion.cpu().numpy()
            # pred_traj = (seq_out.squeeze() + data['scene_orig']).cpu().numpy()
            # pred_traj_gt = data['fut_motion'].cpu().numpy()
            # if sample_num == 1:
            #     plot_traj_img(obs_traj, save_fn, pred_traj=pred_traj, pred_traj_gt=pred_traj_gt, attn_ped_i=attn_ped_i,
            #                   ped_radius=0.1, pred_traj_fake=None, self_attn=s_attn_w, cross_attn=x_attn_w)
            # import ipdb; ipdb.set_trace()

            # create dec_in_z
            in_arr = [out_in, z_in]
            for key in self.input_type:
                if key == 'heading':
                    in_arr.append(heading)
                elif key == 'map':
                    in_arr.append(map_enc)
                else:
                    raise ValueError('wrong decoder input type!')
            if self.use_sfm:
                assert self.pred_type == 'scene_norm'
                pos = out_in + data['scene_orig']
                tmp_pos = pre_motion_scene_norm[-1].view(-1, sample_num, self.forecast_dim) if i == 0 else last_pos
                vel = pos - tmp_pos
                state = torch.cat([pos, vel], dim=-1)
                sf_feat = compute_grad_feature(state, self.sfm_params, self.sfm_learnable_hparams)
                in_arr.append(sf_feat)
                last_pos = pos
            out_in_z = torch.cat(in_arr, dim=-1)
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)
            # print("dec_in_z:", dec_in_z[...,:2])
            if approx_grad:
                pass
                dec_in_z = dec_in_z.detach()#.requires_grad_()

        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        # print("context:", context[0, 0])
        # import ipdb; ipdb.set_trace()
        data[f'{mode}_seq_out'] = seq_out
        # print("seq_out:", seq_out)
        # print("seq_out.shape:", seq_out.shape)
        # TODO
        # import ipdb; ipdb.set_trace()

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

    def forward(self, data, mode, sample_num=1, approx_grad=False, ped_one_at_a_time=False, autoregress=True, z=None, need_weights=False):
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

        if ped_one_at_a_time:
            self.decode_traj_ar_1aaat(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
                                      need_weights=need_weights, approx_grad=approx_grad)
        elif autoregress:
            self.decode_traj_ar(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
                                need_weights=need_weights, approx_grad=approx_grad)
        else:
            self.decode_traj_batch(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num)


""" AgentFormer """
class AgentFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg

        input_type = cfg.get('input_type', 'pos')
        pred_type = cfg.get('pred_type', input_type)
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = cfg.get('fut_input_type', input_type)
        dec_input_type = cfg.get('dec_input_type', [])
        self.ctx = {
            'tf_cfg': cfg.get('tf_cfg', {}),
            'nz': cfg.nz,
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
            'use_sfm': cfg.get('use_sfm', False),
            'use_sfm_context': cfg.get('use_sfm_context', False),
            'sfm_params': cfg.get('sfm_params', dict())
        }
        self.past_frames = self.ctx['past_frames']
        self.future_frames = self.ctx['future_frames']
        self.use_sfm = self.ctx['use_sfm']
        self.use_sfm_context = self.ctx['use_sfm_context']
        self.use_map = self.ctx['use_map']
        self.rand_rot_scene = cfg.get('rand_rot_scene', False)
        self.discrete_rot = cfg.get('discrete_rot', False)
        self.map_global_rot = cfg.get('map_global_rot', False)
        self.ar_train = cfg.get('ar_train', True)
        self.ped_one_at_a_time = cfg.get('ped_one_at_a_time', False)
        self.optimize_trajectory = cfg.get('optimize_trajectory', False)
        self.approx_grad = cfg.get('approx_grad', False)
        self.max_train_agent = cfg.get('max_train_agent', 100)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names
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

        # sfm map encoder
        if self.use_sfm_context:
            self.sfm_map_encoder = MapEncoder(cfg.sfm_map_encoder)
            self.ctx['sfm_map_enc_dim'] = self.sfm_map_encoder.out_dim

        # models
        self.context_encoder = ContextEncoder(cfg.context_encoder, self.ctx)
        self.future_encoder = FutureEncoder(cfg.future_encoder, self.ctx)
        if self.ctx['sfm_params'].get('learnable_hparams', False):
            print("SHOULDNT BE HEREn\n\n\n\n")
            self.recon_weight = nn.Parameter(torch.ones(1) * 5)#torch.rand(1) * 10)
            self.sample_weight = nn.Parameter(torch.ones(1) * 5) # torch.rand(1) * 10)
            self.sigma_d = nn.Parameter(torch.zeros(1))#torch.ones(1))
            self.sfm_learnable_hparams = {'recon_weight': self.recon_weight,
                                          'sample_weight': self.sample_weight,
                                          'sigma_d': self.sigma_d}
            self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx, self.loss_cfg, self.sfm_learnable_hparams)
        else:
            self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx, self.loss_cfg)
            self.sfm_learnable_hparams = None

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
        if self.training and len(data['pre_motion_3D']) > self.max_train_agent:
            in_data = {}
            ind = np.random.choice(len(data['pre_motion_3D']), self.max_train_agent).tolist()
            for key in ['pre_motion_3D', 'fut_motion_3D', 'fut_motion_mask', 'pre_motion_mask', 'heading']:
                in_data[key] = [data[key][i] for i in ind if data[key] is not None]
        else:
            in_data = data

        self.data = defaultdict(lambda: None)
        self.data['cfg'] = self.cfg
        self.data['batch_size'] = len(in_data['pre_motion_3D'])
        self.data['agent_num'] = len(in_data['pre_motion_3D'])
        self.data['pre_motion'] = torch.stack(in_data['pre_motion_3D'], dim=0).to(device).transpose(0, 1).contiguous()
        self.data['fut_motion'] = torch.stack(in_data['fut_motion_3D'], dim=0).to(device).transpose(0, 1).contiguous()
        self.data['fut_motion_orig'] = torch.stack(in_data['fut_motion_3D'], dim=0).to(device)   # future motion without transpose
        self.data['fut_mask'] = torch.stack(in_data['fut_motion_mask'], dim=0).to(device)
        self.data['pre_mask'] = torch.stack(in_data['pre_motion_mask'], dim=0).to(device)
        scene_orig_all_past = self.cfg.get('scene_orig_all_past', False)
        if scene_orig_all_past:
            self.data['scene_orig'] = self.data['pre_motion'].view(-1, 2).mean(dim=0)
        else:
            self.data['scene_orig'] = self.data['pre_motion'][-1].mean(dim=0)
        if in_data['heading'] is not None:
            self.data['heading'] = torch.tensor(in_data['heading']).float().to(device)

        # rotate the scene
        if self.rand_rot_scene and self.training:
            if self.discrete_rot:
                theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2
            for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
                self.data[f'{key}'], self.data[f'{key}_scene_norm'] = rotation_2d_torch(self.data[key], theta, self.data['scene_orig'])
            if in_data['heading'] is not None:
                self.data['heading'] += theta
        else:
            theta = torch.zeros(1).to(device)
            for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
                self.data[f'{key}_scene_norm'] = self.data[key] - self.data['scene_orig']   # normalize per scene

        self.data['pre_vel'] = self.data['pre_motion'][1:] - self.data['pre_motion'][:-1, :]
        self.data['fut_vel'] = self.data['fut_motion'] - torch.cat([self.data['pre_motion'][[-1]], self.data['fut_motion'][:-1, :]])
        self.data['cur_motion'] = self.data['pre_motion'][[-1]]
        self.data['pre_motion_norm'] = self.data['pre_motion'][:-1] - self.data['cur_motion']   # normalize pos per agent
        self.data['fut_motion_norm'] = self.data['fut_motion'] - self.data['cur_motion']
        if in_data['heading'] is not None:
            self.data['heading_vec'] = torch.stack([torch.cos(self.data['heading']), torch.sin(self.data['heading'])], dim=-1)

        # agent maps
        if self.use_map:
            scene_map = data['scene_map']
            scene_points = np.stack(in_data['pre_motion_3D'])[:, -1] * data['traj_scale']
            if self.map_global_rot:
                patch_size = [50, 50, 50, 50]
                rot = theta.repeat(self.data['agent_num']).cpu().numpy() * (180 / np.pi)
            else:
                patch_size = [50, 10, 50, 90]
                rot = -np.array(in_data['heading'])  * (180 / np.pi)
            self.data['agent_maps'] = scene_map.get_cropped_maps(scene_points, patch_size, rot).to(device)

        # sfm context
        if self.use_sfm_context:
            scene_map = data['scene_map']
            scene_points = np.stack(in_data['pre_motion_3D'])[:, -1] * data['traj_scale']
            patch_size = [50, 10, 50, 90]
            rot = -np.array(in_data['heading']) * (180 / np.pi)
            self.data['agent_maps'] = scene_map.get_cropped_maps(scene_points, patch_size, rot).to(device)
            pos = self.data['pre_motion'][i]
            vel = self.data['pre_vel'][max(0, i - 1)]
            state = torch.cat([pos, vel], dim=-1)
            grad = compute_grad_feature(state, self.cfg.sfm_params, self.sfm_learnable_hparams)


        # agent shuffling
        if self.training and self.ctx['agent_enc_shuffle']:
            self.data['agent_enc_shuffle'] = torch.randperm(self.ctx['max_agent_len'])[:self.data['agent_num']].to(device)
        else:
            self.data['agent_enc_shuffle'] = None

        conn_dist = self.cfg.get('conn_dist', 100000.0)
        cur_motion = self.data['cur_motion'][0]
        if conn_dist < 1000.0:
            threshold = conn_dist / self.cfg.traj_scale
            pdist = F.pdist(cur_motion)
            D = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
            D[np.triu_indices(cur_motion.shape[0], 1)] = pdist
            D += D.T
            mask = torch.zeros_like(D)
            mask[D > threshold] = float('-inf')
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        self.data['agent_mask'] = mask

        # social force features
        if self.use_sfm_context:
            # self.grid = torch.array()
            pass
        if self.use_sfm:
            # past
            sf_feat = []
            for i in range(self.past_frames):
                pos = self.data['pre_motion'][i]
                vel = self.data['pre_vel'][max(0, i - 1)]
                state = torch.cat([pos, vel], dim=-1)
                grad = compute_grad_feature(state, self.cfg.sfm_params, self.sfm_learnable_hparams)
                sf_feat.append(grad)
            sf_feat = torch.stack(sf_feat)
            self.data['pre_sf_feat'] = sf_feat
            # future
            sf_feat = []
            for i in range(self.future_frames):
                pos = self.data['fut_motion'][i]
                vel = self.data['fut_vel'][i]
                state = torch.cat([pos, vel], dim=-1)
                grad = compute_grad_feature(state, self.cfg.sfm_params, self.sfm_learnable_hparams)
                if torch.any(torch.isnan(grad)):
                    print('NaN:', torch.where(torch.isnan(grad)))
                sf_feat.append(grad)
            sf_feat = torch.stack(sf_feat)
            self.data['fut_sf_feat'] = sf_feat

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
        self.future_decoder(self.data, mode='train', ped_one_at_a_time=self.ped_one_at_a_time, autoregress=self.ar_train)
        if self.compute_sample:
            self.inference(sample_num=self.loss_cfg['sample']['k'])
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
                            ped_one_at_a_time=self.ped_one_at_a_time, autoregress=True, need_weights=need_weights)
        return self.data[f'{mode}_dec_motion'], self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            # todo HEY here
            # if 'sfm' in loss_name:
            #     params = [self.data, self.loss_cfg[loss_name], self.sfm_learnable_hparams]
            # else:
            params = [self.data, self.loss_cfg[loss_name]]
            loss, loss_unweighted = loss_func[loss_name](*params)
            total_loss += loss.squeeze()
            # if 'mse' in loss_name:
            #     print("loss_unweighted:", loss_unweighted)
            #     print("loss:", loss)
            #     import ipdb; ipdb.set_trace()
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict
