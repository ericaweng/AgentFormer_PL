from torchmetrics import Metric
from metrics import get_collisions_mat_old_torch, check_collision_per_sample_no_gt


def decode_traj_ar(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
                   need_weights=False):
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

    for i in range(self.future_frames):
        traj_in = dec_in_z.view(-1, dec_in_z.shape[-1])
        print("dec_in_z.shape (raw context + latent + other info. input into the decoder, each round of preds):",
              dec_in_z.shape)
        if self.input_norm is not None:
            traj_in = self.input_norm(traj_in)
        print("traj_in.shape:", traj_in.shape)
        tf_in = self.input_fc(traj_in).view(dec_in_z.shape[0], -1, self.model_dim)
        print("tf_in.shape:", tf_in.shape)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle,
                                     t_offset=self.past_frames - 1 if self.pos_offset else 0)
        print("tf_in_pos.shape:", tf_in_pos.shape)
        # tf_in_pos = tf_in
        mem_mask = generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
        print("mem_mask.shape:", mem_mask.shape)
        tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)
        print("tgt_mask.shape:", tgt_mask.shape)
        print("before tf decoder")

        tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask,
                                               num_agent=data['agent_num'], need_weights=need_weights)
        print("tf_out.shape (after decoder):", tf_out.shape)

        out_tmp = tf_out.view(-1, tf_out.shape[-1])
        print("out_tmp.shape:", out_tmp.shape)
        if self.out_mlp_dim is not None:
            out_tmp = self.out_mlp(out_tmp)
        seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
        print("seq_out.shape:", seq_out.shape)
        if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
            norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
            print("norm_motion.shape:", norm_motion.shape)
            if self.sn_out_type == 'vel':
                norm_motion = torch.cumsum(norm_motion, dim=0)
            if self.sn_out_heading:
                angles = data['heading'].repeat_interleave(sample_num)
                norm_motion = rotation_2d_torch(norm_motion, angles)[0]
            seq_out = norm_motion + pre_motion_scene_norm[[-1]]
            seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
            print("seq_out.shape after adding norm motion:", seq_out.shape)
        if self.ar_detach:
            out_in = seq_out[-agent_num:].clone().detach()
        else:
            out_in = seq_out[-agent_num:]
        print("out_in.shape (after taking just the current timestep predictions and relevant agents "
              "for the next prediction):", out_in.shape)
        # create dec_in_z
        in_arr = [out_in, z_in]
        print("z_in.shape (to be concatted with out_in):", z_in.shape)
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
        print("out_in_z.shape (to be concatted with context + previous predicted agent-ts, for next round):",
              out_in_z.shape)
        dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)
        print("dec_in_z.shape:", dec_in_z.shape)
        import ipdb;
        ipdb.set_trace()

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

    dec_motion = dec_motion.transpose(0, 1).contiguous()  # M x frames x 7
    if mode == 'infer':
        dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])  # M x Samples x frames x 3
    data[f'{mode}_dec_motion'] = dec_motion
    if need_weights:
        data['attn_weights'] = attn_weights


def point_to_segment_dist_old(x1, y1, x2, y2, p1, p2):
    """
    Calculate the closest distance between start(p1, p2) and a line segment with two endpoints (x1, y1), (x2, y2)
    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((p1 - x1, p2 - y1), axis=-1)

    u = ((p1 - x1) * px + (p2 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest start to (p1, p2) on the line segment
    x = x1 + u * px
    y = y1 + u * py
    return np.linalg.norm((x - p1, y - p2), axis=-1)


def get_collisions_mat_old_torch(pred_traj_fake, threshold):
    """threshold: radius + discomfort distance of agents"""
    pred_traj_fake = pred_traj_fake.transpose(1, 0)
    ts, num_peds, _ = pred_traj_fake.shape
    collision_mat = torch.full((ts, num_peds, num_peds), False)
    collision_mat_vals = torch.full((ts, num_peds, num_peds), np.inf)
    # test initial timesteps
    for ped_i, x_i in enumerate(pred_traj_fake[0]):
        for ped_j, x_j in enumerate(pred_traj_fake[0]):
            if ped_i == ped_j:
                continue
            closest_dist = torch.norm(x_i - x_j) - threshold * 2
            if closest_dist < 0:
                collision_mat[0, ped_i, ped_j] = True
            collision_mat_vals[0, ped_i, ped_j] = closest_dist

    # test t-1 later timesteps
    for t in range(ts - 1):
        for ped_i, ((ped_ix, ped_iy), (ped_ix1, ped_iy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
            for ped_j, ((ped_jx, ped_jy), (ped_jx1, ped_jy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
                if ped_i == ped_j:
                    continue
                px = ped_ix - ped_jx
                py = ped_iy - ped_jy
                ex = ped_ix1 - ped_jx1
                ey = ped_iy1 - ped_jy1
                closest_dist = point_to_segment_dist_old(px, py, ex, ey, 0, 0) - threshold * 2
                if closest_dist < 0:
                    collision_mat[t+1, ped_i, ped_j] = True
                collision_mat_vals[t + 1, ped_i, ped_j] = closest_dist

    return torch.any(torch.any(collision_mat, dim=0), dim=0), collision_mat


def get_collisions_mat_old(pred_traj_fake, threshold):
    """pred_traj_fake: shape (num_peds, num_samples, ts, 2)
    threshold: radius + discomfort distance of agents"""
    pred_traj_fake = pred_traj_fake.transpose(1, 0, 2)
    ts, num_peds, _ = pred_traj_fake.shape
    collision_mat = np.full((ts, num_peds, num_peds), False)
    collision_mat_vals = np.full((ts, num_peds, num_peds), np.inf)
    # test initial timesteps
    for ped_i, x_i in enumerate(pred_traj_fake[0]):
        for ped_j, x_j in enumerate(pred_traj_fake[0]):
            if ped_i == ped_j:
                continue
            closest_dist = np.linalg.norm(x_i - x_j) - threshold * 2
            if closest_dist < 0:
                collision_mat[0, ped_i, ped_j] = True
            collision_mat_vals[0, ped_i, ped_j] = closest_dist

    # test t-1 later timesteps
    for t in range(ts - 1):
        for ped_i, ((ped_ix, ped_iy), (ped_ix1, ped_iy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
            for ped_j, ((ped_jx, ped_jy), (ped_jx1, ped_jy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
                if ped_i == ped_j:
                    continue
                px = ped_ix - ped_jx
                py = ped_iy - ped_jy
                ex = ped_ix1 - ped_jx1
                ey = ped_iy1 - ped_jy1
                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist_old(px, py, ex, ey, 0, 0) - threshold * 2
                if closest_dist < 0:
                    collision_mat[t+1, ped_i, ped_j] = True
                collision_mat_vals[t + 1, ped_i, ped_j] = closest_dist

    return np.any(np.any(collision_mat, axis=0), axis=0), collision_mat  # collision_mat_pred_t_bool


def compute_ADE(pred_arr, gt_arr, return_sample_vals=False, return_argmin=False, **kwargs):
    ade = 0.0
    ped_ades_per_sample = 0
    argmins = []
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist.mean(axis=-1)  # samples
        ped_ades_per_sample += dist
        ade += dist.min(axis=0)  # (1, )
        argmins.append(dist.argmin(axis=0))
    ade /= len(pred_arr)
    return_vals = [ade]
    if return_sample_vals:  # for each sample: the avg ped ADE
        return_vals.append(ped_ades_per_sample / len(pred_arr))
    if return_argmin:  # for each ped: index of sample that is argmin
        return_vals.append(np.array(argmins))
    # ade2, ped_ades_per_sample2, argmin2 = compute_ADE_fast(pred_arr, gt_arr, True, True)
    # assert np.all(np.abs(ade2 - ade) < 1e-6), f"ade not equal\n{ade}\n\n{ade2}"
    # assert np.all(np.abs(ped_ades_per_sample / len(pred_arr) - ped_ades_per_sample2) < 1e-6), f"ped_ades_per_sample not equal\n{ped_ades_per_sample}\n\n{ped_ades_per_sample2}"
    # assert np.all(np.abs(np.array(argmins) - argmin2) < 1e-6), f"argmins not equal\n{argmins}\n\n{argmin2}"
    return return_vals[0] if len(return_vals) == 1 else return_vals


def compute_FDE(pred_arr, gt_arr, return_sample_vals=False, **kwargs):
    """pred_arr (num_peds, num_samples): """
    fde = 0.0
    peds = 0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist[..., -1]  # samples
        peds += dist
        fde += dist.min(axis=0)  # (1, )
    fde /= len(pred_arr)
    fde2 = compute_FDE_fast(pred_arr, gt_arr, return_sample_vals=False, return_argmin=False)
    assert np.all(np.abs(fde2 - fde) < 1e-6), f"ade not equal\n{fde}\n\n{fde2}"
    if return_sample_vals:
        return fde, peds / len(pred_arr)
    return fde


self.stats = Stats(self.collision_rad)

class Stats(Metric):
    full_state_update = True
    higher_is_better = False
    is_differentiable = True

    def __init__(self, collision_rad):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.add_state("num_peds", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("ade", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("fde", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("cr_max", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("cr_mean", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.collision_rad = collision_rad

    def update(self, gt: torch.Tensor, preds: torch.Tensor):
        assert gt.shape[0] == preds.shape[0]
        assert gt.shape[1:] == preds.shape[2:], f"gt.shape[1:] ({gt.shape[1:]}) != preds.shape[2:] ({preds.shape[2:]})"
        self.num_peds += gt.shape[0]
        for pred, ped_gt in zip(preds, gt):
            diff = pred - ped_gt.unsqueeze(0)  # samples x frames x 2
            dist = torch.norm(diff, dim=-1)  # samples x frames
            ade_dist = dist.mean(dim=-1)  # samples
            self.ade += ade_dist.min(dim=0)[0].item()  # (1, )
            fde_dist = dist[..., -1]  # samples
            self.fde += fde_dist.min(dim=0)[0].item()  # (1, )

        n_ped, n_sample, _, _ = preds.shape
        col_pred = torch.zeros((n_sample))  # cr_pred
        col_mats = []
        if n_ped > 1:
            for sample_idx in range(n_sample):
                n_ped_with_col_pred, col_mat = get_collisions_mat_old_torch(preds[:,sample_idx], self.collision_rad)
                col_mats.append(col_mat)
                col_pred[sample_idx] += (n_ped_with_col_pred.sum())

        self.cr_mean += col_pred.mean(dim=0).item()
        self.cr_max += col_pred.max(dim=0)[0].item()
        # self.cr_min = col_pred.min(axis=0)

    def compute(self):
        return [self.ade / self.num_peds, self.fde / self.num_peds, self.cr_max / self.num_peds, self.cr_mean / self.num_peds]
