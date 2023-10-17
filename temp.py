# ewta stuff
def ewta_recon_loss(data, cfg):
    """soft multiple choice loss': weighted average over samples, rather than min"""
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    print("data:", data.keys())
    import ipdb; ipdb.set_trace()
    loss_unweighted = torch.topk(dist, k, largest=False, dim=-1).values.mean(-1)
    print("loss_unweighted:", loss_unweighted)
    import ipdb; ipdb.set_trace()
    # loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted

def ewta_joint_recon_loss(data, cfg):
    """soft multiple choice loss': weighted average over samples, rather than min"""
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)  # (num_peds, num_samples)
    if cfg.get('normalize', True):
        samples = dist.mean(axis=0)
    else:
        samples = dist.sum(axis=0)  # (num_samples)
    loss_unweighted = torch.topk(samples, k, largest=False).values.mean()
    # loss_unweighted = samples.min()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


'sample_sfm': compute_sample_sfm,
'ewta_joint_recon': ewta_joint_recon_loss,
'ewta_recon': ewta_recon_loss,


# gen configs already done

# agentformer_pre with joint recon loss
cfg2 = cfg.replace('''  sample:
weight: 1.0''', '''  joint_sample:
weight: 5.0''')
dest_filename = f'{prefix}/{dset}_af_pre_jr_w-5.yml'
if not os.path.exists(os.path.dirname(dest_filename)):
    os.makedirs(os.path.dirname(dest_filename))
save_new_cfg(dest_filename, cfg2)

# higher joint recon weights
cfg2 = cfg.replace('''  recon:
weight: 5.0''', '''  joint_recon:
weight: 10.0''')
dest_filename = f'{prefix}/{dset}_af_dlow_jr_w-10.yml'
if not os.path.exists(os.path.dirname(dest_filename)):
    os.makedirs(os.path.dirname(dest_filename))
save_new_cfg(dest_filename, cfg2)

cfg2 = cfg.replace('''  recon:
weight: 5.0''', '''  joint_recon:
weight: 12.5''')
dest_filename = f'{prefix}/{dset}_af_dlow_jr_w-12.5.yml'
if not os.path.exists(os.path.dirname(dest_filename)):
    os.makedirs(os.path.dirname(dest_filename))
save_new_cfg(dest_filename, cfg2)

# trajnet sdd

weight = 30.0
b = 1.0
cfg2 = f"""# {dset} plain dlow with collision rejection \n{cfg}

# ------------------- Testing Parameters -------------------------
collisions_ok: False
collision_rad: 0.1
"""
dest_filename = f'{cfgs_path}/trajnet-sdd_agentformer_nocol.yml'
save_new_cfg(dest_filename, cfg2)

# replace gaussian distance fn with laplacian distance fn; use joint recon loss
cfg2 = cfg.replace("""recon:
    weight: 5.0""", f"""joint_recon:
    weight: 5.0  
  sample_sfm:
    weight: {weight:0.1f}
    sfm_params:
      dist_fn: laplacian
      b: {b}
      use_w: False
      loss_reduce: mean""")
cfg2 = f"# {dset} sfm laplacian distance fn. weight {weight:0.1f} b {b:0.2f} training cfg for dlow models trained w/ collision loss\n{cfg2}"
dest_filename = f'{cfgs_path}/trajnet-sdd_agentformer_dlow_sfm_jr_laplacian_w-{weight:0.1f}_b-{b:0.2f}.yml'
save_new_cfg(dest_filename, cfg2)
weight = 25.0
sigma_d = 0.75
cfg2 = cfg.replace("""recon:
    weight: 5.0""", f"""joint_recon:
    weight: 5.0  
  sample_sfm:
    weight: {weight:0.1f}
    sfm_params:
      sigma_d: {sigma_d}
      beta: 1.2
      use_w: False
      loss_reduce: mean""")

cfg2 = f"# {dset} sfm weight {weight:0.1f} sigma_d {sigma_d:0.2f} training cfg for dlow models trained w/ collision loss\n{cfg2}"
# write the new cfg to new file with new filename: joint recon loss with train noise
dest_filename = f'{cfgs_path}/trajnet-sdd_agentformer_dlow_sfm_jr_w-25.0_s-0.75.yml'
save_new_cfg(dest_filename, cfg2)

# replace marginal_recon with joint_recon
cfg = cfg.replace('recon:', 'joint_recon:')
# write the new cfg to new file with new filename: joint recon loss
dest_filename = f'{cfgs_path}/{dset}_agentformer_dlow_joint_recon_loss.yml'
save_new_cfg(dest_filename, cfg)

cfg = cfg.replace('recon:', 'joint_recon:')
cfg2 = cfg.replace('train_w_mean: true\n', 'test_w_mean: false\n')
cfg2 = f"""# training cfg for models trained with dlow noise with joint recon loss term (instead of the original marginal term)\n{cfg2}
# ------------------- Testing Parameters -------------------------
collisions_ok: False
collision_rad: 0.1
"""
dest_filename = f'{cfgs_path}/{dset}_agentformer_dlow_joint_recon_loss_noise_nocol.yml'
save_new_cfg(dest_filename, cfg2)


# replace gaussian distance fn with laplacian distance fn; use joint recon loss
cfg2 = cfg.replace("""recon:
  weight: 5.0""", f"""joint_recon:
  weight: 5.0  
sample_sfm:
  weight: {weight:0.1f}
  sfm_params:
    dist_fn: laplacian
    b: {b}
    use_w: False
    loss_reduce: mean""")
cfg2 = f"# {dset} sfm laplacian distance fn. weight {weight:0.1f} b {b:0.2f} training cfg for dlow models trained w/ collision loss\n{cfg2}"
dest_filename = f'{cfgs_path}/{dset}_dlow_sfm_laplacian/{dset}_agentformer_dlow_sfm_jr_laplacian_w-{weight:0.1f}_b-{b:0.2f}.yml'
if not os.path.exists(os.path.dirname(dest_filename)):
    os.makedirs(os.path.dirname(dest_filename))
save_new_cfg(dest_filename, cfg2)

# 300 modes
                cfg2 = cfg.replace('sample_k                     : 20',
                                   f'sample_k                     : {k}')
                cfg2 = f"# training cfg for dlow w/ {k} sample modes (instead of 20) \n{cfg2}"
                # write the new cfg to new file with new filename: joint recon loss with train noise
                dest_filename = f'{cfgs_path}/{dset}_nk/{dset}_agentformer_dlow_nk-{k}.yml'
                if not os.path.exists(os.path.dirname(dest_filename)):
                    os.makedirs(os.path.dirname(dest_filename))
                save_new_cfg(dest_filename, cfg2)

# dlow sfm collision loss try 2 weight and sigma_d
cfg2 = cfg.replace("""recon:
  weight: 5.0""", f"""recon:
  weight: 5.0  
sample_sfm:
  weight: {weight:0.1f}
  sfm_params:
    sigma_d: {sigma_d}
    beta: 1.2
    use_w: False
    loss_reduce: mean""")
cfg2 = f"# {dset} sfm weight {weight:0.1f} sigma_d {sigma_d:0.2f} training cfg for dlow models trained w/ collision loss\n{cfg2}"
# write the new cfg to new file with new filename: joint recon loss with train noise
dest_filename = f'{cfgs_path}/{dset}_dlow_sfm/{dset}_agentformer_dlow_sfm_w-{weight:0.1f}_s-{sigma_d:0.2f}.yml'
if f'agentformer_dlow_sfm_w-{weight:0.1f}_s-{sigma_d:0.2f}' not in cfgs_to_do:
    continue
if not os.path.exists(os.path.dirname(dest_filename)):
    os.makedirs(os.path.dirname(dest_filename))
save_new_cfg(dest_filename, cfg2)

# 300 modes
cfg2 = cfg.replace('sample_k                     : 20',
                   f'sample_k                     : {k}')
cfg2 = f"# training cfg for dlow w/ {k} sample modes (instead of 20) \n{cfg2}"
# write the new cfg to new file with new filename: joint recon loss with train noise
dest_filename = f'{cfgs_path}/{dset}_nk/{dset}_agentformer_dlow_nk-{k}.yml'
save_new_cfg(dest_filename, cfg2)

# sfm (collision loss) try 1 just weight
cfg2 = cfg.replace("""  recon:
    weight: 5.0""", f"""  recon:
    weight: 5.0  
  sample_sfm:
    weight: {weight:0.1f}
    sfm_params:
      sigma_d: 1.0
      beta: 1.2
      use_w: False
      loss_reduce: mean""")
cfg2 = f"# {dset} sfm weight {weight:0.1f} training cfg for dlow models trained w/ collision loss\n{cfg2}"
# write the new cfg to new file with new filename: joint recon loss with train noise
dest_filename = f'{cfgs_path}/{dset}/{dset}_agentformer_dlow_sfm_{"7-5" if weight == 7.5 else str(weight)}.yml'
save_new_cfg(dest_filename, cfg2)

cfg = cfg.replace('recon:', 'joint_recon:')
# replace train_w_mean: True
cfg2 = cfg.replace('train_w_mean: true\n', 'test_w_mean: false\n')
cfg2 = f"""# training cfg for models trained with dlow noise with joint recon loss term (instead of the original marginal term)\n{cfg2}
# ------------------- Testing Parameters -------------------------
collisions_ok: False
collision_rad: 0.1
"""
# write the new cfg to new file with new filename: joint recon loss with train noise
rm_fn = f'{cfgs_path}/{dset}/{dset}_agentformer_dlow_joint_recon_loss_noise.yml'
dest_filename = f'{cfgs_path}/{dset}/{dset}_agentformer_dlow_joint_recon_loss_noise_nocol.yml'
save_new_cfg(dest_filename, cfg2)

# train normal dlow (marginal loss) with noise
# replace train_w_mean: True
cfg2 = cfg.replace('train_w_mean: true\n', '')
cfg2 = f"# {dset} training cfg for models trained with dlow noise instead of mean\n{cfg2}"
# write the new cfg to new file with new filename: joint recon loss with train noise
dest_filename = f'{cfgs_path}/{dset}/{dset}_agentformer_dlow_noise.yml'
save_new_cfg(dest_filename, cfg2)

# replace marginal_recon with joint_recon
cfg = cfg.replace('recon:', 'joint_recon:')
# write the new cfg to new file with new filename: joint recon loss
dest_filename = f'{cfgs_path}/{dset}/{dset}_agentformer_dlow_joint_recon_loss.yml'
save_new_cfg(dest_filename, cfg)


                # train normal dlow (marginal loss) with noise
                # replace train_w_mean: True
                cfg2 = cfg.replace('train_w_mean: true\n', 'test_w_mean: false\n')
                cfg2 = f"""# {dset} training cfg for models trained with dlow noise instead of mean\n{cfg2}
# ------------------- Testing Parameters -------------------------
collisions_ok: False
collision_rad: 0.1
"""
                # write the new cfg to new file with new filename: joint recon loss with train noise
                dest_filename = f'{cfgs_path}/{dset}/{dset}_agentformer_dlow_noise.yml'
                save_new_cfg(dest_filename, cfg2)
