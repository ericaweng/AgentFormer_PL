"""generate configs for new experiments"""
import os
import itertools


def save_new_cfg(filename, cfg):
    print("cfg:", cfg)
    print(f"save to {filename}?")
    # import ipdb; ipdb.set_trace()
    with open(filename, 'w') as f:
        f.write(cfg)


def main():
    cfgs_path = 'cfg'
    datasets = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    total_new_cfgs = 0


    for dset in ['trajnet-sdd']: # zara2', 'eth', 'univ', 'hotel', 'zara1']:#,
        for dlow_weight_joint in [5,7.5,10]:
            for dlow_weight_marginal in [5]:#,7.5,10]: 1,2,4]:#[
                # for marg_weight in [0.75, 1]:
                # for jr_weight in [0.75]:
                prefix = f'{cfgs_path}/eth_ucy/{dset}' if dset != 'trajnet-sdd' else f'{cfgs_path}/trajnet_sdd'
                # source_filename = f'{prefix}/{dset}_agentformer_pre.yml'
                source_filename = f'{prefix}/{dset}_agentformer.yml'
                with open(source_filename, 'r') as f:
                    cfg = f.read()
                save_prefix = f"{prefix}_tune"
                # replace root dir
                cfg = cfg.replace(': results\n', ': results-joint\n')
                # cfg = cfg.replace(': results', ': results-joint')
                # ----------------------------------------------------
                # agentformer_pre with joint recon loss
    #             cfg2 = cfg.replace('''  sample:
    # weight: 1.0''', f'''  joint_sample:
    # weight: {jr_weight}''')
    #             dest_filename = f'{save_prefix}/{dset}_af_pre_jr-{jr_weight}.yml'
    #             if not os.path.exists(os.path.dirname(dest_filename)):
    #                 os.makedirs(os.path.dirname(dest_filename))
    #             save_new_cfg(dest_filename, cfg2)

                marg_weight = 1
                jr_weight = 1

  #               cfg2 = cfg.replace('''  sample:
  #   weight: 1.0''', f'''  joint_sample:
  #   weight: {jr_weight}
  # sample:
  #   weight: {marg_weight}''')
  #               dest_filename = f'{save_prefix}/{dset}_af_pre_mg-{marg_weight}-jr-{jr_weight}.yml'
  #               if not os.path.exists(os.path.dirname(dest_filename)):
  #                   os.makedirs(os.path.dirname(dest_filename))
  #               save_new_cfg(dest_filename, cfg2)
  #
    #             best_pre_weight = 0.75
    #
    #             cfg2 = cfg.replace('''  recon:
    # weight: 5.0''', f'''  joint_recon:
    # weight: {dlow_weight}''')
    #             cfg2 = cfg2.replace(f'pred_cfg: {dset}_agentformer_pre', f'pred_cfg: {dset}_af_pre_jr-{best_pre_weight}')
    #             dest_filename = f'{save_prefix}/{dset}_af_jrjr_{best_pre_weight}-w-{dlow_weight}.yml'
    #             if not os.path.exists(os.path.dirname(dest_filename)):
    #                 os.makedirs(os.path.dirname(dest_filename))
    #             save_new_cfg(dest_filename, cfg2)

                cfg2 = cfg.replace('''  recon:
    weight: 5.0''', f'''  recon:
    weight: {dlow_weight_marginal}
  joint_recon:
    weight: {dlow_weight_joint}''')
                cfg2 = cfg2.replace(f'pred_cfg: {dset}_agentformer_pre',
                                    f'pred_cfg: {dset}_af_pre_mg-{marg_weight}-jr-{jr_weight}')
                dest_filename = f'{save_prefix}/{dset}_af_mg-{marg_weight},{dlow_weight_marginal}_jr-{jr_weight},{dlow_weight_joint}.yml'
                if not os.path.exists(os.path.dirname(dest_filename)):
                    os.makedirs(os.path.dirname(dest_filename))
                save_new_cfg(dest_filename, cfg2)

                # ----------------------------------------------------
                total_new_cfgs += 1
                continue

    print(f"total new cfgs: {total_new_cfgs}")


if __name__ == '__main__':
    main()
