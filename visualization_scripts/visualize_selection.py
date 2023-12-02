import glob

from utils.utils import mkdir_if_missing
from eval import *
from viz_utils import plot_anim_grid


def get_traj_from_file(data_file, indices=None):
    # for reconsutrction or deterministic
    if isfile(data_file):
        all_traj = np.loadtxt(data_file, delimiter=' ', dtype='object')[:, indices].astype(np.float32)#, dtype='float32')  # (frames x agents) x 4
        all_traj = np.expand_dims(all_traj, axis=0)  # 1 x (frames x agents) x 4
    # for stochastic with multiple samples
    elif isfolder(data_file):
        sample_list, _ = load_list_from_folder(data_file)
        sample_all = []
        for sample in sample_list[:20]:
            sample = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            sample_all.append(sample)
        all_traj = np.stack(sample_all, axis=0)  # samples x (framex x agents) x 4
    else:
        assert False, 'error'
    return all_traj


def align_gt(pred, gt, no_obs=False):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    obs_len = 8
    if not no_obs:
        obs_idx = np.arange(min(index_list1) - obs_len, min(index_list1))
        assert np.all(obs_idx >= 0)
        obs = gt[obs_idx]
        assert obs.shape == (obs_len, 4)
    else:
        obs=None
    return pred_new, gt_new, obs


def main1(args):
    """plot:
    1. GT
    2. AF
    3. AF + col rej
    4. AF + Dagger"""

    seq_name = 'crowds_zara02'
    cfgs = ['zara2_agentformer_pre', 'zara2_agentformer_pre_nocol', 'zara2_dagger_tune_test']#'zara2_agentformer_pre_nocol']#, 'zara2_dagger_tune_test']  # , 'zara2_dagger_tune'

    cfg = cfgs[-1]
    last_epoch = int(str(sorted(glob.glob(os.path.join(f'results/{cfg}/results/', 'epoch_*')))[-1][-4:]))
    dir_to_search_frames = f'results/{cfg}/results/epoch_{last_epoch:04d}/test/samples/{seq_name}/*'
    print("dir_to_search_frames:", dir_to_search_frames)
    all_frames = [int(f.split('_')[-1]) for f in glob.glob(dir_to_search_frames)]
    print("len(all_frames):", len(all_frames))

    if args.multiprocess:
        with multiprocessing.Pool(multiprocessing.cpu_count() - 5) as p:
            p.starmap(plot_frame, list(zip(all_frames, [cfgs for _ in range(len(all_frames))])))
    else:
        for frame in all_frames:
            plot_frame(frame, cfgs)


def plot_frame(frame, cfgs):
    plot_args_lists = [None] * 12
    seq_name = 'crowds_zara02'
    skip = False
    gt_dir = f'../datasets/eth_ucy/zara2'
    indices = [0, 1, 13, 15]

    # load GT raw data
    gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name + '.txt'))
    gt_raw = []
    for line_data in gt_data:
        line_data = np.array([line_data.split(' ')])[:, indices][0].astype('float32')
        if line_data[1] == -1: continue
        gt_raw.append(line_data)
    gt_raw = np.stack(gt_raw)

    pred_trajs = None
    cfg_previous = None
    agent_traj = None
    for cfg_i, cfg in enumerate(cfgs):
        last_epoch = int(str(sorted(glob.glob(os.path.join(f'results/{cfg}/results/', 'epoch_*')))[-1][-4:]))
        samples_dir = f'results/{cfg}/results/epoch_{last_epoch:04d}/test/samples/{seq_name}/frame_{frame:06d}'
        # gather samples
        all_traj = get_traj_from_file(samples_dir, indices)

        # convert raw data to our format for evaluation
        id_list = np.unique(all_traj[..., 1])
        # skip sequences with only 1 ped
        if len(id_list) == 1:
            skip = True
            break
        agent_trajs_previous = agent_traj
        agent_traj = []
        gt_traj = []
        obs_traj = []
        for idx in id_list:
            # GT traj
            gt_idx = gt_raw[gt_raw[:, 1] == idx]  # frames x 4
            # predicted traj
            ind = np.unique(np.where(all_traj[..., 1] == idx)[1].tolist())
            pred_idx = all_traj[:, ind, :]  # sample x frames x 4
            # filter data
            pred_idx, gt_idx, obs = align_gt(pred_idx, gt_idx)
            obs = obs[..., 2:]

            # append
            agent_traj.append(pred_idx)
            # if cfg_i == 0:
            gt_traj.append(gt_idx)
            obs_traj.append(obs)

        cr, all_crs, col_mats = compute_CR(agent_traj, return_sample_crs=True, return_collision_mat=True, collision_rad=0.1)
        # col_mats = np.stack(col_mats)
        ade, all_ades = compute_ADE(agent_traj, gt_traj, return_sample_ades=True)
        num_agents = len(agent_traj)
        fde, all_fdes = compute_FDE(agent_traj, gt_traj, return_sample_fdes=True)
        text_fixed = f'CR: {cr:0.1f}\nADE: {ade:0.2f}\nFDE: {fde:0.2f}'
        pred_trajs_previous = pred_trajs
        pred_trajs = np.stack(agent_traj).transpose(1, 2, 0, 3)  # (num_samples, pred_steps, num_agents, 2)
        pred_obs_traj = np.stack(obs_traj).transpose(1, 0, 2)  # (obs_steps, num_agents, 2)
        pred_gt_traj = np.stack(gt_traj).transpose(1, 0, 2)  # (pred_steps, num_agents, 2)

        if cfg_i == 0:  # gt
            args_dict = {'plot_title': 'obs + gt only',
                         'obs_traj': pred_obs_traj,
                         'pred_traj_gt': pred_gt_traj,}
            # plot_args_lists.append(args_dict)
            plot_args_lists[0] = args_dict

        cfg = 'AF+dagger' if 'dagger' in cfg else 'AF+no_collisions' if 'nocol' in cfg else 'plain-AF'
        if 'dagger' in cfg:  # and agent_trajs_previous is not None:
            # print("all_crs:", all_crs)
            # print("cr:", cr)
            max_cr = np.max(all_crs)
            if max_cr > 0:
                max_cr_idxes = np.argsort(all_crs)[::-1]
                max_cr_idx = max_cr_idxes[0]#np.argmax(all_crs)
                text_fixed = f'CR: {all_crs[max_cr_idx]:0.1f}\nADE: {all_ades[max_cr_idx]:0.2f}\nFDE: {all_fdes[max_cr_idx]:0.2f}'
                args_dict = {'plot_title': cfg + ', sample w/ max CR',
                             'obs_traj': pred_obs_traj,
                             # 'pred_traj_gt': pred_gt_traj,
                             'pred_traj_fake': pred_trajs[max_cr_idx],
                             'text_fixed': text_fixed,
                             'collision_mats': col_mats[max_cr_idx]}
                # plot_args_lists.append(args_dict)
                plot_args_lists[7] = args_dict

                text_fixed = f'CR: {all_crs[max_cr_idxes[1]]:0.1f}\nADE: {all_ades[max_cr_idxes[1]]:0.2f}\nFDE: {all_fdes[max_cr_idxes[1]]:0.2f}'
                args_dict = {'plot_title': cfg + ', sample w/ second-max CR',
                             'obs_traj': pred_obs_traj,
                             # 'pred_traj_gt': pred_gt_traj,
                             'pred_traj_fake': pred_trajs[max_cr_idxes[1]],
                             'text_fixed': text_fixed,
                             'collision_mats': col_mats[max_cr_idxes[1]]}
                # plot_args_lists.append(args_dict)
                plot_args_lists[11] = args_dict
                # text_fixed = f'CR: {all_crs[max_cr_idxes[2]]:0.1f}\nADE: {all_ades[max_cr_idxes[2]]:0.2f}\nFDE: {all_fdes[max_cr_idxes[2]]:0.2f}'
                # args_dict = {'plot_title': cfg + ', sample w/ second-max CR',
                #              'obs_traj': pred_obs_traj,
                #              # 'pred_traj_gt': pred_gt_traj,
                #              'pred_traj_fake': pred_trajs[max_cr_idxes[2]],
                #              'text_fixed': text_fixed,
                #              'collision_mats': col_mats[max_cr_idxes[2]]}
                # plot_args_lists.append(args_dict)

                num_samples_plotted = 10 #if num_agents < 3 else 5
                plot_idxs = np.arange(num_samples_plotted)
                text_fixed = f'CR: {cr:0.1f}\nADE: {ade:0.2f}\nFDE: {fde:0.2f}'
                args_dict = {'plot_title': cfg + ", first 10 samples",
                             'obs_traj': pred_obs_traj,
                             # 'pred_traj_gt': pred_gt_traj,
                             'pred_traj_fake': pred_trajs[plot_idxs],
                             'text_fixed': text_fixed,
                             'pred_alpha': 0.4}
                plot_args_lists[3] = args_dict
                # plot_args_lists.append(args_dict)

            else:
                skip = True
                # print(f"skipping frame {frame} bc max_cr = {max_cr}")
                break


        if 'plain' in cfg:
            num_samples_plotted = 3 if num_agents < 2 else 1
            plot_idxs = np.arange(num_samples_plotted)
            text_fixed = f'CR: {all_crs[plot_idxs[0]]:0.1f}\nADE: {all_ades[plot_idxs[0]]:0.2f}\nFDE: {all_fdes[plot_idxs[0]]:0.2f}'
            args_dict = {'plot_title': cfg + f', sample {num_samples_plotted}',
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_trajs[plot_idxs],
                         'collision_mats': col_mats[num_samples_plotted - 1] if num_samples_plotted == 1 else None,
                         'text_fixed': text_fixed}
            # plot_args_lists.append(args_dict)
            plot_args_lists[5] = args_dict

            num_samples_plotted2 = 3 if num_agents < 2 else 1
            plot_idxs = np.arange(num_samples_plotted, num_samples_plotted+num_samples_plotted2)
            text_fixed = f'CR: {all_crs[plot_idxs[0]]:0.1f}\nADE: {all_ades[plot_idxs[0]]:0.2f}\nFDE: {all_fdes[plot_idxs[0]]:0.2f}'
            args_dict = {'plot_title': cfg + f', sample 2',#{num_samples_plotted} to {num_samples_plotted + num_samples_plotted2}',
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_trajs[plot_idxs],
                         'collision_mats': col_mats[num_samples_plotted - 1] if num_samples_plotted == 1 else None,
                         'text_fixed': text_fixed}
            # plot_args_lists.append(args_dict)
            plot_args_lists[9] = args_dict

            num_samples_plotted = 10  # if num_agents < 3 else 5
            plot_idxs = np.arange(num_samples_plotted)
            text_fixed = f'CR: {cr:0.1f}\nADE: {ade:0.2f}\nFDE: {fde:0.2f}'
            args_dict = {'plot_title': cfg + ", first 10 samples",
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_trajs[plot_idxs],
                         'text_fixed': text_fixed,
                         'pred_alpha': 0.4}
            plot_args_lists[1] = args_dict
            # plot_args_lists.append(args_dict)

        if 'no_collisions' in cfg:
            num_samples_plotted = 3 if num_agents < 2 else 1
            plot_idxs = np.arange(num_samples_plotted)
            text_fixed = f'CR: {all_crs[plot_idxs[0]]:0.1f}\nADE: {all_ades[plot_idxs[0]]:0.2f}\nFDE: {all_fdes[plot_idxs[0]]:0.2f}'
            args_dict = {'plot_title': cfg + f', sample {num_samples_plotted}',
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_trajs[plot_idxs],
                         'collision_mats': col_mats[num_samples_plotted - 1] if num_samples_plotted == 1 else None,
                         'text_fixed': text_fixed}
            # plot_args_lists.append(args_dict)
            plot_args_lists[6] = args_dict

            num_samples_plotted2 = 3 if num_agents < 2 else 1
            plot_idxs = np.arange(num_samples_plotted, num_samples_plotted+num_samples_plotted2)
            text_fixed = f'CR: {all_crs[plot_idxs[0]]:0.1f}\nADE: {all_ades[plot_idxs[0]]:0.2f}\nFDE: {all_fdes[plot_idxs[0]]:0.2f}'
            args_dict = {'plot_title': cfg + f', sample 2',#{num_samples_plotted} to {num_samples_plotted + num_samples_plotted2}',
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_trajs[plot_idxs],
                         'collision_mats': col_mats[num_samples_plotted - 1] if num_samples_plotted == 1 else None,
                         'text_fixed': text_fixed}
            # plot_args_lists.append(args_dict)
            plot_args_lists[10] = args_dict

            num_samples_plotted = 10  # if num_agents < 3 else 5
            plot_idxs = np.arange(num_samples_plotted)
            text_fixed = f'CR: {cr:0.1f}\nADE: {ade:0.2f}\nFDE: {fde:0.2f}'
            args_dict = {'plot_title': cfg + ", first 10 samples",
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': pred_trajs[plot_idxs],
                         'text_fixed': text_fixed,
                         'pred_alpha': 0.4}
            plot_args_lists[2] = args_dict
            # plot_args_lists.append(args_dict)

            # num_samples_plotted2 = 3 if num_agents < 2 else 1
            # plot_idxs = np.arange(num_samples_plotted, num_samples_plotted+num_samples_plotted2)
            # text_fixed = f'CR: {all_crs[plot_idxs[0]]:0.1f}\nADE: {all_ades[plot_idxs[0]]:0.2f}\nFDE: {all_fdes[plot_idxs[0]]:0.2f}'
            # args_dict = {'plot_title': cfg + f', sample 3',#{num_samples_plotted} to {num_samples_plotted + num_samples_plotted2}',
            #              'obs_traj': pred_obs_traj,
            #              # 'pred_traj_gt': pred_gt_traj,
            #              'pred_traj_fake': pred_trajs[plot_idxs],
            #              'collision_mats': col_mats[num_samples_plotted - 1] if num_samples_plotted == 1 else None,
            #              'text_fixed': text_fixed}
            # plot_args_lists.append(args_dict)

        # num_samples_plotted = 10 #if num_agents < 3 else 5
        # plot_idxs = np.arange(num_samples_plotted)
        # text_fixed = f'CR: {cr:0.1f}\nADE: {ade:0.2f}\nFDE: {fde:0.2f}'
        # args_dict = {'plot_title': cfg + ", first 10 samples",
        #              'obs_traj': pred_obs_traj,
        #              # 'pred_traj_gt': pred_gt_traj,
        #              'pred_traj_fake': pred_trajs[plot_idxs],
        #              'text_fixed': text_fixed,
        #              'pred_alpha': 0.4}
        # plot_args_lists.append(args_dict)

        if cfg_i == len(cfgs) - 1:
            # dagger_vs_agent_traj_DE = compute_FDE(agent_traj, agent_trajs_previous)
            num_samples_plotted = 1 if num_agents < 3 else 1
            plot_idxs = np.arange(num_samples_plotted)
            # args_dict = {'plot_title': f"both {cfg_previous} and {cfg}, first {num_samples_plotted} samples, w/o GT",
            #              'obs_traj': pred_obs_traj,
            #              'pred_traj_fake': [pred_trajs_previous[plot_idxs], pred_trajs[plot_idxs]],
            #              'cfg_names': [cfg_previous, cfg]}
            # plot_args_lists.append(args_dict)
            args_dict = {'plot_title': f"both {cfg_previous} and {cfg}, first {num_samples_plotted} samples",
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': [pred_trajs_previous[plot_idxs], pred_trajs[plot_idxs]],
                         # 'pred_traj_gt': pred_gt_traj if num_samples_plotted == 1 else None,
                         # 'text_fixed': f'DE between dagger and AF {dagger_vs_agent_traj_DE:0.2f}',
                         'cfg_names': [cfg_previous, cfg]}
            # plot_args_lists.append(args_dict)
            plot_args_lists[8] = args_dict
            plot_idxs = np.arange(num_samples_plotted)
            args_dict = {'plot_title': f"both {cfg_previous} and {cfg}\nsample 0 of {cfg_previous} and max CR sample of {cfg}",
                         'obs_traj': pred_obs_traj,
                         # 'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': [pred_trajs_previous[plot_idxs], pred_trajs[[max_cr_idx]]],
                         # 'text_fixed': f'DE between dagger and AF {dagger_vs_agent_traj_DE:0.2f}',
                         'cfg_names': [cfg_previous, cfg]}
            # plot_args_lists.append(args_dict)
            plot_args_lists[4] = args_dict
        cfg_previous = cfg

    if skip:
        return

    """plot layered animation"""
    mkdir_if_missing('viz/comparison')
    anim_save_fn = f'viz/comparison/frame-{frame}.mp4'
    print(f'plotting frame {frame}')
    plot_anim_grid(anim_save_fn, f"frame {frame}", *plot_args_lists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--stats', action='store_true', default=False)
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--cfg2', default=None)
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--split', default='test')
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--multiprocess', '-mp', action='store_true', default=False)
    args = parser.parse_args()

    main1(args)