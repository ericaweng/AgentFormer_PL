import glob
import numpy as np

from utils.config import Config
from utils.utils import mkdir_if_missing
from eval import *
from viz_utils import plot_traj_anim, plot_fig


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


def main(args):
    cfg = Config(args.cfg)
    cfg2 = Config(args.cfg2)
    dataset = cfg.dataset
    results_dir = cfg.result_dir
    results_dir2 = cfg2.result_dir

    if args.epochs is None:
        epoch1 = int(sorted(map(str,glob.glob(f'{results_dir}/epoch_*')))[-1][-4:])
        epoch2 = int(sorted(map(str,glob.glob(f'{results_dir2}/epoch_*')))[-1][-4:])
        print("epochs:", epoch1, epoch2)
    else:
        epoch1, epoch2 = map(int, args.epochs.split(','))

    resize = 1.0
    if dataset == 'nuscenes_pred':  # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{args.split}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = locals()[f'seq_{args.split}']
    elif dataset == 'trajnet_sdd':
        data_root = 'datasets/trajnet_split'
        # data_root = 'datasets/stanford_drone_all'
        gt_dir = f'{data_root}/{args.split}'
        seq_train, seq_val, seq_test = get_stanford_drone_split()
        seq_eval = locals()[f'seq_{args.split}']
        # resize = 0.25
        indices = [0, 1, 2, 3]
    elif dataset == 'sdd':
        data_root = 'datasets/stanford_drone_all'
        gt_dir = f'{data_root}/{args.split}'
        seq_train, seq_val, seq_test = get_stanford_drone_split()
        seq_eval = locals()[f'seq_{args.split}']
        indices = [0, 1, 2, 3]
    else:  # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{cfg.dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(cfg.dataset)
        seq_eval = locals()[f'seq_{args.split}']
        indices = [0, 1, 13, 15]

    if args.log_file is None:
        log_file = os.path.join(results_dir, 'log_eval_viz.txt')
    else:
        log_file = args.log_file
    log_file = open(log_file, 'a+')
    print_log('loading results from %s' % results_dir, log_file)
    print_log('loading results from %s' % results_dir2, log_file)
    print_log('loading GT from %s' % gt_dir, log_file)

    stats_func = {
            'ADE': compute_ADE,
            'FDE': compute_FDE,
            'CR_pred': compute_CR,
            'CR_gt': compute_CR,
            'CR_pred_mean': compute_CR,
            'CR_gt_mean': compute_CR,
            'ACFL': compute_ACFL
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    print_log('number of gt sequences to evaluate is %d' % num_seq, log_file)
    for seq_name in seq_eval:
        # load GT raw data
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name + '.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, indices][0].astype('float32')
            line_data[2:4] = line_data[2:4] * resize
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)
        if dataset == 'trajnet_sdd':
            gt_raw[:, 0] = np.round(gt_raw.astype(np.float)[:, 0] / 12.0)

        samples_dir = f'{results_dir}/epoch_{epoch1:04d}/{args.split}/samples'
        samples_dir2 = f'{results_dir2}/epoch_{epoch2:04d}/{args.split}/samples'
        data_filelist, _ = load_list_from_folder(os.path.join(samples_dir, seq_name))
        print_log('number of real sequences 1 to evaluate is %d' % len(data_filelist), log_file)
        data_filelist2, _ = load_list_from_folder(os.path.join(samples_dir2, seq_name))
        print_log('number of real sequences 2 to evaluate is %d' % len(data_filelist2), log_file)

        for data_file, data_file2 in zip(data_filelist, data_filelist2):  # each example e.g., seq_0001 - frame_000009
            all_traj = get_traj_from_file(data_file)
            all_traj2 = get_traj_from_file(data_file2)

            # convert raw data to our format for evaluation
            id_list = np.unique(all_traj[:, :, 1])
            id_list2 = np.unique(all_traj2[:, :, 1])
            frame_list = np.unique(all_traj[:, :, 0])
            agent_traj = []
            agent_traj2 = []
            gt_traj = []
            obs_traj = []
            for idx in id_list:
                # GT traj
                gt_idx = gt_raw[gt_raw[:, 1] == idx]  # frames x 4

                # predicted traj
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]  # sample x frames x 4
                # filter data
                pred_idx, _, _ = align_gt(pred_idx, gt_idx)
                # same for second model
                pred_idx2 = all_traj2[:, ind, :]  # sample x frames x 4
                pred_idx2, gt_idx, obs = align_gt(pred_idx2, gt_idx)
                obs = obs[..., 2:]

                # append
                agent_traj.append(pred_idx)
                agent_traj2.append(pred_idx2)
                gt_traj.append(gt_idx)
                obs_traj.append(obs)

            """compute stats"""
            frame = int(frame_list[0])
            stats_str = ''
            if args.stats:
                for stats_name, meter in stats_meter.items():
                    func = stats_func[stats_name]
                    stats_func_args = {'pred_arr': agent_traj, 'gt_arr': gt_traj}
                    if stats_name == 'CR_pred':
                        stats_func_args['pred'] = True
                    elif stats_name == 'CR_gt':
                        stats_func_args['gt'] = True
                    elif stats_name == 'CR_pred_mean':
                        stats_func_args['pred'] = True
                        stats_func_args['aggregation'] = 'mean'
                    elif stats_name == 'CR_gt_mean':
                        stats_func_args['gt'] = True
                        stats_func_args['aggregation'] = 'mean'

                    value = func(**stats_func_args)
                    # if value > 0 and stats_name == 'CR_pred':
                    #     import ipdb; ipdb.set_trace()
                    meter.update(value, n=len(agent_traj))

                stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
                print_log(
                        f'evaluating seq {seq_name:s}, forecasting frame {frame:06d} to {int(frame_list[-1]):06d} {stats_str}',
                        log_file)

            """plot layered animation"""
            mkdir_if_missing('viz')
            anim_save_fn = f'viz/{cfg.id}-vs-{cfg2.id}_seq-{seq_name}_frame-{frame}.mp4'
            obs_traj = np.stack(obs_traj).transpose(1,0,2)
            pred_traj1 = np.stack(agent_traj).transpose(1,2,0,3)[:args.num_samples]
            pred_traj2 = np.stack(agent_traj2).transpose(1,2,0,3)[:args.num_samples]
            pred_gt_traj = np.stack(gt_traj).transpose(1,0,2)
            # cfg_names = ['sfm' if 'sfm' in cfg_id else
            #                           'pre_nocol' if 'pre' in cfg_id and 'nocol' in cfg_id else
            #                           'dlow_nocol' if 'nocol' in cfg_id else
            #                           'pre' if 'pre' in cfg_id else
            #                           'dlow'
            #                           for cfg_id in [cfg.id, cfg2.id]]
            cfg_names = [cfg.id, cfg2.id]
            plot_traj_anim(obs_traj,
                           pred_traj_gt=pred_gt_traj,
                           pred_traj_fake=[pred_traj1, pred_traj2],
                           save_fn=anim_save_fn,#)
                           cfg_names=cfg_names)


def main1(args):
    """plot:
    1. GT
    2. AF
    3. AF + col rej
    4. AF + Dagger"""

    seq_name = 'crowds_zara02'
    cfgs = ['zara2_agentformer_pre_nocol', 'zara2_dagger_tune_test']#'zara2_agentformer_pre_nocol']#, 'zara2_dagger_tune_test']  # , 'zara2_dagger_tune'
    # cfgs = [Config(cfg) for cfg in cfgs]

    stats_func = {
            'ADE': compute_ADE,
            'FDE': compute_FDE,
            'CR_pred': compute_CR,
            'CR_pred_mean': compute_CR,
    }
    # stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    cfg = cfgs[0]
    last_epoch = int(str(sorted(glob.glob(os.path.join(f'results/{cfg}/results/', 'epoch_*')))[-1][-4:]))
    dir_to_search_frames = f'results/{cfg}/results/epoch_{last_epoch:04d}/test/samples/{seq_name}/*'
    print("dir_to_search_frames:", dir_to_search_frames)
    all_frames = [int(f.split('_')[-1]) for f in glob.glob(dir_to_search_frames)]
    print("len(all_frames):", len(all_frames))

    if args.multiprocess:
        with Pool() as p:
            p.starmap(plot_frame, list(zip(all_frames, [cfgs for _ in range(len(all_frames))])))
    else:
        for frame in all_frames:
            plot_frame(frame, cfgs)


def plot_frame(frame, cfgs):
    plot_args_lists = []
    seq_name = 'crowds_zara02'
    skip = False
    gt_dir = f'datasets/eth_ucy/zara2'
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

        cr, all_crs, col_mats = compute_CR(agent_traj, return_sample_crs=True, return_collision_mat=True)
        # col_mats = np.stack(col_mats)
        ade = compute_ADE(agent_traj, gt_traj)
        fde = compute_FDE(agent_traj, gt_traj)
        text_fixed = f'CR: {cr:0.1f}\nADE: {ade:0.2f}\nFDE: {fde:0.2f}'
        pred_trajs_previous = pred_trajs
        pred_trajs = np.stack(agent_traj).transpose(1, 2, 0, 3)  # (num_samples, pred_steps, num_agents, 2)
        pred_obs_traj = np.stack(obs_traj).transpose(1, 0, 2)  # (obs_steps, num_agents, 2)
        pred_gt_traj = np.stack(gt_traj).transpose(1, 0, 2)  # (pred_steps, num_agents, 2)

        cfg = 'AF + dagger' if 'dagger' in cfg else 'AF + collision rejection'#'plain AF'
        if 'dagger' in cfg or True:  # and agent_trajs_previous is not None:
            # print("all_crs:", all_crs)
            # print("cr:", cr)
            max_cr = np.max(all_crs)
            # print("max_cr:", max_cr)
            if max_cr > 0:
                max_cr_idx = np.argmax(all_crs)
                # args_dict = {'plot_title': cfg + ' only max CR',
                #              'obs_traj': pred_obs_traj,
                #              'pred_traj_gt': pred_gt_traj,
                #              'pred_traj_fake': pred_trajs[max_cr_idx],
                #              'text_fixed': text_fixed,
                #              'collision_mats': col_mats[max_cr_idx]}
                # plot_args_lists.append(args_dict)
                # print("max_cr_idx.shape:", pred_trajs[max_cr_idx].shape)
                # import ipdb; ipdb.set_trace()
                args_dict = {'plot_title': cfg + ' only max CR, w/o GT',
                             'obs_traj': pred_obs_traj,
                             # 'pred_traj_gt': pred_gt_traj,
                             'pred_traj_fake': pred_trajs[max_cr_idx],
                             'text_fixed': text_fixed,
                             'collision_mats': col_mats[max_cr_idx]}
                plot_args_lists.append(args_dict)
            else:
                skip = True
                break
            # if cr < 30:
            #     skip = True
            #     break

        if cfg_i == 0:  # gt
            args_dict = {'plot_title': 'obs + gt only',
                         'obs_traj': pred_obs_traj,
                         'pred_traj_gt': pred_gt_traj,}
            plot_args_lists.append(args_dict)

        plot_idxs = np.arange(5)

        args_dict = {'plot_title': cfg + ' w/o GT',
                     'obs_traj': pred_obs_traj,
                     # 'pred_traj_gt': pred_gt_traj,
                     'pred_traj_fake': pred_trajs[plot_idxs],
                     'text_fixed': text_fixed}
        plot_args_lists.append(args_dict)
        args_dict = {'plot_title': cfg,
                     'obs_traj': pred_obs_traj,
                     'pred_traj_gt': pred_gt_traj,
                     'pred_traj_fake': pred_trajs[plot_idxs],
                     'text_fixed': text_fixed}
        plot_args_lists.append(args_dict)

        if cfg_i == 1:
            # dagger_vs_agent_traj_DE = compute_FDE(agent_traj, agent_trajs_previous)
            args_dict = {'plot_title': f"both {cfg_previous} and {cfg}",
                         'obs_traj': pred_obs_traj,
                         'pred_traj_gt': pred_gt_traj,
                         'pred_traj_fake': [pred_trajs_previous[plot_idxs], pred_trajs[plot_idxs]],
                         # 'text_fixed': f'DE between dagger and AF {dagger_vs_agent_traj_DE:0.2f}',
                         'cfg_names': [cfg_previous, cfg]}
            plot_args_lists.append(args_dict)

        cfg_previous = cfg

    if skip:
        return

    """plot layered animation"""
    mkdir_if_missing('viz/comparison')
    anim_save_fn = f'viz/comparison/frame-{frame}.mp4'
    print(f'plotting frame {frame}')
    plot_fig(anim_save_fn, *plot_args_lists)


def main0(args):
    """plot nocol data turned into dagger data"""
    dataset = 'zara2'
    gt_dir = f'datasets/eth_ucy/{dataset}'
    seq_train, seq_val, seq_test = get_ethucy_split(dataset)
    seq_eval = locals()[f'seq_{args.split}']
    indices = [0, 1, 13, 15]

    log_file = os.path.join('viz', 'log_eval_viz.txt')
    log_file = open(log_file, 'a+')
    print_log('loading GT from %s' % gt_dir, log_file)

    stats_func = {
            'ADE': compute_ADE,
            'FDE': compute_FDE,
            'CR_pred': compute_CR,
            'CR_pred_mean': compute_CR,
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    print_log('number of gt sequences to evaluate is %d' % num_seq, log_file)
    for seq_name in seq_eval:
        # load GT raw data
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name + '.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, indices][0].astype('float32')
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)

        data_filelist, _ = load_list_from_folder('datasets/eth_ucy/zara2/pred_data')#os.path.join(samples_dir, seq_name))
        data_filelist2, _ = load_list_from_folder('datasets/eth_ucy/zara2/pred_data2')#os.path.join(samples_dir, seq_name))

        # for data_file, data_file2 in zip(data_filelist, data_filelist2):  # each example e.g., seq_0001 - frame_000009
        #     seq_name_data_file = data_file.split('/')[-1].split('-')[0]
        #     frame = data_file.split('/')[-1].split('-')[0].split('.')[0]
        #     if seq_name != seq_name_data_file:
        #         continue
        #     do_data_file_plot(data_file, data_file2, indices, gt_raw, seq_name)
        with Pool() as p:
            p.starmap(do_data_file_plot, list(zip(data_filelist, data_filelist2,
                                                  [indices for _ in range(len(data_filelist2))],
                                                  [gt_raw for _ in range(len(data_filelist2))],
                                                  [seq_name for _ in range(len(data_filelist2))])))


def do_data_file_plot(data_file, data_file2, indices, gt_raw, seq_name):
    frame = data_file.split('/')[-1].split('-')[-1].split('.')[0]
    all_trajss, _ = load_txt_file(data_file)
    all_traj = []
    for line_data in all_trajss:
        line_data = np.array([line_data.split(' ')])[:, indices][0].astype('float32')
        if line_data[1] == -1: continue
        all_traj.append(line_data)
    all_traj = np.stack(all_traj)[np.newaxis, ...]

    all_trajss, _ = load_txt_file(data_file2)
    all_traj2 = []
    for line_data in all_trajss:
        line_data = np.array([line_data.split(' ')])[:, indices][0].astype('float32')
        if line_data[1] == -1: continue
        all_traj2.append(line_data)
    all_traj2 = np.stack(all_traj2)[np.newaxis, ...]

    # all_traj = get_traj_from_file(data_file, indices)
    # all_traj2 = get_traj_from_file(data_file2, indices)

    # convert raw data to our format for evaluation
    id_list = np.unique(all_traj[..., 1])
    agent_traj = []
    agent_traj2 = []
    gt_traj = []
    for idx in id_list:
        # GT traj
        gt_idx = gt_raw[gt_raw[:, 1] == idx]  # frames x 4
        # predicted traj
        ind = np.unique(np.where(all_traj[..., 1] == idx)[1].tolist())
        pred_idx = all_traj[:, ind, :]  # sample x frames x 4
        # filter data
        pred_idx, _, _ = align_gt(pred_idx, gt_idx, no_obs=True)
        # same for second model
        pred_idx2 = all_traj2[:, ind, :]  # sample x frames x 4
        pred_idx2, gt_idx, _ = align_gt(pred_idx2, gt_idx, no_obs=True)
        # obs = obs[..., 2:]

        # append
        agent_traj.append(pred_idx)
        agent_traj2.append(pred_idx2)
        gt_traj.append(gt_idx)
        # obs_traj.append(obs)

    """plot layered animation"""
    mkdir_if_missing('viz')
    anim_save_fn = f'viz/dagger_data-vs-gt_seq-{seq_name}_frame-{frame}.mp4'
    pred_traj1 = np.stack(agent_traj).transpose(1, 2, 0, 3)[0:1]
    pred_traj2 = np.stack(agent_traj2).transpose(1, 2, 0, 3)[0:1]
    pred_gt_traj = np.stack(gt_traj).transpose(1, 0, 2)
    assert np.all(pred_traj1[0,:8] == pred_traj2[0,:8]) and np.all(pred_traj2[0,:8] == pred_gt_traj[:8])
    obs_traj = pred_traj1[0,:8]
    pred_traj1 = pred_traj1[:,8:]
    pred_traj2 = pred_traj2[:,8:]
    pred_gt_traj = pred_gt_traj[8:]
    # cfg_names = ['sfm' if 'sfm' in cfg_id else
    #                           'pre_nocol' if 'pre' in cfg_id and 'nocol' in cfg_id else
    #                           'dlow_nocol' if 'nocol' in cfg_id else
    #                           'pre' if 'pre' in cfg_id else
    #                           'dlow'
    #                           for cfg_id in [cfg.id, cfg2.id]]
    cfg_names = ['pred1', 'pred2']  # [cfg.id, cfg2.id]
    plot_traj_anim(obs_traj,
                   pred_traj_gt=pred_gt_traj,
                   pred_traj_fake=[pred_traj1, pred_traj2],
                   save_fn=anim_save_fn,  # )
                   cfg_names=cfg_names)


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