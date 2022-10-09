import glob
import os
import numpy as np
from utils.utils import mkdir_if_missing, load_list_from_folder, isfolder, isfile, load_txt_file
from data.ethucy_split import get_ethucy_split

from visualize import align_gt


def main(args):
    model_name = 'zara2_agentformer_pre_nocol'
    print("model_name:", model_name)
    # model_name = 'zara2_agentformer_pre_nocol'
    results_root = os.path.join('results', model_name, 'results')
    dataset = 'zara2'
    # dataset_root = 'datasets'
    # output_path = os.path.join(results_root, model_name)
    # glob.glob(os.path.join(output_path, '*.pt'))

    gt_dir = f'datasets/eth_ucy/{dataset}'
    seq_train, _, _ = get_ethucy_split(dataset)
    total = 0
    for seq_name in seq_train:
        # load GT raw data
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name + '.txt'))
        gt_raw = []
        resize = 1
        indices = [0,1,13,15]
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, indices][0].astype('float32')
            line_data[2:4] = line_data[2:4] * resize
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)

        # load samples
        samples_dir = os.path.join(sorted(glob.glob(os.path.join(results_root, 'epoch_*')))[-1], 'train/samples')
        data_filelist, _ = load_list_from_folder(os.path.join(samples_dir, seq_name))
        total += len(data_filelist)
        # print("len(data_filelist):", len(data_filelist))
        # continue
        for data_file_i, data_file in enumerate(data_filelist):
            if isfile(data_file):
                all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')  # (frames x agents) x 4
                all_traj = np.expand_dims(all_traj, axis=0)  # 1 x (frames x agents) x 4
            # for stochastic with multiple samples
            elif isfolder(data_file):
                sample_list, _ = load_list_from_folder(data_file)
                sample_all = []
                for sample_i, sample in enumerate(sample_list):
                    if sample_i == 0:
                        continue
                    sample = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
                    sample_all.append(sample)
                    if args.save_first:
                        break
                    # if args.save_first:
                    #     all_traj = sample_all[0]
                    # else:
                    #     compare samples with gt and find the one with biggest deviation, etc.
                        # sample_all[0]
                all_traj = np.stack(sample_all, axis=0)  # samples x (framex x agents) x 4
            else:
                raise RuntimeError

            # convert raw data to our format for evaluation
            id_list = np.unique(all_traj[:, :, 1])
            save_trajs = []
            for idx in id_list:
                # GT traj
                gt_idx = gt_raw[gt_raw[:, 1] == idx]  # frames x 4
                # predicted traj
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]  # sample x frames x 4
                _, gt, obs = align_gt(pred_idx, gt_idx)
                save_arr = np.concatenate([obs, pred_idx[0]], axis=0)
                save_trajs.append(save_arr)
            save_trajs = np.concatenate(save_trajs)

            save_arr = np.full((save_trajs.shape[0], 17), '-1.0', dtype='object')
            save_arr[:, [0,1,13,15]] = save_trajs
            save_arr[:, 2] = 'Pedestrian'
            save_dir = os.path.join(gt_dir, 'pred_data')
            mkdir_if_missing(save_dir)
            save_path = os.path.join(save_dir, f'{seq_name}-{data_file_i:06d}.txt')
            np.savetxt(save_path, save_arr, fmt="%s")
            print("saved:", save_path)

    print("total:", total)


def save_prediction(pred, data, suffix, save_dir, indices, num_future_frames, scale=1.0):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(num_future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[indices[-2:]] = pred[i, j] / scale  # [13, 15] or [2, 3] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_first', '-sf', action='store_true', default=True)
    # parser.add_argument('--', '-', type=lambda x: list(map(int, x.split())))
    # parser.add_argument('--', '-', action='store_true')
    args = parser.parse_args()
    main(args)