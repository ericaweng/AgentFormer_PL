import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import multiprocessing
import pickle as pkl

from eval import eval_one_seq
from utils.utils import load_list_from_folder, mkdir_if_missing
from eval import stats_func


def peds_pandas_way(data, labels, indices, verbose=False):
    """change shape of array from individual tracks to grouped by
    frame number and peds"""
    ### ALTERNATE data processing OPTION using pandas
    df = pd.DataFrame(data, columns=labels)
    # df2 = df.copy()

    # deal with frame_number
    # https://stackoverflow.com/questions/39534129/change-index-value-in-pandas-dataframe
    for index in indices:
        df = df.set_index([index])
        df.index = pd.Index(pd.Categorical(df.index).codes,
                            dtype="int",
                            name=index)
        df = df.reset_index()
    df = df.set_index(indices)

    # create an empty array of NaN of the right dimensions
    shape = tuple(map(len, df.index.levels)) + (len(df.columns),)
    arr = np.full(shape, np.nan)

    # fill it using Numpy's advanced indexing
    arr[tuple(df.index.codes)] = df.values
    if verbose:
        print("ped_pandas_way final dataset shape:", arr.shape)

    return arr


def do_one(frame_path):
    sample_list, _ = load_list_from_folder(frame_path)
    sample_all = []
    gt = None
    if len(sample_list) == 0:
        print(f'No samples in {frame_path}')
        import ipdb;
        ipdb.set_trace()
    for sample in sample_list:
        if 'gt' in sample:
            gt = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            gt = peds_pandas_way(gt, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id']).swapaxes(0, 1)
        if 'sample' not in sample:
            continue
        sample = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
        sample = peds_pandas_way(sample, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id']).swapaxes(0, 1)
        sample_all.append(sample)

    samples = np.stack(sample_all, axis=0).swapaxes(0, 1)  # samples x (framex x agents) x 4
    assert gt is not None, os.listdir(frame_path)
    num_agents = gt.shape[0]
    assert num_agents == samples.shape[0]

    vals = eval_one_seq(samples, gt, 0.1)
    return vals, num_agents
    # print("vals:", vals)

SEQUENCE_NAMES = {
            'eth': ['biwi_eth'],
            'hotel': ['biwi_hotel'],
            'zara1': ['crowds_zara01'],
            'zara2': ['crowds_zara02'],
            'univ': ['students001', 'students003'],
            # 'eth_ucy': [
            #     'biwi_eth', 'biwi_hotel', 'crowds_zara01', 'crowds_zara02',
            #     'students001', 'students003'
            # ],
            # 'sdd': [
            #     'coupa_0', 'coupa_1', 'gates_2', 'hyang_0', 'hyang_1', 'hyang_3',
            #     'hyang_8', 'little_0', 'little_1', 'little_2', 'little_3', 'nexus_5',
            #     'nexus_6', 'quad_0', 'quad_1', 'quad_2', 'quad_3'
            # ]
    }
all_models = ['agentformer_sfm2', 'agentformer_sfm', 'sgan', 'social_force', 'ynet', 'trajectron', 'agentformer', 'pecnet', 'vanilla_social_force']
models = ['agentformer_sfm2', 'agentformer_sfm', 'sgan', 'social_force', 'ynet', 'trajectron', 'agentformer', 'pecnet', 'vanilla_social_force']


def main(args):
    # model to dataset
    results_dict = defaultdict(lambda: defaultdict(lambda: {}))

    # models = [model.split('/')[-1] for model in glob.glob('../trajectory_reward/results/trajectories/*')]
    # print("models:", models)
    model_paths = glob.glob('../trajectory_reward/results/trajectories/*')

    if not args.cached:
        for model_path in model_paths:
            model = model_path.split('/')[-1]
            if model not in models:
                print(f'skipping {model}')
                continue
            for dataset in SEQUENCE_NAMES:
                print(f"trying {model} {dataset}")
                frames = []
                for seq in SEQUENCE_NAMES[dataset]:
                    if model == 'trajectron':
                        glob_str = f'{model_path}/{dataset}_dyn/{seq}/*'
                    else:
                        glob_str = f'{model_path}/{seq}/*'
                    frames.extend(glob.glob(glob_str))

                if len(frames) == 0:
                    print(f"missing predictions for {model} {dataset}")
                    continue

                with multiprocessing.Pool(60) as pool:
                    all_metrics = pool.map(do_one, frames)

                all_metrics, num_agent_per_seq = zip(*all_metrics)

                total_num_agents = np.sum(num_agent_per_seq)
                for key, values in zip(stats_func.keys(), zip(*all_metrics)):
                    if '_seq' in key:  # sequence-based metric
                        value = np.mean(values)
                    else:  # agent-based metric
                        value = np.sum(values * np.array(num_agent_per_seq)) / np.sum(num_agent_per_seq)
                    results_dict[model][dataset][key] = value

                # log and print results
                print(f"{model} {dataset}")
                for key, value in results_dict[model][dataset].items():
                    print(f"{key}: {value:.4f}")
                for key, value in results_dict[model][dataset].items():
                    print(f"{value:.4f}")
                print("total_num_agents:", total_num_agents)

                with open(f'metrics-new/{model}-{dataset}.pkl', 'wb') as f:
                    pkl.dump(results_dict[model][dataset], f)

    for metric in stats_func:
        get_table_for_metric(metric)


def get_table_for_metric(metric):
    table = np.full((len(all_models),len(SEQUENCE_NAMES)), np.nan)#,dtype=object)
    for model_i, model in enumerate(all_models):
        for dataset_i, dataset in enumerate(SEQUENCE_NAMES):
            try:
                with open(f'metrics-new/{model}-{dataset}.pkl', 'rb') as f:
                    results = pkl.load(f)
                    table[model_i][dataset_i] = results[metric]
            except FileNotFoundError:
                print(f'no stats for {model} {dataset}')
                pass

    table = table.swapaxes(0,1)

    # print("table:", table)
    df = pd.DataFrame(table, columns=all_models)
    df['datasets'] = SEQUENCE_NAMES.keys()
    df = df.set_index('datasets')
    table_save_path = f'results-n/{metric}.tsv'
    mkdir_if_missing(table_save_path)
    df.to_csv(table_save_path, float_format='%.3f')
    # np.savetxt('metrics-new/.tsv', table, fmt='%.3f', delim='\t')


if __name__ == "__main__":
    # logging
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cached', '-c', action='store_true')
    args = parser.parse_args()
    main(args)
