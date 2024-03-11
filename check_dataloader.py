import numpy as np
from multiprocessing import Pool


def data_pair_generator(dl1, dl2):
    for i, (data1, data2) in enumerate(zip(dl1, dl2)):
        train_data1 = {'gt_motion': np.stack(data1['fut_motion'], 1),
                       'obs_motion': np.stack(data1[f'pre_motion'], 1),
                       'frame': data1['frame'],
                       'seq': data1['seq'],
                       'heading_avg': data1['heading_avg'],
                       'heading': data1['heading']}
        train_data2 = {'gt_motion': np.stack(data2['fut_motion'], 1),
                       'obs_motion': np.stack(data2[f'pre_motion'], 1),
                       'frame': data2['frame'],
                       'seq': data2['seq'],
                       'heading_avg': data2['heading_avg'],
                       'heading': data2['heading']}

        yield i, train_data1, train_data2


def check_data_pair(tup):
    i, train_data1, train_data2 = tup
    assert np.allclose(train_data1['gt_motion'],
                       train_data2['gt_motion']), f"i: {i}, {train_data1['gt_motion']} != {train_data2['gt_motion']}"
    assert np.allclose(train_data1['obs_motion'],
                       train_data2['obs_motion']), f"i: {i}, {train_data1['obs_motion']} != {train_data2['obs_motion']}"
    assert train_data1['frame'] == train_data2['frame'], f"i: {i}, {train_data1['frame']} != {train_data2['frame']}"
    assert train_data1['seq'] == train_data2['seq'], f"i: {i}, {train_data1['seq']} != {train_data2['seq']}"
    assert np.allclose(train_data1['heading_avg'], train_data2[
        'heading_avg']), f"i: {i}, {train_data1['heading_avg']} != {train_data2['heading_avg']}"
    assert np.allclose(train_data1['heading'],
                       train_data2['heading']), f"i: {i}, {train_data1['heading']} != {train_data2['heading']}"
    print(f"i: {i} passed")


def process_data_in_parallel(dl1, dl2, num_workers):
    with Pool(num_workers) as pool:
        for i in pool.imap_unordered(check_data_pair, data_pair_generator(dl1, dl2)):
            pass