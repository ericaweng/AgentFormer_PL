import json
import numpy as np

from data.jrdb_split import get_jrdb_split_full
from jrdb_toolkit.visualisation.visualize_constants import TEST_LOCATION_TO_ID

TRAIN_SEQS, VAL_SEQS, TEST_SEQS = get_jrdb_split_full()

stats_dict = {}
# for seq_name in TRAIN_SEQS:
#     gt_train = f'datasets/jrdb/train/labels/labels_3d/{seq_name}.json'
#     with open(gt_train, 'r') as f:
#         num_frames_gt = int(max(json.load(f)['labels'].keys()).split(".")[0])
#     det_train = f'datasets/jrdb/processed/labels/labels_detections_3d/{seq_name}.json'
#     with open(det_train, 'r') as f:
#         num_frames_det = int(max(json.load(f)['labels'].keys()).split(".")[0])
#     print(f"TRAIN {seq_name} gt {num_frames_gt} det {num_frames_det} should be the same")
#     assert num_frames_gt == num_frames_det

for seq_name in TEST_SEQS:
    det_test_txt_ss3d = f'datasets/jrdb/test/labels/ss3d_mot/{TEST_LOCATION_TO_ID[seq_name]:04d}.txt'
    num_frames_ss3d = max(np.genfromtxt(det_test_txt_ss3d, delimiter=' ')[:,0].astype(int))
    det_test_txt_pife = f'datasets/jrdb/test/labels/PiFeNet/{TEST_LOCATION_TO_ID[seq_name]:04d}.txt'
    num_frames_pife = max(np.genfromtxt(det_test_txt_pife, delimiter=' ')[:,0].astype(int))
    print(f"TEST {seq_name} ss3d_mot {num_frames_ss3d} vs. PiFeNet {num_frames_pife}")
