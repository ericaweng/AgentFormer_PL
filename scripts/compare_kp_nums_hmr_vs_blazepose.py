import os
import numpy as np
from pathlib import Path

from data.jrdb_split import get_jrdb_split_full

TRAIN_SEQS, VAL_SEQS, TEST_SEQS = get_jrdb_split_full()

stats_dict = {}
for seq_name in TRAIN_SEQS+TEST_SEQS:
    # print(f"analyzing {seq_name}")
    blazepose_kp_data = np.load(f'datasets/jrdb_og_odo_mdr-1000_ss3d_mot/agent_keypoints/{seq_name}_kp.npz', allow_pickle=True)['arr_0'].item()
    hmr_labels_3d_pose = np.load(f"datasets/jrdb/jrdb_hmr2_raw_stitched/{seq_name}_kp3d.npz", allow_pickle=True)['arr_0'].item()

    blazepose_num_ped_frames_with_at_least_one_kp = np.sum([
            np.sum([1 for ped in peds.values()if np.sum(~np.isnan(ped)) > 10])
                                                            for peds in blazepose_kp_data.values()])
    hmr_num_ped_frames_with_at_least_one_kp = np.sum([
            np.sum([1 for ped in peds.values() if ped is not None and np.sum(~np.isnan(ped)) > 10])
                                                            for peds in hmr_labels_3d_pose.values()])

    print(f"{seq_name} hmr {hmr_num_ped_frames_with_at_least_one_kp} blazepose {blazepose_num_ped_frames_with_at_least_one_kp} "
          f"ratio {round(hmr_num_ped_frames_with_at_least_one_kp/blazepose_num_ped_frames_with_at_least_one_kp, 2)}")
    stats_dict[seq_name] = [blazepose_num_ped_frames_with_at_least_one_kp, hmr_num_ped_frames_with_at_least_one_kp]

print(f'blazepose total ped-frames: {sum([k[0] for k in stats_dict.values()])} '
      f'hmr2 {sum([k[1] for k in stats_dict.values()])} '
      f'diff {sum([k[1] for k in stats_dict.values()]) - sum([k[0] for k in stats_dict.values()])} '
      f'ratio {round(sum([k[1] for k in stats_dict.values()]) / sum([k[0] for k in stats_dict.values()]), 2)}')

# Output
# bytes-cafe-2019-02-07_0 hmr 32064 blazepose 18132.0 ratio 1.77
# clark-center-2019-02-28_0 hmr 22908 blazepose 1508.0 ratio 15.19
# clark-center-2019-02-28_1 hmr 52054 blazepose 4325.0 ratio 12.04
# clark-center-intersection-2019-02-28_0 hmr 27735 blazepose 4484.0 ratio 6.19
# cubberly-auditorium-2019-04-22_0 hmr 33725 blazepose 10450.0 ratio 3.23
# forbes-cafe-2019-01-22_0 hmr 41745 blazepose 7750.0 ratio 5.39
# gates-159-group-meeting-2019-04-03_0 hmr 3965 blazepose 2807.0 ratio 1.41
# gates-ai-lab-2019-02-08_0 hmr 7967 blazepose 1590.0 ratio 5.01
# gates-basement-elevators-2019-01-17_1 hmr 13128 blazepose 5302.0 ratio 2.48
# gates-to-clark-2019-02-28_1 hmr 6428 blazepose 2276.0 ratio 2.82
# hewlett-packard-intersection-2019-01-24_0 hmr 37460 blazepose 6204.0 ratio 6.04
# huang-2-2019-01-25_0 hmr 8161 blazepose 1488.0 ratio 5.48
# huang-basement-2019-01-25_0 hmr 31796 blazepose 4123.0 ratio 7.71
# huang-lane-2019-02-12_0 hmr 30262 blazepose 3204.0 ratio 9.45
# jordan-hall-2019-04-22_0 hmr 23166 blazepose 14179.0 ratio 1.63
# memorial-court-2019-03-16_0 hmr 17774 blazepose 2581.0 ratio 6.89
# meyer-green-2019-03-16_0 hmr 5463 blazepose 245.0 ratio 22.3
# nvidia-aud-2019-04-18_0 hmr 15770 blazepose 7001.0 ratio 2.25
# packard-poster-session-2019-03-20_0 hmr 19139 blazepose 4854.0 ratio 3.94
# packard-poster-session-2019-03-20_1 hmr 31612 blazepose 10362.0 ratio 3.05
# packard-poster-session-2019-03-20_2 hmr 61676 blazepose 19988.0 ratio 3.09
# stlc-111-2019-04-19_0 hmr 25703 blazepose 16000.0 ratio 1.61
# svl-meeting-gates-2-2019-04-08_0 hmr 9106 blazepose 3516.0 ratio 2.59
# svl-meeting-gates-2-2019-04-08_1 hmr 11159 blazepose 4206.0 ratio 2.65
# tressider-2019-03-16_0 hmr 9316 blazepose 626.0 ratio 14.88
# tressider-2019-03-16_1 hmr 11253 blazepose 399.0 ratio 28.2
# tressider-2019-04-26_2 hmr 81771 blazepose 29604.0 ratio 2.76
# cubberly-auditorium-2019-04-22_1 hmr 5641 blazepose 3209.0 ratio 1.76
# discovery-walk-2019-02-28_0 hmr 4241 blazepose 1168.0 ratio 3.63
# discovery-walk-2019-02-28_1 hmr 5773 blazepose 2233.0 ratio 2.59
# food-trucks-2019-02-12_0 hmr 26100 blazepose 13081 ratio 2.0
# gates-ai-lab-2019-04-17_0 hmr 13730 blazepose 13467 ratio 1.02
# gates-basement-elevators-2019-01-17_0 hmr 6532 blazepose 3267 ratio 2.0
# gates-foyer-2019-01-17_0 hmr 7297 blazepose 5567.0 ratio 1.31
# gates-to-clark-2019-02-28_0 hmr 1477 blazepose 814.0 ratio 1.81
# hewlett-class-2019-01-23_0 hmr 9751 blazepose 6368 ratio 1.53
# hewlett-class-2019-01-23_1 hmr 4463 blazepose 3273 ratio 1.36
# huang-2-2019-01-25_1 hmr 357 blazepose 1535 ratio 0.23
# huang-intersection-2019-01-22_0 hmr 13198 blazepose 3299.0 ratio 4.0
# indoor-coupa-cafe-2019-02-06_0 hmr 18287 blazepose 11590 ratio 1.58
# lomita-serra-intersection-2019-01-30_0 hmr 6608 blazepose 3353 ratio 1.97
# meyer-green-2019-03-16_1 hmr 4150 blazepose 2585.0 ratio 1.61
# nvidia-aud-2019-01-25_0 hmr 10387 blazepose 6756 ratio 1.54
# nvidia-aud-2019-04-18_1 hmr 3798 blazepose 1806 ratio 2.1
# nvidia-aud-2019-04-18_2 hmr 4699 blazepose 2360 ratio 1.99
# outdoor-coupa-cafe-2019-02-06_0 hmr 10015 blazepose 4254.0 ratio 2.35
# quarry-road-2019-02-28_0 hmr 2036 blazepose 1090.0 ratio 1.87
# serra-street-2019-01-30_0 hmr 9184 blazepose 3730.0 ratio 2.46
# stlc-111-2019-04-19_1 hmr 4295 blazepose 2955 ratio 1.45
# stlc-111-2019-04-19_2 hmr 2277 blazepose 1468 ratio 1.55
# tressider-2019-03-16_2 hmr 5634 blazepose 1705.0 ratio 3.3
# tressider-2019-04-26_0 hmr 15490 blazepose 38.0 ratio 407.63
# tressider-2019-04-26_1 hmr 23503 blazepose 94.0 ratio 250.03
# tressider-2019-04-26_3 hmr 23700 blazepose 84.0 ratio 282.14
# blazepose total ped-frames: 288353.0 hmr2 914929 diff 626576.0 ratio 3.17