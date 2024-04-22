""" problem: memory error with multiprocessing (so either when loading data in dataloader, or plotting videos) """

# number of sequences:



# 1. check how much worse scenes / frames with movement vs. without
# 2. visualize with poses vs. without
# 3. allow empty poses and positions timesteps


# if self.global_step > 3400:
"""
3452, frame: 798, seq: jordan-hall-2019-04-22_0                                                                                                              [988/3761]
3453, frame: 891, seq: jordan-hall-2019-04-22_0
3454, frame: 199, seq: huang-lane-2019-02-12_0
3455, frame: 119, seq: bytes-cafe-2019-02-07_0
------------------------------------
3407, frame: 826, seq: hewlett-packard-intersection-2019-01-24_0
3408, frame: 615, seq: packard-poster-session-2019-03-20_1
Traceback (most recent call last):
  File "/root/mambaforge/envs/p3d_/lib/python3.9/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/root/mambaforge/envs/p3d_/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/root/mambaforge/envs/p3d_/lib/python3.9/site-packages/torch/multiprocessing/reductions.py", line 366, in reduce_storage
    fd, size = storage._share_fd_cpu_()
RuntimeError: unable to mmap 64 bytes from file </torch_39104_1377978244_63479>: Cannot allocate memory (12)
3409, frame: 567, seq: packard-poster-session-2019-03-20_2
3410, frame: 817, seq: hewlett-packard-intersection-2019-01-24_0
"""
# print(f"{self.global_step}, frame: {batch['frame']}, seq: {batch['seq']}, num_agents: {len(batch['fut_motion'])}")


"""
3430, frame: 496, seq: packard-poster-session-2019-03-20_2, num_agents: 58
3431, frame: 1343, seq: hewlett-packard-intersection-2019-01-24_0, num_agents: 18
Traceback (most recent call last):
  File "/root/mambaforge/envs/p3d_/lib/python3.9/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/root/mambaforge/envs/p3d_/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/root/mambaforge/envs/p3d_/lib/python3.9/site-packages/torch/multiprocessing/reductions.py", line 366, in reduce_storage
    fd, size = storage._share_fd_cpu_()
RuntimeError: unable to mmap 64 bytes from file </torch_48816_3235653068_63483>: Cannot allocate memory (12)
3432, frame: 432, seq: hewlett-packard-intersection-2019-01-24_0, num_agents: 25
3433, frame: 764, seq: packard-poster-session-2019-03-20_1, num_agents: 44
3434, frame: 708, seq: bytes-cafe-2019-02-07_0, num_agents: 25
3435, frame: 762, seq: hewlett-packard-intersection-2019-01-24_0, num_agents: 20
3436, frame: 615, seq: hewlett-packard-intersection-2019-01-24_0, num_agents: 21
3437, frame: 1188, seq: bytes-cafe-2019-02-07_0, num_agents: 22
3438, frame: 516, seq: packard-poster-session-2019-03-20_2, num_agents: 58
3439, frame: 515, seq: huang-lane-2019-02-12_0, num_agents: 55
Epoch 0:  41%|██████████████████████████████████████▌                                                        | 3440/8483 [14:34<21:21,  3.93it/s, loss=117, v_num=jneo]3440, frame: 1020, seq: bytes-cafe-2019-02-07_0, num_agents: 23
3441, frame: 436, seq: jordan-hall-2019-04-22_0, num_agents: 8
3442, frame: 289, seq: bytes-cafe-2019-02-07_0, num_agents: 22
3443, frame: 328, seq: huang-lane-2019-02-12_0, num_agents: 49
3444, frame: 1167, seq: packard-poster-session-2019-03-20_2, num_agents: 57
3445, frame: 284, seq: gates-ai-lab-2019-02-08_0, num_agents: 21
3446, frame: 1324, seq: hewlett-packard-intersection-2019-01-24_0, num_agents: 18
3447, frame: 862, seq: jordan-hall-2019-04-22_0, num_agents: 12
3448, frame: 141, seq: packard-poster-session-2019-03-20_2, num_agents: 53
3449, frame: 755, seq: hewlett-packard-intersection-2019-01-24_0, num_agents: 21
3450, frame: 1269, seq: hewlett-packard-intersection-2019-01-24_0, num_agents: 19
3451, frame: 644, seq: gates-basement-elevators-2019-01-17_1, num_agents: 13
3452, frame: 798, seq: jordan-hall-2019-04-22_0, num_agents: 12
3453, frame: 891, seq: jordan-hall-2019-04-22_0, num_agents: 11
3454, frame: 199, seq: huang-lane-2019-02-12_0, num_agents: 45
3455, frame: 119, seq: bytes-cafe-2019-02-07_0, num_agents: 21
"""

# excluding poseless, dataloader 2
"""
-------------------------- loading val data --------------------------
loading sequence stlc-111-2019-04-19_0 ...
loading sequence svl-meeting-gates-2-2019-04-08_0 ...
loading sequence svl-meeting-gates-2-2019-04-08_1 ...
loading sequence tressider-2019-03-16_0 ...
loading sequence tressider-2019-03-16_1 ...
total num samples: 3835
invalid_peds in stlc-111-2019-04-19_0: 25968140, valid_peds in stlc-111-2019-04-19_0: 6701045
ratio of invalid_peds in stlc-111-2019-04-19_0: 0.79
invalid_peds in svl-meeting-gates-2-2019-04-08_0: 3239484, valid_peds in svl-meeting-gates-2-2019-04-08_0: 642734
ratio of invalid_peds in svl-meeting-gates-2-2019-04-08_0: 0.83
invalid_peds in svl-meeting-gates-2-2019-04-08_1: 2575047, valid_peds in svl-meeting-gates-2-2019-04-08_1: 446996
ratio of invalid_peds in svl-meeting-gates-2-2019-04-08_1: 0.85
invalid_peds in tressider-2019-03-16_0: 1240929, valid_peds in tressider-2019-03-16_0: 97936
ratio of invalid_peds in tressider-2019-03-16_0: 0.93
len(data) before frame_skip downsample: 3534
len(data) after frame_skip downsample: 3534                                                                                                                            total_invalid_peds: 35382024, total_valid_peds: 8019891
ratio of invalid_peds: 0.82
using 3534 num samples

-------------------------- loading train data --------------------------
loading sequence bytes-cafe-2019-02-07_0 ...
loading sequence gates-ai-lab-2019-02-08_0 ...
loading sequence gates-basement-elevators-2019-01-17_1 ...
loading sequence hewlett-packard-intersection-2019-01-24_0 ...
loading sequence huang-lane-2019-02-12_0 ...
loading sequence jordan-hall-2019-04-22_0 ...
loading sequence packard-poster-session-2019-03-20_1 ...
loading sequence packard-poster-session-2019-03-20_2 ...
total num samples: 8483
invalid_peds in bytes-cafe-2019-02-07_0: 20594879, valid_peds in bytes-cafe-2019-02-07_0: 9620129
ratio of invalid_peds in bytes-cafe-2019-02-07_0: 0.68
invalid_peds in gates-ai-lab-2019-02-08_0: 1299435, valid_peds in gates-ai-lab-2019-02-08_0: 100279
ratio of invalid_peds in gates-ai-lab-2019-02-08_0: 0.93
invalid_peds in gates-basement-elevators-2019-01-17_1: 4871213, valid_peds in gates-basement-elevators-2019-01-17_1: 759318
ratio of invalid_peds in gates-basement-elevators-2019-01-17_1: 0.87
invalid_peds in hewlett-packard-intersection-2019-01-24_0: 33310583, valid_peds in hewlett-packard-intersection-2019-01-24_0: 3064387
ratio of invalid_peds in hewlett-packard-intersection-2019-01-24_0: 0.92
invalid_peds in huang-lane-2019-02-12_0: 9086911, valid_peds in huang-lane-2019-02-12_0: 508861
ratio of invalid_peds in huang-lane-2019-02-12_0: 0.95
invalid_peds in jordan-hall-2019-04-22_0: 7736270, valid_peds in jordan-hall-2019-04-22_0: 978712
ratio of invalid_peds in jordan-hall-2019-04-22_0: 0.89
invalid_peds in packard-poster-session-2019-03-20_1: 11625680, valid_peds in packard-poster-session-2019-03-20_1: 1133557
ratio of invalid_peds in packard-poster-session-2019-03-20_1: 0.91
len(data) before frame_skip downsample: 7697
len(data) after frame_skip downsample: 7697
total_invalid_peds: 127046889, total_valid_peds: 23417184
ratio of invalid_peds: 0.84
using 7697 num samples
"""


# not excluding poseless, dataloader 2

"""
-------------------------- loading val data --------------------------
total num samples: 3835                                                                                                                                       [11/2382]
invalid_peds in stlc-111-2019-04-19_0: 7229875, valid_peds in stlc-111-2019-04-19_0: 28203612
ratio of invalid_peds in stlc-111-2019-04-19_0: 0.2
invalid_peds in svl-meeting-gates-2-2019-04-08_0: 314960, valid_peds in svl-meeting-gates-2-2019-04-08_0: 3593570
ratio of invalid_peds in svl-meeting-gates-2-2019-04-08_0: 0.08
invalid_peds in svl-meeting-gates-2-2019-04-08_1: 561368, valid_peds in svl-meeting-gates-2-2019-04-08_1: 3943007
ratio of invalid_peds in svl-meeting-gates-2-2019-04-08_1: 0.12
invalid_peds in tressider-2019-03-16_0: 104080, valid_peds in tressider-2019-03-16_0: 1251838
ratio of invalid_peds in tressider-2019-03-16_0: 0.08
len(data) before frame_skip downsample: 3835
len(data) after frame_skip downsample: 3835                                                                                                                            total_invalid_peds: 8412491, total_valid_peds: 39316266
ratio of invalid_peds: 0.18
using 3835 num samples

-------------------------- loading train data --------------------------
loading sequence bytes-cafe-2019-02-07_0 ...
loading sequence gates-ai-lab-2019-02-08_0 ...
loading sequence gates-basement-elevators-2019-01-17_1 ...
loading sequence hewlett-packard-intersection-2019-01-24_0 ...
loading sequence huang-lane-2019-02-12_0 ...
loading sequence jordan-hall-2019-04-22_0 ...
loading sequence packard-poster-session-2019-03-20_1 ...
loading sequence packard-poster-session-2019-03-20_2 ...
total num samples: 8483
invalid_peds in bytes-cafe-2019-02-07_0: 1036453, valid_peds in bytes-cafe-2019-02-07_0: 29215764
ratio of invalid_peds in bytes-cafe-2019-02-07_0: 0.03
invalid_peds in gates-ai-lab-2019-02-08_0: 85238, valid_peds in gates-ai-lab-2019-02-08_0: 1359717
ratio of invalid_peds in gates-ai-lab-2019-02-08_0: 0.06
invalid_peds in gates-basement-elevators-2019-01-17_1: 1035751, valid_peds in gates-basement-elevators-2019-01-17_1: 5758394
ratio of invalid_peds in gates-basement-elevators-2019-01-17_1: 0.15
invalid_peds in hewlett-packard-intersection-2019-01-24_0: 5625200, valid_peds in hewlett-packard-intersection-2019-01-24_0: 31100539
ratio of invalid_peds in hewlett-packard-intersection-2019-01-24_0: 0.15
invalid_peds in huang-lane-2019-02-12_0: 641025, valid_peds in huang-lane-2019-02-12_0: 8995509
ratio of invalid_peds in huang-lane-2019-02-12_0: 0.07
invalid_peds in jordan-hall-2019-04-22_0: 6290813, valid_peds in jordan-hall-2019-04-22_0: 8025130
ratio of invalid_peds in jordan-hall-2019-04-22_0: 0.44
invalid_peds in packard-poster-session-2019-03-20_1: 1475054, valid_peds in packard-poster-session-2019-03-20_1: 12021030
ratio of invalid_peds in packard-poster-session-2019-03-20_1: 0.11
len(data) before frame_skip downsample: 8483
len(data) after frame_skip downsample: 8483
total_invalid_peds: 17887875, total_valid_peds: 140587475
ratio of invalid_peds: 0.11
using 8483 num samples
"""