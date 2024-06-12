import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from data.nuscenes_pred_split import get_nuscenes_pred_split
from .preprocessor import preprocess
from .preprocessor_sdd import SDDPreprocess
from .jrdb_kp import jrdb_preprocess
from .jrdb_kp2 import jrdb_preprocess as jrdb_preprocess_new
from .jrdb_kp3 import jrdb_preprocess as jrdb_preprocess_w_action_label
from .jrdb_kp4 import jrdb_preprocess as jrdb_preprocess_full
from .jrdb_kp5 import jrdb_preprocess as jrdb_preprocess_like_hst
from .jrdb_kp6 import jrdb_preprocess as jrdb_preprocess_w_learned_action_label
from .jrdb import jrdb_preprocess as jrdb_vanilla
from .pedx import PedXPreprocess
from .stanford_drone_split import get_stanford_drone_split
from .jrdb_split import *#get_jackrabbot_split, get_jackrabbot_split_easy, get_jackrabbot_split_sanity
from .ethucy_split import get_ethucy_split


class AgentFormerDataset(Dataset):
    """ torch Dataset """

    def __init__(self, parser, split='train', phase='training', trial_ds_size=None, randomize_trial_data=None,
                 frames_list=None, start_frame=None, args=None):
        self.args = args
        if self.args.seq_frame is not None:
            self.single_seq = self.args.seq_frame.split(" ")[0]
            self.single_frame = int(self.args.seq_frame.split(" ")[1])
        self.past_frames = parser.past_frames
        self.exclude_kpless_data = parser.get('exclude_kpless_data', False)
        self.data_root_jrdb = parser.data_root_jrdb
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        if split == 'train':
            self.data_skip = parser.get('data_skip', 1)
        elif parser.get('keep_subsamples', False):
            self.data_skip = 1
        else:
            self.data_skip = self.frame_skip
        self.phase = phase
        self.split = split
        self.dataset = parser.dataset
        self.trial_ds_size = trial_ds_size
        self.data_max_agents = parser.get('data_max_agents', np.inf)
        self.data_min_agents = parser.get('data_min_agents', 0)
        self.randomize_trial_data = randomize_trial_data
        self.ped_categories = parser.get('ped_categories', None)
        if self.ped_categories is not None:
            if isinstance(self.ped_categories, str):
                self.ped_categories = list(map(int, self.ped_categories.split(',')))
            elif isinstance(self.ped_categories, int):
                self.ped_categories = [self.ped_categories]
            else:
                raise ValueError('ped_categories should be a string or an int')

        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        elif parser.dataset == 'trajnet_sdd':
            data_root = parser.data_root_trajnet_sdd
            seq_train, seq_val, seq_test = get_stanford_drone_split()
        elif parser.dataset == 'jrdb':
            data_root = parser.data_root_jrdb
            self.split_type = split_type = parser.get('split_type', 'full')
            if split_type == 'no_egomotion':
                seq_train, seq_val, seq_test = get_jrdb_split_no_egomotion()
            elif split_type == 'egomotion':
                seq_train, seq_val, seq_test = get_jrdb_split_egomotion()
            elif split_type == 'jrdb_full':
                seq_train, seq_val, seq_test = get_jrdb_split_full()
            elif split_type == 'hst_full':
                seq_train, seq_val, seq_test = get_jrdb_hst_split()
            elif split_type == 'half_and_half':
                seq_train, seq_val, seq_test = get_jackrabbot_split_half_and_half()
            elif split_type == 'half_and_half_tiny':
                seq_train, seq_val, seq_test = get_jackrabbot_split_half_and_half_tiny()
            elif split_type == 'sanity':
                seq_train, seq_val, seq_test = get_jackrabbot_split_sanity()
            else:
                assert split_type == 'training_only'
                seq_train, seq_val, seq_test = get_jrdb_training_split_erica()
            self.init_frame = 0
        elif parser.dataset == 'pedx':
            data_root = parser.data_root_pedx
            # use capture date as sequences
            # split 1
            # seq_train, seq_val, seq_test = (['20171130T2000_2', '20171130T2000_3', '20171207T2024'],
            #                                 ['20171130T2000_0'],
            #                                 ['20171130T2000_1'])
            # split 2
            seq_train, seq_val, seq_test = (['20171130T2000_2', '20171130T2000_3', '20171130T2000_4', '20171207T2024_0', ],
                                            ['20171130T2000_0', '20171207T2024_1'],
                                            ['20171130T2000_1', '20171207T2024_2'])
        else:
            raise ValueError('Unknown dataset!')

        if 'sdd' in parser.dataset:
            process_func = SDDPreprocess
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            process_func = preprocess
        elif parser.dataset == 'nuscenes_pred':
            process_func = preprocess
        elif parser.dataset == 'jrdb':# and np.any(['kp' in it for it in parser.input_type]):
            dl_v = parser.get('dataloader_version', 6)
            print(f"dl_v: {dl_v}")
            if dl_v == 1:
                process_func = jrdb_preprocess
            elif dl_v == 2:
                process_func = jrdb_preprocess_new
            elif dl_v == 3:
                process_func = jrdb_preprocess_w_action_label
            elif dl_v == 4:
                process_func = jrdb_preprocess_full
            elif dl_v == 5:
                process_func = jrdb_preprocess_like_hst
            elif dl_v == 6:
                process_func = jrdb_preprocess_w_learned_action_label
            else:
                assert dl_v == 0
                import ipdb; ipdb.set_trace()
                process_func = jrdb_vanilla
        else:
            assert parser.dataset == 'pedx'
            process_func = PedXPreprocess
        self.data_root = data_root

        print("\n-------------------------- loading %s data --------------------------" % split)
        if self.split == 'train' and not parser.get('sanity', False):
            self.sequence_to_load = seq_train
        elif self.split == 'val':
            self.sequence_to_load = seq_val
        elif self.split == 'test' or parser.get('sanity', False):
            self.sequence_to_load = seq_test
        else:
            assert False, 'error'

        num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        self.seq_names = []
        self.d = {}
        self.dnf = {}
        for seq_name in self.sequence_to_load:
            if self.args.seq_frame is not None:
                if self.single_seq != seq_name:
                    continue
            print("loading sequence {} ...".format(seq_name))
            preprocessor = process_func(data_root, seq_name, parser, self.split, self.phase)
            min_window_size = parser.min_past_frames + parser.min_future_frames

            # num_seq_samples = (preprocessor.num_fr - (min_window_size - 1) * self.frame_skip) // self.data_skip + 1
            num_seq_samples = int(np.ceil((preprocessor.num_fr - (min_window_size - 1) * self.frame_skip) / self.data_skip))
            self.d[seq_name] = num_seq_samples
            self.dnf[seq_name] = preprocessor.num_fr

            num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            self.seq_names.append(seq_name)
            if self.trial_ds_size is not None and num_total_samples >= 20 * self.trial_ds_size:
                break

        print(f'total num samples: {num_total_samples}')
        self.num_total_samples = num_total_samples

        self.preprocess_data = True if trial_ds_size is not None else parser.get('preprocess_data', False)  # or args.test_certain_frames_only
        if args.test_certain_frames_only:
            self.get_certain_frames(args.frames)
        elif self.preprocess_data:
            self.get_preprocessed_data(num_total_samples, frames_list, start_frame)
        else:
            self.sample_list = [None for _ in range(num_total_samples)]

    def get_certain_frames(self, frames):
        datas = []
        for seq_name, frame in frames:
            try:
                seq_index = self.seq_names.index(seq_name)
            except ValueError:
                print(f"seq_name {seq_name} not found in {self.seq_names}")
                continue
            data = self.sequence[seq_index](frame)
            if data is None:
                continue
            datas.append(data)
        self.sample_list = datas
        self.num_total_samples = len(datas)

    def get_preprocessed_data(self, num_total_samples, frames_list, start_frame):
        """ get data sequence windows from raw data blocks txt files"""
        datas = []

        from collections import defaultdict
        seq_to_frame_ids = defaultdict(list)
        seq_to_num_agents = defaultdict(list)
        seq_to_frame_to_agent_ids = defaultdict(dict)
        for idx in tqdm(range(num_total_samples), desc='preprocessing data'):
            seq_index, frame = self.get_seq_and_frame(idx)
            seq = self.sequence[seq_index]
            data = seq(frame)
            if data is None:
                continue
            # if last_seq is not None and data['seq'] != last_seq:
                # print(f'invalid_peds in {last_seq}: {invalid_peds_this_environment}, '
                #       f'valid_peds in {last_seq}: {valid_peds_this_environment}')
                # print(f'ratio of invalid_peds in {last_seq}: {round(invalid_peds_this_environment / (invalid_peds_this_environment + valid_peds_this_environment), 2)}')
                # invalid_peds_this_environment = 0
                # valid_peds_this_environment = 0
            # last_seq = data['seq']
            # total_invalid_peds += seq.total_invalid_peds
            # total_valid_peds += seq.total_valid_peds
            # invalid_peds_this_environment += seq.total_invalid_peds
            # valid_peds_this_environment += seq.total_valid_peds
            if frames_list is not None and isinstance(frames_list, list) and len(frames_list) > 0 \
                    and frame not in frames_list:
                continue
            if start_frame is not None and frame < start_frame:
                continue
            if self.args.seq_frame is not None and not (self.single_seq == data['seq'] and int(self.single_frame) == data['frame']):
                continue
            if self.ped_categories is not None:  # filter out pedestrians that are not in the specified categories
                masks_dir = '../trajectory_reward/results/interactions'
                mask_path = os.path.join(masks_dir, data['seq'], f"frame_{data['frame']*10:06d}.txt")
                try:
                    mask = np.loadtxt(mask_path)
                    if len(mask.shape) == 1:
                        mask = np.expand_dims(mask, axis=0)
                    num_peds_in_cats = np.sum(mask[:, self.ped_categories])
                    if num_peds_in_cats == 0:
                        continue
                except FileNotFoundError:
                    print(f"mask file not found: {mask_path}")
            num_agents = len(data['pre_motion'])
            if num_agents > self.data_max_agents:
                continue
            if num_agents < self.data_min_agents:
                continue
            datas.append(data)
            if self.trial_ds_size is not None and len(datas) == self.trial_ds_size and not self.randomize_trial_data:
                print(f"test mode: limiting to ds of size {self.trial_ds_size}")
                break
            seq_to_frame_ids[self.seq_names[seq_index]].append(frame)
            seq_to_num_agents[self.seq_names[seq_index]].append(num_agents)
            seq_to_frame_to_agent_ids[self.seq_names[seq_index]][frame] = sorted(data['valid_id'])

        if self.randomize_trial_data:
            print(f"taking elements from different parts of the dataset for diverse data")
            # np.shuffle(datas)
            # datas = datas[:self.trial_ds_size]
            skip = len(datas) // self.trial_ds_size
            datas = datas[::skip]
        # print(f"len(data) before frame_skip downsample: {len(datas)}")
        self.sample_list = datas#[::self.data_skip]

        print(f"num sequences per scene {self.d=}")
        # print(f"which frame ids (samples) per scene are being used {seq_to_frame_ids=}")
        # save frame ids:
        import pickle
        split2 = 'test' if self.split == 'val' else self.split
        ts_or_kp = 'kp' if self.exclude_kpless_data else 'ts'
        mdr = 1000 if '1000' in self.data_root_jrdb else 15

        print('total num sequences', sum(self.d.values()))
        print(f"num total frames {self.dnf=}")
        print(f"total num sequences being used: {sum([len(v) for k,v in seq_to_frame_ids.items()])}")  # 5354
        # num_seq_per_scene = {k: len(v) for k, v in seq_to_frame_ids.items()}
        # print(f"{num_seq_per_scene=}")
        # agent_ids_save_path = f'../human-scene-transformer/human_scene_transformer/af_{split2}_seq_to_frame_id_to_agent_ids_full-{ts_or_kp}_mdr-{mdr}.pkl'
        # with open(agent_ids_save_path, 'wb') as f:
        #     pickle.dump(seq_to_frame_to_agent_ids, f)
        #     print("saved to", agent_ids_save_path)

        print('avg num agents per frame per scene=', {scene: round(np.mean([len(agent_ids) for frame_id, agent_ids in frames.items()]),2) for scene, frames in seq_to_frame_to_agent_ids.items()})
        # print('unique agents across entire scene=', {scene: len(set().union(*frames.values())) for scene, frames in seq_to_frame_to_agent_ids.items()})
        print('num unique agents across entire scene=', {scene: set().union(*frames.values()) for scene, frames in seq_to_frame_to_agent_ids.items()})

        # num sequences
        # total num sequences 3697
        # num total frames self.dnf={'clark-center-2019-02-28_1': 1439, 'forbes-cafe-2019-01-22_0': 1446, 'gates-basement-elevators-2019-01-17_1': 1009, 'huang-2-2019-01-25_0': 872, 'jordan-hall-2019-04-22_0': 1522, 'nvidia-aud-2019-04-18_0': 1012, 'packard-poster-session-2019-03-20_2': 1371, 'svl-meeting-gates-2-2019-04-08_1': 871, 'tressider-2019-04-26_2': 1440, 'discovery-walk-2019-02-28_1': 789, 'gates-basement-elevators-2019-01-17_0': 638, 'hewlett-class-2019-01-23_0': 791, 'huang-intersection-2019-01-22_0': 1659, 'meyer-green-2019-03-16_1': 1006, 'nvidia-aud-2019-04-18_2': 498, 'serra-street-2019-01-30_0': 1399, 'tressider-2019-03-16_2': 642, 'tressider-2019-04-26_3': 1659}

        # full kp
        # total num sequences being used: 1993
        # num_seq_per_scene={'clark-center-2019-02-28_1': 147, 'forbes-cafe-2019-01-22_0': 241, 'gates-basement-elevators-2019-01-17_1': 162, 'huang-2-2019-01-25_0': 71, 'jordan-hall-2019-04-22_0': 267, 'nvidia-aud-2019-04-18_0': 179, 'packard-poster-session-2019-03-20_2': 256, 'svl-meeting-gates-2-2019-04-08_1': 120, 'tressider-2019-04-26_2': 267, 'discovery-walk-2019-02-28_1': 6, 'gates-basement-elevators-2019-01-17_0': 12, 'hewlett-class-2019-01-23_0': 1, 'meyer-green-2019-03-16_1': 35, 'serra-street-2019-01-30_0': 176, 'tressider-2019-04-26_3': 53}
        # avg_num_agents_per_frame_per_scene={'clark-center-2019-02-28_1': 1.46, 'forbes-cafe-2019-01-22_0': 2.49, 'gates-basement-elevators-2019-01-17_1': 2.48, 'huang-2-2019-01-25_0': 1.24, 'jordan-hall-2019-04-22_0': 3.41, 'nvidia-aud-2019-04-18_0': 4.15, 'packard-poster-session-2019-03-20_2': 7.51, 'svl-meeting-gates-2-2019-04-08_1': 2.14, 'tressider-2019-04-26_2': 4.99, 'discovery-walk-2019-02-28_1': 1.0, 'gates-basement-elevators-2019-01-17_0': 1.0, 'hewlett-class-2019-01-23_0': 1.0, 'meyer-green-2019-03-16_1': 1.0, 'serra-street-2019-01-30_0': 1.05, 'tressider-2019-04-26_3': 1.04}
        # avg num unique agents across entire scene= {'clark-center-2019-02-28_1': 18, 'forbes-cafe-2019-01-22_0': 19, 'gates-basement-elevators-2019-01-17_1': 22, 'huang-2-2019-01-25_0': 6, 'jordan-hall-2019-04-22_0': 58, 'nvidia-aud-2019-04-18_0': 21, 'packard-poster-session-2019-03-20_2': 28, 'svl-meeting-gates-2-2019-04-08_1': 14, 'tressider-2019-04-26_2': 50, 'discovery-walk-2019-02-28_1': 1, 'gates-basement-elevators-2019-01-17_0': 1, 'hewlett-class-2019-01-23_0': 1, 'meyer-green-2019-03-16_1': 1, 'serra-street-2019-01-30_0': 3, 'tressider-2019-04-26_3': 6}

        # full ts
        # total num sequences being used: 3615
        # num_seq_per_scene={'clark-center-2019-02-28_1': 267, 'forbes-cafe-2019-01-22_0': 272, 'gates-basement-elevators-2019-01-17_1': 184, 'huang-2-2019-01-25_0': 141, 'jordan-hall-2019-04-22_0': 287, 'nvidia-aud-2019-04-18_0': 185, 'packard-poster-session-2019-03-20_2': 257, 'svl-meeting-gates-2-2019-04-08_1': 157, 'tressider-2019-04-26_2': 270, 'discovery-walk-2019-02-28_1': 140, 'gates-basement-elevators-2019-01-17_0': 107, 'hewlett-class-2019-01-23_0': 107, 'huang-intersection-2019-01-22_0': 312, 'meyer-green-2019-03-16_1': 184, 'nvidia-aud-2019-04-18_2': 82, 'serra-street-2019-01-30_0': 262, 'tressider-2019-03-16_2': 111, 'tressider-2019-04-26_3': 290}
        # avg num agents per frame per scene= {'clark-center-2019-02-28_1': 6.35, 'forbes-cafe-2019-01-22_0': 13.35, 'gates-basement-elevators-2019-01-17_1': 9.15, 'huang-2-2019-01-25_0': 4.74, 'jordan-hall-2019-04-22_0': 7.07, 'nvidia-aud-2019-04-18_0': 10.85, 'packard-poster-session-2019-03-20_2': 26.0, 'svl-meeting-gates-2-2019-04-08_1': 4.96, 'tressider-2019-04-26_2': 21.85, 'discovery-walk-2019-02-28_1': 4.72, 'gates-basement-elevators-2019-01-17_0': 2.56, 'hewlett-class-2019-01-23_0': 2.61, 'huang-intersection-2019-01-22_0': 3.38, 'meyer-green-2019-03-16_1': 3.04, 'nvidia-aud-2019-04-18_2': 3.56, 'serra-street-2019-01-30_0': 2.98, 'tressider-2019-03-16_2': 4.41, 'tressider-2019-04-26_3': 5.77}
        # unique agents across entire scene= {'clark-center-2019-02-28_1': 54, 'forbes-cafe-2019-01-22_0': 54, 'gates-basement-elevators-2019-01-17_1': 28, 'huang-2-2019-01-25_0': 26, 'jordan-hall-2019-04-22_0': 79, 'nvidia-aud-2019-04-18_0': 31, 'packard-poster-session-2019-03-20_2': 59, 'svl-meeting-gates-2-2019-04-08_1': 19, 'tressider-2019-04-26_2': 107, 'discovery-walk-2019-02-28_1': 20, 'gates-basement-elevators-2019-01-17_0': 26, 'hewlett-class-2019-01-23_0': 19, 'huang-intersection-2019-01-22_0': 40, 'meyer-green-2019-03-16_1': 14, 'nvidia-aud-2019-04-18_2': 16, 'serra-street-2019-01-30_0': 23, 'tressider-2019-03-16_2': 21, 'tressider-2019-04-26_3': 75

        # save all this to ../viz/af_data_stats.npy
        # import numpy as np
        # np.save('../viz/af_data_stats.npy', {'d': self.d, 'dnf': self.dnf, 'seq_to_frame_ids': seq_to_frame_ids})

        ### jrdb split
        # print("differences (af - hst)")
        # hst sequence counts by scene
        # {'cubberly-auditorium-2019-04-22_1_test': 201, 'discovery-walk-2019-02-28_0_test': 128, 'discovery-walk-2019-02-28_1_test': 143, 'food-trucks-2019-02-12_0_test': 306, 'gates-ai-lab-2019-04-17_0_test': 229, 'gates-basement-elevators-2019-01-17_0_test': 112, 'gates-foyer-2019-01-17_0_test': 317, 'gates-to-clark-2019-02-28_0_test': 70, 'hewlett-class-2019-01-23_0_test': 143, 'hewlett-class-2019-01-23_1_test': 142, 'huang-2-2019-01-25_1_test': 99, 'huang-intersection-2019-01-22_0_test': 317, 'indoor-coupa-cafe-2019-02-06_0_test': 318, 'lomita-serra-intersection-2019-01-30_0_test': 220, 'meyer-green-2019-03-16_1_test': 186, 'nvidia-aud-2019-01-25_0_test': 230, 'nvidia-aud-2019-04-18_1_test': 85, 'nvidia-aud-2019-04-18_2_test': 84, 'outdoor-coupa-cafe-2019-02-06_0_test': 315, 'quarry-road-2019-02-28_0_test': 72, 'serra-street-2019-01-30_0_test': 265, 'stlc-111-2019-04-19_1_test': 84, 'stlc-111-2019-04-19_2_test': 80, 'tressider-2019-03-16_2_test': 113, 'tressider-2019-04-26_0_test': 229, 'tressider-2019-04-26_1_test': 317, 'tressider-2019-04-26_3_test': 317}
        # hst sequence counts by scene, minus sequences with 0 should_predict values
        # {'cubberly-auditorium-2019-04-22_1_test': 187, 'discovery-walk-2019-02-28_0_test': 121, 'discovery-walk-2019-02-28_1_test': 136, 'food-trucks-2019-02-12_0_test': 299, 'gates-ai-lab-2019-04-17_0_test': 222, 'gates-basement-elevators-2019-01-17_0_test': 105, 'gates-foyer-2019-01-17_0_test': 308, 'gates-to-clark-2019-02-28_0_test': 61, 'hewlett-class-2019-01-23_0_test': 136, 'hewlett-class-2019-01-23_1_test': 135, 'huang-2-2019-01-25_1_test': 92, 'huang-intersection-2019-01-22_0_test': 310, 'indoor-coupa-cafe-2019-02-06_0_test': 311, 'lomita-serra-intersection-2019-01-30_0_test': 213, 'meyer-green-2019-03-16_1_test': 179, 'nvidia-aud-2019-01-25_0_test': 223, 'nvidia-aud-2019-04-18_1_test': 78, 'nvidia-aud-2019-04-18_2_test': 77, 'outdoor-coupa-cafe-2019-02-06_0_test': 308, 'quarry-road-2019-02-28_0_test': 65, 'serra-street-2019-01-30_0_test': 258, 'stlc-111-2019-04-19_1_test': 77, 'stlc-111-2019-04-19_2_test': 72, 'tressider-2019-03-16_2_test': 106, 'tressider-2019-04-26_0_test': 222, 'tressider-2019-04-26_1_test': 310, 'tressider-2019-04-26_3_test': 310}
        # number of sequences with 0 should_predict values: 201
        # total number of sequences: 4921

        # for k in d:
        #     print(f'{k}: {d[k] - d2[k+"_test"]}')
        # import ipdb; ipdb.set_trace()

        # print(f"len(data) after frame_skip downsample: {len(self.sample_list)}")
        # print(f'total_invalid_peds: {total_invalid_peds}, total_valid_peds: {total_valid_peds}')
        # print(f'ratio of invalid_peds: {round(total_invalid_peds / (total_invalid_peds + total_valid_peds), 2)}')
        print(f'using {len(self.sample_list)} num samples')

        print("------------------------------ done --------------------------------\n")
        if len(self.sample_list) < 10:
            print("frames_list:", [data['frame'] for data in self.sample_list])

    def __len__(self):
        if self.preprocess_data:
            return len(self.sample_list)
        return self.num_total_samples

    def get_seq_and_frame(self, index):
        index_tmp = index
        # find the seq_name (environment) this index corresponds to, and then the frame in that environment
        for seq_index in range(len(self.num_sample_list)):  # 0-indexed  # for each seq_name
            if index_tmp < self.num_sample_list[seq_index]:  # if the index is in this environment scene
                frame_index = (index_tmp * self.data_skip
                               + (self.min_past_frames - 1) * self.frame_skip  # return frame is the last obs frame
                               + self.sequence[seq_index].init_frame)  # from 0-indexed list index to 1-indexed frame index (for mot)
                if "half_and_half" not in self.split_type:
                    assert self.sequence[seq_index].init_frame == 0
                return seq_index, frame_index
            else: # keep going through scenes
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def get_item_preprocessed(self, idx):
        return self.sample_list[idx]

    def get_item_online(self, idx):
        seq_index, frame = self.get_seq_and_frame(idx)
        seq = self.sequence[seq_index]
        data = seq(frame)
        return data

    def __getitem__(self, idx, scene=None):
        """Returns a single example from dataset
            Args:
                idx: Index of scenario
            Returns:
                output: Necessary values for scenario
        """
        if scene is not None:
            seq_idx = self.seq_names.index(scene)
            return self.sequence[seq_idx](idx)
        if self.preprocess_data:
            return self.get_item_preprocessed(idx)
        if self.sample_list[idx] is not None:
            return self.sample_list[idx]
        self.sample_list[idx] = self.get_item_online(idx)
        return self.sample_list[idx]
        # return self.get_item_online(idx)

    @staticmethod
    def collate(batch):
        """batch: list of data objects """
        return batch[0]