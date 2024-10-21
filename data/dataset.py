import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from data.tbd_split import *
from .jrdb_split import *
from .tbd import TBDPreprocess
from .jrdb_kp import jrdb_preprocess as jrdb_preprocess_full
from .jrdb_kp_missing_ts import jrdb_preprocess as jrdb_preprocess_like_hst
from .jrdb_kp_action import jrdb_preprocess as jrdb_preprocess_w_learned_action_label
from .pedx import PedXPreprocess
from traj_toolkit.data_scripts.tbd_interesting_scenes import INTERESTING_SCENES


class AgentFormerDataset(Dataset):
    """ torch Dataset """

    def __init__(self, parser, split='train', phase='training', trial_ds_size=None, randomize_trial_data=None,
                 frames_list=None, start_frame=None, args=None):
        self.args = args
        if self.args.seq_frame is not None:
            self.single_seq = self.args.seq_frame.split(" ")[0]
            self.single_frame = int(self.args.seq_frame.split(" ")[1])
        self.past_frames = parser.past_frames
        self.future_frames = parser.future_frames
        self.exclude_kpless_data = parser.get('exclude_kpless_data', False)
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        if split == 'train':
            self.data_skip = parser.get('data_skip_train', None)
        else:
            self.data_skip = parser.get('data_skip_eval', None)
        if parser.get('data_skip_train', None) is None or parser.get('data_skip_eval', None) is None:
            import ipdb; ipdb.set_trace()
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

        self.split_type = split_type = parser.get('split_type', 'full')
        if split in ['val','test'] and parser.get('test_split', None) is not None:
            self.split_type =split_type = parser.test_split
            print(f"testing with {split_type=}")
        if parser.dataset == 'jrdb':
            data_root = parser.data_root_jrdb
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
        elif parser.dataset == 'tbd':
            data_root = parser.data_root_tbd
            if split_type == 'full':
                seq_train, seq_val, seq_test = get_tbd_split()
            elif split_type == 'sanity2':
                seq_train, seq_val, seq_test = get_tbd_split_sanity2()
            elif split_type == 'sanity':
                seq_train, seq_val, seq_test = get_tbd_split(sanity=True)
            elif split_type == 'small':
                seq_train, seq_val, seq_test = get_tbd_split_small()
            elif split_type == 'allan':
                seq_train, seq_val, seq_test = get_tbd_split_allan()
            elif 'interesting_test' in split_type:
                seq_train, seq_val, seq_test = get_test_tbd_interesting_scenes()
                print(f"{seq_test=}")
            elif 'interesting' in split_type:
                seq_train, seq_val, seq_test = get_tbd_interesting_scenes()
            else:
                raise ValueError('Unknown split type!')
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        if parser.dataset == 'jrdb':# and np.any(['kp' in it for it in parser.input_type]):
            dl_v = parser.get('dataloader_version', 6)
            print(f"dl_v: {dl_v}")
            if dl_v == 4:
                process_func = jrdb_preprocess_full
            elif dl_v == 5:
                process_func = jrdb_preprocess_like_hst
            elif dl_v == 6:
                process_func = jrdb_preprocess_w_learned_action_label
            else:
                raise ValueError('Unknown dataloader version!')
        elif parser.dataset == 'tbd':
            data_root = parser.data_root_tbd
            process_func = TBDPreprocess
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
            if 'interesting' in split_type:
                margin=2
                num_seq_samples = len(INTERESTING_SCENES[seq_name]) + margin*2
                self.scenes = get_frame_ids_with_margin(INTERESTING_SCENES, frame_skip=self.frame_skip, margin=margin)
                print(f"{seq_name} {num_seq_samples=}")
            else:
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
        self.get_items_online = parser.get('get_items_online', False)
        if args.test_certain_frames_only:
            self.get_certain_frames(args.peds)
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
            skip = len(datas) // self.trial_ds_size
            datas = datas[::skip]
        self.sample_list = datas

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
                if 'interesting' in self.split_type:
                    frame_index = self.scenes[self.seq_names[seq_index]][index_tmp]
                else:
                    frame_index = (index_tmp * self.data_skip
                                + (self.min_past_frames - 1) * self.frame_skip  # return frame is the last obs frame
                                + self.sequence[seq_index].init_frame)  # from 0-indexed list index to 1-indexed frame index (for mot)
                    if "half_and_half" not in self.split_type:
                        assert self.sequence[seq_index].init_frame == 0
                        # if self.sequence[seq_index].init_frame != 0:
                            # subtract init_frame to get the correct frame index
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
        if self.get_items_online:
            return self.get_item_online(idx)
        import ipdb; ipdb.set_trace()
        # not get online if possible
        if self.sample_list[idx] is not None:
            return self.sample_list[idx]
        self.sample_list[idx] = self.get_item_online(idx)
        return self.sample_list[idx]

    @staticmethod
    def collate(batch):
        """batch: list of data objects """
        return batch[0]