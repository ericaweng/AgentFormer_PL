import os

from torch.utils.data import Dataset

from data.nuscenes_pred_split import get_nuscenes_pred_split
import numpy as np

from .preprocessor import preprocess
from .preprocessor_sdd import SDDPreprocess
from .jrdb_kp import jrdb_preprocess
from .jrdb_kp2 import jrdb_preprocess as jrdb_preprocess_new
from .jrdb_kp3 import jrdb_preprocess as jrdb_preprocess_w_action_label
from .jrdb_kp4 import jrdb_preprocess as jrdb_preprocess_full
from .jrdb_kp5 import jrdb_preprocess as jrdb_preprocess_like_hst
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
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        if split == 'train':
            self.data_skip = parser.get('data_skip', 1)
        else:
            self.data_skip = 1
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
            split_type = parser.get('split_type', 'full')
            if split_type == 'no_egomotion':
                seq_train, seq_val, seq_test = get_jrdb_split_no_egomotion()
            elif split_type == 'egomotion':
                seq_train, seq_val, seq_test = get_jrdb_split_egomotion()
            elif split_type == 'full':
                seq_train, seq_val, seq_test = get_jrdb_split_full()
            elif split_type == 'half_and_half':
                seq_train, seq_val, seq_test = get_jackrabbot_split_half_and_half()
            elif split_type == 'half_and_half_tiny':
                seq_train, seq_val, seq_test = get_jackrabbot_split_half_and_half_tiny()
            elif split_type == 'easy':
                seq_train, seq_val, seq_test = get_jackrabbot_split_easy()
            elif split_type == 'sanity':
                seq_train, seq_val, seq_test = get_jackrabbot_split_sanity()
            else:
                assert split_type == 'normal'
                seq_train, seq_val, seq_test = get_jackrabbot_split()
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
            dl_v = parser.get('dataloader_version', 5)
            print(f"dl_v: {dl_v}")
            if dl_v == 1:
                import ipdb; ipdb.set_trace()
                process_func = jrdb_preprocess
            elif dl_v == 2:
                process_func = jrdb_preprocess_new
            elif dl_v == 3:
                process_func = jrdb_preprocess_w_action_label
            elif dl_v == 4:
                process_func = jrdb_preprocess_full
            elif dl_v == 5:
                process_func = jrdb_preprocess_like_hst
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
        for seq_name in self.sequence_to_load:
            if self.args.seq_frame is not None:
                if self.single_seq != seq_name:
                    continue
            print("loading sequence {} ...".format(seq_name))
            preprocessor = process_func(data_root, seq_name, parser, self.split, self.phase)

            num_seq_samples = (preprocessor.num_fr - (
                        parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip) // self.data_skip + 1

            num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            if self.trial_ds_size is not None and num_total_samples >= 20 * self.trial_ds_size:
                break

        print(f'total num samples: {num_total_samples}')
        self.num_total_samples = num_total_samples

        self.preprocess_data = True if trial_ds_size is not None else parser.get('preprocess_data', False)
        if self.preprocess_data:
            self.get_preprocessed_data(num_total_samples, frames_list, start_frame)
        else:
            self.sample_list = [None for _ in range(num_total_samples)]

    def get_preprocessed_data(self, num_total_samples, frames_list, start_frame):
        """ get data sequence windows from raw data blocks txt files"""
        datas = []

        # from collections import defaultdict
        # d = defaultdict(int)
        for idx in list(range(num_total_samples)):
            seq_index, frame = self.get_seq_and_frame(idx)
            seq = self.sequence[seq_index]
            data = seq(frame)
            if data is None:
                continue
            # d[data['seq']] += 1
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
        if self.randomize_trial_data:
            print(f"taking elements from different parts of the dataset for diverse data")
            # np.shuffle(datas)
            # datas = datas[:self.trial_ds_size]
            skip = len(datas) // self.trial_ds_size
            datas = datas[::skip]
        print(f"len(data) before frame_skip downsample: {len(datas)}")
        self.sample_list = datas#[::self.data_skip]
        # print(d)
        # print('total', sum(d.values()))
        # import ipdb; ipdb.set_trace()
        print(f"len(data) after frame_skip downsample: {len(self.sample_list)}")
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
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = (index_tmp * self.data_skip
                               + (self.min_past_frames - 1) * self.frame_skip  # return frame is the last obs frame
                               + self.sequence[seq_index].init_frame)  # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def get_item_preprocessed(self, idx):
        return self.sample_list[idx]

    def get_item_online(self, idx):
        seq_index, frame = self.get_seq_and_frame(idx)
        seq = self.sequence[seq_index]
        data = seq(frame)
        return data

    def __getitem__(self, idx):
        """Returns a single example from dataset
            Args:
                idx: Index of scenario
            Returns:
                output: Necessary values for scenario
        """
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