import os
import torch

from itertools import zip_longest
from functools import partial
# from torch import string_classes
from torch.utils.data import Dataset

from data.nuscenes_pred_split import get_nuscenes_pred_split
import numpy as np

from .preprocessor import preprocess
from .preprocessor_sdd import SDDPreprocess
from .jrdb import jrdb_preprocess
from .pedx import PedXPreprocess
from .stanford_drone_split import get_stanford_drone_split
from .jrdb_split import get_jackrabbot_split
from .ethucy_split import get_ethucy_split
from human_scene_transformer.jrdb import torch_dataset

from data.human_scene_transformer.jrdb import dataset_params as jrdb_dataset_params
from data.human_scene_transformer.jrdb import input_fn

import tensorflow_datasets as tfds

import gin

class HSTDataset(Dataset):
    """ torch Dataset """

    def __init__(self,
               dataset_params: jrdb_dataset_params.JRDBDatasetParams,
               train: bool = False):

        self.dataset_params = dataset_params
        self.train = train

        if self.train:
            self.dataset = list(tfds.as_numpy(input_fn.get_train_dataset(self.dataset_params, shuffle=False, repeat=False)))
        else:
            self.dataset = list(tfds.as_numpy(input_fn.get_eval_dataset(self.dataset_params)))

    def __getitem__(self, idx):
        return self.preprocess(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def zero_nans(self, arr):
        return np.where(np.isnan(arr), 0, arr)

    def preprocess(self, feature_dict):
        """
        Takes dictionary of features from one hst datapoint

        Converts this into preprocessed format for Agentformer
        """
        past_frames = int(self.dataset_params.num_history_steps) + 1
        future_frames = int(self.dataset_params.num_steps) - int(self.dataset_params.num_history_steps) - 1

        A, T, _ = np.shape(feature_dict['agents/position'])

        centered_motion = feature_dict['agents/position'] # - feature_dict['robot/position']
        pre_motion = centered_motion[:, :past_frames, :]
        fut_motion = centered_motion[:, past_frames:, :]

        keypoints = np.reshape(feature_dict['agents/keypoints'], (A, T, 33, 3))

        pre_motion_kp = self.zero_nans(keypoints[:, :past_frames, :, :])
        fut_motion_kp = self.zero_nans(keypoints[:, past_frames:, :, :])

        fut_motion_mask = np.where(np.isnan(pre_motion[:, :, 0]), 0, 1)
        pre_motion_mask = np.where(np.isnan(fut_motion[:, :, 0]), 0, 1)

        pre_motion = self.zero_nans(pre_motion)
        fut_motion = self.zero_nans(fut_motion)

        heading = self.zero_nans(feature_dict['agents/orientation'][:, -1, :].squeeze())

        agent_mask = np.zeros((A))

        data = {
            'pre_motion' : pre_motion,
            'pre_kp' : pre_motion_kp,
            'fut_motion' : fut_motion,
            'fut_kp' : fut_motion_kp,
            'fut_motion_mask' : fut_motion_mask,
            'pre_motion_mask' : pre_motion_mask,
            'heading' : heading,
            'traj_scale': 1,
            'frame' : feature_dict['scene/timestep']
        }

        for key, val in data.items():
            data[key] = torch.from_numpy(val)

        return data

