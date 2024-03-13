import os
import numpy as np
import torch
import argparse
import json


# ACTIONS = ["standing","walking", "sitting", "holding sth",
#         "listening to someone", "talking to someone", "looking at robot",
#         "looking into sth",  "cycling", "looking at sth", "going upstairs",
#         "bending", "typing", "interaction with door", "eating sth",
#         "talking on the phone", "going downstairs","scootering","pointing at sth",
#         "pushing","reading", "skating", "running", "greeting gestures",'writing','lying']
# actions_to_idx = {action: i for i, action in enumerate(ACTIONS)}
# print(f"actions_to_idx: {actions_to_idx}")

ACTIONS_TO_IDX = {'impossible': -1, 'standing': 0, 'walking': 1, 'sitting': 2, 'holding sth': 3, 'listening to someone': 4,
                  'talking to someone': 5, 'looking at robot': 6, 'looking into sth': 7, 'cycling': 8,
                  'looking at sth': 9, 'going upstairs': 10, 'bending': 11, 'typing': 12, 'interaction with door': 13,
                  'eating sth': 14, 'talking on the phone': 15, 'going downstairs': 16, 'scootering': 17,
                  'pointing at sth': 18, 'pushing': 19, 'reading': 20, 'skating': 21, 'running': 22,
                  'greeting gestures': 23, 'writing': 24, 'lying': 25, 'pulling': 26,}

# frequencies
# {'impossible': 3628, 'standing': 314107, 'walking': 320120, 'sitting': 164872, 'holding sth': 170228,
#  'listening to someone': 84825, 'talking to someone': 80483, 'looking at robot': 25411, 'looking into sth': 25270,
#  'cycling': 15583, 'looking at sth': 15998, 'going upstairs': 11381, 'bending': 7834, 'typing': 5594,
#  'interaction with door': 4689, 'eating sth': 4251, 'talking on the phone': 3685, 'going downstairs': 2346,
#  'scootering': 1382, 'pointing at sth': 2398, 'pushing': 1972, 'reading': 2246, 'skating': 1480,
#  'running': 256, 'greeting gestures': 566, 'writing': 687, 'lying': 201, 'pulling': 206}

# 'pulling': 206
# 'lying': 201

ACTIONS = ['standing', 'walking', 'sitting', 'cycling', ]#'going upstairs', 'bending', 'going downstairs', 'scootering', 'skating', 'running', 'lying']
# ACTIONS = ["walking", "standing", "sitting"]

NO_EGOMOTION = [
    'bytes-cafe-2019-02-07_0',
    'clark-center-2019-02-28_0',
    'clark-center-2019-02-28_1',
    'clark-center-intersection-2019-02-28_0',
    'cubberly-auditorium-2019-04-22_0',
    'forbes-cafe-2019-01-22_0',
    'gates-159-group-meeting-2019-04-03_0',
    'gates-ai-lab-2019-02-08_0',
    'gates-basement-elevators-2019-01-17_1',
    'gates-to-clark-2019-02-28_1',
    'hewlett-packard-intersection-2019-01-24_0',
    'huang-2-2019-01-25_0',
    'huang-basement-2019-01-25_0',
    'huang-lane-2019-02-12_0',
    'jordan-hall-2019-04-22_0',
    'memorial-court-2019-03-16_0',
    'meyer-green-2019-03-16_0',
    'nvidia-aud-2019-04-18_0',
    'packard-poster-session-2019-03-20_0',
    'packard-poster-session-2019-03-20_1',
    'packard-poster-session-2019-03-20_2',
    'stlc-111-2019-04-19_0',
    'svl-meeting-gates-2-2019-04-08_0',
    'svl-meeting-gates-2-2019-04-08_1',
    'tressider-2019-03-16_0',
    'tressider-2019-03-16_1',
    'tressider-2019-04-26_2'
]

class PoseDataset(torch.utils.data.Dataset):

    def __init__(self, root, actions, partition="train"):
        self.actions = actions

        self.data = np.load(f"{root}/{partition}/data.npy")
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        pose = torch.tensor(self.data[ind, :34]).float()
        action = torch.tensor(self.data[ind, 34]).long()
        return pose, action


def main(args):
    # load ronnie preprocessed data
    # data_labels = np.load('datasets/jrdb_action_labels/train/data.npy')
    # print(f"data_labels: {dir(data_labels)}")

    # preprocess my own data by adding action labels into the pose data

    action_labels = {}
    action_label_scores = {}
    frequencies = {action: 0 for action in ACTIONS_TO_IDX}
    for scene in NO_EGOMOTION:
        f = open(f"{args.data_root}/train/labels/labels_2d_stitched/{scene}.json")
        labels = json.load(f)["labels"]

        action_labels[scene] = {}
        action_label_scores[scene] = {}
        for frame, labels_this_frame in labels.items():
            frame = int(frame.split(".")[0])

            action_labels[scene][frame] = {}
            action_label_scores[scene][frame] = {}
            for ped_label in labels_this_frame:
                action_ids = []
                action_scores = []
                for action_name, action_value in ped_label["action_label"].items():
                    frequencies[action_name] += 1
                    action_ids.append(ACTIONS_TO_IDX[action_name])#ACTIONS_TO_IDX.get(action_name, len(ACTIONS_TO_IDX)+1))
                    action_scores.append(action_value)

                ped_id = int(ped_label["label_id"].split(":")[1])
                action_labels[scene][frame][ped_id] = np.array(action_ids)
                action_label_scores[scene][frame][ped_id] = np.array(action_scores)

    print('frequencies:', frequencies)
    poses_2d = np.load("datasets/jrdb_adjusted/poses_2d.npz", allow_pickle=True)['arr_0'].item()
    import ipdb; ipdb.set_trace()

    # save one big file
    # np.savez(f"datasets/jrdb_adjusted/poses_2d_action_labels.npz",
    #          {**poses_2d, 'action_labels': action_labels, 'action_scores': action_label_scores})

    # save one file per scene
    os.makedirs(f"datasets/jrdb_adjusted/poses_2d_action_labels", exist_ok=True)
    for scene in NO_EGOMOTION:
        np.savez(f"datasets/jrdb_adjusted/poses_2d_action_labels/{scene}.npz",
                 {**{k:v[scene] for k,v in poses_2d.items()},
                  'action_labels': action_labels[scene], 'action_scores': action_label_scores[scene]})


if __name__=="__main__":
    parser = argparse.ArgumentParser(
            prog = "Jackrabbot Pose Classification Preprocessor",
            description = """Preprocessor for jrdb poses in action
                             classification.""")
    parser.add_argument("-i", "--data_root", default="datasets/jrdb")
    parser.add_argument("-o", "--data_out", default="datasets/jrdb_adjusted")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
