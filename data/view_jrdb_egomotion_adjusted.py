"""
view adjusted egomotion trajectories on 2d BEV jrdb data, optionally adding in egomotion data from any source
2024-04-11 updated to accomodate kiss-icp tum data
"""

import pandas as pd
import glob
import numpy as np
import os
import argparse
import pyquaternion as pyq
import multiprocessing as mp


# from jrdb_split import TRAIN, TEST, WITH_MOVEMENT, NO_MOVEMENT, WITH_MOVEMENT_ADJUSTED
from jrdb_viz_utils import visualize_BEV_trajs


TRAIN = [
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
# 28

TEST = [
    'cubberly-auditorium-2019-04-22_1',
    'discovery-walk-2019-02-28_0',
    'discovery-walk-2019-02-28_1',
    'food-trucks-2019-02-12_0',
    'gates-ai-lab-2019-04-17_0',
    'gates-basement-elevators-2019-01-17_0',
    'gates-foyer-2019-01-17_0',
    'gates-to-clark-2019-02-28_0',
    'hewlett-class-2019-01-23_0',
    'hewlett-class-2019-01-23_1',
    'huang-2-2019-01-25_1',
    'huang-intersection-2019-01-22_0',
    'indoor-coupa-cafe-2019-02-06_0',
    'lomita-serra-intersection-2019-01-30_0',
    'meyer-green-2019-03-16_1',
    'nvidia-aud-2019-01-25_0',
    'nvidia-aud-2019-04-18_1',
    'nvidia-aud-2019-04-18_2',
    'outdoor-coupa-cafe-2019-02-06_0',
    'quarry-road-2019-02-28_0',
    'serra-street-2019-01-30_0',
    'stlc-111-2019-04-19_1',
    'stlc-111-2019-04-19_2',
    'tressider-2019-03-16_2',
    'tressider-2019-04-26_0',
    'tressider-2019-04-26_1',
    'tressider-2019-04-26_3'
]
# 27

def main(scene, args):
    # Example usage (Note: Replace these with your actual data)
    print(f"scene: {scene}")
    split = 'train' if scene in TRAIN else 'test'

    # load in ego-perspective camera rgb images for plotting
    images_0 = sorted(glob.glob(f"{args.images_dir}/{split}/images/image_0/{scene}/*"))
    images_2 = sorted(glob.glob(f"{args.images_dir}/{split}/images/image_2/{scene}/*"))
    images_4 = sorted(glob.glob(f"{args.images_dir}/{split}/images/image_4/{scene}/*"))
    images_6 = sorted(glob.glob(f"{args.images_dir}/{split}/images/image_6/{scene}/*"))
    images_8 = sorted(glob.glob(f"{args.images_dir}/{split}/images/image_8/{scene}/*"))
    assert len(images_0) == len(images_2) == len(images_4) == len(images_6) == len(images_8), \
        (f"len(images_0): {len(images_0)}, len(images_2): {len(images_2)}, len(images_4): {len(images_4)}, "
         f"len(images_6): {len(images_6)}, len(images_8): {len(images_8)}")

    # load BEV 2d trajs
    bev_traj_dir = args.input_traj_dir  # '/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw'
    bev_traj_path = f'{bev_traj_dir}/{scene}.csv'

    df = pd.read_csv(bev_traj_path, sep=' ', header=None)  # , usecols=[0, 1, 10, 11])
    df.columns = ['frame', 'id', 'x', 'y', 'heading']

    assert len(df['frame'].unique()) == len(images_0), \
        f"len(df['frame'].unique()): {len(df['frame'].unique())}, len(image_list): {len(images_0)}"

    df_ego = df.copy()

    ego_positions = np.zeros((len(images_0), 2))
    ego_rotations = np.zeros(len(images_0))

    if args.egomotion_dir is not None:
        egomotion = np.load(os.path.join(args.egomotion_dir, f'{scene}.npy'))
        ego_positions = egomotion[:, :2]
        # negate y
        # ego_positions[:, 1] = ego_positions[:, 1]
        ego_rotations = egomotion[:, 2]
        delta_x, delta_y = ego_positions.T
        # egomotion = pd.read_csv(os.path.join(args.egomotion_dir, f'{scene}.csv'))

        # Apply rotation to existing trajectory data
        frame_to_ego_rot_mat = {frame: np.array([
                [np.cos(ego_rotations[i]), -np.sin(ego_rotations[i])],
                [np.sin(ego_rotations[i]), np.cos(ego_rotations[i])]
        ]) for i, frame in enumerate(df['frame'].unique())}

        def apply_rotation(row):
            frame = row['frame']
            rotation_matrix = frame_to_ego_rot_mat[frame]
            pos = np.array([row['x'], row['y']])
            new_pos = pos.dot(rotation_matrix.T)
            return new_pos

        df[['x', 'y']] = np.array(df.apply(apply_rotation, axis=1).tolist())

        # Apply translation
        x_value_map = pd.Series(delta_x, index=df['frame'].unique())
        y_value_map = pd.Series(delta_y, index=df['frame'].unique())
        df['x'] = df['x'] + df['frame'].map(x_value_map)
        df['y'] = df['y'] + df['frame'].map(y_value_map)

        # apply rotation
        rot_map = pd.Series(ego_rotations, index=df['frame'].unique())
        df['heading'] = df['heading'] + df['frame'].map(rot_map)

    # add egomotion in as an additional pedestrian to the df
    if args.include_robot_as_ped:
        EGO_ID = 1000
        if EGO_ID not in df['id'].unique():
            ego_ped_df = pd.DataFrame(
                    {'frame': np.arange(len(images_0)), 'id': EGO_ID, 'x': ego_positions[:, 0],
                     'y': ego_positions[:, 1],
                     'heading': ego_rotations})
            df = pd.concat([df, ego_ped_df], ignore_index=True)
            df = df.sort_values(by=['frame', 'id']).reset_index(drop=True)

    # save new trajectories
    if args.save_traj:
        bev_traj_adjusted_path = f'{args.output_traj_dir}/{scene}.txt'

        if not os.path.exists(os.path.dirname(bev_traj_adjusted_path)):
            os.makedirs(os.path.dirname(bev_traj_adjusted_path))
        # print(f"df: {df}")
        # df = df[df['frame'] % skip == 0]
        # print(f"df: {df}")
        df.to_csv(bev_traj_adjusted_path, index=False, header=False, sep=' ')

    ####################
    ##### plotting #####
    ####################

    if args.length is None:
        args.length = len(images_0)

    viz_dir = f'{args.output_viz_dir}/{scene}.mp4'
    if args.visualize and not os.path.exists(viz_dir):
        visualize_BEV_trajs(df, df_ego, images_0, images_2, images_4, images_6, images_8,
                            scene, args)


def process(path, egomotion_save_dir):
    scene = path.split('_poses_tum')[0].split('/')[-1]

    if os.path.exists(f'{egomotion_save_dir}/{scene}.npy'):
        return

    df = pd.read_csv(path, sep=' ', header=None)
    df.columns = ['frame', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

    df.to_csv(f'{egomotion_save_dir}/{scene}_3d.csv', index=False, header=True, sep=' ')
    # np.save(f'{egomotion_save_dir}/{scene}_3d.npy', df.values)

    # # translate into x, y, yaw using pyquaterion
    # def get_yaw(row):
    #     q = pyq.Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
    #     return q.yaw_pitch_roll[0]
    #
    # df['heading'] = df.apply(get_yaw, axis=1)
    # df = df[['frame', 'x', 'y', 'heading']]
    #
    # # save x,y,heading to numpy
    # np.save(f'{egomotion_save_dir}/{scene}.npy', df[['x', 'y', 'heading']].values)


def preprocess_tum_egomotion(args):
    """ preprocesses data in tum format into x, y, yaw format """
    egomotion_save_dir, tum_dir = args.egomotion_dir, args.tum_dir

    if not os.path.exists(egomotion_save_dir):
        os.makedirs(egomotion_save_dir)

    all_paths = []
    for path in os.listdir(tum_dir):
        if path[:4] != '2024':
            continue
        paths = list(glob.glob(f'{tum_dir}/{path}/*_poses_tum.txt'))
        all_paths.extend([(p, egomotion_save_dir) for p in paths])

    if args.mp:
        with mp.Pool(mp.cpu_count()) as p:
            p.starmap(process, all_paths)
    else:
        for path in all_paths:
            process(*path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', '-mp', action='store_true')
    parser.add_argument('--save_traj', '-st', action='store_true')
    parser.add_argument('--dont_visualize', '-dv', dest='visualize', action='store_false')
    parser.add_argument('--input_images_dir', '-ii', dest='images_dir', type=str, default='../AgentFormerSDD/datasets/jrdb')
    parser.add_argument('--input_traj_dir', '-it', type=str, default='../AgentFormerSDD/datasets/jrdb_raw_all')
    parser.add_argument('--output_traj_dir', '-ot', type=str, default='../AgentFormerSDD/datasets/jrdb_adjusted')
    parser.add_argument('--output_viz_dir', '-ov', type=str, default=f'../viz/jrdb_egomotion_rosbag')
    parser.add_argument('--skip','-s', type=int, default=6)
    parser.add_argument('--tum_dir', '-td', default='results',
                        help='path to egomotion data in TUM format (x y z qx qy qz qw) for conversion to (x y yaw).'
                             'converts to (x y yaw) first before generating egomotion-adjusted visualization')
    parser.add_argument('--length', '-l', type=int, default=None)
    parser.add_argument('--egomotion_dir', '-ed', type=str, default=None,#"jrdb/rosbag_egomotion/")
                        help='path to egomotion data in numpy format (x y yaw) for generating egomotion-adjusted visualization')
    parser.add_argument('--include_robot_as_ped', '-ir', action='store_true')

    args = parser.parse_args()
    __spec__ = None
    mp.set_start_method('spawn')

    if args.tum_dir is not None:
        preprocess_tum_egomotion(args)
    import ipdb; ipdb.set_trace()

    if args.mp:
        list_of_args = []
        for scene in TRAIN + TEST:#WITH_MOVEMENT_ADJUSTED:
            list_of_args.append((scene, args))
        with mp.Pool(min(len(list_of_args), mp.cpu_count())) as p:
            p.starmap(main, list_of_args)

    else:
        main('clark-center-intersection-2019-02-28_0', args)
