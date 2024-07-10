"""Preprocesses the raw test split of JRDB.
this file is decommissioned bc of a bug in agents_to_odometry_frame, where the robot was not be transformed to world frame
(and thus the agents where not being transformed to world frame either)
"""

import tqdm

from data.preprocess_w_odometry import *
from pyquaternion import Quaternion

AGENT_KEYPOINTS = True
FROM_DETECTIONS = True
_TRACKING_CONFIDENCE_THRESHOLD = 0.00#1


def list_test_scenes(input_path):
    scenes = os.listdir(os.path.join(input_path, 'images', 'image_0'))
    scenes.sort()
    return scenes


def get_agents_features_df_with_box(
        input_path, scene_id, max_distance_to_robot=10.0, tracking_method=None,
):
    """Returns agents features with bounding box from raw leaderboard data."""
    jrdb_header = [
            'frame',
            'track id',
            'type',
            'truncated',
            'occluded',
            'alpha',
            'bb_left',
            'bb_top',
            'bb_width',
            'bb_height',
            'x',
            'y',
            'z',
            'height',
            'width',
            'length',
            'rotation_y',
            'score',
    ]
    scene_data_file = get_file_handle(
            os.path.join(
                    input_path, 'labels', tracking_method,
                    f'{scene_id:04}' + '.txt'
            )
    )
    df = pd.read_csv(scene_data_file, sep=' ', names=jrdb_header)

    def camera_to_lower_velodyne(p):
        return np.stack([p[..., 2], -p[..., 0], -p[..., 1]+.8], axis=-1)
        # return np.stack(
        #         [p[..., 2], -p[..., 0], -p[..., 1] + (0.742092 - 0.606982)], axis=-1
        # )

    df = df[df['score'] >= _TRACKING_CONFIDENCE_THRESHOLD]

    df['p'] = df[['x', 'y', 'z']].apply(
            lambda s: camera_to_lower_velodyne(s.to_numpy()), axis=1
    )
    df['distance'] = df['p'].apply(lambda s: np.linalg.norm(s, axis=-1))
    # df['l'] = df['height']
    # df['h'] = df['width']
    # df['w'] = df['length']
    df['l'] = df['length']  # height']
    df['h'] = df['height']  # width']
    df['w'] = df['width']  # length']
    df['yaw'] = df['rotation_y']

    df['id'] = df['track id'].apply(lambda s: f'pedestrian:{s}')
    df['timestep'] = df['frame']

    df = df.set_index(['timestep', 'id'])

    df = df[df['distance'] <= max_distance_to_robot]

    return df[['p', 'yaw', 'l', 'h', 'w']]


def df_to_boxes(df):
    """Returns 3d detection boxes from dataframe."""
    agents_pos_dict = {}
    for (ts, agent_id_), row in df.iterrows():
        if f"{ts:06}.pcb" not in agents_pos_dict:
            agents_pos_dict[f"{ts:06}.pcb"] = []
        agents_pos_dict[f"{ts:06}.pcb"].append(
            {
                    'label_id': agent_id_,
                    'box': {'cx': row['p'][0], 'cy': row['p'][1], 'cz': row['p'][2],
                            'l': row['l'], 'w': row['w'], 'h': row['h'], 'rot_z': row['yaw']},
                    'attributes': {}
            })
    return agents_pos_dict


def jrdb_preprocess_test(args):
    input_path, output_path = args.input_path, args.output_path
    """Preprocesses the raw test split of JRDB."""
    AGENT_KEYPOINTS = args.save_keypoints

    scenes = list_test_scenes(os.path.join(input_path, 'test'))
    subsample = 1
    for scene in tqdm.tqdm(scenes):
        scene_save_name = scene + '_test'
        agents_df = get_agents_features_df_with_box(os.path.join(input_path, 'test'),
                                                    scenes.index(scene),
                                                    max_distance_to_robot=args.max_distance_to_robot,
                                                    tracking_method=args.tracking_method,
        )

        if args.old_odometry:
            print('using old odometry')
            robot_odom = get_robot(
                    os.path.join(input_path, 'processed', 'odometry', 'test'), scene
            )
        else:
            robot_odom = get_robot_kiss_icp(
                    os.path.join('datasets/jrdb_egomotion_kiss-icp_3d'), scene
            )

        if AGENT_KEYPOINTS:
            keypoints = get_agents_keypoints(
                    os.path.join(
                            input_path, 'processed', 'labels',
                            'labels_3d_keypoints', 'test', args.tracking_method
                    ),
                    scene,
            )
            keypoints_df = pd.DataFrame.from_dict(
                    keypoints, orient='index'
            ).rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads

            agents_df = agents_df.join(keypoints_df)
            agents_df.keypoints.fillna(
                    dict(
                            zip(
                                    agents_df.index[agents_df['keypoints'].isnull()],
                                    [np.ones((33, 3)) * np.nan]
                                    * len(
                                            agents_df.loc[
                                                agents_df['keypoints'].isnull(), 'keypoints'
                                            ]
                                    ),
                                    )
                    ),
                    inplace=True,
            )

        robot_df = pd.DataFrame.from_dict(robot_odom, orient='index').rename_axis(  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads
                ['timestep']
        )
        # Remove extra data odometry datapoints
        robot_df = robot_df.iloc[agents_df.index.levels[0]]

        assert (agents_df.index.levels[0] == robot_df.index).all()

        # Subsample
        assert len(agents_df.index.levels[0]) == agents_df.index.levels[0].max() + 1
        agents_df_subsampled_index = agents_df.unstack('id').iloc[::subsample].index
        agents_df = (
                agents_df.unstack('id')
                .iloc[::subsample]
                .reset_index(drop=True)
                .stack('id', dropna=True)
        )
        # plot_pedestrian_trajectories(agents_df.rename_axis(['timestep', 'id']), scene, 'before_odo', end=1000, skip=10)

        if args.adjust_w_odometry:
            agents_in_odometry_df = agents_to_odometry_frame(
                agents_df, robot_df.iloc[::subsample].reset_index(drop=True)
            )
            # plot_pedestrian_trajectories(agents_in_odometry_df, scene, 'after_odo', end=1000, skip=10)
            # save agents_in_odometry_df to txt
        else:
            agents_in_odometry_df = agents_df.rename_axis(['timestep', 'id'])

        os.makedirs(output_path, exist_ok=True)
        ## change df to out format

        agents_in_odometry_df = agents_in_odometry_df.reset_index()
        agents_in_odometry_df['id2'] = agents_in_odometry_df['id'].apply(lambda x: int(x.split(":")[-1]))
        agents_in_odometry_df['x'] = agents_in_odometry_df['p'].apply(lambda x: round(x[0],6))
        agents_in_odometry_df['y'] = agents_in_odometry_df['p'].apply(lambda x: round(x[1],6))

        # save agents_in_odometry_df to txt
        if args.save_trajectories:
            agents_in_odometry_df[['timestep', 'id2', 'x', 'y', 'yaw']].to_csv(f'{output_path}/{scene}.txt', sep=' ',
                                                                               header=False, index=False)

        if args.save_robot:
            robot_df['x'] = robot_df['p'].apply(lambda x: round(x[0], 6))
            robot_df['y'] = robot_df['p'].apply(lambda x: round(x[1], 6))
            robot_df['yaw'] = robot_df['q'].apply(lambda x: Quaternion(x).yaw_pitch_roll[0])
            if not os.path.exists(f'{output_path}/robot_poses'):
                os.makedirs(f'{output_path}/robot_poses')
            robot_df[['x', 'y', 'yaw']].to_csv(f'{output_path}/robot_poses/{scene}_robot.txt', sep=' ', header=False,
                                               index=False)

        # Transforming the dataframe into the nested dictionary
        if args.save_keypoints:
            nested_dict = {}
            for index, row in agents_in_odometry_df.iterrows():
                timestep = row['timestep']
                agent_id = row['id2']
                keypoints = row['keypoints']

                if timestep not in nested_dict:
                    nested_dict[timestep] = {}
                nested_dict[timestep][agent_id] = keypoints

            if not os.path.exists(f'{output_path}/agent_keypoints'):
                os.makedirs(f'{output_path}/agent_keypoints')
            np.savez(f'{output_path}/agent_keypoints/{scene}_kp.npz', nested_dict)
            print(f"saved to {output_path=}")


def main():
    parser = argparse.ArgumentParser(description='Process JRDB2022 dataset with additional tracking and confidence options.')
    parser.add_argument('--input_path', '-ip', default='datasets/jrdb', help='Path to jrdb2022 dataset.')
    parser.add_argument('--output_path', '-op', default=None, help='Path to output folder.')
    parser.add_argument('--no_odometry', '-no', dest='adjust_w_odometry', action='store_false')
    parser.add_argument('--save_keypoints', '-sk', action='store_true', default=False,
                        help='Whether to save keypoints.')
    parser.add_argument('--save_trajectories', '-st', action='store_true', default=False, )
    parser.add_argument('--process_pointclouds', action='store_true', default=True,
                        help='Whether to process pointclouds.')
    parser.add_argument('--max_distance_to_robot', '-mdr', type=float, default=15.,
                        help='Maximum distance of agent to the robot to be included in the processed dataset.')
    parser.add_argument('--max_pc_distance_to_robot', type=float, default=10.,
                        help='Maximum distance of pointcloud point to the robot to be included in the processed dataset.')
    parser.add_argument('--tracking_confidence_threshold', type=float, default=0.0,
                        help='Confidence threshold for tracked agent instance to be included in the processed dataset.')
    parser.add_argument('--save_robot', '-sr', action='store_true', default=False,
                        help='Whether to save robot poses.')
    parser.add_argument('--old_odometry', '-oo', action='store_true', default=False,)
    parser.add_argument('--tracking_method', '-tm', default=None, required=True, choices=['ss3d_mot', 'PiFeNet'],)


    args = parser.parse_args()

    # Now use the arguments
    jrdb_preprocess_test(args)


if __name__ == '__main__':
    main()
