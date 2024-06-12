"""Preprocesses the raw train split of JRDB.
this file is decommissioned bc of a bug in agents_to_odometry_frame, where the robot was not be transformed to world frame
(and thus the agents where not being transformed to world frame either)
"""

import collections
import json
import os
import argparse

import numpy as np
import pandas as pd

from preprocess_utils import pose3
from preprocess_utils import quaternion
from preprocess_utils import rotation3
import tqdm

from visualize_preprocessed_data import plot_pedestrian_trajectories


def get_agents_dict(input_path, scene):
  """Returns agents GT data from raw data."""
  scene_data_file = get_file_handle(
      os.path.join(input_path, 'labels', 'labels_3d', scene + '.json')
  )
  scene_data = json.load(scene_data_file)

  agents = collections.defaultdict(list)

  for frame in scene_data['labels']:
    ts = int(frame.split('.')[0])
    for det in scene_data['labels'][frame]:
      agents[det['label_id']].append((ts, det))
  return agents


def get_agents_dict_from_detections(input_path, scene):
  """Returns agents data from fused detections raw data."""
  scene_data_file = get_file_handle(
      os.path.join(
          input_path, 'labels', 'labels_detections_3d', scene + '.json'
      )
  )
  scene_data = json.load(scene_data_file)

  agents = collections.defaultdict(list)

  for frame in scene_data['labels']:
    ts = int(frame.split('.')[0])
    for det in scene_data['labels'][frame]:
      agents[det['label_id']].append((ts, det))
  return agents


def get_file_handle(path, mode='rt'):
  file_handle = open(path, mode)
  return file_handle


def get_robot_kiss_icp(input_path, scene):
  """Returns robot features from raw data."""
  odom_data_file = get_file_handle(
      os.path.join(input_path, scene + '_3d.csv'))
  df = pd.read_csv(odom_data_file, sep=' ')

  robot = collections.defaultdict(list)

  for row in df.values:
    ts = row[0]
    robot[ts] = {
        'p': np.array([row[1], row[2], row[3]]),
        'q': np.array([row[4], row[5], row[6], row[7]])
    }

  return robot

def get_robot(input_path, scene):
  """Returns robot features from raw data."""
  odom_data_file = get_file_handle(
      os.path.join(input_path, scene + '.json'))
  odom_data = json.load(odom_data_file)

  robot = collections.defaultdict(list)

  for pc_ts, pose in odom_data['odometry'].items():
    ts = int(pc_ts.split('.')[0])
    robot[ts] = {
        'p': np.array([pose['position']['x'],
                       pose['position']['y'],
                       pose['position']['z']]),
        'q': np.array([pose['orientation']['x'],
                       pose['orientation']['y'],
                       pose['orientation']['z'],
                       pose['orientation']['w']]),
    }

  return robot
def robot_to_odometry_frame(robot_df):
  """Transforms robot features into odometry frame."""
  # initial world pose
  world_pose_odometry = pose3.Pose3(
      rotation3.Rotation3(
          quaternion.Quaternion(robot_df.loc[0]['q'])), robot_df.loc[0]['p'])
  # initial odometry pose in world frame
  odometry_pose_world = world_pose_odometry.inverse()

  # translate robot such that initial position is at origin
  robot_dict = {}
  for ts, row in robot_df.iterrows():
    world_pose_robot = pose3.Pose3(
        rotation3.Rotation3(quaternion.Quaternion(row['q'])), row['p'])
    odometry_pose_robot = odometry_pose_world * world_pose_robot

    robot_dict[ts] = {
        'p': odometry_pose_robot.translation,
        'yaw': odometry_pose_robot.rotation.euler_angles(radians=True)[-1]
        }
  return pd.DataFrame.from_dict(
      robot_dict, orient='index').rename_axis(['timestep'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads


def agents_to_odometry_frame(agents_df, robot_df):
  """Transforms agents features into odometry frame."""
  world_pose_odometry = pose3.Pose3(
      rotation3.Rotation3(
          quaternion.Quaternion(robot_df.loc[0]['q'])), robot_df.loc[0]['p'])
  odometry_pose_world = world_pose_odometry.inverse()

  agents_dict = {}
  for index, row in agents_df.iterrows():
    ts = index[0]
    robot_odometry_dp = robot_df.loc[ts]

    world_pose_robot = pose3.Pose3(
        rotation3.Rotation3(
            quaternion.Quaternion(robot_odometry_dp['q'])),
        robot_odometry_dp['p'])

    robot_pose_agent = pose3.Pose3(
        rotation3.Rotation3.from_euler_angles(
            rpy_radians=[0., 0., row['yaw']]), row['p'])

    odometry_pose_agent = (odometry_pose_world * world_pose_robot
                           * robot_pose_agent)

    agents_dict[index] = {
        'p': odometry_pose_agent.translation,
        'yaw': odometry_pose_agent.rotation.euler_angles(radians=True)[-1]}

    if 'l' in row:
      agents_dict[index]['l'] = row['l']
      agents_dict[index]['w'] = row['w']
      agents_dict[index]['h'] = row['h']

    if 'keypoints' in row:
      world_rot_robot = rotation3.Rotation3(
          quaternion.Quaternion(robot_odometry_dp['q']))
      odometry_rot_robot = odometry_pose_world.rotation * world_rot_robot
      rot_keypoints = []
      for keypoint in row['keypoints']:
        if np.isnan(keypoint).any():
          rot_keypoints.append(keypoint)
        else:
          rot_keypoints.append(odometry_rot_robot.rotate_point(keypoint))
      rot_keypoints = np.array(rot_keypoints)
      agents_dict[index]['keypoints'] = rot_keypoints

  return pd.DataFrame.from_dict(
      agents_dict, orient='index').rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads


def get_agents_features_with_box(agents_dict, max_distance_to_robot=10):
  """Returns agents features with bounding box from raw data dict."""
  agents_pos_dict = collections.defaultdict(dict)
  for agent_id, agent_data in agents_dict.items():
    for (ts, agent_instance) in agent_data:
      if agent_instance['attributes']['distance'] <= max_distance_to_robot:
        agents_pos_dict[(ts, agent_id)] = {
            'p': np.array([agent_instance['box']['cx'],
                           agent_instance['box']['cy'],
                           agent_instance['box']['cz']]),
            # rotation angle is relative to negatiev x axis of robot
            'yaw': np.pi - agent_instance['box']['rot_z'],
            'l': agent_instance['box']['l'],
            'w': agent_instance['box']['w'],
            'h': agent_instance['box']['h']
        }
  return agents_pos_dict

def list_scenes(input_path):
  scenes = os.listdir(os.path.join(input_path, 'labels', 'labels_3d'))
  scenes.sort()
  return [scene[:-5] for scene in scenes]


def get_agents_keypoints(input_path, scene):
  """Returns agents keypoints from raw data."""
  scene_data_file = get_file_handle(
      os.path.join(input_path, scene + '.json'))
  scene_data = json.load(scene_data_file)

  agents_keypoints = collections.defaultdict(dict)

  for frame in scene_data['labels']:
    ts = int(frame.split('.')[0])
    for det in scene_data['labels'][frame]:
      agents_keypoints[(ts, det['label_id'])] = {
          'keypoints': np.array(det['keypoints']).reshape(33, 3)}
  return agents_keypoints


def jrdb_preprocess_train(args):
  """Preprocesses the raw train split of JRDB."""
  input_path, output_path = args.input_path, args.output_path
  FROM_DETECTIONS = True
  AGENT_KEYPOINTS = args.save_keypoints#True

  subsample = 1

  scenes = list_scenes(
      os.path.join(input_path, 'train')
  )
  for scene in tqdm.tqdm(scenes):
    if scene != 'clark-center-2019-02-28_1':#clark-center-intersection-2019-02-28_0':  # clark-center-2019-02-28_1':
    # if scene != 'gates-to-clark-2019-02-28_1':#clark-center-intersection-2019-02-28_0':
        continue

    if not FROM_DETECTIONS:
      agents_dict = get_agents_dict(
          os.path.join(input_path, 'train'), scene
      )
    else:
      agents_dict = get_agents_dict_from_detections(
          os.path.join(input_path, 'processed'), scene
      )


    agents_features = get_agents_features_with_box(
        agents_dict, max_distance_to_robot=args.max_distance_to_robot
    )

    if args.old_odometry:
      print('using old odometry')
      robot_odom = get_robot(
          os.path.join(input_path, 'processed', 'odometry', 'train'), scene
      )
    else:
      robot_odom = get_robot_kiss_icp(
          os.path.join('datasets/jrdb_egomotion_kiss-icp_3d'), scene
      )

    agents_df = pd.DataFrame.from_dict(
        agents_features, orient='index'
    ).rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads

    if AGENT_KEYPOINTS:
      if args.use_hmr_kp:
        keypoints = np.load(f'../AgentFormerSDD/datasets/jrdb_hmr2_raw/{scene}.npz', allow_pickle=True)['arr_0'].item()
      else:
        keypoints = get_agents_keypoints(
            os.path.join(
                input_path, 'processed', 'labels',
                'labels_3d_keypoints', 'train'),
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

    # plot_pedestrian_trajectories(robot_df, scene, 'robot')
    # plot_pedestrian_trajectories(agents_df.rename_axis(['timestep', 'id']), scene, 'before_odo', end=1000, skip=10, plot_robot=True)
    if args.adjust_w_odometry:
      agents_in_odometry_df0 = agents_to_odometry_frame(
          agents_df, robot_df.iloc[::subsample].reset_index(drop=True)
      )
      filename = f'after_odo_{"og_odo" if args.old_odometry else "kiss"}'
      plot_pedestrian_trajectories(agents_in_odometry_df0, scene, filename, end=1000, skip=10, robot_df=robot_df)
      print(f"{robot_df=}")
    else:
      agents_in_odometry_df0 = agents_df.rename_axis(['timestep', 'id'])

    import ipdb; ipdb.set_trace()

    os.makedirs(output_path, exist_ok=True)
    ## change df to out format

    agents_in_odometry_df = agents_in_odometry_df0.reset_index()
    agents_in_odometry_df['id2'] = agents_in_odometry_df['id'].apply(lambda x: int(x.split(":")[-1]))
    agents_in_odometry_df['x'] = agents_in_odometry_df['p'].apply(lambda x: round(x[0],6))
    agents_in_odometry_df['y'] = agents_in_odometry_df['p'].apply(lambda x: round(x[1],6))

    # save agents_in_odometry_df to txt
    if args.save_trajectories:
      agents_in_odometry_df[['timestep', 'id2', 'x', 'y', 'yaw']].to_csv(f'{output_path}/{scene}.txt', sep=' ', header=False, index=False)

    if args.save_robot:
      robot_df = robot_to_odometry_frame(robot_df)
      robot_df['x'] = robot_df['p'].apply(lambda x: round(x[0],6))
      robot_df['y'] = robot_df['p'].apply(lambda x: round(x[1],6))
      if not os.path.exists(f'{output_path}/robot_poses'):
        os.makedirs(f'{output_path}/robot_poses')
      robot_df[['x', 'y', 'yaw']].to_csv(f'{output_path}/robot_poses/{scene}_robot.txt', sep=' ', header=False, index=False)

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
    parser = argparse.ArgumentParser(description='JRDB2022 Dataset Preprocessing')
    parser.add_argument('--input_path', '-ip', default='datasets/jrdb', help='Path to jrdb2022 dataset.')
    parser.add_argument('--output_path', '-op', default='datasets/jrdb_adjusted/odometry_adjusted', help='Path to output folder.')  #  datasets/jrdb_adjusted_kiss-icp_lower_velodyne
    parser.add_argument('--no_odometry','-no', dest='adjust_w_odometry', action='store_false')
    parser.add_argument('--save_keypoints', '-sk', action='store_true', default=False,
                        help='Whether to save keypoints.')
    parser.add_argument('--save_trajectories', '-st', action='store_true', default=False,)
    parser.add_argument('--process_pointclouds', '-pp', action='store_true', default=False,
                        help='Whether to process pointclouds.')
    parser.add_argument('--max_distance_to_robot', '-mdr', type=float, default=15.,
                        help='Maximum distance of agent to the robot to be included in the processed dataset.')
    parser.add_argument('--max_pc_distance_to_robot', type=float, default=10.,
                        help='Maximum distance of pointcloud point to the robot to be included in the processed dataset.')
    parser.add_argument('--save_robot', '-sr', action='store_true', default=False,
                        help='Whether to save robot poses.')
    parser.add_argument('--old_odometry', '-oo', action='store_true', default=False,)
    parser.add_argument('--use_hmr_kp', '-hmr', action='store_true', default=False,
                        help='Whether to use HMR keypoints instead of Google BlazePose keypoints.')


    args = parser.parse_args()

    # Using the parsed arguments
    # seq_name = 'gates-to-clark-2019-02-28_1'
    # data_root = 'datasets/jrdb_adjusted'
    # seq_name = 'gates-to-clark-2019-02-28_1'
    #
    # path = f'{data_root}/poses_2d_action_labels/{seq_name}.npz'
    # path = f'{data_root}/poses_2d_action_labels/{seq_name}.npz'
    # data = np.load(path, allow_pickle=True)['arr_0'].item()
    # all_kp_data = data['poses']
    # all_score_data = data['scores']
    # cam_ids = data['cam_ids']
    # cam_extrinsics = data['extrinsics']
    # cam_intrinsics = data['intrinsics']
    # action_labels = data['action_labels']
    # action_scores = data['action_scores']

    jrdb_preprocess_train(args)


if __name__ == '__main__':
    main()
