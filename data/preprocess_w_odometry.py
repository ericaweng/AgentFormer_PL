"""Preprocesses the raw train split of JRDB. """

import collections
import json
import os
import argparse

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import tqdm

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_pedestrian_trajectories(df, scene_name, name='trajectory_video', end=-1, skip=1):
  # Create the directory if it doesn't exist
  directory = f"../viz/hst_jrdb_data/{scene_name}"
  os.makedirs(directory, exist_ok=True)

  # Prepare video writer
  video_name = f"{directory}/{name}.mp4"
  frame_size = (640, 480)
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  frame_rate = 15 / skip
  video_writer = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)

  # Unique timesteps
  timesteps = df.index.get_level_values('timestep').unique()
  x_min, x_max = df['p'].apply(lambda x: x[0]).min(), df['p'].apply(lambda x: x[0]).max()
  y_min, y_max = df['p'].apply(lambda x: x[1]).min(), df['p'].apply(lambda x: x[1]).max()

  for timestep in timesteps[:end:skip]:
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    ax.set_xlim([x_min, x_max])  # Set these limits to fit your data
    ax.set_ylim([y_min, y_max])

    # Filter data for the current timestep
    timestep_data = df.loc[timestep]

    # Plot each pedestrian
    for _, row in timestep_data.iterrows():
      position = row['p'][:2]  # Get x, y position
      yaw = row['yaw']

      # Plot pedestrian as a circle
      ax.scatter(*position, s=100)  # Adjust s for size

      # Calculate end point of the arrow based on yaw
      end_point = (position[0] + np.cos(yaw) * 0.5, position[1] + np.sin(yaw) * 0.5)
      ax.annotate('', xy=end_point, xytext=position,
                  arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_aspect('equal')
    # plt.axis('off')

    # Convert plot to image
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    # Convert RGBA to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # Resize for video
    image = cv2.resize(image, frame_size)

    video_writer.write(image)

    plt.close(fig)

  video_writer.release()
  print(f"Video saved as {video_name}")


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


def get_agents_features(agents_dict, max_distance_to_robot=10):
  """Returns agents features from raw data dict."""
  agents_pos_dict = collections.defaultdict(dict)
  for agent_id, agent_data in agents_dict.items():
    for ts, agent_instance in agent_data:
      if agent_instance['attributes']['distance'] <= max_distance_to_robot:
        agents_pos_dict[(ts, agent_id)] = {
            'p': np.array([
                agent_instance['box']['cx'],
                agent_instance['box']['cy'],
                agent_instance['box']['cz'],
            ]),
            # rotation angle is relative to negative x axis of robot
            'yaw': np.pi - agent_instance['box']['rot_z'],
        }
  return agents_pos_dict


def get_file_handle(path, mode='rt'):
  file_handle = open(path, mode)
  return file_handle

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



def agents_to_odometry_frame(agents_df, robot_df):
  """Transforms agents features into odometry frame using pyquaternion."""
  # Calculate the world pose odometry quaternion and position
  robot_initial_quat = Quaternion(robot_df.loc[0]['q'])
  robot_initial_pos = np.array(robot_df.loc[0]['p'])

  # The inverse transformation is obtained by inverting the quaternion and negating the position
  odometry_quat_world = robot_initial_quat.inverse
  odometry_pos_world = -odometry_quat_world.rotate(robot_initial_pos)

  agents_dict = {}
  for index, row in agents_df.iterrows():
    ts = index[0]
    robot_odometry_dp = robot_df.loc[ts]

    # Quaternion and position for the world pose of the robot
    world_quat_robot = Quaternion(robot_odometry_dp['q'])
    world_pos_robot = np.array(robot_odometry_dp['p'])

    # Quaternion and position for the robot pose relative to the agent
    agent_yaw_rotation = Quaternion(axis=[0, 0, 1], radians=row['yaw'])
    agent_pos = np.array(row['p'])

    # Calculate the odometry pose of the agent
    odometry_pos_agent = odometry_quat_world.rotate(
            world_quat_robot.rotate(agent_pos) + world_pos_robot
    ) + odometry_pos_world
    odometry_yaw_agent = (odometry_quat_world * world_quat_robot * agent_yaw_rotation).yaw_pitch_roll[0]

    # Store the transformed position and yaw
    agents_dict[index] = {
            'p': odometry_pos_agent,
            'yaw': odometry_yaw_agent
    }

    # Add length, width, height if available
    if 'l' in row:
      agents_dict[index]['l'] = row['l']
      agents_dict[index]['w'] = row['w']
      agents_dict[index]['h'] = row['h']

    # Rotate keypoints if available
    if 'keypoints' in row:
      rot_keypoints = []
      for keypoint in row['keypoints']:
        if np.isnan(keypoint).any():
          rot_keypoints.append(keypoint)
        else:
          rotated_keypoint = odometry_quat_world.rotate(
                  world_quat_robot.rotate(keypoint)
          )
          rot_keypoints.append(rotated_keypoint)
      rot_keypoints = np.array(rot_keypoints)
      agents_dict[index]['keypoints'] = rot_keypoints

  return pd.DataFrame.from_dict(agents_dict, orient='index').rename_axis(['timestep', 'id'])


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
  AGENT_KEYPOINTS = True

  subsample = 1

  scenes = list_scenes(
      os.path.join(input_path, 'train')
  )
  for scene in tqdm.tqdm(scenes):
    if not FROM_DETECTIONS:
      agents_dict = get_agents_dict(
          os.path.join(input_path, 'train'), scene
      )
    else:
      agents_dict = get_agents_dict_from_detections(
          os.path.join(input_path, 'processed'), scene
      )


    agents_features = get_agents_features_with_box(
        agents_dict, max_distance_to_robot=1000
    )

    robot_odom = get_robot(
        os.path.join(input_path, 'processed', 'odometry', 'train'), scene
    )

    agents_df = pd.DataFrame.from_dict(
        agents_features, orient='index'
    ).rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads

    if AGENT_KEYPOINTS:
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
    # plot_pedestrian_trajectories(agents_df.rename_axis(['timestep', 'id']), scene, 'before_odo', end=1000, skip=10)
    if args.adjust_w_odometry:
      agents_in_odometry_df0 = agents_to_odometry_frame(
          agents_df, robot_df.iloc[::subsample].reset_index(drop=True)
      )
      print(f"{agents_in_odometry_df0=}")
      # plot_pedestrian_trajectories(agents_in_odometry_df, scene, 'after_odo', end=1000, skip=10)
      # save agents_in_odometry_df to txt
    else:
      agents_in_odometry_df0 = agents_df.rename_axis(['timestep', 'id'])

    os.makedirs(output_path, exist_ok=True)
    ## change df to out format

    agents_in_odometry_df = agents_in_odometry_df0.reset_index()
    agents_in_odometry_df['id2'] = agents_in_odometry_df['id'].apply(lambda x: int(x.split(":")[-1]))
    agents_in_odometry_df['x'] = agents_in_odometry_df['p'].apply(lambda x: round(x[0],6))
    agents_in_odometry_df['y'] = agents_in_odometry_df['p'].apply(lambda x: round(x[1],6))

    if args.save_trajectories:
      agents_in_odometry_df[['timestep', 'id2', 'x', 'y', 'yaw']].to_csv(f'{output_path}/{scene}.txt', sep=' ', header=False, index=False)

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

      np.savez(f'{output_path}/{scene}_kp.npz', nested_dict)
      print(f"saved to {output_path=}")


def main():
    parser = argparse.ArgumentParser(description='JRDB2022 Dataset Preprocessing')
    parser.add_argument('--input_path', '-ip', default='datasets/jrdb', help='Path to jrdb2022 dataset.')
    parser.add_argument('--output_path', '-op', default='datasets/jrdb_adjusted/odometry_adjusted', help='Path to output folder.')
    parser.add_argument('--no_odometry','-no', dest='adjust_w_odometry', action='store_false')
    parser.add_argument('--save_keypoints', '-sk', action='store_true', default=False,
                        help='Whether to save keypoints.')
    parser.add_argument('--save_trajectories', '-st', action='store_true', default=False,)
    parser.add_argument('--process_pointclouds', '-pp', action='store_true', default=False,
                        help='Whether to process pointclouds.')
    parser.add_argument('--max_distance_to_robot', type=float, default=15.,
                        help='Maximum distance of agent to the robot to be included in the processed dataset.')
    parser.add_argument('--max_pc_distance_to_robot', type=float, default=10.,
                        help='Maximum distance of pointcloud point to the robot to be included in the processed dataset.')

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
