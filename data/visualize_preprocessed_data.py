import numpy as np
import pandas as pd
from data.jrdb_split import get_jrdb_split_egomotion
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import os
from pyquaternion import Quaternion


def convert_quaternion(xyzw):
  """Convert quaternion from xyzw to wxyz."""
  return Quaternion(xyzw[3], xyzw[0], xyzw[1], xyzw[2])

def convert_back_quaternion(wxyz_quat):
  """Convert quaternion from wxyz to xyzw."""
  w, x, y, z = wxyz_quat.elements
  return np.array([x, y, z, w])


TRAIN, TEST, _ = get_jrdb_split_egomotion()

def plot_pedestrian_trajectories(df, scene_name, name='trajectory_video', end=-1, skip=1, robot_df=None, plot_robot=False):
    # Create the directory if it doesn't exist
    directory = f"../viz/af_jrdb_data_preprocess/{scene_name}"
    os.makedirs(directory, exist_ok=True)

    # Prepare video writer
    video_name = f"{directory}/{name}.mp4"
    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_rate = 15 / skip
    video_writer = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)

    # Unique timesteps
    timesteps = sorted(df.index.get_level_values('timestep').unique())
    x_min, x_max = df['p'].apply(lambda x: x[0]).min(), df['p'].apply(lambda x: x[0]).max()
    y_min, y_max = df['p'].apply(lambda x: x[1]).min(), df['p'].apply(lambda x: x[1]).max()

    if robot_df is not None:
        x_min = min(x_min, robot_df['p'].apply(lambda x: x[0]).min())
        x_max = max(x_max, robot_df['p'].apply(lambda x: x[0]).max())
        y_min = min(y_min, robot_df['p'].apply(lambda x: x[1]).min())
        y_max = max(y_max, robot_df['p'].apply(lambda x: x[1]).max())

    for timestep in timesteps[:end:skip]:
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        ax.set_xlim([x_min, x_max])  # Set these limits to fit your data
        ax.set_ylim([y_min, y_max])

        if plot_robot:
            ax.scatter(0,0, s=100, color='red')
            # annotate robot position
            ax.annotate('robot', (0, 0), textcoords="offset points", xytext=(0,10), ha='center')

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
            ax.annotate('', xy=end_point, xytext=position, arrowprops=dict(facecolor='black', shrink=0.05))

        # Plot the robot if robot_df is provided
        if robot_df is not None:
            robot_position = robot_df.iloc[int(timestep)]['p'][:2]
            ax.scatter(*robot_position, s=100, color='red')  # Robot position in red
            # Plot the robot path
            robot_path = np.array(robot_df[robot_df.index <= timestep]['p'].apply(lambda x: x[:2]).tolist())
            ax.plot(robot_path[:, 0], robot_path[:, 1], color='red')
            # plot current arrow showing current orientation
            robot_quat = robot_df.iloc[int(timestep)]['q']
            assert robot_quat.shape[0] == 4
            robot_yaw = convert_quaternion(robot_quat).yaw_pitch_roll[0]
            end_point = (robot_position[0] + np.cos(robot_yaw) * 0.5, robot_position[1] + np.sin(robot_yaw) * 0.5)
            ax.annotate('', xy=end_point, xytext=robot_position, arrowprops=dict(facecolor='red', shrink=0.05))

        ax.set_aspect('equal')

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


if __name__ == "__main__":

    for scene in TEST + TRAIN:
        for kiss_or_og in ['jrdb_og_odo_mdr-1000_ss3d_mot', 'jrdb_adjusted_kiss-icp_lower_velodyne_mdr-1000_ss3d_mot']:
            trajectories_path = f'datasets/{kiss_or_og}/{scene}.txt'
            data = np.genfromtxt(trajectories_path, delimiter=' ', dtype=float)
            df = pd.DataFrame(data, columns=['timestep', 'agent_id', 'p_x', 'p_y', 'yaw'])
            # combine p_x, p_y into a single column
            df['p'] = df[['p_x', 'p_y']].apply(lambda s: s.to_numpy(), axis=1)
            # remove p_x, p_y, p_z columns
            df = df.drop(columns=['p_x', 'p_y'])
            robot_path = f'datasets/{kiss_or_og}/robot_poses/{scene}_robot.txt'
            robot_data = np.genfromtxt(robot_path, delimiter=' ', dtype=float)
            robot_df = pd.DataFrame(robot_data, columns=['p_x', 'p_y', 'yaw'])
            robot_df['p'] = robot_df[['p_x', 'p_y']].apply(lambda s: s.to_numpy(), axis=1)
            robot_df['q'] = robot_df['yaw'].apply(lambda x: np.array([x]))
            robot_df = robot_df.drop(columns=['p_x', 'p_y'])
            df = df.set_index(['timestep', 'agent_id'])
            robot_df.index.name = 'timestep'

            plot_pedestrian_trajectories(df, scene, f'{kiss_or_og}_loaded_from_preprocessed',
                                         end=1000, skip=10, robot_df=robot_df)

