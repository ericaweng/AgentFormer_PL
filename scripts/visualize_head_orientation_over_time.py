""" plot static chart and animation of head orientation over time for each pedestrian in the dataset, as well as aggregate head orientation statistics for each location in jrdb dataset"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from traj_toolkit.visualisation.calculate_geometric_features_from_hmr_kp import calculate_geometric_features_from_hmr_kp
from traj_toolkit.visualisation.constants import MAX_DISTANCE_TO_ROBOT, TRAIN, TEST_LOCATION_TO_ID, TRACKING_METHOD, TEST, HMR_KEYPOINT_DIM, OPENPOSE44_CONNECTIONS, ALL_SESSIONS_TBD
from traj_toolkit.data_scripts.preprocess_w_odometry import get_agents_dict_from_detections, get_agents_features_with_box, get_agents_keypoints, get_robot, get_robot_kiss_icp, robot_to_odometry_frame
from data_scripts.load_utils import get_agents_keypoints_hmr, get_agents_df_from_txt
from traj_toolkit.data_scripts.preprocess_test_w_odometry import get_agents_features_df_with_box
from traj_toolkit.visualisation.viz_utils_univ import draw_pose_3d_single_frame

import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_head_orientation_over_time_plotly(agents_df, location, save_dir, downsample_interval=4, show_plot=False):
    # Plotting for each pedestrian id separately
    unique_ids = agents_df.index.get_level_values('id').unique()

    # Loop through each pedestrian id
    num_plot = 3  # plot 3 pedestrians per location
    num_skip = len(unique_ids) // num_plot
    for pedestrian_id in unique_ids[::num_skip]:
        # Filter the DataFrame for the current pedestrian id
        df_filtered = agents_df.xs(pedestrian_id, level='id')
        num_timesteps_to_plot = 19  # or another value as per your requirement

        # Downsample according to the downsample_interval argument
        timesteps_every_nth = df_filtered.index.get_level_values('timestep').min() + downsample_interval * np.arange(num_timesteps_to_plot)
        timesteps_every_nth = np.intersect1d(timesteps_every_nth, df_filtered.index.get_level_values('timestep').unique())
        df_filtered = df_filtered.loc[timesteps_every_nth]

        # ped-level statistics
        if np.all(np.isnan(np.stack(df_filtered['gaze']))):
            print(f"Location: {location}, pedestrian_id: {pedestrian_id}, gaze is NaN")
            continue

        # Calculate the angles (azimuth and elevation) for spherical coordinates
        head_orientation = np.stack(df_filtered['head_orientation'])

        # Spherical coordinates
        theta = np.arctan2(head_orientation[:, 1], head_orientation[:, 0])  # Azimuth angle
        radius = df_filtered.index.get_level_values('timestep')  # Using time as the radius

        # Cartesian coordinates for unit sphere
        phi = np.arccos(head_orientation[:, 2])  # Polar angle
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Create the unit sphere mesh (for visualization purposes)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.sin(v), np.cos(u)).flatten()
        y_sphere = np.outer(np.sin(v), np.sin(u)).flatten()
        z_sphere = np.outer(np.cos(v), np.ones_like(u)).flatten()

        sphere_mesh = go.Mesh3d(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            opacity=0.1,  # Increased transparency for the unit sphere
            color='lightblue',
            name='Unit Sphere'
        )

        # Create a subplot with three columns: one polar plot and two 3D plots with different views
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'polar'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=[f"Polar Plot for Pedestrian {pedestrian_id}",
                            f"3D Orientation (View 1) for Pedestrian {pedestrian_id}",
                            f"3D Orientation (View 2) for Pedestrian {pedestrian_id}"]
        )

        # First plot (polar plot)
        polar_trace = go.Scatterpolar(
            r=radius,
            theta=np.degrees(theta),  # Convert radians to degrees for polar plot
            mode='lines',
            name=f'Pedestrian {pedestrian_id}',
            line=dict(color='blue')
        )

        fig.add_trace(polar_trace, row=1, col=1)
        fig.update_polars(radialaxis=dict(range=[radius.min(), radius.max()]),
                          angularaxis=dict(direction='counterclockwise'))

        # Second plot (3D unit sphere plot, View 1)
        sphere_trace_1 = go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'Pedestrian {pedestrian_id} - 3D View 1',
            line=dict(color='blue', width=3),
            marker=dict(size=4)
        )

        fig.add_trace(sphere_trace_1, row=1, col=2)
        fig.add_trace(sphere_mesh, row=1, col=2)

        # Third plot (3D unit sphere plot, View 2)
        sphere_trace_2 = go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'Pedestrian {pedestrian_id} - 3D View 2',
            line=dict(color='blue', width=3),
            marker=dict(size=4)
        )

        fig.add_trace(sphere_trace_2, row=1, col=3)
        fig.add_trace(sphere_mesh, row=1, col=3)

        # Update layout to customize the figure size and add different view angles
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            scene2=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))  # Set camera for the second 3D view
            ),
            scene3=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(eye=dict(x=0, y=0, z=0))  # Set camera for the third 3D view
            ),
            title=f'Head orientation over time for pedestrian {pedestrian_id} (Location: {location})',
            showlegend=False,
            height=800,  # Increased height
            width=2000,   # Increased width for wider display
            margin=dict(l=50, r=50, b=50, t=50) # add some border
        )

        # Save the plot
        if show_plot:
            fig.show()
        else:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_image(f'{save_dir}/{location}_ped-{pedestrian_id}_head_orientation_polar_3d.png')
            print(f'Saved plot for pedestrian {pedestrian_id}, location {location} to {save_dir}/{location}_ped-{pedestrian_id}_head_orientation_polar_3d.png')


def get_avg_std_dev_orientation(head_orientations):
    mean_sin = np.nanmean(np.sin(head_orientations))
    mean_cos = np.nanmean(np.cos(head_orientations))
    mean_angle = np.arctan2(mean_sin, mean_cos)
    # Calculate the resultant length R
    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
    # Calculate the circular standard deviation
    circular_std_dev = np.sqrt(-2 * np.log(R))
    return mean_angle, circular_std_dev, R


def load_robot_df(location, use_old_robot_odo=False):
    root_location = os.path.expanduser("~/code/AgentFormerSDD/datasets/jrdb")  # path to JRDB
    if use_old_robot_odo:
        robot_path = f"{root_location}/processed/odometry/{'train' if location in TRAIN else 'test'}"
    else:
        robot_path = "datasets/jrdb_egomotion_kiss-icp_3d"

    if use_old_robot_odo:
        robot_odom = get_robot(robot_path, location)
    else:
        robot_odom = get_robot_kiss_icp(robot_path, location)
    robot_df = pd.DataFrame.from_dict(robot_odom, orient='index').rename_axis(['timestep'])
    robot_df = robot_to_odometry_frame(robot_df)
    return robot_df


def load_agents_df_tbd(location):
    root_location = os.path.expanduser("~/code/AgentFormerSDD/datasets/tbd/")  # path to TBD
    tracks_location = f"{root_location}/2/Pedestrian_labels/3d_traj"
    label_3d_pose_path = f"{root_location}/tbd_hmr2_raw/{location}_kp3d.npz"
    try:
        agents_df = get_agents_df_from_txt(tracks_location, location, keypoints_path=label_3d_pose_path, load_geometric_features=True, is_jrdb=False)
    except FileNotFoundError as e:
        label_3d_pose_path = f"{root_location}/tbd_hmr2_raw/{location}_kp_3d.npz"
        agents_df = get_agents_df_from_txt(tracks_location, location, keypoints_path=label_3d_pose_path, load_geometric_features=True, is_jrdb=False)

    return agents_df


def load_agents_df_jrdb(location, pose_type):
    root_location = os.path.expanduser("~/code/AgentFormerSDD/datasets/jrdb")  # path to JRDB

    if location in TRAIN:
        agents_path = root_location + "/processed"
    else:
        assert location in TEST
        agents_path = root_location + "/test"

    if location in TRAIN:
        agents_dict = get_agents_dict_from_detections(agents_path, location)
        agents_features = get_agents_features_with_box(agents_dict,
                                                        max_distance_to_robot=MAX_DISTANCE_TO_ROBOT)
        agents_df = pd.DataFrame.from_dict(agents_features, orient='index').rename_axis(['timestep', 'id'])
    else:
        agents_df = get_agents_features_df_with_box(agents_path, TEST_LOCATION_TO_ID[location],
                                                    max_distance_to_robot=MAX_DISTANCE_TO_ROBOT,
                                                    tracking_method=TRACKING_METHOD)

    if pose_type.lower() == "hmr":
        label_3d_pose_path = f"{root_location}/jrdb_hmr2_raw_stitched/{location}_kp_3d.npz"
    else:
        assert pose_type.lower() == "blazepose"
        if location in TRAIN:
            label_3d_pose_path = f"{root_location}/processed/labels/labels_3d_keypoints/train/{location}.json"
        else:
            assert location in TEST
            label_3d_pose_path = f"{root_location}/processed/labels/labels_3d_keypoints/test/{TRACKING_METHOD}/{location}.json"

    if pose_type.lower() == 'hmr':
        keypoints = get_agents_keypoints_hmr(label_3d_pose_path)
    else:
        assert pose_type.lower() == 'blazepose'
        keypoints = get_agents_keypoints(Path(label_3d_pose_path).parent, location)

    keypoints_df = pd.DataFrame.from_dict(keypoints, orient='index').rename_axis(['timestep', 'id'])
    agents_df = agents_df.join(keypoints_df)
    agents_df.keypoints = agents_df.keypoints.fillna(dict(
            zip(agents_df.index[agents_df['keypoints'].isnull()],
                [np.ones((HMR_KEYPOINT_DIM, 3)) * np.nan] * len(agents_df.loc[agents_df['keypoints'].isnull(), 'keypoints']),
    )))
    agents_df = calculate_geometric_features_from_hmr_kp(agents_df.copy())
    return agents_df


def anim_head_orientation_over_time(agents_df, location, save_dir):
    # Plotting for each pedestrian id separately
    unique_ids = agents_df.index.get_level_values('id').unique()

    # Loop through each pedestrian id
    num_plot = 3  # plot 3 pedestrians per location
    num_skip = len(unique_ids) // num_plot
    for pedestrian_id in unique_ids[::num_skip]:
        df_filtered = agents_df.xs(pedestrian_id, level='id')
        num_timesteps_to_plot = 19  # Adjust as necessary

        timesteps_every_5th = df_filtered.index.get_level_values('timestep').min() + 5 * np.arange(num_timesteps_to_plot)
        timesteps_every_5th = np.intersect1d(timesteps_every_5th, df_filtered.index.get_level_values('timestep').unique())
        assert np.all(np.isin(timesteps_every_5th, df_filtered.index.get_level_values('timestep').unique())), "Mismatch in timesteps"
        df_filtered = df_filtered.loc[timesteps_every_5th]

        if np.all(np.isnan(np.stack(df_filtered['gaze']))):
            continue
        mean_angle, circular_std_dev, resultant = get_avg_std_dev_orientation(np.stack(df_filtered['gaze']))

        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f'Head orientation and keypoints over time in location {location}')
        # Add mean_angle text to bottom of the fig
        fig.text(0.5, 0.05, f'Mean angle: {mean_angle:.2f}, Circular std dev: {circular_std_dev:.2f}, Resultant: {resultant:.2f}', ha='center')

        # Create subplots manually
        ax1 = fig.add_subplot(221, projection='polar')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')

        ax1.set_theta_direction(-1)
        ax1.set_theta_offset(np.pi / 2.0)
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Z Orientation')

        # Initialize data for animation
        angles = np.arctan2(np.stack(df_filtered['head_orientation'])[..., 1], np.stack(df_filtered['head_orientation'])[..., 0])
        radii = df_filtered.index.get_level_values('timestep')
        z_orientation = np.stack(df_filtered['head_orientation'])[..., 2]
        keypoints = np.stack(df_filtered['keypoints']).transpose(1, 0, 2)  # Transpose for easier indexing

        # Calculate axis limits
        min_radii, max_radii = np.nanmin(radii), np.nanmax(radii)
        min_z, max_z = np.nanmin(z_orientation), np.nanmax(z_orientation)
        min_keypoints = np.nanmin(keypoints, axis=(0,1))
        max_keypoints = np.nanmax(keypoints, axis=(0,1))
      
        # Animation update function
        def update(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # Update polar plot (ax1)
            ax1.plot(angles[:frame], radii[:frame], linestyle='-', color='b')
            ax1.set_title(f'Head x-y orientation for ped {pedestrian_id}')
            ax1.set_ylim(min_radii, max_radii)  # Fix radii bounds

            # Update Z orientation plot (ax2)
            ax2.plot(df_filtered.index[:frame], z_orientation[:frame], linestyle='-', color='b')
            ax2.set_title(f'Head z orientation for {pedestrian_id}')
            ax2.set_xlim(min_radii, max_radii)  # Fix timestep bounds
            ax2.set_ylim(min_z, max_z)  # Fix Z orientation bounds

            # 3D plot of keypoints over time
            for ax, view_angle in zip([ax3, ax4], [(15, 45), (15, 135)]):
                ax.set_xlim(min_keypoints[0], max_keypoints[0])  # Fix X bounds
                ax.set_ylim(min_keypoints[1], max_keypoints[1])  # Fix Y bounds
                ax.set_zlim(min_keypoints[2], max_keypoints[2])  # Fix Z bounds
                draw_pose_3d_single_frame(keypoints[None], ax, thin=False, frame_idx=frame, connectivities=OPENPOSE44_CONNECTIONS)
                ax.view_init(*view_angle)
                ax.set_title(f'Keypoints over time for {pedestrian_id} - View Angle: {view_angle}')

            fig.suptitle(f'Head orientation and keypoints over time in location {location} - Frame {frame}')

        # Create animation
        ani = FuncAnimation(fig, update, frames=len(radii), interval=1000/3)

        os.makedirs(save_dir, exist_ok=True)
        gif_filename = f'{save_dir}/{location}_ped-{pedestrian_id.split(":")[-1]}_head_orientation_ts-19.gif'
        ani.save(gif_filename)
        print(f"Saved animation for pedestrian {pedestrian_id}, location {location} to {gif_filename}")
        plt.close()
    

def plot_head_orientation_std_vs_distance_to_robot(agents_df, robot_df, location, dont_plot=False, save_dir=None):
    """Plot the standard deviation of head orientation vs distance of the agent to robot, for each agent"""
    # aggregate std dev of head orientations
    unique_ids = agents_df.index.get_level_values('id').unique()
    circular_std_devs = []
    resultants = []
    distances_to_robot = []
    for pedestrian_id in unique_ids:
        # Filter the DataFrame for the current pedestrian id
        df_filtered = agents_df.xs(pedestrian_id, level='id')
        num_timesteps_to_plot = 19  # len(df_filtered) // 5  # the whole length of the pedestrian trajectory

        # downsample to every 5th timestep, making sure the actual timestep value are every 5th, rather than just taking every fifth
        timesteps_every_5th = df_filtered.index.get_level_values('timestep').min() + 5 * np.arange(
            num_timesteps_to_plot)
        timesteps_every_5th = np.intersect1d(timesteps_every_5th,
                                             df_filtered.index.get_level_values('timestep').unique())
        df_filtered = df_filtered.loc[timesteps_every_5th]

        # ped-level statistics
        if np.all(np.isnan(np.stack(df_filtered['gaze']))):
            # print(f"Location: {location}, pedestrian_id: {pedestrian_id}, circular_std_dev is NaN")
            continue
        mean_angle, circular_std_dev, resultant = get_avg_std_dev_orientation(np.stack(df_filtered['gaze']))

        circular_std_devs.append(circular_std_dev)
        resultants.append(resultant)

        # get distance to robot
        robot_timesteps = robot_df.index.get_level_values('timestep').unique()
        robot_timesteps = np.intersect1d(robot_timesteps, df_filtered.index.get_level_values('timestep').unique())
        robot_df_filtered = robot_df.loc[robot_timesteps]
        distance_to_robot = np.linalg.norm(np.stack(df_filtered['p'].values - robot_df_filtered['p'].values), axis=1)
        avg_distance_to_robot = np.mean(distance_to_robot)
        distances_to_robot.append(avg_distance_to_robot)

    if dont_plot:
        return circular_std_devs, resultants, distances_to_robot
    # plot
    plt.scatter(distances_to_robot, circular_std_devs)
    plt.xlabel('Distance to robot')
    plt.ylabel('Circular std dev of head orientation')
    plt.title(f'Location: {location} - Circular std dev of head orientation vs distance to robot')
    plt.savefig(f'{save_dir}/{location}_circular_std_dev_vs_distance_to_robot.png')
    plt.close()

    # plot distance vs resultant
    plt.scatter(distances_to_robot, resultants)
    plt.xlabel('Distance to robot')
    plt.ylabel('Resultant of head orientation')
    plt.title(f'Location: {location} - Resultant of head orientation vs distance to robot')
    plt.savefig(f'{save_dir}/{location}_resultant_vs_distance_to_robot.png')
    plt.close()

    print("saved plots for location: ", location)


def aggregate_head_orientation_std_dev(agents_df, location, save_dir):
    # aggregate std dev of head orientations
    unique_ids = agents_df.index.get_level_values('id').unique()
    circular_std_devs = []
    resultants = []
    for pedestrian_id in unique_ids:
        # Filter the DataFrame for the current pedestrian id
        df_filtered = agents_df.xs(pedestrian_id, level='id')
        num_timesteps_to_plot = 19  # len(df_filtered) // 5  # the whole length of the pedestrian trajectory

        # downsample to every 5th timestep, making sure the actual timestep value are every 5th, rather than just taking every fifth
        timesteps_every_5th = df_filtered.index.get_level_values('timestep').min() + 5 * np.arange(
            num_timesteps_to_plot)
        timesteps_every_5th = np.intersect1d(timesteps_every_5th,
                                             df_filtered.index.get_level_values('timestep').unique())
        df_filtered = df_filtered.loc[timesteps_every_5th]

        # ped-level statistics
        if np.all(np.isnan(np.stack(df_filtered['gaze']))):
            # print(f"Location: {location}, pedestrian_id: {pedestrian_id}, circular_std_dev is NaN")
            continue
        mean_angle, circular_std_dev, resultant = get_avg_std_dev_orientation(np.stack(df_filtered['gaze']))

        circular_std_devs.append(circular_std_dev)
        resultants.append(resultant)

    print(f"Location: {location} mean circular std dev: {np.mean(circular_std_devs):.2f}, std dev circular std dev: {np.std(circular_std_devs):.2f}, mean resultant: {np.mean(resultants):.2f}, std dev resultant: {np.std(resultants):.2f}")



def plot_head_orientation_over_time(agents_df, location, save_dir):
    # Plotting for each pedestrian id separately
    unique_ids = agents_df.index.get_level_values('id').unique()

    # Loop through each pedestrian id
    num_plot = 3  # plot 3 pedestrians per location
    num_skip = len(unique_ids) // num_plot
    for pedestrian_id in unique_ids[::num_skip]:
        # Filter the DataFrame for the current pedestrian id
        df_filtered = agents_df.xs(pedestrian_id, level='id')
        num_timesteps_to_plot = 19 #len(df_filtered) // 5  # the whole length of the pedestrian trajectory

        # downsample to every 5th timestep, making sure the actual timestep value are every 5th, rather than just taking every fifth
        timesteps_every_5th = df_filtered.index.get_level_values('timestep').min() + 5 * np.arange(num_timesteps_to_plot)
        timesteps_every_5th = np.intersect1d(timesteps_every_5th, df_filtered.index.get_level_values('timestep').unique())
        # make sure everything in timestep_ever_5th is in the index
        assert np.all(np.isin(timesteps_every_5th, df_filtered.index.get_level_values('timestep').unique())), f"timesteps_every_5th: {timesteps_every_5th}\nagents_df.index.get_level_values('timestep').unique(): {agents_df.index.get_level_values('timestep').unique()}"
        df_filtered = df_filtered.loc[timesteps_every_5th]

        # ped-level statistics
        if np.all(np.isnan(np.stack(df_filtered['gaze']))):
            print(f"Location: {location}, pedestrian_id: {pedestrian_id}, gaze is NaN")
            continue
        mean_angle, circular_std_dev, resultant = get_avg_std_dev_orientation(np.stack(df_filtered['gaze']))

        # Polar plot for x and y
        fig = plt.figure(figsize=(12, 6))

        # Circular plot for x and y orientations
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_direction(-1)
        ax1.set_theta_offset(np.pi / 2.0)

        # Calculate the angle and radius for x and y
        angles = np.arctan2(np.stack(df_filtered['head_orientation'])[...,1], np.stack(df_filtered['head_orientation'])[...,0])
        radii = df_filtered.index.get_level_values('timestep')

        ax1.plot(angles, radii, linestyle='-', color='b')
        fig.suptitle(f'Head orientation over time in location {location}')
        ax1.set_title(f'Head x-y orientation over time for ped {pedestrian_id}. mean_angle: {mean_angle:.2f}, circular_std_dev: {circular_std_dev:.2f}, resultant: {resultant:.2f}')

        ax2 = fig.add_subplot(122)
        ax2.plot(df_filtered.index, np.stack(df_filtered['head_orientation'])[...,2], linestyle='-', color='b')
        ax2.set_title(f'Head z orientation over time for {pedestrian_id}')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Z Orientation')

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        if isinstance(pedestrian_id, str) and ":" in pedestrian_id:
            ped_id = pedestrian_id.split(":")[-1]
        else:
            ped_id = pedestrian_id
        plt.savefig(f'{save_dir}/{location}_ped-{ped_id}_head_orientation_ts-19.png')
        print(f'Saved plot for pedestrian {ped_id}, location {location} to {save_dir}/{location}_ped-{ped_id}_head_orientation_ts-19.png')
        plt.close()


def main_jrdb():
    OUTPUT_DIR = '../viz/jrdb_keypoint_head_orientations'
    pose_type = 'hmr'  # 'hmr' or 'blazepose'
    circular_std_devs = []
    resultants = [] 
    distances_to_robot = []

    for location in TRAIN + TEST:
        agents_df = load_agents_df_jrdb(location, pose_type)
        plot_head_orientation_over_time(agents_df, location, OUTPUT_DIR)
        anim_head_orientation_over_time(agents_df, location, OUTPUT_DIR)
        # aggregate_head_orientation_std_dev(agents_df, location, OUTPUT_DIR)

        robot_df = load_robot_df(location)
        # remove extra data odometry datapoints
        robot_df = robot_df.iloc[agents_df.index.levels[0]]
        circular_std_dev, resultant, distance_to_robot = plot_head_orientation_std_vs_distance_to_robot(agents_df, robot_df, location, dont_plot=True)
        circular_std_devs.extend(circular_std_dev)
        resultants.extend(resultant)
        distances_to_robot.extend(distance_to_robot)

    # plot all locations distance to robot vs circular std dev
    plt.scatter(distances_to_robot, circular_std_devs)
    plt.ylabel('circular std dev of head orientation')
    plt.title(f'location: {location} - circular std dev of head orientation vs distance to robot')
    plt.savefig(f'{OUTPUT_DIR}/all_circular_std_dev_vs_distance_to_robot.png')
    plt.close()

    # plot distance vs resultant
    plt.scatter(distances_to_robot, resultants)
    plt.xlabel('distance to robot')
    plt.ylabel('resultant of head orientation')
    plt.title(f'location: {location} - resultant of head orientation vs distance to robot')
    plt.savefig(f'{OUTPUT_DIR}/all_resultant_vs_distance_to_robot.png')
    plt.close()


def main_tbd():
    OUTPUT_DIR = '../viz/tbd_keypoint_head_orientations'
    for location in ALL_SESSIONS_TBD:
        agents_df = load_agents_df_tbd(location)
        # plot_head_orientation_over_time(agents_df, location, OUTPUT_DIR)
        plot_head_orientation_over_time_plotly(agents_df, location, OUTPUT_DIR)
        # anim_head_orientation_over_time(agents_df, location, OUTPUT_DIR)
        # aggregate_head_orientation_std_dev(agents_df, location, OUTPUT_DIR)


if __name__ == "__main__":
    main_tbd()
