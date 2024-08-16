import os
import open3d as o3d
import numpy as np
import torch
import cv2
import copy
import yaml
import matplotlib.pyplot as plt
from jrdb_toolkit.visualisation.visualize_constants import TRAIN
from rotate_and_transform_hmr_kp import project_ref_to_image_torch, calculate_median_param_value
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def move_lidar_to_camera_frame(self, pointcloud, upper=True):
    # assumed only rotation about z axis
    if upper:
        pointcloud[:, :3] = \
            pointcloud[:, :3] - torch.Tensor(self.global_config_dict['calibrated']
                                             ['lidar_upper_to_rgb']['translation']).type(pointcloud.type())
        theta = self.global_config_dict['calibrated']['lidar_upper_to_rgb']['rotation'][-1]
    else:
        pointcloud[:, :3] = \
            pointcloud[:, :3] - torch.Tensor(self.global_config_dict['calibrated']
                                             ['lidar_lower_to_rgb']['translation']).type(pointcloud.type())
        theta = self.global_config_dict['calibrated']['lidar_lower_to_rgb']['rotation'][-1]

    rotation_matrix = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).type(
            pointcloud.type())
    pointcloud[:, :2] = torch.matmul(rotation_matrix, pointcloud[:, :2].unsqueeze(2)).squeeze()
    pointcloud[:, :3] = self.project_velo_to_ref(pointcloud[:, :3])
    return pointcloud


def build_pcd(global_config_dict,
              image_path=None,
              lidar_upper_path=None,
              lidar_lower_path=None,
              color_pcd=False,
median_focal_length_y=None, median_optical_center_y=None, img_shape=None
              ):
    geo = []
    all_points = None
    for i, pcd_path in enumerate([lidar_upper_path, lidar_lower_path]):
        if pcd_path == None:
            continue
        pcd = o3d.io.read_point_cloud(pcd_path)
        if i == 0:
            translation = np.asarray(global_config_dict['calibrated']['lidar_upper_to_rgb']['translation'])
            theta = global_config_dict['calibrated']['lidar_upper_to_rgb']['rotation'][-1]
        else:
            translation = np.asarray(global_config_dict['calibrated']['lidar_lower_to_rgb']['translation'])
            theta = global_config_dict['calibrated']['lidar_lower_to_rgb']['rotation'][-1]
        # Get the points
        points = torch.tensor(np.array(pcd.points)).type(torch.float32)
        # points = points[::5,:]

        # if all_points is None:
        #     all_points = np.asarray(points)
        # else:
        #     all_points = np.concatenate((all_points, points), axis=0)
        # print(points[::5,:].shape)
        # exit()


        if color_pcd:
            image = cv2.imread(image_path)

            projected_points = self.move_lidar_to_camera_frame(copy.deepcopy(points),
                                                               upper=True if i == 0 else False)
            projected_points = self.project_ref_to_image_torch(projected_points)
            projected_points = np.floor(np.asarray(projected_points).T).astype(np.int64)

            true_where_x_on_img = (0 <= projected_points[1]) & (projected_points[1] < image.shape[1])
            true_where_y_on_img = (0 <= projected_points[0]) & (projected_points[0] < image.shape[0])
            true_where_point_on_img = true_where_x_on_img & true_where_y_on_img

            points = points[true_where_point_on_img]
            projected_points = projected_points.T[true_where_point_on_img].T

            colors = image[projected_points[1], projected_points[0]]  # BGR
            colors = np.squeeze(cv2.cvtColor(np.expand_dims(colors, 0), cv2.COLOR_BGR2RGB))

        # Get rotation matrix from lidar to camera
        rotation_matrix = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).type(
                points.type())
        # Start rotating
        points[:, :2] = torch.matmul(rotation_matrix, points[:, :2].unsqueeze(2)).squeeze()
        # Translate the position of points
        points -= translation
        # points = self.filter_points(points, range=[-100,100,-100,100,-0.5,1])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        if color_pcd:
            pc.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
        geo.append(pc)

    return geo


def main():
    location = 'cubberly-auditorium-2019-04-22_1'#'clark-center-2019-02-28_1'
    root_dir = f'datasets/jrdb/{"train" if location in TRAIN else "test"}'
    file_idx = '000000'
    img_path = f"{root_dir}/images/image_stitched/{location}/{file_idx}.jpg"
    lidar_upper_path = f"{root_dir}/pointclouds/upper_velodyne/{location}/{file_idx}.pcd"
    lidar_lower_path = f"{root_dir}/pointclouds/lower_velodyne/{location}/{file_idx}.pcd"
    color_pcd = False

    calib_folder = 'datasets/jrdb/train/calibration'
    global_config = os.path.join(calib_folder, 'defaults.yaml')
    with open(global_config) as f:
        global_config_dict = yaml.safe_load(f)
    # camera_config = os.path.join(calib_folder, 'cameras.yaml')
    # with open(camera_config) as f:
    #     camera_config_dict = yaml.safe_load(f)

    median_focal_length_y = calculate_median_param_value(param='f_y')
    median_optical_center_y = calculate_median_param_value(param='t_y')
    img_shape = 3, global_config_dict['image']['height'], global_config_dict['image']['width']

    geo = build_pcd(global_config_dict, img_path, lidar_upper_path, lidar_lower_path,
                    color_pcd, median_focal_length_y, median_optical_center_y, img_shape)
    print(f"{geo=}")

    # display the filtered_points
    geo = geo[0] + geo[1]
    # CODE HERE

    for threshold in [-0.2]:
        # Filter filtered_points with z-value > 0.5
        filtered_points = np.asarray(geo.points)
        filtered_points = filtered_points[filtered_points[:, 2] > threshold]

        # Plot the filtered_points on the x and y plane
        plt.figure(figsize=(10, 10))
        plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Filtered Point Cloud (z > {threshold})')
        plt.axis('equal')

        # Save the plot
        output_path = f'../viz/jrdb_pc/projected_pcd_env_image-{threshold}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        # plot distance fn
        if True:
            filtered_points = filtered_points[..., :2]
            from data.preprocess_w_odometry import pc_to_odometry_frame, filter_agents_and_ground_from_point_cloud
            scene_pc_odometry = pc_to_odometry_frame({int(file_idx): filtered_points}, robot_df)[0]
            pc_this_frame_global = filter_agents_and_ground_from_point_cloud(
                    agents_in_odometry_df, {int(file_idx): scene_pc_odometry}, robot_df, max_dist=100,)

            pc_min = np.floor(filtered_points.min(axis=0)).astype(int)
            pc_max = np.ceil(filtered_points.max(axis=0)).astype(int)

            grid_size_x, grid_size_y = pc_max - pc_min
            x = np.linspace(pc_min[0], pc_max[0], grid_size_x)
            y = np.linspace(pc_min[1], pc_max[1], grid_size_y)
            xx, yy = np.meshgrid(x, y)
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Compute the distance from each grid point to the nearest point in the point cloud
            tree = cKDTree(filtered_points)
            distances, _ = tree.query(grid_points)

            # Reshape the distances to match the grid shape
            distance_grid = distances.reshape((grid_size_y, grid_size_x))

            # Plot the resulting distance function
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, distance_grid, levels=50, cmap='viridis')
            plt.colorbar(label='Distance to nearest point')
            # plt.scatter(filtered_points[:, 0], filtered_points[:, 1], color='red', edgecolor='white', s=1)
            plt.title('2D Distance Function')
            plt.xlabel('X')
            plt.ylabel('Y')

            # Save the plot
            output_path=f'../viz/jrdb_pc/projected_pcd_env_image-dist_fn.png'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()

if __name__ == "__main__":
    main()

