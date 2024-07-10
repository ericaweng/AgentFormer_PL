import os
import numpy as np
import cv2
import json
import pandas as pd
import yaml
import torch

from jrdb_toolkit.visualisation.visualize_constants import *

from jrdb_toolkit.visualisation.visualize_utils import *
from data.preprocess_w_odometry import (get_robot, get_robot_kiss_icp, get_agents_dict,
                                        get_agents_dict_from_detections, get_agents_features_with_box,
                                        get_agents_keypoints, get_agents_keypoints_hmr)
from data.preprocess_test_w_odometry import get_agents_features_df_with_box, df_to_boxes


def project_ref_to_image_torch(pointcloud, img_shape, median_focal_length_y, median_optical_center_y):
    theta = (torch.atan2(pointcloud[:, 0], pointcloud[:, 2]) + np.pi) % (2 * np.pi)
    horizontal_fraction = theta / (2 * np.pi)

    x = (horizontal_fraction * img_shape[2]) % img_shape[2]
    y = -median_focal_length_y * (
            pointcloud[:, 1] * torch.cos(theta) / pointcloud[:, 2]) + median_optical_center_y
    pts_2d = torch.stack([x, y], dim=1)

    return pts_2d


def calculate_median_param_value(param):
    if param == 'f_y':
        idx = 4
    elif param == 'f_x':
        idx = 0
    elif param == 't_y':
        idx = 5
    elif param == 't_x':
        idx = 2
    elif param == 's':
        idx = 1
    else:
        raise 'Wrong parameter!'

    omni_camera = ['sensor_0', 'sensor_2', 'sensor_4', 'sensor_6', 'sensor_8']
    parameter_list = []

    camera_config = f'../AgentFormerSDD/datasets/jrdb/train/calibration/cameras.yaml' ##os.path.join(calib_folder, 'cameras.yaml')
    with open(camera_config) as f:
        camera_config_dict = yaml.safe_load(f)
    for sensor, camera_params in camera_config_dict['cameras'].items():
        if sensor not in omni_camera:
            continue
        K_matrix = camera_params['K'].split(' ')
        parameter_list.append(float(K_matrix[idx]))
    return np.median(parameter_list)


def project_velo_to_ref(pointcloud):

    pointcloud = pointcloud[:, [1, 2, 0]]
    pointcloud[:, 0] *= -1
    pointcloud[:, 1] *= -1

    return pointcloud



def get_bboxes_stitched(scene_name):
    """ get only one dict for all cams, add cam info at the innermost level (alongside bbox) """

    # load bboxes
    dataroot_labels = f"../AgentFormerSDD/datasets/jrdb/train/labels/labels_2d_stitched"
    bboxes_path = os.path.join(dataroot_labels, f'{scene_name}.json')
    with open(bboxes_path, 'r') as f:
        bbox_labels = json.load(f)['labels']
        # frame id to ped id to label
    all_bboxes = {int(image_path.split('.')[0]): {int(label['label_id'].split(":")[-1]): label for label in labels}
                           # if 'visible' in label['attributes']['occlusion']}
                           for image_path, labels in sorted(bbox_labels.items())}

    # remove boxes with too-low occlusion score
    best_bboxes = {}
    for frame, peds in all_bboxes.items():
        for ped, label in peds.items():
            if frame not in best_bboxes:
                best_bboxes[frame] = {}
            assert ped not in best_bboxes[frame], f"frame: {frame}, ped: {ped} already in best_bboxes"
            best_bboxes[frame][ped] = label['box'][:2]

    return best_bboxes


def get_3d_box(all_labels, frame_id, ped_id):
    if f"{frame_id:06}.pcd" in all_labels:
        for ped in all_labels[f"{frame_id:06}.pcd"]:
            if ped['label_id'] == f'pedestrian:{ped_id}':
                return np.array([ped['box']['cx'], ped['box']['cy'], ped['box']['cz']])
    elif f"{frame_id:06}.pcb" in all_labels:
        for ped in all_labels[f"{frame_id:06}.pcb"]:
            if ped['label_id'] == f'pedestrian:{ped_id}':
                return np.array([ped['box']['cx'], ped['box']['cy'], ped['box']['cz']])
    return None


def main():
    for seq_name in TEST:
        hmr_labels_3d_pose = np.load(f"datasets/jrdb/jrdb_hmr2_raw_stitched/{seq_name}_kp3d.npz", allow_pickle=True)['arr_0'].item()
        root_location = "datasets/jrdb"
        new_save_path = f"datasets/jrdb/jrdb_hmr2_raw_stitched/{seq_name}_kp_3d.npz"

        calib_folder = f"{root_location}/train/calibration"
        global_config = os.path.join(calib_folder, 'defaults.yaml')
        with open(global_config) as f:
            global_config_dict = yaml.safe_load(f)

        if seq_name in TRAIN:
            bbox_2d = get_bboxes_stitched(seq_name)
            root_dir = "datasets/jrdb/train"
            labels_3d = None
            # label_3d_path = f"{root_dir}/labels/labels_3d/{seq_name}.json"
            # with open(label_3d_path, 'r') as f:
            #     labels_3d = json.load(f)
            #     labels_3d = labels_3d['labels']
        else:
            assert seq_name in TEST
            root_dir = "datasets/jrdb/test"
            bbox_2d = None
            agents_df = get_agents_features_df_with_box(root_dir, TEST_LOCATION_TO_ID[seq_name],
                                                        max_distance_to_robot=MAX_DISTANCE_TO_ROBOT,
                                                        tracking_method=TRACKING_METHOD)
            labels_3d = df_to_boxes(agents_df)

        # image shape is (color channels, height, width)
        median_focal_length_y = calculate_median_param_value(param='f_y')
        median_optical_center_y = calculate_median_param_value(param='t_y')
        img_shape = 3, global_config_dict['image']['height'], global_config_dict['image']['width']

        # rotate and transform
        print(f"rotating and transforming {seq_name}")

        for frame_id, frames in hmr_labels_3d_pose.items():
            for ped_id, peds in frames.items():
                if peds is None:
                    continue

                pts_2d = None
                if bbox_2d is not None:
                    pts_2d = bbox_2d.get(frame_id, None)
                    if pts_2d is not None:
                        pts_2d = pts_2d.get(ped_id, None)

                if pts_2d is None:
                    bbox_3d = get_3d_box(labels_3d, frame_id, ped_id)

                    center_pos = np.floor(
                            project_ref_to_image_torch(
                                    project_velo_to_ref(
                                            torch.from_numpy(np.array(bbox_3d[:3])).reshape(1, 3)),
                                    img_shape, median_focal_length_y, median_optical_center_y
                            ).numpy().reshape(-1)).astype(np.int64)
                    if center_pos[0] < 0 or center_pos[0] >= img_shape[2]:
                        center_pos[0] = 0
                    if center_pos[1] < 0 or center_pos[1] >= img_shape[1]:
                        center_pos[1] = 0
                else:
                    center_pos = pts_2d[0:2]

                kp_robot_frame = peds
                # finish camera to lidar coords transformation
                kp_robot_frame = kp_robot_frame[:, [1, 0, 2]]
                kp_robot_frame[:, 1] = -kp_robot_frame[:, 1]

                # rotate the pose too
                yaw = -determine_global_yaw(center_pos, img_shape)
                kp_robot_frame = rotate_3d_pts_about_z_axis(kp_robot_frame, yaw)
                hmr_labels_3d_pose[frame_id][ped_id] = kp_robot_frame

        # save
        np.savez(new_save_path, hmr_labels_3d_pose)


if __name__ == '__main__':
    main()
