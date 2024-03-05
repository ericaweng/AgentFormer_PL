""" Format JRDB 2D body keypoints into AgentFormer-consumable format. Try 2.A
(load in pre-processed data, and also save image width and height for the purpose of constructing a pose embedding
using WHAM pretrained models"""

import os
import json
import numpy as np
import argparse
import yaml
from pyquaternion import Quaternion


COCO_CONNECTIVITIES_LIST = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12], [4, 8],
                            [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]


def box_to_2d_corners(box):
    x, y, w, h = box
    x0, x1 = x, x + w
    y0, y1 = y, y + h
    return np.array([x0, y0, x1, y1])


def load_h_w_bboxes(scene, cam_num):
    """ detect 3d poses for all detections that have a fully_visible or partially_visible bbox detection
     OR have a 2d pose keypoint annotations """
    scene_name_w_image = f"{scene}_image{cam_num}"

    # load poses
    dataroot_poses = f"../PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_coco/"
    with open(os.path.join(dataroot_poses, f'{scene_name_w_image}.json'), 'r') as f:
        pose_labels = json.load(f)
    image_id_to_pose_annos = {
            int(image['file_name'].split('/')[-1].split('.')[0]):
                [ann for ann in pose_labels['annotations'] if ann['image_id'] == image['id']]
            for image in pose_labels['images']}

    # load bboxes
    dataroot_labels = f"../PoseFormer/datasets/jackrabbot/train/labels/labels_2d/"
    bboxes_path = os.path.join(dataroot_labels, f'{scene_name_w_image}.json')
    with open(bboxes_path, 'r') as f:
        bbox_labels = json.load(f)['labels']
    all_bboxes = {int(image_path.split('.')[0]): {int(label['label_id'].split(":")[-1]): label for label in labels}
                  for image_path, labels in sorted(bbox_labels.items())}

    # count = 0
    # total = 0
    # for image_id in image_id_to_pose_annos:
    #     for ped_id in image_id_to_pose_annos[image_id]:
    #         if ped_id['track_id'] not in all_bboxes[image_id]:
    #             count += 1
    #         total += 1
    # print(f"ped_id not in all_bboxes[image_id]: {count}/{total}")


    # convert poses visibility to score percentages
    h_w_bboxes = {}
    for image_id, annos in sorted(image_id_to_pose_annos.items(), key=lambda x: x[0]):
        poses_this_frame = {}
        for ann in annos:
            poses_this_frame[ann['track_id']] = {'height': pose_labels['images'][image_id]['height'],
                                                 'width': pose_labels['images'][image_id]['width']}

        for track_id, box in all_bboxes[image_id].items():
            if track_id in poses_this_frame:
                poses_this_frame[track_id]['box'] = box_to_2d_corners(box['box'])

        h_w_bboxes[image_id] = poses_this_frame

    # video_length, num_objects, {box: (4,), height: (1,), width: (1,)}
    return h_w_bboxes


def round_to_orthogonal(matrix):
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(matrix)

    # Reconstruct the nearest orthogonal matrix
    orthogonal_matrix = np.dot(U, Vt)

    return orthogonal_matrix


def main():
    parser = argparse.ArgumentParser(description='Pedestrian Trajectory Visualization')
    parser.add_argument('--dataroot_poses', '-dr', type=str,
                        default=f"../PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_stitched_coco/")
    parser.add_argument('--use_all_gt_boxes', action='store_true', default=False)
    args = parser.parse_args()

    # load camera calibration params for jrdb
    jrdb_calib_path = '../AgentFormerSDD/datasets/jrdb/train/calibration/cameras.yaml'
    with open(jrdb_calib_path) as f:
        camera_config_dict = yaml.safe_load(f)

    intrinsic_params = {}
    extrinsic_params = {}
    for cam_num in [0, 2, 4, 6, 8]:
        camera_params = camera_config_dict['cameras'][f'sensor_{cam_num}']
        K = camera_params['K'].split(' ')
        fx, fy, cx, cy = K[0], K[2], K[4], K[5]
        intrinsic_params[cam_num] = np.array(
            list(map(float, [fx, fy, cx, cy, *camera_params['D'].split(' ')])))  # intrinsic + distortion

        R = np.array(list(map(float, camera_params['R'].splitlines()[0].split(' ')))).reshape(3, 3)
        T = np.array(list(map(float, camera_params['T'].splitlines()[0].split(' '))))
        try:
            q = Quaternion(matrix=R)
        except ValueError:
            q = Quaternion(matrix=round_to_orthogonal(R))
        # get R and T as length 7 vector (*xyz, *quaternion)
        extrinsic_params[cam_num] = np.array([*T, *q])

    # load in previous preprocessed 2d kp from format_jrdb_2d_kp0.py
    poses_2d = np.load('../AgentFormerSDD/datasets/jrdb_adjusted/poses_2d.npz', allow_pickle=True)['arr_0']
    all_scores = poses_2d.item()['scores']
    all_poses = poses_2d.item()['poses']
    all_cam_ids = poses_2d.item()['cam_ids']

    # make a single dict for all data, move data type to the finest-grained level
    all_data = {}
    for scene, this_scene_poses in all_poses.items():
        all_data[scene] = {}
        cam_id_to_hwbox = {}
        for frame, this_frame_poses in this_scene_poses.items():
            for ped_id, kp in this_frame_poses.items():
                if ped_id not in all_data[scene]:
                    all_data[scene][ped_id] = {}
                if frame not in all_data[scene][ped_id]:
                    all_data[scene][ped_id][frame] = {}

                cam_num = all_cam_ids[scene][frame][ped_id]
                if cam_num not in cam_id_to_hwbox:
                    cam_id_to_hwbox[cam_num] = load_h_w_bboxes(scene, cam_num)

                height = cam_id_to_hwbox[cam_num][frame][ped_id]['height']
                width = cam_id_to_hwbox[cam_num][frame][ped_id]['width']
                all_data[scene][ped_id][frame] = {'pose': kp,
                                                  'score': all_scores[scene][frame][ped_id],
                                                  'cam_id': cam_num,
                                                  'intrinsics': intrinsic_params[cam_num],
                                                  'extrinsics': extrinsic_params[cam_num],
                                                  'height': height,
                                                  'width': width,}

                if 'box' in cam_id_to_hwbox[cam_num][frame][ped_id]:
                    all_data[scene][ped_id][frame]['box'] = cam_id_to_hwbox[cam_num][frame][ped_id]['box']

            # make sure all frames are in increments

    np.savez('../AgentFormerSDD/datasets/jrdb_adjusted/poses_2d_for_wham_pose_embedding.npz', data=all_data)


if __name__ == '__main__':
    main()
