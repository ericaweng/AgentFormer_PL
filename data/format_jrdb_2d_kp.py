""" Format JRDB 2D body keypoints into AgentFormer-consumable format. Try 2.B
(starting from scratch. load in both bboxes and poses, and also save image width and height,
for the purpose of constructing a pose embedding using WHAM pretrained models"""

import os
import json
import numpy as np
import argparse
import yaml
from pyquaternion import Quaternion

from jrdb_split import TRAIN


COCO_CONNECTIVITIES_LIST = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12], [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]


def box_to_2d_corners(box):
    x, y, w, h = box
    x0, x1 = x, x + w
    y0, y1 = y, y + h
    return np.array([x0, y0, x1, y1])


def load_bboxes_and_poses(scene_name_w_image, args):
    """ detect 3d poses for all detections that have a fully_visible or partially_visible bbox detection
     OR have a 2d pose keypoint annotations """

    # load poses
    dataroot_poses = f"../PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_coco/"
    with open(os.path.join(dataroot_poses, f'{scene_name_w_image}.json'), 'r') as f:
        pose_labels = json.load(f)
    image_id_to_pose_annos = {
            int(image['file_name'].split('/')[-1].split('.')[0]): [ann for ann in pose_labels['annotations']
                                                                   if ann['image_id'] == image['id']]
            for image in pose_labels['images']}

    # load bboxes
    dataroot_labels = f"../PoseFormer/datasets/jackrabbot/train/labels/labels_2d/"
    bboxes_path = os.path.join(dataroot_labels, f'{scene_name_w_image}.json')
    with open(bboxes_path, 'r') as f:
        bbox_labels = json.load(f)['labels']
    all_bboxes = {int(image_path.split('.')[0]): {int(label['label_id'].split(":")[-1]): label
                   for label in labels}  # if 'visible' in label['attributes']['occlusion']}
                  for image_path, labels in sorted(bbox_labels.items())}

    for image_id in image_id_to_pose_annos:
        assert image_id in all_bboxes, f"all images with poses should have a bbox detection, but image_id {image_id} is not have a bbox. only these image_ids have bboxes: {all_bboxes.keys()}"

    all_poses_and_bboxes = {}
    for image_id, annos in sorted(image_id_to_pose_annos.items(), key=lambda x: x[0]):
        poses_this_frame = {}
        for ann in annos:
            pose_reformatted = np.array(ann['keypoints']).reshape(17, 3)

            # convert kp visibility to confidence score
            score = pose_reformatted[:, -1]
            score = np.where(score == 0, 0.1, score)
            score = np.where(score == 1, 0.5, score)
            score = np.where(score == 2, 1, score)
            assert np.all([np.isclose(p, 0.1) or np.isclose(p, 0.5) or np.isclose(p, 1) for p in score]), f"score: {score}"

            poses_this_frame[ann['track_id']] = {'pose': pose_reformatted[...,:2], 'score': score,
                                                 'height': pose_labels['images'][image_id]['height'],
                                                 'width': pose_labels['images'][image_id]['width']}

        for track_id, box in all_bboxes[image_id].items():
            if track_id in poses_this_frame:
                poses_this_frame[track_id]['box'] = box_to_2d_corners(box['box'])

        # add boxes to those that do not have one
        for track_id, pose_bbox_dict in poses_this_frame.items():
            if 'box' not in pose_bbox_dict:
                ## get max and min of the keypoints as the box
                poses_this_frame[track_id]['box'] = np.concatenate([np.min(pose_bbox_dict['pose'], axis=0)[:2],
                                                                    np.max(pose_bbox_dict['pose'], axis=0)[:2]])
            x0,y0,x1,y1 = pose_bbox_dict['box']
            assert x1 > x0 and y1 > y0, f"invalid box: {pose_bbox_dict['box']}"

        # all_poses_and_bboxes.append(poses_this_frame)
        all_poses_and_bboxes[image_id] = poses_this_frame

    # video_length, num_objects, {pose: (17, 2), box: (4,), height: (1,), width: (1,), score: (17, 1)}
    return all_poses_and_bboxes


def round_to_orthogonal(matrix):
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(matrix)

    # Reconstruct the nearest orthogonal matrix
    orthogonal_matrix = np.dot(U, Vt)

    return orthogonal_matrix


def main():
    parser = argparse.ArgumentParser(description='Pedestrian Trajectory Visualization')
    parser.add_argument('--dataroot_poses', '-dr', type=str, default=f"../PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_stitched_coco/")
    parser.add_argument('--use_all_gt_boxes', action='store_true', default=False)
    args = parser.parse_args()

    # 2d poses stitched
    # poses_stitched = {}
    # for scene in TRAIN:
    #     poses_stitched[scene] = load_poses(scene, args)
    # np.savez('../AgentFormerSDD/datasets/jrdb_adjusted/poses_stitched_2d.npz', **poses_stitched)
    # print(f"saved to: ../AgentFormerSDD/datasets/jrdb_adjusted/poses_stitched_2d.npz")
    # import ipdb; ipdb.set_trace()

    # 2d poses, individual cameras
    args.dataroot_poses = "../PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_coco/"

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
        intrinsic_params[cam_num] = np.array(list(map(float, [fx, fy, cx, cy, *camera_params['D'].split(' ')])))  # intrinsic + distortion

        R = np.array(list(map(float, camera_params['R'].splitlines()[0].split(' ')))).reshape(3, 3)
        T = np.array(list(map(float, camera_params['T'].splitlines()[0].split(' '))))
        try:
            q = Quaternion(matrix=R)
        except ValueError:
            q = Quaternion(matrix=round_to_orthogonal(R))
        # get R and T as length 7 vector (*xyz, *quaternion)
        extrinsic_params[cam_num] = np.array([*T, *q])

    # merge poses from different cams. if two cams have the same kp, then pick the kp with greater number of visible keypoints
    all_poses = {}
    for cam_num in [0, 2, 4, 6, 8]:
        all_poses[cam_num] = {}
        for scene in TRAIN:
            scene_name = f'{scene}_image{cam_num}'
            all_poses[cam_num][scene] = load_bboxes_and_poses(scene_name, args)
    print("done loading all poses")

    for cam_num in [2, 4, 6, 8]:
        for scene, this_scene_poses in all_poses[cam_num].items():
            for frame, this_frame_poses in this_scene_poses.items():
                for ped_id, kp in this_frame_poses.items():
                    if (ped_id in combined_poses[scene][frame]
                            and np.sum(kp['score']) > np.sum(scores[scene][frame][ped_id])
                    or ped_id not in combined_poses[scene][frame]):
                        combined_poses[scene][frame][ped_id] = kp['kp']
                        cam_intrinsics[scene][frame][ped_id] = intrinsic_params[cam_num]
                        cam_id[scene][frame][ped_id] = cam_num
                        cam_extrinsics[scene][frame][ped_id] = extrinsic_params[cam_num]
                        scores[scene][frame][ped_id] = kp['score']


    combined_poses = {}
    cam_intrinsics = {}
    cam_extrinsics = {}
    scores = {}
    cam_id = {}
    for scene, this_scene_poses in all_poses[0].items():
        combined_poses[scene] = {}
        cam_intrinsics[scene] = {}
        cam_extrinsics[scene] = {}
        scores[scene] = {}
        cam_id[scene] = {}
        for frame, poses in this_scene_poses.items():
            combined_poses[scene][frame] = {}
            cam_intrinsics[scene][frame] = {}
            cam_extrinsics[scene][frame] = {}
            scores[scene][frame] = {}
            cam_id[scene][frame] = {}
            for ped_id, kp in poses.items():
                combined_poses[scene][frame][ped_id] = kp['kp']#{'kp': kp, 'cam_id': 0, 'intrinsics': intrinsic_params[0]}
                cam_intrinsics[scene][frame][ped_id] = intrinsic_params[0]
                cam_id[scene][frame][ped_id] = 0
                cam_extrinsics[scene][frame][ped_id] = extrinsic_params[0]
                scores[scene][frame][ped_id] = kp['score']

    for cam_num in [2, 4, 6, 8]:
        for scene, this_scene_poses in all_poses[cam_num].items():
            for frame, this_frame_poses in this_scene_poses.items():
                for ped_id, kp in this_frame_poses.items():
                    if (ped_id in combined_poses[scene][frame]
                            and np.sum(kp['score']) > np.sum(scores[scene][frame][ped_id])
                    or ped_id not in combined_poses[scene][frame]):
                        combined_poses[scene][frame][ped_id] = kp['kp']
                        cam_intrinsics[scene][frame][ped_id] = intrinsic_params[cam_num]
                        cam_id[scene][frame][ped_id] = cam_num
                        cam_extrinsics[scene][frame][ped_id] = extrinsic_params[cam_num]
                        scores[scene][frame][ped_id] = kp['score']

    print(f"combined_poses: {combined_poses['bytes-cafe-2019-02-07_0'][0][1]}")
    print(f"combined_poses: {scores['bytes-cafe-2019-02-07_0'][0][1]}")

    counts = {'num_scenes':0, 'num_frames':0, 'num_unique_peds':0,'ped_id_to_set':set(), 'num_each_cam_id':{0:0, 2:0, 4:0, 6:0, 8:0}}
    for scene, this_scene_poses in combined_poses.items():
        counts['num_scenes'] += 1
        for frame, this_frame_poses in this_scene_poses.items():
            counts['num_frames'] += 1
            for ped_id, kp in this_frame_poses.items():
                if ped_id not in counts['ped_id_to_set']:
                    counts['num_unique_peds'] += 1
                counts['ped_id_to_set'].add(ped_id)
                counts['num_each_cam_id'][cam_id[scene][frame][ped_id]] += 1

    # avg num_frames
    counts['num_frames'] = counts['num_frames'] / counts['num_scenes']
    # avg num_unique_peds
    counts['num_unique_peds'] = counts['num_unique_peds'] / counts['num_scenes']

    # print stats nicely, one per line
    print("\n".join([f"{k}: {v}" for k, v in counts.items() if k != 'ped_id_to_set']))
    import ipdb; ipdb.set_trace()

    np.savez('../AgentFormerSDD/datasets/jrdb_adjusted/poses_2d.npz', {'poses': combined_poses,
                                                                     'intrinsics': cam_intrinsics,
                                                                     'extrinsics': cam_extrinsics,
                                                                     'scores': scores,
                                                                     'cam_ids': cam_id})


if __name__ == '__main__':
    main()
