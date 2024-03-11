import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box, Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import numpy as np
import cv2
import os

def visualize_children_videos(datapath, version, num_videos=10):
    nusc = NuScenes(version=version, dataroot=datapath, verbose=True)
    os.makedirs('viz', exist_ok=True)

    video_count = 0

    for scene in nusc.scene:
        if video_count >= num_videos:
            break

        has_child = False
        current_token = scene['first_sample_token']

        output_file = os.path.join(args.output_root, f"{scene['name']}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = None

        while current_token:
            sample = nusc.get('sample', current_token)
            camera_token = sample['data']['CAM_FRONT']
            camera_data = nusc.get('sample_data', camera_token)
            camera_filepath = os.path.join(datapath, camera_data['filename'])

            img = cv2.imread(camera_filepath)
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                category = ann['category_name']

                if 'human.pedestrian.child' in category:
                    if video_writer is None:
                        h, w, _ = img.shape
                        video_writer = cv2.VideoWriter(video_name, fourcc, 2.0, (w, h))

                    has_child = True
                    sample_data = nusc.get('sample_data', camera_token)
                    _, boxes, _ = nusc.get_sample_data(camera_token, box_vis_level=BoxVisibility.ANY,
                                                       selected_anntokens=[ann_token])
                    calibrated_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                    camera_intrinsic = calibrated_sensor_data['camera_intrinsic']
                    for box in boxes:
                        corners = view_points(box.corners(), np.array(camera_intrinsic), normalize=True)[:2, :]
                        min_x = min(corners[0])
                        max_x = max(corners[0])
                        min_y = min(corners[1])
                        max_y = max(corners[1])
                        cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

                    video_writer.write(img)
                    current_token = sample['next']

        if video_writer is not None:
            video_writer.release()

        if has_child:
            video_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize videos with children and bounding boxes')
    parser.add_argument('-d', '--datapath', type=str, required=True, help='Path to nuScenes data')
    parser.add_argument('-v', '--version', type=str, default='v1.0-mini', help='Dataset version')

    args = parser.parse_args()
    visualize_children_videos(args.datapath, args.version)
