from nuscenes.nuscenes import NuScenes
import argparse


def count_child_pedestrian_stats(datapath, version):
    # Initialize the NuScenes class
    nusc = NuScenes(version=version, dataroot=datapath, verbose=True)

    total_tracks = 0
    child_tracks = 0
    child_time = 0.0

    # Loop through all the scenes
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        current_token = first_sample_token
        scene_duration = 0.0
        prev_time_stamp = 0.0

        while current_token:
            sample = nusc.get('sample', current_token)
            current_time_stamp = sample['timestamp'] / 1e6  # Convert to seconds

            # Calculate time difference between the current and previous sample
            if prev_time_stamp:
                time_diff = current_time_stamp - prev_time_stamp

            import ipdb; ipdb.set_trace()

            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                category = ann['category_name']
                print(f"category: {category}")

                total_tracks += 1

                if category == 'human.pedestrian.child':
                    child_tracks += 1
                    if prev_time_stamp:
                        scene_duration += time_diff

            prev_time_stamp = current_time_stamp
            current_token = sample['next']

        child_time += scene_duration

    # Calculate statistics
    if total_tracks == 0:
        percentage_child_tracks = 0.0
    else:
        percentage_child_tracks = (child_tracks / total_tracks) * 100

    print(f"Percentage of child pedestrian tracks: {percentage_child_tracks}%")
    print(f"Total time of child pedestrian tracks: {child_time} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate child pedestrian stats in nuScenes dataset')
    parser.add_argument('--datapath', '-d', type=str, required=True, help='Path to nuScenes data')
    parser.add_argument('--version', '-v', type=str, default='v1.0-mini', help='Dataset version')

    args = parser.parse_args()

    count_child_pedestrian_stats(args.datapath, args.version)
