from nuscenes.nuscenes import NuScenes
import argparse
import matplotlib.pyplot as plt


def calc_avg_percentage(datapath, version):
    percentages = []
    scene_total = 0
    for version in ['v1.0-mini', 'v1.0-trainval']:
        nusc = NuScenes(version=version, dataroot=datapath, verbose=True)

        scene_total += len(nusc.scene)

        for scene in nusc.scene:
            num_unique_peds = 0
            num_unique_children = 0
            unique_pedestrians = set()

            first_sample_token = scene['first_sample_token']
            current_token = first_sample_token

            while current_token:
                sample = nusc.get('sample', current_token)
                for ann_token in sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    category = ann['category_name']
                    instance_token = ann['instance_token']

                    if 'human.pedestrian' in category:
                        if instance_token not in unique_pedestrians:
                            unique_pedestrians.add(instance_token)
                            num_unique_peds += 1

                            if category == 'human.pedestrian.child':
                                num_unique_children += 1

                current_token = sample['next']

            if num_unique_peds > 0:
                percentage = (num_unique_children / num_unique_peds) * 100
                percentages.append(percentage)

    print(f"There are {scene_total} scenes, not 1000.")

    avg_percentage = sum(percentages) / len(percentages) if percentages else 0.0
    print(f"Average percentage of pedestrians that are children: {avg_percentage}%")

    # Plotting the bar chart
    plt.hist(percentages, bins=14, range=(1, 15), edgecolor='black')
    plt.xlabel('Percentage of Children among Pedestrians')
    plt.ylabel('Number of Scenes')
    plt.title('Distribution of Percentage of Children among Pedestrians in Scenes')
    plt.savefig('viz/children_stats.png')
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate and plot average percentage of child pedestrians in nuScenes dataset')
    parser.add_argument('-d', '--datapath', type=str, required=True, help='Path to nuScenes data')
    parser.add_argument('-v', '--version', type=str, default='v1.0-mini', help='Dataset version')

    args = parser.parse_args()
    calc_avg_percentage(args.datapath, args.version)
