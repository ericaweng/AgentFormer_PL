# Importing necessary libraries
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# Initialize NuScenes and NuScenesMap instances (example assumes NuScenes dataset is available)
nusc = NuScenes(version='v1.0-mini', dataroot='path_to_nuscenes_data', verbose=True)
nusc_map = NuScenesMap(dataroot='path_to_nuscenes_data', map_name='singapore-onenorth')

# Example: Visualizing an annotated scene with map overlay
# Selecting a scene
scene = nusc.scene[0]  # Replace with the desired scene index

# Extracting the map location from the scene
log = nusc.get('log', scene['log_token'])
map_location = log['location']

# Plotting the map
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
nusc_map.render_map_patch(ax, nusc_map.extract_map_patch([-500, 1500, -1000, 1000], map_location),
                          layers=['drivable_area', 'lane', 'ped_crossing', 'walkway'])

# Displaying the plot
plt.show()