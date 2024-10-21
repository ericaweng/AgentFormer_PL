""" remove pedestrians that are mislabelled, as well as combining pedestrians with different labels that are the same person """

import glob
import numpy as np
from traj_toolkit.data_scripts.tbd_interesting_scenes import SAME_PEDESTRIAN_GROUPS, REMOVE_PEDESTRIANS


def process_pedestrians(arr, same_pedestrian_groups, remove_pedestrians):
    # Remove rows where the second column has values in remove_pedestrians
    arr = arr[~np.isin(arr[:, 1], remove_pedestrians)]
    
    # For each group in same_pedestrian_groups, set all second column values to the first element in the group
    for group in same_pedestrian_groups:
        first_value = min(group)
        mask = np.isin(arr[:, 1], group)  # Create a mask for rows where the second column is in the group
        arr[mask, 1] = first_value  # Set the second column value to the first element of the group

    # Ensure there is only one row per frame and pedestrian, otherwise delete the duplicates
    # We use np.unique with the 'return_index' option to keep only the first occurrence of each combination
    unique_rows, unique_indices = np.unique(arr[:, :2], axis=0, return_index=True)
    
    # Use the unique indices to filter the array
    arr = arr[unique_indices]
    
    return arr


def save_array_custom_format(filename, arr):
    # Convert the first two columns to integers and keep the last two as floats with 10 decimal precision
    fmt = ['%d', '%d', '%.10f', '%.10f']
    
    # Save the array to a txt file with the specified format
    np.savetxt(filename, arr, fmt=fmt, delimiter=' ')


for file_path in glob.glob('datasets/tbd/2/Pedestrian_labels/3d_traj/*.txt'):
    print(f"{file_path=}")
    arr = np.genfromtxt(file_path, delimiter=' ')
    new_arr = process_pedestrians(arr, SAME_PEDESTRIAN_GROUPS, REMOVE_PEDESTRIANS)
    # save new arr to file_path
    save_array_custom_format(file_path+'.temp', arr)
    