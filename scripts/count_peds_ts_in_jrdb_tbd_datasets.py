# count peds files in jrdb dataset
import os
import pandas as pd
import numpy as np


def get_agents_df(path, scene):
    # Construct the full file path
    file_path = os.path.join(path, f"{scene}.txt")

    # Initialize the dictionary to store the results
    data_dict = {}

    # Open and read the file
    with open(file_path, 'r') as file:
      for line in file:
        # Split the line into columns
        cols = line.strip().split()

        # Extract frame_id, ped_id, x, and y
        frame_id = int(float(cols[0]))
        ped_id = int(float(cols[1]))
        x = float(cols[2])
        y = float(cols[3])

        # Store the coordinates in a numpy array
        p = np.array([x, y])

        # Map the (frame_id, ped_id) to the numpy array
        data_dict[(frame_id, ped_id)] = {'p': p}

    agents_df = pd.DataFrame.from_dict(data_dict, orient='index').rename_axis(['timestep', 'id'])
    if len(agents_df.index.levels[0]) != agents_df.index.levels[0].max() + 1:
      agents_df.index = agents_df.index.set_levels(
          agents_df.index.levels[0] - agents_df.index.levels[0].min(),
          level=0,
      )

    return agents_df


def count_peds_in_jrdb_tbd_datasets():
  from data.tbd_split import get_tbd_split
  from data.jrdb_split import get_jrdb_hst_split

  tbd_labels_path = 'datasets/tbd/2/Pedestrian_labels/3d_traj'
  jrdb_labels_path = 'datasets/jrdb_adjusted_kiss-icp_lower_velodyne_mdr-1000_PiFeNet/'
  
  TRAIN, TEST, _ = get_tbd_split()
  num_peds = 0
  num_timesteps = 0
  for scene in TRAIN + TEST:
    tbd_df = get_agents_df(tbd_labels_path, scene).reset_index()
    num_peds += tbd_df['id'].nunique()
    total_length_of_peds = tbd_df.groupby('id').size().sum()
    num_timesteps += total_length_of_peds
    # count unique peds, avg trajectory length (num ts per ped)

  TRAIN_JRDB, TEST_JRDB,_ = get_jrdb_hst_split()
  num_timesteps_jrdb = 0  
  num_peds_jrdb = 0
  for scene in TRAIN_JRDB + TEST_JRDB:
    jrdb_df = get_agents_df(jrdb_labels_path, scene)
    num_peds_jrdb += jrdb_df.index.levels[1].nunique()
    # count unique peds
    total_length_of_peds = jrdb_df.groupby('id').size().sum()
    # avg trajectory length (num ts per ped)
    num_timesteps_jrdb += total_length_of_peds  

  print("total peds in tbd dataset: ", num_peds, "num peds in jrdb dataset: ", num_peds_jrdb)
  print("total timesteps in tbd dataset: ", num_timesteps, "num timesteps in jrdb dataset: ", num_timesteps_jrdb)
  print("avg trajectory length in tbd dataset: ", num_timesteps/num_peds, "avg trajectory length in jrdb dataset: ", num_timesteps_jrdb/num_peds_jrdb)   


if __name__ == '__main__':
  count_peds_in_jrdb_tbd_datasets()
  