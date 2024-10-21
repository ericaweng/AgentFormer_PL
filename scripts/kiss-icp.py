""" run kiss-icp on all files in a directory """

import os
from multiprocessing import Pool


def process_file(file_path):
    """
    Function to run the kiss_icp_pipeline command on a file.
    """
    if os.path.isdir(file_path):
        command = f"kiss_icp_pipeline {file_path}"
        os.system(command)  # Using os.system to execute the command, replace with subprocess for better control


def main():
    directory = "datasets/jrdb/test/pointclouds/lower_velodyne/"

    # Get all files in the directory
    files = [os.path.join(directory, file) for file in os.listdir(directory)]

    # Set up a pool of worker processes
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_file, files)


if __name__ == "__main__":
    main()
