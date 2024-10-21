""" reformats tbd dataset pedestrian track annotation txt files such that the start frame = 0 """

import os

# Function to adjust the first column and save to the same file
def adjust_first_column(file_path):
    # Open and read the content of the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Process each line and adjust the first column
    new_lines = []
    first_value = float(lines[0].split()[0])
    if first_value != 0.0:
        pass
    else:
        return None
    for line in lines:
        parts = line.split()
        parts[0] = str(float(parts[0]) - first_value)  # Adjust the first column to start at 0.0
        new_lines.append(" ".join(parts) + "\n")

    # Write to a temporary file for validation
    temp_file_path = file_path + '.temp'
    with open(temp_file_path, 'w') as f:
        f.writelines(new_lines)
    
    return temp_file_path

# Example usage for a single file
import glob
for file_path in glob.glob('datasets/tbd/2/Pedestrian_labels/3d_traj/*.txt'):
    print(f"{file_path=}")
    adjusted_file_path = adjust_first_column(file_path)
    # Replace the original file with the adjusted file
    if adjusted_file_path is not None:
        os.replace(adjusted_file_path, file_path)
