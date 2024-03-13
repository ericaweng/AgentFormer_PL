"""this script rotates a camera extrinsic matrix by a given angle counter-clockwise around the z-axis.
and it visualizes it to test if the function works."""
from pyquaternion import Quaternion
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import mkdir_if_missing


def rotate_extrinsic_matrix(quat, theta):
    """
    Create an extrinsic camera matrix with an additional rotation about the Z axis using pyquaternion.

    Parameters:
    - quat: A tuple containing the translation (x, y, z) and quaternion rotation (qx, qy, qz, qw).
    - theta: Additional rotation angle (in radians) about the Z axis (counterclockwise).

    Returns:
    - A 4x4 numpy array representing the extrinsic matrix with the additional rotation.
    """
    x, y, z, qx, qy, qz, qw = quat
    initial_rotation = Quaternion(qw, qx, qy, qz)

    # Create the additional rotation quaternion about the Z axis
    z_rotation = Quaternion(axis=[0, 0, 1], angle=theta)

    # Combine the rotations
    combined_rotation = z_rotation * initial_rotation

    # Convert combined quaternion to rotation matrix
    rotation_matrix = combined_rotation.rotation_matrix

    # Create the extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = [x, y, z]

    return extrinsic_matrix[:3]


def plot_camera_extrinsic(ax, extrinsic_matrix, color='r'):
    """
    Visualize the camera extrinsics as pyramids in a 3D scene.

    Parameters:
    - ax: The matplotlib axis to plot on.
    - extrinsic_matrix: The 4x4 extrinsic matrix representing the camera pose.
    - color: Color of the pyramid.
    """
    # Camera pyramid vertices in camera coordinates
    pyramid_vertices = np.array([
            [0, 0, 0, 1],  # Camera position
            [-0.5, -0.5, 1, 1],  # Bottom left
            [0.5, -0.5, 1, 1],  # Bottom right
            [0.5, 0.5, 1, 1],  # Top right
            [-0.5, 0.5, 1, 1]  # Top left
    ])

    # Transform pyramid vertices to world coordinates
    pyramid_vertices_world = pyramid_vertices @ extrinsic_matrix.T

    # Plot pyramid edges
    edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # Camera to corners
            (1, 2), (2, 3), (3, 4), (4, 1)  # Base edges
    ]

    for edge in edges:
        start, end = edge
        ax.plot(
                [pyramid_vertices_world[start, 0], pyramid_vertices_world[end, 0]],
                [pyramid_vertices_world[start, 1], pyramid_vertices_world[end, 1]],
                [pyramid_vertices_world[start, 2], pyramid_vertices_world[end, 2]],
                color=color
        )


def quat_to_extrinsic_matrix(quat):
    """ converts shape (7,) translation, rotation_as_quaternion representation to 4x3 extrinsic matrix """
    x, y, z, qx, qy, qz, qw = quat
    initial_rotation = Quaternion(qw, qx, qy, qz)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = initial_rotation.rotation_matrix
    extrinsic_matrix[:3, 3] = [x, y, z]
    return extrinsic_matrix[:3]

def round_to_orthogonal(matrix):
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(matrix)

    # Reconstruct the nearest orthogonal matrix
    orthogonal_matrix = np.dot(U, Vt)

    return orthogonal_matrix

def main():
    # Example camera extrinsic parameters
    quat = (1, 2, 3, 0, 0, 0, 1)  # Translation (1, 2, 3) and Quaternion rotation (0, 0, 0, 1)
    theta = np.radians(45)  # Additional rotation about Z axis (45 degrees)

    # now load real params from file

    # load camera calibration params for jrdb
    jrdb_calib_path = '../AgentFormerSDD/cameras.yaml'
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
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = T
        extrinsic_params[cam_num] = extrinsic_matrix[:3]

        try:
            q = Quaternion(matrix=R)
        except ValueError:
            q = Quaternion(matrix=round_to_orthogonal(R))
        # get R and T as length 7 vector (*xyz, *quaternion)
        extrinsic_params[cam_num] = np.array([*T, *q])

    fig = plt.figure(figsize=(15, 5))
    num_plots = 3
    for i in range(num_plots):
        ax = fig.add_subplot(int(f"1{num_plots}{i+1}"), projection='3d')
        for cam_num in [0, 2, 4, 6, 8]:
            extrinsics = extrinsic_params[cam_num]
            # plot_camera_extrinsic(ax, extrinsics, 'b')
            plot_camera_extrinsic(ax, quat_to_extrinsic_matrix(extrinsics), 'b')
            # plot_camera_extrinsic(ax, rotate_extrinsic_matrix(extrinsics, theta), 'r')

        # Setting the viewing angle for better visualization
        ax.view_init(elev=i*30, azim=i*30)

        # Setting labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    mkdir_if_missing('../viz/test_rotate_cam_extrinsic')
    plt.savefig('../viz/test_rotate_cam_extrinsic/camera.png')


if __name__ == '__main__':
    main()
