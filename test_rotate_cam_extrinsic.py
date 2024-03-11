"""this function rotates a camera extrinsic matrix by a given angle counter-clockwise around the z-axis.
and it visualizes it to test if the function works."""

x = np.array([0, 0, 0, 1])

import numpy as np


def create_extrinsic_matrix(quat, theta):
    """
    Create an extrinsic camera matrix with an additional rotation about the Z axis.

    Parameters:
    - x, y, z: Translations in the respective directions.
    - qx, qy, qz, qw: Quaternion components representing the rotation.
    - theta: Additional rotation angle (in radians) about the Z axis (counterclockwise).

    Returns:
    - A 4x4 numpy array representing the extrinsic matrix with the additional rotation.
    """
    x, y, z, qx, qy, qz, qw = quat

    # Create a rotation matrix from the quaternion
    quat_rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = quat_rotation.as_matrix()

    # Create the additional rotation matrix about the Z axis
    z_rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                  [np.sin(theta), np.cos(theta), 0],
                                  [0, 0, 1]])

    # Combine the rotations
    combined_rotation_matrix = np.dot(z_rotation_matrix, rotation_matrix)

    # Create the quaternion rotation matrix

    return extrinsic_matrix


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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

if __name__ == '__main__':
    # Example usage
    x, y, z = 1, 2, 3  # Translation
    qx, qy, qz, qw = 0, 0, 0, 1  # Quaternion (no rotation)
    theta = np.radians(45)  # Additional rotation about Z axis (45 degrees)

    extrinsic_matrix = create_extrinsic_matrix(x, y, z, qx, qy, qz, qw, theta)
    print(extrinsic_matrix)

    # Example usage
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create and plot extrinsic matrix with additional rotation
    extrinsic_matrix = create_extrinsic_matrix(x, y, z, qx, qy, qz, qw, theta)
    plot_camera_extrinsic(ax, extrinsic_matrix, 'r')

    # Setting the viewing angle for better visualization
    ax.view_init(elev=20, azim=30)

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
