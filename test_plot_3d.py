import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define start points and directions
    start_x, start_y, start_z = [0], [0], [0]  # Start points
    dir_x, dir_y, dir_z = [1], [1], [1]  # Direction vectors

    # Create the 3D quiver plot
    ax.quiver(start_x, start_y, start_z, dir_x, dir_y, dir_z, length=1, color='k')

    # Set the limits of the plot
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])

    plt.savefig("../viz/test_plot_3d.png")

if __name__ == "__main__":
    main()
