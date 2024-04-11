import numpy as np


def nearest_neighbor(src, dst):
    """
    For each point in the source, find the closest point in the destination.
    """
    indices = []
    for s in src:
        min_dist = np.inf
        index = -1
        for i, d in enumerate(dst):
            dist = np.linalg.norm(s - d)
            if dist < min_dist:
                min_dist = dist
                index = i
        indices.append(index)
    return indices


def kiss_icp(src, dst, max_iterations=10, tolerance=1e-3):
    """
    Simple implementation of the KISS-ICP algorithm.
    """
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    for _ in range(max_iterations):
        indices = nearest_neighbor(src_centered, dst_centered)
        matched_points = np.array([dst_centered[i] for i in indices])

        # Compute the transformation between src and the matched points in dst
        H = np.dot(src_centered.T, matched_points)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        t = dst_centroid - np.dot(R, src_centroid)

        # Apply the transformation
        src = np.dot(src, R.T) + t
        src_centered = src - src_centroid

        # Check for convergence
        if np.abs(np.sum(matched_points - src_centered)) < tolerance:
            break

    return src, R, t


# Example usage:
src_points = np.random.rand(10, 2)  # Source point cloud
dst_points = np.random.rand(10, 2)  # Destination point cloud

# Align the source point cloud to the destination point cloud
aligned_src, rotation, translation = kiss_icp(src_points, dst_points)
print("Aligned Source Points:", aligned_src)
