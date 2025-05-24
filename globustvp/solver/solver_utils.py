import numpy as np
from numba import njit
from typing import Dict, List
from scipy.linalg import null_space

from ..utils.geometry import axang2rotm


def generate_bin(all_2D_lines: np.ndarray, param: Dict) -> List[int]:
    """
    Generate a histogram of line directions and return the line IDs from the dominant direction bin,
    excluding cases that are too close to the most dominant bin.

    Parameters:
        all_2D_lines : np.ndarray
            Normalized 2D line segments,
            where each row is [x1, y1, x2, y2], shape (N, 4).
        param : Dict
            Dictionary of configuration parameters,
            including number of direction bins.

    Returns:
        largest_bin_idxes : List[int]
            List of line indices that fall into the dominant direction bin,
            excluding bins that are too similar (i.e., not sufficiently angularly separated).
    """
    lines = all_2D_lines
    num_of_lines = lines.shape[0]

    histogram_len = param.get("histogram_len", 100) # Number of bins for the histogram
    dir_histogram = np.zeros((histogram_len, 2))  # Holds counts of lines in each bin and bin index
    dir_histogram[:, 1] = np.arange(1, histogram_len + 1)
    
    dir_cell = [[] for _ in range(histogram_len)]  # List of bins, each holding the line IDs
    resolution = np.pi / histogram_len  # Bin resolution in radians

    # Calculate direction of each line and assign it to a bin
    dx = lines[:, 2] - lines[:, 0]  # dx = ex - sx
    dy = lines[:, 3] - lines[:, 1]  # dy = ey - sy
    for line_id in range(num_of_lines):
        # Calculate the direction of the line (in radians)
        if dx[line_id] == 0:
            direction = np.pi / 2
        else:
            direction = np.arctan(dy[line_id]/dx[line_id])  # atan(dy/dx), -pi/2 to pi/2
        bin_id = max(int(np.ceil((direction + np.pi / 2) / resolution)), 1) - 1  # Bin index
        dir_histogram[bin_id, 0] += 1  # Increment the count for this bin
        dir_cell[bin_id].append(line_id)  # Append the line ID to the bin (1-based index)

    # Sort histogram bins by line count
    dir_histogram = dir_histogram[dir_histogram[:, 0].argsort()]

    peak_id1 = int(dir_histogram[-1, 1])  # The most populous bin (peak)

    # Select the second peak that is sufficiently different from the first peak
    for i in range(histogram_len):
        test_id = int(dir_histogram[-(i + 1), 1])
        if abs(test_id - peak_id1) >= 4:  # Ensure a significant difference between peaks
            largest_bin_idxes = dir_cell[peak_id1 - 1]  # Return the lines in the first peak bin
            break

    return largest_bin_idxes


def check_eig(W: np.ndarray, param: Dict) -> List[bool]:
    """
    Check whether the ratio between the largest and second-largest eigenvalues
    of each 3x3 matrix slice in an SDP solution exceeds a given threshold.

    Parameters:
        W : np.ndarray
            SDP solution, shape (N, N, 2).
        param : Dict
            Dictionary of configuration parameters,
            including number of ratio threshold.

    Returns:
        pass_flags : List[bool]
            A list of boolean flags where each entry indicates whether the corresponding matrix passed the eigenvalue ratio check.
    """
    threshold = param.get("eigen_threshold", 9)
    pass_flags = []

    for i in range(W.shape[2]):
        # Calculate the eigenvalues of the 3x3 matrix (W[:3, :3, i])
        eigvals = np.linalg.eigvalsh(W[:3, :3, i])
        largest, second_largest = eigvals[-1], eigvals[-2]  # eigenvals is sorted in ascending order
        ratio = largest / second_largest if second_largest != 0 else np.inf
        pass_flags.append(ratio > threshold)

    return pass_flags


def recover_vp(W: np.ndarray) -> np.ndarray:
    """
    Recover the vanishing point from an SDP solution using Singular Value Decomposition (SVD).

    Parameters:
        W : np.ndarray
            SDP solution, shape (N, N, 2).

    Returns:
        estimated_vp : np.ndarray
            Estimated vanishing point derived from the first singular vector, shape (3,).
    """
    # Perform SVD on the first 3x3 matrix slice from W
    _, S, Vt = np.linalg.svd(W[:3, :3, 0], full_matrices=False)  # Use compact SVD
    estimated_vp = S[0] * Vt[0]  # First singular value and its corresponding vector

    return estimated_vp


@njit
def _angle_mask(
    normals: np.ndarray,
    n1_rot: np.ndarray,
    n2_rot: np.ndarray
) -> np.ndarray:
    """
    Efficiently selects normal vectors that are approximately orthogonal
    to either n1_rot or n2_rot using a fast angle check.

    Parameters:
        normals : np.ndarray
            Normal vectors, shape (N, 3).
        n1_rot : np.ndarray
            A rotated perpendicular vector, shape (3,).
        n2_rot : np.ndarray
            Another rotated perpendicular vector, shape (3,).

    Returns:
        inliers : np.ndarray
            Indices of normals that are close to 90 degrees from
            either n1_rot or n2_rot.
    """
    inliers = []
    for i in range(normals.shape[0]):
        n = normals[i]
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8:
            continue

        # Compute cosine similarity to each direction
        cos_n1 = np.dot(n, n1_rot) / norm_n
        cos_n2 = np.dot(n, n2_rot) / norm_n

        # Clamp cosine values to avoid invalid input to arccos
        cos_n1 = min(1.0, max(-1.0, cos_n1))
        cos_n2 = min(1.0, max(-1.0, cos_n2))

        # Convert to angles in degrees
        angle1 = np.arccos(cos_n1) * 180.0 / np.pi
        angle2 = np.arccos(cos_n2) * 180.0 / np.pi

        # Check if either angle is approximately 90 degrees
        if abs(angle1 - 90.0) <= 0.5 or abs(angle2 - 90.0) <= 0.5:
            inliers.append(i)

    return inliers


def find_peak_intervals(
    d: np.ndarray,
    normals: np.ndarray
) -> List[int]:
    """
    Identify dominant normal directions that are nearly orthogonal to two rotating reference vectors.

    Parameters:
        d : np.ndarray
            Rotation axis, shape (3,).
        normals : np.ndarray
            Array of normal vectors, shape (N, 3).

    Returns:
        peak_line_idx : List[int]
            Indices of normals within the peak bin.
    """
    d = d / np.linalg.norm(d)
    null_basis = null_space(d.reshape(1, -1))  # (3,2)
    n1_base = null_basis[:, 0]

    num_bins = 90
    bin_counts = np.zeros(num_bins, dtype=np.int32)
    bin_indices = [[] for _ in range(num_bins)]
    bin_n1 = np.zeros((3, num_bins))
    bin_n2 = np.zeros((3, num_bins))

    for angle_deg in range(num_bins):
        theta = np.deg2rad(angle_deg)
        R = axang2rotm(d, theta)

        n1_rot = R @ n1_base
        n1_rot /= np.linalg.norm(n1_rot)
        n2_rot = np.cross(d, n1_rot)
        n2_rot /= np.linalg.norm(n2_rot)

        inliers = _angle_mask(normals, n1_rot, n2_rot)

        bin_counts[angle_deg] = len(inliers)
        bin_indices[angle_deg] = inliers
        bin_n1[:, angle_deg] = n1_rot
        bin_n2[:, angle_deg] = n2_rot

    peak_idx = np.argmax(bin_counts)
    peak_line_idx = bin_indices[peak_idx]

    return peak_line_idx