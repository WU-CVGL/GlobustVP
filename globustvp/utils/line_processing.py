import cv2
import numpy as np
from sklearn.cluster import KMeans


def filter_lines_by_length(lines: np.ndarray, min_length: float = 30.0) -> np.ndarray:
    """
    Filter 2D line segments by their Euclidean length.

    Parameters:
        lines : np.ndarray
            2D line segments, where each line is [x1, y1, x2, y2], shape (N, 1, 4).
        min_length : float, optional, default=30.0
            Minimum length threshold for keeping detected line segments.

    Returns:
        filtered_lines : np.ndarray
            Line segments that exceed the specified minimum length, shape (M, 1, 4).
    """
    if lines is None or len(lines) == 0:
        return np.empty((0, 1, 4), dtype=np.float32)

    diffs = lines[:, 0, 2:4] - lines[:, 0, 0:2]
    lengths = np.linalg.norm(diffs, axis=1)
    mask = lengths >= min_length

    return lines[mask]


def compute_line_angles(lines: np.ndarray) -> np.ndarray:
    """
    Compute the orientation angle (in radians) of each 2D line segment.

    Parameters:
        lines : np.ndarray
            2D line segments,
            where each line is [x1, y1, x2, y2], shape (N, 1, 4) or (N, 4).

    Returns:
        angles : np.ndarray
            Orientation angles in [0, π), shape (N, 1).
    """
    if lines is None or len(lines) == 0:
        return np.empty((0, 1), dtype=np.float32)
    
    if lines.ndim == 3:
        lines = lines[:, 0, :]  # Convert (N, 1, 4) to (N, 4)

    deltas = lines[:, 2:4] - lines[:, :2]  # (N, 2)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])  # (N,)
    angles = np.mod(angles, np.pi)  # Wrap to [0, π)

    return angles[:, np.newaxis]


def filter_lines_by_main_directions(
    lines: np.ndarray,
    angles: np.ndarray,
    k: int = 3,
    angle_threshold: float = np.deg2rad(10.0)
) -> np.ndarray:
    """
    Filter 2D lines by selecting only those close to dominant orientation clusters.

    Parameters:
        lines : np.ndarray
            2D line segments, where each line is [x1, y1, x2, y2], shape (N, 1, 4).
        angles : np.ndarray
            Line orientation angles (in radians), with values in [0, π), shape (N, 1).
        k : int, optional
            Number of dominant direction clusters. Default is 3.
        angle_threshold : float, optional, default=np.deg2rad(10.0)
            Angular threshold (in radians) to consider a line aligned with a cluster center.

    Returns:
        filtered_lines : np.ndarray
            Subset of input lines that are close to dominant directions, shape (M, 1, 4).
    """
    if lines.shape[0] == 0 or angles.shape[0] == 0:
        return np.empty((0, 1, 4), dtype=lines.dtype)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(angles)
    centers = kmeans.cluster_centers_

    mask = np.abs(angles - centers[labels]) < angle_threshold
    filtered_lines = lines[mask.ravel()]

    return filtered_lines


def detect_and_format_lines(
    gray_image: np.ndarray,
    min_length: float = 30.0,
    angle_cluster_k: int = 3
) -> np.ndarray:
    """
    Detect and filter 2D line segments from a grayscale image using LSD and angular clustering.

    Parameters:
        gray_image : np.ndarray
            Grayscale image, shape (H, W).
        min_length : float, optional, default=30.0
            Minimum length threshold for keeping detected line segments.
        angle_cluster_k : int, optional, default=3
            Number of dominant orientations to retain via KMeans clustering.

    Returns:
        lines_2D : np.ndarray
            Filtered 2D line segments in homogeneous format, shape (6, N).
    """
    assert gray_image.ndim == 2, "Input image must be grayscale."

    # LSD line segment detection
    lsd = cv2.createLineSegmentDetector()
    raw_lines = lsd.detect(gray_image)[0]
    if raw_lines is None or len(raw_lines) == 0:
        raise ValueError("❌ LSD failed to detect any lines.")

    # Filter short lines
    filtered = filter_lines_by_length(raw_lines, min_length=min_length)
    if filtered.size == 0:
        raise ValueError("❌ No lines remain after length filtering.")

    # Cluster by direction
    angles = compute_line_angles(filtered)
    clustered = filter_lines_by_main_directions(filtered, angles, k=angle_cluster_k)
    if clustered.size == 0:
        raise ValueError("❌ No lines remain after dominant direction filtering.")

    # Format lines to (6, N) homogeneous representation
    lines = clustered[:, 0, :]  # shape (N, 4)
    num_lines = lines.shape[0]
    lines_2D = np.ones((6, num_lines))
    lines_2D[0:2, :] = lines[:, 0:2].T
    lines_2D[3:5, :] = lines[:, 2:4].T

    return lines_2D
