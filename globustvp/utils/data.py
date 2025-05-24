import numpy as np
import random
from typing import List, Tuple, Dict

from .geometry import (
    normalize_lines,
    project_and_add_noise,
    compute_line_uncertainties,
    compute_backprojection_normals
)
from .plot import plot_lines


def generate_valid_line_segmentation(
    outlier_ratio: float,
    total_lines: int
) -> List[int]:
    """
    Randomly generate a valid segmentation of line counts into three groups,
    ensuring the third group retains sufficient inliers.

    Parameters:
        outlier_ratio : float
            Expected ratio of outliers among all lines.
        total_lines : int
            Total number of lines to be segmented.

    Returns:
        List[int]
            A list [l1, l2, l3] such that l1 + l2 + l3 == total_lines and
            the third group (l3) contains enough inliers after accounting for outliers
            (i.e., l3 - outliers_per_group >= 4).
    """
    while True:
        l1 = random.randint(30, 45)
        l2 = random.randint(25, 30)
        l3 = total_lines - l1 - l2
        min_required_inliers = 4
        max_outliers_per_group = total_lines * outlier_ratio / 3
        if l3 - max_outliers_per_group >= min_required_inliers:
            return [l1, l2, l3]


def generate_3d_lines(
    num_1: int,
    num_2: int,
    num_3: int,
    total_num: int,
    K: np.ndarray,
    gt_vp_noise: float,
    line_length: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D lines along three orthogonal directions with added Gaussian noise to the main direction.
    Lines are filtered to ensure their image projection lies within valid bounds.

    Parameters:
        num_1 : int
            Number of lines in the first (gravity-aligned) direction.
        num_2 : int
            Number of lines in the second orthogonal direction.
        num_3 : int
            Number of lines in the third orthogonal direction.
        total_num : int
            Total number of lines to generate (should equal num_1 + num_2 + num_3).
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        gt_vp_noise : float
            Standard deviation of noise to apply to the gravity direction.
        line_length : float
            Length of each 3D line.

    Returns:
        Lines_1_dir : np.ndarray
            3D line segments in direction 1, shape (6, num_1).
        Lines_2_dir : np.ndarray
            D line segments in direction 2, shape (6, num_1).
        Lines_3_dir : np.ndarray
            3D line segments in direction 3, shape (6, num_1).
        gt_vps : np.ndarray
            Ground truth vanishing points, shape (3, 3).
    """
    # Direction aligned with gravity (Y axis)
    first_dir = np.array([0.0, 1.0, 0.0])

    # Add noise to the first direction
    noise = gt_vp_noise * np.random.randn(3)
    first_dir_noisy = first_dir + noise
    first_dir_noisy /= np.linalg.norm(first_dir_noisy)

    # Generate orthogonal third direction
    third_dir = np.random.randn(3)
    third_dir -= np.dot(first_dir_noisy, third_dir) * first_dir_noisy
    third_dir /= np.linalg.norm(third_dir)

    # Second direction via right-hand rule
    second_dir = np.cross(third_dir, first_dir_noisy)

    gt_vps = np.vstack([first_dir_noisy, second_dir, third_dir])

    # Initialize storage
    lines_dir1 = np.zeros((6, num_1))
    lines_dir2 = np.zeros((6, num_2))
    lines_dir3 = np.zeros((6, num_3))

    # Projection bounds from camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_bound = cx / fx
    y_bound = cy / fy

    directions = [first_dir_noisy, second_dir, third_dir]
    counters = [0, 0, 0]
    targets = [num_1, num_2, num_3]
    buffers = [lines_dir1, lines_dir2, lines_dir3]

    while sum(counters) < total_num:
        rand_start = np.array([
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(4, 8)
        ])

        # Project start point and check if within image bounds
        proj_start = rand_start[:2] / rand_start[2]
        if not (-x_bound <= proj_start[0] <= x_bound and -y_bound <= proj_start[1] <= y_bound):
            continue

        # Determine which direction this line will take
        if counters[0] < targets[0]:
            dir_idx = 0
        elif counters[1] < targets[1]:
            dir_idx = 1
        elif counters[2] < targets[2]:
            dir_idx = 2
        else:
            break

        end_point = rand_start + directions[dir_idx] * line_length

        # Check projection of end point
        proj_end = end_point[:2] / end_point[2]
        if not (-x_bound <= proj_end[0] <= x_bound and -y_bound <= proj_end[1] <= y_bound):
            continue

        line_3d = np.hstack([rand_start, end_point])
        buffers[dir_idx][:, counters[dir_idx]] = line_3d
        counters[dir_idx] += 1

    return lines_dir1, lines_dir2, lines_dir3, gt_vps


def synthetic_data(
    outlier_ratio: float,
    param: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D line segments from a Manhattan world with three orthogonal 3D line directions.
    Includes inlier generation, endpoint noise addition (per-endpoint, per-dimension), projection,
    normalization, and optional outlier injection.

    Parameters:
        outlier_ratio : float
            Ratio of lines to be replaced with outliers (0.0-0.7).
        param : Dict 
            Configuration dictionary with keys:
                - "line_seg" : List[int]
                    Number of lines for each of the 3 orthogonal directions.
                - "vanishing_point_num" : int
                    Number of VP directions, typically 3.
                - "endpoint_noise" : float
                    Noise scale added to 2D projected endpoints.
                - "K" : np.ndarray
                    Camera intrinsic matrix, shape (3, 3).
                - "line_length" : float, optional, default=4
                    Length of 3D lines.
                - "gt_vp_noise" : float, optional, default=1e-3
                    VP perturbation.
                - "use_uncertainty" : bool, optional
                    Whether to use line-based uncertainty.
                - "uncertainty_scale" : float, optional, default=1e5
                    Scaling factor for uncertainty.
                - "debug" : bool, optional, default=False
                    Whether to visualize lines before and after normalization.

    Returns:
        gt_corrs : np.ndarray
            Ground truth line-VP associations, shape (3, total_num).
        all_2D_lines_norm : np.ndarray
            Normalized 2D line segments, shape (total_num, 4).
        para_lines : np.ndarray
            Normal vectors of the back-projection planes, shape (total_num, 3).
        uncertainty : np.ndarray
            Uncertainty weight for each line, shape (total_num, 1).
        gt_vps : np.ndarray
            Ground truth vanishing points, shape (3, 3).
    """
    num_1, num_2, num_3 = param["line_seg"]
    total_num = num_1 + num_2 + num_3   
    K = np.asarray(param["K"])
    Rw2c = np.eye(3)
    tw2c = np.zeros(3)
    sigma = param.get("endpoint_noise", 0.0)
    gt_vp_noise = param.get("gt_vp_noise", 1e-3)
    line_length = param.get("line_length", 4)

    # Generate 3D lines and ground truth VPs
    first_3d_lines, second_3d_lines, third_3d_lines, gt_vps = generate_3d_lines(
        num_1, num_2, num_3, total_num, K, gt_vp_noise, line_length
    )

    # Initialize GT correspondence matrix
    gt_corrs = np.zeros((param["vanishing_point_num"], total_num))
    gt_corrs[0, :num_1] = 1
    gt_corrs[1, num_1:num_1 + num_2] = 1
    gt_corrs[2, num_1 + num_2:] = 1

    first_2D_lines = project_and_add_noise(K, Rw2c, tw2c, first_3d_lines, num_1, sigma)
    second_2D_lines = project_and_add_noise(K, Rw2c, tw2c, second_3d_lines, num_2, sigma)
    third_2D_lines = project_and_add_noise(K, Rw2c, tw2c, third_3d_lines, num_3, sigma)

    # Inject outliers
    num_outliers = round(outlier_ratio * total_num)
    w, h = K[0, 2] * 2, K[1, 2] * 2
    for i in range(num_outliers):
        s = np.random.rand(2) * [w, h]
        e = np.random.rand(2) * [w, h]
        outlier = np.array([*s, 1, *e, 1])
        flag = i % 3
        idx = i // 3
        if flag == 0 and idx < num_1:
            first_2D_lines[:, num_1 - idx - 1] = outlier
            gt_corrs[0, num_1 - idx - 1] = 0
        elif flag == 1 and idx < num_2:
            second_2D_lines[:, num_2 - idx - 1] = outlier
            gt_corrs[1, num_1 + num_2 - idx - 1] = 0
        elif flag == 2 and idx < num_3:
            third_2D_lines[:, num_3 - idx - 1] = outlier
            gt_corrs[2, total_num - idx - 1] = 0

    # Optional debug plot (unnormalized)
    if param.get("debug", False):
        plot_lines(first_2D_lines, second_2D_lines,third_2D_lines, num_outliers // 3)

    # Normalize projected lines
    first_2D_lines_norm = normalize_lines(K, first_2D_lines)
    second_2D_lines_norm = normalize_lines(K, second_2D_lines)
    third_2D_lines_norm = normalize_lines(K, third_2D_lines)
    all_2D_lines_norm = np.concatenate([first_2D_lines_norm, second_2D_lines_norm, third_2D_lines_norm], axis=1).T

    # Optional debug plot (normalized)
    if param.get("debug", False):
        plot_lines(first_2D_lines_norm, second_2D_lines_norm, third_2D_lines_norm, num_outliers // 3, normalized=True)
    
    # Back-projection plane normals
    para_lines = compute_backprojection_normals(all_2D_lines_norm)

    # Optional uncertainty estimation
    uncertainty = compute_line_uncertainties(all_2D_lines_norm, K, param.get("use_uncertainty", False))

    return gt_corrs, all_2D_lines_norm, para_lines, uncertainty, gt_vps