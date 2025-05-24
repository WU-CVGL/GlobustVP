from numba import njit
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix of a 3D vector.

    The resulting matrix [v]_x satisfies [v]_x @ w == np.cross(v, w) for any 3D vector w.

    Parameters:
        v : np.ndarray
            Input 3D vector, shape (3,).

    Returns:
        np.ndarray
            A 3x3 skew-symmetric matrix corresponding to the input vector.
    """
    assert v.shape == (3,), "Input vector must be 3-dimensional."
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ])


@njit
def axang2rotm(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Compute a 3x3 rotation matrix from a rotation axis and angle using Rodrigues' rotation formula.

    Parameters:
        axis : np.ndarray
            A 3D vector representing the rotation axis, shape (3,).
        theta : float
            Rotation angle in radians.

    Returns:
        R : np.ndarray
            A 3x3 rotation matrix corresponding to the given axis-angle representation.
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c

    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])
    return R


def normalize_lines(K: np.ndarray, lines_2D: np.ndarray) -> np.ndarray:
    """
    Normalize 2D line segments by applying the inverse of the camera intrinsic matrix.

    Parameters:
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        lines_2D : np.ndarray
            2D line segments in homogeneous image coordinates,
            where each column is [x1, y1, 1, x2, y2, 1]^T, shape (6, N).

    Returns:
        np.ndarray
            Normalized line endpoints in camera coordinates,
            where each column is [x1, y1, x2, y2]^T, shape (4, N).
    """
    K_inv = np.linalg.inv(K)

    # Apply inverse calibration to both endpoints
    pts1_h = K_inv @ lines_2D[0:3, :]  # First endpoint in homogeneous coords
    pts2_h = K_inv @ lines_2D[3:6, :]  # Second endpoint

    # Convert to inhomogeneous coordinates
    pts1 = pts1_h[:2, :] / pts1_h[2:, :]
    pts2 = pts2_h[:2, :] / pts2_h[2:, :]

    return np.vstack((pts1, pts2))


def project_3d_points(
    K: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    pts_world: np.ndarray
) -> np.ndarray:
    """
    Projects 3D world points to 2D image plane using camera intrinsics and extrinsics.

    Parameters:
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        R_wc : np.ndarray
            Rotation matrix (world to camera), shape (3, 3).
        t_wc : np.ndarray
            Translation vector (world to camera), shape (3,) or (3, 1).
        pts_world : np.ndarray
            3D world points, where each column represents a point [X, Y, Z], shape (3, N).

    Returns:
        pts_img : np.ndarray
            Projected 2D homogeneous image points,
            where each column represents a point in homogeneous coordinates [x, y, 1], shape (3, N).
    """
    # Ensure shape of translation vector
    t_wc = t_wc.reshape(3, 1)

    # Transform points to camera coordinate frame: X_c = R * X_w + t
    pts_cam = R_wc @ pts_world + t_wc  # shape: (3, N)

    # Project to image plane: x = K * X_c
    pts_img = K @ pts_cam  # shape: (3, N)

    # Normalize homogeneous coordinates
    pts_img /= pts_img[2, :]  # divide each column by its third element

    return pts_img


def project_and_add_noise(
    K: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    lines_3d: np.ndarray,
    num_lines: int,
    sigma: float
) -> np.ndarray:
    """
    Project 3D line segments to 2D image space and add Gaussian noise to endpoints.

    Parameters:
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        R_wc : np.ndarray
            Rotation matrix (world to camera), shape (3, 3).
        t_wc : np.ndarray
            Translation vector (world to camera), shape (3,) or (3, 1).
        lines_3d : np.ndarray
            3D line segments,
            where rows 0:3 represent start points and 3:6 represent end points, shape (6, N).
        num_lines : int
            Number of lines to project.
        sigma : float
            Standard deviation of Gaussian noise applied to each 2D coordinate.

    Returns:
        np.ndarray
            Noisy projected 2D lines, 
            where each column is [x1, y1, 1, x2, y2, 1]^T, shape (6, N).
    """
    # Project 3D start and end points
    pts_start_2d = project_3d_points(K, R_wc, t_wc, lines_3d[0:3, :])
    pts_end_2d = project_3d_points(K, R_wc, t_wc, lines_3d[3:6, :])

    # Add independent Gaussian noise to each endpoint coordinate
    noise_start = sigma * np.random.rand(2, num_lines)
    noise_end = sigma * np.random.rand(2, num_lines)
    pts_start_2d[:2, :] += noise_start
    pts_end_2d[:2, :] += noise_end

    # Stack to 6×N format: [x1, y1, 1, x2, y2, 1]^T
    return np.vstack((pts_start_2d, pts_end_2d))


def compute_backprojection_normals(lines_2D: np.ndarray) -> np.ndarray:
    """
    Compute 3D plane normals by back-projecting 2D image line segments.

    Parameters:
        lines_2D : np.ndarray
            Normalized 2D line segments,
            where each row is [x1, y1, x2, y2], shape (N, 4).

    Returns:
        para_lines : np.ndarray
            Unit-norm 3D normals of the back-projection planes, shape (N, 3).
    """
    assert lines_2D.shape[1] == 4, "Input must have shape (N, 4)."

    x1 = lines_2D[:, 0:2]  # shape (N, 2)
    x2 = lines_2D[:, 2:4]  # shape (N, 2)

    # Convert to homogeneous coordinates
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])  # shape (N, 3)
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])  # shape (N, 3)

    # Cross product to get plane normals
    para_lines = np.cross(x1_h, x2_h)
    para_lines /= np.linalg.norm(para_lines, axis=1, keepdims=True)

    return para_lines


def line_uncertainty(
    K: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray
) -> float:
    """
    Compute the uncertainty of a line defined by two image points, 
    propagated through the camera intrinsics to 3D space.

    Parameters:
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        start_point : np.ndarray
            Start point of the line, shape (2,).
        end_point : np.ndarray
            End point of the line, shape (2,).

    Returns:
        uncertainty : float
            Inverse trace-based uncertainty measure.
    """
    # Homogeneous coordinates
    p1_h = np.append(start_point, 1.0)
    p2_h = np.append(end_point, 1.0)

    # Isotropic 2D point covariance (lifted to 3D)
    Sigma_2D = 2 * np.eye(2)
    Sigma_h = np.zeros((3, 3))
    Sigma_h[:2, :2] = Sigma_2D

    # Transform to normalized camera coordinates
    K_inv = np.linalg.inv(K)
    Sigma_1_h = K_inv @ Sigma_h @ K_inv.T
    Sigma_2_h = Sigma_1_h  # Same covariance for both points

    # 3D line from cross product
    l_3d = np.cross(p1_h, p2_h)
    norm_l = np.linalg.norm(l_3d)
    l_3d_normalized = l_3d / norm_l

    # Covariance propagation for cross product
    Sigma_l = (
        skew(p2_h) @ Sigma_1_h @ skew(p2_h).T +
        skew(p1_h) @ Sigma_2_h @ skew(p1_h).T
    )

    # Jacobian projection matrix for normalization
    J = (np.eye(3) - np.outer(l_3d_normalized, l_3d_normalized)) / norm_l
    Sigma_l_normalized = J @ Sigma_l @ J.T

    # Uncertainty from trace
    uncertainty = 1.0 / np.trace(Sigma_l_normalized)
    return uncertainty


def compute_line_uncertainties(
    lines_2D: np.ndarray,
    K: np.ndarray,
    use_uncertainty: bool = False
) -> np.ndarray:
    """
    Compute normalized uncertainty weights for a set of 2D line segments.

    Parameters:
        lines_2D : np.ndarray
            Normalized 2D line segments,
            where each row is [x1, y1, x2, y2], shape (N, 4).
        K : np.ndarray
            Camera intrinsic matrix, shape (3, 3).
        use_uncertainty : bool, optional, default=False
            Whether to compute uncertainty based on geometry.

    Returns:
        uncertainty : np.ndarray
            Normalized uncertainty weights, shape (N, 1).
    """
    total_num = lines_2D.shape[0]
    uncertainty = np.ones((total_num, 1))

    if use_uncertainty:
        for i in range(total_num):
            u = line_uncertainty(K, lines_2D[i, :2], lines_2D[i, 2:4])
            uncertainty[i] = u
        # Normalize to [0.1, 0.3] for numerical stability
        min_u, max_u = np.min(uncertainty), np.max(uncertainty)
        uncertainty = 0.1 + (uncertainty - min_u) * 0.2 / (max_u - min_u + 1e-8)

    return uncertainty
