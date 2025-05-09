import numpy as np
from typing import Dict, Tuple, List
from scipy.optimize import linear_sum_assignment


def initialize_result_dict(
    n_ratio: int,
    n_iter: int
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    """
    Initialize a dictionary to store results for each ratio and iteration.

    Parameters:
        n_ratio : int
            Number of different outlier ratios.
        n_iter : int
            Number of iterations for each ratio.

    Returns:
        result : Dict[int, Dict[str, List[np.ndarray]]]
            Nested dictionary where:
                - Outer key: ratio index (int).
                - Inner key: result type (e.g., "accuracy", "gt_vps").
                - Values: arrays (for scalars) or lists (for structured outputs) storing results across iterations.
    """
    result = {}
    scalar_keys = {"time", "outlier_ratio", "precision", "recall", "f1_score"}
    
    # Initialize result dictionary with zeros for scalars and None for lists
    for out_idx in range(n_ratio):
        result[out_idx] = {key: np.zeros(n_iter) for key in scalar_keys}
        
        # Initialize lists for specific keys
        for key in ["gt_vps", "est_vps", "gt_corrs", "est_corrs", "parallel_line"]:
            result[out_idx][key] = [None] * n_iter
    
    return result


def evaluate_vp_matching(
    gt_corrs: np.ndarray, 
    est_corrs: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Evaluate vanishing point (VP) estimation quality by comparing ground truth and estimated
    vanishing points and their correspondence maps.

    Parameters:
        gt_corrs : np.ndarray
            Gound truth line-VP associations, shape (3, N).
        est_corrs : np.ndarray
            Estimated line-VP associations, shape (3, N).

    Returns:
        precision : float
            Precision of positive (true) correspondences.
        recall : float
            Recall of positive (true) correspondences.
        f1_score : float
            F1 score of correspondence classification.
    """
    assert gt_corrs.shape[1] == est_corrs.shape[1], \
        "The number of ground-truth and estimated line-to-VP associations must be same."
         
    # Match estimated groups to ground-truth groups by minimizing difference
    diff_map = np.linalg.norm(gt_corrs[:, None, :] - est_corrs[None, :, :], axis=2)
    best_match_ids = np.argmin(diff_map, axis=1)

    # Get reordered estimated correspondence matrix
    aligned_est_corrs = est_corrs[best_match_ids]

    # Compute confusion matrix components
    TP = np.sum((gt_corrs == 1) & (aligned_est_corrs == 1))
    FP = np.sum((gt_corrs == 0) & (aligned_est_corrs == 1))
    FN = np.sum((gt_corrs == 1) & (aligned_est_corrs == 0))

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1_score


def compute_vp_angular_errors(est_vps: np.ndarray, gt_vps: np.ndarray) -> List[float]:
    """
    Compute angular errors (in degrees) between estimated and ground truth vanishing points,
    using optimal matching via the Hungarian algorithm.

    Parameters:
        est_vps : np.ndarray
            Estimated vanishing points, shape (3, 3).
        gt_vps : np.ndarray
            Ground truth vanishing points, shape (3, 3).

    Returns:
        angular_errors : List[int]
            List of angular differences (in degrees) between matched VP pairs.
    """
    # Compute cosine similarities (absolute dot product between column unit vectors)
    similarity_matrix = np.abs(gt_vps.T @ est_vps)

    # Apply Hungarian algorithm for optimal matching
    _, col_ind = linear_sum_assignment(-similarity_matrix)
    est_matched = est_vps[:, col_ind]

    # Compute angular errors
    angular_errors = []
    for i in range(gt_vps.shape[1]):
        gt_vec = gt_vps[:, i]
        est_vec = est_matched[:, i]
        cos_theta = np.abs(np.dot(gt_vec, est_vec)) / (np.linalg.norm(gt_vec) * np.linalg.norm(est_vec))
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angular_errors.append(angle_deg)

    return angular_errors