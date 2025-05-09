import numpy as np
from typing import Dict, Tuple

from .sdp_solver import solve_sdp
from .solver_utils import check_eig, recover_vp, generate_bin, find_peak_intervals


def globustvp(
    all_2D_lines: np.ndarray,
    para_lines: np.ndarray,
    uncertainty: np.ndarray,
    param: Dict
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Fast SDP-based solver to estimate vanishing points from a set of parallel lines.

    Parameters:
        all_2D_lines: np.ndarray
            Normalized 2D line segments, shape (N, 4).
        para_lines : np.ndarray
            Normal vectors of the back-projection planes, shape (N, 3).
        uncertainty : np.ndarray
            Uncertainty weight for each line, shape (N, 1).
        param : Dict
            Dictionary of configuration parameters.

    Returns:
        status : bool
            Boolean flag indicating successful solution.
        est_vps : np.ndarray
            Estimated vanishing points, shape (3, 3).
        est_corrs : np.ndarray
            Estimated line-VP associations, shape (3, N).
    """
    def reset():
        return [], [], np.ones(param["line_num"], dtype=bool), np.arange(param["line_num"])
    
    # Initialize variables
    est_vps, est_corrs, line_id_pool, reverse_pool = reset()
    is_fast_solver = param.get("is_fast_solver", False)

    if is_fast_solver:
        largest_bin_idxes = generate_bin(all_2D_lines, param)

    while True:
        active_lines = para_lines[line_id_pool]
        active_uncertainty = uncertainty[line_id_pool]
        active_size = len(active_lines)

        # Not enough lines left to proceed
        if active_size < 3:
            if len(est_vps) == param["vanishing_point_num"]:
                return True, np.array(est_vps), np.array(est_corrs)
            else:
                est_vps, est_corrs, line_id_pool, reverse_pool = reset()
                continue

        # Sampling
        if not est_vps:  # First vanishing point
            if is_fast_solver:
                sample_ids = np.random.choice(
                    largest_bin_idxes, 
                    min(len(largest_bin_idxes), param["sample_line_num"]),
                    replace=False
                )
            else:
                sample_ids = np.random.choice(
                    active_size, 
                    param["sample_line_num"],
                    replace=False
                )
        else:
            sample_ids = np.random.choice(
                active_size,
                min(param["sample_line_num"], active_size),
                replace=False
            )
        sampled_lines = active_lines[sample_ids]
        sampled_uncertainty = active_uncertainty[sample_ids]
        sample_size = len(sampled_lines)

        # Construct cost matrix C
        C = np.zeros((3 * sample_size + 3, 3 * sample_size + 3, 2))
        for i, (line, unc) in enumerate(zip(sampled_lines, sampled_uncertainty)):
            outer = np.outer(line, line)
            idx = (i + 1) * 3
            C[:3, idx:idx+3, 0] = 0.5 * unc * outer
            C[idx:idx+3, :3, 0] = C[:3, idx:idx+3, 0]
            C[:3, idx:idx+3, 1] = 0.5 * unc * param["c"]**2 * np.eye(3)
            C[idx:idx+3, :3, 1] = C[:3, idx:idx+3, 1]

        # Solve the SDP
        X = solve_sdp(C, C.shape[0], sample_size, param)

        # Check eigenvalues
        if not all(check_eig(X, param)):
            continue

        # Recover results
        est_vp = recover_vp(X)

        # Validate angular orthogonality
        if len(est_vps) == 1:
            angle = np.degrees(np.arccos(est_vps[0] @ est_vp))
            if np.abs(90 - angle) > (90 - np.degrees(np.arccos(param["c"]))):
                est_vps, est_corrs, line_id_pool, reverse_pool = reset()
                continue
        elif len(est_vps) == 2  and (
            np.abs(est_vps[0] @ est_vp) > 1e-3 or 
            np.abs(est_vps[1] @ est_vp) > 1e-3
        ):
            est_vp = np.cross(est_vps[0], est_vps[1])

        est_vps.append(est_vp)

        # Update line-to-vp correspondences
        all_lines = para_lines[line_id_pool]
        corr_line_idx = np.where(np.abs(np.dot(all_lines, est_vp)) < param["c"])[0]
        original_ids = reverse_pool[corr_line_idx]

        if is_fast_solver:
            if len(est_vps) == 1:
                peak_ids = find_peak_intervals(est_vp, para_lines)
                line_id_pool[:] = False
                line_id_pool[peak_ids] = True
            
        line_id_pool[original_ids] = False
        reverse_pool = np.where(line_id_pool)[0]
        corr = np.zeros(param["line_num"])
        corr[original_ids] = 1
        est_corrs.append(corr)

        # Check the required number of vanishing points is reached
        if len(est_vps) == 3:
            if np.linalg.det(est_vps) < 0:
                est_vps[0] = -est_vps[0]
            U, _, Vt = np.linalg.svd(est_vps, full_matrices=True)
            est_vps = U @ Vt

            if np.allclose(est_vps @ est_vps.T, np.eye(3), atol=1e-6):
                return True, np.array(est_vps), np.array(est_corrs)
            else:
                est_vps, est_corrs, line_id_pool, reverse_pool = reset()
                continue