import cvxpy as cp
import numpy as np
from typing import Dict


def solve_sdp(
    C: np.ndarray,
    size_x: int,
    line_current_size: int,
    param: Dict
) -> np.ndarray:
    """
    Defines and returns an SDP solver for a problem using cvxpy.
    
    Parameters:
        C : np.ndarray
            Cost matrix, shape (size_x, size_x, 2).
        size_x : int
            Size of the SDP variable.
        line_current_size : int
            Number of lines considered in the SDP problem.
        param : Dict
            Dictionary of configuration parameters,
            including solver options and outlier constraints.
        
    Returns:
        X : np.ndarray
            Optimized variable, shape (size_x, size_x, 2).
    """
    block_size = 3

    if size_x > C.shape[0]:
        raise ValueError(f"⚠️ size_x={size_x} exceeds available C size ({C.shape[0]}).")

    # Truncate and clean C
    C = np.where(np.abs(C) < 1e-12, 0, C)
    C1, C2 = C[:, :, 0], C[:, :, 1]

    # Define the SDP variable X with 2 block matrices
    X1 = cp.Variable((size_x, size_x), PSD=True)
    X2 = cp.Variable((size_x, size_x), PSD=True)

    # List to hold the constraints
    constraints = []

    # Binary inlier/outlier constraints
    for i in range(line_current_size):
        idx = (i + 1) * block_size
        constraints += [
            X1[:3, idx:idx+3] == X1[idx:idx+3, idx:idx+3],
            X2[:3, idx:idx+3] == X2[idx:idx+3, idx:idx+3]
        ]

    # Trace == 1 constraints
    constraints += [
        cp.trace(X1[:3, :3]) == 1,
        cp.trace(X2[:3, :3]) == 1
    ]

    # Symmetry constraints between different line blocks
    for i in range(1, line_current_size):
        for j in range(i + 1, line_current_size + 1):
            i_idx, j_idx = i * block_size, j * block_size
            constraints += [
                X1[i_idx:i_idx+3, j_idx:j_idx+3] == X1[i_idx:i_idx+3, j_idx:j_idx+3].T,
                X2[i_idx:i_idx+3, j_idx:j_idx+3] == X2[i_idx:i_idx+3, j_idx:j_idx+3].T
            ]

    # Single "1" constraints in each column
    for i in range(line_current_size):
        idx = (i + 1) * block_size
        constraints.append(X1[:3, idx:idx+3] + X2[:3, idx:idx+3] == X1[:3, :3])

    # Redundant constraint
    constraints.append(X1[:3, :3] == X2[:3, :3])

    # Objective and solve
    objective = cp.Minimize(cp.trace(C1 @ X1) + cp.trace(C2 @ X2))
    problem = cp.Problem(objective, constraints)

    try:
        solver = param.get("solver", "MOSEK")
        solver_opts = param.get("solver_opts", {})
        problem.solve(solver=solver, warm_start=False, **solver_opts)

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"❌ Optimization failed. Status: {problem.status}")
    except Exception as e:
        raise ValueError(f"❌ Solver error: {e}")

    X = np.stack([X1.value, X2.value], axis=-1)

    return X