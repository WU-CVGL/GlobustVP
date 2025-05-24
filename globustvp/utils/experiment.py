from typing import Dict
import time
from globustvp.solver.core import globustvp
from globustvp.utils.data import generate_valid_line_segmentation, synthetic_data
from globustvp.utils.metrics import initialize_result_dict, evaluate_vp_matching


def run_single_experiment(param: Dict, outlier_ratio: float, iter_idx: int) -> Dict:
    """
    Run a single experiment iteration with the given outlier ratio.

    Parameters:
        param : Dict
            Dictionary of configuration parameters.
        outlier_ratio : float
            Ratio of outliers in the current experiment.
        iter_idx : int
            Iteration index (for logging/debugging).

    Returns:
        Dict
            All evaluation metrics and intermediate results.
    """
    print(f"\n--- Iteration {iter_idx + 1} | Outlier Ratio: {outlier_ratio:.2f} ---")

    line_seg = generate_valid_line_segmentation(outlier_ratio, param["line_num"])
    inlier_seg = [l - l * outlier_ratio for l in line_seg]

    param.update({
        "line_inlier_num": param["line_num"] * (1 - outlier_ratio),
        "line_seg": line_seg,
        "line_inlier_seg": inlier_seg
    })

    gt_corrs, all_2D_lines, para_lines, uncertainty, gt_vps = synthetic_data(outlier_ratio, param)
    print("GT Manhattan Frame:\n", gt_vps)

    t_start = time.time()
    _, est_vps, est_corrs = globustvp(all_2D_lines, para_lines, uncertainty, param)
    t_end = time.time()

    print("Estimated Manhattan Frame:\n", est_vps)

    prec, rec, f1 = evaluate_vp_matching(gt_corrs, est_corrs)
    print(f"⏱ Time: {t_end - t_start:.4f}s | Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    return {
        "time": t_end - t_start,
        "outlier_ratio": outlier_ratio,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "gt_vps": gt_vps,
        "est_vps": est_vps,
        "gt_corrs": gt_corrs,
        "est_corrs": est_corrs,
        "parallel_line": para_lines
    }


def run_experiments(param: Dict) -> Dict:
    """
    Run synthetic vanishing point estimation experiments across different outlier ratios.

    Parameters:
        param : Dict
            Dictionary of experiment configuration parameters.

    Returns:
        Dict
            Experiment results for each setting.
    """
    num_ratios = len(param["outlier_ratio"])
    num_iters = param["iteration"]
    results = initialize_result_dict(num_ratios, num_iters)

    for out_idx, outlier_ratio in enumerate(param["outlier_ratio"]):
        for iter_idx in range(num_iters):
            result = run_single_experiment(param, outlier_ratio, iter_idx)
            for key, value in result.items():
                results[out_idx][key][iter_idx] = value

    print("\n✅ All experiments completed.")
    return results