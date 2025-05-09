# import argparse
# import json
# import numpy as np
# import time

# from .solver.core import globustvp
# from .utils.data import generate_valid_line_segmentation, synthetic_data
# from .utils.metrics import initialize_result_dict, evaluate_vp_matching
# from .utils.io import save_results
# from .utils.plot import plot_metrics_boxplot


# # ----------------------------
# # Main Run Function
# # ----------------------------
# def run():
#     results = initialize_result_dict(len(param["outlier_ratio"]), param["iteration"])

#     for out_idx, outlier_ratio in enumerate(param["outlier_ratio"]):
#         print(f"\n🔵 Outlier ratio is {outlier_ratio*100:.0f}%")

#         for iter_i in range(param["iteration"]):
#             print(f"\n--- Iteration {iter_i + 1} ---")

#             line_seg = generate_valid_line_segmentation(outlier_ratio, param["line_num"])
#             inlier_seg = [l - l * outlier_ratio for l in line_seg]

#             param.update({
#                 "line_inlier_num": param["line_num"] * (1 - outlier_ratio),
#                 "line_seg": line_seg,
#                 "line_inlier_seg": inlier_seg
#             })

#             gt_corrs, all_2D_lines, para_lines, uncertainty, gt_vps = \
#                 synthetic_data(outlier_ratio, param)

#             print("GT Manhattan frame:\n", gt_vps)

#             t_start = time.time()
#             _, est_vps, est_corrs = globustvp(all_2D_lines, para_lines, uncertainty, param)
#             t_end = time.time()

#             print("Estimated Manhattan frame:\n", est_vps)

#             prec, rec, f1 = evaluate_vp_matching(gt_corrs, est_corrs)

#             print(f"⏱  Elapsed: {t_end - t_start:.4f}s | "
#                   f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

#             # Store results
#             for key, value in zip(
#                 ["time", "outlier_ratio", "precision", "recall", "f1_score",
#                  "gt_vps", "est_vps", "gt_corrs", "est_corrs", "parallel_line"],
#                 [t_end - t_start, outlier_ratio, prec, rec, f1,
#                  gt_vps, est_vps, gt_corrs, est_corrs, para_lines]
#             ):
#                 if isinstance(results[out_idx][key], np.ndarray) and not np.isscalar(value):
#                     print(f"⚠️ Warning: Trying to assign non-scalar to scalar array at key={key}")
#                 results[out_idx][key][iter_i] = value

#     print("\n✅ Experiment completed.")
    
#     return results


# # ----------------------------
# # Entry
# # ----------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True, help="Path to param JSON file")
#     args = parser.parse_args()

#     with open(args.config, "r") as f:
#         param = json.load(f)

#     results = run()

#     save_results(results)
    
#     plot_metrics_boxplot(
#         results=results,
#         metric_keys=["f1_score", "precision", "recall"],
#         iter_param=param,
#         save_path="figures"
#     )


import argparse
import json
import time

from .solver.core import globustvp
from .utils.data import generate_valid_line_segmentation, synthetic_data
from .utils.metrics import initialize_result_dict, evaluate_vp_matching
from .utils.io import save_results
from .utils.plot import plot_metrics_boxplot


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic VP estimation experiments.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON file specifying experiment parameters"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def run_single_experiment(param: dict, outlier_ratio: float, iter_idx: int):
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

    result = {
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
    return result


def run_experiments(param: dict):
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


def main():
    args = parse_args()
    param = load_config(args.config)
    results = run_experiments(param)
    save_results(results)

    plot_metrics_boxplot(
        results=results,
        metric_keys=["f1_score", "precision", "recall"],
        iter_param=param,
        save_path="figures"
    )


if __name__ == "__main__":
    main()
