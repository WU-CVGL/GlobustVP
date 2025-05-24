"""
Script for launching synthetic VP estimation experiments.

Author: GlobustVP Team
CVPR 2025 Oral · https://arxiv.org/abs/2505.04788
"""

from globustvp.utils.io import parse_args, load_config, save_results
from globustvp.utils.experiment import run_experiments
from globustvp.utils.plot import plot_metrics_boxplot


def main() -> None:
    """
    Main entry point for running VP experiments.
    
    Returns:
        None
    """
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
