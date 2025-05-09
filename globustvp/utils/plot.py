import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_lines(
    lines1: np.ndarray,
    lines2: np.ndarray,
    lines3: np.ndarray,
    n_outliers: int,
    normalized: bool = False
) -> None:
    """
    Plot 2D lines with a distinction between inliers and outliers, either for normalized or unnormalized lines.

    Parameters:
        lines1: np.ndarray
            The first set of lines, shape (M, N).
        lines2: np.ndarray
            The second set of lines, shape (M, N).
        lines3: np.ndarray
            The third set of lines, shape (M, N).
        n_outliers : int
            Number of outlier lines to be distinguished in the plot.
        normalized : bool, optional, default=False
            If True, plots normalized lines; otherwise, plots unnormalized lines.

    Returns:
        None
            Displays a plot with the specified lines, distinguishing outliers and inliers.
    """
    plt.figure(figsize=(10, 8))

    # Plot lines from the first set
    for j in range(lines1.shape[1]):
        color = "c" if j >= lines1.shape[1] - n_outliers else "r"
        if normalized:
            plt.plot([lines1[0, j], lines1[2, j]], [lines1[1, j], lines1[3, j]], color)
        else:
            plt.plot([lines1[0, j], lines1[3, j]], [lines1[1, j], lines1[4, j]], color)

    # Plot lines from the second set
    for j in range(lines2.shape[1]):
        color = "c" if j >= lines2.shape[1] - n_outliers else "g"
        if normalized:
            plt.plot([lines2[0, j], lines2[2, j]], [lines2[1, j], lines2[3, j]], color)
        else:
            plt.plot([lines2[0, j], lines2[3, j]], [lines2[1, j], lines2[4, j]], color)

    # Plot lines from the third set
    for j in range(lines3.shape[1]):
        color = "c" if j >= lines3.shape[1] - n_outliers else "b"
        if normalized:
            plt.plot([lines3[0, j], lines3[2, j]], [lines3[1, j], lines3[3, j]], color)
        else:
            plt.plot([lines3[0, j], lines3[3, j]], [lines3[1, j], lines3[4, j]], color)

    plt.axis("equal")
    title = "Normalized 2D lines" if normalized else "Unnormalized 2D lines"
    plt.title(title)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def plot_metrics_boxplot(
    results: List[Dict],
    metric_keys: List[str],
    iter_param: Dict,
    save_path: str = "figures"
) -> None:
    """
    Plot boxplots of the specified metrics against the outlier ratio.

    Parameters:
        results : List[Dict]
            List of dictionaries containing results for each outlier ratio.
        metric_keys : List[str]
            List of metric keys to be plotted.
        iter_param : Dict
            Dictionary of configuration parameters,
            including different outlier ratios.
        save_path : str, optional, default="figures"
            Directory path where the generated plot images will be saved.

    Returns:
        None
            Saves one boxplot figure per metric.
    """
    label_map = {
        "f1_score": "F1-Score (%)",
        "precision": "Precision (%)",
        "recall": "Recall (%)"
    }
    os.makedirs(save_path, exist_ok=True)
    outlier_ratios = [round(float(r), 2) for r in iter_param["outlier_ratio"]]

    for metric in metric_keys:
        # Gather data for each outlier ratio
        metric_data = [results[i][metric]*100 for i in range(len(outlier_ratios))]

        plt.figure()
        plt.boxplot(metric_data, labels=outlier_ratios, showfliers=True)
        plt.xlabel("Outlier ratio (%)")
        # plt.ylabel(metric.replace("_", " ").title())
        plt.ylabel(label_map.get(metric, metric.replace("_", " ").title()))
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.grid(True)
        plt.tight_layout()

        fig_path = os.path.join(save_path, f"{metric}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"📦 Saved boxplot: {fig_path}")