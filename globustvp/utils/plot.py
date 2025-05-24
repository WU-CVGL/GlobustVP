import os
import cv2
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
            Display a plot with the specified lines, distinguishing outliers and inliers.
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


def plot_lines_on_image(
    image: np.ndarray,
    lines: np.ndarray,
    line_color: str = "b",
    line_width: float = 2.5,
    pause_time: float = 3.0,
    title: str = "LSD detected lines"
) -> None:
    """
    Plot extracted line segments by LSD on top of an image.

    Parameters:
        image : np.ndarray
            Input image in grayscale/BGR format, shape (H, W) or (H, W, 3).
        lines : np.ndarray
            2D line segments in homogeneous image coordinates from LSD,
            where each column is [x1, y1, 1, x2, y2, 1]^T, shape (6, N).
        line_color : str, optional, default="b"
            Line color in matplotlib format (e.g., "r", "b", "#FF8800").
        line_width : float, optional, default=2.5
            Line width.
        pause_time : float, optional, default=3.0
            Duration in seconds to display the image.
        title : str, optional, default="LSD detected lines"
            Title of the figure.
    
    Returns:
        None
            Display a plot with line segments extracted from the images using LSD.
    """
    # Convert grayscale/BGR image to RGB for consistent color drawing
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = image.astype(np.uint8)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    
    # Iterate over lines and draw each one
    lines = lines.T
    for x1, y1, _, x2, y2, _ in lines:
        plt.plot([x1, x2], [y1, y2], color=line_color, linewidth=line_width)

    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(pause_time)
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
            Save one boxplot figure per metric.
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


def visualize_line_vp_associations(
    image: np.ndarray,
    lines: np.ndarray,
    est_corrs: np.ndarray,
    colors: List[str] = ["r", "g", "b"],
    line_width: float = 2.5,
    title: str = "Line-VP associations"
) -> None:
    """
    Visualize vanishing point (VP) line associations by overlaying grouped lines on the image.

    Parameters:
        image : np.ndarray
            Original input image in BGR format, shape (H, W, 3).
        lines : np.ndarray
            2D line segments in homogeneous image coordinates from LSD,
            where each column is [x1, y1, 1, x2, y2, 1]^T, shape (6, N).
        est_corrs : np.ndarray
            Estimated line-VP associations, shape (3, N).
        colors : List[str], optional, default=["r", "g", "b"]
            List of colors used to draw lines corresponding to each VP group.
        line_width : float, optional, default=2.5
            Line width.
        title : str, optional, default="Line-VP associations"
            Title of the visualization figure.

    Returns:
        None
            Display a plot with line-VP association result.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for i in range(est_corrs.shape[0]):
        line_ids = np.where(est_corrs[i] == 1)[0]
        for j in line_ids:
            x1, y1, _, x2, y2, _ = lines[:, j]
            ax.plot([x1, x2], [y1, y2], color=colors[i % len(colors)], linewidth=line_width)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
