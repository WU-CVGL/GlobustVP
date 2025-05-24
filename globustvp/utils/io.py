import argparse
import os
import json
import cv2
import numpy as np
from typing import Any, List, Dict, Tuple


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running VP estimation experiments.

    Returns:
        argparse.Namespace
            Parsed arguments containing the path to the configuration JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Run synthetic experiments for vanishing point estimation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON file specifying experiment parameters.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    Load configuration parameters from a JSON file.

    Parameters:
        config_path : str
            Path to the configuration JSON file.

    Returns:
        Dict
            Dictionary containing experiment parameters.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def to_serializable(obj: Any) -> Any:
    """
    Recursively convert various data types (e.g., numpy arrays, dictionaries, lists) 
    into types that are JSON serializable.

    Parameters:
        obj : Any
            The object to be converted into a serializable format. This can include
            numpy arrays, dictionaries, lists, and basic numeric types.

    Returns:
        obj : Any
            The converted object that is JSON serializable. Numpy arrays are converted
            to lists, numpy numeric types are converted to standard Python types, and
            dictionaries and lists are recursively serialized.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, Dict):
        return {k: to_serializable(v) for k, v in obj.items()}  # Recursively handle dicts
    elif isinstance(obj, List):
        return [to_serializable(i) for i in obj]  # Recursively handle lists
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert numpy float types to Python float
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert numpy int types to Python int
    else:
        return obj  # Return the object as-is if it's already serializable


def save_results(
    results: Any,
    save_path: str = "results",
    filename: str = "experiment_results.json"
) -> None:
    """
    Save the experiment results to a specified directory in JSON format.

    Parameters:
        results : Any
            The results to be saved.
            This can be any object that is serializable (e.g., dictionary, list, numpy array).
        save_path : str, optional, default="results"
            The directory path where the results will be saved.
        filename : str, optional, default="experiment_results.json"
            The name of the file where the results will be stored.

    Returns:
        None
            Saves the results to the specified file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Convert the results into a serializable format
    serializable_results = to_serializable(results)
    
    # Construct the full file path
    file_path = os.path.join(save_path, filename)
    
    # Save the results to the file in JSON format
    with open(file_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Confirmation message
    print(f"✅ Results saved to {file_path}")


def load_image_and_gray(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a color image and its corresponding grayscale version.

    Parameters:
        img_path : str
            Path to the input image.

    Returns:
        img : np.ndarray
            Loaded color image (BGR), shape (H, W, 3).
        gray : np.ndarray
            Grayscale image, shape (H, W).
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"❌ Failed to load image at '{img_path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray
