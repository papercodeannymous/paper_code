import sys
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, truncnorm
from tqdm import tqdm

"""
This module provides functionality for generating and processing spatial datasets 
containing 2D coordinates with various statistical distributions. 

Key Features:
- Support for multiple data sizes (TW1: 100K, TW2: 200K, TW5: 500K, HW1: 1M points), you can set the size of the dataset you need.
- Multiple distribution types (Normal, Uniform, Skewed Normal)
- Data conversion from text format to numpy arrays for efficient processing
- Progress tracking for large dataset operations (use of tqdm)
"""

# Dictionary defining dataset sizes with descriptive keys
DATASIZE_TYPE = {
    "TW1": 100000, "TW2": 200000, "TW5": 500000,
    "HW1": 1000000,
}

# Dictionary defining statistical distribution types for data generation
DATADISTRIBUTION_TYPE = {
    "NORMAL": 1, "UNIFORM": 2, "SKEW-NOR8": 3
}


def reading_data(file_path, save_path="model_dataset.npy"):
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    data = np.fromiter(
        tqdm((float(line.strip()) for line in open(file_path)), 
             total=total_lines, desc="Reading Data"), 
        dtype=np.float64, count=total_lines
    )
    edge_size = 1
    rectangles = np.column_stack((data[:-1:2] - edge_size, data[1::2] - edge_size, data[:-1:2], data[1::2]))
    np.save(save_path, rectangles)

    print(f"\nData saved to {save_path}")
    print(f"Total Rectangles: {len(rectangles)}")
    return rectangles

def generate_coordinates(x_min, x_max, y_min, y_max, data_size, output_file, testing_flag, distribution_type="UNIFORM"):
    """
    Generates synthetic 2D coordinate datasets with specified statistical distributions.
    
    Creates datasets of various sizes and distributions for testing spatial algorithms.
    Supports normal, uniform, and skewed normal distributions. Outputs data in both
    text format and optionally as numpy arrays for testing purposes.
    
    Parameters:
    -----------
    x_min, x_max : float
        Minimum and maximum bounds for x-coordinates
    y_min, y_max : float  
        Minimum and maximum bounds for y-coordinates
    data_size : str
        Key from DATASIZE_TYPE specifying number of points to generate
    output_file : str
        Base path for output files
    testing_flag : bool
        If True, generates additional numpy format for testing and modifies filename
    distribution_type : str, optional
        Type of statistical distribution (default: "UNIFORM")
    
    Raises:
    -------
    ValueError
        If unsupported distribution type is specified
    """

    # Determine dataset size from predefined types
    num_points = DATASIZE_TYPE[data_size]

    # Construct output filename based on parameters
    if testing_flag:
        output_file += "/testing_" + data_size
    else:
        output_file += "/" + data_size

    # Generate coordinates based on specified distribution
    if DATADISTRIBUTION_TYPE[distribution_type] == 1:
        """
        Normal (Gaussian) Distribution:
        Generates coordinates using truncated normal distribution to ensure
        all points fall within specified bounds. 
        """
        def truncated_normal(mean, std, lower, upper, size):
            a, b = (lower - mean) / std, (upper - mean) / std
            return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

        # Distribution parameters (mean at center, std as 15% of range)
        x_mean = y_mean = 0.5
        x_std = y_std = 0.15

        # Generate coordinates with truncated normal distribution
        x_coords = truncated_normal(x_mean * (x_max - x_min), x_std * (x_max - x_min), x_min, x_max, num_points)
        y_coords = truncated_normal(y_mean * (y_max - y_min), y_std * (y_max - y_min), y_min, y_max, num_points)
        output_file += "_NORMAL.txt"
        
    elif DATADISTRIBUTION_TYPE[distribution_type] == 2:
        """
        Uniform Distribution:
        Generates coordinates with equal probability across entire specified range.
        Simple random sampling between minimum and maximum bounds.
        """
        x_coords = np.random.uniform(x_min, x_max, num_points)
        y_coords = np.random.uniform(y_min, y_max, num_points)
        output_file += "_UNIFORM.txt"

    elif DATADISTRIBUTION_TYPE[distribution_type] == 3:
        """
        Skewed Normal Distribution:
        Generates coordinates with asymmetric distribution using skewnorm.
        Alpha parameter = 8 creates strong right skew. Raw skewed values are
        normalized to fit within specified coordinate bounds.
        """
        # Alpha > 0: right skew, Alpha < 0: left skew
        alpha = 8
        x_raw = skewnorm.rvs(alpha, size=num_points)
        y_raw = skewnorm.rvs(alpha, size=num_points)

        x_coords = x_min + (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw)) * (x_max - x_min)
        y_coords = y_min + (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw)) * (y_max - y_min)
        output_file += "_SKEW-NOR8.txt"

    else:
        raise ValueError("Unsupported distribution type. Use 'uniform' or 'normal'.")
    
    # Write coordinates to text file in alternating x,y format
    with open(output_file, "w") as f:
        for x, y in zip(x_coords, y_coords):
            f.write(f"{x}\n{y}\n")
    
    # For testing datasets, create additional numpy format
    if testing_flag:
        output_file_npy = output_file.split(".")[0] + ".npy"
        reading_data(output_file, output_file_npy)

    print(f"Successfully saved {num_points} points to {output_file}")

