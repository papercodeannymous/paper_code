import sys
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, truncnorm
from tqdm import tqdm


DATASIZE_TYPE = {
    "H1": 100, "2p5": 25000,
    "K5": 5000, "W1": 10000, "W2": 20000, "W3": 30000, "W4": 40000, "W5": 50000,
    "TW1": 100000, "TW2": 200000, "TW5": 500000,
    "HW1": 1000000, "HW5": 5000000, "KW": 10000000, "KW2": 20000000, "KW5": 50000000,
    "WW1": 100000000
    
}

DATADISTRIBUTION_TRYPE = {
    "NORMAL": 1, "UNIFORM": 2, "SKEW-NOR0": 3, "SKEW-NOR2": 4, "SKEW-NOR4": 5, "SKEW-NOR8": 6
}

DATALOCTION_TYPE = {
    "China": []
}


def reading_data(file_path, save_path="model_dataset.npy"):
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    data = np.fromiter(
        tqdm((float(line.strip()) for line in open(file_path)), 
             total=total_lines, desc="Reading Data"), 
        dtype=np.float64, count=total_lines
    )
    edge_size = 0.01
    rectangles = np.column_stack((data[:-1:2] - edge_size, data[1::2] - edge_size, data[:-1:2], data[1::2]))
    np.save(save_path, rectangles)

    print(f"\nData saved to {save_path}")
    print(f"Total Rectangles: {len(rectangles)}")
    return rectangles

def generate_coordinates(x_min, x_max, y_min, y_max, data_size, output_file, testing_flag, distribution_type="UNIFORM"):
    num_points = DATASIZE_TYPE[data_size]
    if testing_flag:
        output_file += "/testing_" + data_size
    else:
        output_file += "/" + data_size

    if DATADISTRIBUTION_TRYPE[distribution_type] == 1:
        def truncated_normal(mean, std, lower, upper, size):
            a, b = (lower - mean) / std, (upper - mean) / std
            return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

        x_mean = y_mean = 0.5
        x_std = y_std = 0.15

        x_pre = np.random.normal(x_mean, x_std, num_points)
        y_pre = np.random.normal(y_mean, y_std, num_points)

        x_coords = truncated_normal(x_mean * (x_max - x_min), x_std * (x_max - x_min), x_min, x_max, num_points)
        y_coords = truncated_normal(y_mean * (y_max - y_min), y_std * (y_max - y_min), y_min, y_max, num_points)
        output_file += "_NORMAL.txt"
        
    elif DATADISTRIBUTION_TRYPE[distribution_type] == 2:
        x_coords = np.random.uniform(x_min, x_max, num_points)
        y_coords = np.random.uniform(y_min, y_max, num_points)
        output_file += "_UNIFORM.txt"

    elif DATADISTRIBUTION_TRYPE[distribution_type] == 3:
        # > 0, right; < 0, left 
        alpha = 0
        x_raw = skewnorm.rvs(alpha, size=num_points)
        y_raw = skewnorm.rvs(alpha, size=num_points)

        x_coords = x_min + (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw)) * (x_max - x_min)
        y_coords = y_min + (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw)) * (y_max - y_min)
        output_file += "_SKEW-NOR0.txt"
    
    elif DATADISTRIBUTION_TRYPE[distribution_type] == 4:

        # # if skew_factor is 1, then it's uniform 
        # # > 1, low left;    < 1, up right
        # skew_factor = 0.1

        # x_raw = np.random.uniform(0, 1, num_points)
        # y_raw = np.random.uniform(0, 1, num_points)

        # x_skewed = np.power(x_raw, skew_factor)
        # y_skewed = np.power(y_raw, skew_factor)

        # x_coords = x_min + (x_skewed - np.min(x_skewed)) / (np.max(x_skewed) - np.min(x_skewed)) * (x_max - x_min)
        # y_coords = y_min + (y_skewed - np.min(y_skewed)) / (np.max(y_skewed) - np.min(y_skewed)) * (y_max - y_min)
        # output_file += "_SKEW-UNI.txt"
                # > 0, right; < 0, left 
        alpha = 2
        x_raw = skewnorm.rvs(alpha, size=num_points)
        y_raw = skewnorm.rvs(alpha, size=num_points)

        x_coords = x_min + (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw)) * (x_max - x_min)
        y_coords = y_min + (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw)) * (y_max - y_min)
        output_file += "_SKEW-NOR2.txt"
    
    elif DATADISTRIBUTION_TRYPE[distribution_type] == 5:
        # > 0, right; < 0, left 
        alpha = 4
        x_raw = skewnorm.rvs(alpha, size=num_points)
        y_raw = skewnorm.rvs(alpha, size=num_points)

        x_coords = x_min + (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw)) * (x_max - x_min)
        y_coords = y_min + (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw)) * (y_max - y_min)
        output_file += "_SKEW-NOR4.txt"
    
    elif DATADISTRIBUTION_TRYPE[distribution_type] == 6:
        # > 0, right; < 0, left 
        alpha = 8
        x_raw = skewnorm.rvs(alpha, size=num_points)
        y_raw = skewnorm.rvs(alpha, size=num_points)

        x_coords = x_min + (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw)) * (x_max - x_min)
        y_coords = y_min + (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw)) * (y_max - y_min)
        output_file += "_SKEW-NOR8.txt"

    else:
        raise ValueError("Unsupported distribution type. Use 'uniform' or 'normal'.")
    
    with open(output_file, "w") as f:
        for x, y in zip(x_coords, y_coords):
            f.write(f"{x}\n{y}\n")
    
    if testing_flag:
        output_file_npy = output_file.split(".")[0] + ".npy"
        reading_data(output_file, output_file_npy)

    print(f"Successfully saved {num_points} points to {output_file}")



def data_visualization(data_size, output_file, testing_flag, distribution_type="NORMAL"):
    if testing_flag:
        data_filename = output_file + "/testing_" + data_size
    else:
        data_filename = output_file + "/" + data_size
    
    data_filename += "_" + distribution_type + ".txt"

    print(data_filename)
    
    model_dataset = []
    with open(data_filename) as input_file:
        n = 0
        ll_x = ll_y = tr_x = tr_y = 0
        for line in input_file:
            if len(model_dataset) >= DATASIZE_TYPE[data_size]:
                break
            if n % 2 == 0:
                ll_x = float(line[:-1]) - 100
                tr_x = float(line[:-1])
            else:
                ll_y = float(line[:-1]) - 100
                tr_y = float(line[:-1]) 
                model_dataset.append([ll_x, ll_y, tr_x, tr_y])
            n += 1
    
    fig, ax = plt.subplots()
    
    for rect in model_dataset:
        ll_x, ll_y, tr_x, tr_y = rect
        width = tr_x - ll_x
        height = tr_y - ll_y
        ax.add_patch(plt.Rectangle((ll_x, ll_y), width, height, linewidth=1, edgecolor='r', facecolor='none'))
        
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Rectangles Visualization')
    plt.show()

if __name__ == "__main__":
    output_file = rf"data_gen/testing_data"
    distribution_type="NORMAL"
    x_min, x_max, y_min, y_max = 0, 100000, 0, 100000
    data_size = "TW1"
    testing_flag = True          # generate testing data

    generate_coordinates(
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    data_size=data_size,
    output_file=output_file,
    testing_flag=testing_flag,
    distribution_type=distribution_type
    )

    # data_visualization(data_size=data_size, output_file=output_file, testing_flag=testing_flag, distribution_type=distribution_type)
