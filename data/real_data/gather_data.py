import os
from preprocess_utils import *
import numpy as np
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd

"""
Geospatial Data Processing and Sampling Module
"""

np.random.seed(3)

def get_all_shp_dirs(shp_dir):
    """
    Recursively collects all Shapefile directories from a parent directory.
    
    Traverses the specified directory structure to identify all subdirectories
    containing Shapefile data.
    """
    dirs_list = []
    dirs = os.listdir(shp_dir)
    for dir in dirs:
        dir = os.path.join(shp_dir, dir)
        dirs_list.append(dir)
    np.random.shuffle(dirs_list)
    return dirs_list

def check_all_shps_volume(dirs_list, types_num, country="CHINA"):
    """
    Analyzes geometric feature volume across multiple Shapefile directories.

    Iterates through Shapefile directories and calculates the total count of
    different geometry types (Point, Polygon, LineString). 
    """
    sumlen_point = 0
    sumlen_polygon = 0
    sumlen_linestring = 0
    for dir in dirs_list:
        print(dir.split('\\')[-1])
        all_points, point_points, polygon_points, linestring_points = extract_points_from_geometry(dir, types_num, False, country)
        sumlen_point += len(point_points)
        sumlen_polygon += len(polygon_points)
        sumlen_linestring += len(linestring_points)
    return sumlen_point, sumlen_polygon, sumlen_linestring


def get_all_points_data(file_name, dirs_list, types_num, country="CHINA"):
    """
    Generates spatially representative point samples from multiple Shapefile datasets.
    """
    gdf_all = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    for dir in dirs_list:
        all_points, point_points, polygon_points, linestring_points = extract_points_from_geometry(dir, types_num, False, country)
        if len(all_points) == 0:
            continue
            
        new_volume = int(len(all_points) * 0.005)
        if new_volume == 0:
            continue
        
        point_geoms = [Point(xy) for xy in all_points]

        gdf = gpd.GeoDataFrame(geometry=point_geoms, crs="EPSG:4326")
        gdf_all = pd.concat([gdf_all, gdf], ignore_index=True)

    print("Finish writting!")
    return gdf_all

def read_points_from_file(file_name):
    points = []  
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()  
            if line:  
                x, y = line.split(",")  
                x = float(x.strip())  
                y = float(y.strip())  
                points.append((x, y))  
    return points





