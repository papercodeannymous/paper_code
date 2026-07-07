import time
import geopandas as gpd
import fiona
import os
import math
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from geopy.distance import geodesic
from scipy.spatial import KDTree
import numpy as np

"""
Comprehensive Geospatial Data Processing and Analysis Module.
This module provides extensive utilities for reading, processing, analyzing, 
and visualizing geospatial data in ESRI Shapefile format. 

It supports multi-layer shapefile handling, geometry type extraction, 
spatial filtering, and quality assessment for geographic information 
systems (GIS) and spatial analysis applications.
"""

def read_shapefile(shp_path, show_attrs=False):
    """
    Reads a shapefile and returns its contents as a GeoDataFrame.
    """
    gdf = gpd.read_file(shp_path) 
    if show_attrs:      
        print(f".head(): {gdf.head()}")     # show some data
        print(f".info(): {gdf.info()}")     # show basic info of GeoDataFrame
        print(f".crs: {gdf.crs}")           # show Coordinate-Reference-System
        print(f".geometry.type: {gdf.geometry.type}")   # show type of geometry
    return gdf

def read_shapefile_by_type(shp_path, type_name, show_attrs=False):
    """
    Reads a specific layer from a multi-layer shapefile.
    """
    layer_gdf = gpd.read_file(shp_path, layer=type_name)       # .to_crs(epsg=3395)
    if show_attrs:
        print(f".head(): {layer_gdf.head()}")     
        print(f".info(): {layer_gdf.info()}")     
        print(f".crs: {layer_gdf.crs}")          
        print(f".geometry.type: {layer_gdf.geometry.type}")   
    return layer_gdf

def get_shp_layers(shp_path, show_info=False):
    """
    Discovers all available layers in a shapefile directory.
    """
    shp_layers = os.listdir(shp_path)
    layer_names = set()
    for layer in shp_layers:
        rmv_fmt = layer.split('.')[0]
        if rmv_fmt !="README":
            layer_names.add(rmv_fmt)
    if show_info:
        for lname in layer_names:
            print(lname)
    return layer_names
   
def shapefile_visualization(gdf, type_name, save_path=False):
    """
    Creates publication-quality visualizations of geographic data.
    """
    # gdf = gdf.to_crs(epsg=3395)
    ax = gdf.plot(figsize=(10, 10), edgecolor='red', facecolor='lightblue')
    plt.title(type_name, fontsize=16)
    if save_path:
        full_svp = save_path + rf"\{type_name}.png"
        plt.savefig(full_svp, format='png', dpi=300)  
        print(f"Pic has already saved in {full_svp}")
    plt.show()

def get_all_layers_pics(shp_path, pic_save_path):
    """
    Batch processes and visualizes all layers in a shapefile directory.
    """
    layer_names = get_shp_layers(shp_path)
    for type in layer_names:
        layer_gdf = read_shapefile_by_type(shp_path, type)
        shapefile_visualization(layer_gdf, type, pic_save_path)
    
    print("Finished saving all pics of types")

def get_geometry_by_type(shp_path, type_name):
    """
    Extracts detailed geometry information for a specific layer.
    """
    layer_gdf = read_shapefile_by_type(shp_path, type_name)
    fst_geometry_data = layer_gdf.iloc[0]["geometry"]
    geometry_type = layer_gdf.geom_type[0]
    return layer_gdf, geometry_type, fst_geometry_data


def extract_points_from_geometry(shp_path, types_num, show_info=False, country="CHINA"):
    """
        Get all Points data in types like Points and Polygon geometry-type
        types_num:
            1:  Point
            2:  Point, Polygon
            3:  Point, Polygon, LineString
    """
    all_points = []
    point_points = []
    polygon_points = []
    linestring_points = []
    layer_names = get_shp_layers(shp_path)

    for type in layer_names:
        if country == "CHINA" or country == "AMERICA":
            print(f"Traversing layer {type}...", end='\r')
            layer_gdf = read_shapefile_by_type(shp_path, type)
            for geom in layer_gdf.geometry:
                if isinstance(geom, Point):
                    point_points.append((geom.x, geom.y))

                elif isinstance(geom, Polygon) and (types_num == 2 or types_num == 3):
                    exterior_coords = list(geom.exterior.coords)
                    polygon_points.extend(exterior_coords)

                    # if has interior holes, extract them
                    for interior in geom.interiors:
                        interior_coords = list(interior.coords)
                        polygon_points.extend(interior_coords)

                elif isinstance(geom, LineString) and types_num == 3:
                    line_coords = list(geom.coords)
                    linestring_points.extend(line_coords)
        else:
            print(f"Only traversing POIs layer {type}...", end='\r')
            if type == "gis_osm_pois_free_1":
                layer_gdf = read_shapefile_by_type(shp_path, type)
                for geom in layer_gdf.geometry:
                    point_points.append((geom.x, geom.y))
    
    if types_num == 1:
        all_points = point_points
    elif types_num == 2:
        all_points = point_points + polygon_points
    elif types_num == 3:
        all_points = point_points + polygon_points + linestring_points
    
    if show_info:
        print(all_points)
    
    print(f"all_points: {len(all_points)}, point_points: {len(point_points)}, polygon_points: {len(polygon_points)}, linestring_points: {len(linestring_points)}")

    return all_points, point_points, polygon_points, linestring_points

def calculate_distances(points):
    def calculate_euclidean_distance(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    distances = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        distance = calculate_euclidean_distance(p1, p2) 
        distances.append(distance)
    return distances

def filter_points_by_distance(points, threshold=1.0):
    """
    :param points: [(x1, y1), (x2, y2), ...]
    :param threshold: threshold (meter)
    :return: list of being removed duplicated
    """
    print("Filtering data ...")
    points_array = np.array(points)
    tree = KDTree(points_array)
    filtered_points = []
    visited = set()

    n = 0
    for i, point in enumerate(points):
        if i in visited:
            continue
        nearby_indices = tree.query_ball_point(point, threshold)
        for idx in nearby_indices:
            visited.add(idx)
        filtered_points.append(point)
        n += 1
        # print(f"Filtered data {n}", end='\r')
    print(f"Filtered data volume: {len(filtered_points)}")
    return filtered_points



