
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping, shape
import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer

def image_is_geotiff(image_path):
    try:
        with rasterio.open(image_path) as src:
            return src.crs is not None and src.transform is not None
    except Exception:
        return False

def raster_transform_coords(x,y, transform):
    # pixel x,y -> world coordinate using affine transform
    # rasterio uses row (y), col (x) convention -> transform * (col, row)
    X, Y = transform * (x, y)
    return X, Y

def pixel_polygons_to_world(polygons, image_path):
    # polygons are shapely geometries in pixel coordinates (x,y)
    with rasterio.open(image_path) as src:
        tr = src.transform
        crs = src.crs
        world_polys = []
        for poly in polygons:
            coords = []
            for x,y in list(poly.exterior.coords):
                X,Y = raster_transform_coords(x,y,tr)
                coords.append((X,Y))
            from shapely.geometry import Polygon
            world_polys.append(Polygon(coords))
    return world_polys

def ensure_same_crs(gdf, target_epsg):
    try:
        gdf = gdf.set_crs(epsg=target_epsg, allow_override=True)
    except Exception:
        pass
    return gdf
