
from fastkml import kml
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import geopandas as gpd
from pyproj import CRS
import os

def read_kml_boundary(kml_path):
    with open(kml_path, "rb") as f:
        doc = f.read()
    k = kml.KML()
    k.from_string(doc)
    features = list(k.features())
    # attempt find polygon(s)
    polys = []
    def walk(feats):
        for feat in feats:
            try:
                for sub in feat.features():
                    walk([sub])
            except Exception:
                pass
            try:
                geom = feat.geometry
                if geom is not None:
                    shap = shape(geom.geojson())
                    polys.append(shap)
            except Exception:
                pass
    walk(features)
    if len(polys)==0:
        raise ValueError("No polygon found in KML")
    union = unary_union(polys)
    # try detect CRS from KML (not usually stored); default WGS84
    return union, None

def kml_crs_epsg(kml_path):
    # placeholder: KML usually in EPSG:4326 (lat/lon)
    return 4326
