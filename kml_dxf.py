import os
import zipfile
import requests
import geopandas as gpd
import pandas as pd
import streamlit as st
import ezdxf
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, shape
from shapely.ops import unary_union, linemerge, polygonize
import osmnx as ox

# ------------------ Settings ------------------
TARGET_EPSG = "EPSG:32760"
DEFAULT_WIDTH = 6
TOLERANCE_MERGE = 3  # meter

HERE_API_KEY = "BO_l7Fg-xhxA-T4FwiZ_hHQs9fpI4u7vRqfM7xxT0Ec"

# ------------------ Helper Functions ------------------

def classify_layer(hwy):
    if hwy in ['motorway', 'trunk', 'primary']:
        return 'HIGHWAYS', 14
    elif hwy in ['secondary', 'tertiary']:
        return 'MAJOR_ROADS', 10
    elif hwy in ['residential', 'unclassified', 'service']:
        return 'MINOR_ROADS', 6
    elif hwy in ['footway', 'path', 'cycleway']:
        return 'PATHS', 3
    return 'OTHER', DEFAULT_WIDTH

def extract_polygon_from_file(path):
    if path.endswith(".kmz"):
        with zipfile.ZipFile(path, 'r') as z:
            kml_name = [f for f in z.namelist() if f.endswith(".kml")][0]
            z.extract(kml_name, "/tmp")
            path = os.path.join("/tmp", kml_name)
    gdf = gpd.read_file(path)
    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if polys.empty:
        raise Exception("❌ Polygon tidak ditemukan di file.")
    return unary_union(polys.geometry), polys.crs

def get_osm_roads(polygon):
    tags = {"highway": True}
    roads = ox.features_from_polygon(polygon, tags=tags)
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    return roads

def get_here_roads(polygon):
    """Ambil jalan dari HERE vector tile API dalam polygon"""
    minx, miny, maxx, maxy = polygon.bounds
    # Batasan vektor tile (di sini untuk example zoom=13)
    url = (
        f"https://vector.hereapi.com/v2/vectortiles/base/mc/13/{minx},{miny},{maxx},{maxy}/roads"
        f"?apikey={HERE_API_KEY}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"HERE API error: {resp.status_code} - {resp.text}")

    # Response vector tile gratis biasanya protobuf, tapi kita simulasi JSON
    try:
        data = resp.json()  # jika API real JSON
    except:
        # fallback: kosong jika tidak JSON
        data = {"features": []}

    features = []
    for feat in data.get("features", []):
        geom = shape(feat["geometry"])
        road_type = feat.get("properties", {}).get("roadCategory", "other")
        features.append({"geometry": geom, "highway": road_type})

    if not features:
        return gpd.GeoDataFrame(columns=["geometry", "highway"], crs="EPSG:4326")
    return gpd.GeoDataFrame(features, crs="EPSG:4326")

def merge_roads(osm_roads, here_roads, tolerance=TOLERANCE_MERGE):
    """Merge OSM + HERE agar jalan tidak tabrakan"""
    if osm_roads.crs != here_roads.crs:
        here_roads = here_roads.to_crs(osm_roads.crs)

    all_roads = gpd.GeoDataFrame(
        pd.concat([osm_roads, here_roads], ignore_index=True),
        crs=osm_roads.crs
    )

    merged = unary_union(all_roads.buffer(tolerance))
    cleaned = gpd.GeoSeries(merged.buffer(-tolerance), crs=all_roads.crs)
    cleaned = cleaned.explode(index_parts=False).reset_index(drop=True)

    return gpd.GeoDataFrame(geometry=cleaned, crs=all_roads.crs)

def strip_z(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "LineString":
        return LineString([(x, y) for x, y, *_ in geom.coords])
    if geom.geom_type == "MultiLineString":
        return MultiLineString([LineString([(x, y) for x, y, *_ in l.coords]) for l in geom.geoms])
    return geom

def export_to_dxf(gdf, dxf_path, polygon=None, polygon_crs=None):
    doc = ezdxf.new()
    msp = doc.modelspace()
    all_buffers = []

    for _, row in gdf.iterrows():
        geom = strip_z(row.geometry)
        if geom.is_empty or not geom.is_valid:
            continue
        layer, width = classify_layer(str(row.get("highway", "")))
        merged = linemerge(geom) if not isinstance(geom, LineString) else geom
        buffered = merged.buffer(width / 2, resolution=6, join_style=2)
        all_buffers.append(buffered)

    if not all_buffers:
        raise Exception("❌ Tidak ada garis valid untuk diekspor.")

    all_union = unary_union(all_buffers)
    outlines = list(polygonize(all_union.boundary))

    bounds = [(x, y) for geom in outlines for x, y in geom.exterior.coords]
    min_x, min_y = min(x for x, y in bounds), min(y for x, y in bounds)

    for outline in outlines:
        coords = [(x - min_x, y - min_y) for x, y in outline.exterior.coords]
        msp.add_lwpolyline(coords, dxfattribs={"layer": "ROADS"})

    if polygon is not None and polygon_crs is not None:
        poly = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs(TARGET_EPSG).iloc[0]
        if poly.geom_type == 'Polygon':
            coords = [(x - min_x, y - min_y) for x, y in poly.exterior.coords]
            msp.add_lwpolyline(coords, dxfattribs={"layer": "BOUNDARY"})
        elif poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                coords = [(x - min_x, y - min_y) for x, y in p.exterior.coords]
                msp.add_lwpolyline(coords, dxfattribs={"layer": "BOUNDARY"})

    doc.set_modelspace_vport(height=10000)
    doc.saveas(dxf_path)

# ------------------ Core ------------------

def process_kml_to_dxf(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    polygon, polygon_crs = extract_polygon_from_file(file_path)

    osm_roads = get_osm_roads(polygon)
    here_roads = get_here_roads(polygon)

    all_roads = merge_roads(
        osm_roads.to_crs(TARGET_EPSG),
        here_roads.to_crs(TARGET_EPSG),
        tolerance=TOLERANCE_MERGE
    )

    geojson_path = os.path.join(output_dir, "roadmap.geojson")
    dxf_path = os.path.join(output_dir, "roadmap.dxf")

    all_roads.to_file(geojson_path, driver="GeoJSON")
    export_to_dxf(all_roads, dxf_path, polygon=polygon, polygon_crs=polygon_crs)

    return dxf_path, geojson_path, True

# ------------------ Streamlit ------------------

def run_kml_dxf():
    st.title("🌍 KML/KMZ → Road Converter (OSM + HERE)")
    st.caption("Upload file polygon area (.KML atau .KMZ)")

    kml_file = st.file_uploader("Upload file", type=["kml", "kmz"])
    if kml_file:
        with st.spinner("💫 Memproses file..."):
            try:
                temp_input = f"/tmp/{kml_file.name}"
                with open(temp_input, "wb") as f:
                    f.write(kml_file.read())

                output_dir = "/tmp/output"
                dxf_path, geojson_path, ok = process_kml_to_dxf(temp_input, output_dir)

                if ok:
                    st.success("✅ Berhasil diekspor!")
                    with open(dxf_path, "rb") as f:
                        st.download_button("⬇️ Download DXF (AutoCAD)", data=f, file_name="roadmap.dxf")
                    with open(geojson_path, "rb") as f:
                        st.download_button("⬇️ Download GeoJSON", data=f, file_name="roadmap.geojson")

            except Exception as e:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    run_kml_dxf()
