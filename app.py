# streamlit_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import shapely.wkt
import shapely.ops as ops
import zipfile, os, math
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon, box
from shapely.ops import unary_union, polygonize, linemerge
import ezdxf
import osmnx as ox
from io import BytesIO

# ============ Parameters ============
TARGET_EPSG = "EPSG:32760"
DEFAULT_WIDTH = 10
MIN_BUILDING_AREA_M2 = 20
SNAP_GRID_M = 0.2

# ============ Streamlit UI ============
st.title("Roads + Open Buildings Processor")

# Upload KML/KMZ
uploaded_kml = st.file_uploader("Upload KML/KMZ boundary", type=["kml", "kmz"])
uploaded_csv = st.file_uploader("Upload Open Buildings CSV", type=["csv"])

if uploaded_kml and uploaded_csv:
    # --- handle KML/KMZ
    if uploaded_kml.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_kml, 'r') as z:
            z.extractall("kmz_extract")
        kml_files = [f for f in os.listdir("kmz_extract") if f.lower().endswith(".kml")]
        if not kml_files:
            st.error("No KML found in KMZ")
            st.stop()
        kml_path = os.path.join("kmz_extract", kml_files[0])
    else:
        kml_path = uploaded_kml

    gdf_kml = gpd.read_file(kml_path)
    polygons = gdf_kml[gdf_kml.geometry.type.isin(["Polygon","MultiPolygon"])]
    if polygons.empty:
        st.error("No polygon found in boundary")
        st.stop()
    boundary_polygon = unary_union(polygons.geometry)
    boundary_crs = polygons.crs if polygons.crs else "EPSG:4326"

    st.success("Boundary loaded!")

    # --- handle CSV
    df = pd.read_csv(uploaded_csv)
    if "geometry" not in df.columns:
        st.error("CSV must contain 'geometry' column (WKT polygons)")
        st.stop()
    df["geometry"] = df["geometry"].apply(shapely.wkt.loads)
    gdf_buildings = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # filter inside boundary
    boundary_ll = gpd.GeoSeries([boundary_polygon], crs=boundary_crs).to_crs("EPSG:4326").iloc[0]
    gdf_buildings = gdf_buildings[gdf_buildings.centroid.within(boundary_ll)].copy()

    # reproject + filter small
    gdf_buildings = gdf_buildings.to_crs(TARGET_EPSG)
    gdf_buildings["area_m2"] = gdf_buildings.area
    gdf_buildings = gdf_buildings[gdf_buildings["area_m2"] >= MIN_BUILDING_AREA_M2]

    st.write("Buildings after filter:", len(gdf_buildings))

    # roads from OSM
    boundary_wgs84 = gpd.GeoSeries([boundary_polygon], crs=boundary_crs).to_crs("EPSG:4326").iloc[0]
    tags = {"highway": True}
    roads = ox.features_from_polygon(boundary_wgs84, tags=tags)
    roads = roads[roads.geometry.type.isin(["LineString","MultiLineString"])].explode(index_parts=False)

    roads_utm = roads.to_crs(TARGET_EPSG)

    # --- DXF Export
    doc = ezdxf.new()
    msp = doc.modelspace()
    for lname in ["ROADS", "BUILDINGS", "BOUNDARY"]:
        if lname not in doc.layers:
            doc.layers.new(name=lname)

    # simple boundary
    boundary_utm = gpd.GeoSeries([boundary_polygon], crs=boundary_crs).to_crs(TARGET_EPSG).iloc[0]
    if boundary_utm.geom_type == "Polygon":
        coords = [(x, y) for x,y in boundary_utm.exterior.coords]
        msp.add_lwpolyline(coords, dxfattribs={"layer":"BOUNDARY"})

    # buildings
    for geom in gdf_buildings.geometry:
        if geom.geom_type == "Polygon":
            coords = [(x, y) for x,y in geom.exterior.coords]
            msp.add_lwpolyline(coords, close=True, dxfattribs={"layer":"BUILDINGS"})

    # save dxf to buffer
    dxf_bytes = BytesIO()
    doc.saveas(dxf_bytes)
    dxf_bytes.seek(0)

    # save geojson
    geojson_bytes = BytesIO()
    gdf_buildings.to_crs("EPSG:4326").to_file("out.geojson", driver="GeoJSON")
    with open("out.geojson", "rb") as f:
        geojson_bytes.write(f.read())
    geojson_bytes.seek(0)

    # download buttons
    st.download_button("Download DXF", dxf_bytes, "map.dxf")
    st.download_button("Download GeoJSON", geojson_bytes, "buildings.geojson")
