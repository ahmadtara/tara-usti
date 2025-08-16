import streamlit as st
import geopandas as gpd
import pandas as pd
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon, mapping
import shapely.ops as ops
import osmnx as ox
import shapely
import ezdxf
from io import BytesIO
import zipfile

# -----------------------------
# 1) Parameters
# -----------------------------
TARGET_EPSG = "EPSG:32748"  # contoh: UTM zone untuk Indonesia barat
DEFAULT_WIDTH = 10
MIN_BUILDING_AREA_M2 = 20
SNAP_GRID_M = 0.2
CSV_URL = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/31d_buildings.csv.gz"

# -----------------------------
# 2) Utility functions
# -----------------------------
def classify_layer(hwy):
    if hwy in ["motorway", "trunk", "primary"]:
        return "HIGHWAYS", 14
    elif hwy in ["secondary", "tertiary"]:
        return "MAJOR_ROADS", 12
    elif hwy in ["residential", "unclassified", "service"]:
        return "MINOR_ROADS", 8
    elif hwy in ["footway", "path", "cycleway"]:
        return "PATHS", 4
    return "OTHER", DEFAULT_WIDTH

def strip_z(geom):
    """Remove Z if present"""
    if geom.has_z:
        return ops.transform(lambda x, y, z=None: (x, y), geom)
    return geom

def snap_to_grid(geom, grid_size=SNAP_GRID_M):
    """Snap coordinates to grid"""
    return ops.transform(lambda x, y: (round(x/grid_size)*grid_size, round(y/grid_size)*grid_size), geom)

# -----------------------------
# 3) Load buildings from GCS
# -----------------------------
@st.cache_data
def load_buildings():
    df = pd.read_csv(CSV_URL, compression="gzip")
    if "geometry" not in df.columns:
        raise RuntimeError("CSV must contain 'geometry' column with WKT polygons.")
    df["geometry"] = df["geometry"].apply(shapely.wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# -----------------------------
# 4) Streamlit UI
# -----------------------------
st.title("🗺 Roads + Open Buildings Processor")

# Upload boundary file
boundary_file = st.file_uploader("Upload boundary KML/KMZ", type=["kml", "kmz"])

if boundary_file:
    st.success(f"Boundary loaded: {boundary_file.name}")

    # Load boundary
    gdf_boundary = gpd.read_file(boundary_file)
    boundary = gdf_boundary.unary_union

    # Load Open Buildings
    st.info("Downloading Open Buildings CSV from GCS...")
    gdf_buildings = load_buildings()

    # Filter buildings inside boundary
    gdf_filtered = gdf_buildings[gdf_buildings.intersects(boundary)].copy()
    st.write(f"🏠 Buildings after filter: {len(gdf_filtered)}")

    if len(gdf_filtered) == 0:
        st.error("No buildings found in boundary.")
        st.stop()

    # Reproject to target UTM
    gdf_boundary = gdf_boundary.to_crs(TARGET_EPSG)
    gdf_filtered = gdf_filtered.to_crs(TARGET_EPSG)

    # Bersihkan bangunan kecil
    gdf_filtered["area"] = gdf_filtered.area
    gdf_filtered = gdf_filtered[gdf_filtered["area"] > MIN_BUILDING_AREA_M2]

    # Snap grid + strip Z
    gdf_filtered["geometry"] = gdf_filtered["geometry"].apply(strip_z).apply(snap_to_grid)

    # Ambil roads dari OSM
    st.info("Downloading roads from OpenStreetMap...")
    gdf_roads = ox.geometries_from_polygon(boundary, tags={"highway": True})
    gdf_roads = gdf_roads.to_crs(TARGET_EPSG)

    # Tambah layer & width
    gdf_roads["layer_width"] = gdf_roads["highway"].apply(classify_layer)
    gdf_roads["layer"] = gdf_roads["layer_width"].apply(lambda x: x[0])
    gdf_roads["width"] = gdf_roads["layer_width"].apply(lambda x: x[1])

    # Buffer roads
    gdf_roads["geometry"] = gdf_roads.apply(lambda row: row.geometry.buffer(row["width"]), axis=1)

    # Potong bangunan yg kena jalan
    buildings_final = gdf_filtered.overlay(gdf_roads, how="difference")

    st.success(f"✅ Final buildings count: {len(buildings_final)}")

    # -----------------------------
    # 5) Export to GeoJSON & DXF
    # -----------------------------
    geojson_bytes = buildings_final.to_crs(4326).to_json().encode("utf-8")

    # DXF
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Add boundary
    for geom in gdf_boundary.geometry:
        if isinstance(geom, (Polygon, MultiPolygon)):
            msp.add_lwpolyline(list(geom.exterior.coords), dxfattribs={"layer": "BOUNDARY"})

    # Add roads
    for _, row in gdf_roads.iterrows():
        geom = row.geometry
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(list(geom.exterior.coords), dxfattribs={"layer": row["layer"]})

    # Add buildings
    for geom in buildings_final.geometry:
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(list(geom.exterior.coords), dxfattribs={"layer": "BUILDINGS"})

    dxf_bytes = BytesIO()
    doc.write(dxf_bytes)
    dxf_bytes.seek(0)

    # Zip both
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("buildings.geojson", geojson_bytes)
        zf.writestr("output.dxf", dxf_bytes.read())
    zip_buffer.seek(0)

    st.download_button("⬇️ Download results (GeoJSON + DXF)", data=zip_buffer, file_name="results.zip", mime="application/zip")
