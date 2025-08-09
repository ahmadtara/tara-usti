import os
import ee
import geopandas as gpd
import streamlit as st
import ezdxf
import json
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union

# Konstanta
TARGET_EPSG = "EPSG:32760"  # UTM 60S
DEFAULT_WIDTH = 10

# ==============================
# Inisialisasi Earth Engine (pakai st.secrets)
# ==============================
# Ambil info service account dari Streamlit Secrets
service_account_info = st.secrets["gee_service_account"]

# Ubah jadi dict Python
service_account_dict = dict(service_account_info)

# Simpan sementara ke file di /tmp (EE butuh file)
KEY_FILE = "/tmp/privatekey.json"
with open(KEY_FILE, "w") as f:
    json.dump(service_account_dict, f)

SERVICE_ACCOUNT = service_account_dict["client_email"]

# Inisialisasi kredensial
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials)

# ==============================
# Fungsi
# ==============================
def extract_polygon_from_kml(kml_path):
    gdf = gpd.read_file(kml_path)
    polygons = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if polygons.empty:
        raise Exception("❌ Tidak ada polygon di file KML.")
    return unary_union(polygons.geometry), polygons.crs

def get_buildings_and_roads_from_gee(polygon):
    coords = list(polygon.exterior.coords)
    ee_poly = ee.Geometry.Polygon(coords)

    # Dataset Bangunan
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v1/polygons").filterBounds(ee_poly)

    # Dataset Jalan
    roads = ee.FeatureCollection("projects/google/OpenStreetMap/roads").filterBounds(ee_poly)

    buildings_gdf = gpd.GeoDataFrame.from_features(buildings.getInfo())
    roads_gdf = gpd.GeoDataFrame.from_features(roads.getInfo())

    return buildings_gdf, roads_gdf

def export_to_dxf(boundary_poly, buildings_gdf, roads_gdf, polygon_crs, dxf_path):
    doc = ezdxf.new()
    msp = doc.modelspace()

    buildings_gdf = buildings_gdf.to_crs(TARGET_EPSG)
    roads_gdf = roads_gdf.to_crs(TARGET_EPSG)
    boundary_poly = gpd.GeoSeries([boundary_poly], crs=polygon_crs).to_crs(TARGET_EPSG).iloc[0]

    bounds = list(boundary_poly.exterior.coords)
    min_x = min(x for x, y in bounds)
    min_y = min(y for x, y in bounds)

    def offset_coords(coords):
        return [(x - min_x, y - min_y) for x, y in coords]

    # Boundary
    if boundary_poly.geom_type == 'Polygon':
        msp.add_lwpolyline(offset_coords(boundary_poly.exterior.coords), dxfattribs={"layer": "BOUNDARY"})
    elif boundary_poly.geom_type == 'MultiPolygon':
        for p in boundary_poly.geoms:
            msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BOUNDARY"})

    # Bangunan
    for geom in buildings_gdf.geometry:
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(offset_coords(geom.exterior.coords), dxfattribs={"layer": "BUILDINGS"})
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BUILDINGS"})

    # Jalan
    for geom in roads_gdf.geometry:
        if isinstance(geom, LineString):
            msp.add_lwpolyline(offset_coords(geom.coords), dxfattribs={"layer": "ROADS"})
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                msp.add_lwpolyline(offset_coords(line.coords), dxfattribs={"layer": "ROADS"})

    doc.set_modelspace_vport(height=10000)
    doc.saveas(dxf_path)

def process_kml_to_dxf(kml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    polygon, polygon_crs = extract_polygon_from_kml(kml_path)
    buildings_gdf, roads_gdf = get_buildings_and_roads_from_gee(polygon)

    if buildings_gdf.empty and roads_gdf.empty:
        raise Exception("❌ Tidak ada bangunan atau jalan ditemukan di area ini.")

    dxf_path = os.path.join(output_dir, "map_utm60.dxf")
    export_to_dxf(polygon, buildings_gdf, roads_gdf, polygon_crs, dxf_path)
    return dxf_path

# ==============================
# UI Streamlit
# ==============================
def run_app():
    st.title("🏗️ KML → DXF (Google Earth Engine, UTM 60)")
    st.caption("Upload file .KML (boundary cluster), hasil: DXF dengan bangunan & jalan dari GEE.")

    kml_file = st.file_uploader("Upload file .KML", type=["kml"])

    if kml_file:
        with st.spinner("💫 Memproses data dari GEE..."):
            try:
                temp_input = f"/tmp/{kml_file.name}"
                with open(temp_input, "wb") as f:
                    f.write(kml_file.read())

                output_dir = "/tmp/output"
                dxf_path = process_kml_to_dxf(temp_input, output_dir)

                st.success("✅ Berhasil diekspor ke DXF (UTM 60)!")
                with open(dxf_path, "rb") as f:
                    st.download_button("⬇️ Download DXF", data=f, file_name="map_utm60.dxf")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

if __name__ == "__main__":
    run_app()
