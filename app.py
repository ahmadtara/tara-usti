import os
import json
import tempfile
import zipfile
import streamlit as st
import geopandas as gpd
import ezdxf
import ee
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union

# --- Konfigurasi ---
TARGET_EPSG = "EPSG:32760"  # UTM 60S
DEFAULT_MAX_FEATURES = 2000  # batasi request GEE supaya tidak timeout
TARGET_KML_NAME = "BOUNDARY CLUSTER.kml"  # nama file di dalam KMZ

# ==============================
# Inisialisasi Earth Engine
# ==============================
def init_ee_from_streamlit_secrets():
    if "gee_service_account" not in st.secrets:
        raise RuntimeError("Service account GEE tidak ditemukan di streamlit secrets.")
    service_account_info = st.secrets["gee_service_account"]

    if isinstance(service_account_info, str):
        service_account_dict = json.loads(service_account_info)
    else:
        service_account_dict = dict(service_account_info)

    tmp_key = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    with open(tmp_key, "w") as f:
        json.dump(service_account_dict, f)

    SERVICE_ACCOUNT = service_account_dict.get("client_email")
    if not SERVICE_ACCOUNT:
        os.remove(tmp_key)
        raise RuntimeError("client_email tidak ditemukan dalam service account JSON.")

    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, tmp_key)
    ee.Initialize(credentials)

    try:
        os.remove(tmp_key)
    except Exception:
        pass

# ==============================
# Ekstrak KML dari KMZ
# ==============================
def extract_boundary_kml_from_kmz(kmz_path, target_name=TARGET_KML_NAME):
    with zipfile.ZipFile(kmz_path, 'r') as zf:
        names = zf.namelist()
        match = None
        for name in names:
            if os.path.basename(name).lower() == target_name.lower():
                match = name
                break
        if not match:
            raise RuntimeError(f"File '{target_name}' tidak ditemukan di dalam KMZ. Isinya: {names}")

        tmp_kml = tempfile.NamedTemporaryFile(delete=False, suffix=".kml").name
        with open(tmp_kml, "wb") as f:
            f.write(zf.read(match))
        return tmp_kml

# ==============================
# Fungsi pembaca KML → Polygon
# ==============================
def read_kml_to_polygons(kml_path):
    try:
        gdf = gpd.read_file(kml_path, driver="KML")
    except Exception as e:
        raise RuntimeError(f"Gagal membaca KML. Error: {e}")

    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    # Ubah LineString tertutup → Polygon
    new_geoms = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            if len(coords) >= 4 and coords[0] == coords[-1]:
                try:
                    new_geoms.append(Polygon(coords))
                    continue
                except Exception:
                    pass
        elif geom.geom_type == "MultiLineString":
            polys = []
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    try:
                        polys.append(Polygon(coords))
                    except Exception:
                        pass
            if polys:
                new_geoms.append(MultiPolygon(polys))
                continue
        new_geoms.append(geom)

    gdf.geometry = new_geoms

    try:
        gdf = gdf.explode(index_parts=False, ignore_index=True)
    except TypeError:
        gdf = gdf.explode().reset_index(drop=True)

    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    polys["area"] = polys.geometry.area
    polys = polys[polys["area"] > 0]

    if polys.empty:
        raise RuntimeError("KML tidak mengandung polygon. Pastikan boundary benar.")

    merged = unary_union(polys.geometry)
    if merged.is_empty or merged.geom_type not in ["Polygon", "MultiPolygon"]:
        raise RuntimeError(f"Geometry hasil merge tidak valid. Tipe: {merged.geom_type}")

    return merged, polys.crs

# ==============================
# Ambil data GEE
# ==============================
def gee_featurecollection_to_geodataframe(fc, max_features=DEFAULT_MAX_FEATURES):
    try:
        info = fc.limit(int(max_features)).getInfo()
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data dari GEE: {e}")

    if not info or "features" not in info or len(info["features"]) == 0:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame.from_features(info)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    return gdf

def get_buildings_and_roads_from_gee(polygon, max_features=DEFAULT_MAX_FEATURES):
    if polygon.geom_type == "Polygon":
        coords = [list(polygon.exterior.coords)]
    elif polygon.geom_type == "MultiPolygon":
        coords = [list(next(polygon.geoms).exterior.coords)]
    else:
        raise RuntimeError(f"Geometry tipe {polygon.geom_type} tidak didukung.")

    ee_poly = ee.Geometry.Polygon(coords[0])

    buildings_fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v1/polygons").filterBounds(ee_poly)
    roads_fc = ee.FeatureCollection("projects/google/OpenStreetMap/roads").filterBounds(ee_poly)

    buildings_gdf = gee_featurecollection_to_geodataframe(buildings_fc, max_features=max_features)
    roads_gdf = gee_featurecollection_to_geodataframe(roads_fc, max_features=max_features)

    return buildings_gdf, roads_gdf

# ==============================
# Ekspor DXF
# ==============================
def export_to_dxf(boundary_poly, buildings_gdf, roads_gdf, polygon_crs, dxf_path):
    doc = ezdxf.new()
    msp = doc.modelspace()

    if buildings_gdf.crs is None:
        buildings_gdf.set_crs(polygon_crs or "EPSG:4326", inplace=True)
    if roads_gdf.crs is None:
        roads_gdf.set_crs(polygon_crs or "EPSG:4326", inplace=True)

    buildings_gdf = buildings_gdf.to_crs(TARGET_EPSG)
    roads_gdf = roads_gdf.to_crs(TARGET_EPSG)
    boundary_series = gpd.GeoSeries([boundary_poly], crs=polygon_crs or "EPSG:4326").to_crs(TARGET_EPSG)
    boundary_poly_utm = boundary_series.iloc[0]

    bounds = list(boundary_poly_utm.exterior.coords)
    min_x = min(x for x, y in bounds)
    min_y = min(y for x, y in bounds)

    def offset_coords(coords):
        return [(x - min_x, y - min_y) for x, y in coords]

    # Boundary
    if boundary_poly_utm.geom_type == "Polygon":
        msp.add_lwpolyline(offset_coords(boundary_poly_utm.exterior.coords), dxfattribs={"layer": "BOUNDARY"})
    elif boundary_poly_utm.geom_type == "MultiPolygon":
        for p in boundary_poly_utm.geoms:
            msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BOUNDARY"})

    # Buildings
    for geom in buildings_gdf.geometry:
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(offset_coords(geom.exterior.coords), dxfattribs={"layer": "BUILDINGS"})
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BUILDINGS"})

    # Roads
    for geom in roads_gdf.geometry:
        if isinstance(geom, LineString):
            msp.add_lwpolyline(offset_coords(geom.coords), dxfattribs={"layer": "ROADS"})
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                msp.add_lwpolyline(offset_coords(line.coords), dxfattribs={"layer": "ROADS"})

    doc.set_modelspace_vport(height=10000)
    doc.saveas(dxf_path)

# ==============================
# Proses KMZ → DXF
# ==============================
def process_kmz_to_dxf(kmz_path, output_dir, max_features=DEFAULT_MAX_FEATURES):
    os.makedirs(output_dir, exist_ok=True)

    temp_kml = extract_boundary_kml_from_kmz(kmz_path)
    boundary_poly, polygon_crs = read_kml_to_polygons(temp_kml)
    buildings_gdf, roads_gdf = get_buildings_and_roads_from_gee(boundary_poly, max_features=max_features)

    if buildings_gdf.empty and roads_gdf.empty:
        raise RuntimeError("Tidak ada bangunan atau jalan ditemukan di area ini.")

    dxf_path = os.path.join(output_dir, "map_utm60.dxf")
    export_to_dxf(boundary_poly, buildings_gdf, roads_gdf, polygon_crs, dxf_path)
    return dxf_path

# ==============================
# Streamlit UI
# ==============================
def run_app():
    st.title("🏗️ KMZ → DXF (Google Earth Engine, UTM 60)")
    st.caption(f"Upload file .KMZ berisi '{TARGET_KML_NAME}', hasil: DXF dengan bangunan & jalan.")

    try:
        init_ee_from_streamlit_secrets()
    except Exception as e:
        st.error(f"Gagal inisialisasi Earth Engine: {e}")
        st.stop()

    max_features = st.number_input("Batas fitur GEE", value=DEFAULT_MAX_FEATURES, min_value=100, max_value=100000, step=100)
    output_dir = "/tmp/output"

    kmz_file = st.file_uploader("Upload file .KMZ", type=["kmz"])
    if not kmz_file:
        st.info("Silakan upload file KMZ.")
        return

    if st.button("Proses dan Ekspor ke DXF"):
        with st.spinner("Memproses data dari KMZ..."):
            try:
                temp_input = os.path.join(tempfile.gettempdir(), kmz_file.name)
                with open(temp_input, "wb") as f:
                    f.write(kmz_file.read())

                dxf_path = process_kmz_to_dxf(temp_input, output_dir, max_features=max_features)

                st.success("✅ Berhasil diekspor ke DXF!")
                with open(dxf_path, "rb") as f:
                    st.download_button("⬇️ Download DXF", data=f, file_name=os.path.basename(dxf_path))

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

if __name__ == "__main__":
    run_app()
