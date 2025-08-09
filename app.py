# app.py
import os
import zipfile
import tempfile
import json
import time
import streamlit as st
import geopandas as gpd
import ezdxf
import ee
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon, shape, mapping, LineString, MultiLineString
from shapely.ops import unary_union
from pyproj import Transformer

st.set_page_config(page_title="KMZ → DXF (GEE Buildings & Roads)", layout="wide")

# ---------- Config ----------
TARGET_EPSG = "EPSG:32760"   # UTM 60S — ganti jika perlu
DEFAULT_MAX_FEATURES_NONE = True  # kita pakai None (ambil semua) sesuai permintaan

# ---------- Utilities: KMZ / KML ----------
def extract_first_kml_from_kmz(kmz_path):
    """Ekstrak file .kml pertama dari KMZ dan kembalikan path sementara serta nama internal."""
    with zipfile.ZipFile(kmz_path, 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith('.kml'):
                tmp_kml = tempfile.NamedTemporaryFile(delete=False, suffix=".kml").name
                with open(tmp_kml, "wb") as f:
                    f.write(zf.read(name))
                return tmp_kml, name
    raise RuntimeError("Tidak ditemukan file .kml di dalam KMZ.")

def parse_coordinates_text(text):
    """Parse isi <coordinates> tag KML menjadi list (lon, lat) tuples."""
    coords = []
    if not text:
        return coords
    for part in text.strip().split():
        comps = part.split(',')
        if len(comps) >= 2:
            try:
                lon = float(comps[0])
                lat = float(comps[1])
                coords.append((lon, lat))
            except Exception:
                continue
    return coords

def find_boundary_polygons_from_kml(kml_path, folder_name="BOUNDARY CLUSTER"):
    """
    Cari semua polygon di dalam Folder dengan nama folder_name (case-insensitive).
    Jika tidak ditemukan folder yang match, fallback: ambil semua polygon di file KML.
    Return: shapely Polygon or MultiPolygon in EPSG:4326
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    polygons = []

    # Cari folder yang cocok
    for folder in root.findall(".//kml:Folder", ns):
        name_elem = folder.find("kml:name", ns)
        if name_elem is None:
            continue
        if name_elem.text and name_elem.text.strip().upper() == folder_name.upper():
            # Ambil semua coordinates di subtree folder
            for coords_elem in folder.findall(".//kml:coordinates", ns):
                coords = parse_coordinates_text(coords_elem.text)
                if len(coords) >= 4:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid and poly.area > 0:
                            polygons.append(poly)
                    except Exception:
                        continue

    # Fallback: ambil semua polygon di file
    if not polygons:
        for coords_elem in root.findall(".//kml:coordinates", ns):
            coords = parse_coordinates_text(coords_elem.text)
            if len(coords) >= 4:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                try:
                    poly = Polygon(coords)
                    if poly.is_valid and poly.area > 0:
                        polygons.append(poly)
                except Exception:
                    continue

    if not polygons:
        raise RuntimeError("Tidak ditemukan polygon boundary di KML (folder BOUNDARY CLUSTER atau global).")

    merged = unary_union(polygons)
    if merged.is_empty:
        raise RuntimeError("Hasil union polygon kosong.")
    # Pastikan return Polygon / MultiPolygon
    if isinstance(merged, (Polygon, MultiPolygon)):
        return merged
    # Jika geometry collection, filter polygon parts
    poly_parts = [g for g in merged.geoms if isinstance(g, Polygon)]
    if not poly_parts:
        raise RuntimeError("Parsing KML tidak menghasilkan polygon yang valid.")
    return unary_union(poly_parts)

# ---------- Google Earth Engine init & query ----------
def init_ee_from_secrets():
    """Inisialisasi EE dengan service account JSON di st.secrets['gee_service_account']"""
    if "gee_service_account" not in st.secrets:
        raise RuntimeError("st.secrets tidak berisi 'gee_service_account'. Masukkan service account JSON GEE.")
    svc = st.secrets["gee_service_account"]
    if isinstance(svc, str):
        svc_dict = json.loads(svc)
    else:
        svc_dict = dict(svc)
    tmp_key = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    with open(tmp_key, "w") as f:
        json.dump(svc_dict, f)
    client_email = svc_dict.get("client_email")
    if not client_email:
        os.remove(tmp_key)
        raise RuntimeError("Service account JSON tidak berisi 'client_email'.")
    credentials = ee.ServiceAccountCredentials(client_email, tmp_key)
    ee.Initialize(credentials)
    try:
        os.remove(tmp_key)
    except Exception:
        pass

def ee_fc_to_gdf(fc, take_all=True):
    """
    Convert ee.FeatureCollection -> GeoDataFrame.
    If take_all True => use fc.getInfo() (no limit). Beware large results.
    """
    try:
        if take_all:
            info = fc.getInfo()
        else:
            info = fc.limit(10000).getInfo()  # safety default
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data dari GEE: {e}")

    if not info or "features" not in info or len(info["features"]) == 0:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    try:
        gdf = gpd.GeoDataFrame.from_features(info)
    except Exception:
        # fallback: manual build
        features = info.get("features", [])
        geoms = []
        props = []
        for f in features:
            geom = f.get("geometry")
            if geom:
                try:
                    geoms.append(shape(geom))
                    props.append(f.get("properties", {}))
                except Exception:
                    continue
        if geoms:
            gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    return gdf

def query_buildings_and_roads(boundary_shapely, take_all=True):
    """
    Query GEE for buildings and roads within boundary_shapely.
    Return GeoDataFrames in EPSG:4326.
    """
    if isinstance(boundary_shapely, Polygon):
        coords = [list(boundary_shapely.exterior.coords)]
    elif isinstance(boundary_shapely, MultiPolygon):
        coords = [list(next(boundary_shapely.geoms).exterior.coords)]
    else:
        raise RuntimeError("Boundary bukan Polygon / MultiPolygon")

    ee_poly = ee.Geometry.Polygon(coords[0])

    # Open Buildings polygons (Google research)
    buildings_fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v1/polygons").filterBounds(ee_poly)
    # OSM roads (project)
    roads_fc = ee.FeatureCollection("projects/google/OpenStreetMap/roads").filterBounds(ee_poly)

    # Convert
    roads_gdf = ee_fc_to_gdf(roads_fc, take_all=take_all)
    buildings_gdf = ee_fc_to_gdf(buildings_fc, take_all=take_all)

    return buildings_gdf, roads_gdf

# ---------- Export DXF ----------
def export_to_dxf(boundary_shapely, buildings_gdf, roads_gdf, out_path):
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Ensure crs
    if buildings_gdf is None:
        buildings_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    if roads_gdf is None:
        roads_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    if buildings_gdf.crs is None:
        buildings_gdf.set_crs("EPSG:4326", inplace=True)
    if roads_gdf.crs is None:
        roads_gdf.set_crs("EPSG:4326", inplace=True)

    # Reproject to target UTM
    try:
        buildings_utm = buildings_gdf.to_crs(TARGET_EPSG)
        roads_utm = roads_gdf.to_crs(TARGET_EPSG)
    except Exception as e:
        raise RuntimeError(f"Gagal reprojeksi GDF ke {TARGET_EPSG}: {e}")

    # Boundary -> utm
    boundary_series = gpd.GeoSeries([boundary_shapely], crs="EPSG:4326").to_crs(TARGET_EPSG)
    boundary_utm = boundary_series.iloc[0]

    # Compute offset to make DXF coords smaller
    if boundary_utm.geom_type == "Polygon":
        bounds_coords = list(boundary_utm.exterior.coords)
    else:
        bounds_coords = list(next(boundary_utm.geoms).exterior.coords)
    min_x = min(x for x, y in bounds_coords)
    min_y = min(y for x, y in bounds_coords)

    def offset(coords):
        return [(x - min_x, y - min_y) for x, y in coords]

    # Draw boundary
    if boundary_utm.geom_type == "Polygon":
        msp.add_lwpolyline(offset(list(boundary_utm.exterior.coords)), dxfattribs={"layer": "BOUNDARY"})
    else:
        for p in boundary_utm.geoms:
            msp.add_lwpolyline(offset(list(p.exterior.coords)), dxfattribs={"layer": "BOUNDARY"})

    # Draw buildings (polygons) on layer BUILDINGS
    for geom in buildings_utm.geometry:
        if geom is None:
            continue
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(offset(list(geom.exterior.coords)), dxfattribs={"layer": "BUILDINGS"})
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                msp.add_lwpolyline(offset(list(p.exterior.coords)), dxfattribs={"layer": "BUILDINGS"})

    # Draw roads (lines) on layer ROADS
    for geom in roads_utm.geometry:
        if geom is None:
            continue
        if isinstance(geom, LineString):
            msp.add_lwpolyline(offset(list(geom.coords)), dxfattribs={"layer": "ROADS"})
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                msp.add_lwpolyline(offset(list(line.coords)), dxfattribs={"layer": "ROADS"})

    doc.set_modelspace_vport(height=10000)
    doc.saveas(out_path)

# ---------- Streamlit App ----------
def run_app():
    st.title("KMZ → DXF (GEE Buildings & Roads)")
    st.write("Upload KMZ yang berisi boundary (Folder 'BOUNDARY CLUSTER'). Program akan mengambil bangunan & jalan dari GEE sesuai boundary dan meng-export DXF.")
    st.markdown("**Penting:** aplikasi akan mengambil *semua* fitur dari GEE (tanpa limit). Untuk area besar proses bisa memakan waktu dan memory. Pastikan kredensial GEE ada di `st.secrets['gee_service_account']`.")

    uploaded = st.file_uploader("Upload file .kmz", type=["kmz"])
    take_all_checkbox = st.checkbox("Ambil semua fitur dari GEE (tanpa batas)?", value=True, help="Jika dicentang: ambil semua fitur (fc.getInfo()). Untuk area luas, ini bisa berat.")
    # progress placeholders
    progress_bar = st.progress(0)
    progress_text = st.empty()

    if not uploaded:
        st.info("Silakan upload KMZ boundary untuk memulai.")
        return

    # save uploaded kmz to temp
    tmp_kmz = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz").name
    with open(tmp_kmz, "wb") as f:
        f.write(uploaded.read())
    st.success(f"KMZ tersimpan sementara: {os.path.basename(tmp_kmz)}")

    try:
        progress_text.text("10% — Mengekstrak KML dari KMZ...")
        progress_bar.progress(10)
        kml_path, inner_name = extract_first_kml_from_kmz(tmp_kmz)
        progress_text.text(f"20% — Memakai KML internal: {inner_name}")
        progress_bar.progress(20)

        # parse boundary polygon
        progress_text.text("25% — Mencari polygon boundary (folder 'BOUNDARY CLUSTER')...")
        progress_bar.progress(25)
        boundary_poly = find_boundary_polygons_from_kml(kml_path, folder_name="BOUNDARY CLUSTER")
        progress_text.text("35% — Boundary polygon ditemukan.")
        progress_bar.progress(35)
        st.write("Boundary bbox (lon/lat):", boundary_poly.bounds)

        # init GEE
        progress_text.text("40% — Inisialisasi Google Earth Engine...")
        progress_bar.progress(40)
        init_ee_from_secrets()
        progress_text.text("45% — GEE siap.")
        progress_bar.progress(45)

        # query roads
        progress_text.text("55% — Mengambil data jalan dari GEE (OSM roads)...")
        progress_bar.progress(55)
        take_all = True if take_all_checkbox else False
        # query both (we call query_buildings_and_roads, which will fetch both)
        buildings_gdf, roads_gdf = query_buildings_and_roads(boundary_poly, take_all=take_all)
        progress_text.text("75% — Data jalan & bangunan diunduh dari GEE.")
        progress_bar.progress(75)

        # clip to boundary (safety)
        progress_text.text("80% — Memotong (clip) data sesuai boundary...")
        progress_bar.progress(80)
        boundary_gdf = gpd.GeoDataFrame({"geometry": [boundary_poly]}, crs="EPSG:4326")
        try:
            if not buildings_gdf.empty:
                buildings_gdf = gpd.clip(buildings_gdf, boundary_gdf)
            if not roads_gdf.empty:
                roads_gdf = gpd.clip(roads_gdf, boundary_gdf)
        except Exception:
            # fallback: spatial filtering by within/intersects
            if not buildings_gdf.empty:
                buildings_gdf = buildings_gdf[buildings_gdf.geometry.intersects(boundary_poly)]
            if not roads_gdf.empty:
                roads_gdf = roads_gdf[roads_gdf.geometry.intersects(boundary_poly)]
        progress_text.text("85% — Selesai memotong data.")
        progress_bar.progress(85)

        # simple checks
        st.write("Jumlah fitur bangunan setelah clip:", 0 if buildings_gdf.empty else len(buildings_gdf))
        st.write("Jumlah fitur jalan setelah clip:", 0 if roads_gdf.empty else len(roads_gdf))

        # export to DXF
        progress_text.text("90% — Menyusun dan mengekspor DXF...")
        progress_bar.progress(90)
        out_dxf = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
        export_to_dxf(boundary_poly, buildings_gdf, roads_gdf, out_dxf)
        progress_text.text("100% — Selesai. DXF siap diunduh.")
        progress_bar.progress(100)
        st.success("DXF telah dibuat.")

        with open(out_dxf, "rb") as f:
            st.download_button("⬇️ Download DXF", f, file_name="map_buildings_roads.dxf")

    except Exception as e:
        st.error(f"Gagal: {e}")
    finally:
        # cleanup temp files if exist
        try:
            if os.path.exists(tmp_kmz):
                os.remove(tmp_kmz)
        except Exception:
            pass
        try:
            if 'kml_path' in locals() and os.path.exists(kml_path):
                os.remove(kml_path)
        except Exception:
            pass

if __name__ == "__main__":
    run_app()
