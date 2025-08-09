import os
import json
import tempfile
import streamlit as st
import geopandas as gpd
import ezdxf
import ee
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import LinearRing

# --- Konfigurasi ---
TARGET_EPSG = "EPSG:32760"  # UTM 60S (sesuaikan kalau mau EPSG lain)
DEFAULT_MAX_FEATURES = 2000  # batasi request ke GEE supaya tidak timeout

# ==============================
# Inisialisasi Earth Engine (pakai st.secrets)
# ==============================
def init_ee_from_streamlit_secrets():
    if "gee_service_account" not in st.secrets:
        raise RuntimeError("Service account GEE tidak ditemukan di streamlit secrets (st.secrets['gee_service_account']).")
    service_account_info = st.secrets["gee_service_account"]

    # service_account_info bisa berupa dict atau string-json
    if isinstance(service_account_info, str):
        service_account_dict = json.loads(service_account_info)
    else:
        service_account_dict = dict(service_account_info)

    # Simpan sementara ke file karena ee.ServiceAccountCredentials butuh file path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    KEY_FILE = tmp.name
    with open(KEY_FILE, "w") as f:
        json.dump(service_account_dict, f)

    SERVICE_ACCOUNT = service_account_dict.get("client_email")
    if not SERVICE_ACCOUNT:
        os.remove(KEY_FILE)
        raise RuntimeError("client_email tidak ditemukan dalam service account JSON.")

    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials)

    # hapus file kunci jika mau (EE sudah inisialisasi)
    try:
        os.remove(KEY_FILE)
    except Exception:
        pass

# ==============================
# Fungsi pembantu
# ==============================

def read_kml_to_polygons(kml_path):
    """
    Baca KML dan kembalikan (merged_polygon, original_crs).
    Bisa mengkonversi garis tertutup menjadi polygon bila perlu.
    """
    # coba baca dengan driver KML
    try:
        gdf = gpd.read_file(kml_path, driver="KML")
    except Exception as e:
        # jika gagal, oper error yang lebih jelas
        raise RuntimeError(f"Gagal membaca KML dengan geopandas (pastikan GDAL/OGR mendukung driver KML). Error: {e}")

    # Jika crs None, asumsikan WGS84 (EPSG:4326)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    # Normalize geometri: ubah LineString tertutup jadi Polygon
    new_geoms = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        gtype = geom.geom_type
        if gtype in ["LineString"]:
            # Cek tertutup (first == last) -> ubah jadi Polygon
            coords = list(geom.coords)
            if len(coords) >= 4 and coords[0] == coords[-1]:
                try:
                    poly = Polygon(coords)
                    new_geoms.append(poly)
                    continue
                except Exception:
                    pass
        if gtype == "MultiLineString":
            # coba gabungkan lines yang membentuk ring
            converted = []
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    try:
                        converted.append(Polygon(coords))
                    except Exception:
                        pass
            if converted:
                # gabungkan jadi MultiPolygon
                new_geoms.append(MultiPolygon(converted))
                continue
        # default: pakai geom apa adanya
        new_geoms.append(geom)

    # buat GeoDataFrame sementara kalau perlu
    gdf2 = gdf.copy()
    gdf2.geometry = new_geoms

    # Pecah (explode) geometry collection / multiparts
    try:
        gdf2 = gdf2.explode(index_parts=False, ignore_index=True)
    except TypeError:
        # fallback untuk geopandas tua
        gdf2 = gdf2.explode().reset_index(drop=True)

    # Filter hanya polygon / multipolygon dan area > 0
    polys = gdf2[gdf2.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    polys["area"] = polys.geometry.area
    polys = polys[polys["area"] > 0]

    if polys.empty:
        # Jika tidak ada polygon, beri pesan agar user mengecek KML
        raise RuntimeError("File KML tidak mengandung polygon aktif. Jika boundary berupa garis tertutup, pastikan koordinat awal dan akhir sama sehingga dapat diubah menjadi polygon.")

    merged = unary_union(polys.geometry)
    if merged.is_empty or merged.geom_type not in ["Polygon", "MultiPolygon"]:
        raise RuntimeError(f"Geometry hasil merge tidak valid. Tipe: {merged.geom_type}")

    return merged, polys.crs

def gee_featurecollection_to_geodataframe(fc, max_features=DEFAULT_MAX_FEATURES):
    """
    Ambil sebagian FeatureCollection dari GEE dan konversi ke GeoDataFrame.
    Gunakan limit(max_features) untuk menghindari download berlebih.
    """
    try:
        info = fc.limit(int(max_features)).getInfo()
    except ee.EEException as e:
        raise RuntimeError(f"GEE getInfo() gagal (kemungkinan area terlalu besar atau quota). Pesan: {e}")
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data dari GEE. Pesan: {e}")

    if not info or "features" not in info or len(info["features"]) == 0:
        # kembalikan GeoDataFrame kosong
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    try:
        gdf = gpd.GeoDataFrame.from_features(info)
    except Exception:
        # fallback: buat manual dari fitur
        features = info.get("features", [])
        geom_list = []
        props_list = []
        for f in features:
            geom = f.get("geometry")
            props = f.get("properties", {})
            if geom:
                geom_list.append(gpd.GeoSeries.from_geojson(json.dumps(geom)).iloc[0])
                props_list.append(props)
        if geom_list:
            gdf = gpd.GeoDataFrame(props_list, geometry=geom_list, crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    # Pastikan CRS terpasang
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    return gdf

def get_buildings_and_roads_from_gee(polygon, max_features=DEFAULT_MAX_FEATURES):
    # polygon adalah shapely polygon (WGS84 assumed?) - kita ambil koordinatnya dalam lon/lat
    if polygon.geom_type == "Polygon":
        coords = [list(polygon.exterior.coords)]
    elif polygon.geom_type == "MultiPolygon":
        # ambil semua polygon, tapi untuk filterBounds kita bisa pakai polygon pertama
        coords = [list(next(polygon.geoms).exterior.coords)]
    else:
        raise RuntimeError(f"Geometry tipe {polygon.geom_type} tidak didukung untuk query GEE.")

    ee_poly = ee.Geometry.Polygon(coords[0])

    # Feature collections (batasi jumlah fitur)
    buildings_fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v1/polygons").filterBounds(ee_poly)
    roads_fc = ee.FeatureCollection("projects/google/OpenStreetMap/roads").filterBounds(ee_poly)

    buildings_gdf = gee_featurecollection_to_geodataframe(buildings_fc, max_features=max_features)
    roads_gdf = gee_featurecollection_to_geodataframe(roads_fc, max_features=max_features)

    return buildings_gdf, roads_gdf

def export_to_dxf(boundary_poly, buildings_gdf, roads_gdf, polygon_crs, dxf_path):
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Pastikan semua gdf punya CRS; jika None, anggap EPSG:4326
    if buildings_gdf is None:
        buildings_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=polygon_crs or "EPSG:4326")
    if roads_gdf is None:
        roads_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=polygon_crs or "EPSG:4326")

    if buildings_gdf.crs is None:
        buildings_gdf.set_crs(polygon_crs or "EPSG:4326", inplace=True)
    if roads_gdf.crs is None:
        roads_gdf.set_crs(polygon_crs or "EPSG:4326", inplace=True)

    # reprojeksi ke target
    try:
        buildings_gdf = buildings_gdf.to_crs(TARGET_EPSG)
        roads_gdf = roads_gdf.to_crs(TARGET_EPSG)
    except Exception as e:
        raise RuntimeError(f"Gagal transform CRS features ke {TARGET_EPSG}. Pesan: {e}")

    boundary_series = gpd.GeoSeries([boundary_poly], crs=polygon_crs or "EPSG:4326")
    boundary_series = boundary_series.to_crs(TARGET_EPSG)
    boundary_poly_utm = boundary_series.iloc[0]

    bounds = list(boundary_poly_utm.exterior.coords)
    min_x = min(x for x, y in bounds)
    min_y = min(y for x, y in bounds)

    def offset_coords(coords):
        return [(x - min_x, y - min_y) for x, y in coords]

    # Boundary
    if boundary_poly_utm.geom_type == 'Polygon':
        msp.add_lwpolyline(offset_coords(boundary_poly_utm.exterior.coords), dxfattribs={"layer": "BOUNDARY"})
    elif boundary_poly_utm.geom_type == 'MultiPolygon':
        for p in boundary_poly_utm.geoms:
            msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BOUNDARY"})

    # Buildings
    for geom in buildings_gdf.geometry:
        if geom is None:
            continue
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(offset_coords(geom.exterior.coords), dxfattribs={"layer": "BUILDINGS"})
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BUILDINGS"})
        else:
            # kalau geometry point/line skip
            continue

    # Roads (lines)
    for geom in roads_gdf.geometry:
        if geom is None:
            continue
        if isinstance(geom, LineString):
            msp.add_lwpolyline(offset_coords(geom.coords), dxfattribs={"layer": "ROADS"})
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                msp.add_lwpolyline(offset_coords(line.coords), dxfattribs={"layer": "ROADS"})

    doc.set_modelspace_vport(height=10000)
    doc.saveas(dxf_path)

def process_kml_to_dxf(kml_path, output_dir, max_features=DEFAULT_MAX_FEATURES):
    os.makedirs(output_dir, exist_ok=True)

    boundary_poly, polygon_crs = read_kml_to_polygons(kml_path)
    buildings_gdf, roads_gdf = get_buildings_and_roads_from_gee(boundary_poly, max_features=max_features)

    if buildings_gdf.empty and roads_gdf.empty:
        raise RuntimeError("Tidak ada bangunan atau jalan ditemukan di area ini (hasil kosong). Coba periksa area KML apakah terlalu kecil/terlalu jauh dari dataset GEE, atau naikkan max_features.")

    dxf_path = os.path.join(output_dir, "map_utm60.dxf")
    export_to_dxf(boundary_poly, buildings_gdf, roads_gdf, polygon_crs, dxf_path)
    return dxf_path

# ==============================
# Streamlit UI
# ==============================
def run_app():
    st.title("🏗️ KML → DXF (Google Earth Engine, UTM 60)")
    st.caption("Upload file .KML (boundary cluster). Kode akan mengambil bangunan & jalan dari GEE dan mengekspor DXF.")

    # inisialisasi EE
    try:
        init_ee_from_streamlit_secrets()
    except Exception as e:
        st.error(f"Gagal inisialisasi Earth Engine: {e}")
        st.stop()

    st.markdown("**Pengaturan:**")
    max_features = st.number_input("Batas fitur GEE (max_features)", value=DEFAULT_MAX_FEATURES, min_value=100, max_value=100000, step=100)
    output_dir = "/tmp/output"

    kml_file = st.file_uploader("Upload file .KML", type=["kml"])
    if not kml_file:
        st.info("Silakan upload file KML boundary.")
        return

    if st.button("Proses dan Ekspor ke DXF"):
        with st.spinner("Memproses... silakan tunggu (tergantung ukuran area dan koneksi ke GEE)..."):
            try:
                temp_input = os.path.join(tempfile.gettempdir(), kml_file.name)
                with open(temp_input, "wb") as f:
                    f.write(kml_file.read())

                dxf_path = process_kml_to_dxf(temp_input, output_dir, max_features=max_features)

                st.success("Berhasil diekspor ke DXF (UTM 60).")
                with open(dxf_path, "rb") as f:
                    st.download_button("⬇️ Download DXF", data=f, file_name=os.path.basename(dxf_path))

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    run_app()
