import os
import zipfile
import tempfile
import json
import streamlit as st
import geopandas as gpd
import ezdxf
import ee
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, shape
from shapely.ops import unary_union
from pyproj import Transformer

st.set_page_config(page_title="KMZ → DXF (GEE Buildings & Roads)", layout="wide")

# ----- Konfigurasi -----
TARGET_EPSG = "EPSG:32760"   # UTM zone 60S (sesuaikan jika perlu)
DEFAULT_MAX_FEATURES = None   # None = ambil semua fitur dari GEE (tanpa limit)

# ----- Utilities KML / KMZ -----
def extract_first_kml_from_kmz(kmz_path):
    """Ekstrak file .kml pertama (biasanya doc.kml) dari KMZ ke temp file."""
    with zipfile.ZipFile(kmz_path, 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith('.kml'):
                tmp_kml = tempfile.NamedTemporaryFile(delete=False, suffix=".kml").name
                with open(tmp_kml, "wb") as f:
                    f.write(zf.read(name))
                return tmp_kml, name
    raise RuntimeError("Tidak ditemukan file .kml di dalam KMZ.")

def extract_coordinates_from_coordinates_element(text):
    """Parse teks <coordinates>... dan kembalikan list (lon,lat) tuples."""
    coords = []
    if not text:
        return coords
    # coordinates bisa dipisah spasi/linebreak; setiap item: lon,lat[,alt]
    for item in text.strip().split():
        parts = item.split(',')
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except Exception:
                continue
    return coords

def find_boundary_polygons_from_kml(kml_path, folder_name="BOUNDARY CLUSTER"):
    """
    Cari Folder dengan name == folder_name (case-insensitive),
    lalu ambil semua <coordinates> dalam folder itu (term: Placemark/Polygon/LinearRing).
    Kembalikan shapely Polygon / MultiPolygon yang merupakan union dari semua polygon ditemukan.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML namespace (umumnya)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # temukan semua Folder
    polygons = []
    # Cari folder yang nama nya match folder_name (case-insensitive)
    for folder in root.findall(".//kml:Folder", ns):
        name_elem = folder.find("kml:name", ns)
        if name_elem is None:
            continue
        if name_elem.text and name_elem.text.strip().upper() == folder_name.upper():
            # dalam folder ini, cari semua <coordinates> element di subtree
            for coords_elem in folder.findall(".//kml:coordinates", ns):
                coords = extract_coordinates_from_coordinates_element(coords_elem.text)
                # pastikan bentuk polygon (minimal 4 titik dan tertutup)
                if len(coords) >= 4:
                    # jika first != last, coba tutup
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid and poly.area > 0:
                            polygons.append(poly)
                    except Exception:
                        continue
    # Jika tidak ditemukan folder bernama itu, coba fallback: ambil semua polygons di file KML
    if not polygons:
        for coords_elem in root.findall(".//kml:coordinates", ns):
            coords = extract_coordinates_from_coordinates_element(coords_elem.text)
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
        raise RuntimeError("Tidak ditemukan polygon boundary dalam KML (baik di folder BOUNDARY CLUSTER maupun global).")

    merged = unary_union(polygons)
    if merged.is_empty:
        raise RuntimeError("Hasil union polygon kosong.")
    if isinstance(merged, (Polygon, MultiPolygon)):
        return merged
    else:
        # Jika bukan polygon, coba filter geometri polygon saja
        polys = [g for g in merged.geoms if isinstance(g, Polygon)]
        if not polys:
            raise RuntimeError("Hasil parsing KML tidak menghasilkan polygon yang valid.")
        return unary_union(polys)

# ----- Inisialisasi GEE -----
def init_ee_from_st_secrets():
    if "gee_service_account" not in st.secrets:
        raise RuntimeError("st.secrets tidak berisi 'gee_service_account'. Masukkan service account JSON GEE ke Streamlit Secrets.")
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

# ----- Ambil data dari GEE -----
def gee_fc_to_gdf(fc, max_features=None):
    """
    Konversi ee.FeatureCollection ke GeoDataFrame.
    Jika max_features is None -> ambil semua (fc.getInfo()).
    WARNING: untuk area besar ini bisa memakan waktu / memori.
    """
    try:
        if max_features is None:
            info = fc.getInfo()
        else:
            info = fc.limit(int(max_features)).getInfo()
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data dari GEE: {e}")

    if not info or "features" not in info or len(info["features"]) == 0:
        # kosong
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    try:
        gdf = gpd.GeoDataFrame.from_features(info)
    except Exception as e:
        # fallback manual
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

def query_buildings_and_roads(polygon_shapely, max_features=None):
    """
    polygon_shapely: shapely Polygon or MultiPolygon in lon/lat (EPSG:4326)
    Returns: buildings_gdf, roads_gdf (both in EPSG:4326)
    """
    # build ee.Geometry from polygon (use exterior of first polygon if multipoly)
    if isinstance(polygon_shapely, Polygon):
        coords = [list(polygon_shapely.exterior.coords)]
    elif isinstance(polygon_shapely, MultiPolygon):
        coords = [list(next(polygon_shapely.geoms).exterior.coords)]
    else:
        raise RuntimeError("Polygon input bukan Polygon/MultiPolygon")

    ee_poly = ee.Geometry.Polygon(coords[0])

    # Feature collections GEE
    buildings_fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v1/polygons").filterBounds(ee_poly)
    roads_fc = ee.FeatureCollection("projects/google/OpenStreetMap/roads").filterBounds(ee_poly)

    buildings_gdf = gee_fc_to_gdf(buildings_fc, max_features=max_features)
    roads_gdf = gee_fc_to_gdf(roads_fc, max_features=max_features)

    return buildings_gdf, roads_gdf

# ----- Export ke DXF -----
def export_to_dxf(boundary_poly, buildings_gdf, roads_gdf, out_path):
    """
    boundary_poly: shapely Polygon/MultiPolygon in EPSG:4326
    buildings_gdf, roads_gdf: GeoDataFrame in EPSG:4326 (or empty)
    Export to DXF in TARGET_EPSG with layers: BOUNDARY, BUILDINGS, ROADS
    """
    doc = ezdxf.new()
    msp = doc.modelspace()

    # pastikan crs
    if buildings_gdf is None:
        buildings_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    if roads_gdf is None:
        roads_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    if buildings_gdf.crs is None:
        buildings_gdf.set_crs("EPSG:4326", inplace=True)
    if roads_gdf.crs is None:
        roads_gdf.set_crs("EPSG:4326", inplace=True)

    # reprojeksi semua ke target UTM
    try:
        buildings_utm = buildings_gdf.to_crs(TARGET_EPSG)
        roads_utm = roads_gdf.to_crs(TARGET_EPSG)
    except Exception as e:
        raise RuntimeError(f"Gagal reproject GeoDataFrame ke {TARGET_EPSG}: {e}")

    # boundary to UTM
    boundary_series = gpd.GeoSeries([boundary_poly], crs="EPSG:4326").to_crs(TARGET_EPSG)
    boundary_utm = boundary_series.iloc[0]

    # compute offset to make coords relative (optional)
    # use min x,y of boundary to shift coordinates so DXF coordinates are small
    if boundary_utm.geom_type == 'Polygon':
        bounds = list(boundary_utm.exterior.coords)
    else:
        # MultiPolygon: join first exterior
        bounds = list(next(boundary_utm.geoms).exterior.coords)
    min_x = min(x for x, y in bounds)
    min_y = min(y for x, y in bounds)

    def offset_coords(coords):
        return [(x - min_x, y - min_y) for x, y in coords]

    # Boundary layer
    if boundary_utm.geom_type == "Polygon":
        msp.add_lwpolyline(offset_coords(boundary_utm.exterior.coords), dxfattribs={"layer": "BOUNDARY"})
    else:
        for p in boundary_utm.geoms:
            msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BOUNDARY"})

    # Buildings layer (polygons)
    for geom in buildings_utm.geometry:
        if geom is None:
            continue
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(offset_coords(geom.exterior.coords), dxfattribs={"layer": "BUILDINGS"})
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                msp.add_lwpolyline(offset_coords(p.exterior.coords), dxfattribs={"layer": "BUILDINGS"})
        else:
            # if geometry is a LineString/Point skip
            continue

    # Roads layer (lines)
    for geom in roads_utm.geometry:
        if geom is None:
            continue
        if isinstance(geom, LineString):
            msp.add_lwpolyline(offset_coords(list(geom.coords)), dxfattribs={"layer": "ROADS"})
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                msp.add_lwpolyline(offset_coords(list(line.coords)), dxfattribs={"layer": "ROADS"})
        else:
            continue

    doc.set_modelspace_vport(height=10000)
    doc.saveas(out_path)

# ----- Streamlit UI -----
def run_app():
    st.title("KMZ → DXF (ambil Buildings & Roads dari GEE)")
    st.write("Upload file `.kmz` yang berisi boundary (contoh: doc.kml dengan Folder 'BOUNDARY CLUSTER').")
    st.write("**Catatan:** aplikasi akan mengunduh semua fitur dari GEE (tanpa limit). Untuk area besar proses bisa lama atau gagal karena memory/timeout.")

    # inisialisasi GEE
    try:
        init_ee_from_st_secrets()
    except Exception as e:
        st.error(f"Gagal inisialisasi GEE: {e}")
        st.stop()

    uploaded = st.file_uploader("Upload KMZ", type=["kmz"])
    max_features_option = st.checkbox("Batasi jumlah fitur GEE? (uncheck = ambil semua)", value=False)
    max_features_value = None
    if max_features_option:
        max_features_value = st.number_input("Max fitur (masukkan angka)", min_value=100, max_value=200000, value=2000, step=100)

    if uploaded:
        with st.spinner("Menyimpan dan mengekstrak KMZ..."):
            tmp_kmz = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz").name
            with open(tmp_kmz, "wb") as f:
                f.write(uploaded.read())
        st.success("KMZ tersimpan.")

        try:
            kml_path, inner_name = extract_first_kml_from_kmz(tmp_kmz)
            st.info(f"Memakai KML dari KMZ: `{inner_name}`")
        except Exception as e:
            st.error(f"Gagal ekstrak KML dari KMZ: {e}")
            raise

        try:
            st.info("Mencari polygon boundary di dalam KML (folder 'BOUNDARY CLUS
