import streamlit as st
import zipfile
import os
import tempfile
from xml.etree import ElementTree as ET
import ezdxf
from pyproj import Transformer
import math

st.set_page_config(page_title="KMZ → DXF (BOUNDARY CLUSTER Only)", layout="wide")

def extract_kml_from_kmz(kmz_path):
    """Ekstrak file doc.kml dari KMZ."""
    with zipfile.ZipFile(kmz_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith(".kml"):
                tmp_kml = tempfile.NamedTemporaryFile(delete=False, suffix=".kml").name
                with open(tmp_kml, "wb") as f:
                    f.write(zf.read(name))
                return tmp_kml
    raise RuntimeError("Tidak ada file .kml di dalam KMZ")

def parse_kml_for_boundary(kml_file):
    """Ambil koordinat poligon dari folder BOUNDARY CLUSTER."""
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Namespace KML
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    polygons = []
    texts = []

    for folder in root.findall(".//kml:Folder", ns):
        name_elem = folder.find("kml:name", ns)
        if name_elem is not None and name_elem.text.strip().upper() == "BOUNDARY CLUSTER":
            for pm in folder.findall(".//kml:Placemark", ns):
                pm_name = pm.find("kml:name", ns)
                coords_elem = pm.find(".//kml:coordinates", ns)
                if coords_elem is not None:
                    coords = []
                    for coord in coords_elem.text.strip().split():
                        lon, lat, *_ = map(float, coord.split(","))
                        coords.append((lon, lat))
                    if pm_name is not None and "HP COVER" in pm_name.text.upper():
                        texts.append((pm_name.text, coords))
                    else:
                        polygons.append(coords)
    return polygons, texts

def lonlat_to_utm(lon, lat):
    """Konversi koordinat Lon/Lat ke UTM."""
    transformer = Transformer.from_crs("epsg:4326", "epsg:32748", always_xy=True)  # UTM zona 48S (Indonesia barat)
    return transformer.transform(lon, lat)

def point_in_polygon(x, y, poly):
    """Cek apakah titik (x, y) ada di dalam poligon."""
    inside = False
    n = len(poly)
    px, py = zip(*poly)
    for i in range(n):
        j = (i - 1) % n
        if ((py[i] > y) != (py[j] > y)) and \
           (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]):
            inside = not inside
    return inside

def nearest_polygon_center(x, y, polygons):
    """Cari pusat poligon terdekat."""
    min_dist = float("inf")
    nearest_center = None
    for poly in polygons:
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        dist = math.dist((x, y), (cx, cy))
        if dist < min_dist:
            min_dist = dist
            nearest_center = (cx, cy)
    return nearest_center

def create_dxf(polygons, texts, output_path):
    """Buat file DXF dengan teks di dalam kotak terdekat."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Gambar poligon
    for poly in polygons:
        msp.add_lwpolyline(poly, close=True)

    # Tempatkan teks di dalam kotak terdekat
    for text, coords in texts:
        # Ambil titik rata-rata koordinat teks
        avg_lon = sum(p[0] for p in coords) / len(coords)
        avg_lat = sum(p[1] for p in coords) / len(coords)
        x, y = lonlat_to_utm(avg_lon, avg_lat)

        # Cari kotak terdekat
        nearest_center = nearest_polygon_center(x, y, polygons)
        if nearest_center:
            msp.add_text(text, dxfattribs={"height": 2.5}).set_pos(nearest_center, align="CENTER")
        else:
            msp.add_text(text, dxfattribs={"height": 2.5}).set_pos((x, y), align="CENTER")

    doc.saveas(output_path)

st.title("KMZ → DXF Converter (BOUNDARY CLUSTER Only)")

uploaded_kmz = st.file_uploader("Upload file KMZ", type=["kmz"])
if uploaded_kmz:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp_kmz:
        tmp_kmz.write(uploaded_kmz.read())
        kmz_path = tmp_kmz.name

    try:
        kml_file = extract_kml_from_kmz(kmz_path)
        polygons_raw, texts_raw = parse_kml_for_boundary(kml_file)

        # Konversi koordinat Lon/Lat ke UTM
        polygons = [[lonlat_to_utm(lon, lat) for lon, lat in poly] for poly in polygons_raw]
        texts = [(txt, [(lon, lat) for lon, lat in coords]) for txt, coords in texts_raw]

        # Simpan DXF
        output_dxf = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
        create_dxf(polygons, texts, output_dxf)

        with open(output_dxf, "rb") as f:
            st.download_button("Download DXF", f, file_name="boundary_cluster.dxf")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
