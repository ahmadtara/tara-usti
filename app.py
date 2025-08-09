# app.py
import os
import zipfile
import tempfile
import json
import datetime
import requests
import streamlit as st
import ezdxf
import ee
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon, shape, Point, mapping, LineString
from shapely.ops import unary_union
from pyproj import Transformer
import numpy as np
import cv2
import rasterio
from rasterio import transform as rio_transform
from ultralytics import YOLO

st.set_page_config(page_title="KMZ → DXF (Satellite + YOLOv8 Building Detection)", layout="wide")

# Config
TARGET_EPSG = "EPSG:32760"  # ganti sesuai kebutuhan
DESIRED_SCALE = 0.5  # meters per pixel
PREFERRED_COLLECTIONS = [
    "USDA/NAIP/DOQQ",
    "COPERNICUS/S2_SR"
]

# --- Helpers KMZ / KML ---
def extract_first_kml_from_kmz(kmz_path):
    with zipfile.ZipFile(kmz_path, 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith('.kml'):
                tmp_kml = tempfile.NamedTemporaryFile(delete=False, suffix=".kml").name
                with open(tmp_kml, "wb") as f:
                    f.write(zf.read(name))
                return tmp_kml, name
    raise RuntimeError("Tidak ditemukan file .kml di dalam KMZ.")

def parse_coordinates_text(text):
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
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    polygons = []
    for folder in root.findall(".//kml:Folder", ns):
        name_elem = folder.find("kml:name", ns)
        if name_elem is None:
            continue
        if name_elem.text and name_elem.text.strip().upper() == folder_name.upper():
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
        raise RuntimeError("Tidak ditemukan polygon boundary di KML.")

    merged = unary_union(polygons)
    if merged.is_empty:
        raise RuntimeError("Hasil union polygon kosong.")
    if isinstance(merged, (Polygon, MultiPolygon)):
        return merged
    poly_parts = [g for g in merged.geoms if isinstance(g, Polygon)]
    if not poly_parts:
        raise RuntimeError("Parsing KML tidak menghasilkan polygon valid.")
    return unary_union(poly_parts)

# --- Earth Engine init ---
def init_ee_from_st_secrets():
    if "gee_service_account" not in st.secrets:
        raise RuntimeError("st.secrets tidak berisi 'gee_service_account'. Masukkan service account JSON.")
    svc = st.secrets["gee_service_account"]
    svc_dict = json.loads(svc) if isinstance(svc, str) else dict(svc)
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

# --- Get best image from GEE ---
def get_best_image_for_region(region_geojson, start_date_str, end_date_str, desired_scale=DESIRED_SCALE):
    if not start_date_str or not end_date_str:
        raise RuntimeError("start_date or end_date kosong.")

    ee_start = ee.Date(start_date_str)
    ee_end = ee.Date(end_date_str)

    for coll_id in PREFERRED_COLLECTIONS:
        try:
            coll = ee.ImageCollection(coll_id).filterBounds(ee.Geometry(region_geojson)).filterDate(ee_start, ee_end)
            count = coll.size().getInfo()
            if count and int(count) > 0:
                img = coll.median().clip(ee.Geometry(region_geojson))
                st.info(f"Memakai koleksi: {coll_id} (count={count}).")
                return img, coll_id
        except Exception:
            continue

    try:
        coll = ee.ImageCollection("COPERNICUS/S2").filterBounds(ee.Geometry(region_geojson)).filterDate(ee_start, ee_end)
        if coll.size().getInfo() > 0:
            img = coll.median().clip(ee.Geometry(region_geojson))
            st.info("Fallback memakai COPERNICUS/S2.")
            return img, "COPERNICUS/S2"
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil imagery dari GEE: {e}")

    raise RuntimeError("Tidak ada koleksi imagery tersedia di area ini untuk rentang tanggal yang diberikan.")

# --- Download GeoTIFF ---
def download_image_to_geotiff(ee_image, region_geojson, scale, out_path):
    params = {
        'scale': float(scale),
        'region': json.dumps(region_geojson),
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    }
    url = ee_image.getDownloadURL(params)
    r = requests.get(url, stream=True, timeout=1200)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return out_path

# --- Detect buildings (YOLOv8) and roads ---
def detect_buildings_and_roads_from_geotiff(geotiff_path, boundary_shapely, model, progress_callback=None):
    buildings = []
    roads = []

    with rasterio.open(geotiff_path) as src:
        bands = src.count
        transform = src.transform

        if bands >= 3:
            arr = src.read([1,2,3])
        else:
            arr = src.read(1)[None,:,:]

        if arr.shape[0] >= 3:
            img = np.dstack([arr[0], arr[1], arr[2]])
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            # dari RGB ke BGR tidak perlu, cukup RGB
            img_for_yolo = img
        else:
            gray = arr[0]
            if gray.dtype != np.uint8:
                gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
            img_for_yolo = np.stack([gray]*3, axis=-1)  # buat jadi 3 channel

    if progress_callback:
        progress_callback(60, "Menjalankan YOLOv8 untuk deteksi bangunan...")

    results = model.predict(source=img_for_yolo, conf=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if progress_callback:
        progress_callback(75, f"YOLOv8 deteksi {len(boxes)} bounding box.")

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        poly_px = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        lonlat_poly = []
        for px, py in poly_px:
            x, y = rio_transform.xy(transform, int(py), int(px))
            lonlat_poly.append((x, y))

        poly = Polygon(lonlat_poly)
        if poly.centroid.within(boundary_shapely):
            buildings.append(poly)

    if progress_callback:
        progress_callback(80, "Mendeteksi jalan...")

    # Deteksi jalan tanpa cv2: gunakan skimage
    # Konversi ke grayscale
    gray_img = color.rgb2gray(img_for_yolo)

    # Edge detection pakai Canny
    edges = feature.canny(gray_img, sigma=1.0)

    # Perbaiki edges dengan dilasi
    edges_dilated = morphology.dilation(edges, morphology.square(3))

    # Line detection pakai probabilistic Hough transform
    lines = sk_transform.probabilistic_hough_line(edges_dilated, threshold=10, line_length=30, line_gap=10)

    for line in lines:
        (x1, y1), (x2, y2) = line
        lon1, lat1 = rio_transform.xy(transform, int(y1), int(x1))
        lon2, lat2 = rio_transform.xy(transform, int(y2), int(x2))
        line_geom = LineString([(lon1, lat1), (lon2, lat2)])
        if line_geom.intersects(boundary_shapely):
            roads.append(line_geom)

    return buildings, roads


# --- Export ke DXF dengan layer terpisah ---
def export_to_dxf(boundary_shapely, buildings_list, roads_list, out_path):
    doc = ezdxf.new()
    msp = doc.modelspace()
    transformer = Transformer.from_crs("epsg:4326", TARGET_EPSG, always_xy=True)

    if isinstance(boundary_shapely, Polygon):
        bpolys = [boundary_shapely]
    else:
        bpolys = list(boundary_shapely.geoms)

    all_coords = []
    for p in bpolys:
        all_coords.extend(list(p.exterior.coords))
    if all_coords:
        min_x = min(x for x,y in all_coords)
        min_y = min(y for x,y in all_coords)
    else:
        min_x = 0
        min_y = 0

    def to_utm_and_offset(coords):
        out = []
        for lon, lat in coords:
            x,y = transformer.transform(lon, lat)
            out.append((x - min_x, y - min_y))
        return out

    # Boundary di layer "BOUNDARY"
    for p in bpolys:
        coords = to_utm_and_offset(list(p.exterior.coords))
        msp.add_lwpolyline(coords, dxfattribs={"layer": "BOUNDARY"})

    # Bangunan di layer "BANGUNAN"
    for poly in buildings_list:
        try:
            coords = to_utm_and_offset(list(poly.exterior.coords))
            msp.add_lwpolyline(coords, dxfattribs={"layer": "BANGUNAN"})
        except Exception:
            continue

    # Jalan di layer "JALAN"
    for line in roads_list:
        try:
            coords = to_utm_and_offset(list(line.coords))
            msp.add_lwpolyline(coords, dxfattribs={"layer": "JALAN"})
        except Exception:
            continue

    doc.set_modelspace_vport(height=10000)
    doc.saveas(out_path)

# --- Main Streamlit app ---
def run_app():
    st.title("KMZ → DXF (Satellite + YOLOv8 Building Detection)")
    st.markdown("""
    Upload file KMZ boundary.
    App akan mengambil citra resolusi tinggi dari GEE, menjalankan YOLOv8 untuk deteksi bangunan,
    dan mendeteksi jalan menggunakan OpenCV, lalu export ke DXF dengan layer terpisah.
    """)
    st.info("Pastikan `st.secrets['gee_service_account']` sudah berisi service account JSON Earth Engine.")

    uploaded = st.file_uploader("Upload file .kmz", type=["kmz"])
    st.write("Pilih rentang tanggal citra (GEE):")
    today = datetime.date.today()
    default_start = today.replace(year=max(2000, today.year - 3))
    start_date_input = st.date_input("Tanggal mulai", value=default_start)
    end_date_input = st.date_input("Tanggal akhir", value=today)
    run_button = st.button("Mulai proses (ambil citra & deteksi)")

    progress_bar = st.progress(0)
    status = st.empty()

    if not uploaded:
        return

    # Load YOLO model sekali ketika tombol ditekan
    model = None

    if run_button:
        # Validasi tanggal
        if start_date_input > end_date_input:
            st.error("Tanggal mulai harus lebih awal dari tanggal akhir.")
            return

        try:
            status.text("10% — Load model YOLOv8...")
            progress_bar.progress(10)
            model_path = "yolov8s.pt"
            if not os.path.exists(model_path):
                st.error(f"Model YOLO tidak ditemukan: {model_path}")
                return
            model = YOLO(model_path)
            status.text("20% — Model YOLOv8 siap.")
            progress_bar.progress(20)
        except Exception as e:
            st.error(f"Gagal load model YOLOv8: {e}")
            return

        try:
            tmp_kmz = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz").name
            with open(tmp_kmz, "wb") as f:
                f.write(uploaded.read())
            status.text("25% — Ekstrak KML dari KMZ...")
            progress_bar.progress(25)
            kml_path, inner_name = extract_first_kml_from_kmz(tmp_kmz)
            status.text(f"30% — Menggunakan internal KML: {inner_name}")
            progress_bar.progress(30)
        except Exception as e:
            st.error(f"Gagal ekstrak KML: {e}")
            return

        try:
            status.text("35% — Cari polygon boundary di KML...")
            progress_bar.progress(35)
            boundary = find_boundary_polygons_from_kml(kml_path, folder_name="BOUNDARY CLUSTER")
            status.text("45% — Boundary ditemukan.")
            progress_bar.progress(45)
            st.write("Boundary bbox (lon/lat):", boundary.bounds)
        except Exception as e:
            st.error(f"Gagal parsing boundary: {e}")
            return

        try:
            status.text("50% — Inisialisasi Earth Engine...")
            progress_bar.progress(50)
            init_ee_from_st_secrets()
            status.text("55% — Earth Engine siap.")
            progress_bar.progress(55)
        except Exception as e:
            st.error(f"Gagal inisialisasi EE: {e}")
            return

        try:
            status.text("60% — Cari koleksi imagery terbaik...")
            progress_bar.progress(60)
            region_geojson = mapping(boundary)
            start_str = start_date_input.isoformat()
            end_str = end_date_input.isoformat()
            img, coll_used = get_best_image_for_region(region_geojson, start_str, end_str, DESIRED_SCALE)
            status.text(f"65% — Koleksi terpilih: {coll_used}")
            progress_bar.progress(65)
        except Exception as e:
            st.error(f"Gagal pilih imagery: {e}")
            return

        try:
            status.text("70% — Download GeoTIFF dari GEE (bisa lama)...")
            progress_bar.progress(70)
            out_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
            download_image_to_geotiff(img, mapping(boundary), DESIRED_SCALE, out_tif)
            status.text("75% — GeoTIFF diunduh.")
            progress_bar.progress(75)
        except Exception as e:
            st.error(f"Gagal unduh GeoTIFF: {e}")
            return

        try:
            def progress_cb(pct, txt):
                if pct < 100:
                    status.text(f"{pct}% — {txt}")
                    progress_bar.progress(pct)
            status.text("80% — Jalankan deteksi bangunan dan jalan...")
            progress_bar.progress(80)
            buildings, roads = detect_buildings_and_roads_from_geotiff(out_tif, boundary, model, progress_callback=progress_cb)
            status.text("95% — Deteksi selesai.")
            progress_bar.progress(95)
            st.write("Jumlah bangunan terdeteksi:", len(buildings))
            st.write("Jumlah segmen jalan terdeteksi:", len(roads))
        except Exception as e:
            st.error(f"Gagal proses citra: {e}")
            return

        try:
            status.text("98% — Buat file DXF...")
            progress_bar.progress(98)
            out_dxf = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
            export_to_dxf(boundary, buildings, roads, out_dxf)
            status.text("100% — Selesai. DXF siap diunduh.")
            progress_bar.progress(100)
            st.success("DXF siap.")
            with open(out_dxf, "rb") as f:
                st.download_button("⬇️ Download DXF", f, file_name="map_sat_yolov8.dxf")
        except Exception as e:
            st.error(f"Gagal ekspor DXF: {e}")
            return
        finally:
            # Hapus tmp files
            try: os.remove(tmp_kmz)
            except Exception: pass
            try: os.remove(kml_path)
            except Exception: pass

if __name__ == "__main__":
    run_app()

