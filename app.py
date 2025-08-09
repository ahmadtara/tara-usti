# app.py
import os
import zipfile
import tempfile
import json
import time
import requests
import streamlit as st
import geopandas as gpd
import ezdxf
import ee
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon, shape, Point, mapping, LineString
from shapely.ops import unary_union
from pyproj import Transformer
import numpy as np
import cv2
import rasterio
from rasterio import windows

st.set_page_config(page_title="KMZ → DXF (Satellite + Rectangle Detection)", layout="wide")

# ---------- Config ----------
TARGET_EPSG = "EPSG:32760"    # target reprojection for DXF (UTM zone 60S). ganti jika perlu
DESIRED_SCALE = 0.5           # meters per pixel target (attempt)
PREFERRED_COLLECTIONS = [
    # prioritized collections to attempt for higher-res imagery
    # NOTE: many commercial collections require access (Maxar/Planet).
    # If you have a licensed collection name, place it here first.
    # Example placeholders (may require access): "MAXAR/2D", "PLANET/..."
    # We'll try NAIP (US only, ~1m) then Sentinel-2 (~10m) as fallback.
    "USDA/NAIP/DOQQ",           # NAIP (approx 1m) — US only
    # "OPENAI/MAXAR_SAMPLE",    # (placeholder) — commercial; uncomment if available
    "COPERNICUS/S2_SR"          # Sentinel-2 surface reflectance (~10m)
]

# ---------- Helpers: KMZ / KML ----------
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
    # fallback: all coords in file
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
    # filter polygon parts
    poly_parts = [g for g in merged.geoms if isinstance(g, Polygon)]
    if not poly_parts:
        raise RuntimeError("Parsing KML tidak menghasilkan polygon valid.")
    return unary_union(poly_parts)

# ---------- Initialize Earth Engine ----------
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

# ---------- Imagery: select best available collection ----------
def get_best_image_for_region(region_geojson, desired_scale=DESIRED_SCALE):
    """
    Try preferred collections; return ee.Image clipped to region if available.
    If none high-res available, return combined mosaic (e.g., Sentinel-2 median).
    """
    # Try each candidate
    for coll_id in PREFERRED_COLLECTIONS:
        try:
            coll = ee.ImageCollection(coll_id).filterBounds(ee.Geometry(region_geojson)).filterDate('2018-01-01', ee.Date().format('YYYY-MM-dd'))
            count = coll.size().getInfo()
            if count and int(count) > 0:
                # pick median composite
                img = coll.median().clip(ee.Geometry(region_geojson))
                st.info(f"Memakai koleksi: {coll_id} (count={count}).")
                return img, coll_id
        except Exception:
            # collection may not exist or access denied
            continue
    # Fallback: try Sentinel-2 global
    try:
        coll = ee.ImageCollection("COPERNICUS/S2").filterBounds(ee.Geometry(region_geojson)).filterDate('2019-01-01', ee.Date().format('YYYY-MM-dd'))
        if coll.size().getInfo() > 0:
            img = coll.median().clip(ee.Geometry(region_geojson))
            st.info("Fallback memakai COPERNICUS/S2.")
            return img, "COPERNICUS/S2"
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil imagery dari GEE: {e}")
    raise RuntimeError("Tidak ada koleksi imagery tersedia di area ini.")

# ---------- Download GeoTIFF from ee.Image ----------
def download_image_to_geotiff(ee_image, region_geojson, scale, out_path, max_pixels=1e9):
    """
    Request a GeoTIFF download URL from EE and download to out_path.
    Returns out_path when downloaded.
    """
    # build download parameters
    params = {
        'scale': float(scale),
        'region': json.dumps(region_geojson),
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    }
    try:
        url = ee_image.getDownloadURL(params)
    except Exception as e:
        raise RuntimeError(f"Gagal membuat download URL dari EE: {e}")

    # download via requests
    try:
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Gagal mengunduh GeoTIFF dari URL: {e}")

    return out_path

# ---------- Image processing: detect buildings (rectangles) & roads (lines) ----------
def detect_buildings_and_roads_from_geotiff(geotiff_path, boundary_shapely, progress_callback=None):
    """
    - Open GeoTIFF with rasterio
    - Convert to grayscale or pan-sharpened composite
    - Process with OpenCV: findContours -> approxPolyDP untuk rectangles
    - For roads: Canny -> HoughLinesP
    Returns:
      list_of_building_polygons_in_lonlat, list_of_lines_in_lonlat
    """
    buildings = []
    roads = []

    with rasterio.open(geotiff_path) as src:
        # read an RGB composite if possible (bands order depends on file)
        bands = src.count
        # prefer first 3 bands as RGB if available
        if bands >= 3:
            arr = src.read([1,2,3])  # shape (3, H, W)
        else:
            arr = src.read(1)[None,:,:]  # single band

        # build transform for pixel->lonlat
        transform = src.transform
        height = src.height
        width = src.width

        # convert to uint8 image for OpenCV
        if arr.shape[0] >= 3:
            img = np.dstack([arr[0], arr[1], arr[2]])
            # normalize if needed
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr[0]
            if gray.dtype != np.uint8:
                gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)

        # Optional: equalize / denoise
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        # Building detection via adaptive threshold + morphological ops
        # Adjust parameters if needed
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # invert if background white
        white_ratio = np.mean(th == 255)
        if white_ratio > 0.6:
            th = 255 - th

        # morphological opening to remove small noise, closing to fill building roofs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        # find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # progress: contour detection done
        if progress_callback:
            progress_callback(60, "Memproses kontur untuk menemukan bangunan...")

        # iterate contours and approximate polygons
        for cnt in contours:
            # ignore small areas
            area = cv2.contourArea(cnt)
            if area < 50:  # tune threshold as needed
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 4 and len(approx) <= 12:
                # get polygon pixel coords
                poly_px = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
                # convert pixel coords to lonlat using rasterio transform
                lonlat_poly = []
                for px, py in poly_px:
                    # rasterio transform: xy = transform * (col, row)
                    x, y = rasterio.transform.xy(transform, py, px)  # note: row=y, col=x
                    lonlat_poly.append((x, y))
                # check if polygon centroid inside boundary
                centroid = Point(np.mean([p[0] for p in lonlat_poly]), np.mean([p[1] for p in lonlat_poly]))
                if not centroid.within(boundary_shapely):
                    # optional: skip if outside
                    continue
                buildings.append(Polygon(lonlat_poly))

        # Roads detection: use Canny on grayscale then HoughLinesP
        if progress_callback:
            progress_callback(75, "Mendeteksi tepi untuk jalan...")
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # dilate edges to make lines more continuous
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
        # HoughLinesP expects (x,y) coords; returns lines in pixel coords
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                # convert to lonlat
                lon1, lat1 = rasterio.transform.xy(transform, y1, x1)
                lon2, lat2 = rasterio.transform.xy(transform, y2, x2)
                line = LineString([(lon1, lat1), (lon2, lat2)])
                # optionally clip to boundary: keep only if intersects
                if line.intersects(boundary_shapely):
                    roads.append(line)

    return buildings, roads

# ---------- Convert to DXF ----------
def export_to_dxf_single_layer(boundary_shapely, buildings_list, roads_list, out_path):
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Reproject everything to TARGET_EPSG before writing
    transformer = Transformer.from_crs("epsg:4326", TARGET_EPSG, always_xy=True)

    # Convert boundary
    if isinstance(boundary_shapely, Polygon):
        bpolys = [boundary_shapely]
    else:
        bpolys = list(boundary_shapely.geoms)

    # compute global min for offset
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

    # add buildings & roads to same layer "MAP"
    layer_name = "MAP"

    # boundary (draw outer poly)
    for p in bpolys:
        coords = to_utm_and_offset(list(p.exterior.coords))
        msp.add_lwpolyline(coords, dxfattribs={"layer": layer_name})

    # buildings
    for poly in buildings_list:
        try:
            coords = to_utm_and_offset(list(poly.exterior.coords))
            msp.add_lwpolyline(coords, dxfattribs={"layer": layer_name})
        except Exception:
            continue

    # roads
    for line in roads_list:
        try:
            coords = to_utm_and_offset(list(line.coords))
            msp.add_lwpolyline(coords, dxfattribs={"layer": layer_name})
        except Exception:
            continue

    doc.set_modelspace_vport(height=10000)
    doc.saveas(out_path)

# ---------- Streamlit App ----------
def run_app():
    st.title("KMZ → DXF (Satellite + Rectangle Detection)")
    st.markdown("""
    Upload KMZ boundary. App akan mencoba mengambil citra resolusi tinggi (attempt 0.5 m/pix),
    lalu mendeteksi kotak bangunan (approxPolyDP) dan jalan (Canny+Hough), lalu export DXF (single layer).
    """)
    st.info("Pastikan `st.secrets['gee_service_account']` sudah berisi service account JSON untuk Earth Engine.")

    uploaded = st.file_uploader("Upload file .kmz", type=["kmz"])
    take_all_checkbox = st.checkbox("Ambil citra resolusi tertinggi yang tersedia (attempt 0.5m).", value=True)
    run_button = st.button("Mulai proses (ambil citra & deteksi)")

    # progress UI
    progress_bar = st.progress(0)
    status = st.empty()

    if not uploaded:
        return

    if run_button:
        tmp_kmz = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz").name
        with open(tmp_kmz, "wb") as f:
            f.write(uploaded.read())
        status.text("10% — Ekstrak KML dari KMZ...")
        progress_bar.progress(10)

        try:
            kml_path, inner_name = extract_first_kml_from_kmz(tmp_kmz)
            status.text(f"15% — Menggunakan internal KML: {inner_name}")
            progress_bar.progress(15)
        except Exception as e:
            st.error(f"Gagal ekstrak KML: {e}")
            return

        try:
            status.text("20% — Mencari polygon boundary di KML...")
            progress_bar.progress(20)
            boundary = find_boundary_polygons_from_kml(kml_path, folder_name="BOUNDARY CLUSTER")
            status.text("30% — Boundary ditemukan.")
            progress_bar.progress(30)
            st.write("Boundary bbox (lon/lat):", boundary.bounds)
        except Exception as e:
            st.error(f"Gagal parsing boundary: {e}")
            return

        # init EE
        try:
            status.text("35% — Inisialisasi Earth Engine...")
            progress_bar.progress(35)
            init_ee_from_st_secrets()
            status.text("40% — Earth Engine siap.")
            progress_bar.progress(40)
        except Exception as e:
            st.error(f"Gagal inisialisasi EE: {e}")
            return

        # request best image
        try:
            status.text("45% — Mencari koleksi imagery terbaik untuk area...")
            progress_bar.progress(45)
            region_geojson = mapping(boundary)  # shapely -> geojson mapping
            img, coll_used = get_best_image_for_region(region_geojson, DESIRED_SCALE)
            status.text(f"50% — Koleksi terpilih: {coll_used}")
            progress_bar.progress(50)
        except Exception as e:
            st.error(f"Gagal pilih imagery: {e}")
            return

        # request download
        try:
            status.text("55% — Membuat dan mengunduh GeoTIFF dari GEE (ini bisa lama)...")
            progress_bar.progress(55)
            out_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
            # attempt scale param DESIRED_SCALE; note: some collections ignore very small scale
            download_region = region_geojson
            download_image_to_geotiff(img, download_region, DESIRED_SCALE, out_tif)
            status.text("65% — GeoTIFF diunduh.")
            progress_bar.progress(65)
        except Exception as e:
            st.error(f"Gagal unduh GeoTIFF: {e}")
            return

        # image processing
        try:
            def progress_cb(pct, txt):
                if pct < 100:
                    status.text(f"{pct}% — {txt}")
                    progress_bar.progress(pct)
            status.text("70% — Memulai deteksi (OpenCV)...")
            progress_bar.progress(70)
            buildings, roads = detect_buildings_and_roads_from_geotiff(out_tif, boundary, progress_callback=progress_cb)
            status.text("88% — Deteksi selesai.")
            progress_bar.progress(88)
            st.write("Jumlah bangunan terdeteksi:", len(buildings))
            st.write("Jumlah segmen jalan terdeteksi:", len(roads))
        except Exception as e:
            st.error(f"Gagal proses citra: {e}")
            return

        # export DXF
        try:
            status.text("92% — Membuat DXF...")
            progress_bar.progress(92)
            out_dxf = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
            export_to_dxf_single_layer(boundary, buildings, roads, out_dxf)
            status.text("100% — Selesai. DXF siap diunduh.")
            progress_bar.progress(100)
            st.success("DXF siap.")
            with open(out_dxf, "rb") as f:
                st.download_button("⬇️ Download DXF", f, file_name="map_sat_rect_dflt.dxf")
        except Exception as e:
            st.error(f"Gagal ekspor DXF: {e}")
            return
        finally:
            # cleanup temp files (optional keep for debugging)
            try:
                if os.path.exists(tmp_kmz): os.remove(tmp_kmz)
            except Exception: pass
            try:
                if os.path.exists(kml_path): os.remove(kml_path)
            except Exception: pass

if __name__ == "__main__":
    run_app()
