import os
import zipfile
import tempfile
import json
import time
import datetime
import io
import traceback

import streamlit as st
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon, mapping, shape, Point, LineString
from shapely.ops import unary_union
from pyproj import Transformer
import numpy as np
from PIL import Image
import rasterio
from rasterio import transform as rio_transform
import cv2
from ultralytics import YOLO  # optional; use your segmentation framework
import ezdxf
import ee
import requests
from kmz_parser import parse_kmz
from shapely.ops import unary_union

st.set_page_config(page_title="KMZ → DXF (GEE)", layout="wide")

# ---------- CONFIG ----------
TARGET_EPSG = "EPSG:32760"   # output projection for DXF (ubah sesuai kebutuhan)
DESIRED_SCALE = 0.5          # meter per pixel target untuk download (coba 0.5 atau 0.3 jika tersedia)
PREFERRED_COLLECTIONS = [
    # Prioritas: tambahkan koleksi yang akunmu punya akses jika perlu
    "COPERNICUS/S2_SR",          # Sentinel-2 (10m) - fallback
    "LANDSAT/LC08/C01/T1_SR",    # Landsat-8 (30m)
    # Jika akun punya akses ke koleksi satelit resolusi tinggi, masukkan di awal:
    # "MAXAR/..., GOOGLE/GOOGLE_SATELLITE" -- tetapi ketersediaan bergantung akun
]

# ---------- Helpers: KML/KMZ ----------
def fix_geometry(geom):
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom
    except Exception as e:
        print(f"[WARNING] Gagal perbaiki geometry: {e}")
        return None

def load_kmz_with_fix(kmz_path):
    geoms = parse_kmz(kmz_path)
    fixed_geoms = []
    for g in geoms:
        g = fix_geometry(g)
        if g:
            fixed_geoms.append(g)
    return fixed_geoms

if __name__ == "__main__":
    kmz_file = "data/contoh.kmz"
    geoms = load_kmz_with_fix(kmz_file)

    if not geoms:
        print("Tidak ada geometry valid yang ditemukan.")
    else:
        merged_geom = unary_union(geoms)
        print(f"Berhasil load {len(geoms)} geometry valid dari KMZ.")

def extract_first_kml_from_kmz(kmz_path):
    with zipfile.ZipFile(kmz_path, 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith('.kml'):
                tmp_kml = tempfile.NamedTemporaryFile(delete=False, suffix=".kml").name
                with open(tmp_kml, "wb") as f:
                    f.write(zf.read(name))
                return tmp_kml, name
    raise RuntimeError("Tidak ditemukan file .kml di dalam KMZ.")

def parse_polygons_from_kml(kml_path, folder_name=None):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(kml_path)
    root = tree.getroot()
    polygons = []
    # jika folder_name diberikan, cari folder tersebut dulu
    if folder_name:
        for folder in root.findall(".//kml:Folder", ns):
            name_elem = folder.find("kml:name", ns)
            if name_elem is not None and name_elem.text and name_elem.text.strip().upper() == folder_name.strip().upper():
                for coords in folder.findall(".//kml:coordinates", ns):
                    pts = _coords_text_to_list(coords.text)
                    if len(pts) >= 4:
                        polygons.append(Polygon(pts))
    # fallback: semua coordinates
    if not polygons:
        for coords in root.findall(".//kml:coordinates", ns):
            pts = _coords_text_to_list(coords.text)
            if len(pts) >= 4:
                polygons.append(Polygon(pts))
    if not polygons:
        raise RuntimeError("Tidak menemukan polygon di KML.")
    merged = unary_union(polygons)
    return merged

def _coords_text_to_list(txt):
    pts = []
    if not txt:
        return pts
    for part in txt.strip().split():
        comps = part.split(',')
        if len(comps) >= 2:
            try:
                lon = float(comps[0]); lat = float(comps[1])
                pts.append((lon, lat))
            except:
                pass
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts

# ---------- Earth Engine init ----------
def init_ee_from_st_secrets():
    if "gee_service_account" not in st.secrets:
        raise RuntimeError("st.secrets tidak berisi 'gee_service_account' (service account JSON).")
    svc = st.secrets["gee_service_account"]
    svc_dict = json.loads(svc) if isinstance(svc, str) else dict(svc)
    tmp_key = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    with open(tmp_key, "w") as f:
        json.dump(svc_dict, f)
    try:
        client_email = svc_dict.get("client_email")
        if not client_email:
            raise RuntimeError("Service account JSON tidak berisi 'client_email'.")
        credentials = ee.ServiceAccountCredentials(client_email, tmp_key)
        ee.Initialize(credentials)
    finally:
        try:
            os.remove(tmp_key)
        except:
            pass

# ---------- Choose best image from collections ----------
def get_best_image_for_region(region_geojson, start_date_str, end_date_str, desired_scale=DESIRED_SCALE):
    if not start_date_str or not end_date_str:
        raise RuntimeError("start_date or end_date kosong.")
    ee_start = ee.Date(start_date_str)
    ee_end = ee.Date(end_date_str)
    geom = ee.Geometry(region_geojson)

    # Try preferred collections first (user can customize PREFERRED_COLLECTIONS)
    for coll_id in PREFERRED_COLLECTIONS:
        try:
            coll = ee.ImageCollection(coll_id).filterBounds(geom).filterDate(ee_start, ee_end)
            count = int(coll.size().getInfo())
            if count > 0:
                st.info(f"Memakai koleksi: {coll_id} (count={count})")
                img = coll.median().clip(geom)
                return img, coll_id
        except Exception as e:
            # skip if collection not available
            st.warning(f"Skip collection {coll_id}: {e}")
            continue

    # OPTIONAL: try some high-res options (may not be available)
    extras = ["GOOGLE/GOOGLE_SATELLITE", "MAPBOX/NOAA/2014"]  # contoh; kenyataannya availability bergantung akun
    for coll_id in extras:
        try:
            coll = ee.ImageCollection(coll_id).filterBounds(geom).filterDate(ee_start, ee_end)
            if coll.size().getInfo() > 0:
                img = coll.median().clip(geom)
                st.info(f"Memakai koleksi fallback: {coll_id}")
                return img, coll_id
        except Exception:
            continue

    raise RuntimeError("Tidak ada imagery ditemukan untuk area dan rentang tanggal yang diberikan. Periksa PREFERRED_COLLECTIONS atau akses akun GEE.")

# ---------- Download GeoTIFF from ee.Image ----------
def download_image_to_geotiff(ee_image, region_geojson, scale, out_path, crs='EPSG:4326'):
    # Build params. Note: getDownloadURL uses 'scale' in meters if crs is geographic? Use appropriate crs.
    params = {
        'scale': float(scale),
        'region': json.dumps(region_geojson),
        'format': 'GEO_TIFF',
        'crs': crs
    }
    try:
        url = ee_image.getDownloadURL(params)
    except Exception as e:
        raise RuntimeError(f"Gagal membuat download URL dari EE: {e}\n{traceback.format_exc()}")
    # download
    try:
        r = requests.get(url, stream=True, timeout=1200)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Gagal mengunduh GeoTIFF: {e}")
    return out_path

# ---------- Simple segmentation using YOLOv8-seg (user supplies .pt) ----------
def run_segmentation_on_geotiff(geotiff_path, model_path, conf=0.25, use_gpu=False):
    """
    Returns lists of shapely polygons (buildings) and lines/polygons (roads) in lon/lat CRS (same as GeoTIFF)
    """
    # read raster for pixel<->geo transform
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        arr = None
        # read RGB if possible
        if src.count >= 3:
            arr = np.dstack([src.read(1), src.read(2), src.read(3)])
        else:
            arr = src.read(1)
            arr = np.dstack([arr, arr, arr])
        height, width = arr.shape[0], arr.shape[1]

    # save temp image for model
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    im = Image.fromarray(((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8))
    im.save(tmp_png)

    # run model
    device = 0 if use_gpu else "cpu"
    model = YOLO(model_path)
    results = model.predict(source=tmp_png, imgsz=1024, conf=conf, device=device)
    res = results[0]
    masks = getattr(res, "masks", None)
    if masks is None:
        st.warning("Model tidak mengeluarkan masks (pastikan model segmentation).")
        return [], []

    mask_arrs = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else masks.data
    cls_arr = masks.boxes.cls.cpu().numpy().astype(int) if hasattr(masks.boxes.cls, "cpu") else masks.boxes.cls

    buildings = []
    roads = []
    for i, m in enumerate(mask_arrs):
        cls_id = int(cls_arr[i]) if len(cls_arr) > i else 0
        # heuristik class name
        class_name = str(cls_id)
        bin_img = (m * 255).astype('uint8')
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            pts_px = [(int(p[0][0]), int(p[0][1])) for p in approx]
            # convert pixel -> lonlat using rasterio.transform.xy
            geo_coords = [rasterio.transform.xy(transform, int(y), int(x), offset='center') for x,y in pts_px]
            poly = Polygon([(lon, lat) for lon, lat in geo_coords])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area == 0:
                continue
            # simple heuristics: if narrow long -> road
            minx, miny, maxx, maxy = poly.bounds
            w = maxx - minx; h = maxy - miny
            aspect = (w/h) if h>0 else 999
            if aspect > 4 or aspect < 0.25:
                # treat centerline
                roads.append(poly.exterior)
            else:
                buildings.append(poly)
    return buildings, roads

# ---------- DXF export ----------
def export_to_dxf(boundary_geom, buildings_geo, roads_geo, out_path, target_epsg=TARGET_EPSG):
    transformer = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
    doc = ezdxf.new()
    msp = doc.modelspace()
    # boundary
    try:
        if isinstance(boundary_geom, (Polygon, MultiPolygon)):
            parts = [boundary_geom] if isinstance(boundary_geom, Polygon) else list(boundary_geom.geoms)
            for p in parts:
                coords = [(transformer.transform(x, y)) for x, y in list(p.exterior.coords)]
                msp.add_lwpolyline(coords, dxfattribs={"layer":"BOUNDARY"})
    except Exception as e:
        st.warning(f"Error boundary -> DXF: {e}")

    for b in buildings_geo:
        try:
            coords = [(transformer.transform(x, y)) for x, y in list(b.exterior.coords)]
            msp.add_lwpolyline(coords, dxfattribs={"layer":"BUILDING"})
        except Exception:
            continue

    for r in roads_geo:
        try:
            coords = [(transformer.transform(x, y)) for x, y in list(r.coords)]
            msp.add_lwpolyline(coords, dxfattribs={"layer":"ROAD"})
        except Exception:
            continue

    doc.saveas(out_path)

# ---------- Streamlit UI ----------
def run_app():
    st.title("KMZ → DXF (Google Earth Engine source)")
    st.markdown("""
    Alur:
    1. Upload KMZ boundary
    2. Inisialisasi GEE via st.secrets (service account)
    3. Pilih tanggal & scale, ambil imagery dari GEE (coba koleksi di PREFERRED_COLLECTIONS)
    4. Download GeoTIFF, jalankan segmentation (upload model .pt)
    5. Convert -> DXF
    """)

    uploaded = st.file_uploader("Upload .kmz (boundary)", type=["kmz"])
    start_date = st.date_input("Tanggal mulai", value=(datetime.date.today() - datetime.timedelta(days=365)))
    end_date = st.date_input("Tanggal akhir", value=datetime.date.today())
    desired_scale = st.number_input("Desired scale (meter/pixel) untuk download GeoTIFF", min_value=0.1, max_value=10.0, value=DESIRED_SCALE, step=0.1)
    uploaded_model = st.file_uploader("Upload model segmentation (.pt) — yolov8-seg recommended", type=["pt"])
    use_gpu = st.checkbox("Gunakan GPU untuk inference (jika tersedia)", value=False)
    run_btn = st.button("Mulai proses")

    if not uploaded:
        st.info("Upload KMZ untuk memulai.")
        return

    if run_btn:
        tmp_kmz = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz").name
        with open(tmp_kmz, "wb") as f:
            f.write(uploaded.read())

        # extract KML and parse poly
        try:
            kml_path, inner = extract_first_kml_from_kmz(tmp_kmz)
            st.info(f"Memakai internal KML: {inner}")
            boundary = parse_polygons_from_kml(kml_path)
            st.success("Boundary parsed.")
            st.write("Bounds (lon,lat):", boundary.bounds)
        except Exception as e:
            st.error(f"Gagal parse KMZ: {e}")
            return

        # init GEE
        try:
            st.info("Inisialisasi Earth Engine...")
            init_ee_from_st_secrets()
            st.success("Earth Engine siap.")
        except Exception as e:
            st.error(f"Gagal inisialisasi GEE: {e}")
            return

        # get image
        try:
            st.info("Mencari koleksi & image terbaik...")
            img_ee, coll_id = get_best_image_for_region(mapping(boundary), start_date.isoformat(), end_date.isoformat(), desired_scale)
            st.success(f"Koleksi terpilih: {coll_id}")
        except Exception as e:
            st.error(f"Gagal pilih imagery: {e}")
            return

        # download geotiff
        try:
            st.info("Membuat GeoTIFF dari EE (bisa lama)...")
            out_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
            # gunakan CRS EPSG:4326 agar mudah mapping lon/lat; jika ingin pelihara WebMercator gunakan EPSG:3857
            download_image_to_geotiff(img_ee, mapping(boundary), float(desired_scale), out_tif, crs='EPSG:4326')
            st.success("GeoTIFF selesai diunduh.")
            st.write("GeoTIFF path:", out_tif)
            # preview small thumbnail
            with rasterio.open(out_tif) as src:
                arr = src.read([1,2,3]) if src.count>=3 else src.read(1)[None,:,:].repeat(3, axis=0)
                thumb = np.transpose(arr, (1,2,0))
                thumb = ((thumb - thumb.min()) / (thumb.max()-thumb.min()) * 255).astype(np.uint8)
                st.image(thumb, caption="Preview imagery", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal unduh GeoTIFF: {e}")
            return

        # segmentation
        if not uploaded_model:
            st.warning("Model segmentation belum diupload — prosedur berhenti di sini.")
            return

        try:
            st.info("Menjalankan segmentation model...")
            tmp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
            with open(tmp_model_path, "wb") as f:
                f.write(uploaded_model.read())
            buildings, roads = run_segmentation_on_geotiff(out_tif, tmp_model_path, conf=0.25, use_gpu=use_gpu)
            st.success(f"Segmentation selesai: {len(buildings)} bangunan, {len(roads)} roads.")
        except Exception as e:
            st.error(f"Gagal segmentation: {e}\n{traceback.format_exc()}")
            return

        # clip & export
        try:
            # clip to boundary
            buildings_clipped = [b.intersection(boundary) for b in buildings if b.intersects(boundary)]
            roads_clipped = [r.intersection(boundary) for r in roads if r.intersects(boundary)]

            out_dxf = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
            export_to_dxf(boundary, buildings_clipped, roads_clipped, out_dxf, target_epsg=TARGET_EPSG)
            st.success("DXF dibuat.")
            with open(out_dxf, "rb") as f:
                st.download_button("⬇️ Download DXF", f, file_name="map_from_gee.dxf")
        except Exception as e:
            st.error(f"Gagal buat DXF: {e}\n{traceback.format_exc()}")
            return

if __name__ == "__main__":
    run_app()

