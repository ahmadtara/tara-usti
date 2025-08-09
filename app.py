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
from shapely.validation import make_valid  # untuk perbaikan geometri di Shapely 2.x
from pyproj import Transformer
import numpy as np
from PIL import Image
import rasterio
from rasterio import transform as rio_transform
import cv2
from ultralytics import YOLO
import ezdxf
import ee
import requests

st.set_page_config(page_title="KMZ → DXF (GEE)", layout="wide")

# ---------- CONFIG ----------
TARGET_EPSG = "EPSG:32760"   # output projection untuk DXF
DESIRED_SCALE = 0.5          # meter per pixel target untuk download
PREFERRED_COLLECTIONS = [
    "COPERNICUS/S2_SR",
    "LANDSAT/LC08/C01/T1_SR",
]

# ---------- Helpers: KML/KMZ ----------
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

    if folder_name:
        folders = root.findall(".//kml:Folder", ns)
    else:
        folders = [root]  # proses semua

    for folder in folders:
        if folder_name:
            name_elem = folder.find("kml:name", ns)
            if not (name_elem is not None and name_elem.text and name_elem.text.strip().upper() == folder_name.strip().upper()):
                continue
        for coords in folder.findall(".//kml:coordinates", ns):
            pts = _coords_text_to_list(coords.text)
            if len(pts) >= 4:
                poly = Polygon(pts)
                # Perbaiki jika invalid
                if not poly.is_valid:
                    try:
                        poly = make_valid(poly)
                    except Exception:
                        poly = poly.buffer(0)
                # Hapus polygon kosong/kecil
                if not poly.is_empty and poly.area > 1e-12:
                    polygons.append(poly)

    if not polygons:
        raise RuntimeError("Tidak menemukan polygon di KML.")

    # Pastikan semua valid sebelum union
    clean_polygons = []
    for p in polygons:
        if not p.is_valid:
            try:
                p = make_valid(p)
            except Exception:
                p = p.buffer(0)
        if not p.is_empty and p.area > 1e-12:
            clean_polygons.append(p)

    merged = unary_union(clean_polygons)
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
        raise RuntimeError("st.secrets tidak berisi 'gee_service_account'.")
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

# ---------- Choose best image ----------
def get_best_image_for_region(region_geojson, start_date_str, end_date_str, desired_scale=DESIRED_SCALE):
    ee_start = ee.Date(start_date_str)
    ee_end = ee.Date(end_date_str)
    geom = ee.Geometry(region_geojson)

    for coll_id in PREFERRED_COLLECTIONS:
        try:
            coll = ee.ImageCollection(coll_id).filterBounds(geom).filterDate(ee_start, ee_end)
            count = int(coll.size().getInfo())
            if count > 0:
                st.info(f"Memakai koleksi: {coll_id} (count={count})")
                img = coll.median().clip(geom)
                return img, coll_id
        except Exception as e:
            st.warning(f"Skip collection {coll_id}: {e}")
            continue
    raise RuntimeError("Tidak ada imagery ditemukan untuk area dan tanggal.")

# ---------- Download GeoTIFF ----------
def download_image_to_geotiff(ee_image, region_geojson, scale, out_path, crs='EPSG:4326'):
    params = {
        'scale': float(scale),
        'region': json.dumps(region_geojson),
        'format': 'GEO_TIFF',
        'crs': crs
    }
    url = ee_image.getDownloadURL(params)
    r = requests.get(url, stream=True, timeout=1200)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return out_path

# ---------- Segmentation ----------
def run_segmentation_on_geotiff(geotiff_path, model_path, conf=0.25, use_gpu=False):
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        if src.count >= 3:
            arr = np.dstack([src.read(1), src.read(2), src.read(3)])
        else:
            arr = src.read(1)
            arr = np.dstack([arr, arr, arr])

    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    im = Image.fromarray(((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8))
    im.save(tmp_png)

    device = 0 if use_gpu else "cpu"
    model = YOLO(model_path)
    results = model.predict(source=tmp_png, imgsz=1024, conf=conf, device=device)
    res = results[0]
    masks = getattr(res, "masks", None)
    if masks is None:
        st.warning("Model tidak mengeluarkan masks.")
        return [], []

    mask_arrs = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else masks.data
    cls_arr = masks.boxes.cls.cpu().numpy().astype(int) if hasattr(masks.boxes.cls, "cpu") else masks.boxes.cls

    buildings = []
    roads = []
    for i, m in enumerate(mask_arrs):
        bin_img = (m * 255).astype('uint8')
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            pts_px = [(int(p[0][0]), int(p[0][1])) for p in approx]
            geo_coords = [rasterio.transform.xy(transform, int(y), int(x), offset='center') for x,y in pts_px]
            poly = Polygon([(lon, lat) for lon, lat in geo_coords])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            w = maxx - minx; h = maxy - miny
            aspect = (w/h) if h > 0 else 999
            if aspect > 4 or aspect < 0.25:
                roads.append(poly.exterior)
            else:
                buildings.append(poly)
    return buildings, roads

# ---------- DXF export ----------
def export_to_dxf(boundary_geom, buildings_geo, roads_geo, out_path, target_epsg=TARGET_EPSG):
    transformer = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
    doc = ezdxf.new()
    msp = doc.modelspace()
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
        except:
            continue

    for r in roads_geo:
        try:
            coords = [(transformer.transform(x, y)) for x, y in list(r.coords)]
            msp.add_lwpolyline(coords, dxfattribs={"layer":"ROAD"})
        except:
            continue

    doc.saveas(out_path)

# ---------- Streamlit UI ----------
def run_app():
    st.title("KMZ → DXF (Google Earth Engine source)")
    uploaded = st.file_uploader("Upload .kmz (boundary)", type=["kmz"])
    start_date = st.date_input("Tanggal mulai", value=(datetime.date.today() - datetime.timedelta(days=365)))
    end_date = st.date_input("Tanggal akhir", value=datetime.date.today())
    desired_scale = st.number_input("Desired scale (m/pixel)", min_value=0.1, max_value=10.0, value=DESIRED_SCALE, step=0.1)
    uploaded_model = st.file_uploader("Upload model segmentation (.pt)", type=["pt"])
    use_gpu = st.checkbox("Gunakan GPU", value=False)
    run_btn = st.button("Mulai proses")

    if not uploaded:
        return

    if run_btn:
        tmp_kmz = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz").name
        with open(tmp_kmz, "wb") as f:
            f.write(uploaded.read())

        try:
            kml_path, inner = extract_first_kml_from_kmz(tmp_kmz)
            boundary = parse_polygons_from_kml(kml_path)
            st.success("Boundary parsed.")
        except Exception as e:
            st.error(f"Gagal parse KMZ: {e}")
            return

        try:
            init_ee_from_st_secrets()
        except Exception as e:
            st.error(f"Gagal inisialisasi GEE: {e}")
            return

        try:
            img_ee, coll_id = get_best_image_for_region(mapping(boundary), start_date.isoformat(), end_date.isoformat(), desired_scale)
        except Exception as e:
            st.error(f"Gagal pilih imagery: {e}")
            return

        try:
            out_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
            download_image_to_geotiff(img_ee, mapping(boundary), float(desired_scale), out_tif, crs='EPSG:4326')
            with rasterio.open(out_tif) as src:
                arr = src.read([1,2,3]) if src.count>=3 else src.read(1)[None,:,:].repeat(3, axis=0)
                thumb = np.transpose(arr, (1,2,0))
                thumb = ((thumb - thumb.min()) / (thumb.max()-thumb.min()) * 255).astype(np.uint8)
                st.image(thumb, caption="Preview imagery", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal unduh GeoTIFF: {e}")
            return

        if not uploaded_model:
            st.warning("Model segmentation belum diupload.")
            return

        try:
            tmp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
            with open(tmp_model_path, "wb") as f:
                f.write(uploaded_model.read())
            buildings, roads = run_segmentation_on_geotiff(out_tif, tmp_model_path, conf=0.25, use_gpu=use_gpu)
        except Exception as e:
            st.error(f"Gagal segmentation: {e}")
            return

        try:
            buildings_clipped = [b.intersection(boundary) for b in buildings if b.intersects(boundary)]
            roads_clipped = [r.intersection(boundary) for r in roads if r.intersects(boundary)]
            out_dxf = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
            export_to_dxf(boundary, buildings_clipped, roads_clipped, out_dxf, target_epsg=TARGET_EPSG)
            with open(out_dxf, "rb") as f:
                st.download_button("⬇️ Download DXF", f, file_name="map_from_gee.dxf")
        except Exception as e:
            st.error(f"Gagal buat DXF: {e}")
            return

if __name__ == "__main__":
    run_app()
