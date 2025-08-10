
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
from shapely.geometry import Polygon, MultiPolygon, mapping
from skimage import measure
import os

def mask_to_polygons(mask, simplify_tol=1.0, min_area=10):
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    polys = []
    for contour in contours:
        coords = [(float(x), float(y)) for y,x in contour]
        if len(coords) >= 3:
            from shapely.geometry import Polygon
            poly = Polygon(coords).buffer(0)
            if poly.is_valid and poly.area >= min_area:
                polys.append(poly.simplify(simplify_tol))
    return polys

def run_yolo_segmentation(model_path, image_path, conf=0.3, device="cpu", save_overlay=None):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, device=device, imgsz=1280)
    r = results[0]
    # try to get mask arrays
    try:
        mask_data = r.masks.data.cpu().numpy()
    except Exception:
        try:
            mask_data = r.masks.numpy()
        except Exception:
            mask_data = None
    polygons = []
    # build overlay
    pil = Image.open(image_path).convert("RGB")
    overlay = pil.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    if mask_data is not None:
        for i in range(mask_data.shape[0]):
            mask = mask_data[i].astype(np.uint8)
            polys = mask_to_polygons(mask)
            polygons.extend(polys)
            for poly in polys:
                draw.polygon(list(poly.exterior.coords), fill=(0,0,0,200))
    if save_overlay:
        overlay.save(save_overlay)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return {"polygons":polygons, "overlay_png":buf.getvalue()}
