import cv2
import numpy as np
import ezdxf
import streamlit as st
from shapely.geometry import Polygon

st.set_page_config(page_title="Deteksi Atap Presisi → DXF", layout="wide")

uploaded_file = st.file_uploader("Upload Gambar Satelit", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # === 1. Baca gambar ===
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # === 2. Threshold + edge detection ===
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    edges = cv2.Canny(gray, 50, 150)

    # Gabungkan hasil threshold & edges
    combined = cv2.bitwise_or(thresh, edges)

    # === 3. Morphology untuk pisahkan atap ===
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # === 4. Distance transform untuk watershed ===
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(morph, sure_fg)

    # Marker
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    img_ws = img.copy()
    markers = cv2.watershed(img_ws, markers)

    # === 5. Cari kontur ===
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # === 6. Simpan ke DXF ===
    doc = ezdxf.new()
    msp = doc.modelspace()
    count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) > 80:  # area minimum supaya noise hilang
            approx = cv2.approxPolyDP(cnt, 1.5, True)  # lebih detail
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            if len(pts) > 2:
                poly = Polygon(pts)
                if poly.is_valid:
                    msp.add_lwpolyline(pts, close=True)
                    count += 1

    dxf_path = "atap_presisi.dxf"
    doc.saveas(dxf_path)

    st.success(f"DXF berhasil dibuat! Total atap terdeteksi: {count} 🎉")
    st.download_button("Download DXF", data=open(dxf_path, "rb").read(), file_name="atap_presisi.dxf")

    # === 7. Preview hasil ===
    preview = img.copy()
    cv2.drawContours(preview, contours, -1, (0, 255, 0), 1)
    st.image(preview, caption="Hasil Deteksi Atap Presisi")
