import streamlit as st
import cv2
import numpy as np
import tempfile
import ezdxf

st.set_page_config(page_title="Ekstrak Atap → DXF", layout="wide")
st.title("Ekstrak Outline Atap Otomatis ke DXF")

uploaded_file = st.file_uploader("Upload Gambar Satelit (.png, .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Simpan file sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img = cv2.imread(tfile.name)

    st.image(img, caption="Gambar Input", use_column_width=True)

    # === PREPROCESS OTOMATIS ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (lebih presisi daripada fixed Canny)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Bersihkan noise kecil
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Temukan kontur
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Buat DXF
    doc = ezdxf.new()
    msp = doc.modelspace()

    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:  # filter objek kecil
            pts = cnt.reshape(-1, 2)
            poly = [(float(x), float(-y)) for x, y in pts]
            if len(poly) > 2:
                msp.add_lwpolyline(poly, close=True, dxfattribs={"layer": "ATAP", "color": 7})
                count += 1

    # Simpan DXF
    dxf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
    doc.saveas(dxf_file.name)

    st.success(f"DXF berhasil dibuat! Total outline atap terdeteksi: {count} 🎉")
    with open(dxf_file.name, "rb") as f:
        st.download_button("⬇️ Download DXF", f, file_name="atap_outline.dxf", mime="application/dxf")
