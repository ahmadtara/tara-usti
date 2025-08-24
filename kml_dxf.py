import os
import zipfile
import geopandas as gpd
import streamlit as st
import ezdxf
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, linemerge, polygonize
import osmnx as ox

TARGET_EPSG = "EPSG:32760"
DEFAULT_WIDTH = 10

def classify_layer(hwy):
    if hwy in ['motorway', 'trunk', 'primary']:
        return 'HIGHWAYS', 14
    elif hwy in ['secondary', 'tertiary']:
        return 'MAJOR_ROADS', 10
    elif hwy in ['residential', 'unclassified', 'service']:
        return 'MINOR_ROADS', 8
    elif hwy in ['footway', 'path', 'cycleway']:
        return 'PATHS', 4
    return 'OTHER', DEFAULT_WIDTH

def extract_polygon_from_kml_or_kmz(path, target_folder="BONDREY"):
    if path.endswith(".kmz"):
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.endswith(".kml"):
                    z.extract(name, "/tmp")
                    path = os.path.join("/tmp", name)
                    break

    gdf = gpd.read_file(path)

    # Debug: tampilkan kolom untuk cek isi folder
    print("Kolom tersedia:", gdf.columns)

    # --- Filter hanya berdasarkan folder/Name ---
    if "Name" in gdf.columns:
        gdf = gdf[gdf["Name"].str.contains(target_folder, case=False, na=False)]
    elif "FolderPath" in gdf.columns:
        gdf = gdf[gdf["FolderPath"].str.contains(target_folder, case=False, na=False)]
    elif "Description" in gdf.columns:
        gdf = gdf[gdf["Description"].str.contains(target_folder, case=False, na=False)]

    polygons = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if polygons.empty:
        raise Exception(f"❌ Tidak ada Polygon dari folder {target_folder} di KML/KMZ")

    return unary_union(polygons.geometry), polygons.crs

def get_osm_roads(polygon):
    tags = {"highway": True}
    try:
        roads = ox.features_from_polygon(polygon, tags=tags)
    except Exception:
        return gpd.GeoDataFrame()
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    roads = roads.explode(index_parts=False)
    roads = roads[~roads.geometry.is_empty & roads.geometry.notnull()]
    roads = roads.clip(polygon)
    return roads.reset_index(drop=True)

def strip_z(geom):
    if geom.geom_type == "LineString" and geom.has_z:
        return LineString([(x, y) for x, y, *_ in geom.coords])
    elif geom.geom_type == "MultiLineString":
        return MultiLineString([
            LineString([(x, y) for x, y, *_ in line.coords]) if line.has_z else line
            for line in geom.geoms
        ])
    return geom

def export_to_dxf(gdf, dxf_path, polygon=None, polygon_crs=None):
    doc = ezdxf.new()
    msp = doc.modelspace()
    all_buffers = []
    for _, row in gdf.iterrows():
        geom = strip_z(row.geometry)
        hwy = str(row.get("highway", row.get("type", "")))
        layer, width = classify_layer(hwy)
        if geom.is_empty or not geom.is_valid:
            continue
        merged = linemerge(geom) if isinstance(geom, MultiLineString) else geom
        if isinstance(merged, (LineString, MultiLineString)):
            buffered = merged.buffer(width / 2, resolution=8, join_style=2)
            all_buffers.append(buffered)
    if not all_buffers:
        raise Exception("❌ Tidak ada garis valid untuk diekspor.")
    all_union = unary_union(all_buffers)
    outlines = list(polygonize(all_union.boundary))
    min_x = min(pt[0] for geom in outlines for pt in geom.exterior.coords)
    min_y = min(pt[1] for geom in outlines for pt in geom.exterior.coords)
    for outline in outlines:
        coords = [(pt[0] - min_x, pt[1] - min_y) for pt in outline.exterior.coords]
        msp.add_lwpolyline(coords, dxfattribs={"layer": "ROADS"})
    # add boundary
    if polygon is not None and polygon_crs is not None:
        poly = gpd.GeoSeries([polygon], crs=polygon_crs).to_crs(TARGET_EPSG).iloc[0]
        if poly.geom_type == 'Polygon':
            coords = [(pt[0] - min_x, pt[1] - min_y) for pt in poly.exterior.coords]
            msp.add_lwpolyline(coords, dxfattribs={"layer": "BOUNDARY"})
        elif poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                coords = [(pt[0] - min_x, pt[1] - min_y) for pt in p.exterior.coords]
                msp.add_lwpolyline(coords, dxfattribs={"layer": "BOUNDARY"})
    doc.set_modelspace_vport(height=10000)
    doc.saveas(dxf_path)

def process_kml_to_dxf(kml_path, output_dir, target_folder="BONDREY"):
    os.makedirs(output_dir, exist_ok=True)
    polygon, polygon_crs = extract_polygon_from_kml_or_kmz(kml_path, target_folder=target_folder)
    roads = get_osm_roads(polygon)
    if roads.empty:
        raise Exception("❌ Tidak ada jalan ditemukan di OSM.")
    geojson_path = os.path.join(output_dir, "roadmap.geojson")
    dxf_path = os.path.join(output_dir, "roadmap.dxf")
    roads_utm = roads.to_crs(TARGET_EPSG)
    roads_utm.to_file(geojson_path, driver="GeoJSON")
    export_to_dxf(roads_utm, dxf_path, polygon=polygon, polygon_crs=polygon_crs)
    return dxf_path, geojson_path, True

def run_kml_dxf():
    st.title("🌍 KML/KMZ → Road Converter (OSM Only)")
    kml_file = st.file_uploader("Upload file .KML / .KMZ", type=["kml", "kmz"])
    target_folder = st.text_input("Folder name dalam KMZ (contoh: BONDREY)", "BONDREY")
    if kml_file:
        with st.spinner("💫 Memproses file..."):
            try:
                temp_input = f"/tmp/{kml_file.name}"
                with open(temp_input, "wb") as f:
                    f.write(kml_file.read())
                output_dir = "/tmp/output"
                dxf_path, geojson_path, ok = process_kml_to_dxf(temp_input, output_dir, target_folder=target_folder)
                if ok:
                    st.success("✅ Berhasil diekspor ke DXF!")
                    with open(dxf_path, "rb") as f:
                        st.download_button("⬇️ Download DXF (UTM 60)", data=f, file_name="roadmap.dxf")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
