import streamlit as st
import pandas as pd
import geopandas as gpd
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon
import fiona
import ezdxf
import osmnx as ox

# ========== Utility Functions ==========
def classify_layer(highway):
    mapping = {
        "motorway": "PRIMARY",
        "trunk": "PRIMARY",
        "primary": "PRIMARY",
        "secondary": "SECONDARY",
        "tertiary": "SECONDARY",
        "residential": "TERTIARY",
        "unclassified": "TERTIARY",
        "service": "TERTIARY",
    }
    return mapping.get(highway, "OTHER")

def strip_z(geom):
    """Remove Z coordinate if exists"""
    if geom.is_empty:
        return geom
    if geom.has_z:
        return shapely.wkt.loads(shapely.wkt.dumps(geom, output_dimension=2))
    return geom

# ========== Streamlit App ==========
st.title("🏗️ Roads + Open Buildings Processor")

# Upload boundary file
boundary_file = st.file_uploader("Upload KML/KMZ boundary", type=["kml", "kmz"])
csv_file = st.file_uploader("Upload Open Buildings CSV", type=["csv"])

if boundary_file and csv_file:
    try:
        # Read boundary
        st.write("### Loading boundary...")
        boundary_gdf = gpd.read_file(boundary_file)
        boundary = boundary_gdf.geometry.iloc[0]
        st.success(f"Boundary loaded! Type: {boundary.geom_type}")

        # Read Open Buildings
        st.write("### Loading Open Buildings...")
        df = pd.read_csv(csv_file)
        if "geometry" not in df.columns:
            st.error("CSV must contain 'geometry' column with WKT polygons.")
            st.stop()

        df["geometry"] = df["geometry"].apply(shapely.wkt.loads)
        gdf_buildings = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        # Filter by boundary
        st.write("### Filtering buildings inside boundary...")
        gdf_buildings = gdf_buildings[gdf_buildings.intersects(boundary)]
        st.success(f"Buildings after filter: {len(gdf_buildings)}")

        # Roads from OSM
        st.write("### Downloading OSM roads...")
        roads = ox.graph_from_polygon(boundary, network_type="drive")
        gdf_roads = ox.graph_to_gdfs(roads, nodes=False)
        gdf_roads["layer"] = gdf_roads["highway"].apply(classify_layer)

        # Export to DXF
        st.write("### Exporting to DXF...")
        doc = ezdxf.new()
        msp = doc.modelspace()

        # Add boundary
        if boundary.geom_type == "Polygon":
            coords = [(x, y) for x, y in boundary.exterior.coords]
            msp.add_lwpolyline(coords, dxfattribs={"layer": "BOUNDARY"})
        elif boundary.geom_type == "MultiPolygon":
            for poly in boundary.geoms:
                coords = [(x, y) for x, y in poly.exterior.coords]
                msp.add_lwpolyline(coords, dxfattribs={"layer": "BOUNDARY"})

        # Add buildings
        for geom in gdf_buildings.geometry:
            if geom.geom_type == "Polygon":
                coords = [(x, y) for x, y in geom.exterior.coords]
                msp.add_lwpolyline(coords, dxfattribs={"layer": "BUILDINGS"})
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    coords = [(x, y) for x, y in poly.exterior.coords]
                    msp.add_lwpolyline(coords, dxfattribs={"layer": "BUILDINGS"})

        # Add roads
        for geom in gdf_roads.geometry:
            if geom.geom_type == "LineString":
                coords = [(x, y) for x, y in geom.coords]
                msp.add_lwpolyline(coords, dxfattribs={"layer": "ROADS"})
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(x, y) for x, y in line.coords]
                    msp.add_lwpolyline(coords, dxfattribs={"layer": "ROADS"})

        # Save DXF
        out_dxf = "output.dxf"
        doc.saveas(out_dxf)

        # Save GeoJSON
        out_geojson = "output.geojson"
        gdf_buildings.to_file(out_geojson, driver="GeoJSON")

        # Download buttons
        st.success("✅ Processing done!")
        with open(out_dxf, "rb") as f:
            st.download_button("⬇️ Download DXF", f, file_name="output.dxf")
        with open(out_geojson, "rb") as f:
            st.download_button("⬇️ Download GeoJSON", f, file_name="output.geojson")

    except Exception as e:
        st.error(f"Error: {e}")
