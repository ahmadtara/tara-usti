
import ezdxf
from shapely.geometry import mapping, Polygon, MultiPolygon, LineString
import geopandas as gpd
import shapely

def geoms_to_features(polygons):
    features = []
    for poly in polygons:
        features.append({"type":"Feature","geometry":mapping(poly),"properties":{}})
    return features

def save_geojson_polygons_to_dxf(gdf, out_path):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    # ensure layers exist
    try:
        doc.layers.new(name='BUILDING', dxfattribs={'color':7})
    except Exception:
        pass
    try:
        doc.layers.new(name='ROAD', dxfattribs={'color':8})
    except Exception:
        pass
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        if isinstance(geom, (Polygon, MultiPolygon)):
            geoms = [geom] if isinstance(geom, Polygon) else list(geom)
            for pg in geoms:
                exterior = list(pg.exterior.coords)
                msp.add_lwpolyline(exterior, dxfattribs={'layer':'BUILDING', 'closed':True})
                for interior in pg.interiors:
                    msp.add_lwpolyline(list(interior.coords), dxfattribs={'layer':'BUILDING', 'closed':True})
        elif isinstance(geom, LineString):
            coords = list(geom.coords)
            msp.add_lwpolyline(coords, dxfattribs={'layer':'ROAD'})
    doc.saveas(out_path)
