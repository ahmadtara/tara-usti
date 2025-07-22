import zipfile
import os
import math
import requests
from xml.etree import ElementTree as ET
import ezdxf
from pyproj import Transformer

HERE_API_KEY = "iWCrFicKYt9_AOCtg76h76MlqZkVTn94eHbBl_cE8m0"

# Transformer dari WGS84 ke UTM zona 60 Selatan (EPSG:32760)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32760", always_xy=True)

def extract_kmz(kmz_path, extract_dir):
    with zipfile.ZipFile(kmz_path, 'r') as kmz_file:
        kmz_file.extractall(extract_dir)
    return os.path.join(extract_dir, "doc.kml")

def parse_kml(kml_path):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_path)
    root = tree.getroot()
    placemarks = root.findall('.//kml:Placemark', ns)
    points = []
    for pm in placemarks:
        name = pm.find('kml:name', ns)
        coord = pm.find('.//kml:coordinates', ns)
        if name is not None and coord is not None:
            name_text = name.text.strip()
            coord_text = coord.text.strip()
            lon, lat, *_ = coord_text.split(',')
            points.append({
                'name': name_text,
                'latitude': float(lat),
                'longitude': float(lon)
            })
    return points

def parse_boundaries(kml_path):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_path)
    root = tree.getroot()
    boundaries = []

    for folder in root.findall(".//kml:Folder", ns):
        name_tag = folder.find('kml:name', ns)
        if name_tag is not None and name_tag.text.strip().upper() == "BOUNDARY":
            placemarks = folder.findall(".//kml:Placemark", ns)
            for pm in placemarks:
                coords = pm.find('.//kml:coordinates', ns)
                if coords is not None:
                    coord_list = []
                    for pair in coords.text.strip().split():
                        lon, lat, *_ = map(float, pair.split(','))
                        coord_list.append(latlon_to_xy(lat, lon))
                    boundaries.append(coord_list)
    return boundaries

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def classify_points(points):
    classified = {
        "FDT": [], "FAT": [], "POLE": [], "HP_COVER": [],
        "NEW_POLE": [], "EXISTING_POLE": []
    }
    for p in points:
        name = p['name'].upper()
        if "FDT" in name:
            classified["FDT"].append(p)
        elif "FAT" in name:
            classified["FAT"].append(p)
        elif "HP" in name or "HOME" in name or "COVER" in name:
            classified["HP_COVER"].append(p)
        elif "NEW POLE" in name:
            classified["NEW_POLE"].append(p)
        elif "EXISTING" in name or "EMR" in name:
            classified["EXISTING_POLE"].append(p)
        elif "P" in name:
            classified["POLE"].append(p)
    return classified

def latlon_to_xy(lat, lon):
    x, y = transformer.transform(lon, lat)
    return x, y

def find_nearest_pole(hp, poles):
    nearest = None
    min_dist = float('inf')
    for p in poles:
        d = haversine(hp['latitude'], hp['longitude'], p['latitude'], p['longitude'])
        if d < min_dist:
            min_dist = d
            nearest = p
    return nearest

def get_bounds(points):
    lats = [p["latitude"] for p in points]
    lons = [p["longitude"] for p in points]
    return min(lats), min(lons), max(lats), max(lons)

def fetch_roads_from_here(min_lat, min_lon, max_lat, max_lon):
    url = (
        f"https://vector.hereapi.com/v2/vectortiles/base/mc/14/tiles.json"
        f"?apikey={HERE_API_KEY}"
        f"&bbox={min_lat},{min_lon},{max_lat},{max_lon}"
    )
    response = requests.get(url)
    return []

def draw_boundaries(msp, boundaries):
    for polygon in boundaries:
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        msp.add_lwpolyline(polygon, close=True, dxfattribs={"layer": "FAT AREA"})

def draw_dxf(classified, boundaries, road_lines, output_path):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()

    hp_coords = []

    for hp in classified["HP_COVER"]:
        x, y = latlon_to_xy(hp["latitude"], hp["longitude"])
        hp_coords.append((x, y))
        size = 2
        msp.add_lwpolyline([
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
            (x, y)
        ], close=True, dxfattribs={"layer": "FEATURE_LABEL"})

        nearest = find_nearest_pole(hp, classified["POLE"])
        if nearest:
            nx, ny = latlon_to_xy(nearest["latitude"], nearest["longitude"])
            msp.add_line((x + size/2, y + size/2), (nx, ny), dxfattribs={"layer": "LABEL_CABEL"})

    if len(hp_coords) > 1:
        hp_coords_sorted = sorted(hp_coords, key=lambda k: (k[1], k[0]))
        msp.add_lwpolyline(hp_coords_sorted, dxfattribs={"layer": "JALUR_HP"})

    draw_boundaries(msp, boundaries)

    for line in road_lines:
        coords = [latlon_to_xy(lat, lon) for lat, lon in line]
        msp.add_lwpolyline(coords, dxfattribs={"layer": "JALAN_MAPS"})

    for fat in classified["FAT"]:
        x, y = latlon_to_xy(fat["latitude"], fat["longitude"])
        msp.add_text(fat["name"], dxfattribs={"layer": "FAT"}).set_pos((x, y), align='CENTER')

    for fdt in classified["FDT"]:
        x, y = latlon_to_xy(fdt["latitude"], fdt["longitude"])
        msp.add_text(fdt["name"], dxfattribs={"layer": "FDT"}).set_pos((x, y), align='CENTER')

    for pole in classified["POLE"]:
        x, y = latlon_to_xy(pole["latitude"], pole["longitude"])
        msp.add_text(pole["name"], dxfattribs={"layer": "POLE"}).set_pos((x, y), align='CENTER')

    for pole in classified["NEW_POLE"]:
        x, y = latlon_to_xy(pole["latitude"], pole["longitude"])
        msp.add_text(pole["name"], dxfattribs={"layer": "NEW_POLE"}).set_pos((x, y), align='CENTER')

    for pole in classified["EXISTING_POLE"]:
        x, y = latlon_to_xy(pole["latitude"], pole["longitude"])
        msp.add_text(pole["name"], dxfattribs={"layer": "EXISTING_POLE"}).set_pos((x, y), align='CENTER')

    doc.saveas(output_path)

def convert_kmz_to_dwg(kmz_path, output_dwg):
    extract_dir = "temp_kmz"
    os.makedirs(extract_dir, exist_ok=True)
    kml_path = extract_kmz(kmz_path, extract_dir)
    points = parse_kml(kml_path)
    classified = classify_points(points)
    boundaries = parse_boundaries(kml_path)

    min_lat, min_lon, max_lat, max_lon = get_bounds(points)
    road_lines = fetch_roads_from_here(min_lat, min_lon, max_lat, max_lon)

    draw_dxf(classified, boundaries, road_lines, output_dwg)
    print(f"Saved DWG to {output_dwg}")
