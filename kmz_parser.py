from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import wkt

def fix_geometry(geom):
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom
    except Exception as e:
        print(f"[WARNING] Gagal perbaiki geometry: {e}")
        return None

# Saat parsing KMZ
fixed_geoms = []
for geom in original_geoms:
    fixed_geom = fix_geometry(geom)
    if fixed_geom:
        fixed_geoms.append(fixed_geom)

# Kalau mau digabung jadi satu
merged_geom = unary_union(fixed_geoms)
