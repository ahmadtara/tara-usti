
#!/usr/bin/env python3
import argparse, os, json
from utils.yolo_utils import run_yolo_segmentation
from utils.kml_utils import read_kml_boundary, kml_crs_epsg
from utils.geo_utils import image_is_geotiff, raster_transform_coords, pixel_polygons_to_world, ensure_same_crs
from utils.postprocess import postprocess_polygons, extract_centerlines_from_mask
from utils.dxf_utils import save_geojson_polygons_to_dxf, geoms_to_features
import geopandas as gpd

def main(args):
    os.makedirs(args.out, exist_ok=True)
    print("Running YOLO segmentation...")
    res = run_yolo_segmentation(args.model, args.image, conf=args.conf, device=args.device, save_overlay=os.path.join(args.out,"overlay.png"))
    polygons = res.get("polygons", [])
    print(f"Detected {len(polygons)} raw polygons")

    # postprocess: remove small noise, simplify, separate building vs road if label info available
    print("Postprocessing polygons...")
    proc = postprocess_polygons(polygons, min_area=args.min_area, simplify_tol=args.simplify)
    print(f"Postprocessed → {len(proc)} polygons")

    # read kml boundary and its CRS (if any)
    print("Reading KML boundary...")
    boundary_geom, kml_epsg = read_kml_boundary(args.kml)

    # if image geotiff, convert pixel coords to world coords
    if image_is_geotiff(args.image):
        print("Image appears georeferenced (GeoTIFF). Converting pixel polygons to world coords...")
        # convert pixel-space polygons to world using raster affine transform
        world_polys = pixel_polygons_to_world(proc, args.image)
        gdf = gpd.GeoDataFrame.from_features({"type":"FeatureCollection","features":geoms_to_features(world_polys)})
        # ensure same CRS as KML (if KML has EPSG)
        if kml_epsg is not None:
            gdf = ensure_same_crs(gdf, kml_epsg)
    else:
        # non-georef: assume KML is in pixel coords or clipping will be done in pixel space
        gdf = gpd.GeoDataFrame.from_features({"type":"FeatureCollection","features":geoms_to_features(proc)})
    gdf.set_crs(epsg=4326, inplace=True, allow_override=True) # placeholder if missing

    # clip with boundary (attempt to reproject boundary to gdf CRS if needed)
    print("Clipping with boundary...")
    clipped = gpd.clip(gdf, boundary_geom)

    # save intermediate GeoJSON
    geojson_path = os.path.join(args.out, "result.geojson")
    clipped.to_file(geojson_path, driver="GeoJSON")
    print("Saved GeoJSON →", geojson_path)

    # save DXF
    dxf_path = os.path.join(args.out, "result.dxf")
    save_geojson_polygons_to_dxf(clipped, dxf_path)
    print("Saved DXF →", dxf_path)
    print("Done.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--kml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--min_area", type=float, default=50.0)
    p.add_argument("--simplify", type=float, default=1.0)
    args = p.parse_args()
    main(args)
