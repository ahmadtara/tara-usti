
YOLOv8-seg → Postprocess → GeoTIFF support → DXF exporter
========================================================

Contents:
- process_full.py         : Main CLI script to run the full pipeline.
- utils/                  : Utility modules (yolo, kml, geo, postprocess, dxf)
- requirements.txt        : Python packages recommended
- example_run.sh          : Example command-line run (edit paths)
- LICENSE                 : MIT

Pipeline overview (1 → 3 → 2):
1) Run YOLOv8-seg segmentation on aerial image (supports .jpg/.png and GeoTIFF)
2) Post-process masks: remove small objects, simplify, force-rectangles for buildings, extract centerlines for roads
3) If image is georeferenced (GeoTIFF), transform mask pixel geometry → world coords and reproject to KML CRS
4) Clip with uploaded KML boundary and export final DXF (with BUILDING and ROAD layers).

Notes:
- The script defaults to CPU. For GPU, set device='cuda' in the command.
- If your aerial image is not georeferenced, clipping will be done in image pixel space; ensure your KML is in same coordinate space for proper clipping.
- Tested workflow assumes YOLO model has segmentation head and returns masks via ultralytics YOLO predict results.

Example usage:
python process_full.py --model yolov8n-seg.pt --image sample.jpg --kml "BOUNDARY CLUSTER.kml" --out result_dir --device cpu

