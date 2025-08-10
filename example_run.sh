
#!/bin/bash
python process_full.py --model yolov8n-seg.pt --image sample.jpg --kml "BOUNDARY CLUSTER.kml" --out out --device cpu
