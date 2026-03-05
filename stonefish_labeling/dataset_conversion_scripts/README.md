Camera Segmentation utilities
=============================

Short utilities to export segmentation outputs (frame images + masks) into common
training formats (COCO, YOLO, PNG semantic masks). The tools expect a folder with
`frame_*.png` images and corresponding masks named `frame_*_mask.tiff` (or
legacy `*_ids.tiff`) plus `legend.csv` describing id→color mapping.

Quick examples
--------------

- Export COCO (instance segmentation):
	`python3 dataset_conversion_scripts/convert_to_coco.py --seg-dir <resources> --out-dir <coco_out>`
- Export YOLO (bounding boxes):
	`python3 dataset_conversion_scripts/convert_to_yolo.py --seg-dir <resources> --out-dir <yolo_out>`
- Export semantic PNG masks for Roboflow:
	`python3 dataset_conversion_scripts/convert_to_seg_png.py --seg-dir <resources> --out-dir <seg_out>`

Notes
-----
- Color masks are mapped to ids using exact colors in `legend.csv`. If your masks
	contain anti-aliased colors, consider preprocessing or ask to add tolerant
	nearest-color matching.
- Use `--min-pixels` on the converters to filter tiny spurious regions.
