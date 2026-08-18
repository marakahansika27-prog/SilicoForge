# Drift-Sense V2 Dependency Graph

- `ice.py` (Depends on: `cv2`, `numpy`, `logger.py`)
- `gspe.py` (Depends on: `cv2`, `numpy`, `logger.py`)
- `gfee.py` (Depends on: `cv2`, `numpy`, `logger.py`)
- `srae.py` (Depends on: `cv2`, `numpy`, `logger.py`)
- `localization.py` (Depends on: `numpy`, `logger.py`)
- `evaluation.py` (Depends on: `numpy`, `csv`)
- `verify_*.py` (Depends on: all above, `data_loader.py`, `visualizer.py`, `report_generator.py`)
