# utils/

Shared Python helper modules imported by all analysis notebooks.

| Module | Purpose |
|---|---|
| `masking.py` | Apply per-site shapefile land masks to GeoTIFF tiles |
| `viz.py` | RGB display helpers: unnormalize, band selection, contrast stretch, CAM overlay |
| `evaluate.py` | Model inference, metric collection, leakage checks |
| `gradcam.py` | Hooks-based GradCAM and occlusion sensitivity |
