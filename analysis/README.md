# analysis/

Clean, numbered Jupyter notebooks — one analysis per file. Run in order;
each notebook imports helpers from `utils/` rather than re-implementing them.

| Notebook | Purpose |
|---|---|
| `01_data_split.ipynb` | Build leave-one-site-out splits → `data/splits/train.csv`, `val.csv` |
| `02_evaluation.ipynb` | Val metrics, score histograms, ROC curve, per-site accuracy |
| `03_gradcam.ipynb` | Multi-layer GradCAM on TP/FP/FN/TN examples |
| `04_spatial_prob_map.ipynb` | Sub-tile sliding-window probability heatmap on a sampled val tile |
| `05_band_importance.ipynb` | Per-channel importance via grad×input and occlusion |
| `06_failure_analysis.ipynb` | FP/FN pair viewer, score distributions, leakage sanity checks |
| `07_land_masking_demo.ipynb` | Before/after visualization of land mask application |
