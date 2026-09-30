# analysis/

Clean, numbered Jupyter notebooks — one analysis per file. Run in order;
each notebook imports helpers from `utils/` rather than re-implementing them.

| Notebook | Purpose |
|---|---|
| `01_data_split.ipynb` | Build leave-one-site-out splits → `data/splits/<site>/{train,val,test}.csv` |
| `02_evaluation.ipynb` | Held-out-site metrics, score histograms, ROC curve, per-site accuracy |
| `03_gradcam.ipynb` | Multi-layer GradCAM on TP/FP/FN/TN examples |
| `04_spatial_prob_map.ipynb` | Sub-tile sliding-window probability heatmap on a sampled val tile |
| `05_band_importance.ipynb` | Per-channel importance via grad×input and occlusion |
| `06_failure_analysis.ipynb` | FP/FN pair viewer, score distributions, leakage sanity checks |
| `07_land_masking_demo.ipynb` | Before/after visualization of land mask application |

Notebooks 02–06 evaluate the held-out site (`EVAL_SPLIT = "test"`) by default; set it to `"val"` to inspect the inner validation set instead.
