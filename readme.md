# Detection of Coral Bleaching from PlanetScope Imagery Using a Convolutional Neural Network (CNN) Classifier

**Mariam Ayad¹ · Kevin Valencia² · Christine M. Lee³ · Raphael Kudela¹**
¹University of California, Santa Cruz · ²University of California, Los Angeles · ³Jet Propulsion Laboratory

*Funded by the NASA Minority University Research and Education Project (MUREP) Institutional Research Opportunity, Grant 21-MSI21-0034*

---

## Overview

Coral reefs support roughly 25% of all marine species and sustain the livelihoods of hundreds of millions of people globally. Marine heatwaves are intensifying — the 2023–2025 bleaching event affected an estimated 84% of the world's coral reef area — yet large-scale monitoring remains limited by costly and spatially sparse field surveys. Existing satellite thermal products detect heat stress but not bleaching directly.

This repository implements a ResNet-50 binary classifier (healthy vs. bleached) that operates on **paired PlanetScope SuperDove satellite imagery**. Each input consists of two co-registered 8-band scenes from the same reef location at different times: a healthy reference image and a query image to be classified. The 16 spectral bands (8 reference + 8 query) are concatenated into a single 18-channel tensor (including valid-pixel masks) and passed to the model. Training and evaluation follow a **leave-one-site-out** protocol across 12 reef sites in four regions.

**Related publication:**
Ayad, M., Lee, C. M., Porter, J. W., Chirayath, V., Nivison, C. L., Vaughn, K. M., & Kudela, R. (2025). Impacts of the 2023 Marine Heatwave in the Florida Keys: Detection and Analysis of a Mass Coral Bleaching Event Using Spaceborne Remote Sensing Imagery. *Environmental Science & Technology, 59*(29), 15227–15235.

---

## Results

Leave-one-site-out validation accuracy across all 12 reef sites:

| Region | Site | Tiles | Test Accuracy |
|---|---|---|---|
| Florida Keys | Cheeca Rocks | 80 | 80.31% |
| Florida Keys | Eastern Dry Rocks | 52 | 78.80% |
| Florida Keys | Looe Key | 36 | 97.74% |
| Florida Keys | Rock Key | 36 | 98.24% |
| Florida Keys | Sand Key | 36 | 93.04% |
| Mexico | Chachacual | 130 | 95.97% |
| Mexico | Jicaral | 104 | 98.01% |
| Mexico | Riscalillo | 54 | 98.44% |
| Mexico | San Agustín | 60 | 96.72% |
| Mexico | Santa Cruz | 80 | 82.40% |
| Australia | North Point Lizard Island | 96 | 85.81% |
| Malaysia | Pulau Kapas | 40 | 88.89% |

---

## Repository Structure

```
ct_classifier/          Source module — dataset, model, training loop, utilities
configs/                YAML experiment configs (model architecture, hyperparameters)
scripts/                Shell scripts for batch training runs
utils/                  Shared Python helpers imported by analysis notebooks
  masking.py            Land mask application (shapefile → GeoTIFF, site aliases)
  viz.py                RGB display helpers for PlanetScope imagery
  evaluate.py           Inference, metrics, leakage checks
  gradcam.py            Hooks-based Grad-CAM and occlusion sensitivity
analysis/               Clean, numbered Jupyter notebooks — one analysis per file
  01_data_split         Leave-one-site-out split → data/splits/train.csv + val.csv
  02_evaluation         Val metrics, score histograms, ROC curve, per-site accuracy
  03_gradcam            Multi-layer Grad-CAM on TP/FP/FN/TN examples
  04_spatial_prob_map   Sub-tile sliding-window bleaching probability heatmap
  05_band_importance    Per-channel importance via grad×input and occlusion
  06_failure_analysis   FP/FN pair viewer, score distributions, leakage checks
  07_land_masking_demo  Before/after land mask visualization
docs/                   Data processing documentation
environment.yml         Conda environment specification (CUDA 12.4, PyTorch 2.4)
requirements.txt        Minimal pip requirements
```

---

## Setup

**1. Create the conda environment**

```bash
conda env create -f environment.yml
conda activate coral
```

**2. Register as a Jupyter kernel**

```bash
python -m ipykernel install --user --name coral --display-name "Python (coral)"
```

**3. Verify GPU**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

> **On HPC systems (e.g., NERSC Perlmutter):** install the environment into scratch storage to avoid home directory quota limits:
> ```bash
> module load conda
> conda env create --prefix /path/to/your/scratch/envs/coral -f environment.yml
> conda activate /path/to/your/scratch/envs/coral
> ```

---

## Data

PlanetScope SuperDove (PSB.SD) imagery is commercially licensed through [Planet Labs](https://www.planet.com). The preprocessed tile dataset — 12 reef sites across Florida Keys, Mexico, Australia, and Malaysia (~800 labeled 120×120 px image pairs) — **will be made publicly available upon paper publication**.

In the meantime, researchers with existing Planet access can reproduce the dataset from raw imagery using the preprocessing pipeline described in `docs/`. See [Data Processing](#data-processing) below.

**Tile structure after preprocessing:**

```
planet_superdove_landmasked/
  <site>/
    healthy/tiled_360m/<YYYYMMDD>/loc*.tif
    bleached/tiled_360m/<YYYYMMDD>/loc*.tif
```

Each `.tif` is a 120×120 px, 8-band PlanetScope SuperDove tile (GeoTIFF, float32). Band order: coastal blue · blue · green I · green II · yellow · red · red edge · NIR.

---

## Data Processing

A step-by-step PDF guide covering the full data pipeline — from raw PlanetScope download through spatial alignment (GCP), land masking, and patch extraction — is provided in `docs/`. The guide is written so that researchers can **add new reef sites** to the dataset without modifying the training code.

Pipeline overview:

```
Raw PlanetScope scenes
        ↓  Spatial alignment (GCP correction)
        ↓  Land masking (per-site shapefiles, INVERT=True)
        ↓  Patch extraction (120×120 px tiles, ~360 m footprint)
        ↓  Temporal pairing (healthy reference + query image per location)
        ↓  Leave-one-site-out split (01_data_split.ipynb)
Training-ready tile pairs
```

For the land masking step specifically, `utils/masking.py` handles shapefile lookup, site name aliases, and fallback copying for sites without a land mask.

---

## Training

Edit the config to point `data_root` at your split CSVs, then run:

```bash
python ct_classifier/train.py --config configs/exp_resnet18.yaml
```

Key config fields:

| Field | Description |
|---|---|
| `data_root` | Path to directory containing `train.csv` and `val.csv` |
| `layers` | ResNet variant — `18` or `50` |
| `num_classes` | `2` (healthy / bleached) |
| `normalization_factor` | Scalar divisor applied to raw reflectance values |
| `image_size` | Crop size, default `[120, 120]` |
| `batch_size` | Training batch size |

Model checkpoints and a copy of the config are saved to `model_states/`.

---

## Analysis

Run the notebooks in order from the `analysis/` directory. Each notebook requires only the CONFIG cell at the top to be updated with your checkpoint and data paths.

```
01_data_split.ipynb     → generates data/splits/train.csv and val.csv  (run first)
02_evaluation.ipynb     → loads checkpoint, computes metrics and figures
03_gradcam.ipynb        → spatial attention maps on val examples
04_spatial_prob_map.ipynb → sub-tile bleaching probability heatmap
05_band_importance.ipynb  → per-channel importance (grad×input + occlusion)
06_failure_analysis.ipynb → inspect FP/FN pairs, leakage checks
07_land_masking_demo.ipynb → verify land masking before preprocessing
```

---

## Citation

```bibtex
@article{ayad2025coral,
  title   = {Impacts of the 2023 Marine Heatwave in the Florida Keys: Detection and
             Analysis of a Mass Coral Bleaching Event Using Spaceborne Remote Sensing Imagery},
  author  = {Ayad, Mariam and Lee, Christine M. and Porter, Joseph W. and
             Chirayath, Ved and Nivison, Casey L. and Vaughn, Kathleen M. and Kudela, Raphael},
  journal = {Environmental Science \& Technology},
  volume  = {59},
  number  = {29},
  pages   = {15227--15235},
  year    = {2025},
  doi     = {10.1021/acs.est.5c00001}
}
```

---

## Acknowledgements

This work was funded by the NASA Minority University Research and Education Project (MUREP) Institutional Research Opportunity under Grant 21-MSI21-0034. PlanetScope imagery was accessed through the NASA Commercial Smallsat Data Acquisition (CSDA) program.
