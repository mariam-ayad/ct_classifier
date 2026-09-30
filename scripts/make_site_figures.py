"""
make_site_figures.py
====================
Per-site supplemental figures for every leave-one-site-out fold:

  1. Score histograms  — softmax bleaching score, train split vs. held-out site (as Figure 4)
  2. Spectral importance — per-channel mean-replacement occlusion on the held-out site (as Figure 5)
  3. Grad-CAM          — reference | query | Grad-CAM on query, for the most confident
                         correct bleached (TP) and healthy (TN) pair of the held-out site (as Figure 6)

For each runs/<site>/best.pt, writes runs/<site>/figures/{score_hist,band_importance,gradcam}.png
(+ band_importance.csv, scores.csv), then combines all sites into supplement grids in
<out-dir>/{supp_score_hist,supp_band_importance,supp_gradcam}.png and supp_band_importance.csv.

Usage
-----
    python scripts/make_site_figures.py                                  # all folds in runs/
    python scripts/make_site_figures.py --sites sanagustin_mexico pulau_kapas
    python scripts/make_site_figures.py --no-train-hist                  # skip train-split inference

    # a checkpoint outside runs/ (config taken from --config, splits from --data_root):
    python scripts/make_site_figures.py --ckpt sanagustin.pt \\
        --data_root data/splits/sanagustin_mexico --out-dir runs/legacy_sanagustin
"""
import os
import sys
import glob
import argparse

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ct_classifier.dataset import BleachDataset
from ct_classifier.model import CustomResNet
from utils.evaluate import get_model_preds
from utils.gradcam import GradCAM, channel_occlusion_importance
from utils.viz import to_display_rgb, percentile_stretch, overlay_cam
from utils.splits import SITES

SITE_NAMES = {
    "cheeca_flkeys":     "Cheeca Rocks",
    "easterndry_flkeys": "Eastern Dry Rocks",
    "looe_flkey":        "Looe Key",
    "rock_flkey":        "Rock Key",
    "sand_flkey":        "Sand Key",
    "chachacual_mexico": "Chachacual",
    "jicaral_mexico":    "Jicaral",
    "riscalillo_mexico": "Riscalillo",
    "sanagustin_mexico": "San Agustín",
    "santacruz_mexico":  "Santa Cruz",
    "northpoint_lizard": "North Point Lizard",
    "pulau_kapas":       "Pulau Kapas",
}

# Channel order produced by BleachDataset: 8 reference bands, 8 query bands, 2 validity masks.
# SuperDove bands: b1 coastal blue 443, b2 blue 490, b3 green I 531, b4 green 565,
#                  b5 yellow 610, b6 red 665, b7 red edge 705, b8 NIR 865 nm
CHANNEL_LABELS = ([f"ref_b{i}" for i in range(1, 9)] + [f"qry_b{i}" for i in range(1, 9)]
                  + ["ref_mask", "qry_mask"])
CHANNEL_COLORS = ["#2196F3"] * 8 + ["#FF5722"] * 8 + ["#9E9E9E"] * 2
HIST_COLORS = {"bleached": "#1f77b4", "healthy": "#ff7f0e"}


def load_fold(ckpt_path, cfg, device):
    model = CustomResNet(cfg["num_classes"], cfg["layers"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


def make_loader(cfg, split, batch_size=None):
    return DataLoader(
        BleachDataset(cfg, split=split, eval=True),
        batch_size=batch_size or cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )


def plot_hist(ax, df, title):
    for label in ("bleached", "healthy"):
        scores = df.loc[df.str_labels == label, "softmax_bleach_scores"]
        ax.hist(scores, bins=20, range=(0, 1), alpha=0.5, color=HIST_COLORS[label], label=label)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)


def plot_importance(ax, drops, title, xticklabels=True):
    norm = drops / (np.abs(drops).sum() + 1e-12)
    x = np.arange(len(CHANNEL_LABELS))
    ax.bar(x, norm, color=CHANNEL_COLORS)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_LABELS if xticklabels else [], rotation=60, ha="right", fontsize=6)
    ax.set_title(title, fontsize=9)
    return norm


def pick_examples(test_df):
    """Most confident correct bleached (TP) and healthy (TN) pair."""
    pred = (test_df.softmax_bleach_scores > 0.5).astype(int)
    tp = test_df[(pred == 1) & (test_df.labels == 1)].sort_values("softmax_bleach_scores", ascending=False)
    tn = test_df[(pred == 0) & (test_df.labels == 0)].sort_values("softmax_bleach_scores")
    return [("TP", tp.iloc[0]) if len(tp) else None, ("TN", tn.iloc[0]) if len(tn) else None]


def gradcam_panels(model, dataset, row, device):
    """Return (ref_rgb, qry_rgb, cam_overlay) for one test pair."""
    idx = dataset.image_couples.index((row.image_id1, row.image_id2))
    tile = dataset[idx][0]
    cam_extractor = GradCAM(model, [model.feature_extractor.layer3, model.feature_extractor.layer4])
    cam = cam_extractor(tile.to(device), class_idx="pred")
    cam_extractor.remove_hooks()
    # mask CAM outside valid (non-land, non-NaN) query pixels
    cam = cam * tile[17].numpy()
    ref = percentile_stretch(to_display_rgb(tile[:8]))
    qry = percentile_stretch(to_display_rgb(tile[8:16]))
    return ref, qry, overlay_cam(qry, cam, alpha=0.45)


def run_site(site, ckpt_path, cfg, fig_dir, device, train_hist=True):
    os.makedirs(fig_dir, exist_ok=True)
    name = SITE_NAMES.get(site, site)
    model = load_fold(ckpt_path, cfg, device)

    dl_test = make_loader(cfg, "test")
    test_df = get_model_preds(model, dl_test, device=device)
    acc = ((test_df.softmax_bleach_scores > 0.5).astype(int) == test_df.labels).mean()
    out = {"site": site, "name": name, "acc": acc, "test_df": test_df}

    # 1. score histograms
    panels = [("train", None), ("test", test_df)]
    if train_hist:
        panels[0] = ("train", get_model_preds(model, make_loader(cfg, "train"), device=device))
        out["train_df"] = panels[0][1]
    else:
        panels = panels[1:]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.5), squeeze=False)
    for ax, (split, df) in zip(axes[0], panels):
        plot_hist(ax, df, f"{name} — {'train scores' if split == 'train' else 'held-out site scores'}")
        ax.set_xlabel("softmax bleach score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "score_hist.png"), dpi=200)
    plt.close(fig)
    pd.concat([df.assign(split=split) for split, df in panels]).to_csv(
        os.path.join(fig_dir, "scores.csv"), index=False)

    # 2. spectral importance
    drops = channel_occlusion_importance(model, dl_test, device=device)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    norm = plot_importance(ax, drops, f"{name} — channel importance (mean-replacement occlusion)")
    ax.set_ylabel("normalized importance")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in
                       (("#2196F3", "reference"), ("#FF5722", "query"), ("#9E9E9E", "mask"))], fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "band_importance.png"), dpi=200)
    plt.close(fig)
    pd.DataFrame({"channel": CHANNEL_LABELS, "logit_drop": drops, "normalized": norm}).to_csv(
        os.path.join(fig_dir, "band_importance.csv"), index=False)
    out["drops"] = drops

    # 3. Grad-CAM
    examples = [e for e in pick_examples(test_df) if e is not None]
    out["gradcam"] = {}
    if examples:
        fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3.5 * len(examples)), squeeze=False)
        for r, (kind, row) in enumerate(examples):
            ref, qry, cam = gradcam_panels(model, dl_test.dataset, row, device)
            out["gradcam"][kind] = (ref, qry, cam, row.softmax_bleach_scores)
            for c, (img, t) in enumerate(((ref, "reference (healthy)"),
                                          (qry, f"query ({row.str_labels})"),
                                          (cam, f"Grad-CAM  p(bleached)={row.softmax_bleach_scores:.2f}"))):
                axes[r, c].imshow(img)
                axes[r, c].set_title(t, fontsize=9)
                axes[r, c].axis("off")
        fig.suptitle(f"{name} — {', '.join(k for k, _ in examples)} examples", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "gradcam.png"), dpi=200)
        plt.close(fig)

    print(f"[{site}] test OA {100 * acc:.2f}% -> {fig_dir}/")
    return out


def make_supplement(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = len(results)
    ncols = 4 if n > 4 else n
    nrows = int(np.ceil(n / ncols))

    # S1: held-out-site score histograms
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax, r in zip(axes.flat, results):
        plot_hist(ax, r["test_df"], f"{r['name']} ({100 * r['acc']:.1f}%)")
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("softmax bleach score (0 = healthy, 1 = bleached)")
    fig.supylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "supp_score_hist.png"), dpi=200)
    plt.close(fig)

    # S1b: train-split histograms, if computed
    with_train = [r for r in results if "train_df" in r]
    if with_train:
        m = len(with_train)
        fig, axes = plt.subplots(int(np.ceil(m / ncols)), min(ncols, m),
                                 figsize=(3.6 * min(ncols, m), 2.8 * int(np.ceil(m / ncols))), squeeze=False)
        for ax, r in zip(axes.flat, with_train):
            plot_hist(ax, r["train_df"], f"{r['name']} fold — train")
        for ax in axes.flat[m:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7)
        fig.supxlabel("softmax bleach score (0 = healthy, 1 = bleached)")
        fig.supylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "supp_score_hist_train.png"), dpi=200)
        plt.close(fig)

    # S2: spectral importance
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows), squeeze=False, sharey=True)
    for i, (ax, r) in enumerate(zip(axes.flat, results)):
        plot_importance(ax, r["drops"], r["name"], xticklabels=(i // ncols == nrows - 1 or i + ncols >= n))
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in
                        (("#2196F3", "reference"), ("#FF5722", "query"), ("#9E9E9E", "mask"))],
               loc="upper right", fontsize=8)
    fig.supylabel("normalized importance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "supp_band_importance.png"), dpi=200)
    plt.close(fig)
    pd.DataFrame({r["site"]: r["drops"] for r in results}, index=CHANNEL_LABELS).to_csv(
        os.path.join(out_dir, "supp_band_importance.csv"))

    # S3: Grad-CAM, one bleached (TP) example per site
    rows = [r for r in results if "TP" in r["gradcam"]]
    if rows:
        fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.5 * len(rows)), squeeze=False)
        for i, r in enumerate(rows):
            ref, qry, cam, p = r["gradcam"]["TP"]
            for c, img in enumerate((ref, qry, cam)):
                axes[i, c].imshow(img)
                axes[i, c].set_xticks([]); axes[i, c].set_yticks([])
            axes[i, 0].set_ylabel(r["name"], fontsize=9)
            axes[i, 2].text(0.02, 0.04, f"p={p:.2f}", transform=axes[i, 2].transAxes,
                            color="white", fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=1))
        for c, t in enumerate(("Healthy reference", "Bleached query", "Grad-CAM")):
            axes[0, c].set_title(t, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "supp_gradcam.png"), dpi=200)
        plt.close(fig)

    print(f"Supplement figures -> {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--sites", nargs="+", default=None, help="Default: every runs/<site>/best.pt found")
    parser.add_argument("--out-dir", default=None, help="Supplement grids (default: <runs-dir>/supplement)")
    parser.add_argument("--no-train-hist", action="store_true", help="Skip train-split inference/histograms")
    parser.add_argument("--ckpt", default=None, help="Single checkpoint outside runs/ (needs --data_root)")
    parser.add_argument("--data_root", default=None, help="Split folder for --ckpt")
    parser.add_argument("--config", default="configs/loso_resnet50.yaml", help="Config for --ckpt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = []
    if args.ckpt:
        assert args.data_root, "--ckpt needs --data_root"
        cfg = yaml.safe_load(open(args.config))
        cfg["data_root"] = os.path.abspath(args.data_root)
        site = pd.read_csv(os.path.join(cfg["data_root"], "test.csv")).site.unique()[0]
        out_dir = args.out_dir or os.path.join(args.runs_dir, f"figures_{site}")
        jobs.append((site, args.ckpt, cfg, out_dir))
    else:
        sites = args.sites or [s for s in SITES if os.path.exists(os.path.join(args.runs_dir, s, "best.pt"))]
        assert sites, f"No runs/<site>/best.pt found under {args.runs_dir}/"
        for site in sites:
            run = os.path.join(args.runs_dir, site)
            cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
            jobs.append((site, os.path.join(run, "best.pt"), cfg, os.path.join(run, "figures")))
        out_dir = args.out_dir or os.path.join(args.runs_dir, "supplement")

    results = [run_site(site, ckpt, cfg, fig_dir, device, train_hist=not args.no_train_hist)
               for site, ckpt, cfg, fig_dir in jobs]
    make_supplement(results, out_dir)


if __name__ == "__main__":
    main()
