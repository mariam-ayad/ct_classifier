import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import softmax


def get_model_preds(model, loader, device="cpu"):
    """
    Run inference over a dataloader and return a results DataFrame.

    Requires loader to be built with BleachDataset(..., eval=True) so that
    each batch yields (data, labels, (image_id1, image_id2)).

    Returns a DataFrame with columns:
        image_id1, image_id2, softmax_bleach_scores, raw_bleach_scores,
        labels, str_labels, image_path1, image_path2, site, filename
    """
    model.eval()
    model.to(device)

    all_labels, all_raw, all_softmax = [], [], []
    all_id1, all_id2 = [], []

    with torch.no_grad():
        for data, labels, (id1, id2) in loader:
            data = data.to(device)
            logits = model(data).cpu()
            scores = softmax(logits, dim=1)

            all_labels.append(labels.numpy())
            all_raw.append(logits.numpy())
            all_softmax.append(scores.numpy())
            all_id1.append(id1.numpy() if torch.is_tensor(id1) else np.array(id1))
            all_id2.append(id2.numpy() if torch.is_tensor(id2) else np.array(id2))

    labels_np = np.concatenate(all_labels)
    raw_np = np.concatenate(all_raw)
    softmax_np = np.concatenate(all_softmax)
    id1_np = np.concatenate(all_id1)
    id2_np = np.concatenate(all_id2)

    meta = loader.dataset.meta

    def _lookup(ids, col):
        return [meta.query("image_id == @i")[col].values[0] for i in ids]

    df = pd.DataFrame({
        "image_id1":            id1_np,
        "image_id2":            id2_np,
        "softmax_bleach_scores": softmax_np[:, 1],
        "raw_bleach_scores":     raw_np[:, 1],
        "labels":               labels_np.astype(int),
        "str_labels":           ["bleached" if l == 1 else "healthy" for l in labels_np],
        "image_path1":          _lookup(id1_np, "filepath"),
        "image_path2":          _lookup(id2_np, "filepath"),
        "site":                 _lookup(id1_np, "site"),
        "filename":             _lookup(id1_np, "filename"),
    })

    return df


def eval_loss_acc(model, loader, device="cpu"):
    """
    Compute cross-entropy loss and accuracy over a dataloader.

    Works with eval=True or eval=False datasets (ignores the id tuple if present).

    Returns (avg_loss, accuracy) as floats.
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            data, labels = batch[0], batch[1]
            data = data.to(device)
            labels = labels.to(device).long()

            logits = model(data)
            loss_sum += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()

    return loss_sum / total, correct / total


def collect_ids_sites(loader):
    """
    Collect all image IDs (both pair members) and sites from a loader.

    Requires eval=True dataset. Returns a DataFrame with columns:
        image_id1, image_id2, site
    Used for leakage checks before reporting metrics.
    """
    meta = loader.dataset.meta

    id1_list, id2_list, site_list = [], [], []
    for _, _, (id1, id2) in loader:
        id1_arr = id1.numpy() if torch.is_tensor(id1) else np.array(id1)
        id2_arr = id2.numpy() if torch.is_tensor(id2) else np.array(id2)
        id1_list.append(id1_arr)
        id2_list.append(id2_arr)
        site_list.extend(
            meta.query("image_id == @i")["site"].values[0] for i in id1_arr
        )

    return pd.DataFrame({
        "image_id1": np.concatenate(id1_list),
        "image_id2": np.concatenate(id2_list),
        "site":      site_list,
    })


def check_leakage(train_loader, val_loader):
    """
    Assert that no tile image ID appears in both train and val sets.

    Raises AssertionError with details if leakage is found.
    Returns (n_train_ids, n_val_ids) if clean.
    """
    train_meta = train_loader.dataset.meta
    val_meta   = val_loader.dataset.meta

    train_ids = set(train_meta["image_id"].unique())
    val_ids   = set(val_meta["image_id"].unique())

    overlap = train_ids & val_ids
    assert not overlap, (
        f"Leakage detected: {len(overlap)} image IDs appear in both train and val.\n"
        f"Sample overlapping IDs: {sorted(overlap)[:10]}"
    )

    return len(train_ids), len(val_ids)
